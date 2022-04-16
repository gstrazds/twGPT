"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# class GPTConfig:
#     """ base GPT config, params common to all GPT versions """
#     embd_pdrop = 0.1
#     resid_pdrop = 0.1
#     attn_pdrop = 0.1
#
#     def __init__(self, vocab_size=None, block_size=128, **kwargs):
#         self.vocab_size = vocab_size
#         self.block_size = block_size
#         for k,v in kwargs.items():
#             setattr(self, k, v)
#
# class GPT1Config(GPTConfig):
#     """ GPT-1 like network roughly 125M params """
#     n_layer = 12
#     n_head = 12
#     n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, d_embd, block_size, n_heads, attn_pdrop, resid_pdrop):
        super().__init__()
        assert d_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(d_embd, d_embd)
        self.query = nn.Linear(d_embd, d_embd)
        self.value = nn.Linear(d_embd, d_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(d_embd, d_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_heads

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, d_embd, block_size, n_heads, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embd)
        self.ln2 = nn.LayerNorm(d_embd)
        self.attn = CausalSelfAttention(d_embd, block_size, n_heads, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_embd, 4 * d_embd),
            nn.GELU(),
            nn.Linear(4 * d_embd, d_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self,
                 vocab_size: int,  # size of the vocabulary (number of possible tokens)
                 block_size: int,  # length of the model's context window in time
                 n_layers: int,  # depth of the model; number of Transformer blocks in sequence
                 d_embd: int,  # the "width" of the model, number of channels in each Transformer
                 n_heads: int,  # number of heads in each multi-head attention inside each Transformer block
                 embd_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on input embeddings
                 resid_pdrop: float = 0.1,  # \in [0,1]: amount of dropout in each residual connection
                 attn_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on the attention matrix
                 **kwargs  # ignore any extra named args
                 ):
        super().__init__()

#        mconf = GPTConfig(**hparams)    # n_layer=8, n_head=8, n_embd=512)
        logger.info(f"IGNORING EXTRA GPT() kwargs: {kwargs}")

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, d_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(d_embd, block_size, n_heads, attn_pdrop, resid_pdrop) for _ in range(n_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(d_embd)
        self.head = nn.Linear(d_embd, vocab_size, bias=False)

        self.block_size = block_size
        self.apply(self._init_weights)


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        if t > self.block_size:
            assert False, f"Cannot forward, model block size is exhausted: {t} ! <= {self.block_size}"

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

    def get_param_groups(self, weight_decay: float):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    # print(type(m), "NO_decay:", pn, "\t\t|   ", fpn)
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    # print(type(m), "NO_decay:", pn, "\t\t|   ", fpn)
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    # print(type(m), "DECAY:", pn, "\t\t|   ", fpn)
                    decay.add(fpn)
                else:
                    # print(type(m), "UNMATCHED:", pn, "\t\t||||||   ", fpn)
                    pass
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        # param_dict = {}
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
        #         param_dict[fpn] = p
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # print("DECAY:") #, sorted(list(decay)))
        # for i, pn in enumerate(sorted(list(decay))):
        #         print(f"[{i}]\t\t {pn}")
        # print("NO_DECAY:")
        # for i, pn in enumerate(sorted(list(no_decay))):
        #         print(f"[{i}]\t\t {pn}")
        # print("INTERSECTION:", sorted(list(inter_params)))
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

