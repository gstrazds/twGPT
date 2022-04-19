"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from xformers.factory.model_factory import xFormer, xFormerConfig
from xformers.components.in_proj_container import InProjContainer
from xformers.components.feedforward import MLP
from xformers.components.multi_head_dispatch import MultiHeadDispatch


logger = logging.getLogger(__name__)


class GPTxf(nn.Module):
    def __init__(self, config):
        super().__init__()

        # A list of the encoder or decoder blocks which constitute the Transformer.
        # model:
        #     attention_type: scaled_dot_product
        #     block_size: 128  # spatial extent of the model for its context
        #     n_layers: 8
        #     n_heads: 8
        #     d_embd: 512
        #     embd_pdrop: 0.1
        #     resid_pdrop: 0.1
        #     mlp_pdrop: 0.1
        #     attn_pdrop: 0.1
        #     vocab_size: 0     # gets filled in after we load the vocabulary
        #     hidden_layer_multiplier: 4

        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": config.n_layers,
                "dim_model": config.d_embd,
                "layer_norm_style": "pre",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": config.block_size,
                    "vocab_size": config.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": config.n_heads,
                    "residual_dropout": config.resid_pdrop,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": config.attention_type,
                        "dropout": config.attn_pdrop,
                        "causal": True,
                        "seq_len": config.block_size,
                        "num_rules": config.n_heads,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",  #"FusedMLP",  # Use MLP if Triton is not available
                    "dropout": config.mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": config.hidden_layer_multiplier,
                },
            }
        ]

        _cfg = xFormerConfig(xformer_config)
        self.xformer = xFormer.from_config(_cfg)
        # decoder head
        self.ln_f = nn.LayerNorm(config.d_embd)
        self.head = nn.Linear(config.d_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        # self.criterion = nn.CrossEntropyLoss()
        # self._tokens_seen = 0

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, tokids):   #, targets=None):
        # b, t = tokids.size()
        # if t > self.block_size:
        #     assert False, f"Cannot forward, model block size is exhausted: {t} ! <= {self.block_size}"

        # # forward the GPT model
        # token_embeddings = self.tok_emb(tokids) # each index maps to a (learnable) vector
        # position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        # x = self.drop(token_embeddings + position_embeddings)
        x = self.xformer(tokids)
        x = self.ln_f(x)
        logits = self.head(x)

        # # if we are given some desired targets also calculate the loss
        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits #, loss

    def get_param_groups(self, weight_decay: float):

        if True:
            return self.get_param_groups2(weight_decay)

        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        return optim_groups

    def get_param_groups2(self, weight_decay: float):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, InProjContainer) #, MLP, MultiHeadDispatch)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias') and "sublayer" not in pn:
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
                    # if "sublayer" not in pn and pn.endswith('weight'):
                    #     print(type(m), "UNMATCHED:", pn, "\t\t||||||   ", fpn)
                    pass

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # param_dict = {}
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
        #         param_dict[fpn] = p
        inter_params = decay & no_decay
        union_params = decay | no_decay
        print("DECAY:") #, sorted(list(decay)))
        for i, pn in enumerate(sorted(list(decay))):
                print(f"[{i}]\t\t {pn}")
        print("NO_DECAY:")
        for i, pn in enumerate(sorted(list(no_decay))):
                print(f"[{i}]\t\t {pn}")
        print("INTERSECTION:", sorted(list(inter_params)))
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        if len(param_dict.keys() - union_params) > 0:
            for key in sorted(list(param_dict.keys() - union_params)):
                print(f"PARAM NOT IN param_groups -- {key}: {type(param_dict[key])}")
            assert len(
                param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params),)
        print("union_params - param_dict:")  #, sorted(list(param_dict.keys())))
        for i, pn in enumerate(sorted(list(union_params))):
            if pn not in param_dict:
                print(f"[{i}]\t\t {pn}")

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

