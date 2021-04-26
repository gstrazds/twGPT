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

import pytorch_lightning as pl

from .utils import sample


logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size=None, block_size=128, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

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

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLitModule(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        mconf = GPTConfig(**config.gpt)    # n_layer=8, n_head=8, n_embd=512)
        self.model = GPT(mconf)
        self.criterion = nn.CrossEntropyLoss()
        self.tokens = 0
        logger.info("number of parameters: %e", sum(p.numel() for p in self.model.parameters()))

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        module = self.model
        self.tokens = 0  # reset count of tokens processed
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in module.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in module.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.hparams.trainer.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer =  torch.optim.AdamW(optim_groups, lr=self.hparams.trainer.learning_rate, betas=self.hparams.trainer.betas)
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
        #                                 optimizer,
        #                                 T_max=self.hparams.trainer.final_tokens,
        #                                 eta_min=self.hparams.trainer.learning_rate*0.05
        #                                 last_epoch=self.hparams.trainer.final_tokens,
        #                                 ),
        #                 'name': 'cos_anneal_lr',
        #                 'interval': 'step',
        #                 'frequency': 1}
#        self._optimizer = optimizer  # shouldn't be necessary for pt_lightning

        return optimizer  #, lr_scheduler

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _unused__pad_len = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
            x, y = batch

        # clf_logits = self.model(x, classify=True)
        # loss = self.criterion(clf_logits, y)
        # loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1).long())

        logits, loss = self.model(x, y)

        # self.adjust_learning_rate(y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

# decay learning rate (moved to lr_decay.py LearningRateDecayCallback
#     def adjust_learning_rate(self, y):
#         # decay the learning rate based on our progress
#         config = self.hparams.trainer
#         if config.lr_decay:
#             self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
#             if self.tokens < config.warmup_tokens:
#                 # linear warmup
#                 lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
#             else:
#                 # cosine learning rate decay
#                 progress = float(self.tokens - config.warmup_tokens) / float(
#                     max(1, config.final_tokens - config.warmup_tokens))
#                 lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
#             lr = config.learning_rate * lr_mult
#             for param_group in self._optimizer.param_groups:
#                 param_group['lr'] = lr
#         else:
#             lr = config.learning_rate

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _unused__pad_len = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
            x, y = batch
        y_hat, loss = self.model(x, y)

        # # 1. calculate loss
        # loss = F.cross_entropy(y_hat, y)

        # 2. log `val_loss`
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        metrics = {'val_loss': loss} #, 'val_acc': acc}
        return metrics

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)

        # return {"val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_loss': metrics['val_loss'],
                   #'test_acc': metrics['val_acc'],
                   }
        self.log_dict(metrics)

    # def test_epoch_end(self, outs):
    #     result = self.validation_epoch_end(outs)
    #
    #     # replace valid stats with test stats becuase we are reusing function
    #     result["log"]["test_loss"] = result["log"].pop("val_loss")
    #     result["test_loss"] = result.pop("val_loss")
    #     # if self.hparams.classify:
    #     #     result["log"]["test_acc"] = result["log"].pop("val_acc")
    #     return result

    def sample_ahead(self, x, n_samples, temperature=1.0, randsampling=False, top_k=None):
        x_in = x[None, ...]
        preds = sample(self.model, self.hparams.gpt.block_size, x_in, steps=n_samples, temperature=temperature, sample=randsampling, top_k=top_k)
        # print(f"sample_ahead: preds.size={preds.size()}")
        return preds.detach()[0]


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


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


    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


def eval_predict_cmd_tokens(trainer, pl_module:GPTLitModule, dataset, tokenizer=None):
    total_cmd_tokens = 0
    total_matched = 0
    full_matches = 0
    total_cmds = 0
    n_printed = 0
    rank = 0
    if trainer:
        if hasattr(trainer, "rank"):
            rank = trainer.rank

    max_eval_games = 1000000  # infinity for all practical purposes
    max_eval_games = pl_module.hparams.trainer.limit_val_batches

    # for idx in range(1, len(dataset.cmd_spans)):   # skip the initial 'start' command
    #     x, y, cmd_start_pos = dataset.get_cmd_prompt(idx, continuation=-1)
    #     if idx % 200 == 0 and total_matched == total_cmd_tokens:
    #         print(idx, "...")  # let them know we're actually doing something...
    for igame in range(min(dataset.num_games, max_eval_games)):
        if igame % 10 == 0:
            if rank == 0:
                print(f"+{igame} [:{dataset.get_num_steps(igame)}] --------------------------")
        if hasattr(pl_module, 'reset_episode'):
            pl_module.reset_episode()
        for istep in range(1, dataset.get_num_steps(igame)):
            total_cmds += 1
            # _span_debug, _, _ = dataset.get_token_idxs(igame, 0, istep)
            # print(f"get_token_idxs(igame={igame}, 0, end_step={istep})  {_span_debug}")
            # print(dataset.data[_span_debug[0]:_span_debug[1]+1])
            x, y, cmd_start_pos = dataset.get_cmd_prompt_for_gamestep(igame, istep, continuation=-1)
            cmd_start_pos = cmd_start_pos.to(pl_module.device)
            x = x.to(pl_module.device)
            y = y.to(pl_module.device)
            cmd_len = len(x) - int(cmd_start_pos) - 1
            x_trunc = x[0:int(cmd_start_pos) + 1]
            y_trunc = y[0:int(cmd_start_pos) + cmd_len]
            # print(f"len(x)={len(x)}, cmd_start_pos={cmd_start_pos}" )
            # print("cmd_len", cmd_len)
            # print("x:", x)
            # print("x_trunc:", x_trunc)
            # print(f"len(x_trunc) = {len(x_trunc)}")
            assert x_trunc[int(
                cmd_start_pos)] == dataset.cmd_start, f"{cmd_start_pos}: {x_trunc[int(cmd_start_pos)]} {x_trunc}"
            predicted = pl_module.sample_ahead(x_trunc, n_samples=cmd_len, temperature=1.0, randsampling=False,
                                               top_k=None)

            assert len(predicted) == len(y_trunc) + 1, f"{len(predicted)} {len(y_trunc)}"
            assert predicted[1] == y_trunc[0], f"{predicted[0:5]} {y_trunc[0:5]}"

            n_matched_torch = int(
                torch.sum(predicted[-cmd_len:] == y_trunc[-cmd_len:]))  # torch 1.7 has torch.count_nonzero()
            n_cmd_tokens = int(cmd_len)
            # y_predicted = predicted.cpu().tolist()
            # y_ids = y_trunc.detach().cpu().tolist()
            # assert len(y_predicted) == len(y_ids) + 1, f"{len(y_ids)} {len(y_predicted)}"
            # assert y_predicted[1] == y_ids[0], f"{y_predicted[0:5]} {y_ids[0:5]}"

            # n_cmd_tokens = 0
            # n_matched = 0
            # if cmd_len > 1:
            #     for i in range(1, cmd_len + 1):
            #         n_cmd_tokens += 1
            #         if y_predicted[-i] == y_ids[-i]:
            #             n_matched += 1
            # assert n_matched == n_matched_torch, f"{n_matched} {n_matched_torch}"
            # assert n_cmd_tokens == cmd_len

            if n_matched_torch == n_cmd_tokens:
                full_matches += 1
            else:  # n_matched_torch != n_cmd_tokens:
                n_printed += 1
                n_matched = n_matched_torch
                if n_printed < 10 or n_printed % 100 == 0 or igame > dataset.num_games - 3:
                    if rank == 0:
                        print(
                            f" {igame}.{istep}  ...   \t{n_matched} / {n_cmd_tokens}   \tacc: {n_matched / n_cmd_tokens:4f}")
                        if tokenizer:
                            y_predicted = predicted.cpu().tolist()
                            y_ids = y_trunc.detach().cpu().tolist()

                            # show_sample(tokenizer, f"{igame}.{istep}", y_predicted, y_ids, n_sampled=n_cmd_tokens)
                            n_sampled = n_cmd_tokens
                            _idx = f"{igame}.{istep}"
                            print(f"({_idx})", tokenizer.decode(y_ids[0:6]), '[....]',
                                  tokenizer.decode(y_ids[-5 - n_sampled:-n_sampled]), "|",
                                  tokenizer.decode(y_ids[-n_sampled:]))
                            print(f"<{_idx}>", tokenizer.decode(y_predicted[1:7]), '[....]',
                                  tokenizer.decode(y_predicted[-5 - n_sampled:]))
                            print()

            total_cmd_tokens += n_cmd_tokens
            total_matched += n_matched_torch
    return total_matched, total_cmd_tokens, full_matches, total_cmds
