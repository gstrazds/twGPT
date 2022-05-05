import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

import hydra
import pytorch_lightning as pl

from .utils import sample, tokid_from_logits
from .model import GPT

logger = logging.getLogger(__name__)

PADDING_INDEX = 0  # -100 is ignored by default in cross_entropy

def _swapped_first2(n_dims):
    if n_dims == 1:
        return [0]  # don't swap anything
    dims = [1,0]
    for i in range(2, n_dims):
        dims.append(i)
    return dims


class GPTLitModule(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.cmd_start_marker = None   # need to call set_cmd_markers() before running validation_step()
        self.cmd_end_marker = None
        self.transpose_batches = False

        if config.use_framework == 'xf':
            from .model_xf import GPTxf
            self.model = GPTxf(config.model)  # **config.gpt
        elif config.use_framework == 'lml':
            from .model_lml import GPT_lml
            self.model = GPT_lml(config.model)  # **config.gpt
            self.transpose_batches = True
        elif config.use_framework == 'hf':
            from .model_hf import GPThf
            print("***** vocab_size:", config.model.vocab_size)
            self.model = GPThf(config.model, tokenizer=tokenizer)  # **config.gpt
        elif config.use_framework == 'gpt2':
            from .model_gpt2 import GPT2hf
            print("***** vocab_size:", config.model.vocab_size)
            self.model = GPT2hf(config.model, tokenizer=tokenizer)
        elif config.use_framework == 'mingpt':
            self.model = GPT(**config.model)
        else:
            assert False, f"UNKNOWN framework '{config.use_framework}'"

        # self.criterion = nn.CrossEntropyLoss()
        logger.info("number of parameters: %e", sum(p.numel() for p in self.model.parameters()))
        print(self.model)

    @staticmethod
    def adjust_cfg_fields(cfg):
        cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)
        if not cfg.model.d_ff:
            cfg.model.d_ff = cfg.model.d_embd * cfg.model.hidden_layer_multiplier

    @staticmethod
    def adjust_cfg_vocab(cfg, dataset):
        cfg.model.vocab_size = dataset.vocab_size
        try:
            cfg.trainer.decay_tokens = int(cfg.trainer.decay_tokens)
        except (ValueError, TypeError):
            cfg.trainer.decay_tokens = 0
        if not cfg.trainer.decay_tokens:
            cfg.trainer.decay_tokens = cfg.trainer.decay_multiplier * cfg.model.block_size * len(dataset)

    def is_rank_zero(self):
        if hasattr(self, "global_rank"):
            return self.global_rank == 0
        else:
            print("WARNING: NO attr 'global_rank'")
        return False

    def configure_optimizers(self):
        # self._tokens_seen = 0  # reset count of tokens processed
        # separate out all parameters to those that will and won't experience regularizing weight decay
        optim_groups = self.model.get_param_groups(self.hparams.trainer.weight_decay)

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
        return optimizer  #, lr_scheduler

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, targets, _unused__pad_len = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
            x, targets = batch

        if self.transpose_batches:
            x = x.permute(*_swapped_first2(len(x.shape))).contiguous()
            targets = targets.permute(*_swapped_first2(len(targets.shape))).contiguous()
            # x and targets: shape(block_len,batch_len)
        else:
            # x and targets: shape(batch_len,block_len)
            pass

        #print(f"x.shape={x.shape} target.shape={targets.shape}")
        logits = self.model(x)
        # logits: shape(block_len,batch_len,vocab_size) OR shape(batch_len,block_len,vocab_size)
        #print(f"x.shape={x.shape} target.shape={targets.shape} => logits.shape={logits.shape}")
        logits_view = logits.view(-1, logits.size(-1))
        targets_view = targets.view(-1)
        #print(f"::::   logits_view:{logits_view.shape} t_view:{targets_view.shape}")
        # at this point, batch x block_len dimensions have been collapsed into one big sequence:
        # ::::  logits.view: size(txb, vocab_size)  targets.view: size(txb)
        # multi-class cross entropy: # classes = vocab_size. Each token in (bxt) is scored independently
        loss = F.cross_entropy(logits_view, targets_view, ignore_index=PADDING_INDEX)  #-100)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch_cmd_pos = None
        if len(batch) == 3:
            batch_x, batch_y, batch_cmd_pos = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
            batch_x, batch_y = batch

        if self.transpose_batches:
            batch_x = batch_x.permute(*_swapped_first2(len(batch_x.shape))).contiguous()
            batch_y = batch_y.permute(*_swapped_first2(len(batch_y.shape))).contiguous()
            # batch_cmd_pos = batch_cmd_pos.T.contiguous()

        #logits, loss = self.model(batch_x, batch_y)
        logits = self.model(batch_x)
        # loss = F.cross_entropy(logits, batch_y)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), ignore_index=PADDING_INDEX)

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        metrics = {'val_loss': loss} #, 'val_acc': acc}

        n_cmd_tokens = 0
        n_matched_tokens = 0
        n_cmds = 0
        n_matched_cmds = 0
        if self.transpose_batches:   # TODO: fix when transpose_batches == True
            batch_x = batch_x.permute(*_swapped_first2(len(batch_x.shape))).contiguous()  # swap 1st 2 dims back to original order (batch_len, block_len)
            batch_y = batch_y.permute(*_swapped_first2(len(batch_y.shape))).contiguous()
            logits = logits.permute(*_swapped_first2(len(logits.shape))).contiguous()
        if True:
            # print(f"({self.hparams.data.eval_filtering})  batch_y.shape =", batch_y.shape, "logits.shape =", logits.shape)
            if self.hparams.data.eval_filtering == 'cmd_prompts':
                assert batch_cmd_pos is not None
                #print(f"logits.shape={logits.shape}")
                for x, y, cmd_pos, pred in zip(batch_x, batch_y, batch_cmd_pos, logits):  # step through the batch
                    #print(f"calc_cmd_acc: len(x,y)=({len(x),len(y)})\n   x={x}\n  y={y}" )
                    #print(f"pred.shape={pred.shape}")

                    n_toks, n_matched, predicted, y_trunc = \
                        self.calc_cmd_acc(int(cmd_pos), x, y, predicted=pred)
                    n_cmd_tokens += n_toks
                    n_matched_tokens += n_matched
                    n_cmds += 1
                    if n_matched == n_toks:
                        n_matched_cmds += 1
                # self.log('n_cmd_toks', n_cmd_tokens, on_step=True, on_epoch=True, prog_bar=False)
                metrics['n_cmd_toks'] = n_cmd_tokens
                # self.log('n_toks_matched', n_matched_tokens, on_step=True, on_epoch=True, prog_bar=False)
                metrics['n_toks_matched'] = n_matched_tokens
                # self.log('n_cmds', n_cmds, on_step=True, on_epoch=True, prog_bar=True)
                metrics['n_cmds'] = n_cmds
                # self.log('cmd_exact_match', n_matched_cmds, on_step=True, on_epoch=True, prog_bar=True)
                metrics['cmd_exact_match'] = n_matched_cmds
        return metrics

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)

        n_cmd_tokens = sum([x['n_cmd_toks'] for x in outs])
        n_cmds = sum([x['n_cmds'] for x in outs])
        n_matched_tokens = sum([x['n_toks_matched'] for x in outs])
        n_matched_cmds = sum([x['cmd_exact_match'] for x in outs])

        self.log('tok_acc', n_matched_tokens/n_cmd_tokens, on_epoch=True, prog_bar=True)
        self.log('n_cmd_toks', n_cmds, on_epoch=True, prog_bar=False)
        self.log('cmd_acc', n_matched_cmds/n_cmds, on_epoch=True, prog_bar=True)

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

    def set_cmd_markers(self, cmd_start_marker, cmd_end_marker):
        self.cmd_start_marker = cmd_start_marker
        self.cmd_end_marker = cmd_end_marker

    def calc_cmd_acc(self, cmd_start_pos:int, x, y, predicted=None):
        assert self.cmd_start_marker is not None, "Setup ERROR: Need to call pl_module.set_cmd_markers()"
        assert self.cmd_end_marker is not None, "Setup ERROR: Need to call pl_module.set_cmd_markers()"

        if self.hparams.data.eval_filtering == 'cmd_prompts':
            i_end_of_cmd = len(x) - 1
            for i in range(cmd_start_pos, len(x)):
                if x[i].item() == self.cmd_end_marker:
                    # print("len cmd=", i-cmd_start_pos)
                    i_end_of_cmd = i
                    break
            cmd_len = i_end_of_cmd - cmd_start_pos
        else: # self.hparams.data.eval_filtering != 'cmd_prompts':
            cmd_len = len(x) - cmd_start_pos - 1

        x = x.to(self.device)
        y = y.to(self.device)
        x_trunc = x[0:int(cmd_start_pos) + 1]
        y_trunc = y[0:int(cmd_start_pos) + cmd_len]
        #print(f"len(x)={len(x)}, cmd_start_pos={cmd_start_pos} cmd_len={cmd_len}" )
        # print("x:", x)
        # print("x_trunc:", x_trunc)
        # print(f"len(x_trunc) = {len(x_trunc)}")
        assert x_trunc[int(cmd_start_pos)]\
               == self.cmd_start_marker, f"{cmd_start_pos}: {x_trunc[cmd_start_pos]} {x_trunc}"

        if predicted is None:
            predict_out = self.sample_ahead(x_trunc, n_samples=cmd_len, temperature=1.0, randsampling=False,
                                           top_k=None)
            if self.transpose_batches:
                predict_out = predict_out.permute(*_swapped_first2(len(predict_out.shape))).contiguous()
        else:
            out = []
            for x_ in x[0:cmd_start_pos+1]:
                out.append(x_)    # TODO: do this more efficiently with one concat (or refactor to not do it at all)
            for logits in predicted[cmd_start_pos:cmd_start_pos+cmd_len]:
                # print("calc_cmd_acc() - SHAPE of logits =", logits.shape)
                out.append(tokid_from_logits(logits).squeeze(0))
            #print(out)
            predict_out = torch.stack(out)
        assert len(predict_out) == len(y_trunc) + 1, f"{len(predict_out)} {len(y_trunc)}"
        # the following assertion is a sanity check to confirm that x_trunc and y_trunc are aligned correctly
        if cmd_start_pos > 1:  # at start-of-game we have no context, might predict incorrectly
            assert predict_out[1] == y_trunc[0], f"{predict_out[0:5]} {y_trunc[0:5]} (cmd_pos={cmd_start_pos}) {y_trunc.shape}"
        n_matched = int(
            torch.sum(predict_out[-cmd_len:] == y_trunc[-cmd_len:]))  # torch 1.7 has torch.count_nonzero()
        # n_cmd_tokens = int(cmd_len)

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
        return cmd_len, n_matched, predict_out, y_trunc

    def sample_ahead(self, x, n_samples, temperature=1.0, randsampling=False, top_k=None):
        assert len(x.shape) == 1  # expecting a vector of len t
        x_in = torch.unsqueeze(x, 0)  #torch.unsqueeze(x,0) ##== x[None, ...]
        # x_in = x[None, ...]  # x.unsqueeze(0) -- increases tensor rank from 1 to 2 by adding a new dimension 0
        #                      # (consisting of just one row = the original tensor[which, in this case, was a vector])
        assert x_in.shape[0] == 1  # (b=1, t)
        block_size = self.hparams.model.block_size
        if self.transpose_batches:
            preds = sampleT(self.model, block_size, x_in, steps=n_samples, temperature=temperature, sample=randsampling,
                           top_k=top_k)
        else:
            preds = sample(self.model, block_size, x_in, steps=n_samples, temperature=temperature, sample=randsampling, top_k=top_k)
        # print(f"sample_ahead: x_in.size={x_in.size()} preds.size={preds.size()}")
        # return preds.detach()[0]
        return preds.detach().squeeze()  # reduce from (b=1,t) to a single dimension (a vector)


@torch.no_grad()
def sampleT(model, block_size, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    # block_size = model.get_block_size()
    # x.shape == (b,t_orig) [values are indices into vocab]
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x_cond.permute(*_swapped_first2(len(x_cond.shape))).contiguous()  # shape (t,b) [values are indices into vocab]

        logits = model(x_cond)  # logits.shape = (t,b,v)
        # pluck the logits at the final step and scale by temperature
        logits = logits[-1, :, :]
        ix = tokid_from_logits(logits, temperature=temperature, sample=sample, top_k=top_k)
        x = torch.cat((x, ix), dim=-1)  #
    return x  # NOTE: returned tensor has shape (b,t_orig+steps)

def _sample0():  # unused: logic of inner loop of sample(), when transpose_batches == False
                 # NOTE: need sampleT() because each step of for k in steps
                 # not only swaps b,t - but also collapses a different dim of logits:
                 # logits = logits[-1, :, :]
    x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
    logits = model(x_cond)
    logits = logits[:, -1, :]  # use the logits from the last seq pos
    ix = tokid_from_logits(logits, temperature=temperature, sample=sample, top_k=top_k)
    # append to the sequence and continue
    x = torch.cat((x, ix), dim=1)


def eval_predict_cmd_tokens(trainer, pl_module:GPTLitModule, dataset, tokenizer=None):
    total_cmd_tokens = 0
    total_matched = 0
    full_matches = 0
    total_cmds = 0
    n_printed = 0
    rank = -1
    # elif trainer:
    #     if hasattr(trainer, "rank"):
    #         rank = trainer.rank

    print("EVAL_PREDICT_CMD_TOKENS")
    max_eval_games = pl_module.hparams.trainer.limit_val_batches

    # for idx in range(1, len(dataset.cmd_spans)):   # skip the initial 'start' command
    #     x, y, cmd_start_pos = dataset.get_cmd_prompt(idx, continuation=-1)
    #     if idx % 200 == 0 and total_matched == total_cmd_tokens:
    #         print(idx, "...")  # let them know we're actually doing something...
    for igame in range(min(dataset.num_games, max_eval_games)):
        if igame % 10 == 0:
            if pl_module.is_rank_zero():
                print(f"+{igame} [:{dataset.get_num_steps(igame)}] --------------------------")
        if hasattr(pl_module, 'reset_episode'):
            pl_module.reset_episode()
        for istep in range(1, dataset.get_num_steps(igame)):
            total_cmds += 1
            # _span_debug, _, _ = dataset.get_token_idxs(igame, 0, istep)
            # print(f"get_token_idxs(igame={igame}, 0, end_step={istep})  {_span_debug}")
            # print(dataset.data[_span_debug[0]:_span_debug[1]+1])
            x, y, cmd_start_pos = dataset.get_cmd_prompt_for_gamestep(igame, istep, continuation=-1, fetch_data=True)
            # if pl_module.transpose_batches:
            #     x = x.T.contiguous()
            #     y = y.T.contiguous()
                #print("cmd_start_pos:", cmd_start_pos)
            # print( x[cmd_start_pos].item() )

            cmd_len, n_matched, predicted, y_trunc = pl_module.calc_cmd_acc(int(cmd_start_pos), x, y,
                                                                            predicted=None)

            if n_matched == cmd_len:
                full_matches += 1
            else:  # n_matched != n_cmd_tokens:
                n_printed += 1
                if n_printed < 10 or n_printed % 100 == 0 or igame > dataset.num_games - 3:
                    if pl_module.is_rank_zero():
                        print(
                            f" {igame}.{istep}  ...   \t{n_matched} / {cmd_len}   \tacc: {n_matched / cmd_len:4f}")
                        if tokenizer:
                            y_predicted = predicted.cpu().tolist()
                            y_ids = y_trunc.detach().cpu().tolist()

                            # show_sample(tokenizer, f"{igame}.{istep}", y_predicted, y_ids, n_sampled=n_cmd_tokens)
                            n_sampled = cmd_len
                            _idx = f"{igame}.{istep}"
                            print(f"({_idx})", tokenizer.decode(y_ids[0:6]), '[....]',
                                  tokenizer.decode(y_ids[-5 - n_sampled:-n_sampled]), "|",
                                  tokenizer.decode(y_ids[-n_sampled:]))
                            print(f"<{_idx}>", tokenizer.decode(y_predicted[1:7]), '[....]',
                                  tokenizer.decode(y_predicted[-5 - n_sampled:]))
                            print()

            total_cmd_tokens += cmd_len
            total_matched += n_matched
    return total_matched, total_cmd_tokens, full_matches, total_cmds
