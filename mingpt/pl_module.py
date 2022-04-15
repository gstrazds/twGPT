import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

from .utils import sample, tokid_from_logits
from .model import GPT

logger = logging.getLogger(__name__)

class GPTLitModule(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.cmd_start_marker = None   # need to call set_cmd_markers() before running validation_step()
        self.cmd_end_marker = None

        if config.gpt.use_xformers:
            from .model_xf import GPTxf
            self.model = GPTxf(config.gpt)  # **config.gpt
        else:
            self.model = GPT(**config.gpt)
        # self.criterion = nn.CrossEntropyLoss()
        logger.info("number of parameters: %e", sum(p.numel() for p in self.model.parameters()))
        print(self.model)

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

        #logits, loss = self.model(x, targets)
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch_cmd_pos = None
        if len(batch) == 3:
            batch_x, batch_y, batch_cmd_pos = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
            batch_x, batch_y = batch

        #logits, loss = self.model(batch_x, batch_y)
        logits = self.model(batch_x)
        # loss = F.cross_entropy(logits, batch_y)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        metrics = {'val_loss': loss} #, 'val_acc': acc}

        n_cmd_tokens = 0
        n_matched_tokens = 0
        n_cmds = 0
        n_matched_cmds = 0
        if self.hparams.data.eval_filtering == 'cmd_prompts':
            assert batch_cmd_pos is not None
            #print(f"logits.shape={logits.shape}")
            for x, y, cmd_pos, pred in zip(batch_x, batch_y, batch_cmd_pos, logits):
                #print(f"calc_cmd_acc: len(x,y)=({len(x),len(y)})\n   x={x}\n  y={y}" )
                #print(f"pred.shape={pred.shape}")

                n_toks, n_matched, predicted, y_trunc = \
                    self.calc_cmd_acc(int(cmd_pos), x, y, predicted=pred)
                n_cmd_tokens += n_toks
                n_matched_tokens += n_matched
                n_cmds += 1
                if n_matched == n_toks:
                    n_matched_cmds += 1
        self.log('n_cmd_toks', n_cmd_tokens, on_step=True, on_epoch=True, prog_bar=False)
        metrics['n_cmd_toks'] = n_cmd_tokens
        self.log('n_toks_matched', n_matched_tokens, on_step=True, on_epoch=True, prog_bar=False)
        metrics['n_toks_matched'] = n_matched_tokens
        self.log('n_cmds', n_cmds, on_step=True, on_epoch=True, prog_bar=True)
        metrics['n_cmds'] = n_cmds
        self.log('cmd_exact_match', n_matched_cmds, on_step=True, on_epoch=True, prog_bar=True)
        metrics['cmd_exact_match'] = n_matched_cmds
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

        x = x.to(self.device)
        y = y.to(self.device)
        if self.hparams.data.eval_filtering != 'cmd_prompts':
            cmd_len = len(x) - cmd_start_pos - 1
        x_trunc = x[0:int(cmd_start_pos) + 1]
        y_trunc = y[0:int(cmd_start_pos) + cmd_len]
        # print(f"len(x)={len(x)}, cmd_start_pos={cmd_start_pos}" )
        # print("cmd_len", cmd_len)
        # print("x:", x)
        # print("x_trunc:", x_trunc)
        # print(f"len(x_trunc) = {len(x_trunc)}")
        assert x_trunc[int(cmd_start_pos)]\
               == self.cmd_start_marker, f"{cmd_start_pos}: {x_trunc[cmd_start_pos]} {x_trunc}"

        if predicted is None:
            predict_out = self.sample_ahead(x_trunc, n_samples=cmd_len, temperature=1.0, randsampling=False,
                                           top_k=None)
        else:
            out = []
            for x_ in x[0:cmd_start_pos+1]:
                out.append(x_)    # TODO: do this more efficiently with one concat (or refactor to not do it at all)
            for logits in predicted[cmd_start_pos:cmd_start_pos+cmd_len]:
                #print("SHAPE of logits =", logits.shape)
                out.append(tokid_from_logits(logits).squeeze(0))
            #print(out)
            predict_out = torch.stack(out)
        assert len(predict_out) == len(y_trunc) + 1, f"{len(predict_out)} {len(y_trunc)}"
        # the following assertion is a sanity check to confirm that x_trunc and y_trunc are aligned correctly
        if cmd_start_pos > 1:  # at start-of-game we have no context, might predict incorrectly
            assert predict_out[1] == y_trunc[0], f"{predict_out[0:5]} {y_trunc[0:5]}"
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
        x_in = x[None, ...]  # x.unsqueeze(0) -- increases tensor rank from 1 to 2 by adding a new dimension 0
                             # (consisting of just one row = the original tensor[which, in this case, was a vector])

        preds = sample(self.model, self.hparams.gpt.block_size, x_in, steps=n_samples, temperature=temperature, sample=randsampling, top_k=top_k)
        # print(f"sample_ahead: preds.size={preds.size()}")
        return preds.detach()[0]


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

    max_eval_games = 1000000  # infinity for all practical purposes
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
            x, y, cmd_start_pos = dataset.get_cmd_prompt_for_gamestep(igame, istep, continuation=-1)
            # print( x[cmd_start_pos].item() )

            cmd_len, n_matched, predicted, y_trunc = pl_module.calc_cmd_acc(int(cmd_start_pos), x, y)

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
