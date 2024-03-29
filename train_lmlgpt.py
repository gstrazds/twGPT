"""
---
title: Train Feedback Transformer
summary: This is training code with notes for a feedback transformer.
---

# Train Feedback Transformer

This trains a [feedback transformer](index.html) model for auto-regression.
You can pick the original feedback transformer or the new version
where the keys and values are precalculated.

Here's a Colab notebook for training a feedback transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/transformers/feedback/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d8eb9416530a11eb8fb50242ac1c0002)
"""

import os
import logging
import datetime
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback, EarlyStopping

from labml.helpers.module import Module
from labml.nn.transformers.gpt import _init_weights

from mingpt.pthru_dataset import PlaythroughDataModule
from mingpt.char_dataset import CharDataModule
from mingpt.callback import CUDACallback
from mingpt.lr_decay import LearningRateDecayCallback
from mingpt.utils import tokid_from_logits

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(model, block_size, x, steps, temperature=1.0, sample=False, top_k=None):
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
        x_cond = x_cond.T.contiguous()  # shape (t,b) [values are indices into vocab]

        logits, _ = model(x_cond)  # logits.shape = (t,b,v)
        # pluck the logits at the final step and scale by temperature
        logits = logits[-1, :, :]
        ix = tokid_from_logits(logits, temperature=temperature, sample=sample, top_k=top_k)

        # logits = logits / temperature  # shape = (b,v)
        # # optionally crop probabilities to only the top k options
        # if top_k is not None:
        #     logits = top_k_logits(logits, top_k)
        # # apply softmax to convert to probabilities
        # probs = F.softmax(logits, dim=-1)  # (b,v)
        # # sample from the distribution or take the most likely
        # if sample:
        #     ix = torch.multinomial(probs, num_samples=1)
        # else:
        #     _, ix = torch.topk(probs, k=1, dim=-1)  # greedily choose the single largest

        # ix is a tensor shape=(b,1) of indices into the v (2nd) dimension of probs tensor
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=-1)  #

    return x  # NOTE: returned tensor has shape (b,t_orig+steps)


class SamplePredictions(Callback):
    def __init__(self, tokenizer, dataset, how_many=3, out_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.how_many = how_many
        self.out_dir = out_dir

    def on_validation_end(self, trainer, pl_module):
        n_matched, total_cmd_tokens, full_matches, num_cmds = \
                                eval_predict_cmd_tokens(trainer, pl_module, self.dataset, tokenizer=self.tokenizer)
        cmd_token_acc = n_matched / total_cmd_tokens
        cmd_acc = full_matches / num_cmds
        print(f"VALIDATION CMD_TOKEN_ACC = {cmd_token_acc:.5f}  CMD_ACC = {cmd_acc:.5f}")
        # (NOT YET SUPPORTED): pl_module.log("val_acc", n_matched / total_cmd_tokens, on_step=False, on_epoch=True, prog_bar=True)
        pl_module.logger.log_metrics({"cmd_acc": cmd_acc}, step=trainer.global_step)  #n_matched / total_cmd_tokens)
        pl_module.logger.log_metrics({"tok_acc": cmd_token_acc}, step=trainer.global_step)  #n_matched / total_cmd_tokens)
        if self.out_dir and (not hasattr(trainer, "rank") or trainer.rank == 0):
            with open(self.out_dir +
                      f'cmd_acc_{trainer.current_epoch}-step{trainer.global_step:05d}_{cmd_token_acc:.6f}_{cmd_acc:.6f}.txt', 'w') as outfile:
                outfile.write(f"{cmd_token_acc}\t{n_matched}\t{total_cmd_tokens}\t{cmd_acc}\t{full_matches}\t{num_cmds}\t{trainer.current_epoch}\t{trainer.global_step}")


def show_sample(tokenizer, idx, y_predicted, y_ids, n_sampled=5):
    # print(f"!{idx}!", tokenizer.decode(y_ids))
    print(f"[{idx}]", tokenizer.decode(y_ids[0:6]), '[....]',
          tokenizer.decode(y_ids[-5-n_sampled:-n_sampled]), "|",
          tokenizer.decode(y_ids[-n_sampled:]))

    print(f"<{idx}>", tokenizer.decode(y_predicted[1:7]), '[....]',
          tokenizer.decode(y_predicted[-5-n_sampled:]))
    print()


def build_GPT(cfg):
    # when Hydra config (v1.1) supports recursive instantiation, this could be replaced by instantiate-from-config
    # https://github.com/facebookresearch/hydra/issues/566

    from labml.nn.transformers.gpt import GPT
    from labml.nn.transformers.models import Encoder, TransformerLayer, \
        EmbeddingsWithPositionalEncoding, EmbeddingsWithLearnedPositionalEncoding, Generator
    from labml.nn.transformers.mha import MultiHeadAttention
    from labml.nn.transformers.feed_forward import FeedForward

    n_heads = cfg.model.n_heads
    d_model = cfg.model.d_embd
    n_layers = cfg.model.n_layers
    dropout_prob = cfg.model.dropout
    vocab_size = cfg.model.vocab_size
    d_ff = cfg.model.hidden_layer_multiplier * d_model
    self_attn = MultiHeadAttention(n_heads, d_model, dropout_prob=dropout_prob) #, bias: bool = True)

    feedforward = FeedForward(d_model,
                              d_ff=d_ff,   #4*d_model,   #cfg.model.d_ff,
                              dropout=dropout_prob,
                              activation=nn.GELU())

    encoder_layer = TransformerLayer(
                        d_model=d_model,
                        self_attn=self_attn,
                        src_attn=None,
                        feed_forward=feedforward,
                        dropout_prob=dropout_prob)

    encoder = Encoder(encoder_layer, n_layers=n_layers)
    src_embed = EmbeddingsWithLearnedPositionalEncoding(d_model=d_model, n_vocab=vocab_size) #, max_len=5000)
    generator = Generator(n_vocab=vocab_size, d_model=d_model)
    model = GPT(encoder, src_embed, generator)

    # model.init_weights()
    return model

class GPTLitModule(pl.LightningModule):
    """  GPT """
    # def __init__(self,
    #              vocab_size,
    #              n_embd=768,
    #              block_size=128,
    #              embd_pdrop=0.1,
    #              n_layer=12,
    #              n_head=4,
    #              resid_pdrop=0.1,
    #              attn_pdrop=0.1,
    #              weight_decay=0.1,
    #              betas=(0.9, 0.95),
    #              learning_rate=3e-4,
    #              ):
    #     super().__init__()
    #     # auto creates self.hparams from the method signature
    #     self.save_hyperparameters()
    #     config = self.hparams

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        assert config.model.d_embd % config.model.n_heads == 0,\
            f"embedding dim ({config.model.d_embd}) should be a multiple of num heads ({config.model.n_heads})"

        # self.model = GPT(
        #                 Encoder(EncoderLayer(), n_layers=config.model.n_layers),
        #                 c.transformer.src_embed,
        #                 c.transformer.generator,)
        self.model = build_GPT(config)
                        # d_model = config.model.d_embd,
                        # n_heads = config.model.n_heads,
                        # n_layers = config.model.n_layers,
                        # dropout = config.model.dropout,
                        # d_ff = config.model.d_ff,
                        # n_vocab = config.model.vocab_size,
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokens = 0
        if self.model:
            logger.info("number of parameters: %e", sum(p.numel() for p in self.model.parameters()))
        else:
            logger.error(f"FAILED to build model: {self.model}")

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

        # special case the position embedding parameter in the EmbeddingsWithPositionalEncoding module as not decayed
        no_decay.add('src_embed.positional_encodings')

        # # special case the layer weights for feedback transformer memory attn (TODO: ?not sure if should or should not weight decay)
        # no_decay.add('layer_weight')

        #### FOR COMPARISON: corresponding code from labml.nn.transformers.gpt
        #     This applies weight decay only to weights of linear layers.
        #     """
        #     # Collect names of parameters to apply weight decay
        #     decay = set()
        #     for mn, m in c.model.named_modules():
        #         for pn, p in m.named_parameters():
        #             fpn = f'{mn}.{pn}' if mn else pn  # full param name
        #
        #             if fpn.endswith('weight') and isinstance(m, nn.Linear):
        #                 decay.add(fpn)
        #
        #     # Get all the parameters
        #     param_dict = {pn: p for pn, p in c.model.named_parameters()}
        #     # Parameters that are not decayed
        #     no_decay = set(param_dict.keys()) - decay
        #####################################################################

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in module.named_parameters()}
        for pn in param_dict.keys():
            print(pn)
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
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.trainer.learning_rate, betas=self.hparams.trainer.betas)
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
            x, y, _unused__pad_len = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
            x, y = batch

        x = x.T.contiguous()
        y = y.T.contiguous()
        outputs, _ = self.model(x)
        #print(f"training_step x.size={x.size()} outputs.size={outputs.size()}")
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _unused__pad_len = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
            x, y = batch

        x = x.T.contiguous()
        y = y.T.contiguous()
        outputs, _ = self.model(x)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))

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
        assert len(x.shape) == 1  # expecting a vector of len t
        x_in = torch.unsqueeze(x, 0)  #torch.unsqueeze(x,0) ##== x[None, ...]
        assert x_in.shape[0] == 1  # (b=1, t)
        block_size = self.hparams.model.block_size
        preds = sample(self.model, block_size, x_in, steps=n_samples, temperature=temperature, sample=randsampling, top_k=top_k)
        # self.memory = memory
        # print(f"sample_ahead: x_in.size={x_in.size()} preds.size={preds.size()}")
        return preds.detach().squeeze()  # reduce from (b=1,t) to a single dimension (a vector)
        # return preds.detach()[0]

    def reset_episode(self):
        #print("****** RESET EPISODE ****")
        pass


def eval_predict_cmd_tokens(trainer, pl_module: GPTLitModule, dataset, tokenizer=None):
    total_cmd_tokens = 0
    total_matched = 0
    full_matches = 0
    total_cmds = 0
    n_printed = 0
    rank = 0
    if trainer:
        if hasattr(trainer, "rank"):
            rank = trainer.rank

    max_eval_games = pl_module.hparams.trainer.limit_val_batches

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
            x, y, cmd_start_pos, _cmd_len_ = dataset.get_cmd_prompt_for_gamestep(igame, istep, continuation=-1, fetch_data=True)
            cmd_start_pos = cmd_start_pos.to(pl_module.device)
            x = x.to(pl_module.device)
            y = y.to(pl_module.device)
            cmd_len = len(x) - int(cmd_start_pos) - 1
            x_trunc = x[0:int(cmd_start_pos) + 1]
            y_trunc = y[0:int(cmd_start_pos) + cmd_len]
            assert x_trunc[int(
                cmd_start_pos)] == dataset.cmd_start, f"{cmd_start_pos}: {x_trunc[int(cmd_start_pos)]} {x_trunc}"
            predicted = pl_module.sample_ahead(x_trunc, n_samples=cmd_len, temperature=1.0, randsampling=False,
                                               top_k=None)
            # print(f"predicted shape = {predicted.shape}")
            predicted = predicted.T
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


@hydra.main(config_path="conf", config_name="labml-gpt")
def main(cfg: DictConfig) -> None:
    cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)

    print("cwd_path = ", cfg.cwd_path)

    seed_everything(cfg.general.random_seed)

    start_time = datetime.datetime.now()
    print(f"======================================= train_lmlgpt.py - Start time: {start_time}\n{os.getcwd()}\n")


    if cfg.train_ftwc:
        model_data_id = 'lmlgpt'
        _datamodule = PlaythroughDataModule(
            data_file=cfg.data.data_file,
            val_file=cfg.data.val_file,
            tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.model.block_size,
            train_filtering=cfg.data.train_filtering,
            eval_filtering=cfg.data.eval_filtering, )
    else:
        model_data_id = 'lmlchar'
        _datamodule = CharDataModule(
            data_file=cfg.data.data_file,
            val_file=cfg.data.val_file,
            # tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.model.block_size, )

    _datamodule.prepare_data()
    train_dataset = _datamodule.train_dataset
    cfg.trainer.decay_tokens = 2 * len(train_dataset) * cfg.model.block_size  # approx 2 epochs
    cfg.model.vocab_size = _datamodule.vocab_size

    print(OmegaConf.to_yaml(cfg, resolve=True))
    # print(f"Vocabulary size={cfg.fbt.vocab_size}")

    pl_model = GPTLitModule(cfg)

    print("USING PyTorch Lightning Trainer")
    _datamodule.train_dataset.print_info("train_dataset")
    _datamodule.validation_dataset.print_info("validation_dataset")

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_step',
        # dirpath='my/path/',
        filename=model_data_id+'-{epoch}-{step}-{train_loss_step:.3f}-{val_loss:.3f}',   # {val_acc:.2f}'
        save_top_k=cfg.trainer.save_top_k,
        mode='min',
    )

    lr_decay = LearningRateDecayCallback(
        learning_rate=cfg.trainer.learning_rate,
        warmup_tokens=cfg.trainer.warmup_tokens,
        decay_tokens=cfg.trainer.decay_tokens, # = 2 * len(train_dataset) * cfg.gpt.block_size
        #pad_value=tokenizer.PAD_id
    )

    callback_list = [checkpoint_callback, lr_decay, CUDACallback()]
    if cfg.trainer.patience > 0:
        early_stopping = EarlyStopping('train_loss_step', mode='min', patience=cfg.trainer.patience)
        callback_list.append(early_stopping)
    #early_stopping = EarlyStopping('val_acc', mode='max', patience=5)

    if cfg.train_ftwc:
        show_samples_callback = SamplePredictions(_datamodule.tokenizer, _datamodule.validation_dataset, out_dir="./", how_many=5)
        callback_list.append(show_samples_callback)

    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.trainer.max_epochs,
                         val_check_interval=cfg.trainer.val_check_interval,
                         limit_val_batches=cfg.trainer.limit_val_batches,
                         callbacks=callback_list)
    #torch.autograd.set_detect_anomaly(True)
    trainer.fit(pl_model, _datamodule, ckpt_path=cfg.resume_from_checkpoint)

    finish_time = datetime.datetime.now()
    print(f"================ train_lmlgpt.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")



if __name__ == '__main__':
        main()

