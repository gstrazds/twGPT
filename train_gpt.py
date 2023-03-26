import os, sys
from pathlib import Path
# import argparse
from typing import List, Dict, Tuple, Any, Optional

import datetime
import hydra
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info

from mingpt.pthru_dataset import PlaythroughDataModule, PlaythroughDataset
from mingpt.char_dataset import CharDataModule
from mingpt.pl_module import GPTLitModule, PADDING_INDEX
from mingpt.callback import CUDACallback
from mingpt.lr_decay import LearningRateDecayCallback



@hydra.main(config_path="conf", config_name="pthru-gpt")
def train_gpt(cfg: DictConfig) -> None:
    sys.path.append(Path(__file__).parent.absolute())   # need to access python modules in subdirs

    seed_everything(cfg.general.random_seed)

    rank_zero_info(f"original_cwd: {hydra.utils.get_original_cwd()}")

    GPTLitModule.adjust_cfg_fields(cfg)
    # cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)

    model_id = f"{cfg.use_framework}:{cfg.model.arch}"
    model_data_id = f"{model_id}:{'pthru' if cfg.train_ftwc else 'char'}"

    start_time = datetime.datetime.now()
    rank_zero_info(f"======================================= {__file__} - Start time: {start_time}\n{os.getcwd()}\n")

    if cfg.train_ftwc:
        _datamodule = PlaythroughDataModule(
            dataset_dir=cfg.data.dataset_dir,
            data_file=cfg.data.data_file,
            val_file=cfg.data.val_file,
            splits_list=None,   #['train', 'valid', 'test'],
            tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.model.block_size,
            train_filtering=cfg.data.train_filtering,
            eval_filtering=cfg.data.eval_filtering,
            ignore_kg=cfg.data.ignore_kg,
            max_pthru_steps=cfg.data.max_pthru_steps, )
    else:
        _datamodule = CharDataModule(
            data_file=cfg.data.data_file,
            val_file=cfg.data.val_file,
            # tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.model.block_size, )

    _datamodule.prepare_data()
    assert PADDING_INDEX == _datamodule.pad_tok, f"PADDING TOKEN index = {_datmodule.pad_tok} (should be {PADDING_INDEX})"

    # dynamically set some config/hparam values (ensures that they get saved with results of each run)
    train_dataset = _datamodule.train_dataset
    GPTLitModule.adjust_cfg_vocab(cfg, train_dataset)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    # print(f"Vocabulary size={cfg.model.vocab_size}")

    pl_model = GPTLitModule(cfg, tokenizer=None)  #_datamodule.tokenizer)

    pl_model.set_cmd_markers(_datamodule.cmd_start_marker, _datamodule.cmd_end_marker)

    # pl_model.load_from_checkpoint(checkpoint_path=cfg.cwd_path + "/saved_models/dec18-startofepoch2.ckpt")

    _datamodule.train_dataset.print_info("train_dataset")
    _datamodule.validation_dataset.print_info("validation_dataset")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  #'train_loss_step',
        # dirpath='my/path/',
        filename=model_data_id+'-{epoch}-{step}-{val_acc:.3f}-{val_loss:.3f}',   # {val_acc:.2f}'
        save_top_k=cfg.trainer.save_top_k,
        mode='max', #'min',
    )

    callback_list = [checkpoint_callback]

    if cfg.trainer.lr_decay:
        lr_decay = LearningRateDecayCallback(
            learning_rate=cfg.trainer.learning_rate,
            warmup_tokens=cfg.trainer.warmup_tokens,
            decay_tokens=cfg.trainer.decay_tokens,  # = 2 * len(train_dataset) * cfg.model.block_size
        )
        callback_list.append(lr_decay)

    if cfg.trainer.patience > 0:
        early_stopping = EarlyStopping('val_acc', mode='max', patience=cfg.trainer.patience)
        # early_stopping = EarlyStopping('val_acc', mode='max', patience=5)
        callback_list.append(early_stopping)

    if cfg.train_ftwc:

        pl_model._debug_tokenizer = _datamodule.tokenizer
        # DEBUG HACK
        # if hasattr(cfg.trainer, 'eval_predict') and cfg.trainer.eval_predict:

        if cfg.eval.show_samples:
            assert _datamodule.validation_dataset.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS, \
                f"trainer.show_samples requires data.eval_filtering='cmd_prompts' INCOMPATIBLE:{_datamodule.validation_dataset.span_filtering}"
            val_dataloader = _datamodule.val_dataloader()
            show_samples_callback = SamplePredictions(_datamodule.tokenizer, _datamodule.validation_dataset, out_dir="./", how_many=5,
                                                      dataloader=val_dataloader)
            callback_list.append(show_samples_callback)

    callback_list.append(CUDACallback())

    EXPERIMENT_NAME = cfg.wandb.experiment if cfg.wandb.experiment else ''
    tb_logger = TensorBoardLogger(save_dir='logs/', name=EXPERIMENT_NAME)
    loggers_list = [tb_logger]
    if cfg.use_wandb:
        if pl_model.is_rank_zero():
            wandb.init(project=cfg.wandb.proj, name=EXPERIMENT_NAME, settings=wandb.Settings(start_method="thread"))
            wandb.define_metric('val_acc', summary='max')
        wandb_logger = WandbLogger(project=cfg.wandb.proj, name=EXPERIMENT_NAME)
        loggers_list.append(wandb_logger)


    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.trainer.max_epochs,
                         val_check_interval=cfg.trainer.val_check_interval,
                         limit_val_batches=cfg.trainer.limit_val_batches,
                         callbacks=callback_list,
                         strategy='ddp',
                         logger=loggers_list)
    trainer.fit(pl_model, _datamodule, ckpt_path=cfg.resume_from_checkpoint)

    finish_time = datetime.datetime.now()
    rank_zero_info(f"================ {__file__} - Finished : {finish_time} -- elapsed: {finish_time-start_time}")


def show_sample(tokenizer, idx, y_predicted, y_ids, n_sampled=5):
    # print(f"!{idx}!", tokenizer.decode(y_ids))
    print(f"[{idx}]", tokenizer.decode(y_ids[0:6]), '[....]',
          tokenizer.decode(y_ids[-5-n_sampled:-n_sampled]), "|",
          tokenizer.decode(y_ids[-n_sampled:]))

    print(f"<{idx}>", tokenizer.decode(y_predicted[1:7]), '[....]',
          tokenizer.decode(y_predicted[-5-n_sampled:]))
    print()


class SamplePredictions(Callback):
    def __init__(self, tokenizer, dataset, how_many=3, out_dir=None, dataloader=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataloader = dataloader
        self.how_many = how_many
        self.out_dir = out_dir

    def on_validation_end(self, trainer, pl_module):
        if pl_module.is_rank_zero():
            if False and hasattr(pl_module.hparams.eval, "show_samples"):
                show_samples = pl_module.hparams.eval.show_samples
            else:
                show_samples = True
            if True:
                n_matched, total_cmd_tokens, full_matches, num_cmds = \
                    pl_module.eval_predict_cmd_tokens(self.dataset, tokenizer=self.tokenizer, show_samples=show_samples)
            else:
                n_matched, total_cmd_tokens, full_matches, num_cmds = \
                    pl_module.eval_predict_cmds_batched(self.dataset, self.dataloader, tokenizer=self.tokenizer, show_samples=show_samples)
            tok_acc = n_matched / total_cmd_tokens
            cmd_acc = full_matches / num_cmds
            rank_zero_info(f"\nSAMPLED CMD_TOKEN_ACC = {tok_acc*100:02.2f} %  CMD_ACC = {cmd_acc*100:02.2f} %")
            pl_module.logger.log_metrics({"cmd_acc": cmd_acc}, step=trainer.global_step)  #n_matched / total_cmd_tokens)
            pl_module.logger.log_metrics({"tok_acc": tok_acc}, step=trainer.global_step)  #n_matched / total_cmd_tokens)
            if self.out_dir:  #(not hasattr(trainer, "rank") or trainer.rank == 0):
                with open(self.out_dir +
                          f'cmd_acc_{trainer.current_epoch}-step{trainer.global_step:05d}_{tok_acc:.4f}_{cmd_acc:.4f}.txt', 'w') as outfile:
                    outfile.write(f"{tok_acc}\t{n_matched}\t{total_cmd_tokens}\t{cmd_acc}\t{full_matches}\t{num_cmds}\t{trainer.current_epoch}\t{trainer.global_step}")



if __name__ == '__main__':
        train_gpt()
