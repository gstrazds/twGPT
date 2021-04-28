import os
import glob
# import argparse
from typing import List, Dict, Tuple, Any, Optional

import datetime
import hydra
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from mingpt.pthru_dataset import PlaythroughDataModule
from mingpt.char_dataset import CharDataModule
from mingpt.model import GPTLitModule, eval_predict_cmd_tokens
from mingpt.callback import CUDACallback
from mingpt.lr_decay import LearningRateDecayCallback


@hydra.main(config_path="conf", config_name="pthru-gpt")
def main(cfg: DictConfig) -> None:
    cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)

    # print(OmegaConf.to_yaml(cfg, resolve=True))
    print("cwd_path = ", cfg.cwd_path)

    seed_everything(cfg.general.random_seed)

    start_time = datetime.datetime.now()
    print(f"======================================= train_gpt.py - Start time: {start_time}\n{os.getcwd()}\n")

    if cfg.train_ftwc:
        model_data_id = 'instrgpt'
        _datamodule = PlaythroughDataModule(
            data_file=cfg.data.data_file,
            val_file=cfg.data.val_file,
            tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.gpt.block_size,
            train_filtering=cfg.data.train_filtering,
            eval_filtering=cfg.data.eval_filtering, )
    else:
        model_data_id = 'char'
        _datamodule = CharDataModule(
            data_file=cfg.data.data_file,
            val_file=cfg.data.val_file,
            # tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.gpt.block_size, )

    _datamodule.prepare_data()
    train_dataset = _datamodule.train_dataset

    # dynamically set some config/hparam values (ensures that they get saved with results of each run)
    cfg.gpt.vocab_size = _datamodule.vocab_size
    cfg.trainer.decay_tokens = 2 * len(train_dataset) * cfg.gpt.block_size

    print(OmegaConf.to_yaml(cfg, resolve=True))
    # print(f"Vocabulary size={cfg.gpt.vocab_size}")

    pl_model = GPTLitModule(cfg)

    # pl_model.load_from_checkpoint(checkpoint_path=cfg.cwd_path + "/saved_models/dec18-startofepoch2.ckpt")

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

    callback_list = [checkpoint_callback]

    if cfg.trainer.lr_decay:
        lr_decay = LearningRateDecayCallback(
            learning_rate=cfg.trainer.learning_rate,
            warmup_tokens=cfg.trainer.warmup_tokens,
            decay_tokens=cfg.trainer.decay_tokens, # = 2 * len(train_dataset) * cfg.gpt.block_size
        )
        callback_list.append(lr_decay)

    if cfg.trainer.patience > 0:
        early_stopping = EarlyStopping('val_loss', mode='min', patience=cfg.trainer.patience)
        # early_stopping = EarlyStopping('val_acc', mode='max', patience=5)
        callback_list.append(early_stopping)

    if cfg.train_ftwc:
        show_samples_callback = SamplePredictions(_datamodule.tokenizer, _datamodule.validation_dataset, out_dir="./", how_many=5)
        callback_list.append(show_samples_callback)

    callback_list.append(CUDACallback())

    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.trainer.max_epochs,
                         val_check_interval=cfg.trainer.val_check_interval,
                         limit_val_batches=cfg.trainer.limit_val_batches,
                         resume_from_checkpoint=cfg.resume_from_checkpoint,
                         callbacks=callback_list)
    trainer.fit(pl_model, _datamodule)

    finish_time = datetime.datetime.now()
    print(f"================ train_gpt.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")


def show_sample(tokenizer, idx, y_predicted, y_ids, n_sampled=5):
    # print(f"!{idx}!", tokenizer.decode(y_ids))
    print(f"[{idx}]", tokenizer.decode(y_ids[0:6]), '[....]',
          tokenizer.decode(y_ids[-5-n_sampled:-n_sampled]), "|",
          tokenizer.decode(y_ids[-n_sampled:]))

    print(f"<{idx}>", tokenizer.decode(y_predicted[1:7]), '[....]',
          tokenizer.decode(y_predicted[-5-n_sampled:]))
    print()


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


        # SAMPLE_LEN = 4
        # for idx in range(self.how_many):
        #     x, y = self.dataset[idx * 10]
        #     x = x.to(pl_module.device)
        #     # print(x, y)  #two tensors, identical except for offset by 1 postiion
        #
        #     x_trunc = x[:-(SAMPLE_LEN-1)]  # x is already 1 position behind y, so chop off only SAMPLE_LEN-1
        #
        #     predicted = pl_module.sample_ahead(x_trunc,
        #                     n_samples=SAMPLE_LEN, temperature=1.0, randsampling=False, top_k=None)
        #     y_predicted = predicted.cpu().tolist()
        #     y_ids = y.detach().cpu().tolist()
        #     show_sample(self.tokenizer, idx*10, y_predicted, y_ids, n_sampled=SAMPLE_LEN)



if __name__ == '__main__':
        main()
