import os
import glob
# import argparse
from typing import List, Dict, Tuple, Any, Optional

import datetime
import hydra
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from tqdm import tqdm

from mingpt.pthru_dataset import PlaythroughDataModule
from mingpt.model import GPTModule
import mingpt
from mingpt.trainer import TrainerConfig

@hydra.main(config_path=".", config_name="pthru-gpt")
def main(cfg: DictConfig) -> None:
    cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("cwd_path = ", cfg.cwd_path)

    seed_everything(cfg.general.random_seed)

    start_time = datetime.datetime.now()
    print(f"======================================= train_gpt.py - Start time: {start_time}\n{os.getcwd()}\n")
    pass

    # mconf = GPTConfig(**cfg.gpt)
    # #                 n_layer=8, n_head=8, n_embd=512)
    # model = GPT(mconf)
    #
    _datamodule = PlaythroughDataModule(
        data_file=cfg.data.data_file,
        val_file=None,
        tokenizer_file=cfg.data.tokenizer_file,
        vocab_size=cfg.data.vocab_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.general.random_seed,
        batch_size=cfg.trainer.batch_size,
        block_size=cfg.gpt.block_size)
    _datamodule.prepare_data()
    train_dataset = _datamodule.train_dataset
    cfg.trainer.final_tokens = 2 * len(train_dataset) * train_dataset.block_size
    cfg.gpt.vocab_size = _datamodule.vocab_size
    pl_model = GPTModule(cfg)

    # initialize a trainer instance and kick off training
    if not cfg.use_lightning:
        tconf = TrainerConfig(**cfg.trainer)
        print(f"Trainer Config: betas={tconf.betas}, final_tokens={tconf.final_tokens}")
        trainer = mingpt.trainer.Trainer(pl_model.model, train_dataset, None, tconf)
        optimizer = pl_model.configure_optimizers()
        trainer.train(optimizer)
    else:
        print("USING PyTorch Lightning Trainer")

        print("Training dataset length =", len(_datamodule.train_dataset))
        pl.Trainer(gpus=cfg.gpus).fit(pl_model, _datamodule)
    finish_time = datetime.datetime.now()
    print(f"================ train_gpt.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")


if __name__ == '__main__':
    main()
