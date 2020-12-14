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

from mingpt.char_dataset import CharDataModule
from mingpt.model import GPTModule
import mingpt
from mingpt.trainer import TrainerConfig

@hydra.main(config_path=".", config_name="char-gpt")
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
    char_datamodule = CharDataModule(
        data_file=cfg.data.input,
        val_file=None,
        num_workers=8,
        seed=cfg.general.random_seed,
        batch_size=cfg.trainer.batch_size,
        block_size=cfg.gpt.block_size)
    char_datamodule.prepare_data()
    train_dataset = char_datamodule.train_dataset
    cfg.trainer.final_tokens = 2 * len(train_dataset) * train_dataset.block_size
    cfg.gpt.vocab_size = train_dataset.vocab_size
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
        # from pl_bolts.models.vision.image_gpt.gpt2 import GPT2
        # from pl_bolts.models.vision import ImageGPT
        # from pl_bolts.datamodules import MNISTDataModule
        # dm = MNISTDataModule('.')
        # pl_model = ImageGPT(dm)

        pl.Trainer(gpus=1).fit(pl_model, char_datamodule)
    finish_time = datetime.datetime.now()
    print(f"================ train_gpt.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")


if __name__ == '__main__':
    main()
