import os
import glob
# import argparse
from typing import List, Dict, Tuple, Any, Optional

import datetime
import hydra
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from tqdm import tqdm

from mingpt.pthru_dataset import PlaythroughDataModule
from mingpt.char_dataset import CharDataModule
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

    if cfg.train_ftwc:
        model_data_id = 'mingpt'
        _datamodule = PlaythroughDataModule(
            data_file=cfg.data.data_file,
            val_file=cfg.data.val_file,
            tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.gpt.block_size)
    else:
        model_data_id = 'char'
        _datamodule = CharDataModule(
            data_file=f'{cfg.cwd_path}/input.txt',
            val_file=f'{cfg.cwd_path}/input.txt',
            # tokenizer_file=cfg.data.tokenizer_file,
            num_workers=cfg.data.num_workers,
            seed=cfg.general.random_seed,
            batch_size=cfg.trainer.batch_size,
            block_size=cfg.gpt.block_size)

    _datamodule.prepare_data()
    train_dataset = _datamodule.train_dataset
    cfg.trainer.final_tokens = 2 * len(train_dataset) * train_dataset.block_size
    cfg.gpt.vocab_size = _datamodule.vocab_size

    print(f"Vocabulary size={cfg.gpt.vocab_size}")

    pl_model = GPTModule(cfg)

    # pl_model.load_from_checkpoint(checkpoint_path=cfg.cwd_path + "/saved_models/dec18-startofepoch2.ckpt")

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

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            # dirpath='my/path/',
            filename=model_data_id+'-{epoch:02d}-{step:05d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )
        callback_list = [checkpoint_callback]
        show_samples_callback = SamplePredictions(_datamodule.tokenizer, _datamodule.train_dataset, how_many=5)
        callback_list.append(show_samples_callback)
        trainer = pl.Trainer(gpus=cfg.gpus,
                             val_check_interval=cfg.val_check_interval,
                             limit_val_batches=cfg.limit_val_batches,
                             resume_from_checkpoint=cfg.resume_from_checkpoint,
                             callbacks=callback_list)
        trainer.fit(pl_model, _datamodule)

    finish_time = datetime.datetime.now()
    print(f"================ train_gpt.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")


from pytorch_lightning.callbacks import Callback

from mingpt.model import sample_ahead

def show_sample(model, tokenizer, idx, x, y, n_samples=5):
    y_ids = y.detach().cpu().tolist()
    print(f"[{idx}]", tokenizer.decode(y_ids[0:6]), '....',
          tokenizer.decode(y_ids[-5:-1]), "|",
          tokenizer.decode(y_ids[-1:]))
    x_in = x[None, ...]
    preds = sample_ahead(model, x, n_samples=n_samples, temperature=1.0, randsampling=False, top_k=None)
    y_out = preds.detach().cpu().tolist()[0]
    # if idx > 16:
    #     print(x_in.size(), preds.size(), y_out[125:])
    print(f"<{idx}>", tokenizer.decode(y_out[1:7]), '....',
          tokenizer.decode(y_out[-(5 - 1) - n_samples:]))
    print()

# def show_sample(tokenizer, idx, y_predicted, y_ids, n_sampled=5):
#     # print(f"!{idx}!", tokenizer.decode(y_ids))
#     print(f"[{idx}]", tokenizer.decode(y_ids[0:6]), '....',
#           tokenizer.decode(y_ids[-5-n_sampled:-n_sampled]), "|",
#           tokenizer.decode(y_ids[-n_sampled:]))
#
#     print(f"<{idx}>", tokenizer.decode(y_predicted[1:7]), '....',
#           tokenizer.decode(y_predicted[-5-n_sampled:]))
#     print()


class SamplePredictions(Callback):
    def __init__(self, tokenizer, dataset, how_many=3, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.how_many = how_many

    def on_validation_end(self, trainer, pl_module):
        for idx in range(self.how_many):
            x, y = self.dataset[idx * 10]
            x = x.to(pl_module.device)
            # print(x, y)  #two tensors, identical except for offset by 1 postiion
            show_sample(pl_module.model, self.tokenizer, idx * 10, x, y, n_samples=4)


if __name__ == '__main__':
        main()
