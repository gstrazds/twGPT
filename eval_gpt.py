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
from mingpt.utils import sample

@hydra.main(config_path=".", config_name="pthru-gpt")
def main(cfg: DictConfig) -> None:
    cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("cwd_path = ", cfg.cwd_path)

    seed_everything(cfg.general.random_seed)

    start_time = datetime.datetime.now()
    print(f"======================================= eval_gpt.py - Start time: {start_time}\n{os.getcwd()}\n")
    pass

    # mconf = GPTConfig(**cfg.gpt)
    # #                 n_layer=8, n_head=8, n_embd=512)
    # model = GPT(mconf)
    #
    _datamodule = PlaythroughDataModule(
        data_file=cfg.data.data_file,
        val_file=cfg.data.val_file,
        tokenizer_file=cfg.data.tokenizer_file,
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
    if False and not cfg.use_lightning:
        tconf = TrainerConfig(**cfg.trainer)
        # print(f"Trainer Config: betas={tconf.betas}, final_tokens={tconf.final_tokens}")
        # trainer = mingpt.trainer.Trainer(pl_model.model, train_dataset, None, tconf)
        # optimizer = pl_model.configure_optimizers()
        # trainer.train(optimizer)
    else:
        print("USING PyTorch Lightning")

        context = "O God, O God!"

        pl_model.load_from_checkpoint(checkpoint_path=cfg.eval.checkpoint)

        # x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(trainer.device)
        # y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
        # completion = ''.join([train_dataset.itos[int(i)] for i in y])
        # print(completion)

        # pl.Trainer(gpus=cfg.gpus).fit(pl_model, _datamodule)
        print(len(_datamodule.train_dataset), len(_datamodule.validation_dataset))
        dataset = _datamodule.train_dataset
        datalen = len(dataset)
        for idx in range(20):
            x, y = dataset[idx*10]
            # print(x, y)  #two tensors, identical except for offset by 1 postiion
            show_sample(pl_model.model, _datamodule.tokenizer, idx*10, x, y, n_samples=4)

    finish_time = datetime.datetime.now()
    print(f"================ eval_gpt.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")

from mingpt.model import sample_ahead

def show_sample(model, tokenizer, idx, x, y, n_samples=5):
    y_ids = y.detach().cpu().tolist()
    print(f"[{idx}]", tokenizer.decode(y_ids[0:6]), '....',
          tokenizer.decode(y_ids[-5:-1]), "|",
          tokenizer.decode(y_ids[-1:]))
    x_in = x[None, ...]
    # preds = sample(gpt_pt.model, x_in, steps=sample_ahead, temperature=1.0, sample=False, top_k=None)
    ## y_in = y[None,...]
    ## preds, loss = gpt_pt.model(x_in, y_in)
    # y_out = preds.detach().cpu().tolist()[0]
    y_out = sample_ahead(model, x, n_samples=n_samples, temperature=1.0, randsampling=False, top_k=None)
    # if idx > 16:
    #     print(x_in.size(), preds.size(), y_out[125:])
    print(f"<{idx}>", tokenizer.decode(y_out[1:7]), '....',
          tokenizer.decode(y_out[-(5 - 1) - n_samples:]))
    print()


if __name__ == '__main__':
    main()
