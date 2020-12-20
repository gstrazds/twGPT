#!/usr/bin/env python
# coding: utf-8

# ## Train a character-level GPT on some text data
# 
# The inputs here are simple text files,
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from torch.utils.data import Dataset

class PlaythroughDataset(Dataset):

    def __init__(self, data, block_size):
        data_size = len(data)
        print("PlaythroughDataset datalen=", data_size)
        self.block_size = block_size
        self.data = list(data)  # make a copy

        # chars = sorted(list(set(data)))
        # vocab_size = len(chars)
        # self.vocab_size = vocab_size
        # print('data has %d characters, %d unique.' % (data_size, vocab_size))
        # self.stoi = { ch:i for i,ch in enumerate(chars) }
        # self.itos = { i:ch for i,ch in enumerate(chars) }

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        # dix = self.data[self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y



from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from tokenizers import Tokenizer

class PlaythroughDataModule(LightningDataModule):
    """
    """

    name = "ftwc_pthru"

    def __init__(
        self,
        data_file: str = "./mingpt-training-all.pthru",
        val_file: str = None,
        num_workers: int = 16,
        tokenizer_file: str = "ftwc_tokenizer.json",
        seed: int = 42,
        batch_size: int = 192,
        block_size: int = 128,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
        """
        super().__init__(*args, **kwargs)

        self.data_file = data_file
        self.tokenizer_file = tokenizer_file
        self.val_file = val_file
        self.num_workers = num_workers
        self.seed = seed
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_dataset = None
        self.validation_dataset = None
        self.vocab_size = 0

    def read_and_encode(self, filepath):
        with open(filepath, 'r') as file:
            text = file.read()
        encoded_data = self.tokenizer.encode(text)
        #encoded_data.ids
        #encoded_data.tokens
        return encoded_data

    def prepare_data(self):
        """
        """
        self.tokenizer = Tokenizer.from_file(self.tokenizer_file)
        if not self.vocab_size:
            self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        # TODO: get dataset length after loading data and use it to compute final_tokens
        encoded_data = self.read_and_encode(self.data_file)
        print("PlaythroughDataModule.prepare_data: ", len(encoded_data.ids))
        self.train_dataset = PlaythroughDataset(encoded_data.ids, self.block_size)  # one line of poem is roughly 50 characters
        if self.val_file:
            encoded_eval_data = self.read_and_encode(self.val_file)
            self.validation_dataset = PlaythroughDataset(encoded_eval_data.ids, self.block_size)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        if not self.validation_dataset:
            return None
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

