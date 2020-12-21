#!/usr/bin/env python
# coding: utf-8
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from torch.utils.data import Dataset

class PlaythroughDataset(Dataset):

    def __init__(self, data, block_size, cmd_markers=Tuple[int,int]):
        data_size = len(data)
        print("PlaythroughDataset datalen=", data_size)
        self.block_size = block_size
        self.data = np.array(data)  # make a copy of the given list of token ids
        self.cmd_spans = None
        if cmd_markers:
            self.cmd_start = cmd_markers[0]
            self.cmd_end = cmd_markers[1]
            cmd_start_idxs = np.where(self.data == self.cmd_start)[0]
            if cmd_start_idxs.size > 0:
                cmd_end_idxs = np.where(self.data[cmd_start_idxs[0]:] == self.cmd_end)[0]
                if cmd_end_idxs.size > 0:
                    if cmd_end_idxs.size < cmd_start_idxs.size:  # fewer end markers than starts
                        cmd_start_idxs = cmd_start_idxs[:cmd_end_idxs.size]  # truncate to same length
                    np_spans = np.stack((cmd_start_idxs, cmd_end_idxs), axis=1)
                    if np_spans[0][0] == 0:
                        np_spans = np_spans[1:]  # skip initial 'start' command
                    self.cmd_spans = np_spans
                    print("PlaythroughDataset cmd_spans =", self.cmd_spans)
                    for span in self.cmd_spans:
                        assert np.all(span[0] < span[1]), f"Bad dataset: inconsistent cmd markers: {self.cmd_spans}"
        else:
            self.cmd_start = None
            self.cmd_end = None
            self.cmd_spans = None

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

    def get_cmd_prompt(self, icmd:int, continuation=-1, fill_id=0):
        """ returns a span that ends with the ith cmd_start marker
        (of length block_size, or less if the cmd marker position is less than block_size).
        if continuation > 0, span length is extended, and x_out is padded with fill_id
        if continuation < 0, continuation is auto-adjusted to a length up to and including corresponding cmd_end_marker
        """
        if self.cmd_spans is None or icmd >= len(self.cmd_spans):
            return None, None
        cmd_start_idx, cmd_end_idx = self.cmd_spans[icmd]  # span includes the start,end markers
        cmd_start_pos = self.block_size-1   # where in the output buffer the start marker will be
        prompt_len = self.block_size
        start_idx = cmd_start_idx - self.block_size + 1
        if start_idx < 0:
            prompt_len += start_idx
            cmd_start_pos += start_idx
            start_idx = 0
        if continuation < 0:
            continuation = cmd_end_idx - cmd_start_idx

        output_len = prompt_len + continuation
        x_len = output_len - cmd_start_pos
        if start_idx + x_len >= len(self.data):  # NOTE: numpy automatically truncates slices at max pos
            x_len = len(self.data) - start_idx
        x_out = np.full(output_len, fill_value=fill_id)
        y_out = np.full(output_len, fill_value=fill_id)
        # print(f"({cmd_start_idx,cmd_end_idx}) output_len={output_len} start_idx={start_idx} cmd_start_pos={cmd_start_pos} x_len={x_len}")
        x_out[:output_len] = self.data[start_idx:start_idx+cmd_start_pos+x_len]
        y_out[:output_len] = self.data[start_idx+1:start_idx+cmd_start_pos+x_len+1]
        # print(x_out)
        return torch.tensor(x_out, dtype=torch.long), torch.tensor(y_out, dtype=torch.long), cmd_start_pos


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
        self.vocab_dict = {}
        self.cmd_start_marker = None
        self.cmd_end_marker = None

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
        self.vocab_dict = self.tokenizer.get_vocab(with_added_tokens=True)
        self.cmd_start_marker = self.tokenizer.token_to_id('>>>[')
        self.cmd_end_marker = self.tokenizer.token_to_id(']<<<')
        if not self.vocab_size:
            self.vocab_size = len(self.vocab_dict)
            # self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)  # this seems to be wrong!
            # (maybe tokenizers=0.10.0rc1 impl of WordLevel model has a bug?
            assert self.vocab_size == len(self.vocab_dict)
        # TODO: get dataset length after loading data and use it to compute final_tokens
        encoded_data = self.read_and_encode(self.data_file)
        print("PlaythroughDataModule.prepare_data: ", len(encoded_data.ids))

        cmd_markers = (self.cmd_start_marker, self.cmd_end_marker)
        self.train_dataset = PlaythroughDataset(encoded_data.ids, self.block_size, cmd_markers=cmd_markers)

        if self.val_file:
            eval_encoded = self.read_and_encode(self.val_file)
            self.validation_dataset = PlaythroughDataset(eval_encoded.ids, self.block_size, cmd_markers=cmd_markers)

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

