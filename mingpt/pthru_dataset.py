#!/usr/bin/env python
# coding: utf-8
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from torch.utils.data import Dataset


CMD_START_TOKEN = '>>>['
CMD_END_TOKEN = ']<<<'
GAME_START_CMD = 'start'



def _span_len(span):
    return span[1]-span[0]+1

class PlaythroughDataset(Dataset):
    TARGET_CMD_TOKENS = "cmd_tokens"
    TARGET_CMD_PROMPTS = "cmd_prompts"

    def __init__(self, data, block_size, cmd_markers: Tuple[int,int] = None, game_start_tok:int = None, pad_tok:int=0, span_filtering=None):
        self.block_size = block_size
        self.data = np.array(data)  # make a copy of the given list of token ids
        self.cmd_spans = None
        self.game_spans = []  # each span (index in cmd_spans of game start, index into cmd_spans of start of next game)
        self.game_start_tok = game_start_tok
        self.pad_tok = pad_tok
        self.span_filtering = span_filtering

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
                    # if np_spans[0][0] == 0:
                    #     np_spans = np_spans[1:]  # skip initial 'start' command
                    self.cmd_spans = np_spans
                    print("PlaythroughDataset cmd_spans =", self.cmd_spans)
                    current_game = [None, None]
                    for ispan, span in enumerate(self.cmd_spans):
                        assert np.all(span[0] < span[1]), f"Bad dataset: inconsistent cmd markers: {self.cmd_spans}"
                        if self.data[span[0]+1] == game_start_tok:
                            if self.data[span[0]+2] != self.cmd_end:
                                print("WARNING: skipping false start", span)
                                continue
                            if current_game[0] is None:
                                current_game[0] = ispan
                            elif current_game[1] is None:
                                current_game[1] = ispan
                                self.game_spans.append((current_game[0], current_game[1]))
                                current_game = [ispan, None]
                            else:
                                assert False, f"Shouldn't be possible: {current_game} {ispan} {span}"
                    assert ispan == len(self.cmd_spans)-1, f"{ispan} {len(self.cmd_spans)}"
                    if current_game[0] is not None:
                        assert current_game[1] is None, f"{current_game} {ispan} {span}"
                        self.game_spans.append((current_game[0], -1))
                    print("####################  # Games in dataset:", len(self.game_spans))
                    print(self.game_spans[0:3], self.game_spans[-2:])
        else:
            self.cmd_start = None
            self.cmd_end = None
            self.cmd_spans = None

        self.build_index()

    def print_info(self, name="PlaythroughDataset"):
        num_cmd_tokens = 0
        num_cmd_tokens = 0
        if self.cmd_spans is not None:
            num_spans = len(self.cmd_spans)
            for span in self.cmd_spans:
                num_cmd_tokens += _span_len(span)
        print(f"{name} filtering={self.span_filtering} datalen={len(self.data)}"
              f" len={len(self)} #games={self.num_games} #cmd_spans={num_spans} #cmd_tokens={num_cmd_tokens}")

    @property
    def num_games(self):
        return len(self.game_spans)

    def get_num_steps(self, igame:int):  # returns number of steps in the playthrough data for a single game
        if igame < 0 or igame >= len(self.game_spans):
            return None
        game_span = self.game_spans[igame]
        if game_span[1] < 0:  # last game in the dataset
            return len(self.cmd_spans) - game_span[0]
        return game_span[1] - game_span[0]  # number of cmd_spans


    def get_token_idxs(self, igame, start_step=0, end_step=-1, inclusive=(True,True)):
        # returns a span: start and end index into the token id data
        # inclusive[0]: include the command sequence at the beginning of the start_step
        # inclusive[1]: include the command sequence at the end of the end_step (if there is one)

        assert 0 <= igame < len(self.game_spans), f"{igame} {len(self.game_spans)}"
        game_span = self.game_spans[igame]
        num_game_steps = self.get_num_steps(igame)
        if end_step < 0:
            assert end_step == -1
            end_step = num_game_steps
        elif start_step >= num_game_steps or end_step > num_game_steps:
            print(f"WARNING: get_token_idxs({start_step}, {end_step}) out of range for game {igame} {game_span}")
            end_step = min(num_game_steps, end_step)
            start_step = min(num_game_steps-1, start_step)

        icmd_start = game_span[0] + start_step  # index into self.cmd_spans
        icmd_end = game_span[0] + end_step      # index into self.cmd_spans

        start_cmd_span = self.cmd_spans[icmd_start]
        if inclusive[0]:
            start_idx = start_cmd_span[0]
        else:
            start_idx = start_cmd_span[1]+1
        if icmd_end >= len(self.cmd_spans):
            end_cmd_span = (len(self.data), len(self.data)-1)  # fake span of length zero
        else:
            end_cmd_span = self.cmd_spans[icmd_end]
        if not inclusive[1] or end_step == num_game_steps:  # don't include the next cmd sequence
            end_idx = end_cmd_span[0]-1
        else:
            end_idx = end_cmd_span[1]
        return (start_idx, end_idx), _span_len(start_cmd_span), _span_len(end_cmd_span)


    def num_steps_total(self) -> int:
        n_total = 0
        for igame in range(self.num_games):
            n_total += self.get_num_steps(igame)
        return n_total

    def _add_to_index(self, value):
        pos = len(self._index)
        if value not in self._index:
            self._index[value] = pos
            assert len(self._index) == pos+1
        return len(self._index)

    def build_index(self):
        self._index = {}  # NOTE: WE DEPEND ON DICTIONARIES PRESERVING INSERTION ORDER (Python 3.6+)
        if self.cmd_spans is None:
            return  # we can't index anything
        if self.span_filtering == PlaythroughDataset.TARGET_CMD_TOKENS \
            or self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
            # index only within-game spans that end within a cmd_span
            for igame in range(self.num_games):
                # game_start_idx = self.get_token_idxs(igame)[0][0]  # idx of start of this game
                for step in range(self.get_num_steps(igame)):
                    # from token spans that end with a cmd sequence
                    if self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
                        self._add_to_index((igame, step))
                    else:  # self.span_filtering == PlaythroughDataset.TARGET_CMD_TOKENS
                        span, cmd0_len, cmd1_len = self.get_token_idxs(igame, 0, step, inclusive=(True,True))
                        game_start_idx = span[0]  # idx of start of this game
                        # if _span_len(span) >= self.block_size:
                        for j in range(cmd1_len):  # for each subspan that ends within the next_cmd token seq
                            _start_idx = span[1]-self.block_size-j+1
                            if _start_idx < game_start_idx:
                                # print(f"Discarding span {subspan} from eval index")
                                # continue  # skip this one, it doesn't satisfy reqs
                                _start_idx = game_start_idx  # clip to start of game (will be left-padded on retrieval)
                            subspan = (_start_idx, span[1]-j) # clipped span, len == block_size or less
                            self._add_to_index(subspan)  # this subspan gets included in the dataset
        else:  # index all within-game spans of len self.block_size
            if self.span_filtering:
                assert False, f"Misconfiguration Error: unrecognized span_filtering option: {self.span_filtering}"
            for igame in range(self.num_games):
                span, _, _ = self.get_token_idxs(igame)  # all the tokens for this game
                if _span_len(span) < self.block_size+1:
                    print(f"_index does not include game {igame} because it is too short {span}")
                    continue
                for j in range(_span_len(span)-self.block_size):  # for each subspan of len blocksize
                    subspan = (span[0]+j, span[0]+j+self.block_size+1)
                    self._add_to_index(subspan[0])  # this subspan gets included in the dataset
        self._index_by_idx = list(self._index)  # python array: O(1) for access by position (idx)

    def __len__(self):
        if self._index:
            return len(self._index)
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        if self._index:
            assert self._index_by_idx
            assert len(self._index) == len(self._index_by_idx)   # consistency check

            if self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
                igame, istep = self._index_by_idx[idx]
                return self.get_cmd_prompt_for_gamestep(igame, istep)

            elif self.span_filtering == PlaythroughDataset.TARGET_CMD_TOKENS:
                start_idx, end_idx = self._index_by_idx[idx]
                return self.get_left_padded_block(start_idx, end_idx)
            #else:
            print("WARNING: using legacy version of self._index", idx, self._index[idx])
            idx = self._index_by_idx[idx]
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

    def get_cmd_prompt_for_gamestep(self, igame:int, istep:int, continuation=-1, fill_id=None):
        if self.cmd_spans is None or not self.game_spans:
            assert self.cmd_spans
            assert self.game_spans
            return None, None
        icmd = 0
        if fill_id is None:
            fill_id = self.pad_tok
        gamestep_span, _, cmd1_len = self.get_token_idxs(igame, 0, istep, inclusive=(True,True))
        cmd_span = (gamestep_span[1]-cmd1_len+1, gamestep_span[1])
        # np_icmd = np.where((self.cmd_spans == cmd_span).all(axis=1))[0]
        # if len(np_icmd):
        #     icmd = np_icmd[0]  # pull out the value
        # else:
        #     print(f"Failed to find {cmd_span} in self.cmd_spans")
        return self._prompt_for_cmd_span(cmd_span[0], cmd_span[1], game_start_idx=gamestep_span[0],
                                         continuation=continuation, fill_id=fill_id)

    def get_cmd_prompt(self, icmd:int, continuation=-1, fill_id=None):
        """ returns a span that ends with the ith cmd_start marker
        (of length block_size, or less if the cmd marker position is less than block_size).
        if continuation > 0, span length is extended, and x_out is padded with fill_id
        if continuation < 0, continuation is auto-adjusted to a length up to and including corresponding cmd_end_marker
        """
        if self.cmd_spans is None or icmd >= len(self.cmd_spans):
            return None, None
        cmd_start_idx, cmd_end_idx = self.cmd_spans[icmd]  # span includes the start,end markers
        if fill_id is None:
            fill_id = self.pad_tok
        return self._prompt_for_cmd_span(cmd_start_idx, cmd_end_idx, continuation=continuation, fill_id=fill_id)

    def _prompt_for_cmd_span(self, cmd_start_idx, cmd_end_idx, game_start_idx=0,
                             continuation=-1, fill_id=None):
        if fill_id is None:
            fill_id = self.pad_tok
        if continuation < 0:
            continuation = cmd_end_idx - cmd_start_idx   # cmd_len
        cmd_start_pos = self.block_size-1   # where in the output buffer the start marker will be
        prompt_len = self.block_size
        start_idx = cmd_start_idx - self.block_size + 1
        if start_idx < game_start_idx:
            prompt_len -= (game_start_idx - start_idx)
            cmd_start_pos -= (game_start_idx - start_idx)
            start_idx = game_start_idx

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
        return torch.tensor(x_out, dtype=torch.long), torch.tensor(y_out, dtype=torch.long), torch.tensor(cmd_start_pos)

    def get_left_padded_block(self, start_idx, end_idx, fill_id=None):
        if fill_id is None:
            fill_id = self.pad_tok
        if end_idx >= len(self.data):
            # assert False, "THIS SHOULD NOT HAPPEN!"
            print(f"ASSERTION FAILURE get_left_padded_block({start_idx},{end_idx}) data len={len(self.data)}!!!")
            end_idx = len(self.data)-1
            start_idx = end_idx - self.block_size+1
        if start_idx < 0:
            start_idx = 0;
        output_len = end_idx - start_idx + 1
        if output_len > self.block_size:
            print(f"WARNING (UNEXPECTED!): get_left_padded_block({start_idx},{end_idx}) will truncate block to len={self.block_size}")
            start_idx += output_len - self.block_size
            output_len = self.block_size
        pad_length = self.block_size - output_len
        if pad_length > 0:    # output_len < self.block_size:
            x_out = np.full(self.block_size, fill_value=fill_id)
            y_out = np.full(self.block_size, fill_value=fill_id)
            x_out[-output_len:] = self.data[start_idx:start_idx+output_len]
            y_out[-output_len:] = self.data[start_idx+1:start_idx+1+output_len]
        else:
            x_out = self.data[start_idx:start_idx+output_len]
            y_out = self.data[start_idx+1:start_idx+1+output_len]
        # print(x_out)
        return torch.tensor(x_out, dtype=torch.long), torch.tensor(y_out, dtype=torch.long), torch.tensor(pad_length)

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
        train_filtering = None,
        eval_filtering = PlaythroughDataset.TARGET_CMD_TOKENS,
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
        self.train_filtering = train_filtering
        self.eval_filtering = eval_filtering

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
        self.cmd_start_marker = self.tokenizer.token_to_id(CMD_START_TOKEN)
        self.cmd_end_marker = self.tokenizer.token_to_id(CMD_END_TOKEN)
        self.game_start_tok = self.tokenizer.token_to_id(GAME_START_CMD)
        if not self.vocab_size:
            self.vocab_size = len(self.vocab_dict)
            # self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)  # this seems to be wrong!
            # (maybe tokenizers=0.10.0rc1 impl of WordLevel model has a bug?
            assert self.vocab_size == len(self.vocab_dict)
        # TODO: get dataset length after loading data and use it to compute final_tokens
        encoded_data = self.read_and_encode(self.data_file)
        print("PlaythroughDataModule.prepare_data: ", len(encoded_data.ids))

        cmd_markers = (self.cmd_start_marker, self.cmd_end_marker)
        self.train_dataset = PlaythroughDataset(encoded_data.ids, self.block_size,
                                                cmd_markers=cmd_markers,
                                                game_start_tok=self.game_start_tok,
                                                span_filtering=self.train_filtering)  #PlaythroughDataset.TARGET_CMD_TOKENS)

        if self.val_file:
            eval_encoded = self.read_and_encode(self.val_file)
            self.validation_dataset = PlaythroughDataset(eval_encoded.ids, self.block_size,
                                                         cmd_markers=cmd_markers,
                                                         game_start_tok=self.game_start_tok,
                                                         span_filtering=self.eval_filtering)   #PlaythroughDataset.TARGET_CMD_TOKENS)
                                        #    span_filtering = PlaythroughDataset.TARGET_CMD_PROMPTS)  # eval accuracy

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

        if self.validation_dataset.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
            batch_size = 1
        else:
            batch_size = self.batch_size

        loader = DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

