#!/usr/bin/env python
# coding: utf-8
from typing import List, Dict, Optional, Any, Tuple, Iterable, Mapping
import os
import logging
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from pytorch_lightning import LightningDataModule


logger = logging.getLogger(__name__)

MAX_PTHRU_STEPS = 30

CMD_START_TOKEN = '>>>['
CMD_END_TOKEN = ']<<<'
GAME_START_CMD = 'start'
PAD_TOKEN = '<PAD>'

def _span_len(span):
    return span[1]-span[0]+1  # includes 2 extra tokens: cmd_start, cmd_end

# NOTE: in most of the code for this project, cmd_start_pos points to the cmd_start token
# but cmd_len is _span_len(cmd_span)-1  (it includes only the cmd_end token, not the cmd_start)

class PlaythroughDataset(Dataset):
    TARGET_CMD_TOKENS = "cmd_tokens"     # one data sample per cmd token of each step of each game (ending on each token)
    TARGET_CMD_PROMPTS = "cmd_prompts"   # one data sample per step of each game (ending on the cmd_end token)

    def __init__(self, data, block_size, vocab_size: int = 0,
                 cmd_markers: Tuple[int,int] = None, game_start_tok: int = None,
                 pad_tok: int = 0, span_filtering=None, batch_size=1, prompt_extra_len=None):
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.data = np.array(data)  # make a copy of the given list of token ids
        self.cmd_spans = None
        self.game_spans = []  # each span (index in cmd_spans of game start, index into cmd_spans of start of next game)
        self.game_start_tok = game_start_tok
        self.pad_tok = pad_tok
        self.span_filtering = span_filtering
        self.prompt_extra_len = prompt_extra_len if prompt_extra_len is not None else -10

        assert cmd_markers, "REQUIRED: token ids for cmd_start, cmd_end"
        print("cmd_markers = ", cmd_markers)
        self.cmd_start = cmd_markers[0]
        self.cmd_end = cmd_markers[1]
        cmd_start_idxs = np.where(self.data == self.cmd_start)[0]
        if cmd_start_idxs.size > 0:
            cmd_end_idxs = np.where(self.data == self.cmd_end)[0]
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
                print(f"####################  # Games in dataset: {len(self.game_spans)}  # cmds: {len(self.cmd_spans)}")
                print(self.game_spans[0:3], self.game_spans[-2:])

        self.build_index()

    def __len__(self):
        if self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
            return len(self._index_by_idx) if self._index_by_idx else 0
        elif self.span_filtering == PlaythroughDataset.TARGET_CMD_TOKENS:
            return len(self._tokspans) if self._tokspans else 0
        return len(self.data) - self.block_size


    def print_info(self, name="PlaythroughDataset"):
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

    def num_steps_total(self) -> int:
        n_total = 0
        for igame in range(self.num_games):
            n_total += self.get_num_steps(igame)
        return n_total

    def _add_to_index(self, value):
        nextpos = len(self._index)
        if value not in self._index:
            self._index[value] = nextpos
            assert len(self._index) == nextpos+1
        return len(self._index)

    def _add_to_tokspan_index(self, spanwlen):
        nextpos = len(self._tokspans)
        if spanwlen not in self._tokspans:
            self._tokspans[spanwlen] = nextpos
            assert len(self._tokspans) == nextpos+1
        return len(self._tokspans)

    def build_index(self):
        self._index = {}  # NOTE: HERE WE DEPEND ON DICTIONARIES PRESERVING INSERTION ORDER (Python 3.6+)
        self._tokspans = {}
        if self.cmd_spans is None:
            return  # we can't index anything
        if self.span_filtering == PlaythroughDataset.TARGET_CMD_TOKENS \
            or self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
            # index only within-game spans that end within a cmd_span
            for igame in range(self.num_games):
                # game_start_idx = self.get_token_idx_spans(igame)[0][0]  # idx of start of this game
                for step in range(self.get_num_steps(igame)):
                    # from token spans that end with a cmd sequence
                    # if self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
                    # one data sample per step of each game
                    self._add_to_index((igame, step))
                    # else:  # self.span_filtering == PlaythroughDataset.TARGET_CMD_TOKENS  # return a record ending at each tokan

                    # one data sample per cmd token of each step of each game
                    span, cmd0_span, cmd1_span = self.get_token_spans(igame, 0, step, inclusive=(True, True))
                    #cmd0_len = _span_len(cmd0_span)
                    cmd1_len = _span_len(cmd1_span)
                    game_start_idx = span[0]  # idx of start of this game
                    cmd_start_idx = cmd1_span[0]
                    # if _span_len(span) >= self.block_size:
                    for j in range(cmd1_len):  # for each subspan that ends within the next_cmd token seq
                        _start_idx = span[1]-self.block_size-j+1
                        if _start_idx < game_start_idx:
                            # print(f"Discarding span {subspan} from eval index")
                            # continue  # skip this one, it doesn't satisfy reqs
                            _start_idx = game_start_idx  # clip to start of game (block will be padded on retrieval)
                        subspan = (_start_idx, span[1]-j, cmd_start_idx, cmd1_len-j) # clipped span, len == block_size or less
                        self._add_to_tokspan_index(subspan)  # this subspan gets included in the dataset
        else:  # index all within-game spans of len self.block_size
            if self.span_filtering:
                assert False, f"Misconfiguration Error: unrecognized span_filtering option: {self.span_filtering}"
            for igame in range(self.num_games):
                span, _, _, _ = self.get_token_spans(igame)  # all the tokens for this game
                if _span_len(span) < self.block_size+1:
                    print(f"_index does not include game {igame} because it is too short {span}")
                    continue
                for j in range(_span_len(span)-self.block_size):  # for each subspan of len blocksize
                    subspan = (span[0]+j, span[0]+j+self.block_size+1, -1, -1)
                    self._add_to_tokspan_index(subspan)  # this subspan gets included in the dataset
        # optimization: create secondary index for access by position (idx) in linear time - python array: O(1)
        self._index_by_idx = [gamestep for gamestep in self._index if gamestep[1] > 0]
        self._index_tokspans_by_idx = list(self._tokspans)

    def get_token_spans(self, igame, start_step=0, end_step=-1, inclusive=(True, True)):
        # returns 3 spans: start and end index into the token id data
        # one span for each of 1) the game, 2 & 3) the cmd_seq for start_step & end_step
        # inclusive[0]: include the command sequence at the beginning of the start_step
        # inclusive[1]: include the command sequence at the end of the end_step (if there is one)

        assert 0 <= igame < len(self.game_spans), f"{igame} {len(self.game_spans)}"
        game_span = self.game_spans[igame]
        num_game_steps = self.get_num_steps(igame)
        if end_step < 0:
            assert end_step == -1
            end_step = num_game_steps
        elif start_step >= num_game_steps or end_step > num_game_steps:
            print(f"WARNING: get_token_idx_spans({start_step}, {end_step}) out of range for game {igame} {game_span}")
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
        return (start_idx, end_idx), start_cmd_span, end_cmd_span

    def _remap_idx(self, idx):
        # return len(self._index_by_idx)-idx-1   # for debugging
        return idx

    def get_game_step(self, idx):   # from *unshuffled* dataset idx get (igame, istep)  [only for validation & test]
        idx = self._remap_idx(idx)
        igame, istep = self._index_by_idx[idx]
        nsteps = self.get_num_steps(igame)
        return igame, istep, nsteps

    def __getitem__(self, idx):
        idx = self._remap_idx(idx)
        # grab a chunk of (block_size + 1) characters from the data
        if self._index:
            assert self._index_by_idx  # consistency check
            assert len(self._index) == len(self._index_by_idx) + self.num_games  # initial >>[ start ]<< is excluded from _index_by_idx

            if self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
                igame, istep, _nsteps_ = self.get_game_step(idx)
                start_idx, output_len, cmd_start_idx, cmd_len = self.get_cmd_prompt_for_gamestep(igame, istep,
                                                                                        fetch_data=False,
                                                                                        block_size=(-1 if self.prompt_extra_len else 0),
                                                                                        continuation=self.prompt_extra_len)  # +random extra len from 0 to N
                # if False:
                #     return (*self.fetch_data(start_idx, output_len, cmd_start_idx, cmd_len, pad_left=False, fill_id=self.pad_tok), cmd_len)

            elif self.span_filtering == PlaythroughDataset.TARGET_CMD_TOKENS:
                start_idx, end_idx, cmd_start_idx, cmd_len = self._index_tokspans_by_idx[idx]
                start_idx, output_len, cmd_start_idx = self._limit_to_block_size(start_idx, end_idx, cmd_start_idx)
                # if False:
                #     return (*self.fetch_data(start_idx, output_len, cmd_start_idx, cmd_len, pad_left=True, fill_id=self.pad_tok), cmd_len)
            else:
                assert False, f"UNSUPPORTED span_filtering={self.span_filtering} ({idx}:{self._index[idx]})"
            return start_idx, output_len, cmd_start_idx, cmd_len

        # """
        # arrange data and targets so that the first i elements of x
        # will be asked to predict the i-th element of y. Notice that
        # the eventual language model will actually make block_size
        # individual predictions at the same time based on this data,
        # so we are being clever and amortizing the cost of the forward
        # pass of the network. So for example if block_size is 4, then
        # we could e.g. sample a chunk of text "hello", the integers in
        # x will correspond to "hell" and in y will be "ello". This will
        # then actually "multitask" 4 separate examples at the same time
        # in the language model:
        # - given just "h", please predict "e" as next
        # - given "he" please predict "l" next
        # - given "hel" predict "l" next
        # - given "hell" predict "o" next
        #
        # In addition, because the DataLoader will create batches of examples,
        # every forward/backward pass during traning will simultaneously train
        # a LOT of predictions, amortizing a lot of computation. In particular,
        # for a batched input of integers X (B, T) where B is batch size and
        # T is block_size and Y (B, T), the network will during training be
        # simultaneously training to make B*T predictions, all at once! Of course,
        # at test time we can paralellize across batch B, but unlike during training
        # we cannot parallelize across the time dimension T - we have to run
        # a forward pass of the network to recover the next single character of the
        # sequence along each batch dimension, and repeatedly always feed in a next
        # character to get the next one.
        #
        # So yes there is a big asymmetry between train/test time of autoregressive
        # models. During training we can go B*T at a time with every forward pass,
        # but during test time we can only go B at a time, T times, with T forward
        # passes.
        # """
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def get_cmd_prompt_for_gamestep(self, igame:int, istep:int, continuation=-1, fill_id=None, fetch_data=True,
                                    block_size: int = -1  # default: self.block_size ;
                                    # if block_size==0: return continuation as y instead of merging it into x
                                    ):
        if self.cmd_spans is None or not self.game_spans:
            assert self.cmd_spans
            assert self.game_spans
            return None, None
        if fill_id is None:
            fill_id = self.pad_tok
        gamepthru_span, _, cmd1_span = self.get_token_spans(igame, 0, istep, inclusive=(True, True))
        cmd1_len = _span_len(cmd1_span)
        cmd_span = (gamepthru_span[1]-cmd1_len+1, gamepthru_span[1])
        assert _span_len(cmd1_span) == _span_len(cmd_span), f"{cmd1_span} {cmd_span}"
        assert cmd1_span[0] == cmd_span[0] and cmd1_span[1] == cmd_span[1], f"{cmd1_span} {cmd_span}"

        merge_x_with_continuation = True
        if block_size == -1:
            block_size = _span_len(gamepthru_span)  # - cmd1_len - 1 # full length of playthrough up to istep cmd
            if block_size > self.block_size:
                block_size = self.block_size
            # print(f"get_cmd_prompt_for_game={igame}_step={istep} AUTO block_size={block_size} ({_span_len(gamepthru_span)})")
        elif block_size < 0:
            block_size = self.block_size
        else:  # block_size == 0
            block_size = self.block_size
            merge_x_with_continuation = False
            # print("block_size 0 -- not merging continuation into x")

        ####### 05-16 INLINED _prompt_for_cmd_span(), which was only used in this one place
        cmd_start_idx, cmd_end_idx = cmd_span  # cmd_span[0], cmd_span[1]
        game_start_idx = gamepthru_span[0]
        # start_idx, output_len, cmd_start_idx, _cmd_len_ = self._prompt_for_cmd_span(cmd_span[0], cmd_span[1],
        #                                                                  game_start_idx=gamepthru_span[0],
        #                                                                  continuation=continuation,
        #                                                                  fill_id=fill_id,
        #                                                                  block_size=block_size)
        if continuation == -1:  # -1 by default
            continuation = cmd_end_idx - cmd_start_idx  # cmd_len
        elif continuation < -1:
            continuation = cmd_end_idx - cmd_start_idx + random.randint(0, -continuation)  # cmd_len + some extra randomization

        cmd_start_pos = block_size-1   # where in the output buffer the start marker will be
        prompt_len = block_size
        start_idx = cmd_start_idx - block_size + 1
        if start_idx < game_start_idx:
            prompt_len -= (game_start_idx - start_idx)
            cmd_start_pos -= (game_start_idx - start_idx)
            start_idx = game_start_idx

        output_len = prompt_len
        if merge_x_with_continuation:
            output_len += continuation
        x_len = output_len - cmd_start_pos   # NOTE: can be negative e.g. if output_len is shorter than block_size

        if start_idx + x_len >= len(self.data):  # NOTE: numpy automatically truncates slices at max pos
            x_len = len(self.data) - start_idx

        # adjust to block_size if output len is greater than block_size
        if self.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:  # and (self.batch_size is not None and self.batch_size > 1):
            # x_out = np.full(self.block_size, fill_value=fill_id)
            # y_out = np.full(self.block_size, fill_value=fill_id)
            if output_len > self.block_size:
                # adjust so that we output exactly block_size
                diff_len = output_len - self.block_size
                output_len -= diff_len  # = self.block_size
                cmd_start_pos -= diff_len
                start_idx += diff_len
            elif output_len <= self.block_size:
                # if align_cmds:  # ragged lengths are Ok, pad_collate() will add right padding if needed
                #     diff_len = 0
                # else:  # this is how it used to be -- pad on the right if needed
                #     diff_len = self.block_size - output_len
                #     output_len = self.block_size
                #     x_len += diff_len
                diff_len = 0
        else:
            # x_out = np.full(output_len, fill_value=fill_id)
            # y_out = np.full(output_len, fill_value=fill_id)
            pass
        # print(f"({cmd_start_idx,cmd_end_idx}) output_len={output_len} start_idx={start_idx} cmd_start_pos={cmd_start_pos} x_len={x_len}")
        assert output_len == cmd_start_pos+x_len
        assert cmd_start_pos == cmd_start_idx - start_idx
        assert x_len == output_len - (cmd_start_idx - start_idx)
        # return start_idx, output_len, cmd_start_idx, cmd_end_idx-cmd_start_idx
        _cmd_len_ = cmd_end_idx - cmd_start_idx

        # print(f"{_span_len(cmd_span)} {_cmd_len_}")
        assert _span_len(cmd_span) == cmd1_len
        assert _cmd_len_ == cmd1_len-1
        if fetch_data:
            # print("fetch_data==True:", start_idx, output_len, cmd_start_idx)
            x, y, cmd_start_pos = self._get_data_tensors(start_idx, output_len, self.block_size, cmd_start_idx,
                                                         pad_left=0, fill_id=fill_id)
            return x, y, cmd_start_pos, cmd1_len-1
        else:
            return start_idx, output_len, cmd_start_idx, cmd1_len-1

    def _limit_to_block_size(self, start_idx, end_idx, cmd_start_idx):
        if end_idx >= len(self.data):
            # assert False, "THIS SHOULD NOT HAPPEN!"
            print(f"ASSERTION FAILURE _limit_to_block_size({start_idx},{end_idx}) data len={len(self.data)}!!!")
            end_idx = len(self.data)-1
            # start_idx = end_idx - self.block_size+1
        if start_idx < 0:
            start_idx = 0;
        output_len = end_idx - start_idx + 1
        if output_len > self.block_size:
            print(f"ASSERTION FAILURE! _limit_to_block_size({start_idx},{end_idx}) truncating to len={self.block_size}")
            start_idx += output_len - self.block_size
            output_len = self.block_size
        # pad_length = self.block_size - output_len
        if output_len <= 0 or start_idx < 0 or start_idx+output_len > len(self.data):
            logger.error(f"ASSERTION FAILURE! _limit_to_block_size({start_idx},{end_idx}) truncating to len={self.block_size}")
            assert False

        return (start_idx, output_len, cmd_start_idx)
        # return self.get_data_tensor(start_idx, output_len, cmd_start_idx, pad_left)

    def _get_data_tensors(self, start_idx, output_len, buffer_size, cmd_start_idx, pad_left=0, fill_id=0):
        if pad_left > 0:    # output_len < self.block_size:
            #logger.info(f"_get_data_tensors(pad_left) {start_idx} {output_len} {buffer_size} {self.data.shape}")
            x_out = np.full(buffer_size, fill_value=fill_id)
            y_out = np.full(buffer_size, fill_value=fill_id)
            x_out[pad_left:pad_left+output_len] = self.data[start_idx:start_idx+output_len]
            y_out[pad_left-1:pad_left+output_len] = self.data[start_idx:start_idx+output_len+1]
                                                        #self.data[start_idx+1:start_idx+1+output_len]
        elif output_len < buffer_size: #pad_right
            # logger.info(f"_get_data_tensors(pad_right) {start_idx} {output_len} {buffer_size} {self.data.shape}")
            if output_len <= 0:
                assert False, f"{output_len}"
            if start_idx >= len(self.data):
                assert False, f"{self.data.shape}"
            x_out = np.full(buffer_size, fill_value=fill_id)
            y_out = np.full(buffer_size, fill_value=fill_id)
            x_out[:output_len] = self.data[start_idx:start_idx + output_len]
            y_out[:output_len] = self.data[start_idx + 1:start_idx + 1 + output_len]
            # logger.info(f"_get_data_tensors(pad_right:{start_idx}) {x_out.shape} {y_out.shape} {self.data.shape}")
            # cmd_start_pos = cmd_start_idx - start_idx
        else:
            x_out = self.data[start_idx:start_idx+output_len]
            y_out = self.data[start_idx+1:start_idx+1+output_len]
        cmd_start_pos = cmd_start_idx - start_idx
        if pad_left:
            cmd_start_pos += pad_left
        # print(x_out)
        return torch.tensor(x_out, dtype=torch.long), torch.tensor(y_out, dtype=torch.long), cmd_start_pos

    def pad_collate(self, batch):
        # print(f"cmd_start:{self.cmd_start} cnd_end:{self.cmd_end} {len(batch)}", type(batch), type(batch[0]))  # a list of tuples, len=batch_size
        # print(len(batch[0]))  # 3 : each tuple = (start_idx, output_len, cmd_start_idx)
        max_output_len = 0
        for _, output_len, _, _cmd_len_ in batch:
            if output_len > max_output_len:
                max_output_len = output_len
        # print(f"align_cmd=False max_output_len={max_output_len}")

        xx = []
        yy = []
        cmd_start_pos = []
        cmd_len = []

        for i, (start_idx, output_len, cmd_start_idx, _cmd_len_) in enumerate(batch):
            # logger.info(f"[{i}] output_len={output_len} (start={start_idx} cmd_start={cmd_start_idx}")
            if not output_len > 0:
                err_msg = f"[{i}] UNEXPECTED! output_len={output_len} (start={start_idx} cmd_start={cmd_start_idx}"
                logger.error(err_msg)
                assert False, err_msg
            x, y, cmd_pos = self._get_data_tensors(start_idx, output_len, max_output_len, cmd_start_idx,
                                                       pad_left=0, fill_id=self.pad_tok)
            xx.append(x); yy.append(y); cmd_start_pos.append(cmd_pos), cmd_len.append(_cmd_len_)

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.pad_tok)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.pad_tok)
        for i in range(len(cmd_start_pos)):
            assert len(xx_pad[i]) == len(yy_pad[i]), f"[{i}] {xx_pad[i]} {yy_pad[i]}"
            if xx_pad[i,cmd_start_pos[i]] != self.cmd_start:
                err_msg = f"[{i}:{cmd_start_pos[i]}] {xx_pad[i, cmd_start_pos[i]]} {xx_pad[i,:]}"
                print(err_msg)
                assert xx_pad[i, cmd_start_pos[i]] == self.cmd_start, err_msg
        return xx_pad, yy_pad, cmd_start_pos, cmd_len

    def pad_collate_for_eval(self, batch):
        # print(f"cmd_start:{self.cmd_start} cnd_end:{self.cmd_end} {len(batch)}", type(batch), type(batch[0]))  # a list of tuples, len=batch_size
        # print(len(batch[0]))  # 3 : each tuple = (start_idx, output_len, cmd_start_idx)
        max_output_len, max_tail, align_pos = 0, 0, 0
        for start_idx, output_len, cmd_start_idx, _cmd_len in batch:
            # print(start_idx, cmd_start_idx-start_idx, output_len )
            cmd_pos = cmd_start_idx - start_idx
            tail_len = _cmd_len
            assert 0 <= cmd_pos < output_len, f"{cmd_start_idx} {start_idx} {output_len}"
            assert output_len <= self.block_size, f"{output_len} {self.block_size}"
            if tail_len > max_tail:
                max_tail = tail_len
            if output_len > max_output_len:
                max_output_len = output_len
        # print(f"max_tail={max_tail} max_output_len={max_output_len} align_pos={align_pos}")

        xx = []
        yy = []
        cmd_start_pos = []
        cmd_len = []
        buffer_size = max_output_len   #self.block_size
        for i, (start_idx, output_len, cmd_start_idx, _cmd_len_) in enumerate(batch):
            cmd_pos = cmd_start_idx - start_idx
            cmd_start_idx += 1   # point to the first token after the special [cmd_start] token
            # logger.info(f"[{i}] output_len={output_len} (start={start_idx} cmd_start={cmd_start_idx}")
            if not output_len > 0:
                err_msg = f"[{i}] UNEXPECTED! output_len={output_len} (start={start_idx} cmd_start={cmd_start_idx}"
                logger.error(err_msg)
                assert False, err_msg
            assert start_idx + output_len == cmd_start_idx, f"{start_idx}+{output_len} == {cmd_start_idx}"
            if output_len < buffer_size:  # pad_left
                # logger.info(f"_get_data_tensors(pad_right) {start_idx} {output_len} {buffer_size} {self.data.shape}")
                if output_len <= 0:
                    assert False, f"{output_len}"
                if start_idx >= len(self.data):
                    assert False, f"{self.data.shape}"
                pad_left = buffer_size - output_len
                x_out = np.full(buffer_size, fill_value=self.pad_tok)
                x_out[pad_left:pad_left + output_len] = self.data[start_idx:start_idx + output_len]
                cmd_pos += pad_left
            else:
                x_out = self.data[start_idx:start_idx + output_len]
            y_out = self.data[cmd_start_idx:cmd_start_idx + _cmd_len_]

            assert len(y_out) == _cmd_len_
            xx.append(torch.tensor(x_out, dtype=torch.long))
            yy.append(torch.tensor(y_out, dtype=torch.long))
            cmd_start_pos.append(cmd_pos)
            cmd_len.append(_cmd_len_)

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.pad_tok)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.pad_tok)
        for i in range(len(cmd_start_pos)):
            if xx_pad[i, cmd_start_pos[i]] != self.cmd_start:
                err_msg = f"[{i}:{cmd_start_pos[i]}] {xx_pad[i, cmd_start_pos[i]]} {xx_pad[i, :]}"
                print(err_msg)
                assert xx_pad[i, cmd_start_pos[i]] == self.cmd_start, err_msg
        return xx_pad, yy_pad, cmd_start_pos, cmd_len

    def pad_collate_aligned(self, batch):
        # print(f"cmd_start:{self.cmd_start} cnd_end:{self.cmd_end} {len(batch)}", type(batch), type(batch[0]))  # a list of tuples, len=batch_size
        # print(len(batch[0]))  # 3 : each tuple = (start_idx, output_len, cmd_start_idx)
        max_output_len, max_tail, align_pos = 0, 0, 0
        for start_idx, output_len, cmd_start_idx, _cmd_len in batch:
            # print(start_idx, cmd_start_idx-start_idx, output_len )
            cmd_pos = cmd_start_idx - start_idx
            tail_len = output_len - cmd_pos
            assert 0 <= cmd_pos < output_len, f"{cmd_start_idx} {start_idx} {output_len}"
            assert output_len <= self.block_size, f"{output_len} {self.block_size}"
            if tail_len > max_tail:
                max_tail = tail_len
            # if cmd_pos > align_pos:  # move align_pos to the right as far as possible
            #     align_pos = min(cmd_pos, self.block_size - max_tail)
            align_pos = min(max(cmd_pos, align_pos), self.block_size - max_tail)
            max_output_len = max(align_pos + tail_len, max_output_len)
            assert max_output_len <= self.block_size, f"{max_output_len} {self.block_size}"
        # print(f"max_tail={max_tail} max_output_len={max_output_len} align_pos={align_pos}")

        xx = []
        yy = []
        cmd_start_pos = []
        cmd_len = []

        for i, (start_idx, output_len, cmd_start_idx, _cmd_len_) in enumerate(batch):
            # logger.info(f"[{i}] output_len={output_len} (start={start_idx} cmd_start={cmd_start_idx}")
            if not output_len > 0:
                err_msg = f"[{i}] UNEXPECTED! output_len={output_len} (start={start_idx} cmd_start={cmd_start_idx}"
                logger.error(err_msg)
                assert False, err_msg

            cmd_pos = cmd_start_idx - start_idx
            tail_len = output_len - cmd_pos
            assert align_pos + tail_len <= max_output_len, f"{align_pos} +{tail_len} <= {max_output_len} ({max_tail})"
            delta_shift = align_pos - cmd_pos
            if delta_shift > 0:  # need to pad on the left
                x, y, cmd_pos = self._get_data_tensors(start_idx, output_len, max_output_len, cmd_start_idx,
                                                       pad_left=delta_shift, fill_id=self.pad_tok)
            else:
                if delta_shift < 0:  # need to truncate on the left
                    start_idx -= delta_shift  # NOTE: delta_shift is negative: start_idx gets INCREMENTED
                    output_len += delta_shift  # NOTE: delta_shift is negative: output_len gets DECREMENTED
                x, y, cmd_pos = self._get_data_tensors(start_idx, output_len, max_output_len, cmd_start_idx,
                                                       pad_left=0, fill_id=self.pad_tok)

            xx.append(x);
            yy.append(y);
            cmd_start_pos.append(cmd_pos), cmd_len.append(_cmd_len_)

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.pad_tok)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.pad_tok)
        assert yy_pad.shape[1] == max_tail, f"{yy_pad.shape} {_max_tail}"
        for i in range(len(cmd_start_pos)):
            assert len(xx_pad[i]) == len(yy_pad[i]), f"[{i}] {xx_pad[i]} {yy_pad[i]}"
            if xx_pad[i, cmd_start_pos[i]] != self.cmd_start:
                err_msg = f"[{i}:{cmd_start_pos[i]}] {xx_pad[i, cmd_start_pos[i]]} {xx_pad[i, :]}"
                print(err_msg)
                assert xx_pad[i, cmd_start_pos[i]] == self.cmd_start, err_msg
        return xx_pad, yy_pad, cmd_start_pos, cmd_len


def _check_skills_list(filepath):
    print(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
        while not first_line:
            first_line = f.readline()
        first_rec = json.loads(first_line)
        if first_rec['skills']:
            skills_list = first_rec['skills']
            if len(skills_list) == 1 and ',' in skills_list[0]:
                print(f"NEED TO FIX skills list in {filepath} {skills_list}")
                return False
    return True


def _fix_skills_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        with open(filepath + ".fixed", "w", encoding="utf-8") as fout:
            json_lines = f.readlines()
            for line in json_lines:
                line = line.strip()
                if line:  # filter out any empty lines
                    json_rec = json.loads(line)
                    if 'skills' in json_rec and len(json_rec['skills']) == 1:
                        json_rec['skills'] = [skill for skill in json_rec['skills'][0].split(',')]  # split the list of skills
                    fout.write(json.dumps(json_rec))
                    fout.write("\n")
    os.rename(filepath, filepath+".bad_skills")
    os.rename(filepath+".fixed", filepath)
    return True


class PlaythroughDataModule(LightningDataModule):
    name = "ftwc_pthru"

    def __init__(
        self,
        data_file: str = None,  #"./mingpt-training-all.pthru",
        val_file: str = None,
        dataset_dir: str = '/ssd2tb/ftwc/playthru_data/',
        splits_list: Optional[Iterable[str]] = None,   # ['train', 'valid', 'test']
        num_workers: int = 16,
        tokenizer_file: str = "ftwc_tokenizer_new.json",
        seed: int = 42,
        batch_size: int = 192,
        block_size: int = 128,
        train_filtering = None,
        eval_filtering = PlaythroughDataset.TARGET_CMD_TOKENS,
        ignore_kg = False,
        max_pthru_steps = MAX_PTHRU_STEPS,
        filter_out_skills = None,
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

        self.dataset_dir = dataset_dir
        self.splits_list = splits_list
        self.data_file = data_file
        self.val_file = val_file
        self.tokenizer_file = tokenizer_file
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
        self.pad_tok = None
        self.train_filtering = train_filtering
        self.eval_filtering = eval_filtering
        self.ignore_kg = ignore_kg
        self.max_pthru_steps = max_pthru_steps
        self.filter_out_skills = filter_out_skills

    def read_and_encode(self, filepath):
        with open(filepath, 'r') as file:
            text = file.read()
        encoded_data = self.tokenizer.encode(text)
        #encoded_data.ids
        #encoded_data.tokens
        return encoded_data

    def load_from_textds(self, dirpath, splits_list=None, no_kg=False):
        def _normalize_splitname(splitname):
            name_parts = splitname.split('-')
            if 'train' in name_parts:
                return 'train'
            elif 'valid' in name_parts:
                return 'valid'
            elif 'test' in name_parts:
                return 'test'
            return splitname

        _tokenizer = self.tokenizer
        _text_field = 'text0' if no_kg else 'text'
        def _tokenize_text(data: dict):
            return _tokenizer(data[_text_field])

        if not splits_list:
            splits_list = ['train', 'valid', 'test']
        dsfiles = {_normalize_splitname(split): f"{dirpath}/{split}.textds" for split in splits_list}
        print(f"load_from_textds({_text_field}, {dsfiles})")

        for filepath in dsfiles.values():
            if not _check_skills_list(filepath):
                _fix_skills_list(filepath)

        _dataset = load_dataset('json', data_files=dsfiles)        # ,download_mode='force_redownload')
        if self.max_pthru_steps and self.max_pthru_steps > 0:
            for splitname in _dataset:  # don't include records that have trajectories longer than than max_pthru_steps
                _dataset[splitname] = _dataset[splitname].filter(lambda rec: rec['numsteps'] <= self.max_pthru_steps)
        if self.filter_out_skills:
            exclude_skills = set(self.filter_out_skills)  # don't include records that list one or more of these skills
            for splitname in _dataset:
                _dataset[splitname] = _dataset[splitname].filter(lambda rec: not bool(set(rec['skills']) & exclude_skills))

        tokenized_ds = _dataset.map(_tokenize_text, batched=True, load_from_cache_file=False)
        tokenized_ds.set_format(type='numpy', columns=['input_ids'])
        print(tokenized_ds)
        return tokenized_ds

    def prepare_data(self):
        """
        """
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_file)
        # self.tokenizer = Tokenizer.from_file(self.tokenizer_file)
        # self.vocab_dict = self.tokenizer.get_vocab(with_added_tokens=True)
        self.cmd_start_marker = self.tokenizer.convert_tokens_to_ids(CMD_START_TOKEN)   # .token_to_id(CMD_START_TOKEN)
        self.cmd_end_marker = self.tokenizer.convert_tokens_to_ids(CMD_END_TOKEN)
        self.game_start_tok = self.tokenizer.convert_tokens_to_ids(GAME_START_CMD)
        self.pad_tok = self.tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        cmd_markers = (self.cmd_start_marker, self.cmd_end_marker)

        if not self.vocab_size:
            self.vocab_size = self.tokenizer.vocab_size   # len(self.vocab_dict)
            # self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)  # this seems to be wrong!
            # (maybe tokenizers=0.10.0rc1 impl of WordLevel model has a bug?
            # assert self.vocab_size == len(self.vocab_dict)
        # TODO: get dataset length after loading data and use it to compute final_tokens
        if self.dataset_dir:  # load from .json array of dicts (a common format for huggingface datasets)
            self.tokenized_ds = self.load_from_textds(self.dataset_dir, splits_list=self.splits_list, no_kg=self.ignore_kg)
            eval_dataset = None
            for splitkey in self.tokenized_ds:
                encoded_data_ids = np.concatenate(self.tokenized_ds[splitkey]['input_ids'][:])
                # print(encoded_data_ids[:100])
                print(f"PlaythroughDataModule.prepare_data({splitkey}): ", len(encoded_data_ids))
                if 'train' in splitkey:
                    self.train_dataset = PlaythroughDataset(encoded_data_ids, self.block_size,
                                                            vocab_size=self.vocab_size,
                                                            cmd_markers=cmd_markers,
                                                            game_start_tok=self.game_start_tok,
                                                            pad_tok=self.pad_tok,
                                                            span_filtering=self.train_filtering,  #PlaythroughDataset.TARGET_CMD_TOKENS)
                                                            batch_size=self.batch_size,
                                                            prompt_extra_len=-10)  # include extra len after cmd (random range(0,-10)
                else:
            # print(eval_encoded_ids[:100])
                    eval_dataset = PlaythroughDataset(encoded_data_ids, self.block_size,
                                                         vocab_size=self.vocab_size,
                                                         cmd_markers=cmd_markers,
                                                         game_start_tok=self.game_start_tok,
                                                         pad_tok=self.pad_tok,
                                                         span_filtering=PlaythroughDataset.TARGET_CMD_PROMPTS,
                                                         batch_size=self.batch_size,
                                                         prompt_extra_len=0)  # DO NOT include the cmd after the prompt
                    if 'valid' in splitkey:
                        print(f"NOTE: using {splitkey} dataset as .validaton_dataset")
                        self.validation_dataset = eval_dataset
                    else:
                        print(f"NOTE: loaded {splitkey} -> eval_dataset. (might discard or use as .validation_dataset...")
            if eval_dataset and not self.validation_dataset:  # (if only a test split has been loaded)
                print("NOTE: using previously loaded eval_dataset as .validation_dataset")
                self.validation_dataset = eval_dataset        # maybe use the test split as self.validation_dataset

        else:   # load directly from .pthru text files
            encoded_data = self.read_and_encode(self.data_file)
            print("PlaythroughDataModule.prepare_data: ", len(encoded_data.ids))
            eval_encoded = self.read_and_encode(self.val_file) if self.val_file else None

            self.train_dataset = PlaythroughDataset(encoded_data.ids, self.block_size,
                                                    vocab_size=self.vocab_size,
                                                    cmd_markers=cmd_markers,
                                                    game_start_tok=self.game_start_tok,
                                                    pad_tok=self.pad_tok,
                                                    span_filtering=self.train_filtering,  #PlaythroughDataset.TARGET_CMD_TOKENS)
                                                    batch_size=self.batch_size,
                                                    prompt_extra_len=-10)  # include extra len after cmd (random range(0,-10)


            if eval_encoded:
                batch_size = self.batch_size
                # if self.eval_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
                #     batch_size = 1
                self.validation_dataset = PlaythroughDataset(eval_encoded.ids, self.block_size,
                                                             vocab_size=self.vocab_size,
                                                             cmd_markers=cmd_markers,
                                                             game_start_tok=self.game_start_tok,
                                                             pad_tok=self.pad_tok,
                                                             span_filtering=PlaythroughDataset.TARGET_CMD_PROMPTS,
                                                             batch_size=batch_size,
                                                             prompt_extra_len=0)  # DO NOT include the cmd after the prompt

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,  # NOTE: if need to transpose batches to (t, b, ..),  drop_last=True can help
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=lambda batch: self.train_dataset.pad_collate(batch)  #, align_cmds=False)
        )
        return loader

    def val_dataloader(self):
        if not self.validation_dataset:
            return None

        # if self.validation_dataset.span_filtering == PlaythroughDataset.TARGET_CMD_PROMPTS:
        #     batch_size = 1
        # else:
        #     batch_size = self.batch_size

        loader = DataLoader(
            self.validation_dataset,
            batch_size=self.validation_dataset.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=lambda batch: self.validation_dataset.pad_collate_for_eval(batch)  #, align_cmds=False)
        )
        return loader
