import os
import glob
import pathlib
# import argparse
from typing import List, Dict, Tuple, Any, Optional

import datetime
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from twutils.playthroughs import TW_VALIDATION_DIR
from twutils.redis_playthroughs import format_playthrough_step, start_game_for_playthrough, \
    playthrough_step_to_json, step_game_for_playthrough

from mingpt.pthru_dataset import PlaythroughDataModule
from mingpt.model import GPTModule


def predict_cmd(pl_model, tokenizer, pthru_so_far: str) -> str:
    N_AHEAD = 9
    encoded_ptru = tokenizer.encode(pthru_so_far)
    pthru_token_ids = encoded_ptru.ids[:]
    if len(pthru_token_ids) >= pl_model.model.block_size:
        pthru_token_ids = pthru_token_ids[-pl_model.model.block_size + 1:]

    pthru_token_ids.append(tokenizer.token_to_id('>>>['))  # start of command marker

    x = torch.tensor(np.array(pthru_token_ids))
    x.to(pl_model.device)

    predicted = pl_model.sample_ahead(x, n_samples=N_AHEAD, temperature=1.0, randsampling=False, top_k=None)

    y_predicted = predicted.cpu().tolist()

    print("****** PROMPT:", tokenizer.decode(y_predicted[max(0, len(y_predicted)-20-N_AHEAD):-N_AHEAD]))
    predicted_cmd = y_predicted[-N_AHEAD:]

    end_marker = tokenizer.token_to_id("]<<<")
    try:
        idx_ = predicted_cmd.index(end_marker)
        predicted_cmd = predicted_cmd[:idx_]
    except ValueError:
        print("end of command marker NOT FOUND")

    action_str = tokenizer.decode(predicted_cmd)
    if ' - ' in action_str:
        action_str = action_str.replace(' - ', '-')
    print("******* PREDICTED CMD:", action_str)
    return action_str  # send the command to the game


def format_step_json(agent_kg, step_json):
    step_json = list(step_json.values())[0]
    prev_action = step_json.get('prev_action', None)
    if prev_action and " the " in prev_action:
        prev_action = prev_action.replace(" the ", " ")
        step_json['prev_action'] = prev_action

    #         kg_descr = get_kg_descr(kg_accum, step_json)
    kg_descr = agent_kg.describe_room(agent_kg.player_location.name, obs_descr=step_json['description'])
    # simplify_raw_obs_feedback=False because we start_game_for_playthrough(raw_obs_feedback=False)
    # and thus, the ConsistentFeedbackWrapper already does exactly the same thing at each game step
    outstr, pthru = format_playthrough_step(kg_descr, step_json, simplify_raw_obs_feedback=False)
    return outstr, pthru


def play_game(gamename, pl_model, tokenizer, gamedir=TW_VALIDATION_DIR, max_steps=45):
    _gamefile = f"{gamedir}/{gamename}.z8"
    _dones = [0]
    _rewards = [0]
    num_steps = 0
    pthru_all = ""
    next_cmds = ['start']

    gymenv, _obs, _infos = start_game_for_playthrough(_gamefile,
                                                      raw_obs_feedback=False,  # simplify obs and feedback text
                                                      passive_oracle_mode=True)

    agent_kg = gymenv.tw_oracles[0].gi.kg

    step_json = playthrough_step_to_json(next_cmds, _dones, _infos, _obs, _rewards, num_steps)
        # save_playthrough_step_info_to_redis(gamename, num_steps, _obs, _rewards, _dones, _infos,
        #                                                          next_cmds,
        #                                                          redis=None, do_write=False)

    _, pthru = format_step_json(agent_kg, step_json)
    pthru_all += pthru

    if 'tw_o_step' in _infos:
        next_cmds = _infos['tw_o_step']
    else:
        next_cmds = [None] * len(_obs)
    predicted_cmd = predict_cmd(pl_model, tokenizer, pthru_all)
    print(f"Oracle: |{next_cmds[0]}|  Model: |{predicted_cmd}|")
    next_cmds[0] = predicted_cmd
    success = False
    while not _dones[0] and num_steps < max_steps:
        num_steps += 1
        _obs, _rewards, _dones, _infos = step_game_for_playthrough(gymenv, next_cmds)
        step_json = playthrough_step_to_json(next_cmds, _dones, _infos, _obs, _rewards, num_steps)
        # _redis_ops_, step_json = save_playthrough_step_info_to_redis(gamename, num_steps, _obs, _rewards, _dones,
        #                                                              _infos, next_cmds,
        #                                                              redis=None, do_write=False)

        if _dones[0] and _rewards[0] and next_cmds[0] == 'eat meal':
            success = True

        if 'tw_o_step' in _infos:
            next_cmds = _infos['tw_o_step']
        else:
            next_cmds = [None] * len(_obs)
        print(step_json.keys())
        assert len(step_json) == 1, f"Expecting one key like 'step_NN' {list(step_json.keys())}"
        _, pthru = format_step_json(agent_kg, step_json)
        pthru_all += pthru

        predicted_cmd = predict_cmd(pl_model, tokenizer, pthru_all)
        print(f"Oracle: |{next_cmds[0]}|  Model: |{predicted_cmd}|")
        if not _dones[0]:
            next_cmds[0] = predicted_cmd
        print("============================================")
        print(pthru)
        print("============================================")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(pthru_all)

    gymenv.close()
    return num_steps, success

@hydra.main(config_path=".", config_name="pthru-gpt")
def main(cfg: DictConfig) -> None:
    cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("cwd_path = ", cfg.cwd_path)

    seed_everything(cfg.general.random_seed)

    start_time = datetime.datetime.now()
    print(f"======================================= eval_gpt.py - Start time: {start_time}\n{os.getcwd()}\n")
    pass

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

    _datamodule.prepare_data()
    tokenizer = _datamodule.tokenizer
    train_dataset = _datamodule.train_dataset
    cfg.trainer.final_tokens = 2 * len(train_dataset) * train_dataset.block_size
    cfg.gpt.vocab_size = _datamodule.vocab_size

    print("USING PyTorch Lightning")

    pl_model = GPTModule.load_from_checkpoint(checkpoint_path=cfg.eval.checkpoint)
    # pl_model.to(torch.device('cuda'))

    print(f"Training dataset length={len(_datamodule.train_dataset)} (raw:{len(_datamodule.train_dataset.data)})")
    print(f"Validation dataset length={len(_datamodule.validation_dataset)} (raw:{len(_datamodule.validation_dataset.data)})")
    dataset = _datamodule.validation_dataset
    if cfg.eval.play_games:
        filelist = glob.glob(f"{cfg.eval.pthru_data_dir}/*.pthru")
        print(len(filelist))
        maybe_ok = 0
        num_successful = 0
        total_played = 0
        n_steps_dict = {}
        for i, filepath in enumerate(filelist[:]):
            total_played += 1
            print(f"[{i}] ------------ PLAYING: {filepath}")
            gn = pathlib.Path(filepath).stem
            n_steps, success = play_game(gn, pl_model, tokenizer, gamedir=f"{cfg.eval.games_dir}")
            print(f"[{i}] n_steps={n_steps} \t---- {gn} ")
            if n_steps < 45:
                maybe_ok += 1
            if success:
                num_successful += 1
            n_steps_dict[gn] = n_steps
        print(f"PLAYED: {total_played} success={num_successful} maybe_ok={maybe_ok}")
        print(n_steps_dict)
    else:
        debug_print_some_spans(dataset)

        total_matched, total_cmd_tokens = eval_predict_cmd_tokens(pl_model, dataset, tokenizer=_datamodule.tokenizer)
        print(f"MATCHED {total_matched}/{total_cmd_tokens} acc={total_matched / total_cmd_tokens}")

    finish_time = datetime.datetime.now()
    print(f"================ eval_gpt.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")


def eval_predict_cmd_tokens(pl_model, dataset, tokenizer=None):
    total_cmd_tokens = 0
    total_matched = 0
    n_printed = 0
    # for idx in range(1, len(dataset.cmd_spans)):   # skip the initial 'start' command
    #     x, y, cmd_start_pos = dataset.get_cmd_prompt(idx, continuation=-1)
    #     if idx % 200 == 0 and total_matched == total_cmd_tokens:
    #         print(idx, "...")  # let them know we're actually doing something...
    for igame in range(dataset.num_games):
        if igame % 10 == 0:
            print(f"+{igame} [:{dataset.get_num_steps(igame)}] --------------------------")
        for istep in range(1, dataset.get_num_steps(igame)):
            #_span_debug, _, _ = dataset.get_token_idxs(igame, 0, istep)
            #print(f"get_token_idxs(igame={igame}, 0, end_step={istep})  {_span_debug}")
            #print(dataset.data[_span_debug[0]:_span_debug[1]+1])
            x, y, cmd_start_pos = dataset.get_cmd_prompt_for_gamestep(igame, istep, continuation=-1)
            cmd_start_pos = cmd_start_pos.to(pl_model.device)
            x = x.to(pl_model.device)
            y = y.to(pl_model.device)
            cmd_len = len(x) - int(cmd_start_pos) - 1
            x_trunc = x[0:int(cmd_start_pos)+1]
            y_trunc = y[0:int(cmd_start_pos)+cmd_len]
            #print(f"len(x)={len(x)}, cmd_start_pos={cmd_start_pos}" )
            #print("cmd_len", cmd_len)
            #print("x:", x)
            #print("x_trunc:", x_trunc)

            assert x_trunc[int(cmd_start_pos)] == dataset.cmd_start, f"{cmd_start_pos}: {x_trunc[int(cmd_start_pos)]} {x_trunc}"
            predicted = pl_model.sample_ahead(x_trunc,
                                     n_samples=cmd_len, temperature=1.0, randsampling=False, top_k=None)

            assert len(predicted) == len(y_trunc)+1, f"{len(predicted)} {len(y_trunc)}"
            assert predicted[1] == y_trunc[0], f"{predicted[0:5]} {y_trunc[0:5]}"

            n_matched_torch = int(torch.sum(predicted[-cmd_len:] == y_trunc[-cmd_len:]))  # torch 1.7 has torch.count_nonzero()
            n_cmd_tokens = int(cmd_len)
            # y_predicted = predicted.cpu().tolist()
            # y_ids = y_trunc.detach().cpu().tolist()
            # assert len(y_predicted) == len(y_ids) + 1, f"{len(y_ids)} {len(y_predicted)}"
            # assert y_predicted[1] == y_ids[0], f"{y_predicted[0:5]} {y_ids[0:5]}"

            # n_cmd_tokens = 0
            # n_matched = 0
            # if cmd_len > 1:
            #     for i in range(1, cmd_len + 1):
            #         n_cmd_tokens += 1
            #         if y_predicted[-i] == y_ids[-i]:
            #             n_matched += 1
            # assert n_matched == n_matched_torch, f"{n_matched} {n_matched_torch}"
            # assert n_cmd_tokens == cmd_len

            if n_matched_torch != n_cmd_tokens:
                n_printed += 1
                n_matched = n_matched_torch
                if n_printed < 10 or n_printed % 100 == 0 or igame > dataset.num_games - 3:
                    print(f" {igame}.{istep}  ...   \t{n_matched} / {n_cmd_tokens}   \tacc: {n_matched / n_cmd_tokens:4f}")
                    if tokenizer:
                        y_predicted = predicted.cpu().tolist()
                        y_ids = y_trunc.detach().cpu().tolist()
                        show_sample(tokenizer, f"{igame}.{istep}", y_predicted, y_ids, n_sampled=n_cmd_tokens)

            total_cmd_tokens += n_cmd_tokens
            total_matched += n_matched_torch
    return total_matched, total_cmd_tokens


def debug_print_some_spans(dataset):
    print("eval dataset # cmd_spans =", len(dataset.cmd_spans))
    for i in range(5):
        num_steps = dataset.get_num_steps(i)
        game_span = dataset.game_spans[i]
        print("Game", i, "num_steps:", num_steps, game_span)
        for j in range(num_steps):
            print(f"\tcmd_span[{game_span[0] + j}] {dataset.cmd_spans[game_span[0] + j]}", end=' ')
            print(f"{dataset.get_token_idxs(i, 0, j + 1, inclusive=(True, True))[0]}")
            # print("cmd_prompt_for_gamestep:", dataset.get_cmd_prompt_for_gamestep(i,j))
        print("Game", i, "token span:", dataset.get_token_idxs(i))
        print()
    # print the same info for the last game in the dataset
    i = dataset.num_games - 1
    num_steps = dataset.get_num_steps(i)
    game_span = dataset.game_spans[i]
    print("Game", i, "num_steps:", num_steps, game_span)
    for j in range(num_steps):
        print(f"\tcmd_span[{game_span[0] + j}] {dataset.cmd_spans[game_span[0] + j]}", end=' ')
        print(f"{dataset.get_token_idxs(i, j, j + 1, inclusive=(True, True))[0]}")
    last_range, cmd0_len, cmd1_len = dataset.get_token_idxs(i)
    print("Game", i, "token span:", last_range, cmd0_len, cmd1_len)
    # token_list = dataset.data[last_range[0]:last_range[1]]
    # print(_datamodule.tokenizer.decode(token_list))
    # print()


def show_sample(tokenizer, idx, y_predicted, y_ids, n_sampled=5):
    # print(f"!{idx}!", tokenizer.decode(y_ids))
    print(f"({idx})", tokenizer.decode(y_ids[0:6]), '[....]',
          tokenizer.decode(y_ids[-5-n_sampled:-n_sampled]), "|",
          tokenizer.decode(y_ids[-n_sampled:]))

    print(f"<{idx}>", tokenizer.decode(y_predicted[1:7]), '[....]',
          tokenizer.decode(y_predicted[-5-n_sampled:]))
    print()


if __name__ == '__main__':
    main()
