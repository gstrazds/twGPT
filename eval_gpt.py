import os, sys
import glob
import logging
import pathlib

# import argparse
from typing import List, Dict, Tuple, Any, Optional

import datetime
import hydra
import omegaconf
from omegaconf import OmegaConf, DictConfig
import torch
import numpy as np

from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info

from twutils.playthroughs import TW_TRAINING_DIR, CMD_START_TOKEN, CMD_END_TOKEN, GAME_START_CMD
from twutils.playthroughs import start_twenv_for_playthrough, step_twenv_for_playthrough
from twutils.playthroughs import playthrough_step_to_json, format_playthrough_step, concat_pthru_step
from twutils.twlogic import get_name2idmap, subst_names

from mingpt.pthru_dataset import PlaythroughDataModule
from mingpt.pl_module import GPTLitModule


def predict_cmd(pl_module, tokenizer, pthru_so_far: str, failed_cmds: List[str] = None) -> str:
    # if True:  # TODO: TMP DEBUGGING
    #     return "ignore this debugging cmd"
    N_AHEAD = 9
    encoded_pthru_ids = tokenizer.encode(pthru_so_far)
    pthru_token_ids = encoded_pthru_ids[:]
    if len(pthru_token_ids) >= pl_module.model.block_size:
        pthru_token_ids = pthru_token_ids[-pl_module.model.block_size + 1:]

    pthru_token_ids.append(tokenizer.convert_tokens_to_ids(CMD_START_TOKEN))  # start of command marker

    x = torch.tensor(np.array(pthru_token_ids))
    x_dev = x.to(pl_module.device)

    if failed_cmds:
        attempts = 5
        temp = 1.0 + .5 * (len(failed_cmds)-1)  # increase the temperature if we fail repeatedly
        sample = True
        top_k = None
    else:
        attempts = 1
        temp = 1.0
        sample = False
        top_k = None

    end_marker = tokenizer.convert_tokens_to_ids(CMD_END_TOKEN)
    period_marker = tokenizer.convert_tokens_to_ids('.')
    while attempts > 0:
        attempts -= 1
        predicted = pl_module.sample_ahead(x_dev, n_samples=N_AHEAD, temperature=temp, randsampling=sample, top_k=top_k)
        y_predicted = predicted.cpu().tolist()
        if True:  # debug print
            x_decoded = x_dev.cpu().tolist()
            x_decoded = tokenizer.decode(x_decoded[max(0, len(x_decoded)-20-N_AHEAD):])
            print("****** PROMPT:", x_decoded)  # tokenizer.decode(y_predicted[max(0, len(y_predicted)-20-N_AHEAD):-N_AHEAD]))

        predicted_cmd = y_predicted[-N_AHEAD:]
        try:
            idx_ = predicted_cmd.index(end_marker)
            predicted_cmd = predicted_cmd[:idx_]
        except ValueError:
            print("end of command marker NOT FOUND")
            if period_marker in predicted_cmd:
                idx_ = predicted_cmd.index(period_marker)
                print("WARNING: interpreting '.' as an end of command marker")
                predicted_cmd = predicted_cmd[:idx_]

        action_str = tokenizer.decode(predicted_cmd)
        if ' - ' in action_str:
            action_str = action_str.replace(' - ', '-')
        print("******* PREDICTED CMD:", action_str)
        if not failed_cmds:
            break
        elif action_str not in failed_cmds:
            print(f"Trying alternate cmd: '{action_str}' previously tried: {failed_cmds}")
            break   # try this, it hasn't failed yet

    return action_str  # send the command to the game


def format_step_json(agent_kg, step_json, map_names2ids=None):
    # print("WIP DEBUGGGING format_step_json:", step_json)
    prev_action = step_json.get('prev_action', None)
    if prev_action and " the " in prev_action:
        prev_action = prev_action.replace(" the ", " ")
        step_json['prev_action'] = prev_action

    #         kg_descr = get_kg_descr(kg_accum, step_json)
    kg_descr = agent_kg.describe_room(agent_kg.player_location.name, obs_descr=step_json['description'])
    # simplify_raw_obs_feedback=False because we start_game_for_playthrough(raw_obs_feedback=False)
    # and thus, the ConsistentFeedbackWrapper already does exactly the same thing at each game step
    outstr, pthru = format_playthrough_step(kg_descr, step_json, simplify_raw_obs_feedback=False)
    if map_names2ids:
        pthru = subst_names(pthru, map_names2ids)
        outstr = subst_names(outstr, map_names2ids)
    return outstr, pthru


def _first_word_of(cmd_str:str):
    words = cmd_str.split()
    if words:
        return words[0]
    return cmd_str


def grow_pthru_if_cmd_ok(pthru_so_far, prev_cmd, infos, reward, pthru_step):
    feedback = infos['feedback'][0]
    feedback = feedback.lower()
    if reward > 0:
        cmd_was_ok = True
    elif feedback.startswith(f'you {_first_word_of(prev_cmd)} '):
        cmd_was_ok = True
    elif feedback.startswith("you can't"):
        cmd_was_ok = False
    elif feedback.startswith('you need to'):
        cmd_was_ok = False
    elif feedback.startswith('you already'):
        cmd_was_ok = False
    elif feedback.startswith("you haven't"):
        cmd_was_ok = False
    elif feedback.startswith('i '):
        cmd_was_ok = False
    elif feedback.startswith('what do'):
        cmd_was_ok = False
    elif feedback.startswith('which do'):
        cmd_was_ok = False
    elif feedback.startswith('can only'):
        cmd_was_ok = False
    else:
        cmd_was_ok = True

    if cmd_was_ok:
        new_pthru = concat_pthru_step(pthru_so_far, pthru_step, keep_objectives=True)
        return new_pthru, cmd_was_ok  # caller doesn't need to try anything different
    print(f"%%% CMD NOT OK: |{prev_cmd}| :", feedback)
    return pthru_so_far, False  # try a different command


def play_game(gamename, pl_module, tokenizer, gamedir=TW_TRAINING_DIR, cmds=None, max_steps=45, using_internal_names=False, step_infos=None):
    _gamefile = f"{gamedir}/{gamename}.z8"
    #_gamefile = f"{gamedir}/{gamename}.json"
    _dones = [0]
    _rewards = [0]
    num_steps = 0
    pthru_so_far = ""
    next_cmds = [GAME_START_CMD]
    attempted_cmds = []  # cmds we've already tried for a given step that DIDN'T do anything

    # twenv, _obs, _infos = start_gym_for_playthrough([_gamefile],
    #                                                   raw_obs_feedback=False,  # simplify obs and feedback text
    #                                                   passive_oracle_mode=True,
    #                                                   use_internal_names=use_internal_names)
    twenv, _obs, _infos = start_twenv_for_playthrough([_gamefile], pthru_cmds=cmds, step_infos=step_infos)

    if using_internal_names:
        map_names2ids = get_name2idmap(twenv.tw_oracle.get_game_data())
    else:
        map_names2ids = None
    agent_kg = twenv.tw_oracle.gi.kg

    # step_json = playthrough_step_to_json(next_cmds, _dones, _infos, _obs, _rewards, num_steps)
    playthru_step_data = playthrough_step_to_json(next_cmds, _dones, _infos, _obs, _rewards, num_steps)
    # playthru_step_data is a list of list of json dicts (with data for a single game step),
    #   one entry for each game in the batch
    step_json = playthru_step_data[0]
    _, pthru_step = format_step_json(agent_kg, step_json, map_names2ids=map_names2ids)
    print(f"WIP DEBUGGING -- initial step: {pthru_step}")
    pthru_so_far, cmd_was_ok = grow_pthru_if_cmd_ok(pthru_so_far,
                                                        GAME_START_CMD, _infos, 0, pthru_step)

    if 'tw_o_step' in _infos:
        next_cmds = _infos['tw_o_step']
    else:
        next_cmds = [None] * len(_obs)

    predicted_cmd = predict_cmd(pl_module, tokenizer, pthru_so_far, failed_cmds=None)
    if using_internal_names:
        predicted_cmd = subst_names(predicted_cmd, map_names2ids, ids2names=True)

    # TODO: use GT command from the eval dataloader (instead of oracle)
    print(f"Oracle: |{next_cmds[0]}|  Model: |{predicted_cmd}|")
    if True:  #if False:  # temporary, WIP debugging
        next_cmds[0] = predicted_cmd  # replace oracle command with the one generated by the model
    won = None
    lost = None
    stuck = None
    while not all(_dones) and num_steps < max_steps:
        # _obs, _rewards, _dones, _infos = step_gym_for_playthrough(gymenv, next_cmds)
        _obs, _rewards, _dones, _infos = step_twenv_for_playthrough(twenv, next_cmds)
        playthru_step_data = playthrough_step_to_json(next_cmds, _dones, _infos, _obs, _rewards, num_steps)
        #   returned list has one entry for each game in the batch
        step_json = playthru_step_data[0]
        prev_cmd = next_cmds[0]

        _, pthru_step = format_step_json(agent_kg, step_json, map_names2ids=map_names2ids)
        # TODO: get step_num from game engine or tw_oracle
        step_num = num_steps
        # step_num = list(step_json.keys())[0]
        # step_num = int(step_num.split('_')[1])

        if _dones[0]:
            if _rewards[0] and prev_cmd == 'eat meal':
                won = step_num
            elif num_steps < max_steps:
                lost = step_num
            # else:
            #     stuck = step_num

        pthru_so_far, cmd_was_ok = grow_pthru_if_cmd_ok(pthru_so_far,
                                                        prev_cmd, _infos, _rewards[0], pthru_step)
        if cmd_was_ok:
            num_steps += 1
            attempted_cmds = []
        else:
            if prev_cmd in attempted_cmds or len(attempted_cmds) > 4:  # our previous attempts to randomize failed to find any other option
                print("Stop trying to find alternatives:", prev_cmd, " has already been tried:", attempted_cmds)
                num_steps += 1
            attempted_cmds.append(prev_cmd)

        if not _dones[0]:
            predicted_cmd = predict_cmd(pl_module, tokenizer, pthru_so_far, attempted_cmds)
            if using_internal_names:
                predicted_cmd = subst_names(predicted_cmd, map_names2ids, ids2names=True)

            if 'tw_o_step' in _infos:
                next_cmds = _infos['tw_o_step']   # this is what the tw_oracle recommends
            else:
                next_cmds = [None] * len(_obs)

            # TODO: use GT command from the eval dataloader (instead of oracle)
            print(f"Oracle: |{next_cmds[0]}|  Model: |{predicted_cmd}|  previous attempts: {attempted_cmds}")
            if True:  #if False:    # temporary, WIP debugging
                next_cmds[0] = predicted_cmd       # replace oracle command with the one generated by the model

            print("============================================")
            if CMD_START_TOKEN in pthru_so_far:
                _cmd_start_idx = pthru_so_far.rindex(CMD_START_TOKEN)  # last command
                print(pthru_so_far[_cmd_start_idx:])
            else:
                print("NO CMD_START FOUND in pthru_so_far! -- ", pthru_step)
            print("============================================")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(pthru_so_far)
    if not won and not lost:
        stuck = step_num
    twenv.close()
    return num_steps, won, lost, stuck

@hydra.main(config_path="conf", config_name="pthru-gpt")
def main(cfg: DictConfig) -> None:

    sys.path.append(pathlib.Path(__file__).parent.absolute())   # need to access python modules in subdirs

    seed_everything(cfg.general.random_seed)

    rank_zero_info(f"original_cwd: {hydra.utils.get_original_cwd()}")

    logger = None
    if True:
        # fh = logging.FileHandler(',', 'a')
        # fh.setLevel(logging.DEBUG)
        # # create console handler with a higher log level
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # ch.setFormatter(formatter)
        # fh.setFormatter(formatter)
        logger = logging.getLogger('eval_log')
        # logger.addHandler(fh)
        # print("ADDED logging HANDLER", fh)
        # logger.addHandler(ch)
        # print("ADDED logging HANDLER", ch)
        # logger.info("SETUP LOGGING -- info() test")
        # logger.debug("TEST DBG LOGGING -- dbg() test")


    GPTLitModule.adjust_cfg_fields(cfg)
    # cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("cwd_path = ", cfg.cwd_path)

    start_time = datetime.datetime.now()
    rank_zero_info(f"======================================= {__file__} - Start time: {start_time}\n{os.getcwd()}\n")
    pass
    if logger:
        logger.info(f"======================================= {__file__} - Start time: {start_time}\n{os.getcwd()}\n")

    _datamodule = PlaythroughDataModule(
        dataset_dir=cfg.data.dataset_dir,
        # data_file=cfg.data.data_file,
        # val_file=cfg.data.val_file,
        splits_list=[cfg.eval.which_set],  # =None loads all splits ['train', 'valid', 'test']
        tokenizer_file=cfg.data.tokenizer_file,
        num_workers=cfg.data.num_workers,
        seed=cfg.general.random_seed,
        batch_size=cfg.trainer.batch_size,
        block_size=cfg.model.block_size,
        train_filtering=cfg.data.train_filtering,
        eval_filtering=cfg.data.eval_filtering,
        ignore_kg=cfg.data.ignore_kg,
        max_pthru_steps=cfg.data.max_pthru_steps,
        filter_out_skills=cfg.data.filter_out_skills,
        which_games=cfg.eval.which_games
    )

    _datamodule.prepare_data()
    tokenizer = _datamodule.tokenizer

    # dynamically set some config/hparam values (ensures that they get saved with results of each run)
    if cfg.eval.which_set == 'test':
        dataset = _datamodule.test_dataset
    elif cfg.eval.which_set == 'train':
        dataset = _datamodule.train_dataset
    else:
        assert cfg.eval.which_set == 'valid', cfg.eval.which_set
        dataset = _datamodule.validation_dataset

    GPTLitModule.adjust_cfg_vocab(cfg, dataset)
    # cfg.model.vocab_size = _datamodule.vocab_size
    # cfg.trainer.decay_tokens = 2 * len(dataset) * dataset.block_size

    pl_model = GPTLitModule.load_from_checkpoint(checkpoint_path=cfg.eval.checkpoint)
    pl_model.set_cmd_markers(_datamodule.cmd_start_marker, _datamodule.cmd_end_marker)

    if torch.cuda.is_available() and cfg.gpus is not None:
        # print(cfg.gpus, type(cfg.gpus))
        if isinstance(cfg.gpus, omegaconf.listconfig.ListConfig):
            print("USING CUDA device=",cfg.gpus[0])
            pl_model.eval().cuda(device=cfg.gpus[0])
        else:
            pl_model.eval().cuda(device=cfg.gpus)

    # print(f"Training dataset length={len(_datamodule.train_dataset)} (raw:{len(_datamodule.train_dataset.data)})")
    # print(f"Validation dataset length={len(_datamodule.validation_dataset)} (raw:{len(_datamodule.validation_dataset.data)})")
    #_datamodule.train_dataset.print_info("train_dataset")
    dataset.print_info("eval dataset")
    if cfg.eval.which_set == 'test':
        dataloader = _datamodule.test_dataloader()
    elif cfg.eval.which_set == 'valid':
        dataloader = _datamodule.val_dataloader()
    else:
        # assert False, f"eval_gpt with cfg.eval.which_set={cfg.eval.which_set} is not supported"
        if _datamodule.validation_dataset is None and dataset is not None:  # a hack to allow eval with train_dataset (see above)
            print("WARNING: explicitly setting _datamodule.validation_dataset=", dataset)
            _datamodule.validation_dataset = dataset  # We hack this so that _datamodule._val_dataloader() will work
            dataloader = _datamodule.val_dataloader()

    if dataset.span_filtering == 'cmd_prompts' and pl_model.hparams.data.eval_filtering != 'cmd_prompts':
        print("***** ADJUSTING cfg.data.eval_filtering for backward compatibility *****")
        pl_model.hparams.data.eval_filtering = 'cmd_prompts'
        pl_model.hparams.data.eval_filtering = 'cmd_prompts'
    if cfg.eval.play_games:
        # for each .pthru in the dataset, play the corresponding game from cfg.eval.games_dir
        # NOTE: the .pthru data is not used, except to select a game name with matching pathlib.Path(filepath).stem
        eval_ds = _datamodule.tokenized_ds[cfg.eval.which_set]
        eval_gameids = set(eval_ds['game'])
        print(f"#---- play_games({cfg.eval.which_set}) NUMBER OF GAMES in eval set: {len(eval_gameids)}")
        games_glob = f"{cfg.eval.games_dir}/*.json"
        filelist = glob.glob(games_glob)
        print(f"#---- play_games({cfg.eval.which_set}) NUMBER OF FILES matching {games_glob}: {len(filelist)} ")
        file_stems = set(map(lambda fp: pathlib.Path(fp).stem, filelist))
        gameids_for_eval = eval_gameids.intersection(file_stems)
        print(f"#---- play_games({cfg.eval.which_set}) will eval {len(gameids_for_eval)} GAMES")
        wins = []
        losses = []
        loopers = []
        n_wins = 0
        n_losses = 0
        n_stuck = 0

        maybe_ok = 0
        num_successful = 0
        total_played = 0
        n_steps_dict = {}
        #for i, filepath in enumerate(filelist[:]):
        for idx in range(len(eval_ds)):
            gn = eval_ds['game'][idx]
            print(f"[{total_played}]({idx}) ------------ PLAYING: {gn}")
            total_played += 1
            cmds_list = _datamodule.list_cmds(idx, split=cfg.eval.which_set)
            step_times = _datamodule.get_step_times(idx, split=cfg.eval.which_set)
            print(f"#---- [{idx}] play_game({gn}) cmds_list={cmds_list} step_times={step_times}")
            num_steps, won, lost, stuck = play_game(gn, pl_model, tokenizer,
                                                    using_internal_names=cfg.data.use_internal_names,
                                                    gamedir=f"{cfg.eval.games_dir}",
                                                    cmds=cmds_list,
                                                    step_infos=step_times)
            if won:
                n_steps = won
                wins.append((n_steps,gn))
                n_wins += 1
            elif lost:
                n_steps = lost
                losses.append((n_steps,gn))
                n_losses += 1
            else:
                n_steps = stuck
                loopers.append((n_steps,gn))
                n_stuck += 1
            print(f"[{idx}] n_steps={n_steps} \t---- {gn} ")
            if num_steps < 45:  # n_steps is not None and n_steps < 45:
                maybe_ok += 1
            if won:
                num_successful += 1
            n_steps_dict[gn] = (num_steps, n_steps, won is not None, lost is not None, stuck is not None)
        results_str = f"PLAYED:{total_played} won:{n_wins} lost:{n_losses} stuck:{n_stuck}"
        if logger:
            logger.info(results_str)
        else:
            print(results_str)
        rank_zero_info(results_str)
        dict_out = str(n_steps_dict)
        print(dict_out)
        with open(cfg.eval.checkpoint+".play_games.results", "a+") as f:
            f.write(dict_out+'\n\n')
            finish_time = datetime.datetime.now()
            f.write(f"================ {__file__} - Finished : {finish_time} -- elapsed: {finish_time-start_time}\n")
    else:
        # debug_print_some_spans(dataset)

        results_dir = cfg.eval.results_dir
        trainer_epoch = 0
        trainer_global_step = 0
        eval_start_time = datetime.datetime.now()

        use_old_version = True
        if use_old_version:
            tokens_matched, total_cmd_tokens, full_matches, num_cmds = pl_model.eval_predict_cmd_tokens(dataset,
                tokenizer=_datamodule.tokenizer, show_samples=cfg.eval.show_samples)
        else:
            tokens_matched, total_cmd_tokens, full_matches, num_cmds = pl_model.eval_predict_cmds_batched(dataset, dataloader,
                tokenizer=_datamodule.tokenizer, show_samples=cfg.eval.show_samples)

        eval_done_time = datetime.datetime.now()
        rank_zero_info(f"----------- eval : {eval_done_time} -- elapsed: {eval_done_time - eval_start_time}")
        cmd_acc = 0.0 if num_cmds == 0 else full_matches / num_cmds
        token_acc = 0.0 if total_cmd_tokens == 0 else tokens_matched / total_cmd_tokens
        results_str = f"TOKENS: {tokens_matched}/{total_cmd_tokens} tok_acc={token_acc*100:02.2f} % \t" +\
            f"CMDS: {full_matches}/{num_cmds} cmd_acc={cmd_acc*100:02.2f} %"
        if logger:
            logger.info(results_str)
        else:
            print(results_str)
        rank_zero_info(results_str)

        if results_dir:  # (not hasattr(trainer, "rank") or trainer.rank == 0):
            results_file = f'{results_dir}/epoch{trainer_epoch:02d}_step{trainer_global_step:04d}_{token_acc:.4f}_{cmd_acc:.4f}.txt'
            with open(results_file,'w') as outfile:
                outfile.write(
                    f"{token_acc}\t{tokens_matched}\t{total_cmd_tokens}\t{cmd_acc}\t{full_matches}\t{num_cmds}\t{trainer_epoch}\t{trainer_global_step}")

        finish_time = datetime.datetime.now()
    rank_zero_info(f"================ {__file__} - Finished : {finish_time} -- elapsed: {finish_time-start_time}")
    if logger:
        logger.info(f" {__file__} - Finished : {finish_time} -- elapsed: {finish_time-start_time}")
    else:
        print(f"================ {__file__} - Finished : {finish_time} -- elapsed: {finish_time-start_time}")

def debug_print_some_spans(dataset):
    print("eval dataset # cmd_spans =", len(dataset.cmd_spans))
    for i in range(5):
        num_steps = dataset.get_num_steps(i)
        game_span = dataset.game_spans[i]
        print("Game", i, "num_steps:", num_steps, game_span)
        for j in range(num_steps):
            print(f"\tcmd_span[{game_span[0] + j}] {dataset.cmd_spans[game_span[0] + j]}", end=' ')
            print(f"{dataset.get_token_spans(i, 0, j + 1, inclusive=(True, True))[0]}")
            # print("cmd_prompt_for_gamestep:", dataset.get_cmd_prompt_for_gamestep(i,j))
        print("Game", i, "token span:", dataset.get_token_spans(i))
        print()
    # print the same info for the last game in the dataset
    i = dataset.num_games - 1
    num_steps = dataset.get_num_steps(i)
    game_span = dataset.game_spans[i]
    print("Game", i, "num_steps:", num_steps, game_span)
    for j in range(num_steps):
        print(f"\tcmd_span[{game_span[0] + j}] {dataset.cmd_spans[game_span[0] + j]}", end=' ')
        print(f"{dataset.get_token_spans(i, j, j + 1, inclusive=(True, True))[0]}")
    game_span, cmd0_span, cmd1_span = dataset.get_token_spans(i)
    print("Game", i, "token span:", game_span, cmd0_span, cmd1_span)
    # token_list = dataset.data[last_range[0]:last_range[1]]
    # print(_datamodule.tokenizer.decode(token_list))
    # print()



if __name__ == '__main__':
    main()
