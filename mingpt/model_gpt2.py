"""
---
title: Train OpenAI GPT2
---

GPT2hf(
  (gpt): GPT2LMHeadModel(
    (transformer): GPT2Model(
      (wte): Embedding(157, 512)
      (wpe): Embedding(128, 512)
      (drop): Dropout(p=0.1, inplace=False)
      (h): ModuleList(
        (0): GPT2Block(
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): GPT2Attention(
            (c_attn): Conv1D()
            (c_proj): Conv1D()
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (resid_dropout): Dropout(p=0.1, inplace=False)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): GPT2MLP(
            (c_fc): Conv1D()
            (c_proj): Conv1D()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (1): GPT2Block(
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): GPT2Attention(
            (c_attn): Conv1D()
            (c_proj): Conv1D()
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (resid_dropout): Dropout(p=0.1, inplace=False)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): GPT2MLP(
            (c_fc): Conv1D()
            (c_proj): Conv1D()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (2): GPT2Block(
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): GPT2Attention(
            (c_attn): Conv1D()
            (c_proj): Conv1D()
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (resid_dropout): Dropout(p=0.1, inplace=False)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): GPT2MLP(
            (c_fc): Conv1D()
            (c_proj): Conv1D()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (ln_f): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (lm_head): Linear(in_features=512, out_features=157, bias=False)
  )
)

"""

import logging

import torch
from torch import nn

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#print("MODEL_TYPES", MODEL_TYPES)
#print("MODEL_CONFIG_CLASSES", MODEL_CONFIG_CLASSES)


class GPT2hf(nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()

        assert config.d_embd % config.n_heads == 0,\
            f"embedding dim ({config.d_embd}) should be a multiple of num heads ({config.n_heads})"

        hfconfig = CONFIG_MAPPING['gpt2']()
        # print(hfconfig)
        # GPT2Config
        # {
        #     "activation_function": "gelu_new",
        #     "attn_pdrop": 0.1,
        #     "bos_token_id": 50256,
        #     "embd_pdrop": 0.1,
        #     "eos_token_id": 50256,
        #     "initializer_range": 0.02,
        #     "layer_norm_epsilon": 1e-05,
        #     "model_type": "gpt2",
        #     "n_embd": 768,
        #     "n_head": 12,
        #     "n_inner": null,
        #     "n_layer": 12,
        #     "n_positions": 1024,
        #     "reorder_and_upcast_attn": false,
        #     "resid_pdrop": 0.1,
        #     "scale_attn_by_inverse_layer_idx": false,
        #     "scale_attn_weights": true,
        #     "summary_activation": null,
        #     "summary_first_dropout": 0.1,
        #     "summary_proj_to_labels": true,
        #     "summary_type": "cls_index",
        #     "summary_use_proj": true,
        #     "transformers_version": "4.14.1",
        #     "use_cache": true,
        #     "vocab_size": 50257
        # }


        hfconfig.n_embd = config.d_embd
        hfconfig.n_head = config.n_heads
        hfconfig.n_layer = config.n_layers
        hfconfig.attn_pdrop = config.attn_pdrop
        hfconfig.embd_pdrop = config.embd_pdrop
        hfconfig.resid_pdrop = config.resid_pdrop
        hfconfig.n_positions = config.block_size
        hfconfig.vocab_size = config.vocab_size
        hfconfig.n_inner = config.hidden_layer_multiplier
        hfconfig.activation_function = 'gelu_new'

        print(hfconfig)

        self.gpt = AutoModelForCausalLM.from_config(hfconfig)

        if tokenizer:
            print(tokenizer)
            print(tokenizer.vocab_size)
            self.gpt.resize_token_embeddings(tokenizer.vocab_size)
        if self.gpt:
            logger.info("number of parameters: %e", sum(p.numel() for p in self.gpt.parameters()))
        else:
            logger.error(f"FAILED to build model: {self.gpt}")

    def forward(self, tokids):   #, targets=None):
        # b, t = tokids.size()
        # if t > self.block_size:
        #     assert False, f"Cannot forward, model block size is exhausted: {t} ! <= {self.block_size}"

        # # forward the GPT model
        ret_dict = self.gpt(tokids, return_dict=True)
        return ret_dict.logits #, loss

    def get_param_groups(self, weight_decay: float):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        self.tokens = 0  # reset count of tokens processed
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, transformers.modeling_utils.Conv1D)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        decay_params = []   # [param_dict[pn] for pn in sorted(list(decay))]
        nodecay_params = []  # [param_dict[pn] for pn in sorted(list(no_decay))]
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or \
                    pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # all biases will not be decayed
                    # weights of blacklist modules will NOT be weight decayed

                    #print(type(m), "NO_decay:", pn, "\t\t>>>   ", fpn)
                    no_decay.add(fpn)
                    if not any(id(p) == id(p0) for p0 in nodecay_params):
                        if any(id(p) == id(p0) for p0 in decay_params):
                            print("NOT ADDING TO NODECAY --- already in DECAY:", id(p), fpn)
                        else:
                            print("**** ADDING TO NODECAY:", id(p), fpn)
                            nodecay_params.append(p)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    #print(type(m), "DECAY:", pn, "\t\t<<##   ", fpn)
                    decay.add(fpn)
                    if not any(id(p) == id(p0) for p0 in decay_params):
                        if any(id(p) == id(p0) for p0 in nodecay_params):
                            print("NOT ADDING TO DECAY --- already in NODECAY:", id(p), fpn)
                        else:
                            print("**** ADDING TO DECAY:", id(p), fpn)
                            decay_params.append(p)
                else:
                    # print(type(m), "UNMATCHED:", pn, "\t\t||||||   ", fpn)
                    pass

        # special case the position embedding parameter i as not decayed
        # no_decay.add('gpt.transformer.positions_embed.weight')
        #NOTE: gpt.lm_head.weight is the same parameter (===) as gpt.transformer.tokens_embed.weight

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # param_dict = {}
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
        #         param_dict[fpn] = p
        for pn in param_dict.keys():
            print(pn)
        inter_params = decay & no_decay
        union_params = decay | no_decay
        print("DECAY:") #, sorted(list(decay)))
        for i, pn in enumerate(sorted(list(decay))):
                print(f"[{i}]\t\t {pn}")
        print("NO_DECAY:")
        for i, pn in enumerate(sorted(list(no_decay))):
                print(f"[{i}]\t\t {pn}")
        print("INTERSECTION:", sorted(list(inter_params)))
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        if len(param_dict.keys() - union_params) > 0:
            for key in sorted(list(param_dict.keys() - union_params)):
                print(f"PARAM NOT IN param_groups -- {key}: {type(param_dict[key])}")
            assert len(
                param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params),)
        print("union_params - param_dict:")  #, sorted(list(param_dict.keys())))
        for i, pn in enumerate(sorted(list(union_params))):
            if pn not in param_dict:
                print(f"[{i}]\t\t {pn}")
        # create the pytorch optimizer object
        # decay_params = []   # [param_dict[pn] for pn in sorted(list(decay))]
        # for pn in sorted(list(decay)):
        #     print("ADDING DECAY:", id(param_dict[pn]), pn)
        #     decay_params.append(param_dict[pn])
        # nodecay_params = []  # [param_dict[pn] for pn in sorted(list(no_decay))]
        # for pn in sorted(list(no_decay)):
        #     print("ADDING NO DECAY:", id(param_dict[pn]), pn )
        #     nodecay_params.append(param_dict[pn])
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return optim_groups

