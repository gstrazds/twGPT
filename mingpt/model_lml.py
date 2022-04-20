"""
---
title: Train Feedback Transformer
summary: This is training code with notes for a feedback transformer.
---

# Train Feedback Transformer

This trains a [feedback transformer](index.html) model for auto-regression.
You can pick the original feedback transformer or the new version
where the keys and values are precalculated.

Here's a Colab notebook for training a feedback transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/transformers/feedback/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d8eb9416530a11eb8fb50242ac1c0002)
"""

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)

def build_GPT(cfg):
    # when Hydra config (v1.1) supports recursive instantiation, this could be replaced by instantiate-from-config
    # https://github.com/facebookresearch/hydra/issues/566

    from labml.nn.transformers.gpt import GPT
    from labml.nn.transformers.models import Encoder, TransformerLayer, \
        EmbeddingsWithPositionalEncoding, EmbeddingsWithLearnedPositionalEncoding, Generator
    from labml.nn.transformers.mha import MultiHeadAttention
    from labml.nn.transformers.feed_forward import FeedForward

    n_heads = cfg.n_heads
    d_model = cfg.d_embd
    n_layers = cfg.n_layers
    dropout_prob = cfg.dropout
    vocab_size = cfg.vocab_size
    d_ff = cfg.hidden_layer_multiplier * d_model
    self_attn = MultiHeadAttention(n_heads, d_model, dropout_prob=dropout_prob) #, bias: bool = True)

    feedforward = FeedForward(d_model,
                              d_ff=d_ff,   #4*d_model,   #cfg.model.d_ff,
                              dropout=dropout_prob,
                              activation=nn.GELU())

    encoder_layer = TransformerLayer(
                        d_model=d_model,
                        self_attn=self_attn,
                        src_attn=None,
                        feed_forward=feedforward,
                        dropout_prob=dropout_prob)

    encoder = Encoder(encoder_layer, n_layers=n_layers)
    src_embed = EmbeddingsWithLearnedPositionalEncoding(d_model=d_model, n_vocab=vocab_size) #, max_len=5000)
    generator = Generator(n_vocab=vocab_size, d_model=d_model)
    model = GPT(encoder, src_embed, generator)

    # model.init_weights()
    return model


class GPT_lml(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_embd % config.n_heads == 0,\
            f"embedding dim ({config.d_embd}) should be a multiple of num heads ({config.n_heads})"

        self.gpt = build_GPT(config)
                        # d_model = config.model.d_embd,
                        # n_heads = config.model.n_heads,
                        # n_layers = config.model.n_layers,
                        # dropout = config.model.dropout,
                        # d_ff = config.model.d_ff,
                        # n_vocab = config.model.vocab_size,
        if self.gpt:
            logger.info("number of parameters: %e", sum(p.numel() for p in self.gpt.parameters()))
        else:
            logger.error(f"FAILED to build model: {self.gpt}")

    def forward(self, tokids):   #, targets=None):
        # b, t = tokids.size()
        # if t > self.block_size:
        #     assert False, f"Cannot forward, model block size is exhausted: {t} ! <= {self.block_size}"

        # # forward the GPT model
        # (second return value is for state, since lml trainer is used with RNNs also)
        logits, _ = self.gpt(tokids)
        return logits #, loss

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
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the EmbeddingsWithPositionalEncoding module as not decayed
        no_decay.add('gpt.src_embed.positional_encodings')

        # # special case the layer weights for feedback transformer memory attn (TODO: ?not sure if should or should not weight decay)
        # no_decay.add('layer_weight')

        #### FOR COMPARISON: corresponding code from labml.nn.transformers.gpt
        #     This applies weight decay only to weights of linear layers.
        #     """
        #     # Collect names of parameters to apply weight decay
        #     decay = set()
        #     for mn, m in c.model.named_modules():
        #         for pn, p in m.named_parameters():
        #             fpn = f'{mn}.{pn}' if mn else pn  # full param name
        #
        #             if fpn.endswith('weight') and isinstance(m, nn.Linear):
        #                 decay.add(fpn)
        #
        #     # Get all the parameters
        #     param_dict = {pn: p for pn, p in c.model.named_parameters()}
        #     # Parameters that are not decayed
        #     no_decay = set(param_dict.keys()) - decay
        #####################################################################

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        for pn in param_dict.keys():
            print(pn)
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    # def training_step(self, batch, batch_idx):
    #     if len(batch) == 3:
    #         x, y, _unused__pad_len = batch
    #     else:
    #         assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
    #         x, y = batch
    #
    #     x = x.T.contiguous()
    #     y = y.T.contiguous()
    #     outputs, _ = self.model(x)
    #     #print(f"training_step x.size={x.size()} outputs.size={outputs.size()}")
    #     loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
    #
    #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     return {"loss": loss}
    #
    # def validation_step(self, batch, batch_idx):
    #     if len(batch) == 3:
    #         x, y, _unused__pad_len = batch
    #     else:
    #         assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y,(padding) "+int(len(batch))
    #         x, y = batch
    #
    #     x = x.T.contiguous()
    #     y = y.T.contiguous()
    #     outputs, _ = self.model(x)
    #     loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
    #
    #     self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     metrics = {'val_loss': loss} #, 'val_acc': acc}
    #     return metrics
    #
    #
    # def sample_ahead(self, x, n_samples, temperature=1.0, randsampling=False, top_k=None):
    #     assert len(x.shape) == 1  # expecting a vector of len t
    #     x_in = torch.unsqueeze(x, 0)  #torch.unsqueeze(x,0) ##== x[None, ...]
    #     assert x_in.shape[0] == 1  # (b=1, t)
    #     block_size = self.hparams.model.block_size
    #     preds = sample(self.model, block_size, x_in, steps=n_samples, temperature=temperature, sample=randsampling, top_k=top_k)
    #     # self.memory = memory
    #     # print(f"sample_ahead: x_in.size={x_in.size()} preds.size={preds.size()}")
    #     return preds.detach().squeeze()  # reduce from (b=1,t) to a single dimension (a vector)
    #     # return preds.detach()[0]
    #
    # def reset_episode(self):
    #     #print("****** RESET EPISODE ****")
    #     pass

