#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

dataset=corsair/algo_seq_3var/

data_path=./data/$dataset

# Generate data if needed
python ./data/gen_data_algo_seq.py --variables 3 --path $data_path

# Number of GPUs available on your machine
ngpus=2

export PYTHONPATH=$HOME/work/0_magd3/tw13/minGPT

# Baseline
#python -m torch.distributed.launch --nproc_per_node=$ngpus main.py \
#    --nepochs 50 --nbatches 1000 --batch-sz 512 --test-batch-sz 32 \
#    --data $data_path --data-omit-labels "_" \
#    --nlayers 8 --nheads 4 --hid-sz 256 --inner-hid-sz 1024 \
#    --mem-sz 64 --attn-lim 100 \
#    --lr 0.0001 --momentum 0 --dropout 0.2 --optim adam --lr-warmup 1000 \
#    --grad-clip 0.1 --pre-norm \
#    --checkpoint checkpoints/$dataset/xt_3var.ckpt --checkpoint-freq 5

# Feedback
python -m torch.distributed.launch --nproc_per_node=$ngpus main.py \
    --nepochs 50 --nbatches 1000 --batch-sz 512 --test-batch-sz 32 \
    --data $data_path --data-omit-labels "_" \
    --nlayers 4  --nheads 4 --hid-sz 256 --inner-hid-sz 1024 \
    --mem-sz 64 --attn-lim 100 \
    --lr 0.0001 --momentum 0 --dropout 0.2 --optim adam --lr-warmup 1000 \
    --grad-clip 0.1 --pre-norm \
    --feedback \
    --checkpoint checkpoints/$dataset/fb0_3var.ckpt --checkpoint-freq 5
