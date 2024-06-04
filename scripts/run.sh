#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=5004 main.py \
    --conf confs/gens.conf ${@:1}