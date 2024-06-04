#!/usr/bin/env bash

scenes=("scan24" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105" "scan106" "scan110" "scan114" "scan118" "scan122")
ref_views=(23 43)

for ref_view in ${ref_views[@]};
do
    for scene in ${scenes[@]};
    do
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --mode finetune \
        --scene $scene \
        --ref_view $ref_view \
        --conf confs/gens_finetune.conf ${@:1}
    done
done