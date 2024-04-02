#!/bin/bash

CUDA_VISIBLE_DEVICES=$2
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llama-vid-7b-full-224-video-fps-1"
# CKPT="llama-vid-7b-full-224-long-video"
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamavid/eval/mvbench.py \
        --model-path ./work_dirs/$CKPT \
        --output_dir ./work_dirs/llamavid/$CKPT \
        --output_name $1 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 
done
