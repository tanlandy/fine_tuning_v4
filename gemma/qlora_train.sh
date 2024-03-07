#! /usr/bin/env bash

set -ex

LR=2e-4
LORA_RANK=8

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=gemma_qlora
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

DATA_FS="/opt/models"
DATA_FS_NAME="/opt/models/datasets"
LOG_STEPS=30

LOCAL_RANK=-1 CUDA_VISIBLE_DEVICES=0 python qlora_tutorial.py \
    --use_flash_attention_2 true \
    --lora_r $LORA_RANK \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --dataset_name $DATA_FS_NAME \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --optim "paged_adamw_8bit" \
    --save_steps ${LOG_STEPS} \
    --logging_steps ${LOG_STEPS} \
    --learning_rate $LR \
    --max_grad_norm 1.0 \
    --max_steps 1000 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing true \
    --fp16 \
    --packing \
    --max_seq_length 2048 \
    --seed 23 2>&1 | tee ${OUTPUT_DIR}/train.log
