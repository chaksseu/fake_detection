#!/bin/bash

# ------------------------------------------------------------------------------------
# Contrastive Learning Training Script (Hardcoded Arguments with W&B API Key)
# ------------------------------------------------------------------------------------

# W&B API Key 설정
export WANDB_API_KEY="13d9653af6ac2f7f8ea29413f4c8eaaea35a2717"  # 여기에 본인의 W&B API Key를 입력하세요

# 실행 시작 시간 기록
START_TIME=$(date +%s)

# Python 스크립트 실행
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" main.py \
  --augmented_file_path "./data/augmented/" \
  --output_dir "./results" \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --lr 1e-4 \
  --epochs 1000 \
  --eval_epoch 1 \
  --save_epoch 100 \
  --num_workers 4 \
  --pretrained \
  --wandb_project "fake_detection" \
  --wandb_entity "dohyun12-korea-university" \
  --wandb_run_name "first"

# 실행 종료 시간 기록
END_TIME=$(date +%s)

# 실행 시간 출력
echo "** Training takes $(($END_TIME - $START_TIME)) seconds."
