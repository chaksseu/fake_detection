#!/bin/bash

# 기본 설정
DATA_PATH="./data/processed"
OUTPUT_PATH="./data/augmented"
NUM_COLOR_JITTER=10
NUM_NEG=100

# mode별 설정
MODES=("train" "valid" "test")
DATA_TYPES=("normal" "fraud")

for MODE in "${MODES[@]}"; do
    for DATA_TYPE in "${DATA_TYPES[@]}"; do
        echo "Processing $MODE data for $DATA_TYPE..."

        if [ "$DATA_TYPE" == "fraud" ]; then
            # fraud 데이터 처리 (negative 생성 비활성화)
            python data/test-gpt.py \
                --data_path "$DATA_PATH" \
                --data_type "$DATA_TYPE" \
                --mode "$MODE" \
                --output_path "$OUTPUT_PATH" \
                --num_color_jitter "$NUM_COLOR_JITTER"
        else
            # normal 데이터 처리
            python data/test-gpt.py \
                --data_path "$DATA_PATH" \
                --data_type "$DATA_TYPE" \
                --mode "$MODE" \
                --output_path "$OUTPUT_PATH" \
                --num_pseudo "$NUM_NEG" \
                --num_color_jitter "$NUM_COLOR_JITTER" \
                #--generate_negatives
        fi
    done
done
