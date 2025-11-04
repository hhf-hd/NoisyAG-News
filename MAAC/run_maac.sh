#!/bin/bash

# --- 1. GPU 设备 ---
export CUDA_VISIBLE_DEVICES=0

# --- 2. 关键路径 (请修改为你自己的路径) ---
# 包含 'text' 和 'noisy_col' 的训练数据
PKL_PATH="NoisyAG-News.pkl"
# 你的初始 Q^S 矩阵 (.npy 或 .csv)
# !!! 注意：MAAC 期望一个 "full Q^S" (对角线非零)。
# 如果你提供了 off-diag 矩阵, --qs_include_diag 会在
# update_qs_online 的第一次运行时将其修正为 full 矩阵。
QS_PATH="out_qs_human_best/QS_full.npy" 
# 用于评估的 test.csv
TEST_CSV_PATH="test.csv"

# --- 3. 关键列名和输出 ---
# 你在 .pkl 中使用的噪声标签列 (例如: human_worst_label)
NOISY_COL="human_best_label"
# 实验结果输出目录
OUT_DIR="./exp_human_best_maac-epoch6"


# --- 4. 执行训练 ---
python train.py \
    --pkl "$PKL_PATH" \
    --noisy_col "$NOISY_COL" \
    --qs_path "$QS_PATH" \
    --test_csv "$TEST_CSV_PATH" \
    --out_dir "$OUT_DIR" \
    \
    `# --- 模型与训练 ---` \
    --model "roberta-base" \
    --epochs 6 \
    --batch_size 32 \
    --max_len 256 \
    --lr 2e-5 \
    --warmup 0.1 \
    --seed 2027 \
    \
    `# --- 新方法 (MAAC) 策略 ---` \
    --use_maac \
    --maac_boost_factor 2.0 \
    --maac_world_class_idx 0 \
    --alpha0 0.1 \
    --E0 4 \
    \
    `# --- 关键：MAAC 必须使用 full Q^S ---` \
    --qs_include_diag \
    \
    `# --- (可选) 仍然可以与 ASO 结合使用 ---` \
    --use_aso \
    --lam 0.5 \
    \
    `# --- (推荐) 保持 Q^S 在线更新 ---` \
    --update_qs_online \
    --qs_update_until 6 \
    --qs_eta 0.30
    
echo "MAAC 实验完成: $OUT_DIR"