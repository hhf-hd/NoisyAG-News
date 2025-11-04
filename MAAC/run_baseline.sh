#!/bin/bash
# --- 1. GPU 设备 ---
export CUDA_VISIBLE_DEVICES=0


PKL_PATH="NoisyAG-News.pkl"
QS_PATH="out_qs_human_best/QS_full.npy" 
TEST_CSV_PATH="test.csv"

# --- 3. 关键列名和输出 ---
# 你在 .pkl 中使用的噪声标签列 (例如: human_worst_label)
NOISY_COL="human_worst_label"
# 实验结果输出目录
OUT_DIR="./exp_baseline_cds_aso"

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
    --epochs 12 \
    --batch_size 32 \
    --max_len 256 \
    --lr 2e-5 \
    --warmup 0.1 \
    --seed 2027 \
    \
    `# --- Baseline (1.8%) 策略 ---` \
    --use_cds \
    --use_aso \
    --alpha0 0.1 \
    --lam 0.5 \
    --E0 4 \
    \
    `# --- Q^S 在线更新 (可选) ---` \
    --update_qs_online \
    --qs_update_until 6 \
    --qs_eta 0.30 \
    \
    `# --- 关键：Baseline 使用 off-diag Q^S (不设置 --qs_include_diag) ---` \
    # --qs_include_diag=False (这是默认值, 无需显式设置)
    
echo "Baseline 实验完成: $OUT_DIR"