#!/bin/bash
# ============================================
# calcQs.py 使用脚本
# 功能：从 NoisyAG-News.pkl 计算 Q^S 矩阵
# ============================================

# 基本用法（使用默认参数）
python calcQs.py \
    --pkl ./NoisyAG-News.pkl \
    --noisy_col human_best_label \
    --gt_col ground_truth \
    --out_dir ./out_qs_human_best

# ============================================
# 参数说明：
# --pkl:        必需，指定输入的 pickle 文件路径
# --noisy_col:  噪声标签列名（默认: human_worst_label）
# --gt_col:     真实标签列名（默认: ground_truth）
# --out_dir:    输出目录（默认: ./out_qs_human_worst）
# ============================================

# 其他使用示例：
# 
# 1. 只指定必需参数（使用默认值）
# python calcQs.py --pkl ./NoisyAG-News.pkl
#
# 2. 指定不同的输出目录
# python calcQs.py --pkl ./NoisyAG-News.pkl --out_dir ./my_output
#
# 3. 使用不同的列名
# python calcQs.py --pkl ./data.pkl --noisy_col noisy_labels --gt_col true_labels
#
# ============================================
# 输出文件说明：
# - QS_full.npy/csv:        完整的 Q^S 矩阵（对角线≠0，用于分析）
# - QS_off.npy/csv:         对角线为0的 Q^S 矩阵（用于训练/CDS）
# - QS_full_heatmap.png:    完整矩阵的热力图
# - QS_off_heatmap.png:     对角线为0矩阵的热力图
# - QS_metrics.json:        解释性指标（纯度、吸引子强度、熵等）
# ============================================

