#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_qs_from_pkl.py
从 NoisyAG-News.pkl 计算 Q^S (human_worst_label):
- QS_full:  Q^S[a,b] = P(true=b | noisy=a)  （对角≠0，做分析/解释）
- QS_off:   仅 off-diag，行归一（对角=0，做训练/CDS）
并导出热力图与解释性指标。
"""

import os, argparse, json
import numpy as np
import pandas as pd

def save_heatmap(Q, path_png, title='Q^S'):
    try:
        import matplotlib.pyplot as plt
        C = Q.shape[0]
        fig = plt.figure(figsize=(4.8, 4.4), dpi=180)
        plt.imshow(Q, aspect='equal')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.xlabel('true class b'); plt.ylabel('noisy class a')
        for i in range(C):
            for j in range(C):
                plt.text(j, i, f'{Q[i,j]:.2f}', ha='center', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(path_png)
        plt.close(fig)
    except Exception as e:
        print('[Warn] plot heatmap failed:', e, flush=True)

def ensure_zero_based(arr):
    arr = np.asarray(arr).astype(int)
    amin, amax = int(arr.min()), int(arr.max())
    # 若是 1..K，转成 0..K-1
    if amin >= 1 and amax <= 4:
        return arr - 1
    return arr

def qs_full_from_counts(counts, eps=1e-12):
    """counts[a,b] = # {noisy=a, true=b} -> row-normalized Q^S (对角可能>0)"""
    Q = counts.astype(np.float64)
    row = Q.sum(axis=1, keepdims=True)
    # 行全0用极小伪计数避免 NaN（基本不会发生）
    row = row + eps
    Q = Q / row
    return Q

def qs_off_from_qs_full(Q_full, eps=1e-12):
    """从 full 版得到 off-diag 版（对角=0，行归一为1）"""
    Q = Q_full.copy()
    np.fill_diagonal(Q, 0.0)
    row = Q.sum(axis=1, keepdims=True)
    # 行和为0（极少数）则用均匀 off-diag
    C = Q.shape[0]
    U = (np.ones_like(Q) - np.eye(C)) / (C - 1)
    Q = np.where(row > eps, Q / (row + eps), U)
    return Q

def analysis_metrics(Q_full):
    """给第五章用的几个解释性指标"""
    C = Q_full.shape[0]
    # off-diag 方向分布
    Qoff = Q_full.copy()
    np.fill_diagonal(Qoff, 0.0)
    row = Qoff.sum(axis=1, keepdims=True) + 1e-12
    Qoff = Qoff / row

    purity = np.diag(Q_full).tolist()                     # 各 noisy 行的“纯度” Q[a,a]
    attract = Qoff.sum(axis=0).tolist()                   # 列入流（吸引子强度）
    asym = (Qoff - Qoff.T)                                # 不对称矩阵
    row_entropy = (-(Qoff * (np.log(Qoff + 1e-12))).sum(axis=1)).tolist()
    # 短板效应：对角-最大 off
    max_off = (Q_full - np.diag(np.diag(Q_full))).max(axis=1)
    short_plank = (np.diag(Q_full) - max_off).tolist()

    return {
        "purity_diag": purity,
        "attractor_colsum_off": attract,
        "row_entropy_off": row_entropy,
        "short_plank_margin": short_plank,
        # 需要矩阵时另存：
        "asym_off_matrix": asym.tolist()
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', type=str, required=True, help='NoisyAG-News.pkl')
    ap.add_argument('--noisy_col', type=str, default='human_worst_label')
    ap.add_argument('--gt_col', type=str, default='ground_truth')
    ap.add_argument('--out_dir', type=str, default='./out_qs_human_worst')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_pickle(args.pkl)

    noisy = ensure_zero_based(df[args.noisy_col].values)
    truey = ensure_zero_based(df[args.gt_col].values)
    C = int(max(noisy.max(), truey.max())) + 1
    # 构造全类别 crosstab，避免缺类
    noisy_cat = pd.Categorical(noisy, categories=list(range(C)))
    true_cat  = pd.Categorical(truey, categories=list(range(C)))
    ct = pd.crosstab(noisy_cat, true_cat, dropna=False)
    counts = ct.to_numpy().astype(np.int64)  # shape (C,C)

    # === Q^S 计算 ===
    QS_full = qs_full_from_counts(counts)           # 对角≠0（分析用）
    QS_off  = qs_off_from_qs_full(QS_full)          # 对角=0（训练/CDS用）

    # === 导出 ===
    np.save(os.path.join(args.out_dir, 'QS_full.npy'), QS_full)
    np.save(os.path.join(args.out_dir, 'QS_off.npy'),  QS_off)
    pd.DataFrame(QS_full).to_csv(os.path.join(args.out_dir, 'QS_full.csv'), index=False)
    pd.DataFrame(QS_off).to_csv(os.path.join(args.out_dir, 'QS_off.csv'), index=False)

    save_heatmap(QS_full, os.path.join(args.out_dir, 'QS_full_heatmap.png'),
                 title='Q^S (full) row=noisy col=true')
    save_heatmap(QS_off, os.path.join(args.out_dir, 'QS_off_heatmap.png'),
                 title='Q^S (off-diag) row=noisy col=true')

    # 解释性指标
    metrics = analysis_metrics(QS_full)
    with open(os.path.join(args.out_dir, 'QS_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "C": C,
            "counts": counts.tolist(),
            "metrics": metrics
        }, f, indent=2, ensure_ascii=False)

    # 控制台简报
    print('[QS_full row-sum ~1]:', np.round(QS_full.sum(axis=1), 6))
    print('[QS_off  row-sum =1]:', np.round(QS_off.sum(axis=1), 6))
    print('[purity(diag)]:', np.round(np.diag(QS_full), 3))
    print('[attractor(colsum_off)]:', np.round(np.array(metrics["attractor_colsum_off"]), 3))

if __name__ == '__main__':
    main()
