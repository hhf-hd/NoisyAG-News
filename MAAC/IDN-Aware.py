#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_cds_aso_evaltest.py
- 噪声训练 + 在 test.csv 上定点评估（共 36 次）
- 支持 CDS / ASO 独立或同时开启（--use_cds / --use_aso）
- ✅ NEW: 支持 Q^S 在线细化（GT-free, 仅用训练集+当前模型预测）并 EMA 融合 (--update_qs_online)
- 全程保存可复现实验与画图所需的中间与最终产物

[MODIFIED]: 集成了 MAAC (Mechanism-Aware Adaptive Correction) 方法。
"""

import os, json, math, argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup, set_seed)

# ==================== MAAC CONFIG ====================
# [MODIFIED]: Added global V_FALLBACK set based on eswa.pdf Figure 12
# 基于 eswa.pdf Figure 12 的 Fallback 触发词
V_FALLBACK = {
    "government", "nasa", "federal", "court", "china", 
    "washington", "president", "hurricane", "international", "space"
}
# =====================================================


# ==================== Data ====================
class TxtDataset(Dataset):
    # [MODIFIED]: Added v_fallback param and pre-lowercased texts for MAAC
    def __init__(self, texts, labels, tok, max_len=256, v_fallback=None):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len
        self.v_fallback = v_fallback or set() # <-- NEW
        # Pre-lowercase texts for efficient trigger word checking
        self.texts_lower = [t.lower() for t in self.texts] # <-- NEW

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i], # Use original case for tokenizer
            truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        
        # --- [MODIFIED] MAAC: Add has_trigger flag ---
        has_trigger = 0
        if self.v_fallback:
            # Check for trigger words in the pre-lowercased text
            text_lower = self.texts_lower[i]
            if any(kw in text_lower for kw in self.v_fallback):
                has_trigger = 1
        item['has_trigger'] = torch.tensor(has_trigger, dtype=torch.long)
        # --- End MAAC ---
        
        return item

def preprocess_test_csv(csv_path):
    df = pd.read_csv(csv_path)
    # 列名兼容
    col_class = [c for c in df.columns if c.strip().lower() in ["class index","class","label"]][0]
    col_title = [c for c in df.columns if c.strip().lower()=="title"][0]
    col_desc  = [c for c in df.columns if c.strip().lower() in ["description","desc"]][0]
    y = df[col_class].astype(int).to_numpy() - 1  # 1..4 -> 0..3
    texts = (df[col_title].fillna("").astype(str) + ". " + df[col_desc].fillna("").astype(str)).tolist()
    return texts, y

def set_all_seeds(seed=2027):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# ==================== 可视化保存 ====================
def save_qs_heatmap(QS, path_png, title='Q^S (row=noisy, col=direction)'):
    try:
        import matplotlib.pyplot as plt
        C = QS.shape[0]
        fig = plt.figure(figsize=(4.8, 4.4), dpi=180)
        plt.imshow(QS, aspect='equal')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.xlabel('b'); plt.ylabel('a')
        for i in range(C):
            for j in range(C):
                # [MODIFIED]: Only show off-diag text for clarity
                if i != j:
                    plt.text(j, i, f'{QS[i,j]:.2f}', ha='center', va='center', fontsize=8)
                # [MODIFIED]: Show diag text
                else:
                    plt.text(j, i, f'{QS[i,j]:.2f}', ha='center', va='center', fontsize=8, color='white')
        plt.tight_layout()
        plt.savefig(path_png)
        plt.close(fig)
    except Exception as e:
        print('[Warn] plot heatmap failed:', e, flush=True)

# ==================== CDS / ASO ====================
def cds_loss(logits, y_noisy, QS_t, alpha):
    """
    CDS: 目标分布 = (1-alpha)*onehot(y_noisy) + alpha*Q^S[y_noisy,:]
    
    [MODIFIED]: alpha can now be a scalar (standard CDS) or a [B] vector (MAAC)
    """
    C = logits.size(-1)

    # --- [MODIFIED] MAAC: Handle vector alpha ---
    # Ensure alpha is [B, 1] for broadcasting
    if not isinstance(alpha, float) and alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1) # [B] -> [B, 1]
    # --- End MAAC ---

    y1 = F.one_hot(y_noisy, num_classes=C).float()
    qy = QS_t[y_noisy]                  # [B,C] 取对应行
    
    # Broadcasting works for both scalar and [B, 1] alpha
    tgt = (1 - alpha) * y1 + alpha * qy 

    # 数值保险：精确归一并截断
    tgt = tgt / (tgt.sum(dim=-1, keepdim=True) + 1e-12)
    tgt = tgt.clamp_min(1e-8)
    logp = F.log_softmax(logits, dim=-1)
    return -(tgt * logp).sum(dim=-1).mean()

@torch.no_grad()
def pick_attractors(QS_t, k_attr=1):
    """
    ASO 吸引子：列和（入度）最大的前 k 个类
    """
    col_mass = QS_t.sum(dim=0)          # [C]
    return torch.topk(col_mass, k=k_attr).indices.tolist()

def apply_aso_(logits, y_noisy, attractors, lam, r):
    """
    ASO：对吸引子类列施加负偏置，抑制早期坍塌/回退（只对 y != k 的样本）。
    衰减系数 r 随 epoch 线性衰减到 0。
    """
    if r <= 0 or lam <= 0 or len(attractors)==0: return
    for k in attractors:
        mask = (y_noisy != k).float().to(logits.device)
        logits[:, k] -= lam * r * mask

# ==================== Eval on test ====================
@torch.no_grad()
def eval_on_test(model, tok, test_texts, test_labels,
                 device="cuda", batch_size=64, max_len=256, infer_workers=0):
    """
    在 test 集上评估，返回：
    - accuracy: 准确率
    - all_preds: 所有预测结果 (numpy array)
    - all_labels: 所有真实标签 (numpy array)
    - confusion_mat: 混淆矩阵 (numpy array)
    """
    ds = TxtDataset(test_texts, test_labels, tok, max_len=max_len) # Note: v_fallback not needed for eval
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,  # shuffle=False 确保顺序固定
        num_workers=infer_workers, pin_memory=device.startswith('cuda')
    )
    
    all_preds = []
    all_labels = []
    
    for batch in dl:
        # [MODIFIED]: Pop 'has_trigger' during eval
        batch.pop('has_trigger', None) 
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop('labels')
        logits = model(**batch).logits
        pred = logits.argmax(dim=-1)
        
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    
    # 合并所有batch
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算准确率
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / max(total, 1)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    C = int(max(all_labels.max(), all_preds.max())) + 1
    confusion_mat = confusion_matrix(all_labels, all_preds, labels=list(range(C)))
    
    return accuracy, all_preds, all_labels, confusion_mat

# ==================== Eval 节点调度（确保每 epoch 指定评测次数） ====================
def eval_nodes_per_epoch(num_batches, n_evals):
    """返回严格递增的（1..num_batches）中的 n_evals 个评测节点。"""
    if n_evals <= 0 or num_batches <= 0: return []
    nodes = np.linspace(1, num_batches, num=n_evals, endpoint=True)
    idx = [int(round(x)) for x in nodes]
    idx = [min(max(1, v), num_batches) for v in idx]
    used = set(); out = []
    for v in idx:
        w = v
        while w in used and w < num_batches:
            w += 1
        if w in used:
            w = v
            while w in used and w > 1:
                w -= 1
        if w not in used:
            used.add(w); out.append(w)
    if len(out) < n_evals:
        cur = 1
        while len(out) < n_evals and cur <= num_batches:
            if cur not in used:
                used.add(cur); out.append(cur)
            cur += 1
    out.sort()
    return out[:n_evals]

def cumulative_target_evals(epoch_idx):
    """训练到第 epoch_idx 个 epoch 结束时，总评测次数应为多少（总计 36 次）。"""
    if epoch_idx <= 3:
        return 6 * epoch_idx
    elif epoch_idx <= 6:
        return 18 + 4 * (epoch_idx - 3)
    else:
        return 30 + 1 * (epoch_idx - 6)

# ==================== ✅ NEW: Q^S 在线细化（GT-free） ====================
@torch.no_grad()
def refine_qs_epoch(model, tok, texts, y_noisy, C, device="cuda",
                      max_len=256, batch_size=64,
                      gate_conf=0.75, gate_margin=0.25,
                      topk=2, include_diag=False, mu_online=0.02):
    """
    用“训练集噪声标签 + 当前模型预测”重估一个 Q^S_new：
      - gate：只用置信度与 top-2 边距都足够的样本；
      - topk：只把概率质量分配给 top-k 方向，降低噪声；
      - 仅基于训练集；不使用 test.csv；
      - [MODIFIED]: include_diag controls if this is Q^S_full or Q^S_off
    """
    model.eval()
    # [MODIFIED]: Pass v_fallback=None, we don't need triggers for Q^S estimation
    dl = DataLoader(
        TxtDataset(texts, y_noisy, tok, max_len=max_len, v_fallback=None),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.startswith("cuda")
    )
    M = np.zeros((C, C), dtype=np.float64)
    for batch in dl:
        # [MODIFIED]: Pop 'has_trigger'
        batch.pop('has_trigger', None)
        y = batch.pop('labels').numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()   # [B,C]

        for i in range(p.shape[0]):
            a = int(y[i])           # 噪声标签（行）
            pb = p[i].copy()
            conf = float(pb.max())
            top2 = np.partition(pb, -2)[-2:]
            margin = float(top2[-1] - top2[-2])
            if conf < gate_conf or margin < gate_margin:
                continue

            if not include_diag:
                pb[a] = 0.0

            if topk >= C:
                q = pb / (pb.sum() + 1e-12)
                M[a, :] += q
            else:
                idx = np.argpartition(pb, -topk)[-topk:]
                q = np.zeros(C, dtype=np.float64); q[idx] = pb[idx]
                s = q.sum() + 1e-12
                if s > 0:
                    q /= s
                    M[a, :] += q
    
    # [MODIFIED]: Logic for smoothing/normalization depends on include_diag
    if include_diag:
        # --- Q^S_full logic ---
        # 1. Normalize full rows
        row_sums = M.sum(axis=1, keepdims=True) + 1e-12
        Q_new = M / row_sums
        # 2. Apply smoothing (e.g., uniform smoothing)
        U = np.ones((C, C), dtype=np.float64) / C
        Q_new = (1 - mu_online) * Q_new + mu_online * U
    else:
        # --- Q^S_off logic (original code) ---
        # 1. Fill diag with 0
        np.fill_diagonal(M, 0.0)
        # 2. Normalize off-diag rows
        row_sums = M.sum(axis=1, keepdims=True) + 1e-12
        Q_new = M / row_sums
        # 3. Apply off-diag uniform smoothing
        U = np.ones((C, C), dtype=np.float64); np.fill_diagonal(U, 0.0)
        U /= max(1, (C - 1))
        Q_new = (1 - mu_online) * Q_new + mu_online * U
        
    return torch.tensor(Q_new, dtype=torch.float32, device=device)

# ==================== Train ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', type=str, required=True)
    ap.add_argument('--noisy_col', type=str, required=True)  # e.g. human_middle_label
    ap.add_argument('--qs_path', type=str, required=True)    # 初始 Q^S：.npy or .csv
    ap.add_argument('--test_csv', type=str, required=True)
    ap.add_argument('--text_col', type=str, default='text')
    ap.add_argument('--model', type=str, default='roberta-base')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--alpha0', type=float, default=0.10)
    ap.add_argument('--E0', type=int, default=4)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--k_attr', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out_dir', type=str, default='./ckpt_cds_aso_eval')

    # 稳定性 & 开关
    ap.add_argument('--num_workers', type=int, default=0, help='DataLoader workers for training (0 safest)')
    ap.add_argument('--infer_workers', type=int, default=0, help='DataLoader workers for eval')
    ap.add_argument('--use_cds', action='store_true', help='enable CDS')
    ap.add_argument('--use_aso', action='store_true', help='enable ASO')

    # ✅ NEW: 在线细化 Q^S 的参数
    ap.add_argument('--update_qs_online', action='store_true', help='enable online refine of Q^S each epoch')
    ap.add_argument('--qs_eta', type=float, default=0.30, help='EMA 融合系数; QS <- (1-eta)*QS + eta*Q_new')
    ap.add_argument('--qs_update_until', type=int, default=6, help='仅在前 N 个 epoch 更新，后面冻结')
    ap.add_argument('--qs_gate_conf', type=float, default=0.75)
    ap.add_argument('--qs_gate_margin', type=float, default=0.25)
    ap.add_argument('--qs_topk', type=int, default=2)
    ap.add_argument('--qs_mu_online', type=float, default=0.02)

    # --- [MODIFIED] MAAC: Add new args ---
    ap.add_argument('--use_maac', action='store_true', 
                    help='Enable Mechanism-Aware Adaptive Correction (MAAC).')
    ap.add_argument('--maac_boost_factor', type=float, default=2.0, 
                    help='Boost factor for high-risk samples (alpha_boost = alpha_base * factor).')
    ap.add_argument('--maac_world_class_idx', type=int, default=0, 
                    help='Class index for the "World" category (the fallback target).')
    ap.add_argument('--qs_include_diag', action='store_true', 
                    help='Use full-posterior Q^S (Strategy B). Recommended for MAAC.')
    # --- End MAAC ---

    args = ap.parse_args()

    # --- [MODIFIED] MAAC: Add logic check ---
    if args.use_maac:
        if not args.use_cds:
            print("[Info] MAAC requires --use_cds. Enabling --use_cds.", flush=True)
            args.use_cds = True
        if not args.qs_include_diag:
            print("[Info] MAAC logic requires --qs_include_diag=True. Forcing it to True.", flush=True)
            args.qs_include_diag = True
    # --- End MAAC ---

    # 设备与并发设置
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[Warn] CUDA not available, fallback to CPU.', flush=True)
        args.device = 'cpu'
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print('[Args]', vars(args), flush=True)
    set_all_seeds(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load train noisy data ----
    df = pd.read_pickle(args.pkl)
    texts = df[args.text_col].astype(str).tolist()
    y_noisy_np = df[args.noisy_col].astype(int).to_numpy()
    C = int(max(y_noisy_np)) + 1

    # ---- load test data (GT only for eval) ----
    test_texts, test_labels = preprocess_test_csv(args.test_csv)
    assert len(test_texts) == len(test_labels)

    # ---- load Q^S (initial) ----
    QS = np.load(args.qs_path) if args.qs_path.endswith('.npy') else pd.read_csv(args.qs_path).to_numpy()
    assert QS.shape == (C, C), f"QS shape mismatch. expect {(C,C)} got {QS.shape}"
    QS_t = torch.tensor(QS, dtype=torch.float32, device=args.device)
    # 保存初始热力图，便于与第五章对齐
    save_qs_heatmap(QS, os.path.join(args.out_dir, "QS_init_heatmap.png"),
                      title='Q^S (init, row=noisy, col=direction)')

    # ---- tokenizer/model ----
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=C).to(args.device)

    # ---- train loader ----
    # --- [MODIFIED] MAAC: Pass v_fallback to Dataset ---
    v_fallback_set = V_FALLBACK if args.use_maac else None
    ds_tr = TxtDataset(texts, y_noisy_np, tok, max_len=args.max_len, v_fallback=v_fallback_set)
    # --- End MAAC ---
    
    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.device.startswith('cuda')
    )

    # ---- optimizer/scheduler ----
    optim = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(args.warmup * total_steps),
        num_training_steps=total_steps
    )

    # ---- 吸引子（ASO 用；可每 epoch 重算，这里按需在线刷新） ----
    attractors = pick_attractors(QS_t, k_attr=args.k_attr)

    # ---- 日志容器：用于画图讲故事 ----
    acc_series = []              # 36 次 test 精度序列
    preds_series = []            # 36 次预测结果
    labels_series = []           # 36 次真实标签（应该都一样，但记录下来保险）
    confusion_series = []        # 36 次混淆矩阵
    attractors_log = {}          # {epoch: [attr_ids]}
    qs_snapshots = []            # 保存每次在线更新后的快照文件名

    # ---- 训练循环 ----
    for ep in range(1, args.epochs + 1):
        model.train()
        
        # --- [MODIFIED] MAAC: Decouple CDS/ASO annealing ---
        # 1. ASO's annealing (r_aso) - remains as original 'r'
        r_aso = max(0.0, 1.0 - (ep - 1) / max(args.E0, 1))
        
        # 2. CDS/MAAC's base strength (alpha_base)
        if args.use_cds: # This check covers both CDS and MAAC
            # Fix "Mismatch 3": Use linear warmup for E0 epochs, then hold at alpha0.
            r_cds = min(1.0, ep / max(args.E0, 1.0)) 
            alpha_base = args.alpha0 * r_cds
        else:
            alpha_base = 0.0
        # --- End MAAC ---

        print(f"[Epoch {ep}] use_cds={args.use_cds} (alpha_base={alpha_base:.3f}), "
              f"use_aso={args.use_aso} (r_aso={r_aso:.3f}, lam={args.lam}), "
              f"use_maac={args.use_maac}, " # <-- [MODIFIED]
              f"target_evals_this_epoch={(6 if ep<=3 else 4 if ep<=6 else 1)}",
              flush=True)

        # 记录本 epoch 的吸引子（用于附录/可视化）
        attractors_log[ep] = list(attractors)

        # 设定该 epoch 的评测次数
        if ep <= 3: n_eval = 6
        elif ep <= 6: n_eval = 4
        else: n_eval = 1

        num_batches = len(dl_tr)
        eval_nodes = set(eval_nodes_per_epoch(num_batches, n_eval))

        # batch loop
        for bidx, batch in enumerate(tqdm(dl_tr, desc=f"Epoch {ep}/{args.epochs}")):
            
            # --- [MODIFIED] MAAC: Instance-adaptive alpha_i ---
            y_noisy = batch.pop('labels').to(args.device) # [B]

            if args.use_maac and (alpha_base > 0):
                has_trigger = batch.pop('has_trigger').to(args.device) # [B]
                
                # Identify high-risk samples: (label == World) AND (has_trigger == 1)
                is_fallback_target = (y_noisy == args.maac_world_class_idx)
                is_high_risk = (is_fallback_target & has_trigger.bool()).float() # [B], 0.0 or 1.0
                
                # Calculate boosted alpha
                alpha_boosted = min(alpha_base * args.maac_boost_factor, 0.99)
                
                # Final alpha_i vector [B]
                alpha_to_use = alpha_base + is_high_risk * (alpha_boosted - alpha_base)
            
            elif args.use_cds and (alpha_base > 0):
                # Standard CDS, use scalar alpha_base
                alpha_to_use = alpha_base
            
            else:
                # No CDS
                alpha_to_use = 0.0

            # Ensure 'has_trigger' is consumed if it exists but MAAC isn't active
            if 'has_trigger' in batch:
                batch.pop('has_trigger')
            # --- End MAAC ---

            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch); logits = out.logits

            # ASO：先改 logits (use r_aso)
            if args.use_aso and (r_aso > 0) and (args.lam > 0) and (len(attractors) > 0):
                apply_aso_(logits, y_noisy, attractors, lam=args.lam, r=r_aso)

            # 损失：CDS 或 CE
            if args.use_cds and (alpha_base > 0):
                # [MODIFIED]: Pass vector or scalar alpha_to_use
                loss = cds_loss(logits, y_noisy, QS_t, alpha=alpha_to_use)
            else:
                loss = F.cross_entropy(logits, y_noisy)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()

            # 到评测节点：在 test.csv 上评一次
            if (bidx + 1) in eval_nodes:
                model.eval()
                acc, preds, labels, conf_mat = eval_on_test(
                    model, tok, test_texts, test_labels,
                    device=args.device, batch_size=64, max_len=args.max_len,
                    infer_workers=args.infer_workers
                )
                acc_series.append(float(acc))
                preds_series.append(preds)
                labels_series.append(labels)
                confusion_series.append(conf_mat)
                model.train()

        # epoch 结束：按"累计目标次数"补齐（确保总数满足 36）
        print("acc_series: ", acc_series)
        target_total = cumulative_target_evals(ep)
        while len(acc_series) < target_total:
            model.eval()
            acc, preds, labels, conf_mat = eval_on_test(
                model, tok, test_texts, test_labels,
                device=args.device, batch_size=64, max_len=args.max_len,
                infer_workers=args.infer_workers
            )
            acc_series.append(float(acc))
            preds_series.append(preds)
            labels_series.append(labels)
            confusion_series.append(conf_mat)
            model.train()

        # ✅ NEW：在线细化 Q^S（仅前 qs_update_until 个 epoch）
        if args.update_qs_online and ep <= args.qs_update_until:
            
            # --- [MODIFIED] MAAC: Fix "Mismatch 1" ---
            # Pass args.qs_include_diag (forced to True if --use_maac)
            Q_new = refine_qs_epoch(
                model, tok, texts, y_noisy_np, C, device=args.device,
                max_len=args.max_len, batch_size=64,
                gate_conf=args.qs_gate_conf, gate_margin=args.qs_gate_margin,
                topk=args.qs_topk, 
                include_diag=args.qs_include_diag, # <-- MODIFIED
                mu_online=args.qs_mu_online
            )
            # --- End MAAC ---

            # EMA 融合
            QS_t = (1 - args.qs_eta) * QS_t + args.qs_eta * Q_new
            # 保存快照
            qs_np = QS_t.detach().cpu().numpy()
            snap_npy = os.path.join(args.out_dir, f"QS_epoch{ep}.npy")
            np.save(snap_npy, qs_np)
            snap_png = os.path.join(args.out_dir, f"QS_epoch{ep}.png")
            save_qs_heatmap(qs_np, snap_png, title=f'Q^S epoch {ep}')
            qs_snapshots.append(os.path.basename(snap_npy))
            # 吸引子更新
            attractors = pick_attractors(QS_t, k_attr=args.k_attr)
            
            # [MODIFIED] Added include_diag to log
            print(f"[Epoch {ep}] Q^S online-updated (eta={args.qs_eta}, include_diag={args.qs_include_diag}); "
                  f"attractors={attractors}", flush=True)

    # ---- 训练完毕：输出结果 ----
    # assert len(acc_series) == 36, f"Expect 36 evals, got {len(acc_series)}"
    acc_max = max(acc_series); argmax_idx = int(np.argmax(acc_series)) + 1  # 1-based

    # 最终 Q^S（可能已在线更新）
    QS_final = QS_t.detach().cpu().numpy()
    np.save(os.path.join(args.out_dir, "QS_final.npy"), QS_final)
    save_qs_heatmap(QS_final, os.path.join(args.out_dir, "QS_final_heatmap.png"),
                      title='Q^S (final)')

    # === 保存详细评估结果 ===
    # 1. 保存所有预测结果（每次评估的预测）
    np.save(os.path.join(args.out_dir, "predictions_series.npy"), 
            np.array(preds_series, dtype=object))  # shape: (n_evals,) 每个元素是 (n_test,) 的数组
    
    # 2. 保存所有真实标签（每次评估的标签，应该都一样）
    np.save(os.path.join(args.out_dir, "labels_series.npy"), 
            np.array(labels_series, dtype=object))
    
    # 3. 保存所有混淆矩阵
    np.save(os.path.join(args.out_dir, "confusion_matrices_series.npy"), 
            np.array(confusion_series))  # shape: (n_evals, C, C)
    
    # 4. 保存为更易读的格式：每次评估单独保存一个CSV（混淆矩阵）
    confusion_dir = os.path.join(args.out_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    for idx, conf_mat in enumerate(confusion_series, start=1):
        pd.DataFrame(conf_mat).to_csv(
            os.path.join(confusion_dir, f"confusion_matrix_eval{idx}.csv"), 
            index=True
        )
    
    # 5. 保存预测和标签的CSV（便于检查）
    predictions_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    for idx, (preds, labels) in enumerate(zip(preds_series, labels_series), start=1):
        pd.DataFrame({
            'sample_idx': np.arange(len(preds)),
            'true_label': labels,
            'pred_label': preds,
            'correct': (preds == labels).astype(int)
        }).to_csv(
            os.path.join(predictions_dir, f"predictions_eval{idx}.csv"),
            index=False
        )

    # 结果总表
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "acc_series": acc_series,
            "acc_max": acc_max,
            "acc_max_at_eval_idx": argmax_idx,
            "attractors_per_epoch": attractors_log,
            "qs_snapshots": qs_snapshots,
            "num_evaluations": len(acc_series),
            "num_test_samples": len(test_labels)
        }, f, indent=2)

    # 精度序列 CSV（便于画曲线）
    pd.DataFrame({"eval_idx": np.arange(1, len(acc_series) + 1), "acc": acc_series}).to_csv(
        os.path.join(args.out_dir, "acc_series.csv"), index=False
    )

    print(f"[Done] {len(acc_series)} test accuracies:\n{[round(x,4) for x in acc_series]}", flush=True)
    print(f"[Max] {acc_max:.4f} at evaluation #{argmax_idx}", flush=True)
    print(f"[Saved] 详细结果已保存:", flush=True)
    print(f"  - predictions_series.npy: 所有预测结果", flush=True)
    print(f"  - labels_series.npy: 所有真实标签", flush=True)
    print(f"  - confusion_matrices_series.npy: 所有混淆矩阵", flush=True)
    print(f"  - confusion_matrices/: 每次评估的混淆矩阵CSV", flush=True)
    print(f"  - predictions/: 每次评估的预测详情CSV", flush=True)

if __name__ == "__main__":
    main()