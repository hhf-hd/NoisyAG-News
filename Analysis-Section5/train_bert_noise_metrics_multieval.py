import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


@dataclass
class Metrics:
    # bookkeeping
    epoch: int
    eval_idx: int
    global_step: int
    epoch_step: int
    epoch_batches: int
    epoch_progress: float
    # global metrics
    train_acc_gt: float
    train_acc_noisy: float
    val_acc_gt: float
    val_acc_noisy: float
    # subset metrics
    train_clean_acc: float
    train_noisy_gt_acc: float
    train_noisy_nl_acc: float
    val_clean_acc: float
    val_noisy_gt_acc: float
    val_noisy_nl_acc: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "eval_idx": self.eval_idx,
            "global_step": self.global_step,
            "epoch_step": self.epoch_step,
            "epoch_batches": self.epoch_batches,
            "epoch_progress": round(self.epoch_progress, 6),
            "train_acc_gt": self.train_acc_gt,
            "train_acc_noisy": self.train_acc_noisy,
            "val_acc_gt": self.val_acc_gt,
            "val_acc_noisy": self.val_acc_noisy,
            "train_clean_acc": self.train_clean_acc,
            "train_noisy_gt_acc": self.train_noisy_gt_acc,
            "train_noisy_nl_acc": self.train_noisy_nl_acc,
            "val_clean_acc": self.val_clean_acc,
            "val_noisy_gt_acc": self.val_noisy_gt_acc,
            "val_noisy_nl_acc": self.val_noisy_nl_acc,
        }


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def compute_metrics(
    epoch: int,
    eval_idx: int,
    global_step: int,
    epoch_step: int,
    epoch_batches: int,
    yhat_train: np.ndarray,
    y_gt_train: np.ndarray,
    y_noisy_train: np.ndarray,
    yhat_val: np.ndarray,
    y_gt_val: np.ndarray,
    y_noisy_val: np.ndarray,
) -> Metrics:
    t_clean = (y_noisy_train == y_gt_train)
    t_noisy = ~t_clean
    v_clean = (y_noisy_val == y_gt_val)
    v_noisy = ~v_clean

    train_acc_gt = accuracy(y_gt_train, yhat_train)
    train_acc_noisy = accuracy(y_noisy_train, yhat_train)
    val_acc_gt = accuracy(y_gt_val, yhat_val)
    val_acc_noisy = accuracy(y_noisy_val, yhat_val)

    train_clean_acc = accuracy(y_gt_train[t_clean], yhat_train[t_clean])
    train_noisy_gt_acc = accuracy(y_gt_train[t_noisy], yhat_train[t_noisy])
    train_noisy_nl_acc = accuracy(y_noisy_train[t_noisy], yhat_train[t_noisy])

    val_clean_acc = accuracy(y_gt_val[v_clean], yhat_val[v_clean])
    val_noisy_gt_acc = accuracy(y_gt_val[v_noisy], yhat_val[v_noisy])
    val_noisy_nl_acc = accuracy(y_noisy_val[v_noisy], yhat_val[v_noisy])

    return Metrics(
        epoch=epoch,
        eval_idx=eval_idx,
        global_step=global_step,
        epoch_step=epoch_step,
        epoch_batches=epoch_batches,
        epoch_progress=epoch_step / max(1, epoch_batches),
        train_acc_gt=train_acc_gt,
        train_acc_noisy=train_acc_noisy,
        val_acc_gt=val_acc_gt,
        val_acc_noisy=val_acc_noisy,
        train_clean_acc=train_clean_acc,
        train_noisy_gt_acc=train_noisy_gt_acc,
        train_noisy_nl_acc=train_noisy_nl_acc,
        val_clean_acc=val_clean_acc,
        val_noisy_gt_acc=val_noisy_gt_acc,
        val_noisy_nl_acc=val_noisy_nl_acc,
    )


class TextDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def build_label_mapping(y_gt: np.ndarray, y_noisy: np.ndarray) -> Tuple[Dict[str, int], Dict[int, str]]:
    # HF-friendly: label2id keys must be strings
    all_vals = np.concatenate([y_gt, y_noisy], axis=0)
    class_names = sorted({str(v) for v in all_vals})
    label2id = {name: i for i, name in enumerate(class_names)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


def predict_all(model, dataloader, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            pred = torch.argmax(logits, dim=-1)
            preds.append(pred.cpu())
    return torch.cat(preds, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="BERT training with intra-epoch multi-evaluation logging (10 metrics each time).")
    parser.add_argument("--pkl", default="NoisyAG-News.pkl", help="Path to .pkl dataset")
    parser.add_argument("--label", default="human_middle_label", help="Noisy label column for supervision")
    parser.add_argument("--model", default="bert-base-uncased", help="HF model name or path")
    parser.add_argument("--epochs", type=int, default=2)
    # Evaluation schedule: either fixed per-epoch (eval-per-epoch), or staged schedule
    parser.add_argument("--eval-per-epoch", type=int, default=None, help="If set, use a fixed number of evaluations per epoch; otherwise use staged schedule.")
    parser.add_argument("--early-epochs", type=int, default=5, help="First K epochs use --early-evals evaluations per epoch")
    parser.add_argument("--early-evals", type=int, default=5, help="Evaluations per epoch for the first --early-epochs")
    parser.add_argument("--mid-epochs", type=int, default=5, help="Following M epochs use --mid-evals evaluations per epoch")
    parser.add_argument("--mid-evals", type=int, default=3, help="Evaluations per epoch for the following --mid-epochs after the early phase")
    parser.add_argument("--valid-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="metrics_bert_human_middle_multieval.csv")
    args = parser.parse_args()

    if args.eval_per_epoch is not None and args.eval_per_epoch < 1:
        raise ValueError("--eval-per-epoch must be >= 1 when provided")

    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    df = pd.read_pickle(args.pkl)
    need = ["text", "ground_truth", args.label]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[need].dropna(subset=["text", "ground_truth", args.label]).copy()
    df["text"] = df["text"].astype(str)

    X = df["text"].values
    y_gt_raw = df["ground_truth"].values
    y_nl_raw = df[args.label].values

    # split
    X_tr, X_va, y_gt_tr_raw, y_gt_va_raw, y_nl_tr_raw, y_nl_va_raw = train_test_split(
        X, y_gt_raw, y_nl_raw, test_size=args.valid_size, random_state=args.random_state, stratify=y_gt_raw
    )

    label2id, id2label = build_label_mapping(
        np.concatenate([y_gt_tr_raw, y_gt_va_raw]), np.concatenate([y_nl_tr_raw, y_nl_va_raw])
    )
    y_gt_tr = np.array([label2id[str(v)] for v in y_gt_tr_raw])
    y_gt_va = np.array([label2id[str(v)] for v in y_gt_va_raw])
    y_nl_tr = np.array([label2id[str(v)] for v in y_nl_tr_raw])
    y_nl_va = np.array([label2id[str(v)] for v in y_nl_va_raw])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    enc_tr = tokenizer(list(X_tr), truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
    enc_va = tokenizer(list(X_va), truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")

    ds_tr = TextDataset(enc_tr, y_nl_tr)  # train with noisy labels
    ds_va = TextDataset(enc_va, y_nl_va)

    # loaders
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_tr_eval = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, id2label=id2label, label2id=label2id
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dl_tr) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    metrics_rows: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch} training", leave=False)

        # choose evaluation step indices by fixed proportions per epoch
        # positions at i/m of the epoch, i = 1..m
        total_batches = len(dl_tr)
        if total_batches <= 0:
            eval_points = []
        else:
            if args.eval_per_epoch is not None:
                # fixed mode
                m = min(args.eval_per_epoch, total_batches)
            else:
                # staged schedule mode
                if epoch <= args.early_epochs:
                    per_epoch = max(1, args.early_evals)
                elif epoch <= args.early_epochs + args.mid_epochs:
                    per_epoch = max(1, args.mid_evals)
                else:
                    per_epoch = 1  # end-only
                m = min(per_epoch, total_batches)
            eval_points = [int(math.ceil(total_batches * i / m)) for i in range(1, m + 1)]
            # clamp, unique, sorted
            eval_points = [min(max(1, s), total_batches) for s in eval_points]
            eval_points = sorted(set(eval_points))
        next_eval_ptr = 0
        current_eval_idx = 0

        for b_idx, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # intra-epoch evaluation
            if next_eval_ptr < len(eval_points) and b_idx == eval_points[next_eval_ptr]:
                current_eval_idx += 1
                next_eval_ptr += 1
                # full predictions (no shuffle)
                yhat_tr = predict_all(model, dl_tr_eval, device)
                yhat_va = predict_all(model, dl_va, device)
                m = compute_metrics(
                    epoch=epoch,
                    eval_idx=current_eval_idx,
                    global_step=global_step,
                    epoch_step=b_idx,
                    epoch_batches=total_batches,
                    yhat_train=yhat_tr,
                    y_gt_train=y_gt_tr,
                    y_noisy_train=y_nl_tr,
                    yhat_val=yhat_va,
                    y_gt_val=y_gt_va,
                    y_noisy_val=y_nl_va,
                )
                row = m.to_dict()
                metrics_rows.append(row)
                print(
                    f"[Ep {epoch} Eval {current_eval_idx}/{len(eval_points)} step {b_idx}/{total_batches}] "
                    f"train_acc_gt={row['train_acc_gt']:.4f} | train_acc_noisy={row['train_acc_noisy']:.4f} | "
                    f"val_acc_gt={row['val_acc_gt']:.4f} | val_acc_noisy={row['val_acc_noisy']:.4f}"
                )

    out_path = args.output
    pd.DataFrame(metrics_rows).to_csv(out_path, index=False)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
