#!/usr/bin/env python3
"""Disaster Tweets：微调预训练 Transformer（DeBERTa-v3 / BERT / RoBERTa）

与基线共用同一套评估协议：5 折 CV（分层可选 target / keyword）→ 输出 OOF → 搜阈值 → 出提交。

关键设置（对应阶段 2 的推荐配置）：
  - AdamW  lr 2e-5, weight_decay 0.01
  - 线性 warmup 10% + 线性衰减
  - 混合精度(fp16) + 梯度累积
  - EarlyStopping 盯 val F1，patience 2，restore_best_weights
  - 输入 = keyword + 轻清洗原始文本（不做词干化/去停用词）
  - 分类头用 transformers 自带 (num_labels=2)，Dropout 由 config 控制(--dropout)

权重来源（本机 HF 被墙，走 ModelScope）：
  resolve_model: 本地路径 → models/<name> → ModelScope snapshot_download → HF id

用法：
  # 冒烟测试（400 条、2 折、1 epoch，验证流程）
  python finetune_transformer.py --model AI-ModelScope/distilbert-base-uncased \
      --limit 400 --folds 2 --epochs 1 --name smoke

  # 快速基线（BERT-base, 5 折）
  python finetune_transformer.py --model google-bert/bert-base-uncased \
      --epochs 3 --bs 16 --name bert_base_v1 --log --changes "微调BERT-base"

  # 冲分（DeBERTa-v3-base，4GB 显存：bs 8 + accum 2）
  python finetune_transformer.py --model microsoft/deberta-v3-base \
      --epochs 4 --bs 8 --accum 2 --name deberta_v1 --log --changes "微调DeBERTa-v3-base"
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

try:
    import exp_log
except Exception:                      # 允许独立运行
    exp_log = None

from tfidf_lr_baseline import build_text          # 复用同一套文本构造（keyword + 轻清洗）

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"


# ----------------------------------------------------------------------------- 数据
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.labels = list(labels) if labels is not None else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def resolve_model(name: str) -> str:
    """本地 → models/ 缓存 → ModelScope → 原始 HF id。"""
    if Path(name).exists():
        return name
    local = MODELS_DIR / name.replace("/", "__")
    if (local / "config.json").exists():
        print(f"  使用本地权重: {local}")
        return str(local)
    try:
        from modelscope import snapshot_download
        print(f"  从 ModelScope 下载: {name} …")
        path = snapshot_download(name, cache_dir=str(MODELS_DIR))
        print(f"  已下载到: {path}")
        return path
    except Exception as e:                                   # pragma: no cover
        print(f"  ModelScope 失败({e})；尝试直接用 HF id（需能访问 HF）")
        return name


# ----------------------------------------------------------------------------- 训练
@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def load_model(path: str, dropout: float, num_labels: int = 2):
    """兼容不同 transformers 版本：有的版本不接受 dropout 覆盖参数。"""
    base = dict(num_labels=num_labels)
    for extra in ({"hidden_dropout_prob": dropout, "attention_probs_dropout_prob": dropout},
                  {"classifier_dropout": dropout},
                  {}):
        try:
            return AutoModelForSequenceClassification.from_pretrained(path, **base, **extra)
        except TypeError:
            continue
    raise RuntimeError("加载模型失败")


def train_fold(tr_texts, tr_y, va_texts, va_y, te_texts, tok, args, device, fold: int):
    collate = DataCollatorWithPadding(tok)
    tr_loader = DataLoader(TweetDataset(tr_texts, tr_y, tok, args.max_len),
                           batch_size=args.bs, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(TweetDataset(va_texts, va_y, tok, args.max_len),
                           batch_size=args.bs * 2, shuffle=False, collate_fn=collate)
    te_loader = DataLoader(TweetDataset(te_texts, None, tok, args.max_len),
                           batch_size=args.bs * 2, shuffle=False, collate_fn=collate)

    model = load_model(args.model_path, args.dropout).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(tr_loader) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if step % args.accum == 0 or step == len(tr_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        va_prob = predict(model, va_loader, device)
        f1 = f1_score(va_y, (va_prob >= 0.5).astype(int))
        print(f"    fold{fold} ep{ep}: loss={running/max(1,len(tr_loader)):.4f} "
              f"val_F1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best_f1 + 1e-5:
            best_f1, best_ep, bad = f1, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= args.patience:
                print(f"    fold{fold} 早停于 ep{ep}（最佳 ep{best_ep} F1={best_f1:.4f}）")
                break

    if best_state is not None:                       # restore_best_weights
        model.load_state_dict(best_state)
    va_prob = predict(model, va_loader, device)
    te_prob = predict(model, te_loader, device)
    return va_prob, te_prob, best_f1


# ----------------------------------------------------------------------------- 主流程
def main() -> None:
    here = HERE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=here)
    ap.add_argument("--model", default="microsoft/deberta-v3-base", help="骨干（HF id 或 ModelScope id）")
    ap.add_argument("--name", default="transformer_v1", help="实验名（OOF/提交/记录）")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=1, help="多种子平均（>1 时每个种子跑完整 CV）")
    ap.add_argument("--strat", choices=["target", "keyword"], default="target",
                    help="CV 分层依据（keyword 更贴近赛题真实切分）")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="梯度累积（4GB 显存建议 2）")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    ap.add_argument("--no-keyword", action="store_true", help="不要把 keyword 拼进文本")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟测试）")
    ap.add_argument("--log", action="store_true", help="写入 experiments.csv")
    ap.add_argument("--changes", default="", help="本次改动点（记录用）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    if args.limit:
        train = train.groupby("target", group_keys=False).apply(
            lambda d: d.head(max(1, args.limit // 2))).reset_index(drop=True)
        print(f"[冒烟] 训练集子集: train={train.shape}（test 保持完整，保证提交有效）")

    X = train["text"].fillna("").map(__import__("tfidf_lr_baseline").clean)
    if not args.no_keyword:
        X = (train["keyword"].fillna("").astype(str) + " " + X).str.strip()
    y = train["target"].values
    X_test = test["text"].fillna("").map(__import__("tfidf_lr_baseline").clean)
    if not args.no_keyword:
        X_test = (test["keyword"].fillna("").astype(str) + " " + X_test).str.strip()
    print(f"train={train.shape} test={test.shape} 正类比例={y.mean():.4f}  骨干={args.model}")

    print("准备 tokenizer/权重 …")
    args.model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    strat = y if args.strat == "target" else train["keyword"].fillna("__none__").values
    oof_sum = np.zeros(len(X))
    test_sum = np.zeros(len(X_test))
    fold_f1s = []
    n_runs = 0
    for s in range(args.seeds):
        seed = args.seed + s
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (trn, val) in enumerate(skf.split(X, strat), 1):
            print(f"\n== seed{seed} fold{fold}/{args.folds} ==")
            va_prob, te_prob, best = train_fold(
                X.iloc[trn], y[trn], X.iloc[val], y[val], X_test, tok, args, device, fold)
            oof_sum[val] += va_prob
            test_sum += te_prob
            fold_f1s.append(best)
            n_runs += 1

    oof = oof_sum / args.seeds
    test_prob = test_sum / n_runs
    print(f"\n每折最佳 val F1: {[round(f,4) for f in fold_f1s]}")
    print(f"CV F1 = {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"OOF F1(@0.5) = {f1_score(y, (oof >= 0.5).astype(int)):.4f}")

    grid = np.arange(0.20, 0.71, 0.01)
    scored = sorted(((f1_score(y, (oof >= t).astype(int)), float(t)) for t in grid), reverse=True)
    thr, thr_f1 = scored[0][1], scored[0][0]
    print("  阈值候选 top3:", [(round(t, 2), round(f, 4)) for f, t in scored[:3]])
    print(f"阈值搜索: thr={thr:.2f} -> OOF F1={thr_f1:.4f}")

    pred = (test_prob >= thr).astype(int)
    sub_dir = exp_log.SUB_DIR if exp_log else here / "submissions"
    sub_dir.mkdir(exist_ok=True)
    out = sub_dir / f"{args.name}.csv"
    sub = pd.read_csv(args.data_dir / "sample_submission.csv")
    sub["target"] = pred
    sub.to_csv(out, index=False)
    print(f"已写出 {out}；测试集正类比例={pred.mean():.4f}")

    if exp_log is not None:
        exp_log.OOF_DIR.mkdir(exist_ok=True)
        np.save(exp_log.OOF_DIR / f"{args.name}.npy", oof)
        np.save(exp_log.OOF_DIR / f"{args.name}_test.npy", test_prob)   # 供融合出提交
        if not (exp_log.OOF_DIR / "y_true.npy").exists():
            np.save(exp_log.OOF_DIR / "y_true.npy", y)
        print(f"OOF/测试概率已保存: oof/{args.name}.npy, oof/{args.name}_test.npy")
        if args.log:
            exp_id = exp_log.log_experiment(
                name=args.name, model=args.model.split("/")[-1],
                cv_scheme=f"{args.strat}{args.folds}-seed{args.seed}",
                f1_mean=float(np.mean(fold_f1s)), f1_std=float(np.std(fold_f1s)),
                threshold=thr, f1_at_thr=float(thr_f1),
                oof_file=f"oof/{args.name}.npy", submission=str(out),
                changes=args.changes,
                notes=f"bs={args.bs} accum={args.accum} lr={args.lr} ep<={args.epochs} "
                      f"maxlen={args.max_len} seeds={args.seeds} amp={not args.no_amp}")
            print(f"已写入实验记录: {exp_id}")


if __name__ == "__main__":
    main()
