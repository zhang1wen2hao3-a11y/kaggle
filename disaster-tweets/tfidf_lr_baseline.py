#!/usr/bin/env python3
"""Disaster Tweets 基线：TF-IDF + LogisticRegression

特点（相比常见 starter 的改进）：
  - 5 折交叉验证，评估用竞赛指标 F1（正类 = disaster），输出 均值±标准差
  - 同时给出两种分层方式的 CV：
      (a) 按 target 分层（常规做法）
      (b) 按 keyword 分层（复现赛题 train/test 的真实切分结构，CV 与 LB 更相关）
  - 特征里拼上 keyword（本数据里很强的信号）
  - 在 OOF 预测上搜索最优阈值（F1 最大），而不是固定 0.5
  - 输出 submission.csv

用法:
  python tfidf_lr_baseline.py                 # 默认读同目录的 train.csv/test.csv
  python tfidf_lr_baseline.py --folds 5 --out submission.csv
"""
from __future__ import annotations

import argparse
import html
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

try:                       # 允许在没有 exp_log.py 时独立运行
    import exp_log
except Exception:          # pragma: no cover
    exp_log = None

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")


def clean(text: str) -> str:
    """轻量清洗：反转义 HTML 实体，URL/@ 换成占位 token（保留"有没有"这一信息）。"""
    t = html.unescape(str(text))
    t = URL_RE.sub(" url ", t)
    t = MENTION_RE.sub(" user ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_text(df: pd.DataFrame) -> pd.Series:
    """keyword 拼进文本（keyword 是强特征，别浪费）。"""
    kw = df["keyword"].fillna("").astype(str)
    tx = df["text"].fillna("").map(clean)
    return (kw + " " + tx).str.strip()


def make_pipe(C: float = 1.0) -> Pipeline:
    """词级(1-2 gram) + 字符级(char_wb 3-5) TF-IDF 拼接 + 逻辑回归。

    char n-gram 在这份数据上稳定带来约 +0.01 F1（消融见 README/对话记录）。
    """
    feats = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2,
            max_features=60_000, strip_accents="unicode")),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            min_df=3, max_features=60_000)),
    ])
    return Pipeline([
        ("feats", feats),
        ("clf", LogisticRegression(C=C, max_iter=3000, solver="liblinear")),
    ])


def cv_eval(X: pd.Series, y: pd.Series, strat: pd.Series, folds: int, seed: int):
    """返回 (oof 概率, 每折 F1 列表)。strat 为分层依据。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    f1s = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # keyword 分层的类别过少警告
        for trn, val in skf.split(X, strat):
            pipe = make_pipe()
            pipe.fit(X.iloc[trn], y.iloc[trn])
            p = pipe.predict_proba(X.iloc[val])[:, 1]
            oof[val] = p
            f1s.append(f1_score(y.iloc[val], (p >= 0.5).astype(int)))
    return oof, f1s


def report(name: str, y: pd.Series, oof: np.ndarray, f1s: list[float]) -> None:
    pred = (oof >= 0.5).astype(int)
    print(f"\n【{name}】")
    print(f"  每折 F1: {[round(f, 4) for f in f1s]}")
    print(f"  CV F1  = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  OOF: acc={accuracy_score(y, pred):.4f}  P={precision_score(y, pred):.4f}  "
          f"R={recall_score(y, pred):.4f}  F1={f1_score(y, pred):.4f}")


def best_threshold(y: pd.Series, oof: np.ndarray):
    grid = np.arange(0.20, 0.71, 0.01)
    scores = [(f1_score(y, (oof >= t).astype(int)), float(t)) for t in grid]
    s, t = max(scores)
    return t, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--data-dir", type=Path, default=here)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="提交输出路径（默认 submissions/<name>.csv）")
    ap.add_argument("--name", default="tfidf_lr", help="实验名：用于 OOF/提交/实验记录")
    ap.add_argument("--log", action="store_true", help="写入 experiments.csv")
    ap.add_argument("--changes", default="", help="本次改动点（记录用）")
    ap.add_argument("--notes", default="", help="备注")
    args = ap.parse_args()
    if args.out is None:
        sub_dir = (exp_log.SUB_DIR if exp_log else here / "submissions")
        args.out = sub_dir / f"{args.name}.csv"

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    print(f"train={train.shape}  test={test.shape}  正类比例={train['target'].mean():.4f}")

    X = build_text(train)
    y = train["target"]
    X_test = build_text(test)

    # (a) 按 target 分层
    oof_t, f1_t = cv_eval(X, y, y, args.folds, args.seed)
    report("按 target 分层 CV", y, oof_t, f1_t)

    # (b) 按 keyword 分层（复现赛题切分结构）
    kw_strat = train["keyword"].fillna("__none__")
    oof_k, f1_k = cv_eval(X, y, kw_strat, args.folds, args.seed)
    report("按 keyword 分层 CV（更贴近 LB 切分）", y, oof_k, f1_k)

    # 阈值搜索
    thr, thr_f1 = best_threshold(y, oof_t)
    base_f1 = f1_score(y, (oof_t >= 0.5).astype(int))
    print(f"\n阈值搜索: 最优 thr={thr:.2f} -> OOF F1={thr_f1:.4f}  (0.5 时为 {base_f1:.4f})")

    # 全量训练 + 预测
    pipe = make_pipe()
    pipe.fit(X, y)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= thr).astype(int)

    sub = pd.read_csv(args.data_dir / "sample_submission.csv")
    sub["target"] = pred
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\n已写出 {args.out}")
    print(f"测试集预测正类比例 = {pred.mean():.4f}  (训练集正类比例 {y.mean():.4f})")

    # 保存 OOF（后续融合/阈值搜索/对比都要用）
    if exp_log is not None:
        exp_log.OOF_DIR.mkdir(exist_ok=True)
        np.save(exp_log.OOF_DIR / f"{args.name}.npy", oof_t)
        np.save(exp_log.OOF_DIR / f"{args.name}_keyword.npy", oof_k)
        np.save(exp_log.OOF_DIR / f"{args.name}_test.npy", prob)   # 供融合出提交
        if not (exp_log.OOF_DIR / "y_true.npy").exists():
            np.save(exp_log.OOF_DIR / "y_true.npy", y.values)
        print(f"OOF/测试概率已保存: oof/{args.name}.npy, _keyword.npy, _test.npy")
        if args.log:
            exp_id = exp_log.log_experiment(
                name=args.name, model="tfidf_lr",
                cv_scheme=f"target{args.folds}-seed{args.seed}",
                f1_mean=float(np.mean(f1_t)), f1_std=float(np.std(f1_t)),
                threshold=thr, f1_at_thr=float(thr_f1),
                oof_file=f"oof/{args.name}.npy", submission=str(args.out),
                changes=args.changes,
                notes=(args.notes + f" | keyword分层F1={np.mean(f1_k):.4f}").strip(" |"))
            print(f"已写入实验记录: {exp_id}")

    # 最有信息量的词（可选，帮助理解）
    vec = pipe.named_steps["feats"]
    coef = pipe.named_steps["clf"].coef_[0]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(coef)
    print("\n最像灾难的 12 个特征:", ", ".join(names[order[-12:]][::-1]))
    print("最不像灾难的 12 个特征:", ", ".join(names[order[:12]]))


if __name__ == "__main__":
    main()
