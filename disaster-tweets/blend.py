#!/usr/bin/env python3
"""模型融合：把多个模型的 OOF/测试概率加权融合，搜权重+阈值，出提交。

依赖各模型跑完后保存的两份文件（由 tfidf_lr_baseline.py / finetune_transformer.py 产出）：
  oof/<name>.npy       该模型的 OOF 概率（必须同一套切分）
  oof/<name>_test.npy  该模型的测试集概率
  oof/y_true.npy       训练集标签

用法：
  python blend.py --names bert_base_v2 tfidf_lr_v1 --out submissions/blend_v1.csv
  python blend.py --names a b c --log --exp-name blend_v1     # 写入实验记录

做法：
  1) 权重搜索：坐标上升（步长 0.05），目标 = OOF F1
  2) 阈值搜索：在融合后的 OOF 上网格搜 F1 最大
  3) 用最优权重/阈值作用到测试概率，写出提交
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    import exp_log
except Exception:
    exp_log = None

HERE = Path(__file__).resolve().parent
OOF_DIR = HERE / "oof"


def best_threshold(y, p, lo=0.20, hi=0.71, step=0.01):
    grid = np.arange(lo, hi, step)
    scored = max((f1_score(y, (p >= t).astype(int)), float(t)) for t in grid)
    return scored[1], scored[0]


def search_weights(y, probs: list[np.ndarray], step=0.05, rounds=3):
    n = len(probs)
    w = np.ones(n) / n
    def score(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0: return -1.0
        w = w / w.sum()
        blend = sum(wi * pi for wi, pi in zip(w, probs))
        return best_threshold(y, blend)[1]
    cur = score(w)
    for _ in range(rounds):
        improved = False
        for i in range(n):
            for d in (step, -step):
                cand = w.copy(); cand[i] = max(0.0, cand[i] + d)
                if cand.sum() == 0: continue
                s = score(cand)
                if s > cur + 1e-6:
                    cur, w, improved = s, cand / cand.sum(), True
        if not improved:
            break
    return w / w.sum(), cur


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True, help="参与融合的实验名（对应 oof/<name>.npy）")
    ap.add_argument("--out", type=Path, default=HERE / "submissions" / "blend.csv")
    ap.add_argument("--data-dir", type=Path, default=HERE)
    ap.add_argument("--log", action="store_true", help="写入 experiments.csv")
    ap.add_argument("--exp-name", default="blend", help="实验名（记录用）")
    args = ap.parse_args()

    y = np.load(OOF_DIR / "y_true.npy")
    oofs, tests = [], []
    for n in args.names:
        f_oof, f_test = OOF_DIR / f"{n}.npy", OOF_DIR / f"{n}_test.npy"
        if not f_oof.exists() or not f_test.exists():
            raise SystemExit(f"缺少 {f_oof.name} 或 {f_test.name}；该模型需要重跑以保存测试概率")
        oofs.append(np.load(f_oof)); tests.append(np.load(f_test))
        print(f"  {n:22s} OOF F1={f1_score(y,(np.load(f_oof)>=0.5).astype(int)):.4f}")

    w, blend_oof_f1 = search_weights(y, oofs)
    print("\n权重:", {n: round(float(x), 3) for n, x in zip(args.names, w)})
    print(f"融合 OOF F1(优化中) = {blend_oof_f1:.4f}")

    blend_oof = sum(wi * pi for wi, pi in zip(w, oofs))
    thr, f1_thr = best_threshold(y, blend_oof)
    print(f"阈值搜索: thr={thr:.2f} -> 融合 OOF F1={f1_thr:.4f}")

    blend_test = sum(wi * pi for wi, pi in zip(w, tests))
    pred = (blend_test >= thr).astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(args.data_dir / "sample_submission.csv")
    sub["target"] = pred
    sub.to_csv(args.out, index=False)
    print(f"已写出 {args.out}；测试集正类比例={pred.mean():.4f}")

    if args.log and exp_log is not None:
        exp_log.OOF_DIR.mkdir(exist_ok=True)
        np.save(exp_log.OOF_DIR / f"{args.exp_name}.npy", blend_oof)
        np.save(exp_log.OOF_DIR / f"{args.exp_name}_test.npy", blend_test)
        exp_id = exp_log.log_experiment(
            name=args.exp_name, model="blend",
            cv_scheme="target5-seed42",
            f1_mean=float(f1_thr), f1_std=None, threshold=thr, f1_at_thr=float(f1_thr),
            oof_file=f"oof/{args.exp_name}.npy", submission=str(args.out),
            changes="融合: " + " + ".join(f"{n}*{round(float(x),2)}" for n, x in zip(args.names, w)),
            notes="权重与阈值均在 OOF 上搜索")
        print(f"已写入实验记录: {exp_id}")


if __name__ == "__main__":
    main()
