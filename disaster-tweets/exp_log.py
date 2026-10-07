#!/usr/bin/env python3
"""实验记录表：为 Disaster Tweets 项目追加/查看实验结果。

设计目标：让"每次实验"都自动留痕，避免凭感觉判断改动是否有效。

文件（都在脚本同目录）：
  experiments.csv   实验记录表（可用 Excel 打开，utf-8-sig）
  oof/              每次实验的 OOF 预测（后续融合/阈值/对比都靠它）
  submissions/      每次实验产出的提交文件

命令行：
  python exp_log.py list                    # 按 CV F1 降序列出全部实验
  python exp_log.py md                      # 生成 实验记录.md（便于阅读/汇报）
  python exp_log.py add --name X --f1 0.8   # 手工追加一行

训练脚本里自动记录：
  from exp_log import log_experiment
  log_experiment(name="bert_v1", model="deberta-v3-base", cv_scheme="keyword5-42",
                 f1_mean=0.8321, f1_std=0.0075, threshold=0.45, f1_at_thr=0.8350,
                 oof_file="oof/bert_v1.npy", submission="submissions/bert_v1.csv",
                 changes="微调DeBERTa, AdamW 2e-5, 原始文本", notes="")
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "experiments.csv"
OOF_DIR = HERE / "oof"
SUB_DIR = HERE / "submissions"
MD_PATH = HERE / "实验记录.md"

COLUMNS = [
    "exp_id", "date", "name", "model", "cv_scheme",
    "f1_mean", "f1_std", "threshold", "f1_at_thr",
    "oof_file", "submission", "lb_score", "changes", "notes",
]
# 跨实验的公共基准（不同实验必须用同一套切分才可比）
CV_REF = "target5-seed42"


def _ensure() -> None:
    OOF_DIR.mkdir(exist_ok=True)
    SUB_DIR.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def read_all() -> list[dict]:
    _ensure()
    with CSV_PATH.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def next_id() -> str:
    rows = read_all()
    if not rows:
        return "EXP-0001"
    nums = [int(r["exp_id"].split("-")[1]) for r in rows if r.get("exp_id", "").startswith("EXP-")]
    return f"EXP-{max(nums) + 1:04d}"


def log_experiment(*, name: str, model: str = "", cv_scheme: str = CV_REF,
                   f1_mean: float | None = None, f1_std: float | None = None,
                   threshold: float | None = None, f1_at_thr: float | None = None,
                   oof_file: str = "", submission: str = "",
                   changes: str = "", notes: str = "") -> str:
    """追加一条实验记录，返回 exp_id。"""
    _ensure()
    exp_id = next_id()
    row = {
        "exp_id": exp_id,
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": name, "model": model, "cv_scheme": cv_scheme,
        "f1_mean": "" if f1_mean is None else f"{f1_mean:.4f}",
        "f1_std": "" if f1_std is None else f"{f1_std:.4f}",
        "threshold": "" if threshold is None else f"{threshold:.2f}",
        "f1_at_thr": "" if f1_at_thr is None else f"{f1_at_thr:.4f}",
        "oof_file": oof_file, "submission": submission,
        "changes": changes, "notes": notes,
    }
    with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
    return exp_id


def set_lb(exp_id: str, score: float) -> None:
    """为某条实验记录写入 Kaggle 排行榜分数（public LB）。"""
    rows = read_all()
    hit = False
    for r in rows:
        if r.get("exp_id") == exp_id:
            r["lb_score"] = f"{score:.5f}"; hit = True
    if not hit:
        raise SystemExit(f"未找到 {exp_id}")
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except ValueError:
        return 0.0


def cmd_list(_args) -> None:
    rows = read_all()
    if not rows:
        print("还没有实验记录。")
        return
    rows.sort(key=lambda r: -_f(r, "f1_mean"))
    best = _f(rows[0], "f1_mean")
    print(f"{'ID':9s} {'name':22s} {'model':18s} {'CV':16s} {'F1':>7s} {'±std':>7s} "
          f"{'@thr':>7s} {'LB':>9s} {'对比':>7s}")
    print("-" * 106)
    for r in rows:
        delta = _f(r, "f1_mean") - best
        print(f"{r['exp_id']:9s} {r['name'][:22]:22s} {r['model'][:18]:18s} "
              f"{r['cv_scheme'][:16]:16s} {_f(r,'f1_mean'):7.4f} {_f(r,'f1_std'):7.4f} "
              f"{_f(r,'f1_at_thr'):7.4f} {(r.get('lb_score') or '-'):>9s} {delta:+7.4f}")
    print(f"\n共 {len(rows)} 条记录；基准(CV) = {CV_REF}")
    print(f"表文件: {CSV_PATH}")


def cmd_md(_args) -> None:
    rows = read_all()
    rows.sort(key=lambda r: -_f(r, "f1_mean"))
    lines = [
        "# 实验记录（Disaster Tweets）",
        "",
        f"> 基准切分：`{CV_REF}`（所有实验必须同一套切分才可比）  ",
        f"> 生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}  ·  共 {len(rows)} 条",
        "",
        "| ID | 实验 | 模型 | CV 方案 | CV F1 | ±std | 阈值 | @阈值 F1 | **LB** | 改动点 | 备注 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| {exp_id} | {name} | {model} | {cv_scheme} | **{f1_mean}** | {f1_std} | "
                     "{threshold} | {f1_at_thr} | {lb_score} | {changes} | {notes} |".format(**{
                         k: str(r.get(k, "")).replace("|", "/") for k in COLUMNS}))
    lines += [
        "",
        "## 说明",
        "",
        "- **CV F1** 是唯一可信的对比依据；不要用 public LB 反复挑模型。",
        "- 每条实验若产出了 OOF，会存在 `oof/`，融合/阈值搜索都用它。",
        "- 新实验只有 `CV F1 ≥ 当前最佳` 才保留，否则回滚。",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {MD_PATH}")


def cmd_add(args) -> None:
    exp_id = log_experiment(
        name=args.name, model=args.model, cv_scheme=args.cv,
        f1_mean=args.f1, f1_std=args.std, threshold=args.thr, f1_at_thr=args.f1_thr,
        oof_file=args.oof or "", submission=args.sub or "",
        changes=args.changes or "", notes=args.notes or "")
    print(f"已记录 {exp_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出实验（按 CV F1 降序）").set_defaults(func=cmd_list)
    sub.add_parser("md", help="生成 实验记录.md").set_defaults(func=cmd_md)

    lb = sub.add_parser("lb", help="为某条实验写入 Kaggle LB 分数")
    lb.add_argument("--exp", required=True)
    lb.add_argument("--score", type=float, required=True)
    lb.set_defaults(func=lambda a: (set_lb(a.exp, a.score), print(f"{a.exp} LB={a.score:.5f}"))[-1])

    a = sub.add_parser("add", help="手工追加一条实验")
    a.add_argument("--name", required=True)
    a.add_argument("--model", default="")
    a.add_argument("--cv", default=CV_REF)
    a.add_argument("--f1", type=float, required=True)
    a.add_argument("--std", type=float, default=None)
    a.add_argument("--thr", type=float, default=None)
    a.add_argument("--f1-thr", type=float, default=None)
    a.add_argument("--oof", default=None)
    a.add_argument("--sub", default=None)
    a.add_argument("--changes", default=None)
    a.add_argument("--notes", default=None)
    a.set_defaults(func=cmd_add)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
