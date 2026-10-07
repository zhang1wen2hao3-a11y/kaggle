#!/usr/bin/env python3
"""把实验记录里最好的提交文件上传到 Kaggle（nlp-getting-started）。

前置条件（一次性）：
  1) 在 Kaggle 网页上【接受比赛规则】（Join Competition），否则 API 会返回 403
  2) 准备凭据：Kaggle → 头像 → Settings → API → Create New API Token
     下载 kaggle.json 放到 ~/.kaggle/kaggle.json，并 chmod 600
     （或设环境变量 KAGGLE_USERNAME / KAGGLE_KEY）

用法：
  python submit_to_kaggle.py --dry-run                 # 只显示会提交什么
  python submit_to_kaggle.py                           # 提交 CV F1 最高的一条
  python submit_to_kaggle.py --name bert_base_v1       # 指定实验
  python submit_to_kaggle.py --list                    # 查看已提交记录
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "experiments.csv"
COMPETITION = "nlp-getting-started"
KAGGLE = str(Path(sys.executable).parent / "kaggle")


def has_credentials() -> tuple[bool, str]:
    env_ok = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if env_ok:
        return True, "使用环境变量 KAGGLE_USERNAME/KAGGLE_KEY"
    if cfg.exists():
        return True, f"使用 {cfg}"
    return False, "未找到凭据：请放置 ~/.kaggle/kaggle.json 或设置 KAGGLE_USERNAME/KAGGLE_KEY"


def upload_host_reachable(timeout: float = 8.0) -> bool:
    """Kaggle 提交要把文件 PUT 到 Google 云存储；该域在国内常被墙。

    注意：被墙域名连 DNS 解析都会长时间阻塞，所以整体用线程 + 超时兜住。
    """
    import socket
    import concurrent.futures

    def probe() -> bool:
        try:
            socket.create_connection(("www.googleapis.com", 443), timeout=timeout).close()
            return True
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return ex.submit(probe).result(timeout=timeout + 2)
        except Exception:
            return False


def load_rows() -> list[dict]:
    with CSV_PATH.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("submission")]
    def f1(r):
        try:
            return float(r.get("f1_mean") or 0)
        except ValueError:
            return 0.0
    rows.sort(key=f1, reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="实验名（默认取 CV F1 最高且已产出提交的）")
    ap.add_argument("--file", type=Path, help="直接指定提交文件")
    ap.add_argument("--message", help="提交说明（默认包含实验名与 CV F1）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不真正提交")
    ap.add_argument("--list", action="store_true", help="列出该比赛的历史提交")
    ap.add_argument("--force", action="store_true", help="跳过上传目标可达性预检")
    args = ap.parse_args()

    ok, how = has_credentials()
    print(f"凭据: {how}")
    if not ok and not args.dry_run and not args.list:
        sys.exit("缺少 Kaggle 凭据，无法提交。")

    if args.list:
        subprocess.run([KAGGLE, "competitions", "submissions", "-c", COMPETITION])
        return

    if args.file:
        path, f1, name = args.file, None, args.file.stem
    else:
        rows = load_rows()
        if not rows:
            sys.exit("experiments.csv 里没有带 submission 的记录。")
        row = next((r for r in rows if r["name"] == args.name), rows[0]) if args.name else rows[0]
        path = Path(row["submission"])
        f1, name = row.get("f1_mean"), row["name"]
    if not path.exists():
        sys.exit(f"提交文件不存在: {path}")

    msg = args.message or (f"{name} | CV F1={f1}" if f1 else f"{name}")
    cmd = [KAGGLE, "competitions", "submit", "-c", COMPETITION, "-f", str(path), "-m", msg]
    print(f"将提交: {path}\n说明  : {msg}\n命令  : {' '.join(cmd)}")
    if args.dry_run:
        print("\n[dry-run] 未真正提交。")
        return
    if not args.dry_run:
        if not upload_host_reachable():
            print("\n[预检失败] 无法连接 www.googleapis.com（Kaggle 的提交文件上传目标）。\n"
                  "  这一步被墙时，CLI 会一直卡住直到超时。解决办法：\n"
                  "    · 在 Windows 侧开代理(如 Clash)，然后在 WSL 里带代理重试：\n"
                  "        HTTPS_PROXY=http://127.0.0.1:7890 python submit_to_kaggle.py\n"
                  "        (NAT 模式用宿主机 IP: https_proxy=http://172.28.224.1:7890)\n"
                  "    · 或换成能访问 Google 的网络/机器提交\n"
                  "  加 --force 可跳过本预检强行尝试。")
            if not args.force:
                sys.exit(2)

    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout or "", r.stderr or "")
    if r.returncode != 0:
        print("\n提交失败，常见原因：\n"
              "  · 403 -> 还没在网页上【接受比赛规则】(Join Competition)\n"
              "  · 400/403 -> 当日提交次数用尽（该比赛通常每天 5 次）\n"
              "  · 401 -> 凭据无效/过期")
    else:
        print("提交成功。用 `--list` 查看历史提交。")


if __name__ == "__main__":
    main()
