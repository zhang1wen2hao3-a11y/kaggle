#!/usr/bin/env python3
"""经 Windows 侧代理完成 Kaggle 提交。

背景：Kaggle 提交要把文件 PUT 到 www.googleapis.com（Google 云存储），该域在本机
WSL 里被墙；而 Windows 侧的代理(Clash/UniClash 等)只监听 127.0.0.1，WSL 用不了。
做法：三步式提交里，第 1、3 步（kaggle.com API）在 WSL 直接跑；
      第 2 步（上传到 Google 云存储）交给 **Windows 的 curl.exe** 走 Windows 回环代理完成。

前置：Windows 侧代理已开启（核心在运行），并知道其 HTTP 代理端口。
用法：
  python submit_via_proxy.py --file submissions/blend_v1.csv --port 7993
  python submit_via_proxy.py --file submissions/blend_v1.csv --port 7993 --message "blend_v1 CV0.8089"
  python submit_via_proxy.py --check --port 7993        # 只检测代理是否可用
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURL_WIN = "/mnt/c/Windows/System32/curl.exe"
WIN_TMP_ROOT = Path("/mnt/c/Users/zhang/AppData/Local/Temp/kaggle_submit")


def win_path(p: Path) -> str:
    s = str(p)
    if s.startswith("/mnt/") and len(s) > 7:
        drive, rest = s[5].upper(), s[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    raise SystemExit(f"路径不在 Windows 可见盘上: {p}")


def detect_proxy_port() -> int | None:
    """从 Windows 注册表读取系统代理端口（UniClash/Clash 常用做法）。"""
    reg = "/mnt/c/Windows/System32/reg.exe"
    try:
        out = subprocess.run(
            [reg, "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "ProxyServer" in line:
            val = line.split()[-1]                      # 形如 127.0.0.1:7993
            if ":" in val:
                try:
                    return int(val.rsplit(":", 1)[1])
                except ValueError:
                    return None
    return None



def proxy_ok(port: int, timeout: int = 12) -> bool:
    r = subprocess.run(
        [CURL_WIN, "-s", "-o", "NUL", "-w", "%{http_code}", "--max-time", str(timeout),
         "-x", f"http://127.0.0.1:{port}", "https://www.googleapis.com"],
        capture_output=True, text=True)
    code = (r.stdout or "").strip()
    print(f"  代理 127.0.0.1:{port} 访问 googleapis -> {code or '失败'}")
    return code not in ("", "000")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="提交文件")
    ap.add_argument("--port", type=int, help="Windows 侧 HTTP 代理端口（缺省自动读注册表系统代理）")
    ap.add_argument("--competition", default="nlp-getting-started")
    ap.add_argument("--message", default=None)
    ap.add_argument("--check", action="store_true", help="只检测代理连通性")
    args = ap.parse_args()

    if not Path(CURL_WIN).exists():
        sys.exit(f"找不到 Windows curl: {CURL_WIN}")
    if not args.port:
        args.port = detect_proxy_port()
        print(f"自动检测到系统代理端口: {args.port}")
        if not args.port:
            sys.exit("无法自动检测代理端口，请用 --port 指定")
    if not proxy_ok(args.port):
        sys.exit("代理不可用：请在 UniClash/Clash 里确认『核心已启动 + 系统代理已开』，"
                 "并用端口检测确认；也可在客户端里查看当前 HTTP 代理端口。")
    if args.check:
        print("代理可用 ✓")
        return
    if not args.file or not args.file.exists():
        sys.exit("请用 --file 指定提交文件")

    # 复制到 Windows 可见路径（curl.exe 读 WSL 路径不稳）
    WIN_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    dst = WIN_TMP_ROOT / args.file.name
    shutil.copy2(args.file, dst)
    wp = win_path(dst)
    print(f"文件已就位: {wp}  ({dst.stat().st_size} 字节)")

    # 依赖 kaggle 官方库完成第 1、3 步
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import (
        ApiCreateSubmissionRequest, ApiStartSubmissionUploadRequest)

    api = KaggleApi()
    api.authenticate()

    req = ApiStartSubmissionUploadRequest()
    req.competition_name = args.competition
    req.file_name = args.file.name
    req.content_length = dst.stat().st_size
    req.last_modified_epoch_seconds = int(dst.stat().st_mtime)
    with api.build_kaggle_client() as kaggle:
        start = kaggle.competitions.competition_api_client.start_submission_upload(req)
        print(f"上传会话已创建 -> {start.create_url[:60]}…")

        # 第 2 步：交给 Windows curl 走代理上传
        up = subprocess.run(
            [CURL_WIN, "-sS", "-x", f"http://127.0.0.1:{args.port}",
             "-X", "PUT",
             "-H", "Content-Type: application/octet-stream",
             "--data-binary", f"@{wp}",
             "-w", "%{http_code}", "-o", "NUL", start.create_url],
            capture_output=True, text=True)
        code = (up.stdout or "").strip()
        print(f"上传结果 HTTP {code or '失败'} {up.stderr.strip()[:200]}")
        if code not in ("200", "201", "308"):
            sys.exit("上传未成功，请检查代理是否稳定（308 也是正常的分段响应）。")

        submit = ApiCreateSubmissionRequest()
        submit.competition_name = args.competition
        submit.blob_file_tokens = start.token
        if args.message:
            submit.submission_description = args.message
        resp = kaggle.competitions.competition_api_client.create_submission(submit)
    print("提交成功：", resp)
    print("用 `kaggle competitions submissions -c nlp-getting-started` 查看计分状态。")


if __name__ == "__main__":
    main()
