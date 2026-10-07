#!/usr/bin/env bash
# 把本地 the-pokemon-company 工程的最新提交同步到本仓库的 the-pokemon-company/ 子目录。
# 本机专用（路径写死）；只同步 git 已跟踪的内容，data/ 等大文件不会进来。
set -e
PROJ="/home/zhang/code/kaggle/the-pokemon-company"
REPO="$(cd "$(dirname "$0")" && pwd)"

echo "== 拉取远端最新 ==" && git -C "$REPO" pull --rebase -q
echo "== 导出已提交内容 ==" && rm -rf "$REPO/the-pokemon-company" && mkdir -p "$REPO/the-pokemon-company"
git -C "$PROJ" archive HEAD | tar -x -C "$REPO/the-pokemon-company"
echo "== 提交推送 ==" && git -C "$REPO" add -A
git -C "$REPO" commit -q -m "sync: the-pokemon-company @ $(git -C "$PROJ" rev-parse --short HEAD)" || echo "(无变化)"
git -C "$REPO" push -q origin main
echo "完成：$REPO/the-pokemon-company 已同步到 GitHub"
