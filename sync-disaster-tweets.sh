#!/usr/bin/env bash
# 把本地 disaster-tweets 工程的最新提交同步到本仓库的 disaster-tweets/ 子目录。
# 本机专用（路径写死）；只同步 git 已跟踪的内容，models/ 等大文件不会进来。
set -e
PROJ="/home/zhang/code/kaggle/Natural Language Processing with Disaster Tweets"
REPO="$(cd "$(dirname "$0")" && pwd)"

echo "== 拉取远端最新 ==" && git -C "$REPO" pull --rebase -q
echo "== 导出已提交内容 ==" && rm -rf "$REPO/disaster-tweets" && mkdir -p "$REPO/disaster-tweets"
git -C "$PROJ" archive HEAD | tar -x -C "$REPO/disaster-tweets"
echo "== 提交推送 ==" && git -C "$REPO" add -A
git -C "$REPO" commit -q -m "sync: disaster-tweets @ $(git -C "$PROJ" rev-parse --short HEAD)" || echo "(无变化)"
git -C "$REPO" push -q origin main
echo "完成：$REPO/disaster-tweets 已同步到 GitHub"
