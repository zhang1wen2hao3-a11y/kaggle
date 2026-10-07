#!/usr/bin/env bash
# 把本地 the-pokemon-company 工程的最新提交同步到本仓库的 the-pokemon-company/ 子目录。
# 本机专用（路径写死）；只同步 git 已跟踪的内容，data/ 等大文件不会进来。
set -e
PROJ="/home/zhang/code/kaggle/the-pokemon-company"
REPO="$(cd "$(dirname "$0")" && pwd)"
NET_TIMEOUT="${SYNC_TIMEOUT:-60}"     # 网络操作超时（秒）

# 网络操作一律加超时：曾经因为 git pull --rebase 卡住导致整个同步挂死，
# 而且 commit 已经建好、只是没推送，从输出上很难看出卡在哪一步。
net() {
    local desc="$1"; shift
    if ! timeout "$NET_TIMEOUT" "$@"; then
        echo "  ⚠️  '$desc' 超时或失败（${NET_TIMEOUT}s）—— 本地提交已保留，可手动重试" >&2
        return 1
    fi
}

echo "== 拉取远端最新 =="
net "git pull" git -C "$REPO" pull --rebase -q || echo "  (跳过，继续用本地状态)"

echo "== 导出已提交内容 =="
rm -rf "$REPO/the-pokemon-company" && mkdir -p "$REPO/the-pokemon-company"
git -C "$PROJ" archive HEAD | tar -x -C "$REPO/the-pokemon-company"

echo "== 提交 =="
git -C "$REPO" add -A
git -C "$REPO" commit -q -m "sync: the-pokemon-company @ $(git -C "$PROJ" rev-parse --short HEAD)" \
    || echo "  (无变化)"

echo "== 推送 =="
if net "git push" git -C "$REPO" push -q origin main; then
    echo "完成：$REPO/the-pokemon-company 已同步到 GitHub"
else
    echo "本地提交已建好但未推送。手动重试：git -C $REPO push origin main" >&2
    exit 1
fi
