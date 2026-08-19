#!/usr/bin/env bash
# 一键构建前端并同步到后端静态托管目录（app/static/）
# 用法：
#   bash scripts/web-sync.sh             # 完整构建 + 同步
#   bash scripts/web-sync.sh --skip-build # 跳过构建，直接同步现有 web/out 产物
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB="$ROOT/web"
DEST="$ROOT/app/static"

# ---------- 1. 构建 ----------
if [[ "${1:-}" == "--skip-build" ]]; then
  echo "==> 跳过构建，使用现有 web/out 产物"
else
  echo "==> 构建前端 (web/out)..."
  (cd "$WEB" && npm run build)
fi

# ---------- 2. 检查产物 ----------
if [[ ! -d "$WEB/out" ]]; then
  echo "错误: 未找到 $WEB/out，请先构建（去掉 --skip-build）" >&2
  exit 1
fi

# ---------- 3. 同步到后端静态目录 ----------
echo "==> 清空 $DEST 并同步..."
rm -rf "$DEST"/*
cp -r "$WEB/out/." "$DEST/"

echo "==> 完成！前端已部署到 app/static/"
echo "    刷新浏览器（Ctrl+Shift+R 强制刷新）即可看到最新效果"
