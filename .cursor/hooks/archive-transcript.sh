#!/usr/bin/env bash
# stop / sessionEnd hook: 把 ~/.cursor/projects/<encoded>/agent-transcripts/<uuid>/<uuid>.jsonl
# 镜像到本仓库 .cursor/chat-logs/<uuid>.jsonl
# 幂等：以 mtime 比较，已最新则跳过
# failClosed: false（归档失败仅少一份备份，不该卡住会话）
set -uo pipefail

# 读取 hook 输入但不强依赖（不同 hook 类型 schema 不同）
cat >/dev/null 2>&1 || true

PROJ_TRANSCRIPTS="/home/xyjiang/.cursor/projects/home-xyjiang-Desktop-yurupeng-eloftr/agent-transcripts"
DEST="/home/xyjiang/Desktop/yurupeng/eloftr/.cursor/chat-logs"

mkdir -p "$DEST" 2>/dev/null || true

if [ ! -d "$PROJ_TRANSCRIPTS" ]; then
  echo '{}'
  exit 0
fi

shopt -s nullglob
copied=0
total=0

for jsonl in "$PROJ_TRANSCRIPTS"/*/*.jsonl; do
  total=$((total + 1))
  base=$(basename "$jsonl")
  target="$DEST/$base"
  if [ ! -f "$target" ] || [ "$jsonl" -nt "$target" ]; then
    cp -f "$jsonl" "$target" 2>/dev/null && copied=$((copied + 1))
  fi
done

# stop hook output schema: 不接受 additional_context 之类，安全做法是输出空对象
echo "{\"_archived\": $copied, \"_total\": $total}"
exit 0
