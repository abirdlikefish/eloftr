#!/usr/bin/env bash
# preToolUse hook: 拦截 Write/StrReplace/Delete/EditNotebook 工具的越界写入
# 边界约定见仓库根 AGENTS.md §2-3 与 .cursor/rules/00-scope-and-output.mdc
set -uo pipefail

input=$(cat)

# 从 tool_input 里提取目标路径（不同写工具字段名不同）
target=$(echo "$input" | jq -r '
  .tool_input.path
  // .tool_input.target_notebook
  // .tool_input.file_path
  // empty
')

# 没有路径字段（罕见）：放行
if [ -z "$target" ]; then
  echo '{"permission":"allow"}'
  exit 0
fi

# 解析为绝对路径（即便文件不存在也按字面量解析）
abs=$(realpath -m -- "$target" 2>/dev/null || echo "$target")

WHITELIST=(
  "/home/xyjiang/Desktop/yurupeng"
)

BLACKLIST=(
  "/home/xyjiang/anaconda3"
  "/home/xyjiang/.bashrc"
  "/home/xyjiang/.condarc"
  "/home/xyjiang/.gitconfig"
  "/home/xyjiang/.cursor"
  "/home/xyjiang/.cursor-server"
  "/home/xyjiang/projects"
  "/data"
)

# 1. 黑名单优先（即便落在白名单子树之外/内皆拒）
for b in "${BLACKLIST[@]}"; do
  case "$abs" in
    "$b"|"$b"/*)
      jq -nc --arg p "$abs" --arg b "$b" \
        '{permission:"deny",
          agent_message:("写入路径 \($p) 命中黑名单 \($b)，被项目 hook 拒绝。如确需修改请人工 shell 操作（详见 AGENTS.md §3）。"),
          user_message:("agent 试图写入黑名单目录已被拦截: " + $p)}'
      exit 0 ;;
  esac
done

# 2. 白名单放行
for w in "${WHITELIST[@]}"; do
  case "$abs" in
    "$w"|"$w"/*)
      echo '{"permission":"allow"}'
      exit 0 ;;
  esac
done

# 3. 都不命中 → 拒
jq -nc --arg p "$abs" \
  '{permission:"deny",
    agent_message:("路径 \($p) 不在仓库根 /home/xyjiang/Desktop/yurupeng 下，被项目 hook 拒绝（详见 AGENTS.md §2）。"),
    user_message:("agent 试图写入仓库外路径已被拦截: " + $p)}'
exit 0
