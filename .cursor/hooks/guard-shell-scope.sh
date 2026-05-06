#!/usr/bin/env bash
# beforeShellExecution hook: 拦截危险/越界 shell 命令
# 仅做关键词级匹配，不做完整路径解析（shell 太复杂），细粒度交给 agent 自律 + AGENTS.md
set -uo pipefail

input=$(cat)
cmd=$(echo "$input" | jq -r '.command // empty')

deny() {
  jq -nc --arg c "$cmd" --arg r "$1" \
    '{permission:"deny",
      agent_message:("shell 命令被项目 hook 拒绝: \($r)。命令: \($c)"),
      user_message:("危险/越界 shell 命令已被拦截: " + $r)}'
  exit 0
}

# --- 1. 二次提权（已经是 root，禁止扩散） ---
echo "$cmd" | grep -qE '(^|[[:space:]&;|`(])(sudo|su)([[:space:]]|$)' \
  && deny "禁止 sudo/su 二次提权"

# --- 2. rm -rf / -R 命中只读 / 系统路径 ---
echo "$cmd" | grep -qE 'rm[[:space:]]+(-[a-zA-Z]*[rRf][a-zA-Z]*)[[:space:]]+(/|/home/xyjiang/(anaconda3|\.cursor|\.cursor-server|\.bashrc|\.condarc|\.gitconfig|projects)|/data(/|$)|~|\$HOME)' \
  && deny "rm -rf 命中只读/系统路径"
echo "$cmd" | grep -qE 'rm[[:space:]]+(-[a-zA-Z]*[rRf][a-zA-Z]*)[[:space:]]+(\*|/\*)([[:space:]]|$)' \
  && deny "rm 通配 / 整盘"

# --- 3. 重定向写入黑名单文件 ---
echo "$cmd" | grep -qE '>>?[[:space:]]*(/etc/|/home/xyjiang/(anaconda3|\.cursor|\.cursor-server|\.bashrc|\.condarc|\.gitconfig|projects)|/data/|~/\.bashrc|~/\.condarc|~/\.gitconfig)' \
  && deny "重定向写入黑名单路径"

# --- 4. chmod / chown -R 命中只读 / 系统根 ---
echo "$cmd" | grep -qE '(chmod|chown)[[:space:]]+(-[a-zA-Z]*R[a-zA-Z]*)[[:space:]]+(/|/data|/home/xyjiang)([[:space:]]|/|$)' \
  && deny "chmod/chown -R 命中只读/系统路径"

# --- 5. cp / mv / touch / mkdir / tee / dd 等写入 /data ---
echo "$cmd" | grep -qE '(^|[[:space:]&;|`(])(cp|mv|touch|mkdir|install|tee)([[:space:]]+-[a-zA-Z]+)*[[:space:]]+([^|]*[[:space:]])?/data/' \
  && deny "禁止往 /data 写（只读盘）"
echo "$cmd" | grep -qE '(^|[[:space:]&;|`(])dd[[:space:]]+.*of=/data/' \
  && deny "禁止 dd of=/data"
echo "$cmd" | grep -qE '(^|[[:space:]&;|`(])(rsync|scp)[[:space:]]+.*[[:space:]]/data/' \
  && deny "禁止 rsync/scp 写入 /data"

# --- 6. wget / curl 下载到黑名单 ---
echo "$cmd" | grep -qE '(wget[[:space:]]+(-[a-zA-Z]+[[:space:]]+)*-O[[:space:]]+|curl[[:space:]]+(-[a-zA-Z]+[[:space:]]+)*-o[[:space:]]+)(/data/|/home/xyjiang/(anaconda3|\.cursor|\.cursor-server|projects))' \
  && deny "禁止下载到黑名单路径"

# --- 7. pip install 落到 base / 导师 env ---
if echo "$cmd" | grep -qE '(^|[[:space:]&;|`(])pip[[:space:]]+install'; then
  conda_env="${CONDA_DEFAULT_ENV:-}"
  case "$conda_env" in
    ""|"base"|"eloftr"|"eloftr_training"|"loftr"|"Xoftr"|"gim"|"LightGlue"|"RoMa"|"glue-factory"|"minima"|"dust3r")
      deny "pip install 当前 CONDA_DEFAULT_ENV='${conda_env:-<未激活>}'，会污染 base/导师环境。先 conda activate eloftr_yurupeng" ;;
  esac
fi

# --- 8. conda env 操作命中导师环境 ---
echo "$cmd" | grep -qE 'conda[[:space:]]+(env[[:space:]]+remove|remove)[[:space:]]+.*(-n|--name)[[:space:]]+(base|eloftr|eloftr_training|loftr|Xoftr|gim|minima|dust3r|RoMa|LightGlue|glue-factory)' \
  && deny "禁止删除导师 conda env"
echo "$cmd" | grep -qE 'conda[[:space:]]+install[[:space:]]+.*(-n|--name)[[:space:]]+(base|eloftr|eloftr_training|loftr|Xoftr|gim|minima|dust3r|RoMa|LightGlue|glue-factory)' \
  && deny "禁止 conda install 到导师 env"

# --- 9. 系统级破坏命令 ---
echo "$cmd" | grep -qE '(^|[[:space:]&;|`(])(mkfs|shutdown|reboot|poweroff|halt|init[[:space:]]+0|init[[:space:]]+6)([[:space:]]|$)' \
  && deny "禁止系统级破坏命令"

echo '{"permission":"allow"}'
exit 0
