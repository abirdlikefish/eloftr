#!/bin/bash
# ============================================================================
# view_training.sh  (远程服务器版，对照 MyScripts/view_training.bat)
#
# 用法:
#   bash MyScripts/view_training.sh X [Y]
#   bash MyScripts/view_training.sh           # 不带参数会交互询问
#
#   X : 版本号，支持子版本: 6_1 或 6.1（内部统一规范化为 6_1）
#       匹配 logs/tb_logs/*_vX_*；自动跨数据集前缀（roadscene / m3fd / ...）
#   Y : run 类型 (可选，默认 3)
#       1 = debug   -> *_vX_debug
#       2 = small   -> *_vX_small
#       3 = final   -> *_vX_<其他后缀>（排除 _debug / _small）
#
# 行为:
#   - 自动在 6006..6020 内挑第一个空闲 TCP 端口
#   - 用 nohup 后台跑 tensorboard，stdout/stderr -> /tmp/tb_view_<PORT>.log
#   - 写入 PID 到 /tmp/tb_view_<PORT>.pid（方便事后 kill）
#   - HTTP 探活通过后立即返回，打印本地访问 URL 与 SSH 隧道命令
#   - 不再尝试启动浏览器（远程服务器无显示）
#   - 不再自动关闭 tensorboard；用打印出的 `kill <PID>` 自行清理
#
# 在本地（你的 Windows 机器）打开浏览器之前，先建立 SSH 隧道:
#   ssh -L 6006:localhost:6006 -p 8708 xyjiang@222.20.94.235
#   # 如果脚本选了 6007 端口，则相应改成:
#   # ssh -L 6007:localhost:6007 -p 8708 xyjiang@222.20.94.235
#   然后本地浏览器访问 http://localhost:<PORT>
#
# 子版本号过滤逻辑（与 .bat 完全一致）:
#   X=6 时不会撞到 m3fd_v6_1_finetune（_v6_ 后第一个字符是数字 -> 视为子版本，跳过）
#   X=6_1 时只匹配 m3fd_v6_1_finetune，不会撞到 v6
# ============================================================================
set -uo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
cd "$(dirname "$0")/.."

LOGS_DIR="logs/tb_logs"

# ---- read inputs -----------------------------------------------------------
X="${1:-}"
Y="${2:-}"

if [ -z "$X" ]; then
    read -r -p "Enter X [Y] (e.g. '6 1' / '6_1' / '6.1'; X required, Y optional 1=debug 2=small 3=final): " _input
    if [ -n "${_input:-}" ]; then
        IFS=', ' read -r _x _y <<< "$_input"
        X="${_x:-}"
        [ -z "$Y" ] && Y="${_y:-}"
    fi
fi

if [ -z "$X" ]; then
    echo "[ERROR] version number X is required."
    exit 1
fi

# 子版本: 6.1 -> 6_1（无点时 no-op，'5' / '6' 仍可用）
X="${X//./_}"
Y="${Y:-3}"

case "$Y" in
    1) WANT_SUFFIX="_debug"; WANT_FINAL=0 ;;
    2) WANT_SUFFIX="_small"; WANT_FINAL=0 ;;
    3) WANT_SUFFIX="";       WANT_FINAL=1 ;;
    *) echo "[ERROR] invalid Y=$Y (must be 1=debug, 2=small, 3=final)"; exit 1 ;;
esac

# ---- resolve experiment directory ------------------------------------------
EXP=""
shopt -s nullglob
for d in "$LOGS_DIR"/*_v"${X}"_*; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"

    tail_str="${name#*_v${X}_}"
    first="${tail_str:0:1}"
    case "$first" in
        [0-9]) continue ;;
    esac

    if [ "$WANT_FINAL" = "1" ]; then
        case "$name" in
            *_debug|*_small) continue ;;
        esac
        [ -z "$EXP" ] && EXP="$name"
    else
        case "$name" in
            *"$WANT_SUFFIX") [ -z "$EXP" ] && EXP="$name" ;;
        esac
    fi
done
shopt -u nullglob

if [ -z "$EXP" ]; then
    echo "[ERROR] no experiment found for v$X with Y=$Y"
    echo "        looked under: $LOGS_DIR/*_v${X}_*"
    exit 1
fi

TARGET="$LOGS_DIR/$EXP"
if [ ! -d "$TARGET" ]; then
    echo "[ERROR] experiment directory does not exist: $TARGET"
    exit 1
fi

# ---- find a free TCP port in 6006..6020 ------------------------------------
PORT=6006
while [ "$PORT" -le 6020 ]; do
    if ! ss -tln "( sport = :$PORT )" 2>/dev/null | grep -q LISTEN; then
        break
    fi
    PORT=$((PORT + 1))
done
if [ "$PORT" -gt 6020 ]; then
    echo "[ERROR] no free port in 6006-6020. Close some other tensorboard processes."
    exit 1
fi

# ---- header ----------------------------------------------------------------
echo
echo "============================================================"
echo "  Experiment : $EXP"
echo "  Log dir    : $TARGET"
echo "  Port       : $PORT"
echo "  URL (server-local) : http://localhost:$PORT"
echo
echo "  >>> 在本地机器上先建立 SSH 隧道 <<<"
echo "      ssh -L $PORT:localhost:$PORT -p 8708 xyjiang@222.20.94.235"
echo "  >>> 然后本地浏览器访问 <<<"
echo "      http://localhost:$PORT"
echo "============================================================"
echo

# ---- start tensorboard in background, capture PID --------------------------
LOG_FILE="/tmp/tb_view_${PORT}.log"
PID_FILE="/tmp/tb_view_${PORT}.pid"

nohup python -m tensorboard.main \
    --logdir "$TARGET" \
    --port "$PORT" \
    --bind_all \
    --reload_interval 5 \
    > "$LOG_FILE" 2>&1 &
TB_PID=$!
echo "$TB_PID" > "$PID_FILE"

echo "[INFO] tensorboard PID = $TB_PID (log: $LOG_FILE)"
echo "[INFO] waiting for it to bind port $PORT and serve HTTP ..."

# ---- poll TB until it's serving HTTP (cold start can take 5-10s) -----------
TRIES=0
MAX_TRIES=15
while [ "$TRIES" -lt "$MAX_TRIES" ]; do
    TRIES=$((TRIES + 1))
    if curl -fsS --max-time 5 "http://localhost:$PORT" -o /dev/null 2>/dev/null; then
        echo "[INFO] tensorboard is up on http://localhost:$PORT (took ~${TRIES}s)."
        echo
        echo "[STOP] 用以下命令停止 tensorboard:"
        echo "       kill $TB_PID         # 或: kill \$(cat $PID_FILE)"
        echo
        exit 0
    fi
    if ! kill -0 "$TB_PID" 2>/dev/null; then
        echo "[ERROR] tensorboard process died early. Last 30 lines of log:"
        tail -n 30 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep 1
done

echo "[ERROR] tensorboard not responding on http://localhost:$PORT after ${MAX_TRIES} tries."
echo "        check log: $LOG_FILE"
echo "        kill it with: kill $TB_PID"
exit 1
