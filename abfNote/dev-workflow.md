# 开发工作流（2026-05 起）

> 6 个月后的 future-you 重读时：当前的边界是「**本地写代码、服务器跑训练**」。`.cursor/hooks/guard-*.sh` 已废弃删除，靠 agent 自律 + ask mode 纪律 + `.cursor/rules/01-server-ask-only.mdc` 软约束守住服务器。

## 一句话流程

```
本地 Windows (agent mode)  →  git push  →  服务器 vlrlab (ask mode)  →  手动 tmux 跑训练脚本
```

## 实际操作步骤

### A. 本地 Windows（开发端）

1. Cursor agent mode，用 `c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr\` 作为工作区
2. agent 改代码（cfg、模型、脚本都可以）
3. `ReadLints` 简单语法验证
4. `git status` / `git diff` 审改动
5. `git commit -m "..." && git push`

### B. 服务器 vlrlab（运行端）

1. **仅 ask mode**！`.cursor/rules/01-server-ask-only.mdc` 会反向提醒 agent 拒绝写工具，但人脑也要把关
2. SSH 进 tmux：`tmux new -s <task>` 或 `tmux attach -t <task>`
3. `cd /home/xyjiang/Desktop/yurupeng/eloftr && git pull --ff-only`
4. `conda activate eloftr_yurupeng`
5. `nvidia-smi` 看卡占用，`export CUDA_VISIBLE_DEVICES=N` 限定
6. `bash MyScripts/run_<exp>.sh`，detach `Ctrl+B D`
7. 训练完毕 → 看 `logs/tb_logs/<exp>/`，回本地 push 一份 `abfNote/<topic>.md` 实验笔记

### C. 不要做的事

- ❌ 不要在服务器开 agent mode 改代码（除非 §1.5 明示豁免）
- ❌ 不要在服务器 `git commit / push`（除非临时改了 cfg 试效果且明确告知本地）
- ❌ 不要把 `logs/` `weights/` `data/` 加进 git（已 `.gitignore`）
- ❌ 不要在 Windows 上手动转换 `.sh` 行尾符 —— `.gitattributes` 已自动钉 LF

## 一、为什么这样设计（旧方案 vs 新方案）

| 旧方案问题 | 新方案 |
|---|---|
| 服务器 agent mode + guard hook fail-closed → Windows 本地 hook 跑不了，开发被锁死 | 本地 agent mode 自由开发，服务器 ask mode 只读 |
| `.sh` 行尾符 CRLF/LF 跨平台炸 | `.gitattributes` 钉 LF |
| 在服务器跑 IDE 占 root 共用账号资源、agent 写文件 owner=root 污染共用账号 | 服务器只 SSH + tmux，不开 IDE agent |
| guard hook 路径硬编码 `/home/xyjiang/Desktop/yurupeng`，Windows 永不命中 → 100% deny | 删 guard，保留对话归档 hook（跨平台 Python） |
| guard hook 维护成本 vs 实际防护收益不对等 | agent 自律 + ask mode 纪律 + 01-server-ask-only rule 三层软约束 |

## 二、仍保留的边界（agent 自律 + 人脑把关）

详见 `AGENTS.md` §2-§4：

- 服务器禁止写 `/home/xyjiang/anaconda3/`、`/home/xyjiang/.cursor/`、`/data/` 等
- 禁止 `sudo` / `rm -rf` 白名单外 / `pip install` 到导师 env
- 训练前 `nvidia-smi` 看卡占用，限制 `CUDA_VISIBLE_DEVICES`
- 不要 `chown -R` / `chmod -R` 整目录

软约束的本质：agent 自律 + 人审 diff + 服务器 ask mode 不调写工具 = 三层防御。

## 三、例外情况

### 1. 服务器临时小改

SSH 在服务器上发现 cfg 有 typo 想立刻 patch：

- ✅ 手动 `vim` 改，**改完立即** `git add && git commit -m "hotfix: ..."` && `git push`，本地下次 pull 同步
- ❌ 不要借 agent mode 改

### 2. 本地 debug 需要数据

本地通常不跑训练，所以不需要数据集。如果偶尔本地 debug：

- 单独建 `data/<dataset>/` 目录或软链（已 `.gitignore`）
- 不要试图本地复现完整训练（缺 ckpt、缺 GPU）

### 3. chat-logs 自动归档

`.cursor/hooks/archive_transcript.py` + `.cursor/hooks.json`（`failClosed: false`）跨平台归档：

- Linux 用 `python3` 命中
- Windows 用 `python` 命中
- 归档目标 `.cursor/chat-logs/`，已 `.gitignore`，不入版本库
- 失败也无所谓（只是少一份备份）

## 四、新机器/新协作者 onboarding 清单

如果换台机器或别人接手：

1. `git clone <repo>`
2. 本地是 Windows → 直接 Cursor agent mode 开干，无需配置 hook（项目级 hook.json 只做归档）
3. 本地是 Linux 服务器（vlrlab） → 仅 ask mode，按本文档 §B 走
4. 读一遍 `AGENTS.md` 与 `.cursor/rules/00-scope-and-output.mdc`、`.cursor/rules/01-server-ask-only.mdc`
5. 第一次改完代码后跑一次 `git status` 确认 `.gitattributes` 没把现有文件大批"换行"误标 modified（如果有，跑 `git add --renormalize . && git commit -m "normalize line endings"` 一次性归一化）

## 五、相关文档

- `AGENTS.md` —— 项目级 agent 工作守则总纲
- `.cursor/rules/00-scope-and-output.mdc` —— 边界与输出物路径表
- `.cursor/rules/01-server-ask-only.mdc` —— 服务器 ask-only 软约束
- `.cursor/skills/eloftr-yurupeng-workspace/SKILL.md` —— 工作区差异层
- `.cursor/skills/eloftr-vlrlab-server/SKILL.md` —— 服务器通用 SOP
