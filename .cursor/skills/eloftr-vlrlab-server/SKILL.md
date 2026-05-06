---
name: eloftr-vlrlab-server
description: 'VLRLab 4×RTX 3090 服务器（Ubuntu 20.04, CUDA driver 12.8 / nvcc 11.7, 80 CPU, 251GB RAM, 15TB /data）远程开发与代码同步 SOP。覆盖 SSH 私钥配置、Cursor Remote-SSH、git 双向同步、与导师 xyjiang 共用账号的安全守则、复用导师 eloftr_training 环境、/data 目录规划、数据集软链接、tmux 长任务、TensorBoard 端口转发、4 卡 DDP 训练。Use when 配置远程服务器 / 迁移代码到服务器 / 配置 SSH 私钥 / 用 Cursor 远程开发 / 同步本地服务器代码 / git push pull / 服务器上跑训练 / 服务器跑评估 / 多卡 DDP 训练 / Linux 环境装 PyTorch / 共用账号注意事项 / /data 大盘使用 / tmux 保活 / 端口转发 / 防止误删导师文件. Triggers: vlrlab / 3090 服务器 / 222.20.94.235 / 222.20.99.26 / 8708 端口 / xyjiang / Remote-SSH / git clone 到服务器 / scp / rsync / 服务器训练 / 4 卡训练 / DDP / 多卡 / Linux 环境 / conda activate eloftr_training / icacls 私钥权限 / UNPROTECTED PRIVATE KEY FILE / Permission denied publickey, English ''remote ssh'', ''multi-gpu training'', ''DDP launch'', ''shared account'', ''symbolic link dataset''. 与本地 Windows 单卡兼容补丁的关系见 eloftr-windows-setup（Linux 服务器不需要那 6 处补丁）.'
---

# VLRLab 3090×4 服务器使用 SOP

> 本 skill 记录毕设期间使用导师 VLRLab 4×RTX 3090 服务器的全套配置与日常工作流。
> 本地 Windows 单卡兼容补丁见 [eloftr-windows-setup](../eloftr-windows-setup/SKILL.md)（Linux 服务器+多卡上不需要那 6 处补丁）。

## 0. 服务器基本信息

| 项目 | 配置 |
|------|------|
| 主 IP / 备用 IP | `222.20.94.235` / `222.20.99.26` |
| SSH 端口 | `8708` |
| 用户 | `xyjiang`（**与导师共用，注意 §1 安全守则**） |
| OS / 内核 | Ubuntu 20.04.6 LTS / kernel 5.15 |
| GPU | 4× RTX 3090 24GB（共享，先 `nvidia-smi` 看占用） |
| Driver / CUDA | Driver 570.153.02，最高支持 CUDA 12.8；系统 nvcc 是 CUDA 11.7（PyTorch 不依赖系统 nvcc，自带 runtime） |
| CPU / RAM / shm | 80 cores / 251 GB / `/dev/shm` 126 GB |
| 系统盘 `/` | 819 GB（剩 580 GB），放代码、conda 环境 |
| 数据盘 `/data` | **15 TB（剩 7.8 TB），放数据集、日志、ckpt** ⭐ |

## 1. 共用账号安全守则（必读）

`xyjiang` 是导师本人账号，不是给学生新开的。**任何破坏性操作前先三思**：

1. 不在 `/home/xyjiang/` 根目录下乱建文件，集中放在 `/home/xyjiang/projects/<your-name>/`。
2. 不删除、不修改 `/home/xyjiang/anaconda3/envs/` 下导师已有环境，特别是匹配相关的 `eloftr` / `eloftr_training` / `loftr` / `Xoftr` / `LightGlue` / `RoMa` / `gim` / `glue-factory` / `minima` / `dust3r`。
3. **不往 base 环境装包**（污染所有人）。
4. 自己的环境用带名字后缀，如 `eloftr_<your-name>`。
5. 训练前必跑 `nvidia-smi`，避开导师正在用的 GPU；用 `CUDA_VISIBLE_DEVICES=N,M` 限制占用，不要无脑占满 4 卡。
6. 数据/日志/ckpt 全部放 `/data/xyjiang/<your-name>/`，**不要占系统盘**。
7. 第一次连上后，建议跟导师确认：（a）能否单开账号；（b）`/data/xyjiang/<your-name>/` 这个子目录可不可以建；（c）`/data` 上有没有现成的 M3FD / RoadScene。

## 2. 本地 SSH 私钥与 config（Windows）

### 2.1 私钥放置

私钥文件 `vlrlab` 由导师提供（**没有 `.pub` 后缀**才是私钥），放到 Windows 标准位置：

```powershell
mkdir C:\Users\abirdlikefish\.ssh -ErrorAction SilentlyContinue
move <下载位置>\vlrlab C:\Users\abirdlikefish\.ssh\vlrlab
```

设置权限（**必做**，否则 SSH 报 `WARNING: UNPROTECTED PRIVATE KEY FILE!`）：

```powershell
icacls C:\Users\abirdlikefish\.ssh\vlrlab /inheritance:r
icacls C:\Users\abirdlikefish\.ssh\vlrlab /remove "Everyone" "Users" "Authenticated Users" "BUILTIN\Users" 2>$null
icacls C:\Users\abirdlikefish\.ssh\vlrlab /grant:r "${env:USERNAME}:(R)"
```

验证：`icacls C:\Users\abirdlikefish\.ssh\vlrlab` 输出只应包含 `abirdlikefish:(R)` 一行。

### 2.2 SSH config

写入 `C:\Users\abirdlikefish\.ssh\config`：

```ssh-config
Host vlrlab
    HostName 222.20.94.235
    Port 8708
    User xyjiang
    IdentityFile C:\Users\abirdlikefish\.ssh\vlrlab
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host vlrlab-bak
    HostName 222.20.99.26
    Port 8708
    User xyjiang
    IdentityFile C:\Users\abirdlikefish\.ssh\vlrlab
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

之后短命令登录：`ssh vlrlab`（主 IP 不通就 `ssh vlrlab-bak`）。`ServerAliveInterval 60` 是为了 SSH 长时间空闲不被路由器/防火墙踢。

## 3. Cursor Remote-SSH 远程开发

代替命令行 vim 改代码的标准方案：

1. Cursor 装 `Remote - SSH` 扩展（微软官方）。
2. 左下角 `><` → `Connect to Host...` → 选 `vlrlab`。
3. 首次会在服务器装 VS Code Server (~200MB)，之后毫秒响应。
4. `File > Open Folder` 选 `/home/xyjiang/projects/<your-name>/efficient_loftr/`。
5. 终端、调试、AI 助手、Pylance 全在服务器上跑，体验和本地一致。

## 4. Git 同步本地↔服务器（推荐工作流）

### 4.1 一次性初始化

**本地** —— 确认 `.gitignore` 排除 `logs/` `__pycache__` `*.ckpt` `*.pth` `data/` `weights/`，然后：

```powershell
cd C:\Users\abirdlikefish\Desktop\毕设\efficient_loftr
git remote -v   # 确认有 GitHub 远程，没有就 git remote add origin <url>
git add . && git commit -m "init: ready for vlrlab" && git push
```

**服务器**：

```bash
mkdir -p /home/xyjiang/projects/<your-name>
cd /home/xyjiang/projects/<your-name>
git clone https://github.com/<your-username>/efficient_loftr.git
cd efficient_loftr
```

> Private repo 用 GitHub Personal Access Token (PAT) 替代密码：Settings → Developer settings → Personal access tokens (classic)，勾 `repo` 权限，clone 时输入用户名 + token。

### 4.2 日常双向同步

**铁律**：每次开始工作前先 `git pull`，结束工作前先 `git push`。

```bash
git pull          # 拉最新
# ... 改代码 / 跑实验 ...
git add . && git commit -m "..." && git push
```

### 4.3 大文件别走 git

数据集、ckpt、预训练权重单独走 rsync / scp，详见 §5、§6。

## 5. 目录组织（系统盘 vs 数据盘）

```
/home/xyjiang/projects/<your-name>/efficient_loftr/   ← 代码（小，git）
├── configs/  src/  MyScripts/
├── data       → 软链 → /data/xyjiang/<your-name>/efficient_loftr/data
├── logs       → 软链 → /data/xyjiang/<your-name>/efficient_loftr/logs
└── weights    → 软链 → /data/xyjiang/<your-name>/efficient_loftr/weights

/data/xyjiang/<your-name>/efficient_loftr/   ← 大文件（在 15TB 数据盘）
├── data/M3FD/  data/RoadScene/
├── logs/tb_logs/
└── weights/eloftr_outdoor.ckpt
```

建立软链命令：

```bash
mkdir -p /data/xyjiang/<your-name>/efficient_loftr/{data,logs,weights}
cd /home/xyjiang/projects/<your-name>/efficient_loftr
ln -s /data/xyjiang/<your-name>/efficient_loftr/data data
ln -s /data/xyjiang/<your-name>/efficient_loftr/logs logs
ln -s /data/xyjiang/<your-name>/efficient_loftr/weights weights
```

代码里仍然用相对路径 `data/M3FD`、`logs/tb_logs/...`，**无需修改任何 config**。

## 6. 数据集 / 预训练权重

**先找现成的，再决定上传**：

```bash
find /data -maxdepth 4 -type d \( -iname "*m3fd*" -o -iname "*roadscene*" \) 2>/dev/null
find /home/xyjiang -maxdepth 5 -name "eloftr*.ckpt" 2>/dev/null
find /data/xyjiang -maxdepth 5 -name "eloftr*.ckpt" 2>/dev/null
```

找到就软链，找不到就 rsync 上传到 `/data/xyjiang/<your-name>/efficient_loftr/data/` 后再软链。

**rsync 上传数据集**（本地 Git Bash）：

```bash
rsync -avz --progress \
    -e "ssh -i /c/Users/abirdlikefish/.ssh/vlrlab -p 8708" \
    "/c/Users/abirdlikefish/Desktop/毕设/efficient_loftr/data/M3FD/" \
    xyjiang@222.20.94.235:/data/xyjiang/<your-name>/efficient_loftr/data/M3FD/
```

## 7. Conda 环境（克隆导师的，不污染原环境）

导师已有 `eloftr` 与 `eloftr_training` 环境（路径 `/home/xyjiang/anaconda3/envs/`）。**绝不**直接 `conda activate eloftr_training` 然后 `pip install` —— 必须**克隆**：

```bash
conda create --name eloftr_<your-name> --clone eloftr_training
conda activate eloftr_<your-name>
```

激活后验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
# 期望: 2.x.x cuXX 4
```

如果克隆环境缺包，自由 `pip install`（这是你自己的副本）。需要从零建环境时参考：

```bash
conda create -n eloftr_<your-name> python=3.10 -y
conda activate eloftr_<your-name>
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 8. tmux 长任务保活

**铁律**：所有训练/长评估必须在 tmux 里跑，否则 SSH 一断进程就死。

```bash
tmux new -s train                    # 新建会话
conda activate eloftr_<your-name>
bash MyScripts/run_m3fd_v7_pcclahe.sh
# Ctrl+B 然后 D：detach（训练继续在后台跑，可以安全关闭 SSH）

# 下次想看进度
ssh vlrlab
tmux attach -t train

# 翻看历史输出: Ctrl+B 然后 [，q 退出翻页
# 列出会话: tmux ls
# 关闭会话: tmux kill-session -t train
```

## 9. 多卡 DDP 训练

4 × 24 GB = 96 GB 总显存，可比本地激进得多。Linux 上 NCCL 工作正常，可以打开真正的 DDP（不需要 [eloftr-windows-setup](../eloftr-windows-setup/SKILL.md) 第 2 节表 2、3 的单卡守卫，但保留也无害）。

启动模板（写成 `MyScripts/run_xxx.sh`）：

```bash
#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH
python train.py \
    configs/data/m3fd_trainval.py \
    configs/loftr/eloftr_full_v7_pcclahe.py \
    --exp_name=m3fd_v7_pcclahe_4gpu \
    --gpus=4 --num_nodes=1 --accelerator=ddp \
    --batch_size=4 --num_workers=8 \
    --max_epochs=30
```

要点：
- 真实 batch = `batch_size × num_gpus`，学习率按线性或 sqrt scale 调整。
- `num_workers` 可以拉大（80 核根本不愁）。
- 只用部分卡：`CUDA_VISIBLE_DEVICES=0,1 python train.py ... --gpus=2 --accelerator=ddp`。
- Windows 上为单卡禁用过的 DDPPlugin / sync_batchnorm 守卫在 Linux 多卡下无害（条件是 `WORLD_SIZE > 1`，多卡时自动激活）。

## 10. TensorBoard 端口转发

服务器跑 tensorboard，本地浏览器看：

```powershell
# 本地 PowerShell：建立隧道（前台保持开启）
ssh -L 6006:localhost:6006 vlrlab
```

```bash
# 隧道里
cd /home/xyjiang/projects/<your-name>/efficient_loftr
tensorboard --logdir=logs/tb_logs --port=6006
```

本地浏览器开 `http://localhost:6006`。端口被占就改成 16006/26006 等并相应调整 URL。

## 11. 取回结果到本地

```powershell
# 单个 ckpt
scp -i C:\Users\abirdlikefish\.ssh\vlrlab -P 8708 ^
    xyjiang@222.20.94.235:/data/xyjiang/<your-name>/efficient_loftr/logs/tb_logs/<exp>/version_0/checkpoints/last.ckpt ./

# 整个 tensorboard 目录
scp -i C:\Users\abirdlikefish\.ssh\vlrlab -P 8708 -r ^
    xyjiang@222.20.94.235:/data/xyjiang/<your-name>/efficient_loftr/logs/tb_logs/<exp> ./logs/tb_logs/
```

注意 scp 的端口参数是 **大写 -P**（区别于 ssh 的小写 -p）。

## 12. 常见坑速查

| 症状 | 原因 / 修复 |
|------|-------------|
| `WARNING: UNPROTECTED PRIVATE KEY FILE!` | 私钥权限太开放，重做 §2.1 的 icacls 三连。 |
| `Permission denied (publickey)` | 文件不是私钥（可能拿到了 `.pub`），或 IdentityFile 路径拼错。 |
| `Connection timed out` | 主 IP 不通，`ssh vlrlab-bak`；或挂校园 VPN（实验室服务器常常只对校园网开放）。 |
| `ssh -L` 端口被占 | 换端口 `ssh -L 16006:localhost:6006 vlrlab`，本地浏览器开 `:16006`。 |
| `pip install` 卡住 | 走清华源 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>`。 |
| 系统盘 `df -h ~` 暴涨 | 检查 `logs/` 是不是没软链到 `/data`，赶紧迁移。 |
| `nvidia-smi` 看到导师在跑 | `CUDA_VISIBLE_DEVICES=N,M` 限制自己用空闲卡，别挤。 |
| 关掉 SSH 训练就死 | 没用 tmux，回 §8。 |
| Cursor Remote-SSH 装 server 卡住 | 服务器外网受限或速度慢，让 Cursor 配代理或临时挂 VPN；首次安装通常 1-3 分钟。 |
| git push 提示用密码失败 | 现在 GitHub 不接受密码，必须用 PAT，见 §4.1。 |

## 13. 工作流速查（每天打开服务器后做什么）

```bash
# 1. 登录
ssh vlrlab

# 2. 看 GPU 占用，挑空闲卡
nvidia-smi

# 3. 进项目，先 pull
cd /home/xyjiang/projects/<your-name>/efficient_loftr
git pull

# 4. 激活自己的环境
conda activate eloftr_<your-name>

# 5. 进 tmux 跑训练
tmux attach -t train  # 或 tmux new -s train
bash MyScripts/run_xxx.sh
# Ctrl+B D 脱离

# 6. 另开一个本地 PowerShell：开 tensorboard 隧道看曲线
ssh -L 6006:localhost:6006 vlrlab
tensorboard --logdir=logs/tb_logs --port=6006
# 本地浏览器: http://localhost:6006
```
