---
name: eloftr-megadepth-syn-data
description: |
  Megadepth_Syn dataset integration for EfficientLoFTR v10 cross-modal
  scale-up experiment. Megadepth_Syn = MegaDepth Internet tourist
  photos (Mt Rushmore, Colosseum, etc.) + style-transferred "infrared"
  modality, 195 train scenes (~129K (IR_i, VIS_i) co-located pairs,
  pixel-aligned by construction) + 2 test scenes (806 pairs in
  Undistorted_SfM/0015+0022, MegaDepth1500-compatible layout, NOT
  used by v10).
  Storage: /data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/
  on vlrlab (509 GB total, v10 uses train/infrared + train/phoenix
  ~ 222 GB IR + VIS imgs; train/depth/event/normal/paint/sketch
  unused).
  Asymmetric train vs test layout: train/{infrared/phoenix/S6/zl548/
  MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.jpg, phoenix/S6/zl548/
  MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.jpg} (IR has extra
  'infrared/' prefix, both .jpg); test/{infrared/Undistorted_SfM/
  <scene>/images/<stem>.png, Undistorted_SfM/<scene>/images/<stem>
  .jpg} (IR=.png, VIS=.jpg, different ext).
  v10 only uses train/. scene-disjoint 175/10/10 split via
  MyScripts/build_megadepth_syn_index.py (seed=42, hard ValueError
  if test/val < 2000 imgs or train < 80K imgs to catch high-variance
  scene size 19..3975). Server: 2 sub-symlinks (train/{infrared,
  phoenix}) + 2 _pc real dirs (train/{infrared_pc, phoenix_pc},
  M3FD-style source/pc peer-level layout). Cfg picks up via
  configs/data/megadepth_syn_trainval.py with deep ROAD_*_SUBDIR =
  'train/infrared/phoenix/S6/zl548/MegaDepth_v1' so index file rows
  are short '<scene>/<denseN>/imgs/<stem>.jpg' (M3FD-style).
  PC cache L1+L2+L3 acceleration via precompute_pc_edges.py
  --recursive --max_long_edge 640 --pc_nscale 3 (~14h -> ~46 min).
  Post-precompute df-truncation may break cache aspect mismatch:
  use MyScripts/fix_pc_cache_alignment.py to inplace-resize cache to
  raw's _resize_keep_aspect target shape (~30 min, idempotent).
  Use when adding Megadepth_Syn to repo / running v10 / 接入
  Megadepth_Syn / 数据集软链 / scene-disjoint 划分 / 平衡检查 /
  PC 缓存 L1+L2+L3 / fix_pc_cache_alignment / df 截断破坏 cache 对齐 /
  build_megadepth_syn_index 平衡 ValueError 提示换 seed.
  Triggers: Megadepth_Syn / megadepth_syn / 风格迁移合成 IR /
  scene-disjoint 175/10/10 / build_megadepth_syn_index / 平衡检查
  raise ValueError / fix_pc_cache_alignment / max_long_edge df 截断 /
  6 个 VIS 空目录 / Undistorted_SfM 0015 0022 / MegaDepth1500 兼容布局 /
  M3FD 风格 source pc 同级 / train/infrared/phoenix 双重 phoenix 前缀,
  English 'add Megadepth_Syn dataset', 'scene-disjoint split',
  'aligned IR-VIS pixel-perfect', 'L1 L2 L3 PC acceleration',
  'fix df-truncated cache shape', 'M3FD-style symlink layout for
  Megadepth_Syn', 'asymmetric train vs test path layout'.
  Per-version implementation: eloftr-v10-msyn (training cfg/run/sanity
  gates). Companion data skills: eloftr-m3fd-data (LWIR street scenes,
  similar M3FD-style symlink approach), eloftr-roadscene-data.
  Server boundary / softlink guards: eloftr-yurupeng-workspace §4.
---

# Megadepth_Syn 跨模态合成数据集接入（v10 专用）

> 前置依赖：[eloftr-roadscene-data](../eloftr-roadscene-data/SKILL.md)（数据集 I/O 类、A3 padding、Homography 监督、白名单机制）+ [eloftr-m3fd-data](../eloftr-m3fd-data/SKILL.md)（aligned IR-VIS 第二个一等公民的接入模板）。
> 服务器边界守卫与软链命令通用规则见 [eloftr-yurupeng-workspace §4](../eloftr-yurupeng-workspace/SKILL.md)。
> v10 训练 cfg / DDP / PC 加速 / fix script 设计动机见 [eloftr-v10-msyn](../eloftr-v10-msyn/SKILL.md)。
> 评估侧 apples-to-apples 对比方法见 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)。

## 0. 为什么要加 Megadepth_Syn（背景）

v9 用 M3FD 4200 对真红外街景训完拿到双向 SOTA（in-domain p@1 0.6863 / OOD p@1 0.2128）。但是：

1. **数据规模仍是小数据 finetune**——M3FD 3.78K train 跟 MegaDepth 原版 ~14M pair 相比小 4 个量级
2. **从未在 4 卡 DDP 上跑过 aligned IR-VIS 训练**——[eloftr-server-multigpu §4.3](../eloftr-server-multigpu/SKILL.md) RandomConcatSampler 不分片地雷在 v0-v9 单卡时没暴露，需要在大数据集上验证
3. **真 IR 数据集稀缺**——M3FD 4200 对 + RoadScene 220 对就已经是公开 IR-VIS 数据集 top tier；要再扩数据只能走合成路线

Megadepth_Syn 解决以上 3 点：

| 子集 | 对数 | 来源 | 用途 |
|---|---|---|---|
| **train (v10 用)** | **~129K** | MegaDepth 旅游照 + neural style transfer 合成"伪 IR" | v10 主训练 |
| test (v10 不用) | 806 | 同上, MegaDepth1500 兼容 (Undistorted_SfM 0015 + 0022) | v11+ B 路线 真 epipolar pose AUC eval 预备 |

核心特点：**(IR_i, VIS_i) pixel-perfect 对齐**（IR 是 VIS 风格迁移产物，同图不同模态）→ Homography aug 后得到 cross-modal + synthetic-cross-view 训练样本，跟 M3FD 监督契约完全一致（详见 [eloftr-v10-msyn §3](../eloftr-v10-msyn/SKILL.md)）。

## 1. 数据集事实速查

来源：导师下发的合成数据集，存放在服务器 `/data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/`。**不需要下载**，直接软链到本地即可。

### 1.1 顶层结构

```text
/data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/
├── train/                                 ← v10 主训练源
│   ├── infrared/phoenix/S6/zl548/MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.jpg
│   ├── phoenix/S6/zl548/MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.jpg
│   ├── phoenix/S6/zl548/MegaDepth_v1/<scene>/<denseN>/depths/<stem>.h5  ← 深度（v10 不用）
│   ├── depth/, event/, normal/, paint/, sketch/                          ← 陪跑模态（v10 不用）
└── test/                                  ← MegaDepth1500 兼容布局（v10 不用）
    ├── infrared/Undistorted_SfM/0015/images/*.png
    ├── infrared/Undistorted_SfM/0022/images/*.png
    ├── Undistorted_SfM/0015/images/*.jpg
    └── Undistorted_SfM/0022/images/*.jpg
```

总占用 **509 GB**；v10 实际只用 train/{infrared, phoenix}，**~222 GB**（IR ~19 GB + VIS+depth ~203 GB）。其它模态（depth/event/normal/paint/sketch）目录存在但 v10 不软链不读。

### 1.2 训练数据规模

| 项 | 值 |
|---|---|
| train scene 数（MegaDepth_v1 风格）| **195** |
| train IR/VIS pair 数 | **128,982** （IR 与 VIS 完美一一配对）|
| dense 子目录总数 | IR=269，VIS=275（VIS 多 6 个**空**目录，**安全忽略**） |
| 每 scene 图数分布 | min=19, max=3975, **median=473, mean=661**（高方差！）|
| 单图分辨率 | 1280×~1000 长边（MegaDepth 原版 aspect 多变）|
| 文件扩展名 | **train 两边都 `.jpg`**（与 test IR=.png 不一致）|

**6 个 VIS 空目录**（IR 端也没有，对训练完全不影响，索引脚本会自动跳过 + warn）：
```
0092/dense0, 0265/dense1, 0327/dense2, 0360/dense1, 0360/dense2, 0394/dense1
```

### 1.3 测试数据规模（v10 不用，留 v11+）

| scene | IR 张数 | VIS 张数 | 兼容 |
|---|---|---|---|
| `0015` | 328 (.png) | 328 (.jpg) | MegaDepth1500 标准评测 |
| `0022` | 478 (.png) | 478 (.jpg) | 同上 |
| **合计** | **806 (.png)** | **806 (.jpg)** | — |

> ⚠️ **test 跟 train 扩展名不一致**（IR=.png / VIS=.jpg, 同 stem）+ **物理路径不一致**（test 走 `Undistorted_SfM/`, train 走 `phoenix/S6/zl548/MegaDepth_v1/`）。v10 通过"只用 train + scene-disjoint 自划分"完全规避此不对称。如果将来 v11+ 走 B 路线（真 epipolar），需要新建 `MegaDepthSynDataset` + 第二份 cfg 处理 test 扩展名差异。

### 1.4 路径不对称（接入时关键）

| 模态 | train 路径 | test 路径 |
|---|---|---|
| IR | `train/infrared/phoenix/S6/zl548/MegaDepth_v1/...` | `test/infrared/Undistorted_SfM/<scene>/images/*.png` |
| VIS | `train/phoenix/S6/zl548/MegaDepth_v1/...` | `test/Undistorted_SfM/<scene>/images/*.jpg` |

**train 端 IR 比 VIS 多一段 `infrared/` 前缀**——这导致 cfg 里 `ROAD_IR_SUBDIR / ROAD_VIS_SUBDIR` 必须**深路径吸收**：
```python
ROAD_IR_SUBDIR  = "train/infrared/phoenix/S6/zl548/MegaDepth_v1"
ROAD_VIS_SUBDIR = "train/phoenix/S6/zl548/MegaDepth_v1"
```
让 index 文件每行只存 `<scene>/<denseN>/imgs/<stem>.jpg`（M3FD 风格的简短嵌套相对路径）。

## 2. v10 划分策略：scene-disjoint 175/10/10（vs M3FD 90/5/5 random sample）

### 2.1 为什么 scene-disjoint 而不是 image-random

| 方案 | 优势 | 劣势 |
|---|---|---|
| Image-random（M3FD 90/5/5）| 简单 | val/test 跟 train **共享 scene**，模型可能见过同 scene 的"邻居视角"，OOD 性弱 |
| **Scene-disjoint 175/10/10**（v10 选定）| val/test 来自完全没见过的 scene → 真 in-domain test 上界 | scene 大小高方差（min 19 / max 3975）→ 抽到"偏小 scene"会让 val/test 太小不稳，需要平衡检查 |

scene-disjoint 让 v10 测的是"v10 ckpt 在没见过的 MegaDepth 旅游 scene 上的泛化"，是**比 in-domain 弱、比真 OOD 强**的中间水平。真 OOD（M3FD/RoadScene 真红外）由 §6.3 单独 eval。

### 2.2 划分参数（plan 选定 + 实测验证）

| split | scene 数 | 实测图数 | 占比 |
|---|---|---|---|
| train | 175 | **106,083**（实测）| 89.7% |
| val | 10 | **12,101**（实测，比预估 ~6.6K 大 2×, 因为 seed=42 抽到偏大 scene）| 5.1% |
| test | 10 | ~10,711（推算）| 5.1% |
| **合计** | **195** | **~129K** | 100% |

随机种子 `--seed 42` 固定。如果将来想换 seed 复现实验，记得**同时 re-precompute PC cache 不需要**（PC 是 per-image，跟 scene 划分无关），但 fix_pc_cache_alignment 不需要重跑。

## 3. 索引脚本：`build_megadepth_syn_index.py`

源码：[MyScripts/build_megadepth_syn_index.py](../../../MyScripts/build_megadepth_syn_index.py)。

### 3.1 默认行为

```bash
python MyScripts/build_megadepth_syn_index.py
# 等价于
python MyScripts/build_megadepth_syn_index.py \
    --root data/Megadepth_Syn \
    --ir_subdir  train/infrared/phoenix/S6/zl548/MegaDepth_v1 \
    --vis_subdir train/phoenix/S6/zl548/MegaDepth_v1 \
    --out_subdir index \
    --ext .jpg \
    --train_scene_count 175 \
    --val_scene_count 10 \
    --test_scene_count 10 \
    --seed 42 \
    --min_test_imgs 2000 \
    --min_train_imgs 80000 \
    --min_val_test_ratio 0.3
```

输出 `data/Megadepth_Syn/index/{train,val,test}_pairs.txt`，每行一个**嵌套相对路径带 .jpg**（与 m3fd_splits 行格式同款，dataset 加载时 `osp.join(ir_dir, name)` 直接拼）：

```text
0000/dense0/imgs/1000564847_9a99654012_o.jpg
0000/dense0/imgs/1000570923_c2a177031b_o.jpg
...
```

### 3.2 输入扫描策略

- 扫描 `<root>/<ir_subdir>/<scene>/<denseN>/imgs/*.jpg` 作为 IR 列表（按 IR 列表为准，VIS 端自动配对检查）
- 对每个 IR 文件检查 VIS 同位文件存在（`<root>/<vis_subdir>/<scene>/<denseN>/imgs/<stem>.jpg`），不存在则跳过 + warn（处理 6 个 VIS 空目录）
- **不扫 test/Undistorted_SfM**（v10 不用）

### 3.3 Scene-disjoint 划分逻辑

```python
all_scenes = sorted(scene_to_relpaths.keys())   # 195 个
scenes = list(all_scenes); rng = random.Random(seed); rng.shuffle(scenes)
train = sorted(scenes[:175])
val   = sorted(scenes[175:185])
test  = sorted(scenes[185:195])
# 每个 scene 内的所有 (dense*, imgs/*.jpg) pair 都进对应 split
```

### 3.4 平衡检查（hard ValueError，必修复机制）

由于 scene 大小方差极大（19..3975），random 抽 10 个进 test 极端时可能只 200 张。脚本在生成 list 之后**强制检查图数**：

| 检查 | 阈值 | 失败处置 |
|---|---|---|
| `test_imgs >= --min_test_imgs` | 默认 2000 | `raise ValueError` 列出 test scene + 最小 scene 大小 + 建议换 seed |
| `val_imgs >= --min_test_imgs` | 默认 2000 | 同上, val 端 |
| `train_imgs >= --min_train_imgs` | 默认 80000 | 同上, train 端 |
| `min(val_imgs, test_imgs) / max(...) >= --min_val_test_ratio` | 默认 0.3 | val/test 严重不平衡 |

**实测 seed=42 通过所有检查**：train 106083 / val 12101 / test ~10711 / val-test ratio ~0.88，远超阈值。

如果失败：换 `--seed 7 / 13 / 99 / 2024 / 1234` 重跑。

### 3.5 Sanity 输出（启动后必看）

```text
Megadepth_Syn index built (scene-disjoint, seed=42):
  train scenes : 175 (e.g. 0000, 0001, 0003, ..., 5023)  total imgs: 106083
  val   scenes :  10 (e.g. 0042, 0089, 0156, ...)         total imgs:  12101
  test  scenes :  10 (e.g. 0067, 0123, 0345, ...)         total imgs:  10711
  smallest scene contribution:
    train: 0349 -> 25 imgs
    val  : 0156 -> 88 imgs
    test : 0067 -> 156 imgs
  largest scene contribution:
    train: 0007 -> 3975 imgs
    val  : 5004 -> 1234 imgs
    test : 0123 -> 2345 imgs
  empty VIS dirs warned: 0092/dense0, 0265/dense1, ... (skipped)
  balance check:
    train >= 80000:  PASS (106083)
    val   >=  2000:  PASS (12101)
    test  >=  2000:  PASS (10711)
    val/test ratio >= 0.3:  PASS (0.88)
  -> SUCCESS, written to data/Megadepth_Syn/index/{train,val,test}_pairs.txt
```

## 4. 数据 cfg：`configs/data/megadepth_syn_trainval.py`

源码：[configs/data/megadepth_syn_trainval.py](../../../configs/data/megadepth_syn_trainval.py)。**完全照抄 m3fd_trainval.py 模式**，三个 mode 共享同一个 ROAD_IR_SUBDIR：

```python
MSYN_PATH = "data/Megadepth_Syn"

cfg.DATASET.TRAINVAL_DATA_SOURCE = "Megadepth_Syn"          # 注册在 ALIGNED_IRVIS_SOURCES
cfg.DATASET.TEST_DATA_SOURCE     = "Megadepth_Syn"

# 三组 LIST_PATH 全部指向 data/Megadepth_Syn/index/{train,val,test}_pairs.txt
cfg.DATASET.TRAIN_DATA_ROOT  = MSYN_PATH
cfg.DATASET.TRAIN_LIST_PATH  = f"{MSYN_PATH}/index/train_pairs.txt"
cfg.DATASET.VAL_DATA_ROOT    = MSYN_PATH
cfg.DATASET.VAL_LIST_PATH    = f"{MSYN_PATH}/index/val_pairs.txt"
cfg.DATASET.TEST_DATA_ROOT   = MSYN_PATH
cfg.DATASET.TEST_LIST_PATH   = f"{MSYN_PATH}/index/test_pairs.txt"

# 深路径吸收 IR/VIS 不对称（"infrared/" 前缀只在 IR 端）
cfg.DATASET.ROAD_IR_SUBDIR    = "train/infrared/phoenix/S6/zl548/MegaDepth_v1"
cfg.DATASET.ROAD_VIS_SUBDIR   = "train/phoenix/S6/zl548/MegaDepth_v1"
cfg.DATASET.ROAD_IR_PC_SUBDIR  = "train/infrared_pc/S6/zl548/MegaDepth_v1"   # M3FD 风格 source/_pc 同级
cfg.DATASET.ROAD_VIS_PC_SUBDIR = "train/phoenix_pc/S6/zl548/MegaDepth_v1"

cfg.DATASET.ROAD_IMG_RESIZE  = 480     # 与 v9 持平，apples-to-apples
cfg.DATASET.ROAD_DF          = 32
cfg.DATASET.ROAD_PAD_SIZE    = 480
cfg.DATASET.ROAD_HOMOGRAPHY_AUG  = True
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0

cfg.TRAINER.ENABLE_PLOTTING = False
```

### 4.1 白名单 1 行加入

[src/utils/data_source.py](../../../src/utils/data_source.py)：
```python
ALIGNED_IRVIS_SOURCES = frozenset({"roadscene", "m3fd", "megadepth_syn"})  # 加 megadepth_syn
```

加完之后所有 5 处 dispatch（[eloftr-roadscene-data §2.5](../eloftr-roadscene-data/SKILL.md)）自动走 IR-VIS 路径，跟 M3FD 完全对称。

## 5. 服务器侧软链：M3FD 风格 source/_pc 同级布局

按 [eloftr-yurupeng-workspace §4.1](../eloftr-yurupeng-workspace/SKILL.md) 子目录软链方案 + M3FD 风格命名：

```bash
cd /home/xyjiang/Desktop/yurupeng/eloftr/data
mkdir -p Megadepth_Syn/{train,index}
cd Megadepth_Syn
mkdir -p train/{infrared_pc,phoenix_pc}     # 实体 PC 缓存目录
ln -s /data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/train/infrared  train/infrared
ln -s /data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/train/phoenix   train/phoenix
cd ..

# 验证
ls Megadepth_Syn/train                       # 期望: infrared (软链) phoenix (软链) infrared_pc (实体) phoenix_pc (实体)
ls -L Megadepth_Syn/train/infrared/phoenix/S6/zl548/MegaDepth_v1 | head -3   # 期望: 0000  0001  0003 ...
```

最终目录结构：

```text
data/Megadepth_Syn/                                ← 实体可写目录
├── train/                                         ← 实体可写目录（mkdir）
│   ├── infrared       → /data/.../train/infrared    (软链, 只读源)
│   ├── infrared_pc/                                 ← 实体, PC 缓存（mkdir, 镜像 source 嵌套结构）
│   │   └── S6/zl548/MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.png
│   ├── phoenix        → /data/.../train/phoenix     (软链, 只读源)
│   └── phoenix_pc/                                  ← 实体, PC 缓存
│       └── S6/zl548/MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.png
└── index/                                          ← 实体, build_megadepth_syn_index.py 写入
    ├── train_pairs.txt   ~106083 行
    ├── val_pairs.txt     ~12101 行
    └── test_pairs.txt    ~10711 行
```

跟 M3FD 的 `data/M3FD_Detection/{Ir, Ir_pc, Vis, Vis_pc}` 在 `train/` 父目录下完全对称。**test/Undistorted_SfM 不软链**（v10 不用）。

## 6. PC 缓存 L1+L2+L3 三层加速

`v0-v9` PC 缓存默认 ~14h（M3FD 35-50 min × 数据量 ~30×）。v10 通过三层独立优化压到 **~46 min**：

| 层 | 改动 | 加速比 | 默认值 | v10 用 |
|---|---|---|---|---|
| **L1 pyfftw** | `pip install pyfftw` 让 phasepack 走 pyfftw 而非 scipy fftpack | ~1.3× | （依装机环境）| 装上 |
| **L2 nscale 4→3** | `--pc_nscale 3` CLI（v0-v9 默认仍 4 字节兼容）| ~2× | 4 | 3 |
| **L3 预 resize 640** | `--max_long_edge 640` 把源图预降到 640 长边再算 PC（默认 0 = v0-v9 兼容）| ~7× | 0 | 640 |
| **综合** | — | **~18×** | — | 三层全开 |

### 6.1 启动 PC 预计算

```bash
source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
pip install pyfftw    # L1, 一次性
python MyScripts/precompute_pc_edges.py --dataset Megadepth_Syn \
       --recursive --max_long_edge 640 --pc_nscale 3 --workers 24
# tmux 内跑, 实测 ~46 min
```

`--recursive` 让脚本递归扫描嵌套目录 + 镜像输出（PC 缓存目录结构跟 source 完全对称）。`DATASET_SHORTCUTS` 已在 [precompute_pc_edges.py L107-138](../../../MyScripts/precompute_pc_edges.py) 加 `Megadepth_Syn` 项。

### 6.2 ⚠️ L3 副作用：df 截断破坏 cache aspect

`--max_long_edge 640 --df 32` 联合作用时，对源图做 resize 会单边砍像素：
```
raw (1090, 696) → scale=640/1090=0.587 → new_h=640, new_w=409
new_w 经 df=32 对齐 → 384  ← 砍掉 25 像素！
cache aspect = 640/384 = 1.6667 ≠ raw 1.5661 (差 6.4%)
```

部分 raw aspect（如 1.45）会触发 dataset 加载时 cache 跟 raw 在 `_resize_keep_aspect(_, 480, 32)` 后 **target shape 不一致**，`np.stack([ir_pad, ir_pc_pad])` 失败崩。

→ **必须跑 fix script 修复**（详见 §7）。

## 7. fix_pc_cache_alignment.py（PC df 截断救场）

源码：[MyScripts/fix_pc_cache_alignment.py](../../../MyScripts/fix_pc_cache_alignment.py)。**一次性 patch 脚本**，一次跑完一劳永逸。

### 7.1 核心思路

把所有 cache 强 resize 到"raw 在 dataset 端 `_resize_keep_aspect(_, 480, 32)` 的 target shape"。fix 后：
- cache 已是 dataset 训练目标分辨率（如 (480, 288)）
- dataset L383 `cv2.resize(ir_pc_raw, (w0_r, h0_r))` 是 1.0× no-op
- pad 到 (480, 480) + stack 必然 shape 一致

物理对齐数学验证：每个 (480, 288) tensor 像素对应同一片 raw 物理区域（不论走 raw→直接 还是 raw→cache(640,384)→(480,288)），复合 scale 完全相等。详见 [eloftr-v10-msyn §4.3](../eloftr-v10-msyn/SKILL.md)。

### 7.2 启动 fix

```bash
# dry-run 先看（~15 min, 24 worker, IR + VIS 两路）
python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn --dry-run --workers 24
# 期望输出 fixed: ~257790 (= 128895 IR + 128895 VIS, 100% 都需要修)

# 实跑（~30 min）
python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn --workers 24
# inplace 写回, idempotent (已修过的 cache 是 ok-noop)
```

CLI 参数：
- `--dataset Megadepth_Syn` 走预设 shortcut（IR + VIS 两路自动跑）
- `--workers 24` 24 个 multiprocessing worker
- `--dry-run` 不写回，只统计有多少需要修
- `--img_resize 480 --df 32` 默认值，应跟 dataset cfg 的 `ROAD_IMG_RESIZE / ROAD_DF` 一致

### 7.3 配套 dataset 端 target-shape check

[src/datasets/roadscene.py L353-404](../../../src/datasets/roadscene.py) 的 PC cache shape check 已升级（`_resize_target_shape` helper 与 fix script 共用逻辑）：

```python
target_raw_ir = _resize_target_shape(ir_raw.shape, self.img_resize, self.df)
target_pc_ir = _resize_target_shape(ir_pc_raw.shape, self.img_resize, self.df)
if target_raw_ir != target_pc_ir:
    raise RuntimeError("... Run MyScripts/fix_pc_cache_alignment.py first.")
```

兼容性：
- v0-v9 (M3FD/RoadScene): cache 跟 raw 同 res → `_target_shape` 返回相同 → check pass
- v10 fix 后: cache 已是 dataset target shape → check pass（dataset L383 是 no-op）
- v10 漏 fix: target 不一致 → 报清晰错误提示跑 fix script

### 7.4 速度

| 步骤 | 耗时 | 备注 |
|---|---|---|
| dry-run（IR + VIS 两路）| ~15 min | 第一路 11 min（cold cache）, 第二路 4 min（OS page cache 命中）|
| 实跑（含 cv2.imwrite）| ~30 min | 257K 文件 inplace 写回 |
| **总** | **~45 min** | vs 重跑 precompute ~46 min, **省 ~16 min**（且不损失 PC quality）|

## 8. 加新对齐 IR-VIS 数据集的样板（如果未来要加 LLVIP / KAIST）

按"M3FD = aligned IR-VIS 第二个一等公民" + "Megadepth_Syn = 第三个" 的标准 4 步：

1. **数据落地**：`data/<NAME>/` 实体目录 + 内部子目录软链到只读源（[eloftr-yurupeng-workspace §4.1](../eloftr-yurupeng-workspace/SKILL.md)）
2. **白名单加名**：[src/utils/data_source.py](../../../src/utils/data_source.py) `ALIGNED_IRVIS_SOURCES` 加上 `"<name>"`（小写）
3. **数据 config**：复制 [configs/data/megadepth_syn_trainval.py](../../../configs/data/megadepth_syn_trainval.py)（如果数据集嵌套结构跟 Megadepth_Syn 类似）或 [m3fd_trainval.py](../../../configs/data/m3fd_trainval.py)（如果是扁平单层 Ir/Vis）→ `configs/data/<name>_trainval.py`，改 4 个字段
4. **索引脚本**：复制 `make_m3fd_splits.py`（image-random）或 `build_megadepth_syn_index.py`（scene-disjoint + 平衡检查）→ `<name>` 适配版

可选：
- 如果数据集分辨率超 480（如 LLVIP 1080×720）→ 在 cfg 提 `ROAD_IMG_RESIZE` 到 640/832
- 如果走 PC + CLAHE → 在 `precompute_pc_edges.py` 的 `DATASET_SHORTCUTS` 加项 + 跑 PC cache（如果 max_long_edge 触发 df 截断 → 跑 fix_pc_cache_alignment）

**严禁**：任何地方再写 `== 'roadscene'` 或 `== 'm3fd'` 或 `== 'megadepth_syn'` 的字面量比较。所有 dispatch 必须走 `is_aligned_irvis(...)`。

## 9. 常见错误 → 定位指南

| 错误现象 | 大概率原因 | 修复入口 |
|---|---|---|
| 启动日志没出现 `building RoadSceneDataset (dataset_name=Megadepth_Syn)` | `TRAINVAL_DATA_SOURCE` 拼写不对 / 没在 `ALIGNED_IRVIS_SOURCES` 里 | 检查 [src/utils/data_source.py](../../../src/utils/data_source.py) §4.1 |
| `FileNotFoundError: ...train/infrared/phoenix/...` | 软链失效或路径写错 | `ls -L data/Megadepth_Syn/train/infrared/phoenix/S6/zl548/MegaDepth_v1` 验证 |
| `RuntimeError: PC cache shape (480, 640) resizes to (480, 288) but raw IR shape (1450, 1000) resizes to (480, 320)` | PC 缓存被 L3 df 截断破坏 aspect | 跑 `python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn --workers 24` |
| `ValueError: test split has only 1234 images (< 2000)` | scene-disjoint 抽到偏小 scene | 换 `--seed 7/13/99/...` 重跑 `build_megadepth_syn_index.py` |
| build_megadepth_syn_index.py warn 出 6 个 VIS 空目录 | 数据集本身就有这 6 个空（已知）| 安全忽略，IR 端也对应没有，已在 sanity 输出列出 |
| 4 卡 DDP debug log 不出现 `aligned IR-VIS train pre-shard: 26521/106083` | image-level pre-shard 未触发 | 检查 [src/lightning/data.py L248-273](../../../src/lightning/data.py) get_local_split 路径; mode A 单卡走 fallback (`world_size=1` 不分片) 是预期 |
| Megadepth_Syn 训练 loss 起步就 < 1.0 | **正常** | Megadepth_Syn IR 是 VIS 风格迁移产物, 跟 outdoor.ckpt 训练分布更近, 比 M3FD 真红外 task 容易, 详见 [eloftr-v10-msyn §1 H1 假设](../eloftr-v10-msyn/SKILL.md) |
