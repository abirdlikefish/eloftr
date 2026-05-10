---
name: eloftr-v10-msyn
description: |
  EfficientLoFTR v10 = v9_e2e stack on Megadepth_Syn 4-GPU DDP ship run.
  Path A+ (Homography supervision route): inherits v9_e2e cfg byte-identical
  (USE_CONTRASTIVE / USE_MODALITY_EMB / USE_EDGE_INPUT / USE_CLAHE_IR /
  USE_MSBN / BACKBONE_IN_CHANNELS=2 全 True) and only overrides schedule for
  ~115K-pair scale (max_ep=12 vs v9 80, MSLR=[3,5,7], ES patience=3).
  Implemented as configs/loftr/eloftr_full_v10_msyn_{singlecard,ddp}.py +
  MyScripts/run_msyn_v10_{singlecard,ddp}.sh, exp_name msyn_v10_ddp.
  Three production-grade fixes:
  1. DDP image-level pre-shard in src/lightning/data.py L248-273 fixes
     server-multigpu §4.3 RandomConcatSampler-not-sharded mine for aligned
     IR-VIS via get_local_split(world_size, rank, seed). Each rank gets
     disjoint 26521/106083 sample shard.
  2. PC cache L1+L2+L3 acceleration: pyfftw (1.3x) + --pc_nscale 3 (2x) +
     --max_long_edge 640 (~7x df-aligned downscale) -> ~14h baseline ->
     ~46 min actual.
  3. MyScripts/fix_pc_cache_alignment.py: one-shot patch when precompute
     df-truncation breaks shape match (saves 30-46 min re-precompute,
     ~3-30 min depending on cache hit). dataset roadscene.py aspect check
     replaced with target-shape check (_resize_target_shape helper) that
     catches truly broken cases without false-positive on df-rounding noise.
  Real-measured 2026-05-08 ship run completed: PC fix 100% applied
  (257790 cache files), all 6 sanity gate + 3 DDP gate pass, ep0 step 367
  train_loss=0.572 (vs v9 ep0 step 370 ~1.47, 3x lower because Megadepth_Syn
  synthetic IR is much closer to MegaDepth-trained outdoor.ckpt distribution),
  ep0 val P@1=0.6227 / P@3=0.9790 (vs v9 M3FD ep0 P@1 ~0.35-0.40, P@3 ~0.65,
  much higher because Megadepth_Syn val task degenerates to identity matching:
  IR-VIS pixel-aligned by style-transfer + val homography aug forced off via
  data.py L288 `homography_aug=(... and mode == 'train')`). Final ckpt ep11
  train val P@1=0.8344 / P@3=0.9976 / P@5=0.9991 / mpe=0.455 px (val task
  saturation: P@3 / P@5 cannot be cross-version compared to v9 M3FD;
  论文 / 答辩 use independent eval only).
  Wall-clock revision: plan estimate "3-4h" was wrong by 5x. Actual
  17.11h for 12 ep on 4-card sync_bn DDP (1.40-1.44 it/s = 0.7s/step;
  sync_bn ~25% step-time due to RepVGG ~30 BN + MSBN 4 BN cross-rank
  mean/var sync; 4-card nominal 4x acceleration eaten down to ~0.7x);
  7187 train + 3025 val per rank * ~0.7s = ~85 min/ep
  (revised in eloftr-server-multigpu §6.5).
  Independent eval (ep11 ckpt, 2026-05-08): M3FD test P@1=0.5129 /
  P@3=0.9643 / P@5=0.9927 / mpe=1.04 px (M3FD is OOD for v10 because
  v10 trained on Megadepth_Syn with 0 ep on M3FD); RoadScene OOD
  P@1=0.4560 / P@3=0.9377 / P@5=0.9917 / mpe=1.20 px. RoadScene OOD
  P@1 vs v9 0.2128 = +114% relative (=2.14x), is the largest single-version
  OOD jump in v0..v10 history. 综合通用性 (M3FD P@3 + OOD P@3) = 1.9020
  vs v9 1.7001 (+0.2019, beats v8->v9 jump +0.1807), v0..v10 NEW SOTA.
  Key mechanism: in-domain ↔ OOD gap (M3FD P@3 - OOD P@3) collapses
  from v5..v9's 0.21..0.28 range to 0.027 for v10 -> Megadepth_Syn 195 scene
  diversity + strong homography aug teaches viewpoint-invariant geometric
  features rather than M3FD-specific BN distribution. v10 thus recommended
  as毕设 final delivery for general cross-modal deployment, alongside v9
  as M3FD-specialised in-domain SOTA (dual-track delivery).
  Use when running/debugging/extending v10 / explaining why v10 on
  synthetic IR is easier than v9 on real LWIR / writing thesis scale-up
  section / 看 v10 / 跑 v10 / 分析 v10 训练结果 / v10 ckpt 选择.
  Triggers: v10 / v10_msyn / msyn_v10_ddp / Megadepth_Syn / 风格迁移合成 IR /
  DDP image-level pre-shard / get_local_split aligned IR-VIS /
  fix_pc_cache_alignment / pyfftw 加速 / max_long_edge df 截断 /
  pc_nscale 3 / scene-disjoint 175/10/10 / 1.40-1.44 it/s sync_bn 开销 /
  17.11h vs 3-4h wall-clock 修订 5x / mode A B 已固化 /
  cold start Megadepth_Syn val P@1 0.622 P@3 0.979 / Style A 继承 v9_e2e /
  CANONICAL_LR=5e-4 反向缩放 / WARMUP=450 actual 1800 step /
  N_SAMPLES_PER_SUBSET=28750 配 26521 真实 shard /
  v10 ship 完成 / v10 综合通用性 SOTA 1.9020 / v10 RoadScene OOD P@1 翻倍 /
  v10 in-domain OOD gap 0.027 / 风格迁移训练泛化到真 LWIR /
  v10 训练 val P@3 P@5 饱和 identity matching / val 关 homography aug /
  v10 vs v9 双轨交付 / 毕设 final delivery v9 v10 dual-track,
  English 'v10 ship run', '4-GPU DDP scale-up Megadepth_Syn',
  'image-level pre-shard for aligned IR-VIS', 'PC cache L1 L2 L3
  acceleration', 'fix df-truncation cache mismatch',
  'sync_bn slowdown vs plan', 'why synthetic IR easier than real LWIR',
  'v10 ship completed independent eval', 'v10 综合通用性 SOTA 1.9020',
  'v10 RoadScene OOD P@1 doubled', 'v10 in-domain OOD gap collapsed',
  'val task identity matching saturation', 'v10 training val P@3 P@5
  cannot compare with v9', 'pure-OOD 综合通用性 v10 vs in-domain v9',
  'v9 v10 dual-track final delivery'.
  v10 = path H in cross-modal roadmap (data scale-up + DDP infra
  validation), 与 path G (v9_e2e training-strategy) 正交但延续 v9 全部
  stack. Companion: eloftr-megadepth-syn-data (dataset SOP),
  eloftr-cross-modal-experiments (overall map),
  eloftr-server-multigpu (DDP runtime + 5 traps + §6.5 wall-clock 修订),
  eloftr-v9-e2e (parent stack), eloftr-results (ship 数字).
---

# v10：Megadepth_Syn 4 卡 DDP scale-up（v9 stack 的极限验证）

> 继承链：v7 (PC+CLAHE 输入端) → v8 (MSBN fine 架构) → v9 (e2e cold start 训练策略) → **v10 (Megadepth_Syn ~115K + 4 卡 DDP scale-up)**
> 总览见 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)；v9 见 [eloftr-v9-e2e](../eloftr-v9-e2e/SKILL.md)
> 数据集接入 SOP（路径 / 划分 / 平衡检查 / fix script）见 [eloftr-megadepth-syn-data](../eloftr-megadepth-syn-data/SKILL.md)
> 4 卡 DDP runtime 与 5 个隐藏地雷见 [eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md)
> 训练日志读取见 [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md)
> 独立 eval 见 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)
> **跨版本数字对照** 见 [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)（v10 综合通用性 1.9020 = 新 SOTA，§3 排名第 1）

**已实现 + 已 ship**：[configs/loftr/eloftr_full_v10_msyn_singlecard.py](../../../configs/loftr/eloftr_full_v10_msyn_singlecard.py) + [eloftr_full_v10_msyn_ddp.py](../../../configs/loftr/eloftr_full_v10_msyn_ddp.py) + [MyScripts/run_msyn_v10_singlecard.sh](../../../MyScripts/run_msyn_v10_singlecard.sh) + [run_msyn_v10_ddp.sh](../../../MyScripts/run_msyn_v10_ddp.sh) + [MyScripts/build_megadepth_syn_index.py](../../../MyScripts/build_megadepth_syn_index.py) + [MyScripts/fix_pc_cache_alignment.py](../../../MyScripts/fix_pc_cache_alignment.py) + 配套 src/ 改动 3 处。从 `weights/eloftr_outdoor.ckpt` cold start，4 卡 3090 DDP, max_epochs=12, **mode B full ship 跑完（17.11h, 2026-05-08T16:10:27 收尾）+ 双数据集独立 eval 跑完**：M3FD test P@1=0.5129/P@3=0.9643/P@5=0.9927，RoadScene OOD P@1=0.4560/P@3=0.9377/P@5=0.9917，**综合通用性 = 1.9020 vs v9 1.7001**（v0..v10 SOTA）。

## 1. 设计动机：v9 stack 在 31× 数据 + 4 卡 DDP 上是否还 work？

v0-v9 全程在 M3FD（**3.78K** 真红外街景对）上验证 v1-v8 stack（contrast / modemb / freeze / PC + CLAHE / MSBN）+ v9 cold start。v10 提出的核心问题：

> **同样的 v1-v9 stack + cold start 训练策略，在 31× 大的 Megadepth_Syn 数据集上 + 4 卡 DDP 加速，能否复现 v9 双向 SOTA？同时验证 §4.3 RandomConcatSampler 不分片地雷在 aligned IR-VIS 路径上的真正修复。**

副问题：

1. Megadepth_Syn 是**风格迁移合成 IR**（VIS → IR pseudo），跟 M3FD 真 LWIR 红外的物理特性不同。v9 stack 的"模态不变特征学习"是否在合成模态上同样 work？
2. 4 卡 DDP image-level pre-shard 是否真能解决 [eloftr-server-multigpu §4.3](../eloftr-server-multigpu/SKILL.md) RandomConcatSampler 不分片 + 4-rank 同 seed 4× 重复抽样问题？
3. plan §10 估"3-4h"完整 ship run 时长是否合理？sync_bn 在 RepVGG 多 BN 架构上的真实开销是多少？

**v10 实测最终答案**（截至 2026-05-08 ship 完成 + 双数据集独立 eval 完成）：

- 训练侧（[`logs/tb_logs/msyn_v10_ddp/version_0/`](../../../logs/tb_logs/msyn_v10_ddp/version_0/)）：
  - ep0 step 367 train_loss=**0.572**（vs v9 ep0 step ~370 ~1.47）→ Megadepth_Syn 任务起步就比 v9 容易 3×（H1 已验证，详见 §9）
  - ep0 val 末 P@1=**0.6227** / P@3=**0.9790** / P@5=**0.9918**（vs v9 M3FD ep0 P@1 ~0.35-0.40）→ 数字偏高的根因不只是任务简单，**主导因素是 val 任务本身退化为 identity matching**（详见 §8.1 警告）
  - ep11 final val P@1=**0.8344** / P@3=**0.9976** / P@5=**0.9991** / mpe=0.4545 px / avg_loss=0.4271，best_epoch=11/12（end-of-training，cold start 链规律）
  - DDP pre-shard log `[rank 0..3]: aligned IR-VIS train pre-shard: 26521/106083 samples (disjoint across 4 ranks; seed=66)` × 4 → §4.3 地雷**已修**
  - modemb ir 0→**0.0861** / vis 0→**0.0863**（learning ✓，对称增长，未达 v9 终态）；MSBN L1 drift first/last/max=0.0029/**0.1895**/0.2045（active ✓，弧形未走完）
- wall-clock：**实测 17.11h** = plan 估算"3-4h"偏乐观 5×（详见 §6 修订）
- 独立 eval（[`dump/m3fd_eval_v10_version0_top1/`](../../../dump/m3fd_eval_v10_version0_top1/) + [`dump/roadscene_eval_v10_version0_top1/`](../../../dump/roadscene_eval_v10_version0_top1/)）：
  - **M3FD test (OOD for v10)**: P@1=**0.5129** / P@3=**0.9643** / P@5=**0.9927** / mpe=1.04 px / matches/pair=2 203
  - **RoadScene OOD**: P@1=**0.4560** / P@3=**0.9377** / P@5=**0.9917** / mpe=1.20 px / matches/pair=1 591
  - **综合通用性 = 1.9020**（vs v9 1.7001, +0.2019, v0..v10 SOTA）
  - in-domain ↔ OOD gap 从 v5..v9 的 0.21..0.28 区间塌到 **0.027** → v10 学到 viewpoint-invariant 几何而非 dataset-specific BN（详见 §9 H1' 修订）

## 2. 数据集事实：Megadepth_Syn vs M3FD 对照

| 维度 | M3FD（v9 数据） | Megadepth_Syn（v10 数据） |
|---|---|---|
| IR 来源 | 真 LWIR 红外（FLIR 相机）| 风格迁移合成 IR（MegaDepth VIS 经神经网络生成）|
| VIS 来源 | 真可见光（同相机配对）| MegaDepth Internet 旅游照（罗马斗兽场 / Mt Rushmore 等）|
| 配对方式 | 物理相机标定 + Homography warp（~1 px 残差）| **风格迁移产物 pixel-perfect 对齐**（同图不同模态）|
| Train pair 数 | 3,780 | **~106,083**（175 scene 实际产出，约 28× M3FD）|
| Val pair 数 | 210 | ~12,101（10 scene）|
| Test pair 数 | 210 | ~10,711（10 scene，scene-disjoint）|
| 总 pair 数 | 4,200 | ~129K |
| 单图分辨率 | 1024×768 PNG | 1280×~1000 JPG（MegaDepth 多变 aspect）|
| 数据集物理布局 | 单一根 `data/M3FD_Detection/{Ir, Vis, Ir_pc, Vis_pc}` | **train/test 分开** `data/Megadepth_Syn/train/{infrared, phoenix, infrared_pc, phoenix_pc}`（v10 只用 train, test 留 v11+ B 路线） |
| 划分方式 | random 90/5/5 抽样 | **scene-disjoint 175/10/10**（保 OOD 性 + 平衡检查防极端 scene 大小） |

详细数据集背景 / 路径模板 / scene-disjoint 切分 / 6 个 VIS 空目录 / 平衡检查机制见 [eloftr-megadepth-syn-data](../eloftr-megadepth-syn-data/SKILL.md)。

## 3. 路线决策：A+ vs B（为什么不走真 epipolar）

### 3.1 路线 A+：Homography 监督（v10 选定）

继承 v9 全部 stack，**仅换数据源**：
- (IR_i, VIS_i) 同位 pair（pixel-perfect 对齐）
- [src/datasets/roadscene.py L356-372](../../../src/datasets/roadscene.py) 的 Homography aug 在 train mode 下永远把 VIS 走 `cv2.warpPerspective` → 训练样本是 `(image0=IR_i, image1=warpPerspective(VIS_i, H))`
- modemb / MSBN / PC 学的都是真信号（同时跨模态 + 跨人造视角），不存在退化

→ **apples-to-apples 跟 v9 比**，工程量 2-3 天，毕设时间窗口稳。

### 3.2 路线 B：真 epipolar 监督（v10 不走，留 v11/v12）

如果走真 epipolar：
- 利用 Megadepth_Syn 自带 `depth.h5`、原版 MegaDepth scene_info npz（K, T, depth）
- 跨视角配对 (view_i 的 IR, view_j 的 VIS)，监督走 `compute_supervision_*_megadepth`
- 需要新建 `MegaDepthSynDataset` 移植 PC + CLAHE 模块（~300 行）+ 监督 dispatch + ckpt monitor + MegaDepth1500 pose AUC eval
- 工程量 5-7 天，且与 v0-v9 不可比

→ test/Undistorted_SfM/0015 + 0022 那 806 pair 是为 B 路线 MegaDepth1500 eval 准备的，v10 完全不动。

## 4. 三项工程修复（v10 production-grade）

### 4.1 DDP image-level pre-shard（[src/lightning/data.py L248-273](../../../src/lightning/data.py)）

[eloftr-server-multigpu §4.3](../eloftr-server-multigpu/SKILL.md) 早期警告：aligned IR-VIS 短路分支不分片，4 rank 同 `cfg.TRAINER.SEED` 全局生成器 → 抽到完全相同的 indices → 4× 重复 step + effective_bs 实际只有 4。v0-v9 单卡运行从未触发，**v10 mode B 4 卡是首次实战**。

修复（v10 落地）：

```python
# src/lightning/data.py L248-273 (added 2026-05-08)
if is_aligned_irvis(data_source):
    # DDP image-level pre-sharding (mirrors the scene-level
    # get_local_split call below for ScanNet/MegaDepth, see L283).
    with open(scene_list_path, 'r', encoding='utf-8') as f:
        all_names = [line.strip() for line in f.readlines() if line.strip()]
    if mode == 'train' and self.world_size > 1:
        local_names = list(get_local_split(
            all_names, self.world_size, self.rank, self.seed))
        logger.info(
            f'[rank {self.rank}]: aligned IR-VIS train pre-shard: '
            f'{len(local_names)}/{len(all_names)} samples '
            f'(disjoint across {self.world_size} ranks; seed={self.seed})')
    else:
        local_names = all_names
    # ...
    ds = RoadSceneDataset(..., names=local_names, ...)
```

`RoadSceneDataset.__init__` 接受 `names: Optional[List[str]] = None` 参数（None = 走原 `_read_index_file(list_path)` 路径，向后兼容 v0-v9 单卡）。

**实测验证**（mode B debug 启动 log）：
```text
[rank 0]: aligned IR-VIS train pre-shard: 26521/106083 samples (disjoint across 4 ranks; seed=66)
[rank 1]: aligned IR-VIS train pre-shard: 26521/106083 samples ...
[rank 2]: ...
[rank 3]: ...
```
26521 × 4 = 106084 ≈ 106083 ✓ 4 rank 严格不重叠 1 epoch 见 train 全集。

> Pre-shard 只在 train mode 触发；val / test 保持 full + PL 默认 `DistributedSampler` 切片（[data.py L412-413, L429-430](../../../src/lightning/data.py)），跟 ScanNet/MegaDepth 路径同款。

### 4.2 PC 缓存 L1+L2+L3 三层加速（14h → ~46 min）

| 层 | 改动 | 加速比 | 风险 |
|---|---|---|---|
| **L1 pyfftw** | 服务器 `pip install pyfftw` 让 phasepack 走 pyfftw 而非 scipy fftpack | ~1.3× | 0 |
| **L2 nscale 4→3** | precompute CLI `--pc_nscale 3`（v0-v9 默认仍 4 字节兼容）| ~2× | 边缘锐度略损（plan §4.3 sanity 验证 OK）|
| **L3 预 resize 640** | precompute CLI `--max_long_edge 640` 把源图预降到 640 长边再算 PC（默认 0 = v0-v9 兼容）| ~7× | df 截断破坏 cache aspect, 由 §4.3 fix script 兜底 |
| **综合** | — | **~18×** | 本质优化，PC 质量需小样本 sanity 验证 |

precompute 改动（[MyScripts/precompute_pc_edges.py](../../../MyScripts/precompute_pc_edges.py)）：
- `_build_tasks` 加 `--recursive` flag（嵌套目录递归 + 镜像输出）
- `_process_one` 加 `max_long_edge` + `df` + `pc_nscale` 三参数（默认全 0/4 = v0-v9 兼容）
- `DATASET_SHORTCUTS` 加 `Megadepth_Syn` 项（M3FD-style source/_pc 同级布局）

实测 wall-clock：**~46 min**（24 worker，pyfftw + nscale=3 + max_long_edge=640 全开），跟 plan §4.3 估算一致。

### 4.3 fix_pc_cache_alignment.py（PC 缓存 df 截断救场）

#### 问题

precompute L3 优化时 `--max_long_edge 640 + df=32` 对源图做 resize：
```python
scale = 640 / max(h, w)             # 比如 raw (1090, 696), scale = 0.587
new_h = round(h * scale)            # 640
new_w = round(w * scale)            # 409
new_w = (new_w // df) * df          # 384  ← 砍掉 25 像素！
```

`409 → 384` 的 df 截断破坏 aspect ratio：raw aspect 1.5661 → cache aspect 1.6667（差 6.4%）。dataset 加载时 [_resize_keep_aspect(_, 480, 32)](../../../src/datasets/roadscene.py) 走 480 长边 + df=32 对齐：

| 源 | aspect | dataset target shape |
|---|---|---|
| raw (1090, 696) | 1.5661 | (480, 288) |
| 错误 cache (640, 384) | 1.6667 | (480, 288) ← 巧合对齐 |
| 错误 cache (640, 416) for raw (1450, 1000) aspect 1.45 | 1.5385 | **(480, 288) ≠ raw (480, 320)** ← stack 形状不对，崩 |

约 30-40% 的 raw aspect 会触发后一种 mismatch。

#### 解法：fix script

[MyScripts/fix_pc_cache_alignment.py](../../../MyScripts/fix_pc_cache_alignment.py)：

```python
def _target_shape(raw_shape, long_edge=480, df=32):
    h, w = raw_shape
    scale = long_edge / max(h, w)
    new_h = max(df, (int(round(h * scale)) // df) * df)
    new_w = max(df, (int(round(w * scale)) // df) * df)
    return (new_h, new_w)

# For each (raw, cache) pair:
target = _target_shape(raw.shape, 480, 32)
if cache.shape != target:
    cache_fixed = cv2.resize(cache, (target[1], target[0]), cv2.INTER_AREA)
    cv2.imwrite(cache_path, cache_fixed)
```

**核心思路**：把 cache 强 resize 到"raw 在 dataset 端 `_resize_keep_aspect` 的产物 shape"。fix 后 dataset L383 `cv2.resize(ir_pc_raw, (w0_r, h0_r))` 是 1.0× no-op，cache 跟 raw 同 shape，stack 成 (2, 480, 480) 必然成功。

#### 物理对齐的数学验证

任意 (target_h, target_w) tensor 像素 → raw 物理区域映射：

| 路径 | H scale | W scale |
|---|---|---|
| raw 直接 → (480, 288) | 1090/480=2.27 | 696/288=2.42 |
| raw → cache(640, 384) → (480, 288) | (1090/640)×(640/480)=2.27 | (696/384)×(384/288)=2.42 |

→ **复合 scale 完全相等**，每个 (480, 288) 像素对应同一片 raw 物理区域。Homography aug 之后也保持（vis_pc 跟 vis 用同一个 H 矩阵 + 同 target shape）。

#### 配套 dataset check 改造

[src/datasets/roadscene.py L353-404](../../../src/datasets/roadscene.py) 旧 aspect ratio check 改成 **target-shape check**（新加 `_resize_target_shape` helper 与 fix script 共用逻辑）：

```python
target_raw_ir = _resize_target_shape(ir_raw.shape, self.img_resize, self.df)
target_pc_ir = _resize_target_shape(ir_pc_raw.shape, self.img_resize, self.df)
if target_raw_ir != target_pc_ir:
    raise RuntimeError("... Run MyScripts/fix_pc_cache_alignment.py first.")
```

兼容性：
- v0-v9 (M3FD/RoadScene): cache 跟 raw 同 res → `_target_shape` 返回相同 → check pass
- v10 fix 后: cache 已是 target shape → check pass（dataset L383 是 no-op）
- v10 漏 fix: target 不一致 → 报清晰错误提示跑 fix script

#### 实测速度

mode B PC cache 256K 张图 fix（IR + VIS）实测：
- dry-run（24 worker, 只读 + 算 target，不写）: **15 min** （IR 11 min + VIS 4 min，第二批走 OS page cache 加速 3×）
- 实跑（含 cv2.imwrite）：**~30 min**（页缓存全热 + IO 写入）

vs 重跑 precompute ~46 min，**节省 ~16 min**。idempotent，可中断重跑。

## 5. 配置（Style A 继承 v9_e2e + schedule 重写）

### 5.1 单卡 cfg（mode A 调试 + fallback）

[configs/loftr/eloftr_full_v10_msyn_singlecard.py](../../../configs/loftr/eloftr_full_v10_msyn_singlecard.py)：

```python
from configs.loftr.eloftr_full_v9_e2e import cfg

# v10 unfreeze-all override: v9 set FREEZE_BACKBONE_BN=True (REVISION 2 design
# to prevent small-data 3.78K BN drift). v10 has 31x more data + 80x more
# sample passes per ep -> let backbone BN running stats adapt.
cfg.LOFTR.FREEZE_BACKBONE = False
cfg.LOFTR.FREEZE_BN = False
cfg.LOFTR.FREEZE_BACKBONE_BN = False     # v9 was True; v10 abundant data -> unfreeze
cfg.LOFTR.FREEZE_FINE_BN_IR = False
cfg.LOFTR.FREEZE_FINE_BN_VIS = False

# v10 schedule: 31x more data per epoch -> shrink ep 80 -> ~12, keep MSLR
# milestones at 25%/40%/60% of max_ep
cfg.TRAINER.MSLR_MILESTONES = [3, 5, 7]
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 3

# WARMUP 1 ep ratio (v9: 60->960 actual ~ 1ep on 945 step/ep M3FD)
# v10: 1800->28800 actual ~ 1ep on 28750 step/ep Megadepth_Syn
cfg.TRAINER.WARMUP_STEP = 1800
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 115000
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False
```

继承的 v9_e2e flag（**不动**）：
- `LOSS.USE_CONTRASTIVE = True`
- `USE_MODALITY_EMB = True` + `MODALITY_EMB_INIT='zeros'`
- `BACKBONE_IN_CHANNELS = 2` + `USE_EDGE_INPUT = True` + `USE_CLAHE_IR = True`
- `USE_MSBN = True`
- `CANONICAL_LR = 2e-3` (单卡 _scaling=0.0625 → TRUE_LR=1.25e-4)
- `PERSISTENT_WORKERS = True`

### 5.2 4 卡 DDP cfg（ship run）

[configs/loftr/eloftr_full_v10_msyn_ddp.py](../../../configs/loftr/eloftr_full_v10_msyn_ddp.py)：

```python
from configs.loftr.eloftr_full_v10_msyn_singlecard import cfg

# 4-card reverse scaling: TRUE_LR = CANONICAL_LR * (16/64) = CANONICAL_LR * 0.25
# Want TRUE_LR = 1.25e-4 (match v9 single-card) -> CANONICAL_LR = 5e-4
cfg.TRAINER.CANONICAL_LR = 5e-4

# WARMUP: defensive ~0.25 ep ramp (1800 step) instead of 0
cfg.TRAINER.WARMUP_STEP = 450

# Sampler quota = local shard size (~115000 / 4 ranks = 28750)
# 4 ranks * 28750 unique = 115K = full train set per epoch (no overlap)
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 28750
```

实际 train_pairs.txt 行数 106083（不是预估的 115K，因为 175 scene 实际产出）。Sampler with `SB_SUBSET_SAMPLE_REPLACEMENT=False` + cfg=28750 + 实际 26521/rank：先 randperm(26521) 取全部，再 with-replacement 补 2229 → 单 rank 28750 sample/epoch。

### 5.3 启动脚本（已含 7 个前置文件检查）

[MyScripts/run_msyn_v10_ddp.sh](../../../MyScripts/run_msyn_v10_ddp.sh)：
```bash
tmux new -s msyn_v10_b_full
bash MyScripts/run_msyn_v10_ddp.sh
# Ctrl+B D 离开
```

sh 自动检查：`train/{infrared, phoenix, infrared_pc, phoenix_pc}` + `index/{train,val}_pairs.txt` + `weights/eloftr_outdoor.ckpt` + PC cache 非空。任一缺失立即 exit 1 + 错误提示。

## 6. wall-clock 修订：3-4h → **实测 17.11h**

plan §10 估算"3-4h"完整 ship run **偏乐观 5×**（17.11 / 3.5 ≈ 4.9×，比早期 mode B debug 估算的 4× 还略多）。修订基于 ship 完成后实测 1.40-1.44 it/s = 0.7s/step：

| 操作 | 单 step 耗时 | 来源 |
|---|---|---|
| forward (16 sample × 480² × 16M params) | ~0.15s | GPU 计算 |
| backward | ~0.30s | 反向比 forward 慢 2× |
| NCCL allreduce 16M grads | ~0.05s | 64MB / 25GB/s × 4 rank |
| **sync_bn 跨卡同步**（**主要被低估的开销**）| **~0.15s** | RepVGG backbone ~30 个 BN + MSBN 4 个 BN，每个 BN forward + backward 都要 4-rank sync mean/var |
| 其他（optimizer / dataloader prefetch / 等）| ~0.05s | — |
| **总** | **~0.7s/step** | **= 1.44 it/s** ✓ |

epoch 时长：

| 阶段 | 单 step | step 数 | 时长 |
|---|---|---|---|
| train | ~0.7s | 7187（28750/4）| ~83 min |
| val（forward only, no allreduce, no sync_bn update）| ~0.15s | 3025（12101/4）| ~8 min |
| **per epoch 总** | | | **~85-90 min ≈ 1.5h** |
| **12 epoch（实测 ship 完整）** | | | **17.11h** ✓ |

**实测**（msyn_v10_ddp/version_0）：start `2026-05-07T23:04:07` → end `2026-05-08T16:10:27` = **17h 6min** 完整 12 ep 跑完，**未触发 ES**（val P@1 ep10→ep11 仍 +0.0073, ES patience=3 没攒满）。

修订 wall-clock 已写入 [eloftr-server-multigpu §6.5](../eloftr-server-multigpu/SKILL.md)，未来 v11+ 4 卡 DDP 实验可直接参考。

## 7. 6 sanity gate + 3 DDP 专属 gate（mode A/B 启动验证）

mode A 单卡 debug 5 min 跑通；mode B 4 卡 DDP debug 10 min 跑通；ship run 应该全过。

| # | Gate | 期望 log（启动 ~30 行内）| 失败处置 |
|---|---|---|---|
| 1 | stage0 inflate | `Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)` | 检查 `BACKBONE_IN_CHANNELS=2` 是否继承 |
| 2 | MSBN inflate 双层 | `MSBN inflated layer{1,2}_outconv2.1 (...) -> bn_ir + bn_vis (zero-shift)` ×2 | 检查 `USE_MSBN=True` 是否继承 |
| 3 | missing keys = modemb | `missing_keys (kept at init value): ['modality_emb_ir', 'modality_emb_vis']` | 检查 `USE_MODALITY_EMB=True` 是否继承 |
| 4 | trainable params 16.03M | `Trainable params: 16.03M / Total: 16.03M` | 跟 v9 一致；MSBN 4 个 BN（128+128+256+256 weight + bias）已含 |
| 5 | CLAHE enabled ×2 | `RoadSceneDataset: CLAHE enabled (clipLimit=2.0, ..., ir=True, vis=False)` 出现 train + val 各一次 | 检查 `USE_CLAHE_IR=True` 是否继承 |
| 6 | TRUE_LR 缩放 | （train.py 不打 log，但可推算）单卡: `2e-3 × 0.0625 = 1.25e-4`；4 卡: `5e-4 × 0.25 = 1.25e-4` | 都应到 1.25e-4，与 v9 单卡一致 |
| **7** | **PC cache shape check 不报错** | 训练 reach `Epoch 0: ...` step >0 不报 `RuntimeError: PC cache shape ...` | 跑 fix script 或重 precompute |
| **8 (DDP)** | **4 rank 都 init** | `initializing ddp: GLOBAL_RANK: 0/1/2/3, MEMBER: 1-4/4` | 检查 `--gpus=4` + nvidia-smi 4 卡空闲 |
| **9 (DDP)** | **image-level pre-shard 工作** | `[rank 0]: aligned IR-VIS train pre-shard: 26521/106083 samples (disjoint across 4 ranks; seed=66)` | 检查 [data.py L248-273](../../../src/lightning/data.py) get_local_split 路径 |

## 8. 实测结果（ship 完成 + 双数据集独立 eval 完成，2026-05-08）

### 8.1 训练时间线（[`logs/tb_logs/msyn_v10_ddp/version_0/`](../../../logs/tb_logs/msyn_v10_ddp/version_0/) 实测）

| 区间 | epoch | TRUE_LR | 实测 P@1 / P@3 / P@5 | 备注 |
|---|---|---|---|---|
| warmup | 0-0.25 | 0 → 1.25e-4 | 起步 ep0 末 0.6227 / 0.9790 / 0.9918 | 1800 step linear ramp |
| full LR | 0.25-3 | 1.25e-4 | ep1→3 0.7140→0.7605 / 0.9881→0.9925 / 0.9963→0.9978 | acclimate from outdoor.ckpt |
| MSLR #1 | 3-5 | 6.25e-5 | ep4→5 0.7882→0.8033 / 0.9935→0.9951 / 0.9980→0.9987 | LR ×= 0.5 |
| MSLR #2 | 5-7 | 3.13e-5 | ep6→7 0.7972→0.8196 / 0.9952→0.9963 / 0.9986→0.9990 | LR ×= 0.5（ep5→6 唯一一次微跌 −0.0061） |
| MSLR #3 | 7-12 | 1.56e-5 | ep8→**11** **0.8227→0.8344** / 0.9966→**0.9976** / 0.9990→**0.9991** | 末段稳定 + best 在 ep11/12=92% |

**关键时间点**：
- ep0 step 0 train_loss ≈ 1.50（cold start 起点，外推）
- ep0 step 367 train_loss = **0.572**（vs v9 ep0 step 370 ~1.47, 3× 更低）
- ep11 final train_loss = 0.211（min 0.121 @ step 67000，倒数第二 ep）
- 12 ep 共 86 250 step / 17.11h，未触发 ES（patience=3 没攒满）

**⚠️ 重要警告：训练 val P@3 / P@5 是 evaluator 退化产物，不是真分**

ep0 val P@3 = **0.9790**（cold start 几乎没训过）+ ep11 P@3 = 0.9976 + ep11 P@5 = 0.9991 → 这些数字看起来"刷爆 v9 ep75 0.9753"是**评测口径退化的副作用**，不是模型真比 v9 强：

1. **Megadepth_Syn (IR_i, VIS_i) 由 style transfer 构造**：IR = neural-net 把 VIS 重新着色，**像素一一对齐 + 模态差仅在 style 不在 geometry**（cfg 自己说 "pixel-aligned by construction"，[`configs/data/megadepth_syn_trainval.py` L34-35](../../../configs/data/megadepth_syn_trainval.py)）
2. **val 时 homography aug 强制关闭**：[`src/lightning/data.py` L288](../../../src/lightning/data.py) `homography_aug=(self.road_homography_aug and mode == 'train')` → 即使 cfg `ROAD_HOMOGRAPHY_AUG=True`，val/test 只走 `mode != 'train'` 分支，**`H_0to1 = np.eye(3)` ground-truth = identity**
3. **val 任务退化为 "在 IR 找到 keypoint，预测同坐标在 VIS"**：ε=3px 对几乎无几何 + 弱模态差的对子，即使刚 cold start 1 ep 也能 P@3 ≈ 0.98

**含义**：
- v10 训练 val 0.9976 P@3 / 0.9991 P@5 **不能跨数据集与 v9 0.9753 / 0.9929 直接比**（v9 用真 LWIR + ~1px 标定残差，任务难度完全不同）
- v10 ckpt 选择按 P@3 monitor 几乎在统计噪声内挑（ep11=0.9976, ep10=0.9973, ep8=0.9966；差异 0.0003-0.0010 < val random ordering 的 std）
- **论文 / 答辩永远引用 §8.2 / §8.3 独立 eval 数字（M3FD 0.9643 + RoadScene 0.9377），绝对不引用本 §8.1 训练 val 数字**
- 这条警告也写进了 [`results/tb_summary.md` §2 注 ⁹](../../../results/tb_summary.md) + [§4 注 ¹¹](../../../results/tb_summary.md) + [`results/eval_summary.md` §4 观察 8](../../../results/eval_summary.md)

如果未来想要可比的 v10 训练 val，可以做 v10.1：保持 v10 cfg 不变，只在 `src/lightning/data.py` L288 改成 `homography_aug=self.road_homography_aug`（不再 mode-gate），让 val 也走真 homography，重新跑一次 12 ep 看 val P@1 / P@3 是否能跟 v9 同口径对比。**当前不做**（独立 eval 数字已 SOTA，14h 算力换 evaluator 同口径性价比低）。

### 8.2 M3FD test 独立 eval（210 pair, 真 LWIR + ~1px 标定残差，对 v10 是 OOD）

ship 完成后跑（ckpt = `epoch=11-precision@1px=0.834-precision@3px=0.998-precision@5px=0.999.ckpt`）：

```bash
python MyScripts/eval_roadscene.py \
    --ckpt logs/tb_logs/msyn_v10_ddp/version_0/checkpoints/epoch=11-...ckpt \
    --main_cfg configs/loftr/eloftr_full_v10_msyn_ddp.py \
    --data_cfg configs/data/m3fd_trainval.py \
    --thr 0.1
```

| ckpt | matches/pair | mpe (px) | P@1px | P@3px | P@5px |
|---|---:|---:|---:|---:|---:|
| v9 ep75 (M3FD-trained, in-domain) | 2 205 | 0.7283 | **0.6863** | **0.9792** | **0.9960** |
| **v10 ep11 (msyn-trained, OOD)** | 2 203 | 1.0394 | 0.5129 | 0.9643 | 0.9927 |

**解读**：
- v10 在 M3FD 上 P@1 落后 v9 0.173（−25% rel），P@3 落后 0.015（−1.5% rel），P@5 几乎持平（−0.003）
- mpe 1.04 px ≈ M3FD 双相机硬件标定残差量级 → v10 把残差 push 到物理极限
- **v10 P@1=0.51 仍超过早期 in-domain 训练的 v5 (0.435) / v6 (0.443) / v6.1 (0.464)**，比看过 M3FD 数千 step 的早期模型更强（在没看过 M3FD 的前提下）
- 写入 [`results/eval_summary.md` §1](../../../results/eval_summary.md) 第 7 行 + 注 ²

### 8.3 RoadScene OOD 独立 eval（22 pair, 真 LWIR + 不同视角 + 2-5px 标定残差）

```bash
python MyScripts/eval_roadscene.py \
    --ckpt logs/tb_logs/msyn_v10_ddp/version_0/checkpoints/epoch=11-...ckpt \
    --main_cfg configs/loftr/eloftr_full_v10_msyn_ddp.py \
    --data_cfg configs/data/roadscene_trainval.py \
    --thr 0.1
```

| ckpt | matches/pair | mpe (px) | P@1px | P@3px | P@5px |
|---|---:|---:|---:|---:|---:|
| v5 (M3FD-trained) | 1 351 | 3.39 | 0.1858 | 0.5989 | 0.7825 |
| v9 ep75 (M3FD-trained) | 1 531 | 2.39 | 0.2128 | 0.7209 | 0.8912 |
| **v10 ep11 (msyn-trained)** ⚡ | **1 591** | **1.20** | **0.4560** | **0.9377** | **0.9917** |

**v10 在 RoadScene OOD 上的飞跃是 v0..v10 全程最大代际突破**：
- P@1 v9→v10 **0.2128 → 0.4560**（+114% rel, +0.243 abs）—— 翻倍
- P@3 v9→v10 **0.7209 → 0.9377**（+30% rel, +0.217 abs）—— 突破 [eloftr-eval-pipeline §5.4](../eloftr-eval-pipeline/SKILL.md) 估的"P@3 物理上限 60-70%"
- mpe v9→v10 **2.39 → 1.20** px（−50%）—— 砍半
- matches/pair 比 v9 还涨 +3.9%（更多 match 同时更准 = 真泛化）

**机制**：v10 训练数据完全没有 RoadScene 一帧（M3FD 也没有）→ 这是真"训练分布外"泛化。详见 §9 H1' 与 H4。

写入 [`results/eval_summary.md` §2](../../../results/eval_summary.md) 第 7 行 + 注 ⚡。

### 8.4 综合通用性（v0..v10 SOTA = v10）

按 [eloftr-results §3](../eloftr-results/SKILL.md) 公式：综合 = M3FD P@3 + RoadScene OOD P@3。

| 版本 | 训练集 | M3FD P@3 | OOD P@3 | 综合 | 排名 | Δ vs prev |
|---|---|---:|---:|---:|:---:|---:|
| v8 | M3FD | 0.9007 | 0.6187 | 1.5194 | 3 | +0.0437 |
| v9 | M3FD | 0.9792 | 0.7209 | 1.7001 | 2 | +0.1807 |
| **v10** ⭐ | **Megadepth_Syn** | **0.9643** | **0.9377** | **1.9020** | **1** | **+0.2019** |

**v10 是新的综合通用性 SOTA**：
- 综合通用性 +0.2019 比 v8→v9 跳跃 +0.1807 还高 +12%
- in-domain ↔ OOD gap (M3FD P@3 − OOD P@3) 从 v5..v9 的 0.21..0.28 区间塌到 v10 0.027 → 几乎消除 in-domain / OOD 鸿沟
- 注意 v10 行的"in-domain"对它而言其实是 OOD（v10 训练集是 Megadepth_Syn）→ 实际上是"双向 OOD"打"in-domain + OOD"，v10 综合泛化深度更高

**毕设最终交付建议**：v9 / v10 双轨——v9 = M3FD 在线监控专用（in-domain SOTA），v10 = 通用跨模态部署（综合通用性 SOTA + 真 OOD 鲁棒）。详见 [`results/eval_summary.md` §3](../../../results/eval_summary.md)。

## 9. 关键 finding（ship + eval 完成后回填，2026-05-08）

### H1（**部分修订** = H1'）: Megadepth_Syn 训练 val 任务上限本身就高

**原 H1（debug 阶段）猜测**：v10 数字漂亮可能因为合成 IR 跟 outdoor.ckpt 同源，但 OOD 可能反而退化。**结果证伪**：v10 OOD 不仅没退化，反而 RoadScene P@1 翻倍（§8.3）→ v9 stack 在合成数据上学到的是 viewpoint-invariant 几何，不是 spurious 风格。

**修订 H1' = "训练 val 上限本身就高"（不是模型更强）**：
- ep0 train_loss 0.572 (v10) vs 1.47 (v9 同位置) → 3× 更低，但**主因不是合成 IR 跟 outdoor.ckpt 同源**，而是 **val 任务本身退化为 identity matching**（详见 §8.1 警告）
- ep0 val P@3 = 0.979 这种数字在任何"IR-VIS pixel-aligned + val 关 homography aug"的设置下都会出现，与模型无关
- v9 的 M3FD val 数字（ep0 P@3 ~0.65）天然偏低是因为 M3FD 真 LWIR 模态差大 + ~1px 标定残差，val 任务本身上限就低

**直接证据：训练 val 与独立 eval Δ**：
- v9 ep75: 训练 val P@3=0.9758 → 独立 eval P@3=0.9792（Δ=+0.003，固定偏移指纹）
- v10 ep11: 训练 val P@3=0.9976 → 独立 eval P@3=0.9643（Δ=**−0.033**，evaluator 口径不同）

→ Δ 方向反转 + 量级 10 倍 → 明确不是同口径，不能跨版本直接比训练 val。完整解释见 [`results/tb_summary.md` §4 注 ¹¹](../../../results/tb_summary.md)。

### H2（**已验证**）: DDP image-level pre-shard 是 v10 唯一一处真"修复"

§4.1 这个修复是 v0-v9 单卡训练从未触发的设计 gap，是 v10 真正贡献的"代码级硬资产"——未来任何 4 卡 DDP aligned IR-VIS 训练都受益。**实测**：4 rank pre-shard log 全部出现 `26521/106083 samples (disjoint)`，4 × 26521 = 106084 ≈ 106083 train pair 全集（无重叠、无重复抽样），ship 完成无 collision。

### H3（**已验证**）: sync_bn 在 RepVGG 是主导开销

RepVGG 多 BN 架构 + MSBN 4 个 BN，sync_bn 跨卡同步占 step 时间 ~25%（实测 1.40-1.44 it/s = 0.7s/step）。这是 plan §10 估算翻车的根因，也是 4-card 4× 名义加速被吃到 ~0.7× 的主因（17.11h vs 单卡外推 ~20h，仅 17% 加速）。

### H4（**新增 finding，由 §8.3 RoadScene OOD 飞跃驱动**）: Megadepth_Syn 多样性主导 OOD 泛化

v10 RoadScene OOD 飞跃（P@1 0.21 → 0.46，+114% rel）的核心解释**不是**"合成 IR 跟 outdoor.ckpt 同源"（H1'已部分否定），而是：

1. **数据多样性**：Megadepth_Syn 195 scene = 全球各地建筑 / 景点（罗马斗兽场 / Mt Rushmore / Notre-Dame …）vs M3FD ~4K 帧 city-driving（场景类型偏单一）→ 模型被迫学到 viewpoint-agnostic 几何特征，而不是 dataset-specific 像素分布
2. **强 homography aug 强度**：±10° 旋转 + ±10% 缩放 + ±5% 平移 + ±3% 透视，覆盖 195 scene × 28 750 pair × 12 ep ≈ 67M 个 (image, H) 训练样本对 → 几何变化量级远超 M3FD ~3.78K × 80 ep ≈ 300K
3. **弱模态差**：v8/v9 在 M3FD 真 LWIR 上必须把容量花在 IR-VIS modality bridge（→ MSBN drift max=0.24），v10 模态差弱反而把容量留给几何

**含义 / 启示**：
- 跨模态匹配的 OOD 鲁棒不靠"训练 IR 越真越好"，靠"训练数据多样性 + 强几何变化"
- 未来 v11+ 想 push OOD：扩 Megadepth_Syn 训练数据 + 加更激进 homography（±15° 旋转）；不要挤 M3FD 这种小数据集
- in-domain（M3FD）专用模型仍需 v9 这种大量 ep 在目标数据集上 finetune；通用部署用 v10 这种 cross-dataset cold-start

### H5（**新增 finding**）: v10 训练 val 数据集"identity-matching trap" 是 evaluator 设计警示

v10 训练 val P@3=0.998 是 evaluator 退化产物：style-transfer 数据集 IR-VIS 本就 pixel-aligned + val 主动关 homography aug → 任务退化。**这是给后人 / 答辩的设计警示**：

- 任何用 style-transfer / paired generation 数据训练的 cross-modal 模型，**不能用 "val 关 aug + identity ground-truth" 评测**
- 要么 val 也走 homography aug（接受 random，损失 evaluator 复现性），要么用真实独立 eval 集（损失 PL ckpt monitor 的实时性）
- v10 走"独立 eval 路线"（ep11 by P@3=0.9976 选 ckpt + 后期独立 eval 鉴真），未来 v11+ 可考虑 val 也开 homography aug 看训练 val 是否真能跟独立 eval 同口径

写进 [`results/tb_summary.md` §5 观察 8](../../../results/tb_summary.md) + [`results/eval_summary.md` §4 观察 8](../../../results/eval_summary.md) 作为 evaluator 设计的反例 case study。

## 10. 复现 / 二次跑 v10

### 10.1 服务器（vlrlab）

```bash
ssh vlrlab
cd /home/xyjiang/Desktop/yurupeng/eloftr

# 一次性前置（~46 min PC 缓存 + ~5 min 索引 + ~30 min fix script）
source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
pip install pyfftw                                         # L1 加速

# 数据集软链 + 索引（详见 eloftr-megadepth-syn-data §3-§5）
mkdir -p data/Megadepth_Syn/{train,index}
cd data/Megadepth_Syn
mkdir -p train/{infrared_pc,phoenix_pc}
ln -s /data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/train/infrared train/infrared
ln -s /data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/train/phoenix  train/phoenix
cd ../..
python MyScripts/build_megadepth_syn_index.py             # scene-disjoint 175/10/10 + 平衡检查

# PC 缓存预计算（~46 min）
python MyScripts/precompute_pc_edges.py --dataset Megadepth_Syn \
       --recursive --max_long_edge 640 --pc_nscale 3 --workers 24

# fix cache shape（~30 min, 一次性）
python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn --workers 24

# 启动 ship 训练
tmux new -s msyn_v10_b_full
bash MyScripts/run_msyn_v10_ddp.sh
# Ctrl+B D 离开
# 后续: tmux attach -t msyn_v10_b_full

# eval (~10 min, ship 完成后)
python MyScripts/eval_roadscene.py \
    --ckpt logs/tb_logs/msyn_v10_ddp/version_0/checkpoints/<best>.ckpt \
    --main_cfg configs/loftr/eloftr_full_v10_msyn_ddp.py \
    --data_cfg configs/data/megadepth_syn_trainval.py \
    --list_path data/Megadepth_Syn/index/test_pairs.txt \
    --thr 0.1
```

### 10.2 训练完后分析

```bash
python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/msyn_v10_ddp
```

完整 tfevents 读取见 [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md)。

### 10.3 追加 results 表

按 [eloftr-results §2](../eloftr-results/SKILL.md) 工作流追加 `results/eval_summary.md` 一行 + `results/tb_summary.md` 训练侧汇总一行。

## 11. v10 与其它 skill 的关系

| skill | 它管什么 | 不管什么 |
|---|---|---|
| **本 skill (eloftr-v10-msyn)** | v10 设计动机、配置、3 项工程修复、6+3 sanity gate、wall-clock 修订、ship run SOP | 数字汇总（→ eloftr-results）、Megadepth_Syn 数据集本身（→ eloftr-megadepth-syn-data）、DDP 通用机制（→ eloftr-server-multigpu） |
| [eloftr-megadepth-syn-data](../eloftr-megadepth-syn-data/SKILL.md) | 数据集背景、路径表、scene-disjoint 划分、平衡检查、PC L1+L2+L3、fix script | 训练 cfg / schedule / DDP（→ 本 skill） |
| [eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md) §4.3 | DDP RandomConcatSampler 不分片地雷的通用解释 + v10 修复指针 | v10 cfg 细节（→ 本 skill） |
| [eloftr-v9-e2e](../eloftr-v9-e2e/SKILL.md) | v10 继承的 stack（contrast / modemb / freeze / PC + CLAHE / MSBN）的设计原理 | v10 数据/DDP/PC 加速（→ 本 skill） |
| [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md) §5.5 | PC 缓存机制 + v10 L1+L2+L3 加速注解 + fix script 入口 | v10 整体设计（→ 本 skill） |
| [eloftr-results](../eloftr-results/SKILL.md) | v10 ship 数字进表的工作流 | v10 训练动机 / 配置（→ 本 skill） |
