---
name: eloftr-v11-dualh
description: |
  EfficientLoFTR v11 = v10_msyn_ddp + dual-side aggressive Homography
  augmentation on Megadepth_Syn 4-GPU DDP, cold start from outdoor.ckpt.
  Path I (geometric augmentation route): inherits v10_msyn_ddp cfg
  byte-identical except for 4 cfg overrides (DATASET.ROAD_HOMOGRAPHY_DUAL=
  True / PROB=0.7 / KWARGS=aggressive preset rot_deg=25 scale (0.75,1.25)
  trans 0.12 persp 0.08 / TRAINER.WARMUP_STEP=900) + 1 CLI exp_name. The
  aug strength is ~3x v0-v10 single-side weak preset (rot ±10, scale
  0.9-1.1, trans 0.05, persp 0.03) on every axis; SuperPoint COCO baseline
  comparison: rot ±90 / scale 0.8-1.2 / persp 0.2 (v11 is ~1/3 of
  SuperPoint, ~3x of v10).

  Six files changed: (1) src/datasets/roadscene.py rewrite warp block
  L414-450 to support dual H_ir + H_vis with mask0 &= ir_valid parallel
  to mask1 &= vis_valid; H_0to1 = H_vis @ inv(H_ir) composition formula;
  byte-identical to v0-v10 when DUAL=False (ir_valid=ones makes mask0
  intersection a no-op). (2) src/lightning/data.py +3 lines pull
  ROAD_HOMOGRAPHY_DUAL from cfg + transmit homography_dual= kwarg to
  RoadSceneDataset. (3) src/config/default.py +6 lines declare
  ROAD_HOMOGRAPHY_DUAL=False default + ROAD_HOMOGRAPHY_KWARGS as
  CN(new_allowed=True) so v11 cfg can declare sub-fields without YACS
  strict-mode KeyError. (4) NEW configs/loftr/eloftr_full_v11_dualh_
  aggressive_msyn_ddp.py inherits v10_msyn_ddp + 4 cfg overrides
  (NOT 5: TRAINER.EXP_NAME is NOT a registered field in default.py,
  must come from CLI --exp_name, matching v10_msyn_ddp.py pattern;
  caught this dev-time bug via sanity script first run failing with
  KeyError("Non-existent config key: TRAINER.EXP_NAME") at cfg
  merge_from_file). (5) NEW MyScripts/run_msyn_v11_dualh_ddp.sh
  copies run_msyn_v10_ddp.sh changes only cfg path + exp_name + adds
  --sanity-only fast-path. (6) NEW MyScripts/sanity_v11_dualh.py
  pre-flight verification before 17h ship: 5 checks = shape/dtype
  contract / dual aug firing on IR side via single-vs-dual mask0
  coverage delta >= 0.05 / covisibility floor >= 0.30 / H composition
  recomposition residual < 1.0 px (H_vis @ inv(H_ir) verified independently)
  / backward compat (DUAL=False AUG=False yields H_0to1==I and mask0
  fully True in valid region, byte-identical to v0-v10).

  Key design decisions emerged from chat 2026-05-10:
  (1) Dual + aggressive raw covisibility per pair = ~32% (single aggressive)
  x ~32% (single aggressive) = ~10%, dangerous low signal.
  (2) prob=0.7 mitigates: 0.7 * 10% + 0.3 * 100% = ~37% average covisibility,
  same ballpark as v10 single-weak ~66% so per-batch GT count stays in
  regime where loss has sufficient signal (see covisibility collapse
  analysis in chat).
  (3) WARMUP_STEP doubled 450 -> 900 (actual 1800 -> 3600 step ~0.5 ep)
  is the ONLY hyperparameter changed beyond aug knobs; LR / MSLR /
  patience / sync_bn / freeze / N_SAMPLES_PER_SUBSET all inherited
  from v10 byte-identical for clean ablation. Rationale: dual aggressive
  aug pulls geometric distribution further from outdoor.ckpt prior than
  v10 did, cold start needs longer LR ramp; ~15 min wall-clock cost
  for ~3x reduction in cold-start crash probability.
  (4) outdoor.ckpt cold start (NOT v10 ep11 warm start) keeps v11 vs
  v10 a fair-controlled aug ablation.

  Status: NOT YET shipped as of 2026-05-11. Implementation + sanity
  script done; awaiting server-side `bash MyScripts/run_msyn_v11_dualh_ddp.sh
  --sanity-only` PASS + 17h DDP ship + independent eval. Acceptance
  grading thresholds (set in v11 cfg docstring) match v10 because
  schedule / data / LR identical, only aug differs:
    strong: M3FD-OOD test P@1 >= 0.50, RoadScene OOD P@1 >= 0.45
    medium: M3FD-OOD P@1 in [0.40, 0.50], RoadScene OOD >= 0.40
    weak:   M3FD-OOD P@1 in [0.30, 0.40]
    fail:   ep0 train_loss > 3.0 sustained, OR M3FD-OOD P@1 < 0.30
  Failure modes + retries documented in v11 cfg docstring + run script
  comment block: cold-start crash -> WARMUP 900 -> 1800 + PROB 0.7 -> 0.5;
  aug too aggressive after ep1 -> KWARGS rot=15 scale (0.85,1.15) trans
  0.08 persp 0.05.

  Use when running/debugging/extending v11 / explaining what changed
  vs v10 / answering "why dual H instead of just stronger single-side"
  / 看 v11 / 跑 v11 / 共视坍塌 / dual H 公式怎么推 / mask0 ir_valid
  intersection / sanity 5 项 / EXP_NAME 报错 / WARMUP 加倍理由 / cold
  start aug-shock / 答辩写 v11 实验 / v11 vs v10 ablation comparison.

  Triggers: v11 / v11_dualh / msyn_v11_dualh_aggressive_ddp / dual
  homography / 双侧 H / 双侧激进 H / aggressive aug preset /
  ROAD_HOMOGRAPHY_DUAL / ROAD_HOMOGRAPHY_KWARGS / H_vis @ inv(H_ir) /
  H_0to1 dual composition / mask0 &= ir_valid / 共视坍塌 /
  covisibility collapse / WARMUP 加倍 / WARMUP_STEP 900 / 3600 step /
  cold start aug shock / sanity_v11_dualh / 5 项 sanity check /
  TRAINER.EXP_NAME KeyError / CN new_allowed True YACS strict /
  TRAINER.EXP_NAME not registered / merge_from_file KeyError TRAINER /
  v11 cfg 4 overrides / v11 ship 未完成 / SuperPoint COCO 量级对照 /
  rot 25 scale (0.75 1.25) trans 0.12 persp 0.08 / prob 0.7 /
  v11 vs v10 公平对照 / outdoor.ckpt cold start v11 / dual aug 风险
  与回滚 / 温和档 fallback / 1 ep smoke test 1.5h / DDP image-level
  pre-shard 沿用 v10 / sync_bn 沿用 v10 / 17h wall-clock 估计沿用 v10,
  English 'v11 dual-side aggressive Homography', 'aug strength 3x v10',
  'covisibility collapse 10 pct floor', 'prob=0.7 mitigates 37 pct
  average covisibility', 'WARMUP doubled cold start aug shock',
  'H_vis @ inv(H_ir) composition formula', 'mask0 AND ir_valid parallel
  to mask1', 'CN new_allowed YACS strict mode workaround',
  'TRAINER.EXP_NAME not registered must come from CLI', 'sanity script
  5 checks pre-flight', 'v11 ship not yet completed', 'fair v11 vs v10
  ablation outdoor.ckpt cold start', 'aggressive vs SuperPoint COCO
  baseline 3x weaker still'.

  v11 = path I in cross-modal roadmap (geometric augmentation
  scale-up). Inherits v10 path H stack (data scale-up + DDP infra) +
  v9 path G stack (training-strategy cold start) + v8 path E1 stack
  (MSBN fine-arch) + v7 path F stack (PC + CLAHE input). Companion:
  eloftr-v10-msyn (parent stack), eloftr-cross-modal-experiments (chain
  overview), eloftr-megadepth-syn-data (dataset SOP), eloftr-server-multigpu
  (DDP runtime + sync_bn cost), eloftr-results (跨版本数字总表;
  v11 ship 数字待 backfill), eloftr-tb-summary (训练侧 KPI 总表;
  v11 ship 数字待 backfill).
---

# v11：双侧激进 Homography augmentation（v10 stack + 几何多样性 ×4）

> 继承链：v7 (PC+CLAHE) → v8 (MSBN) → v9 (e2e cold start) → v10 (Megadepth_Syn DDP scale-up) → **v11 (dual-side aggressive H aug)**
> 总览见 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)；v10 stack 见 [eloftr-v10-msyn](../eloftr-v10-msyn/SKILL.md)
> 数据集 SOP 见 [eloftr-megadepth-syn-data](../eloftr-megadepth-syn-data/SKILL.md)（Megadepth_Syn，v11 沿用 v10 同一份数据 + 同一份 PC 缓存）
> 4 卡 DDP runtime + sync_bn 开销见 [eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md)
> **跨版本数字总表**见 [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)（v11 数字待 ship 后 backfill）

**已实现 + 等 ship**：6 个文件（3 个 src 改动 + 3 个 MyScripts/configs 新建），sanity 通过本地 Read+Lint 验证，等用户在服务器 `git pull && bash MyScripts/run_msyn_v11_dualh_ddp.sh --sanity-only` 真正跑 sanity；sanity PASS 后再 `bash MyScripts/run_msyn_v11_dualh_ddp.sh` 进 17h ship。

## 1. 设计动机：v10 几何多样性是否还有空间？

[eloftr-v10-msyn](../eloftr-v10-msyn/SKILL.md) 在 Megadepth_Syn 195 scene × ~115K pair 上拿到综合通用性 1.9020 SOTA。但 v10 用的还是 v0-v10 默认 Homography aug 强度（rot ±10°, scale 0.9-1.1, trans 5%, persp 3%, **仅 warp VIS 一侧**），跟业界基线对比明显保守：

| axis | v0-v10 默认 | **v11 (本次)** | SuperPoint COCO 训练 cfg |
|---|---|---|---|
| rot_deg | ±10° | **±25°** | ±90° |
| scale | 0.9-1.1 | **0.75-1.25** | 0.8-1.2 |
| trans | ±5% | **±12%** | ~±15% |
| persp | ±3% | **±8%** | ±20% |
| 双侧 | 否（仅 VIS）| **是（IR+VIS 各自独立 H）**| 否 |
| prob | 1.0 | **0.7** | 1.0 |

v11 强度是 v10 的 **~3× 单轴 + 双侧（几何多样性 ×4）**，但仍只是 SuperPoint COCO 量级的 1/3。核心问题：

> **Megadepth_Syn 195 scene + v9 stack 在更激进的几何 aug 下能不能进一步刷 OOD？还是会因为共视坍塌反而崩？**

## 2. 共视坍塌（关键约束，必看）

每对训练样本里只有"两边都看得到的像素"才能给 loss 提供监督信号——叫**共视像素**。supervision 代码 [src/loftr/utils/supervision.py:spvs_coarse_roadscene](../../../src/loftr/utils/supervision.py) 把 `valid` 与 4 个条件 AND 收紧（投影数值有效 + 投影落在 image1 内 + mask0 + mask1），任何一个不满足 → 这个 (i,j) 不进 loss。

aug 强度 vs 共视率（60×60 coarse 网格能用的 GT 对数）：

| 模式 | IR 侧 warp 后 | VIS 侧 warp 后 | **共视率** | 每 batch 可用 GT 对 |
|---|---|---|---|---|
| v10 单侧温和 | 100%（不动）| ~66% | **~66%** | ~2400 / 3600 |
| 单侧激进（假设）| 100% | ~32% | **~32%** | ~1150 / 3600 |
| 双侧激进（v11 raw）| ~32% | ~32% | **~10%** | ~360 / 3600 |
| **v11 实际** = 双侧激进 + prob=0.7 | 30% pair 不动 + 70% pair 双侧激进 | 同左 | **~37% 平均** | ~1330 / 3600 |

**为什么 ~10% 危险**：
- 偶发零 GT batch → supervision.py L233 兜底塞 dummy GT (位置=0,0) → 等于喂错标签
- 方差爆炸 → 少数 GT 主导梯度 → 训练震荡
- 冷启动死亡 → outdoor.ckpt 学到的"上下文丰富自然图像"骤然喂只 10% 共视的破碎图，前几百步 loss 不下降甚至发散

**v11 设计选择 prob=0.7**：30% pair 保留 identity 兜底信号 + 70% pair 双侧激进狠抓多样性，平均共视 ~37% 与 v10 ~66% 同一量级（loss 噪声方差差异不大），但几何多样性事实上 ×4。

## 3. 文件改动汇总（6 个文件，~50 行核心代码 + 3 个新建文件）

| 文件 | 性质 | 关键改动 |
|---|---|---|
| [src/datasets/roadscene.py](../../../src/datasets/roadscene.py) | 改 | docstring +14 行（dual 模式语义） / `__init__` +1 参数 `homography_dual` / `__getitem__` 第 414-450 段重写：dual 采样 H_ir + H_vis 独立 / 双侧 cv2.warpPerspective / 双 valid mask / mask0 &= ir_valid 平行 mask1 &= vis_valid / H_t 计算改为 H_vis @ inv(H_ir) |
| [src/lightning/data.py](../../../src/lightning/data.py) | 改 | +3 行：拉取 `cfg.DATASET.ROAD_HOMOGRAPHY_DUAL` + 透传 `homography_dual=` 到 RoadSceneDataset 构造调用 |
| [src/config/default.py](../../../src/config/default.py) | 改 | +6 行：`_CN.DATASET.ROAD_HOMOGRAPHY_DUAL = False` 默认 + `_CN.DATASET.ROAD_HOMOGRAPHY_KWARGS = CN(new_allowed=True)`（**关键**：让 v11 cfg 声明 KWARGS 子字段时 YACS strict 不报错；详见 §5） |
| [configs/loftr/eloftr_full_v11_dualh_aggressive_msyn_ddp.py](../../../configs/loftr/eloftr_full_v11_dualh_aggressive_msyn_ddp.py) | 新建 | 87 行（含 65 行 docstring + 4 行 cfg override + NOTE 解释为什么 EXP_NAME 不在 cfg） |
| [MyScripts/run_msyn_v11_dualh_ddp.sh](../../../MyScripts/run_msyn_v11_dualh_ddp.sh) | 新建 | 117 行 = 复制 v10 ddp script 改 cfg path + EXP_NAME + 加 `--sanity-only` 短路分支 |
| [MyScripts/sanity_v11_dualh.py](../../../MyScripts/sanity_v11_dualh.py) | 新建 | 389 行：5 项 pre-flight 检查（详见 §6）|

向后兼容：DUAL=False 时 H_ir = I → ir_valid = ones → `mask0 &= ir_valid` 是 no-op → v0-v10 byte-identical。

## 4. 核心数学：H_0to1 = H_vis @ inv(H_ir)

设原始 IR 与 VIS 在原坐标系完全对齐（aligned IR-VIS 数据集前提），物理点在原坐标 q：

- 经 IR 侧 warp：物理点出现在 warped IR 坐标 `p_warped_ir = H_ir @ q`
- 经 VIS 侧 warp：物理点出现在 warped VIS 坐标 `p_warped_vis = H_vis @ q`

监督 supervision 需要的是：给 warped IR 上的某点 `p_warped_ir`，找到对应的 warped VIS 坐标。

```
q = inv(H_ir) @ p_warped_ir
p_warped_vis = H_vis @ q = H_vis @ inv(H_ir) @ p_warped_ir
```

⟹ **H_0to1 = H_vis @ inv(H_ir)**

单侧模式下 H_ir = I → H_0to1 = H_vis（v0-v10 byte-identical）。

实现：[src/datasets/roadscene.py L555-560](../../../src/datasets/roadscene.py)

```python
H_0to1_np = (H_vis @ np.linalg.inv(H_ir)).astype(np.float32)
H_0to1_np /= H_0to1_np[2, 2]
H_t = torch.from_numpy(H_0to1_np)
```

`spvs_coarse_roadscene` / `spvs_fine_roadscene` 字段语义保持不变，**0 改动**。

## 5. 训练超参 vs v10 对照（仅 WARMUP 调整）

| 超参 | v10 cfg | v10 actual | v11 cfg | v11 actual | 是否调 | 理由 |
|---|---|---|---|---|---|---|
| CANONICAL_LR | 5e-4 | TRUE_LR=1.25e-4 | 5e-4 | TRUE_LR=1.25e-4 | **不变** | LR 不该跟 aug 强度耦合，loss 内部已对 valid GT 数取 mean |
| **WARMUP_STEP** | **450** | **1800 step (~0.25 ep)** | **900** | **3600 step (~0.5 ep)** | **加倍** | 双侧激进 aug 把几何分布拉离 outdoor.ckpt prior，cold start 需更长 LR ramp；代价 ~15 min wall-clock |
| MSLR_MILESTONES | [3, 5, 7] | ep 3/5/7 衰减 | [3, 5, 7] | 同 | **不变** | 与 v10 公平对照；若 v11 实测 peak 后移留 v11.1 调 |
| EARLY_STOPPING_PATIENCE | 3 | — | 3 | — | **不变** | val 端 H aug 强制关闭（v10 v11 一致），val P@3 饱和模式不变 |
| sync_batchnorm | True | (WORLD_SIZE>1) | True | 同 | **不变** | train.py L237 自动开启 |
| FREEZE_BACKBONE_BN | False | unfreeze | False | unfreeze | **不变** | v10 解锁全 backbone 设计沿用 |
| N_SAMPLES_PER_SUBSET | 28750/rank | 4 rank x 28750 = 115K | 28750/rank | 同 | **不变** | 数据集不变 |
| ckpt_path | outdoor.ckpt | cold start | outdoor.ckpt | cold start | **不变** | 公平对照前提（NOT v10 ep11 warm start） |

LR 时间线（v11 4-card DDP, bs=4, 7187 step/ep/rank）：

```
warmup    : ep 0-0.5         (3600 step linear ramp 0 -> 1.25e-4)   <-- v10 是 0.25 ep
full LR   : ep 0.5-3         (acclimate + dual-aug shock)
MSLR #1   : ep 3-5           (LR x= 0.5 -> 6.25e-5)
MSLR #2   : ep 5-7           (LR x= 0.5 -> 3.13e-5)
MSLR #3   : ep 7-12          (LR x= 0.5 -> 1.56e-5, final stabilisation)
```

## 6. 三个 dev-time 踩坑（避免重复）

### 6.1 `cfg.TRAINER.EXP_NAME = "..."` → YACS strict KeyError

**症状**：sanity 脚本第一次跑挂在 `cfg.merge_from_file(args.main_cfg)`：

```
KeyError: 'Non-existent config key: TRAINER.EXP_NAME'
```

**根因**：`TRAINER.EXP_NAME` 不在 [src/config/default.py](../../../src/config/default.py) 注册。v10_msyn_ddp.py 也**不**设置这个字段——`exp_name` 一直是通过 CLI `--exp_name=...` 传入。我写 v11 cfg 时凭印象多加了一行 `cfg.TRAINER.EXP_NAME = "msyn_v11_dualh_aggressive_ddp"`，触发 YACS strict 检查。

**修复**：删除该行，加 NOTE 注释；exp_name 仍由 [run_msyn_v11_dualh_ddp.sh L101](../../../MyScripts/run_msyn_v11_dualh_ddp.sh) 的 `--exp_name=msyn_v11_dualh_aggressive_ddp` 提供。

**通用教训**：v_x cfg 写新字段前**必须先 grep 一下 default.py 确认字段已注册**；或者加到 default 里。`exp_name` 是约定走 CLI（按 [eloftr-cross-modal-experiments §6](../eloftr-cross-modal-experiments/SKILL.md) "加新 v_x" 工程惯例）。

### 6.2 `_CN.DATASET.ROAD_HOMOGRAPHY_KWARGS = {}` → 后续赋 dict 失败

**问题**：YACS 的 `__setattr__` 把 `{}` 自动包成 CN 节点。然后 v11 cfg 想 `cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS = dict(rot_deg=25.0, ...)` 时，因为类型不一致（dict vs CN）+ 严格模式，会报错。

**修复**：用 `CN(new_allowed=True)` 显式声明节点支持新键，v11 cfg 改成 dot-notation 一项一项设：

```python
# default.py
_CN.DATASET.ROAD_HOMOGRAPHY_KWARGS = CN(new_allowed=True)

# v11 cfg
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 25.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]   # YACS 用 list 不用 tuple
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08
```

**通用教训**：YACS 里需要"用户可声明任意子键的 cfg 节点"用 `CN(new_allowed=True)`，不要用 `{}` 占位。

### 6.3 `mask0 &= ir_valid` 在 dual=False 模式不能误伤 v0-v10

**风险**：新增的 `mask0[:h0_r, :w0_r] &= ir_valid` 行如果 ir_valid 不正确兜底，dual=False 模式下也会让 mask0 变小，破坏 v0-v10 byte-identical。

**保障**：[roadscene.py L487-494](../../../src/datasets/roadscene.py) 兜底分支：

```python
if not np.array_equal(H_ir, np.eye(3, dtype=np.float32)):
    ones0 = np.ones((h0_r, w0_r), dtype=np.uint8)
    ir_valid = cv2.warpPerspective(ones0, H_ir, (w0_r, h0_r), ...).astype(bool)
else:
    ir_valid = np.ones((h0_r, w0_r), dtype=bool)   # <-- 兜底，dual=False 必走这里
```

dual=False → H_ir 永远是 I → 走 else 分支 → ir_valid 全 True → `mask0 &= all-True` 等价 no-op。

sanity 第 5 项专门验证这点（详见 §7）。

## 7. Sanity 脚本的 5 项检查（[MyScripts/sanity_v11_dualh.py](../../../MyScripts/sanity_v11_dualh.py)）

ship 训练前必须在服务器跑：

```bash
bash MyScripts/run_msyn_v11_dualh_ddp.sh --sanity-only
```

| # | 检查 | 抓什么 bug | FAIL 时回滚 |
|---|---|---|---|
| 1 | shape / dtype contract | dataset 输出 tensor 形状 / 类型 / H 矩阵非奇异 + H[2,2]≈1 | dataset patch 写错 |
| 2 | dual aug 真的开了 | cfg→data.py→dataset 透传链 + dual 模式真的 warp IR（用 single-vs-dual mask0 coverage delta >= 0.05 验证，避免非方形图 padding 干扰）| 检查 data.py L91 `road_homography_dual` 拉取 |
| 3 | 共视率 ≥ 0.30 | aug 强度 + prob 组合是否饿死 loss 信号 | aug 太狠：cfg 改 prob=0.5 重跑 sanity |
| 4 | H 复合数学正确 | 公式 `H_vis @ inv(H_ir)` 不能写反；用独立采样 H_ir/H_vis + 64 个测试点的 `H_0to1 @ (H_ir @ q) ?= H_vis @ q` 验证（容差 1.0 px） | dataset L555-560 H_0to1_np 推导改回 |
| 5 | v0-v10 byte-identical | dual=False AUG=False → H_0to1==I + mask0 fully True in valid region | 检查 ir_valid=ones fallback 分支 |

**为什么需要这个脚本**：v11 改了 dataset 核心 warp 逻辑 + 数学公式，比 v10 风险高一档。Sanity 用单卡 + 单 worker + 显式 print 每项数字，1-2 min 内打掉所有坑，比"边跑训练边看 TB"快一个数量级。

## 8. 服务器侧操作流程（vlrlab）

按 `.cursor/rules/01-server-ask-only.mdc`，所有命令必须由用户手动在 tmux 真实 shell 执行。

```bash
# 0. 本地 commit + push
cd c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr
git status
git add src\datasets\roadscene.py src\lightning\data.py src\config\default.py ^
        configs\loftr\eloftr_full_v11_dualh_aggressive_msyn_ddp.py ^
        MyScripts\run_msyn_v11_dualh_ddp.sh MyScripts\sanity_v11_dualh.py
git commit -m "v11: dual-side aggressive Homography aug on Megadepth_Syn"
git push

# 1. 服务器拉代码
ssh xyjiang@222.20.94.235 -p 8708
cd /home/xyjiang/Desktop/yurupeng/eloftr
git pull --ff-only

# 2. 检查卡占用
nvidia-smi      # 真实 shell 看，sandbox 看不到

# 3. Sanity（必须 PASS 才能 ship；约 1-2 min）
source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
bash MyScripts/run_msyn_v11_dualh_ddp.sh --sanity-only

# 4. 进 tmux 跑 12 ep DDP（~17h，沿用 v10 实测，sync_bn 开销 + 4 卡 ~0.7s/step）
tmux new -s msyn_v11
cd /home/xyjiang/Desktop/yurupeng/eloftr
bash MyScripts/run_msyn_v11_dualh_ddp.sh
# detach: Ctrl+B D；attach: tmux attach -t msyn_v11

# 5. 实时监控（前 200 step 关键）
tail -f logs/tb_logs/msyn_v11_dualh_aggressive_ddp/version_*/version_*.log
```

**关键 sanity gate**（前 5 min 内可看出来）：

| 时间 | 期望值 | 不达预期则 |
|---|---|---|
| 启动 1 min 内 | log 出现 `[rank 0..3] building RoadSceneDataset (...) homography_dual=True` 4 次 | 重检查 cfg 加载链 |
| step 10 | `TRUE_LR=1.250e-04, WARMUP_STEP=3600` | 检查 v11 cfg 的 WARMUP_STEP=900 |
| step 100 | train_loss 中位数 < 2.0 | > 3.0：cold start 崩，kill + cfg WARMUP 1800 + PROB 0.5 |
| step 500 | train_loss < 1.5 | > 2.0：aug 太狠，kill + cfg KWARGS 改温和档 |
| ep 0 结束 | val P@3 > 0.4 | < 0.3：模型没收敛 |

## 9. Acceptance grading（与 v10 阈值持平，因为 schedule/data 一致）

```
strong : peak ep 2-7,  M3FD-OOD test P@1 >= 0.50, RoadScene OOD P@1 >= 0.45,  wall-clock <= 18h
medium : peak ep 2-9,  M3FD-OOD P@1 in [0.40, 0.50], RoadScene OOD P@1 >= 0.40,  <= 19h
weak   : peak ep 2-11, M3FD-OOD P@1 in [0.30, 0.40], <= 20h
fail   : ep0 train_loss > 3.0 sustained, OR M3FD-OOD P@1 < 0.30
```

注意：v11 训练数据是 Megadepth_Syn（合成 IR），M3FD/RoadScene 都是 OOD。v10 已建立这个 OOD 基线（M3FD P@1=0.5129 / RoadScene P@1=0.4560），v11 strong = 维持 v10 同档；medium = 略退；weak = 明显退；fail = 几何 aug 反而把模型搞坏。

## 10. 风险与回滚

| 风险 | 检测 | 回滚 |
|---|---|---|
| ep0 train_loss > 3.0（cold start 崩，WARMUP 加倍后概率 ~10%）| 训练前 200 step TB 监控 | cfg 改 WARMUP_STEP=1800 (actual 7200 step ~1 ep) + PROB=0.5 重启 |
| ep1 后 train_loss 持续 > 2.0（aug 太狠学不动）| 看 ep1+ train_loss 中位数 | cfg KWARGS 改温和档 rot=15 / scale (0.85,1.15) / trans 0.08 / persp 0.05 |
| Sanity 第 4 项 FAIL（H 复合写反）| sanity 立即报错 | 回退 dataset L555-560 H_0to1_np 推导，先 fix 再 ship |
| DDP DUAL=False byte-identical 破坏 | sanity 第 5 项 FAIL | 检查 dataset L487-494 ir_valid=ones 兜底分支 |
| 12 ep 训完 RoadScene OOD P@1 < v10 0.456 | eval 阶段发现 | 不算工程回归（aug 强度高本就有变差风险），文档化数字归档；考虑 v11.1 prob 退到 0.5 / KWARGS 退温和档 |

## 11. 不动的部分（重要：避免误改）

- [src/loftr/utils/supervision.py](../../../src/loftr/utils/supervision.py)：`spvs_coarse_roadscene` / `spvs_fine_roadscene` 的 `homography_0to1` 字段语义保持，**0 改动**
- [src/lightning/lightning_loftr.py](../../../src/lightning/lightning_loftr.py)：模型不感知 H 方向 / 来源，**0 改动**
- backbone / coarse / fine module **0 改动**
- PC cache 生成（precompute_pc_edges.py / fix_pc_cache_alignment.py）：仍是未 warp 的边图，warp 在线进行，**0 改动**
- DDP image-level pre-shard（[data.py L267-275 `get_local_split`](../../../src/lightning/data.py)）：与 H 无关，**0 改动**，沿用 v10 修复
- eval 脚本 [MyScripts/eval_roadscene.py](../../../MyScripts/eval_roadscene.py) / `eval_*.bat`：测试时 `homography_aug=False` 强制关闭，DUAL 开关不生效，**0 改动**
- 数据集软链接 / Megadepth_Syn 索引文件 / outdoor.ckpt 等

> **PC 频谱分析发生在 640 long-edge 上**（v11 继承 v10 的 cache 生成链路 `precompute --max_long_edge 640 --pc_nscale 3` + `fix_pc_cache_alignment` INTER_AREA → 480）。这一事实加上 R8 runtime PC fallback（2026-05-11，[`src/utils/pc.py`](../../../src/utils/pc.py)）：**v11 ckpt 部署到新数据集**时，即使没有预计算 PC cache，dataset 也能在 val/test 模式自动 fallback 到 runtime 计算（与 v11 训练分布字节级等价）。Train 模式硬阻断不允许 fallback，保护训练分布锁。详见 [eloftr-eval-pipeline §3.2](../eloftr-eval-pipeline/SKILL.md) 和 [eloftr-v7-pcclahe SKILL §5 R8](../eloftr-v7-pcclahe/SKILL.md)。

## 12. 训练完成后产物归档（按现有 skill 流程）

> v11 ship 完成后回填本节实测数字 + 把 §13 H1/H2 假设结果填入。

- TB 训练侧 KPI 追加到 [results/tb_summary.md](../../../results/tb_summary.md)（按 [eloftr-tb-summary](../eloftr-tb-summary/SKILL.md) skill）
- eval 数字追加到 [results/eval_summary.md](../../../results/eval_summary.md)（按 [eloftr-results](../eloftr-results/SKILL.md) skill）
- 本 SKILL 顶部 description 区"Status: NOT YET shipped"改成"shipped 2026-MM-DD, ep_X final" + 实测数字
- 本 SKILL §9 acceptance grading 替换为实测档位（strong/medium/weak/fail 哪一档）

## 13. 待回填假设（ship 后填实测）

- **H1**：dual + aggressive aug 能否进一步刷综合通用性 SOTA（vs v10 1.9020）？
  - **strong 命中**：v11 综合 ≥ 1.95，证明 v10 仍未榨干几何 aug 空间，论文写"path I 几何 aug scale-up"
  - **持平 ~1.85-1.90**：v10 已接近 aug 上限，v11 是验证负面案例，论文 ablation 章节
  - **退化 < 1.85**：aug 过头 / 共视坍塌 / cold start 失败之一；查 §10 风险表定位

- **H2**：双侧 H 能否让模型学到更对称的 modemb_ir / modemb_vis（v10 ep11 是 0.0861/0.0863 几乎对称）？
  - dual mode 下 IR 也被几何变换，理论上 modemb 学到的特征应该更"模态-几何解耦"
  - 实测看 modemb norm 终态 + IR/VIS 不对称度

- **H3**：in-domain ↔ OOD gap（v10 是 0.027）能否进一步压平到 < 0.02？
  - v11 增加几何多样性，理论上 viewpoint-invariant 特征应该更强
  - 实测看 (M3FD P@3 - RoadScene P@3) 这个差值
