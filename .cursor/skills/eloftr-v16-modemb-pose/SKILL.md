---
name: eloftr-v16-modemb-pose
description: |
  EfficientLoFTR v16 = v14 + v2 cross-modal learnable modality embedding (modemb).
  v16 vs v14 是 1-2 项改动 ablation——cfg.LOFTR.USE_MODALITY_EMB False -> True
  (核心) + cfg.LOFTR.MODALITY_EMB_INIT='zeros' (显式声明保 step 0 byte-identical
  v14), 其他 (IMG_RESIZE=640, NPE 走 train.py fallback [832,832,832,832] 跟
  v0-v13/v14/v15 一致, bs=4, CANONICAL_LR 5e-4, WARMUP 450, MSLR [6,10,14],
  ES patience 5, N_SAMPLES=100, max_ep=18, EVAL_TIMES=1, limit_val_batches=0.2,
  ENABLE_PLOTTING=False, log_every_n_steps=500) 全部继承 v14.

  v2 modemb 简介 (复用 v2 实现, 见 .cursor/skills/eloftr-v2-modemb/SKILL.md):
  - transformer 入口对 IR/VIS coarse features 加 256-d 可学习偏置
    (loftr.py:116-118 在 use_modality_emb=True 时 broadcast add over H,W)
  - 让注意力机制 (Q/K/V 投影) 显式知道模态身份, 学 modality-aware attention
  - 跟 NPE/RoPE 正交: RoPE 是相对位置 (只乘 Q/K), modemb 是绝对模态身份
    (加在 feature 让 Q/K/V 都受影响). v14/v15/v16 NPE 都走 fallback
    [832,832,832,832] 不 stretch, modemb 跟 RoPE 完全独立.
  - 'zeros' init 让 step 0 == v14 baseline (safe finetune, finetune 不会因为
    modemb 反而变差). 失败模式 = norm 卡 0 不动 -> 改 'normal_0.02' (v16b).

  默认超参 (default.py:29-30 已声明, v16 cfg 仅 toggle 不改 hyperparams):
  - cfg.LOFTR.USE_MODALITY_EMB = False (default) -> True (v16)
  - cfg.LOFTR.MODALITY_EMB_INIT = 'zeros' (default, v16 显式声明)
  跟 v2 设计原值完全一致, 让 v16 跟 v2 在 init 上同步, 仅 dataset path
  跟 v2 (RoadScene H 监督) 不同 (v16 走 Megadepth_Syn pose 监督).

  为什么 v14 加 modemb 可能有效:
  - v14 已经走 pose 监督 (cross-view + cross-modal), 但 model 是隐式学
    modality-invariance (通过 BN running stats + transformer 注意力).
  - modemb 提供"显式跨模态信号": transformer 自己知道哪个 token 是 IR /
    哪个是 VIS, 可学 modality-specific Q/K/V projection.
  - 跟 pose 监督正交: pose loss 用反投影 GT 约束几何, modemb 给 transformer
    输入端加模态身份偏置, 信号路径不交叉.
  - 跟 v15 (contrastive) 不同: v15 在 loss 端约束 cross-modal alignment,
    v16 在 input 端给 transformer 模态身份. 两者正交可单独 / 组合 (v17).

  MegaDepth pose path vs RoadScene H path 兼容性 (v2 原设计是 RoadScene):
  - modemb 实现 (loftr.py:116-118) 在 transformer 之前 inject, 跟
    spvs_coarse / spvs_coarse_roadscene 都不冲突——modemb 修改的是 transformer
    输入 feature, dataset path 修改的是 spvs_coarse 输出 GT match. 两者不在
    同一信号通路.
  - v14 走 MegaDepth pose path, modemb 在 transformer 之前 broadcast add,
    spvs_coarse 不感知 modemb. 所以 v16 在 v14 上零代码改动直接可用,
    仅 cfg toggle.

  Fine path 限制 (v2 SKILL §"modemb 信号路径"):
  - modemb 加在 transformer 入口, 残差让信号传到 fine_preprocess
  - 但 fine_preprocess 的 2 个 BN 把 channel-wise 常数偏置完全归零
    (BN(x+b) = BN(x))
  - 即使 BN 不吃, fine_matching argmax 对常数偏置免疫 (row/col softmax
    各自消除 row-only / col-only 偏置)
  - 所以 modemb 主要影响 coarse 阶段, fine refinement 几乎不受影响
  - 这对 v16 不是问题: v14 用 MegaDepth pose 监督, fine 阶段也是 pose-
    supervised sub-pixel regression, 不依赖 modemb 给 fine 信号

  4 个新建文件 (零 src/ 改动):
  (1) configs/loftr/eloftr_full_v16_pose_msyn_ddp.py (~110 行, 继承 v14 ddp
      + 2 行 cfg overrides: USE_MODALITY_EMB=True / MODALITY_EMB_INIT='zeros';
      docstring 含 v2 modemb 设计原理 + v14 v16 ablation 矩阵 + 监控点 +
      acceptance gate)
  (2) configs/loftr/eloftr_full_v16_pose_msyn_singlecard.py (~30 行, 继承
      v16 ddp + 单卡反向缩放 CANONICAL_LR=2e-3 / WARMUP_STEP=1800)
  (3) MyScripts/run_msyn_v16_pose_ddp.sh (~110 行, 基于 v14 ddp.sh 改 3 处:
      main_cfg / exp_name + 注释更新; data_cfg 复用 v14 的
      megadepth_syn_pose_640.py)
  (4) MyScripts/run_msyn_v16_pose_singlecard.sh (~60 行, 基于 v14 singlecard.sh
      改 3 处)

  Ship 状态 (2026-05-15): NOT YET shipped, plan-only.
  cfg + sh 已建好但未 commit. ship 依赖 v14 ship 完成 + METU eval 拿到 v14
  baseline 数字, 才能跑 v16 跟 v14 比 ablation.

  v14 实测对照基础 (待 v14 ship 完成填; v14 又依赖 v13 ship 完成 + METU
  eval 决策):
    v14 best ep / auc@10 / METU auc@20: TODO (依赖 v14 ship)

  v16 schedule 推算 (sample-pass 跟 v14 完全等效, 仅 transformer 输入端
  多 256-d 偏置加法):
    v14 best ep, ES 触发 ep, max_ep 18, MSLR [6,10,14] 全部继承

  理论预期 (METU all auc@20 vs v14):
    加 modemb 通常 +1-5% relative AUC (cross-modal 文献常见)
    但 v14 已强制 modality-invariant (pose 监督 + cross-view), 边际收益
    可能 +-1-3%
    净: 大概率 +-0~+5% (跟 RANSAC 噪声 + v15 同量级)

  工程预期 (wall-clock vs v14):
    显存: 2 个 nn.Parameter(256) fp32 = 2 KB / 卡 + 同样大小梯度. 占 v14
      ~17 GB / 卡 < 0.001%. **可忽略**.
    速度: forward 加 1 次 broadcast add (256-d), 微秒级. **0% step time**.
    TB: 多 2 个 scalar (mod_emb_ir_norm / mod_emb_vis_norm).
    18 ep wall-clock: 跟 v14 ~9-10h **完全一致**.
    events.tfevents: 跟 v14 ~220 MB 基本一致.

  Acceptance gate (METU all auc@20 vs v14):
    Strong  : v16 >= v14 + 1.0pp -> 论文 "modemb in pose supervision works"
    Medium  : v16 in [v14 + 0.3, v14 + 1.0)pp -> 写 ablation 表小贡献
    Flat    : v16 in [v14 - 0.3, v14 + 0.3]pp -> modemb 边际, 论文叙事可省
    Negative: v16 in [v14 - 1.0, v14 - 0.3)pp -> 罕见 (zeros init 起点等价
              v14, 应不会 negative); 检查 modemb dead vs runaway
    Fail    : v16 < v14 - 1.0pp -> 严重负效应, 检查 mod_emb norm trajectory
    Crash   : NaN loss -> zeros init 下几乎不可能

  辅助监控 (TB scalar 来自 lightning_loftr.py:522-530, ENABLE_PLOTTING=False
  仍保留 scalar):
    train/mod_emb_ir_norm 应单调上涨 (从 0 涨到 ~0.5-1.0, 健康)
    train/mod_emb_vis_norm 同上
    ir/vis norm 长期接近 0 -> modemb dead, 改 INIT='normal_0.02' (v16b)
    ir/vis norm 持续 > 5.0 -> modemb runaway, 加 weight_decay 或减 LR

  Use when running/debugging/extending v16 / 解释 v16 vs v14 ablation /
  modemb 在 pose 监督下的兼容性 / v2 modemb 跨数据集 (RoadScene H ->
  Megadepth_Syn pose) 复用 / 监控 mod_emb_ir/vis_norm 单调上涨 / norm 卡 0
  调 INIT='normal_0.02' / norm runaway 加 weight_decay / fine path BN 吃 bias
  / fine_matching argmax 免疫常数偏置 / 写论文 modemb ablation 章节 / 跟
  v15 (contrastive) 正交对比 / 未来 v17 双 buff 组合.

  Triggers: v16 / v16_pose / v16_modemb / v16_pose_modemb / msyn_v16_pose_ddp /
  USE_MODALITY_EMB True / MODALITY_EMB_INIT zeros / modality_emb_ir /
  modality_emb_vis / mod_emb_ir_norm / mod_emb_vis_norm / cross-modal modality
  embedding / v2 modemb in pose path / modemb in MegaDepth pose / loftr.py 47-62
  / loftr.py 116-118 / lightning_loftr.py 522-530 / default.py 29-30 / v2 default
  zeros init / v14 baseline + 1 cfg toggle / modemb runaway / modemb dead /
  正交于 pose 监督 / 跟 v15 contrastive 正交 / fine path BN bias 吃 bias /
  argmax immune constant bias / GPU 散热 v16 复用 v14 风险, English 'v14
  plus modality embedding', 'cross-modal modemb in pose supervision', 'reuse
  v2 modemb across datasets', 'orthogonal to pose supervision', 'two-line
  cfg toggle USE_MODALITY_EMB MODALITY_EMB_INIT zeros'.

  v16 = path J 第二级衍生平行支 (v13 -> v14 -> v15 / v16 平行 ablation),
  在 v14 工程加速框架上叠加 v2 modemb. 测试 "modemb 跟 pose 监督是否正交
  可叠加".
  Inheritance graph:
    eloftr_full.py (v0 baseline)
      -> v13 (B-pure pose 监督, 832, bs=2, eloftr-v13-pose)
        -> v14 (640 + 工程加速, bs=4, eloftr-v14-resolution-640)
          -> v15 (+ USE_CONTRASTIVE=True, eloftr-v15-pose-contrast)
          -> v16 (THIS skill, + USE_MODALITY_EMB=True)
       (未来 v17 = v14 + USE_MODALITY_EMB=True + USE_CONTRASTIVE=True)

  Companion:
    eloftr-v14-resolution-640 (parent / v16 acceptance gate 起点 = v14
      实测数字, 待 v14 ship 完成填)
    eloftr-v13-pose (祖父 / v16 间接继承的 B-pure stack)
    eloftr-v2-modemb (v2 modemb 实现 + 设计原理 + 注入位置分析 + RoadScene
      H 监督下的 ablation 数字 + fine path BN 吃 bias 论证)
    eloftr-v15-pose-contrast (sister / contrastive 单 ablation, v16 modemb
      单 ablation 平行对照)
    eloftr-megadepth-syn-data (Megadepth_Syn dataset SOP)
    eloftr-cross-modal-experiments (chain 路线总览)
    eloftr-server-multigpu (DDP runtime + 5 traps)
    eloftr-eval-pipeline (eval_metu_vistir_finetuned.bat 协议)
    eloftr-results (跨版本数字总表; v16 数字待 backfill)
    eloftr-tb-summary (训练侧 KPI 总表; v16 wall-clock / best_ep /
      mod_emb_norm 曲线 待 backfill)
---

# v16: v14 + v2 Cross-Modal Learnable Modality Embedding (Plan-Only)

> 继承链：`eloftr_full.py` → **`v13` (path J B-pure)** → **`v14` (640 + 工程加速)** → **`v16` (THIS, + modemb)**
> 父对照 → [eloftr-v14-resolution-640](../eloftr-v14-resolution-640/SKILL.md)
> modemb 实现 → [eloftr-v2-modemb](../eloftr-v2-modemb/SKILL.md)
> 平行 sister → [eloftr-v15-pose-contrast](../eloftr-v15-pose-contrast/SKILL.md)
> 跨版本数字 → [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)
> **状态**：plan-only, ship pending. 依赖 v14 ship 完成 + METU eval 拿到 baseline 数字, 才能跟 v16 比 ablation.

## 1. 设计意图：modemb 跟 pose 监督是否正交可叠加？

v2 在 RoadScene H 监督上加 modemb 拿到 +X% AUC (见 [eloftr-v2-modemb](../eloftr-v2-modemb/SKILL.md))。但 v14 已经走 pose 监督 (model 已经被强制学 modality-invariant features), modemb 是不是冗余？

v16 想回答：

> **加 modemb (一个显式的输入端模态身份偏置) 是否在 pose 监督路径下仍有边际收益？**

3 个可能的答案：
- **可叠加正交**: modemb 跟 pose 监督正交 (input 端模态偏置 vs 反投影 GT), 各自补对方信号 → v16 > v14
- **冗余饱和**: pose 监督已经把 modality 学透了, modemb 边际收益 0 → v16 ≈ v14
- **干扰**: modemb 让 transformer 学到 modality-specific bias, 跟 pose 监督的 modality-invariant 目标矛盾 → v16 < v14

## 2. 4 个新建文件 (零 src/ 改动) + cfg 继承链

| 文件 | 关键设计 |
|---|---|
| [`configs/loftr/eloftr_full_v16_pose_msyn_ddp.py`](../../../configs/loftr/eloftr_full_v16_pose_msyn_ddp.py) | ~110 行。继承 v14 ddp + **2 行 cfg overrides**: `cfg.LOFTR.USE_MODALITY_EMB = True` + `cfg.LOFTR.MODALITY_EMB_INIT = 'zeros'`. docstring 含 v2 modemb 设计原理 + v14 v16 ablation 矩阵 + 监控点 + acceptance gate. |
| [`configs/loftr/eloftr_full_v16_pose_msyn_singlecard.py`](../../../configs/loftr/eloftr_full_v16_pose_msyn_singlecard.py) | ~30 行。继承 v16 ddp + 单卡反向缩放 `CANONICAL_LR=2e-3` / `WARMUP_STEP=1800` 让 TRUE_LR=1.25e-4 跟 DDP 一致. |
| [`MyScripts/run_msyn_v16_pose_ddp.sh`](../../../MyScripts/run_msyn_v16_pose_ddp.sh) | ~110 行。基于 v14 ddp.sh **改 3 处**: main_cfg / exp_name + 注释更新; data_cfg 复用 v14 的 [megadepth_syn_pose_640.py](../../../configs/data/megadepth_syn_pose_640.py). |
| [`MyScripts/run_msyn_v16_pose_singlecard.sh`](../../../MyScripts/run_msyn_v16_pose_singlecard.sh) | ~60 行。基于 v14 singlecard.sh 改 3 处. |

**0 src/ 改动**。继承链:

```
eloftr_full.py (v0 baseline)
  └─ v13_pose_msyn_ddp (B-pure, 832, bs=2)
       └─ v14_pose_msyn_ddp (12 项 overrides, 640, bs=4)
            ├─ v15_pose_msyn_ddp (+ 1 行 USE_CONTRASTIVE=True)
            └─ v16_pose_msyn_ddp (THIS, + 2 行 USE_MODALITY_EMB=True / INIT='zeros')
       (未来 v17 = v14 + USE_CONTRASTIVE + USE_MODALITY_EMB)
```

## 3. 唯一改动 + v2 默认超参复用

```python
# eloftr_full_v16_pose_msyn_ddp.py 唯一改动 (相对 v14 ddp)
cfg.LOFTR.USE_MODALITY_EMB = True
cfg.LOFTR.MODALITY_EMB_INIT = 'zeros'
```

USE_MODALITY_EMB / MODALITY_EMB_INIT 走 [`src/config/default.py:29-30`](../../../src/config/default.py) 默认值:
- `USE_MODALITY_EMB = False` (default) → `True` (v16)
- `MODALITY_EMB_INIT = 'zeros'` (default, v16 显式声明)

跟 v2 原值完全一致 (v2 设计目的是给 transformer 显式跨模态信号)。让 v16 跟 v2 在两个超参上完全同步, 仅 dataset path 跟 v2 (RoadScene H 监督) 不同 (v16 走 Megadepth_Syn pose 监督)。

## 4. v2 modemb 兼容 MegaDepth pose path

**关键代码点** ([`src/loftr/loftr.py:47-62`](../../../src/loftr/loftr.py) + [`L116-118`](../../../src/loftr/loftr.py)):

```python
# L47-62: __init__ 注册 nn.Parameter
self.use_modality_emb = config.get('use_modality_emb', False)
if self.use_modality_emb:
    d_model = config['coarse']['d_model']  # 256
    init_mode = config.get('modality_emb_init', 'zeros')
    if init_mode == 'zeros':
        ir_init = torch.zeros(d_model)
        vis_init = torch.zeros(d_model)
    elif init_mode == 'normal_0.02':
        ir_init = torch.randn(d_model) * 0.02
        vis_init = torch.randn(d_model) * 0.02
    self.modality_emb_ir = nn.Parameter(ir_init)
    self.modality_emb_vis = nn.Parameter(vis_init)

# L116-118: forward broadcast add (在 transformer 之前)
if self.use_modality_emb:
    feat_c0 = feat_c0 + self.modality_emb_ir.view(1, -1, 1, 1)
    feat_c1 = feat_c1 + self.modality_emb_vis.view(1, -1, 1, 1)
feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
```

**modemb 修改的是 transformer 输入 feature**, 跟 spvs_coarse (修改 GT match 字段) 信号路径不交叉, 所以 **v16 在 v14 上零代码改动直接可用**, 仅 cfg toggle。

跟 v15 的差异: v15 修改 loss 端 (loftr_loss.py:134-180 加 InfoNCE), v16 修改 forward feature (loftr.py:116-118 加 broadcast bias)。两者**信号路径完全独立**, 未来 v17 可同时启用。

## 5. fine path BN 吃 bias / argmax 免疫论证 (来自 v2 SKILL §"modemb 信号路径")

modemb 加在 transformer 入口, 残差让 modemb 信号通过 self/cross-attention 一路传到 fine_preprocess。但**fine 阶段 modemb 几乎完全失效**:

**第一次吃**：fine_preprocess 的 2 个 BN (FPN 链路 `feat_c -> upsample -> +x2 -> layer2_outconv2 -> +x1 -> layer1_outconv2`)
- `+x2` / `+x1` 来自 backbone 中间层（不带 modemb）, 把 modemb 信号稀释一半
- 紧跟着的 `BatchNorm2d` 把 channel-wise 常数偏置完全归零（`BN(x+b) = BN(x)`）

**第二次吃**：fine_matching argmax (即使 BN 不吃)
- 给 `feat_f0 += b_ir`、`feat_f1 += b_vis`, inner product 多出 3 项: `<b_ir, feat_f1>` 只跟 r 有关、`<feat_f0, b_vis>` 只跟 l 有关、`<b_ir, b_vis>` 是常数
- row/col softmax 各自把 row-only / col-only / 全局常数完全消除
- argmax 位置不变 → fine peak 不变 → fine refinement 跟 modemb 无关

**对 v16 不是问题**: v14 用 MegaDepth pose 监督, fine 阶段也是 pose-supervised sub-pixel regression, 不依赖 modemb 给 fine 信号。v16 modemb 只在 coarse 阶段救场, 跟 v14 fine refinement 行为一致。

## 6. 工程开销 (相对 v14 完全可忽略)

| 项 | v14 baseline | v16 增量 | v16 总 |
|---|---|---|---|
| 显存 / 卡 | ~17 GB | 2 个 nn.Parameter(256) fp32 = 2 KB + grad 2 KB = 4 KB | **~17 GB** (+0.000023%) |
| step time | ~0.83 s/step | forward 加 1 次 broadcast add (256-d, 微秒级) | **~0.83 s/step** (0% 影响) |
| TB scalar | 4 个 (loss/auc) | +2 个 (mod_emb_ir_norm / mod_emb_vis_norm) | 6 个 |
| events.tfevents | ~220 MB | scalar 增量可忽略 | **~220 MB** |
| ep wall-clock | ~33 min | 0 增量 | ~33 min |
| 12-18 ep ship | ~9-10 h | 0 增量 | **~9-10 h** |

## 7. acceptance gate (待 v14 ship 完成 + METU eval 拿到 baseline 数字后填)

| 档位 | METU all auc@20 vs v14 | 解读 |
|---|---|---|
| Strong | ≥ v14 + 1.0pp | 论文 "modemb in pose supervision works", modemb 跟 pose 监督正交可叠加 |
| Medium | in [v14 + 0.3, v14 + 1.0)pp | 写 ablation 表小贡献 |
| Flat | in [v14 − 0.3, v14 + 0.3]pp | modemb 边际, 论文叙事可省略此 ablation |
| Negative | in [v14 − 1.0, v14 − 0.3)pp | 罕见 (zeros init 保护); 检查 mod_emb norm dead vs runaway |
| Fail | < v14 − 1.0pp | 严重负效应, 检查 mod_emb norm trajectory |
| Crash | NaN loss | zeros init 下几乎不可能 |

## 8. 训练时关键监控点 (TB scalar)

ENABLE_PLOTTING=False 关掉 train figure, 但**所有 scalar 仍保留**, v16 多出的 2 个 scalar 都能在 TB 看到:

| scalar | 健康判据 | 病理信号 |
|---|---|---|
| `train/mod_emb_ir_norm` | 单调上涨 (从 0 涨到 ~0.5-1.0) | 长期 < 0.05 → modemb dead, 切 INIT='normal_0.02' (v16b) |
| `train/mod_emb_vis_norm` | 同上 | 同上 |
| 两者对比 | 接近 (IR / VIS 各自学到独立模态身份) | 长期相差 > 5x → 模态偏向 (image0=IR vs image1=VIS 学到不对称信号) |
| 任一 norm | 不超过 ~5.0 | 持续 > 5.0 → modemb runaway, 加 weight_decay 或减 LR |

## 9. ship 时序 (依赖 v14 完成)

```mermaid
flowchart LR
  v13["v13 ship 实测<br/>9 ep, ES 即将触发<br/>best ep6 auc@10=0.4096"]
  v13 --> v13Eval["v13 METU eval"]
  v13Eval --> v14Decide["v14 跑 vs 跳过"]
  
  v14Decide -->|跑| v14Ship["v14 ship<br/>~9h"]
  v14Ship --> v14Eval["v14 METU eval"]
  v14Eval --> route["v15 / v16 平行决定"]
  
  route -->|v15: pose + contrastive| v15Ship["v15 ship<br/>~9-10h"]
  route -->|v16: pose + modemb (THIS)| v16Ship["v16 ship<br/>~9-10h"]
  
  v15Ship --> v15Eval["v15 METU eval"]
  v16Ship --> v16Eval["v16 METU eval"]
  
  v15Eval --> v17Decide["v17? (双 buff 组合)"]
  v16Eval --> v17Decide
  
  v17Decide -->|"v15 + v16 都 strong/medium"| v17["v17 = v14 + USE_CONTRASTIVE + USE_MODALITY_EMB"]
  v17Decide -->|"否则"| stop["停在最佳单变量版本"]
```

**推荐流程**：
1. **现在 (本 skill)**：cfg + sh 已建好, plan-only 状态, 等 v14 ship 完成
2. **v14 ship 完成 + METU eval 后**：看 v14 数字
   - v14 数字 strong/medium → v15 / v16 平行跑确认 modemb / contrastive 是否正交可叠加
   - v14 数字 weak/fail → 跳过 v15 / v16, 反思 v14 设计
3. **v15 / v16 ship 完成后**：backfill 数字到 [eloftr-results](../eloftr-results/SKILL.md)
4. **如果 v15 / v16 都 strong/medium**：v17 (双 buff 组合) 验证 stack 效应

## 10. v16 ship 后的 backfill 工作

```bash
python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/msyn_v16_pose_ddp/version_<N>
```

backfill 进:
- 本 skill description 的 "Ship 状态" + 实测曲线
- [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)
- [eloftr-tb-summary](../eloftr-tb-summary/SKILL.md) → [`results/tb_summary.md`](../../../results/tb_summary.md)
- 跟 v15 (sister ablation) 对比, 写 v17 决策依据

特别关注:
- `train/mod_emb_ir_norm` / `vis_norm` 末尾值（healthy ~0.5-1.0; dead < 0.05 → 标记 v16b 实验; runaway > 5.0 → 标记需要正则）
- v16 vs v14 的 attention pattern 差异（v2 SKILL 提到 modemb 影响 attention head selection）
