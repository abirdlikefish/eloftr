---
name: eloftr-roadscene-data
description: Integrate the RoadScene IR-VIS dataset (and any other aligned IR-VIS dataset reusing the same I/O class) into EfficientLoFTR for cross-modal matching. Use when the user mentions RoadScene, cropinfrared, crop_LR_visible / crop_HR_visible, RoadSceneDataset, A3 padding, ROAD_PAD_SIZE, Homography augmentation, mask propagation bugs, "Trying to resize storage", "Calculated padded input size", asks how to add a custom IR-VIS dataset that returns mask0/mask1/homography_0to1, or asks how data_source dispatch works for aligned IR-VIS datasets (RoadScene + M3FD share RoadSceneDataset / spvs_*_roadscene / _compute_roadscene_metrics via is_aligned_irvis / ALIGNED_IRVIS_SOURCES).
---

# RoadScene IR-VIS 数据集集成（Solution A3）

> 注：自从引入"对齐 IR-VIS 白名单"机制后，本 skill 描述的 `RoadSceneDataset` 与全套 IR-VIS 监督 / metric / plotting 路径**已经不再只服务 RoadScene 本身**——M3FD（以及未来的 MSRS / LLVIP / TNO）通过 `configs/data/<name>_trainval.py` 复用同一套代码。具体见下方第 2.5 节"对齐 IR-VIS 白名单机制"。新接入数据集时优先看那一节，再回到本 skill 处理 padding / mask / Bug 等通用问题。M3FD 的接入细节单独看 [eloftr-m3fd-data](../eloftr-m3fd-data/SKILL.md)。

## ⚠️ 必读：HR vs LR 对齐陷阱

`data/RoadScene/` 下有三个图像目录：

| 目录 | 是否与 `cropinfrared` 像素对齐 | 用途 |
|------|------------------------------|------|
| `cropinfrared`     | —     | IR 输入（image0） |
| `crop_LR_visible`  | ✅ 同分辨率、同 FoV | 与 IR 配对的 VIS（image1） |
| `crop_HR_visible`  | ❌ 分辨率更高、FoV 更宽 | 高分辨率参考图，**不能**作为匹配目标 |

早期所有实验默认用 `crop_HR_visible` 训练，监督信号本身就是错的。已经把 7 处默认值统一改成 `crop_LR_visible`：

1. [src/config/default.py](../../../src/config/default.py) `_CN.DATASET.ROAD_VIS_SUBDIR = 'crop_LR_visible'`
2. [configs/data/roadscene_trainval.py](../../../configs/data/roadscene_trainval.py) `cfg.DATASET.ROAD_VIS_SUBDIR = "crop_LR_visible"`
3. [src/datasets/roadscene.py](../../../src/datasets/roadscene.py) `vis_subdir: str = "crop_LR_visible"`
4. [src/lightning/data.py](../../../src/lightning/data.py) `self.road_vis_subdir = getattr(config.DATASET, 'ROAD_VIS_SUBDIR', 'crop_LR_visible')`
5. [MyScripts/eval_roadscene.py](../../../MyScripts/eval_roadscene.py) 同步
6. [MyScripts/infer_roadscene_official.py](../../../MyScripts/infer_roadscene_official.py) `vis_dir = Path("data/RoadScene/crop_LR_visible")`
7. [MyScripts/make_roadscene_splits.py](../../../MyScripts/make_roadscene_splits.py) 默认参数 `--vis_subdir crop_LR_visible`

**新建 v_x config 时不需要再 override `ROAD_VIS_SUBDIR`**，只在确实想换成 HR 做对照实验时显式设置。

## 1. 数据集架构（Solution A3：独立 resize + bottom-right pad）

源码：[src/datasets/roadscene.py](../../../src/datasets/roadscene.py)。

为什么 A3 而不是 “联合 resize 到同一目标尺寸”：
- IR / VIS 在原始 RoadScene 上虽然 FoV 一致但尺寸偶有 1-2 px 偏差，独立 resize 可以保留各自的物理像素比例。
- `default_collate` 要求 batch 内 tensor 形状完全相同，**不 pad 就只能 `batch_size=1`**。
- bottom-right padding（左上对齐）让 mask 仍然是“左上区域 = 真实像素”的常见结构，与 LoFTR 现有 `coarse_matching` 的 `BORDER_RM` 行为兼容。

`__getitem__` 关键流程（[src/datasets/roadscene.py:216-321](../../../src/datasets/roadscene.py)）：

1. `_resize_keep_aspect`：按 `max(H, W) == ROAD_IMG_RESIZE=480` 等比缩放，再向下取整到 `ROAD_DF=32` 的整数倍。IR 和 VIS 各自走一遍。
2. （仅 train）随机 Homography 在 *未 pad 的 VIS 帧* 上 warp，使旋转中心位于真实内容中心而不是 padding 后的画布中心。
3. `pad_bottom_right` 把两张图都 pad 到 `(ROAD_PAD_SIZE=480, ROAD_PAD_SIZE=480)`，同时返回 `mask0` / `mask1`。
4. `mask1[:h1_r, :w1_r] &= vis_valid`：把 “padding 区域 = False” 与 “warp 后该像素源自原图 = True” 求 AND，确保被 warp 出画外的像素也被标为无效。
5. 把 mask 下采样到 coarse 分辨率 `1/8`（与 MegaDepthDataset 一致）。
6. `scale0/scale1 = [1, 1]`：padded image 自身就是“原始分辨率参考”，原始尺寸只通过 `orig_scale0/1` 留给评估脚本反投影。
7. 输出 dict 不含 `depth0/depth1`、`T_0to1/T_1to0`、`K0/K1`，因为 RoadScene 没有相机几何。

数据 root 路径在 [configs/data/roadscene_trainval.py](../../../configs/data/roadscene_trainval.py) 里设为 `data/RoadScene/`，list 文件位于 `data/RoadScene/index/{train,val,test}_pairs.txt`。

## 2. data 模块的 short-circuit 分支

[src/lightning/data.py:221-251](../../../src/lightning/data.py) 在 `_setup_dataset` 里给"对齐 IR-VIS"数据源加了一个早 return 分支，跳过 ScanNet/MegaDepth 的 per-scene `.npz` 迭代，直接用一个 `RoadSceneDataset`。判断走 `is_aligned_irvis(data_source)`（见第 2.5 节），所以 `'RoadScene'` / `'M3FD'` / 未来加进白名单的任何名字都会走这条分支。所有 `ROAD_*` 字段都在 `__init__` 里 `getattr` 拿到，包括 `ROAD_PAD_SIZE`、`ROAD_HOMOGRAPHY_AUG / PROB / KWARGS`，所以**字段名虽然带 `ROAD_` 前缀，但语义已经是"对齐 IR-VIS 通用开关"**，不要因为名字误以为它仅适用于 RoadScene。

`homography_aug=(self.road_homography_aug and mode == 'train')` 这一行决定了 val/test 一定不增强；评估脚本想强制开 Homography 必须自己改 `mode`（见 `eloftr-eval-pipeline` skill）。

`RoadSceneDataset` 的 `dataset_name` 参数（默认 `'RoadScene'`）由 `data.py` 在调用时透传 `dataset_name=str(data_source)` 注入；`__getitem__` 把它写进输出 dict 的 `dataset_name` / `scene_id` 字段，下游 dispatch 再从 batch 字段读到这个值。

## 2.5 对齐 IR-VIS 白名单机制

**单一可信源**：[src/utils/data_source.py](../../../src/utils/data_source.py)

```python
ALIGNED_IRVIS_SOURCES = frozenset({"roadscene", "m3fd"})

def is_aligned_irvis(name) -> bool:
    if name is None:
        return False
    return str(name).lower() in ALIGNED_IRVIS_SOURCES
```

**5 处 dispatch 全部走 `is_aligned_irvis(...)`**——任何新加判断必须也走它，**禁止再写 `== 'roadscene'` 或 `lower() == 'roadscene'` 这种字面量比较**：

| # | 位置 | 作用 |
|---|---|---|
| 1 | [src/lightning/data.py:225](../../../src/lightning/data.py) | dataset short-circuit（不走 npz） |
| 2 | [src/loftr/utils/supervision.py](../../../src/loftr/utils/supervision.py) `compute_supervision_coarse` | dispatch 到 `spvs_coarse_roadscene` |
| 3 | [src/loftr/utils/supervision.py](../../../src/loftr/utils/supervision.py) `compute_supervision_fine` | dispatch 到 `spvs_fine_roadscene` |
| 4 | [src/lightning/lightning_loftr.py](../../../src/lightning/lightning_loftr.py) `validation_step` | 走 `_compute_roadscene_metrics` |
| 5 | [src/lightning/lightning_loftr.py](../../../src/lightning/lightning_loftr.py) `validation_epoch_end` | 走 `_aggregate_roadscene_metrics` + log `precision@{1,3,5}px` |
| 6 | [src/utils/plotting.py](../../../src/utils/plotting.py) `_compute_conf_thresh` | 用 px 阈值 (3.0) 而不是 epipolar 阈值 |
| 7 | [src/utils/plotting.py](../../../src/utils/plotting.py) `make_matching_figures` | 走 `_make_evaluation_figure_roadscene` |

变量名仍写 `is_roadscene` 是为了最小化 git diff，**语义已经是"是否为对齐 IR-VIS"**。

**[src/loftr/utils/supervision.py:251](../../../src/loftr/utils/supervision.py) 的 `assert len(set(data['dataset_name'])) == 1`** 故意保留：单 batch 内不允许混 RoadScene + M3FD 样本（监督函数本身写得能处理任何对齐 IR-VIS 数据，但当前 lightning data 模块没准备好"per-batch 同源采样"）。**未来**做联合训练再单独立项放宽。

### 加新对齐 IR-VIS 数据集的标准 3 步

不论 MSRS、LLVIP、TNO 还是其他数据集，全程**不要触碰 5 处 dispatch 中的任何一处**：

1. 把数据放成 `data/<NAME>/<ir_dir>/<id>.<ext>` + `data/<NAME>/<vis_dir>/<id>.<ext>`，文件名一一对应。
2. 在 [src/utils/data_source.py](../../../src/utils/data_source.py) 的 `ALIGNED_IRVIS_SOURCES` 里加上数据集小写名（如 `'msrs'`）。
3. 复制 [configs/data/m3fd_trainval.py](../../../configs/data/m3fd_trainval.py) → `configs/data/<name>_trainval.py`，改 `M3FD_PATH`、`TRAINVAL_DATA_SOURCE`、`ROAD_IR_SUBDIR`、`ROAD_VIS_SUBDIR`，必要时改 `ROAD_IMG_RESIZE` / `ROAD_PAD_SIZE`。

可选：若文件名结构不同，复制 [MyScripts/make_m3fd_splits.py](../../../MyScripts/make_m3fd_splits.py) → `make_<name>_splits.py`，改默认 `--root` / `--ir_subdir` / `--vis_subdir` / `--ext`。

## 3. Homography 监督

[src/loftr/utils/supervision.py](../../../src/loftr/utils/supervision.py) 里新增的两个函数：

- `_warp_pts_homography(pts, H)`：把 batch 化的 3x3 H 应用到 2D 点，返回 warp 后坐标和 `valid`（齐次坐标第三维 > 0）。
- `spvs_coarse_roadscene` / `spvs_fine_roadscene`：用 `H_0to1` 把 IR coarse grid 投到 VIS，构造 `conf_matrix_gt`。

`spvs_coarse_roadscene` 关键的 mask AND 逻辑（[src/loftr/utils/supervision.py:213-220](../../../src/loftr/utils/supervision.py)）：

```python
mask0_flat = data['mask0'].reshape(N, -1)
mask1_flat = data['mask1'].reshape(N, -1)
valid = valid & mask0_flat
target_valid = mask1_flat.gather(1, nearest_index1)
valid = valid & target_valid
```

“mask 同步追踪”的含义就是这两行：在 “warp 是否落在 VIS 画布内” 之外，还要要求 “源 IR 格子是真像素 (mask0)” **AND** “目标 VIS 格子也是真像素 (mask1.gather(nearest_index1))”，避免在 zero-pad 区或 warp-out 区域产生伪 GT。

`compute_supervision_coarse / fine` 在 [src/loftr/utils/supervision.py](../../../src/loftr/utils/supervision.py) 里按 `data_source.lower()` dispatch 到 `spvs_*_roadscene` 还是原生 `spvs_*`。

## 4. Transformer / linear_attention 的 mask 传播 BUG（务必同步修）

LoFTR 有两处对 “非矩形 mask + 双维度同时小于画布” 的隐式假设，在 ScanNet/MegaDepth 上恰好不会触发，RoadScene + Homography 一上就崩。

### Bug A：`mask0[0].sum(-2)[0]` 假设 mask 是“左上矩形”

Homography warp 出来的 mask 经常 col 0 或 row 0 上一个 True 都没有，`sum(-2)` 在 “第一列全 0” 时返回 `[0, k, k, k, ...]`，旧代码取 `[0]` 直接拿到 0，crop 出 0 宽特征 → `Calculated padded input size per channel: (36 x 0). Kernel size: (4 x 4)`。

修复：用 `_bound_extent(v) = v.nonzero()[-1].item() + 1`，意为 “最右一个 True 索引 + 1”，等价于 “能装下所有 True 的最小左上 bounding box”。对 ScanNet 那种规则 mask 也仍然成立。

源码：
- [src/loftr/loftr_module/transformer.py:142-149](../../../src/loftr/loftr_module/transformer.py)（`bs == 1` 优化路径）
- [src/loftr/loftr_module/linear_attention.py:17-40](../../../src/loftr/loftr_module/linear_attention.py)（`crop_feature`，`bs > 1` 时按样本走的路径）

`transformer.py` 还多了一个保护：当 `mask_h0/w0/h1/w1` 中任何一个 `< agg_size` 时跳过 cropping 优化，避免 conv aggregator 接 0 输入（见同一文件 159-163 行）。

### Bug B：`elif` + `mask_W0` 在 “双维度都 < 画布” 时少 pad 一边

`pad_feature` / transformer 末尾的 pad-back，原代码：

```python
if mask_h0 != mask_H0:
    feat = torch.cat([feat, zeros(..., mask_H0-mask_h0, mask_W0)], dim=-2)  # 用全宽
elif mask_w0 != mask_W0:                                                    # 排他 elif
    feat = torch.cat([feat, zeros(..., mask_H0, mask_W0-mask_w0)], dim=-1)
```

只能处理 “只 H 或只 W 之一被裁” 的 MegaDepth 情形。RoadScene 由于 IR / VIS 各自独立 resize，**两维同时小于 PAD_SIZE 是常态**。修复是：

1. `elif` → `if`（两个维度都各自判断、各自 pad）。
2. 第一段 pad 用的宽度从 `mask_W0`（全宽）改成 `mask_w0`（**当前已裁宽度**）；这样第二段沿着 “刚 pad 完高的特征” 继续沿 W 拼，张量形状才一致。

源码：
- [src/loftr/loftr_module/transformer.py:186-196](../../../src/loftr/loftr_module/transformer.py)
- [src/loftr/loftr_module/linear_attention.py:43-54](../../../src/loftr/loftr_module/linear_attention.py)

### 复现这两个 bug 的最简方法

跑 `MyScripts/run_roadscene_debug.bat`（`bs=1`）触发 Bug A；跑 `MyScripts/run_roadscene_small.bat`（`bs=2`）触发 Bug B（`bs > 1` 时 transformer 走 linear_attention 的 per-sample 循环，没修 linear_attention 就会再次崩在同样的位置）。

## 5. 评估指标

[src/lightning/lightning_loftr.py:184-235](../../../src/lightning/lightning_loftr.py) 的 `_compute_roadscene_metrics`：

- 用 `batch['homography_0to1']` 把预测的 IR 关键点 (`mkpts0_f`) warp 到 VIS 坐标，与预测的 VIS 关键点 (`mkpts1_f`) 算 `pixel_errs = ||warp(mkpts0) - mkpts1||_2`。
- val 阶段不增强 → `H = I`，等价于直接坐标相减。
- per-pair 收集 `pixel_errs / num_matches / mean_conf`，留到 `validation_epoch_end` 再聚合。

聚合（[src/lightning/lightning_loftr.py:364-386](../../../src/lightning/lightning_loftr.py) `_aggregate_roadscene_metrics`）走 **per-match** 平均：把所有 pair 的所有 match 的 `pixel_errs` 拼成一个长向量，再算 `(errs < t).mean()`。这样匹配数多的 pair 自然权重大，符合 “越多正确匹配越好” 的直觉。

物理上限：RoadScene 的 IR/VIS 标定残差大约在 2-5 px，因此 `precision@3px` 的天花板大致在 60-70%，看到训练 val 突破 80% 几乎一定是 pipeline 问题（参见 `eloftr-eval-pipeline` skill 第 5 节）。

## 6. 重新生成 split 的注意事项

- `cropinfrared / crop_LR_visible / crop_HR_visible` 三个目录里的文件名集合一致（同一个 RoadScene 帧），所以从 HR 切到 LR **不需要重新生成 `train/val/test_pairs.txt`**。
- 想换不同的 80/10/10 → 用 [MyScripts/make_roadscene_splits.py](../../../MyScripts/make_roadscene_splits.py)，默认就是 80/10/10、同种子。
- 生成完一定要 `wc -l data\RoadScene\index\*_pairs.txt` 检查总行数 = 数据集对数（**221 对**，README 上写明）；少了说明哪个目录文件名集合对不上，必须先核对。

### 数据规模与 BN 收敛瓶颈（重要）

当前 split 是 **177 train / 22 val / 22 test**（已用满 221 对）。这个数量级给训练带来两个隐性后果，直接影响 v3/v4 实验的指标解读：

1. **BN running-stat 收敛步数不足**：bs=4 时 ~40 step/epoch，30 epoch 仅 ~1200 step。BN 用 momentum=0.1 EMA 更新 running stats，从 MegaDepth 预训练值漂移到 IR-VIS 域典型需要 ≥3000 步，所以**任何"解冻 fine_preprocess BN"的 v_x 实验都需要至少 80 epoch 或扩数据**才能让 running stats 收敛。已用 v4 REVISION 2 的实测验证（p@1px 暴跌到 0.329 而 p@5px 反超到 0.814）见 [eloftr-cross-modal-experiments §5](../eloftr-cross-modal-experiments/SKILL.md)。
2. **过拟合 + val 噪声大**：22 对 val 的 p@3px 单点波动可达 ±0.02-0.03，EarlyStopping patience=8 在小 val 上**容易误触发**；177 train 配 16M 参数模型也很容易在几个 epoch 内 memorize → v3 必须 `FREEZE_BACKBONE=True`（冻 9.5M backbone）才能勉强稳住。

如果想根治这两个问题，**调 split 比例（80→95%）几乎没用**（177→210，仅 +18%，且 val 缩到 11 对反而更不稳），需要走 [eloftr-cross-modal-experiments §6 路径 B](../eloftr-cross-modal-experiments/SKILL.md)：加 LLVIP（~30k 对，已对齐）/ M3FD（~4.2k 对）作为额外 train 数据，**val/test 仍保留 RoadScene 22+22 对**以保证跟 v0-v4 指标公平可比。

## 7. 最常见错误 → 定位指南

| 错误信息 | 大概率原因 | 修复入口 |
|----------|-----------|---------|
| `RuntimeError: Trying to resize storage that is not resizable` | dataset 输出形状 batch 内不一致 | 检查 `ROAD_PAD_SIZE` 是否设了，且 `bs > 1` 时所有样本都 pad 到一样的画布 |
| `Calculated padded input size per channel: (36 x 0)` | Bug A，mask 经 Homography 后非矩形 | 第 4 节 Bug A |
| `Sizes of tensors must match except in dimension 1/2. Expected size N but got size N+k` | Bug B，双维度都被裁但只 pad 了一维 | 第 4 节 Bug B |
| 训练 val `precision@3px` 高于独立 eval 很多 | HR vs LR 用错了 + pipeline 不一致 | 顶部红警 + `eloftr-eval-pipeline` skill |
| `KeyError: 'depth0'` / `'T_0to1'` | 监督函数走到了 ScanNet/MegaDepth 分支，说明 `is_aligned_irvis(data_source)` 返回了 `False` | 确认 `cfg.DATASET.TRAINVAL_DATA_SOURCE` 的小写名在 [src/utils/data_source.py](../../../src/utils/data_source.py) `ALIGNED_IRVIS_SOURCES` 里。新加数据集必须先把名字加进白名单（第 2.5 节标准 3 步） |
| 启动日志没有 `building RoadSceneDataset (dataset_name=...) from ...` | data.py short-circuit 没匹配到 | 同上，检查 `TRAINVAL_DATA_SOURCE` 拼写大小写无所谓但小写形式必须在白名单里 |
| 监督函数 `assert len(set(data['dataset_name'])) == 1` 触发 | 单 batch 里混了多个数据源 | 不要把多个 `RoadSceneDataset(dataset_name=...)` 直接 `ConcatDataset` 用 `RandomSampler`；联合训练目前未支持，参见第 2.5 节末尾 |
