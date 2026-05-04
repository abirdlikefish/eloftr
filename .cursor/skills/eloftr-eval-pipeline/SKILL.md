---
name: eloftr-eval-pipeline
description: Evaluate EfficientLoFTR checkpoints (official or finetuned v1/v2/v3) on RoadScene IR-VIS in a way that exactly matches training-time validation. Use when the user mentions eval_roadscene, independent eval, "training val higher than independent eval", precision@3px discrepancy, reparameter, overall.txt / summary.csv, official vs finetuned ckpt evaluation, or wants to fairly compare baseline vs finetuned numbers.
---

# RoadScene Evaluation Pipeline（与训练 val 对齐）

> 该 skill 假定数据集已经按 [eloftr-roadscene-data](../eloftr-roadscene-data/SKILL.md) 集成、模型按 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md) 训练完成。

## 0. 为什么必须有专门的 eval skill

历史上有过很大坑：训练阶段 `val precision@3px` 报到 0.52，独立 `eval_roadscene.py` 用同一个 ckpt、同一个 split 只跑出 0.34，差距 ~50%。原因不是模型差，而是 **两条管线在 dataset / mask / 优化路径 / RepVGG 模式 / BORDER_RM 上全都不一样**。

修复方法：把独立 eval 重写成**完全镜像**训练 val 的实现。本 skill 就是那条新管线的使用说明 + 对比表 + 结果解读。

## 1. 两条管线的关键差异（修复前 vs 修复后）

| 维度 | 训练时 validation | 旧 eval（被废弃） | **新 eval（当前实现）** |
|------|------------------|------------------|----------------------|
| DataLoader batch size | 1（PL 强制） | 1 | 1 |
| Resize 策略 | IR / VIS 各自独立 resize | 联合 resize 到同一目标 | 与训练一致：独立 resize |
| Padding | bottom-right pad 到 `(PAD_SIZE, PAD_SIZE)` | 不 pad | 与训练一致：bottom-right pad |
| mask0/mask1 | 来自 `RoadSceneDataset` | 不传，模型走 `mask is None` 分支 | 来自 `RoadSceneDataset` |
| `bs==1` cropping 优化 | 触发（mask 不为 None） | **不触发**（无 mask） | 触发 |
| RepVGG 模式 | multi-branch（训练态） | `reparameter()` deploy 单分支 | multi-branch（不调 `reparameter`） |
| `coarse_matching.BORDER_RM` 影响 | 落在 padding 区，等价无效 | 直接吃掉真实图像边界 | 落在 padding 区，等价无效 |
| MP / HALF | `--disable_mp` ⇒ `MP=False, HALF=False` | 自由 | **强制** `MP=False, HALF=False` |
| 模型 config | yacs，含训练用的所有 v_x extras | 硬编码，可能丢 modemb | yacs，与训练时同 main_cfg |
| 像素误差计算 | `_compute_roadscene_metrics` 用 `homography_0to1` warp | 类似但路径不同 | 与训练一致（直接 mirror 实现） |
| precision@Npx 聚合 | per-match（所有 pair 的 errs 拼起来再算 mean） | per-pair 平均 | per-match（与训练一致） |

把旧 eval 当成 “模型 + 不同的预处理 + 不同的优化路径”，所以 “它给出的 0.34” 根本不是同一函数的取值，跟训练 val 不能比。

## 2. 当前实现：[MyScripts/eval_roadscene.py](../../../MyScripts/eval_roadscene.py)

核心设计（参考脚本顶部 docstring 与 `build_config` / `build_dataset` / `build_matcher`）：

1. **复用 yacs config**：`get_cfg_defaults()` → `merge_from_file(args.main_cfg)` → `merge_from_file(args.data_cfg)`，等价于训练 `train.py` 在第 90 行附近做的事。这样 `USE_MODALITY_EMB / USE_CONTRASTIVE / NPE / DSMAX_TEMPERATURE / MATCH_COARSE.THR` 全部按训练时的值。
2. **强制 MP/HALF=False + 缺省 NPE**：见 [MyScripts/eval_roadscene.py:111-122](../../../MyScripts/eval_roadscene.py)。`disable_mp` 总是被训练命令传，所以这里也强制；`NPE` 默认 `None`，要打补丁成 `[832]*4` 不然 `LoFTR.__init__` 会 assert。
3. **`RoadSceneDataset(mode='val')` + `DataLoader(bs=1)`**：与训练 val 完全一致；`coarse_scale = 1 / cfg.LOFTR.RESOLUTION[0]` 直接读 config，不再硬编码 `1/8`。
4. **不调用 `reparameter()`**：保持 RepVGG 多分支训练态，与训练 val 算的同一组数；想看推理速度时再单独跑。
5. **`_strip_matcher_prefix`**：PL 训练保存的 ckpt 每个 LoFTR 参数都带 `matcher.` 前缀，官方 `eloftr_outdoor.ckpt` 不带。脚本能自动识别，不需要外部转换。
6. **像素误差**：`_compute_pair_pixel_errs` 直接镜像 [src/lightning/lightning_loftr.py:184-235](../../../src/lightning/lightning_loftr.py) 的 `_compute_roadscene_metrics`，复用 `batch['homography_0to1']`。
7. **聚合**：`overall.txt` 用 per-match (`np.concatenate(all_errs)` 后取 `< t).mean()`，与 `_aggregate_roadscene_metrics` 一致。`summary.csv` 仍按 per-pair 写出，仅供检阅个例。

可视化：`_save_pair_figure` 会把 padded canvas crop 到 mask 的 bbox 再画，避免 padding 黑边占满画面。

## 3. CLI 速查

```bat
python MyScripts\eval_roadscene.py ^
  --ckpt <path>                  REM 必传：ckpt 路径（官方或 PL 训练）
  --out_dir dump\xxx             REM 必传：输出目录（含 summary.csv / overall.txt / *_match.png）
  --main_cfg configs\loftr\<...>.py   REM 默认 eloftr_full.py；finetuned 必须传训练用的 main_cfg
  --data_cfg configs\data\roadscene_trainval.py   REM 默认即此值
  --list_path data\RoadScene\index\test_pairs.txt REM 默认走 data_cfg.TEST_LIST_PATH
  --thr 0.1                      REM 覆盖 LOFTR.MATCH_COARSE.THR；finetune 训练用 0.1
  --apply_homography             REM 切 mode='train' 让随机 H 增强生效，给鲁棒性测试用
  --seed 123                     REM 仅 --apply_homography 时影响
  --max_pairs 0                  REM 0 = 全部
```

废弃 / 危险 flag：
- `--img_resize` / `--df`：旧版有，已删，全部由 yacs 提供，确保与训练 numerics 一致。
- 想换 HR/LR：用 `--vis_subdir crop_HR_visible` 临时覆盖，**只在做 “HR 是否影响很大” 的对比时用**，长期请改 config。

## 4. Bat 脚本

| 脚本 | 用途 | 必改字段 |
|------|------|---------|
| [MyScripts/eval_roadscene_official.bat](../../../MyScripts/eval_roadscene_official.bat) | 跑官方 `weights/eloftr_outdoor.ckpt` | 默认开箱即用；想换 split 改 `--list_path` |
| [MyScripts/eval_roadscene_finetuned.bat](../../../MyScripts/eval_roadscene_finetuned.bat) | 跑 v1/v2/v3 finetuned ckpt | `FT_CKPT` + **`FT_CFG`**（必须与训练 main_cfg 一致） |

`eval_roadscene_finetuned.bat` 顶部的对照表必须遵守，否则会发生 “v3 ckpt 用 baseline cfg 加载 → modemb 参数被识别为 unexpected_keys 而丢弃 → 评估的等价于半个 baseline 模型”：

| ckpt | 必须传的 main_cfg |
|------|-------------------|
| baseline (v0) | `configs/loftr/eloftr_full.py` |
| v1 contrast   | `configs/loftr/eloftr_full_v1_contrast.py` |
| v2 modemb     | `configs/loftr/eloftr_full_v2_modemb.py` |
| v3 combined   | `configs/loftr/eloftr_full_v3_combined.py` |

加载完会打印 `missing_keys / unexpected_keys`，理想状态：
- 官方 ckpt + baseline cfg：两者都为空。
- 官方 ckpt + v2/v3 cfg：`missing_keys = ['modality_emb_ir', 'modality_emb_vis']`（modemb 用初值），`unexpected_keys = []`。
- finetuned v_x ckpt + 对应 v_x cfg：两者都为空。
出现额外 key 时立刻停下来核对 cfg 与 ckpt 的对应关系。

## 5. 结果解读

### 5.1 关注 `overall.txt`，不是终端 per-pair 行

终端每行 `[i/N] FLIR_xxx.jpg: matches=... mpe=... p@3px=XX%` 是 per-pair 数字，22 张样本 + 每对 100 ~ 1000 个 match，单 pair `p@3px` 抖到 0/100% 都正常。

`overall.txt` 写出的 `precision@1/3/5px` 才是与训练 val 同口径（per-match 平均）。

### 5.2 样本量噪声

22 张 val pair 上做 per-match 聚合，根据典型 `p@3px ≈ 0.4-0.55` 推算 σ≈13%，标准误 SE≈2.8%。
> **结论：finetuned 与 baseline 的 `p@3px` 差距 < 5% 时，要小心是否在噪声范围内。** 多次重训 + 不同 seed eval 能给信心区间。

### 5.3 finetune 是否有效的判断标准

必须用同一份 `eval_roadscene.py` + 同一份 `--list_path` + 同一份 `--main_cfg`，分别跑 baseline 和 finetuned，比较 `overall.txt`。绝不能拿训练曲线里的 `precision@3px` 直接跟独立 eval 数字比，因为：
- 训练 val 走的是 train-time validation 路径（multi-branch RepVGG + cropping 优化 + masks）。
- 独立 eval 现在虽然也走同样路径，但 random seed / data ordering 可能不同；只比同一脚本的产物。

### 5.4 物理上限提醒

RoadScene 的 IR/VIS 标定本身就有 2-5 px 的对齐残差，因此 `overall.txt` 中 `precision@3px` 的天花板大概在 60-70%。看到训练 val 飙到 80%+ 第一反应是检查 pipeline（HR vs LR、mask 是否生效、是否走错 dispatch），而不是欢呼模型变神。

## 6. 常见 “数字看着不合理” 的检查清单

| 现象 | 大概率原因 | 应检查 |
|------|-----------|--------|
| `unexpected_keys` 里出现 `modality_emb_ir/vis` | `--main_cfg` 没指向训练用的 v_x | 第 4 节 ckpt-cfg 对应表 |
| `missing_keys` 里出现 backbone 或 transformer 层 | `--main_cfg` arch 与 ckpt arch 不同 | 检查继承链是否完整 |
| 独立 eval 远低于训练 val | 误用了旧版 eval 或 pipeline 不一致 | 先 `git log MyScripts/eval_roadscene.py` 确认是新版 |
| 独立 eval 远高于训练 val | `--apply_homography` 未开但训练时 H 增强是开的（数据无难度）；或者用了 `crop_HR_visible` | 第 5.4 节 + HR/LR |
| `overall.txt` 中 `total_matches` 很小 | `--thr` 设高了 / 模型确实退化 | 把 `--thr` 调回训练的 0.1，再看 |
| 终端 p@3px 抖到 0% | per-pair 单点抖动 | 改看 `overall.txt`，per-match 才是稳定指标 |
