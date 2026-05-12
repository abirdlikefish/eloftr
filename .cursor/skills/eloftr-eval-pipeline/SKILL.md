---
name: eloftr-eval-pipeline
description: Evaluate EfficientLoFTR checkpoints (official or finetuned v1..v7, sub-versions like v6_1, v7 input-side PC+CLAHE) on RoadScene IR-VIS or M3FD test splits, mirroring training-time validation. Use when running, writing, debugging, or interpreting an eval / validation / test script in this repo. Triggers: 模型验证脚本 / 验证脚本 / 评估脚本 / 测试脚本 / 跑测试 / 跑评估 / 测试模型 / 验证模型 / 评估 ckpt / 评估模型效果 / 选最好的 ckpt / 对比 baseline 和 finetune / 跨数据集 / in-domain / OOD / 训练 val 高于独立 eval / PC cache, English 'model validation script', 'evaluation script', 'eval bat', 'evaluate ckpt', 'compare baseline vs finetuned', 'independent eval', 'best checkpoint', 'cross-dataset eval', 'in-domain eval'; names eval_roadscene.py, eval_roadscene_official.bat, eval_roadscene_finetuned.bat, eval_m3fd_finetuned.bat, X/Y/Z bat args, -1 sentinel, sub-version v6_1 / v7_pcclahe, sub-version skip filter, precision@3px, unexpected_keys, FileNotFoundError PC cache, overall.txt / summary.csv, dump\\<dataset>_eval_v<X>_version<Y>_<topZ|last>.
---

# Evaluation Pipeline（与训练 val 对齐，支持 RoadScene + M3FD）

> 该 skill 假定数据集已经按 [eloftr-roadscene-data](../eloftr-roadscene-data/SKILL.md) 集成、模型按 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md) 训练完成。
>
> **跑出 overall.txt 之后**：把数字追加到 `results/eval_summary.md`，工作流见 [eloftr-results](../eloftr-results/SKILL.md)。本 skill 只管"产生 overall.txt"，**不管"跨版本横向对照"**——后者唯一来源是 [`results/eval_summary.md`](../../../results/eval_summary.md)。
>
> 同一份 [MyScripts/eval_roadscene.py](../../../MyScripts/eval_roadscene.py) 既能跑 RoadScene test，也能跑 M3FD test（只是名字保留了历史叫法）。两个 bat wrapper 区分评估集：
> - [MyScripts/eval_roadscene_finetuned.bat](../../../MyScripts/eval_roadscene_finetuned.bat) → 评估 RoadScene test
> - [MyScripts/eval_m3fd_finetuned.bat](../../../MyScripts/eval_m3fd_finetuned.bat) → 评估 M3FD test
>
> 两个 wrapper CLI 完全对称，因此 v1..v4（roadscene 训练）和 v5+（M3FD 训练）都可以做 in-domain 与 OOD 双向比较。

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

### 3.1 v7+ ckpt eval 前置条件（v0-v6.1 ckpt 不需要）

v7 cfg 启用 PC 边缘通道（`USE_EDGE_INPUT=True`），dataset 启动时会校验 PC 缓存目录是否存在。**推荐做法**是 eval 前预计算 PC cache（一次性，~35-40 min for M3FD+RoadScene；v10/v11 用 Megadepth_Syn 参数 `--max_long_edge 640 --pc_nscale 3` 大约 46 min）：

```bat
:: 双击 MyScripts/precompute_pc_edges.bat
::    (默认行为 = M3FD + RoadScene 双数据集 PC cache, ~35-40 min)
::    覆盖 v7/v8/v9 训练 + 双向 eval 全部需求
```

**自 2026-05-11 R8 落地后**：v7+ ckpt eval 在 PC cache 缺失时**不再硬抛**，dataset 自动 fallback 到 runtime PC 计算（仅 val/test 模式；train 仍硬抛保护训练分布）。

| eval 命令 | PC cache 需求 | 缺失时行为 |
|---|---|---|
| `eval_roadscene_finetuned.bat 1`-`6_1` | **无**（v0-v6.1 cfg 默认 `USE_EDGE_INPUT=False`，dataset 不读 PC 路径）| — |
| `eval_m3fd_finetuned.bat 1`-`6_1` | **无**（同上，即使 m3fd_trainval.py 设了 `Ir_pc/Vis_pc` 字段，dataset 的 R1 守卫保护不读）| — |
| `eval_roadscene_official.bat`（官方 ckpt） | **无**（baseline cfg 默认 USE_EDGE_INPUT=False）| — |
| **`eval_m3fd_finetuned.bat 7`** (v7 in-domain) | **M3FD PC cache** (`data/M3FD_Detection/Ir_pc/`, `Vis_pc/`)| **R8 fallback**：WARN + runtime PC（~200-400 ms/img on CPU）|
| **`eval_roadscene_finetuned.bat 7`** (v7 OOD) | **RoadScene PC cache** (`data/RoadScene/cropinfrared_pc/`, `crop_LR_visible_pc/`)| 同上 |
| **`eval_m3fd_finetuned.bat 8`** (v8 in-domain) | **M3FD PC cache**（v8 cfg 继承 v7 → USE_EDGE_INPUT=True）| 同上 |
| **`eval_roadscene_finetuned.bat 8`** (v8 OOD) | **RoadScene PC cache** | 同上 |
| **v9_e2e (in-domain / OOD)** — 用 [v9-e2e SKILL §11.4](../eloftr-v9-e2e/SKILL.md) 直接调 `eval_roadscene.py`，bat 暂未支持 X=9 | **M3FD + RoadScene PC cache 都需**（v9 cfg 自有 USE_EDGE_INPUT=True，含 v7 stack）| 同上 |
| **v10/v11 ckpt 部署到新数据集** | 新数据集 PC cache（`precompute_pc_edges --pc_nscale 3 --max_long_edge 640` + `fix_pc_cache_alignment`）| **R8 fallback**：runtime PC 路径**字节级等价** v10/v11 训练分布（实测 v10 同 3 对 cache vs runtime 差异 ≤ 1% absolute）|

### 3.2 R8 runtime PC fallback 路径（2026-05-11 加入）

实现在 [`src/utils/pc.py`](../../../src/utils/pc.py) 的 `compute_pc_v11_runtime(img_u8, target_hw)`，硬编码 v11 训练参数：

| 参数 | 值 | 来源 |
|---|---|---|
| `nscale` | 3 | `precompute_pc_edges.py --pc_nscale 3`（v10/v11） |
| `norient` | 6 | 仓库默认 `PC_NORIENT=6`，全版本一致 |
| `long_edge` | 640 | `precompute_pc_edges.py --max_long_edge 640`（v10/v11）|
| `interpolation` | `INTER_AREA` | `fix_pc_cache_alignment.py:L217` |

runtime 完整链路（4 步，与 v10/v11 cache 生成链路 byte-equivalent，除 phasepack 浮点 reproducibility 微噪声）：

```
raw uint8 (CLAHE之前) -> INTER_AREA downscale 到 640 long-edge df=32 对齐
                     -> phasecong(nscale=3, norient=6)
                     -> stretch_to_uint8（per-image max norm）
                     -> INTER_AREA resize 到 dataset 的 _resize_target_shape (480 long-edge df=32)
```

**关键约束**：

- runtime 输入必须是 **CLAHE 之前** 的 raw 灰度（与 `precompute_pc_edges.py` 读盘 raw 一致）。dataset `__getitem__` 在 CLAHE 块之前对 `ir_raw` / `vis_raw` 做 `.copy()` 快照
- **训练模式（`mode='train'`）硬抛 `FileNotFoundError`**：保护训练分布锁，runtime 永远只走测试路径
- **val/test 模式**：整目录缺失 → `__init__` WARN 一次 + 设 `_pc_runtime_fallback=True`；单图缺失 → `__getitem__` 首次 WARN 后静默
- **v7/v8/v9 ckpt 数值偏差**：v7-v9 cache 用 `nscale=4` 在 raw 上算（非 v11 参数），runtime 用 v11 参数 → PC channel 数值上有偏移；模型不会崩（实测 v5 cfg M3FD 50 对 P@1=0.462 字节级保留 v0-v9 路径），但 v7/v8/v9 ckpt 上 runtime 与 cache 的精度差距未必 ≤ 1%（v10/v11 ckpt 上 ≤ 1%）
- **v12+ 改 PC 参数时演进**：`PC_V11_LONG_EDGE` / `PC_V11_NSCALE` 当前硬编码，若做 PC ablation 提升为 cfg 字段 + dataset 读 cfg 即可（非破坏性）

## 4. Bat 脚本

| 脚本 | 评估集 | CLI |
|------|-------|-----|
| [MyScripts/eval_roadscene_official.bat](../../../MyScripts/eval_roadscene_official.bat) | RoadScene test | 无参数，跑 `weights/eloftr_outdoor.ckpt` |
| [MyScripts/eval_roadscene_finetuned.bat](../../../MyScripts/eval_roadscene_finetuned.bat) | RoadScene test | `X [Y] [Z]` |
| [MyScripts/eval_m3fd_finetuned.bat](../../../MyScripts/eval_m3fd_finetuned.bat) | M3FD test | `X [Y] [Z]` |

### 4.1 X / Y / Z 三参数（两个 finetuned wrapper 共用）

| arg | 含义 | 默认 / 哨兵 |
|-----|------|-------------|
| `X` | 版本号，支持普通整数 `1..6` 与 sub-version `6_1` 或 `6.1`（脚本内部统一归一化为 `6_1`） | 必填，无默认；`-1` 视为缺省并报错 |
| `Y` | Lightning `version_Y` 子目录编号 | 缺省 = 当前 EXP 下数值最大的 `version_N`（编号可不连续）；传 `-1` 也走默认 |
| `Z` | ckpt 选择规则 | 缺省 = `1`；`1..5` = 按文件名里 `precision@3px=` 倒序的第 N 个；`6` = `last.ckpt` |

`-1` 哨兵在三个位置都生效，方便"默认 Y + 自定义 Z"这种需求（必须 `X -1 Z`，因为是位置参数）。

无参数双击 / 直接运行时会交互提示 `Enter X [Y] [Z]`，空格或逗号分隔，未输入的位保留默认。

### 4.2 EXP 与 cfg 自动解析（含 sub-version skip filter）

```
EXP : logs\tb_logs\*_v<X>_*           # 任意 dataset 前缀；自动跳 _debug / _small
                                      # sub-version skip：tail 首字符是数字则丢弃
                                      # 例：X=6 命中 m3fd_v6_finetune，跳过 m3fd_v6_1_finetune
                                      #     X=6_1 命中 m3fd_v6_1_finetune
cfg : configs\loftr\eloftr_full_v<X>_*.py   # 同样的 sub-version skip filter
ckpt: <EXP>\version_<Y>\checkpoints\
        - Z=1..5: epoch=*precision@3px=*.ckpt 倒序第 Z 个
        - Z=6   : last.ckpt
```

完整 OUT_DIR 命名见第 4.4 节。

### 4.3 ckpt ↔ cfg 自动配对（保留对照表只为人脑核对）

由于脚本现在自动按 `eloftr_full_v<X>_*.py` glob，**用户一般不需要手填 cfg 名**。下面的对照表只用来在 `missing_keys / unexpected_keys` 异常时反查脚本是否选对了 cfg：

| X | EXP 目录 | 自动选中的 cfg |
|---|---------|----------------|
| `1` | `roadscene_v1_contrast` | `eloftr_full_v1_contrast.py` |
| `2` | `roadscene_v2_modemb` | `eloftr_full_v2_modemb.py` |
| `3` | `roadscene_v3_combined` | `eloftr_full_v3_combined.py` |
| `4` | `roadscene_v4_combined` | `eloftr_full_v4_combined.py` |
| `5` | `m3fd_v5_combined` | `eloftr_full_v5_m3fd.py` |
| `6` | `m3fd_v6_finetune` | `eloftr_full_v6_finetune.py` |
| `6_1` (= `6.1`) | `m3fd_v6_1_finetune` | `eloftr_full_v6_1_finetune.py` |
| `7` | `m3fd_v7_pcclahe` | `eloftr_full_v7_pcclahe.py`（PC + CLAHE input-side; 需要 PC cache 已生成） |
| `8` | `m3fd_v8_msbn` | `eloftr_full_v8_msbn.py`（v7 stack + MSBN）|
| **v9_e2e** | `m3fd_v9_e2e_outdoor` | `eloftr_full_v9_e2e.py`（**继承 v0 baseline + 显式 opt-in 全 v1-v8 stack**, 不继承 v8 cfg）— bat 暂未支持 X=9，详见 [eloftr-v9-e2e §11.4](../eloftr-v9-e2e/SKILL.md) 直接调 `eval_roadscene.py` |
| baseline (官方) | — | `eloftr_full.py`（仅 `eval_roadscene_official.bat` 走这条） |

加载完仍会打印 `missing_keys / unexpected_keys`，理想状态：
- 官方 ckpt + baseline cfg：两者都为空。
- 官方 ckpt + v2/v3/... cfg：`missing_keys = ['modality_emb_ir', 'modality_emb_vis']`（modemb 用初值），`unexpected_keys = []`。
- finetuned v_x ckpt + 自动选中的 v_x cfg：两者都为空。

如果出现额外 key，先看 echo header 里的 `Cfg` / `Ckpt` 行确认 sub-version skip filter 没把对的 cfg 误丢。

### 4.4 OUT_DIR 命名规则（统一）

```
RoadScene eval : dump\roadscene_eval_v<X>_version<Y>_<topZ|last>
M3FD eval      : dump\m3fd_eval_v<X>_version<Y>_<topZ|last>
```

`<X>` 是归一化后的下划线形式（`6_1`），`<topZ|last>` 由 Z 决定：`Z=1..5 → top1..top5`，`Z=6 → last`。

样例：

| 命令 | OUT_DIR |
|------|---------|
| `eval_roadscene_finetuned.bat 3` | `dump\roadscene_eval_v3_version3_top1` |
| `eval_roadscene_finetuned.bat 5` (cross-dataset, v5 是 M3FD-trained) | `dump\roadscene_eval_v5_version0_top1` |
| `eval_m3fd_finetuned.bat 5` (in-domain) | `dump\m3fd_eval_v5_version0_top1` |
| `eval_m3fd_finetuned.bat 4` (cross-dataset, v4 是 RoadScene-trained) | `dump\m3fd_eval_v4_version0_top1` |
| `eval_m3fd_finetuned.bat 6_1` | `dump\m3fd_eval_v6_1_version0_top1` |
| `eval_m3fd_finetuned.bat 6.1 -1 6` | `dump\m3fd_eval_v6_1_version0_last` |

> 代价：从 dump 目录名看不出"训练 dataset"，但通过 X 与 v1..v4=roadscene / v5+=m3fd 的约定可推；想严格确认时打开目录里的 `overall.txt`，最上面会写 ckpt 路径。

### 4.5 历史目录兼容

旧命名（已不再产出）保留在磁盘上不动：

| 旧命名 | 新命名 |
|--------|--------|
| `dump\roadscene_eval_v3_combined_version3_top1` | `dump\roadscene_eval_v3_version3_top1` |
| `dump\roadscene_eval_m3fd_v5_combined_version0_top1` | `dump\roadscene_eval_v5_version0_top1` |
| `dump\m3fd_eval_v5_combined_version0_top1` | `dump\m3fd_eval_v5_version0_top1` |
| `dump\m3fd_eval_roadscene_v2_modemb_version1_top1` | `dump\m3fd_eval_v2_version1_top1` |

要重新跑老结果直接重跑新版 bat 即可，不要手动改老目录的名字（避免和未来其它实验撞车）。

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
| `unexpected_keys` 里出现 `modality_emb_ir/vis` | 自动选到的 cfg 不是训练用的 v_x（多半是 sub-version skip filter 误吃） | echo header 里 `Cfg` 行 + 第 4.3 节对照表 |
| `missing_keys` 里出现 backbone 或 transformer 层 | cfg arch 与 ckpt arch 不同 | 检查 cfg 继承链是否完整 |
| 独立 eval 远低于训练 val | 误用了旧版 eval 或 pipeline 不一致 | 先 `git log MyScripts/eval_roadscene.py` 确认是新版 |
| 独立 eval 远高于训练 val | `--apply_homography` 未开但训练时 H 增强是开的（数据无难度）；或者用了 `crop_HR_visible` | 第 5.4 节 + HR/LR |
| `overall.txt` 中 `total_matches` 很小 | `--thr` 设高了 / 模型确实退化 | 把 `--thr` 调回训练的 0.1，再看 |
| 终端 p@3px 抖到 0% | per-pair 单点抖动 | 改看 `overall.txt`，per-match 才是稳定指标 |
| `[ERROR] no final experiment found for v6 under logs\tb_logs\*_v6_*` | sub-version skip 把唯一目录也滤掉了（极少见，比如目录叫 `xxx_v6_2hidden_finetune`） | 临时把 X 改成全名（如 `6_2hidden`），或重命名实验目录 |
| `eval_roadscene_finetuned.bat 6` 与 `6_1` 看起来跑了同一个 ckpt | bat 是旧版（没有 sub-version skip filter） | echo header 里 `Experiment` 行应分别是 `m3fd_v6_finetune` 与 `m3fd_v6_1_finetune` |
| `[WARN] multiple cfgs match eloftr_full_vX_*.py` 触发但选错了 | 同一 X 下有多个 non-subver cfg（比如 `_combined` 和 `_finetune` 都属 v5） | 看 WARN 列出的所有候选，删掉不需要的 cfg，或把脚本里 first-match 的逻辑收紧 |
| `RuntimeError: size mismatch for backbone.layer0.rbr_dense.conv.weight ((64,1,3,3) vs (64,2,3,3))` 在 eval 时触发 | v7 cfg 强制 backbone in_ch=2，但传入的 ckpt 是 v0-v6.1 的 in_ch=1，且 `eval_roadscene.py` 的 `build_matcher` **不**调用 `_maybe_inflate_stage0` hook（那个 hook 只在训练时 lightning_loftr.py 内部触发）| 用对应的 cfg：v7 ckpt 用 `eval_*finetuned.bat 7`；v0-v6.1 ckpt 用 `eval_*finetuned.bat 1`-`6_1`，绝不用 X=7 + v0-v6.1 ckpt |
| `FileNotFoundError: PC cache directory not found in train mode` | v7+ cfg + `mode='train'` 时 PC 目录缺失（R8 train 硬阻断，保护训练分布）| 跑一次 `MyScripts/precompute_pc_edges.bat` / `.sh`，详见 §3.1。eval 永远不该见到这条 error，因为 R8 在 val/test 下走 fallback |
| 看到 WARN `PC cache dir missing ... will compute PC on-the-fly` 但 eval 正常出 overall.txt | R8 runtime PC fallback 触发（PC 目录缺失但 v7+ cfg 启用 `USE_EDGE_INPUT=True`）| **预期行为**，详见 §3.2。要消除 WARN 只需要预计算 cache。v10/v11 ckpt 上数值字节级等价；v7/v8/v9 ckpt 上数值有偏移（cache nscale=4 vs runtime nscale=3），建议预计算 |
| OOD（RoadScene）`total_matches` 数远小于 in-domain（M3FD），如 33K vs 463K (≈14× 差距) | **不是模型问题，是数据集差异 × pair 数差的复合**：14× = 9.5× (210 vs 22 pair) × 1.44× (每对 matches 差)。每对 matches 差距来自 RoadScene 原图低质量 (~500×329 jpg ~22KB) + 道路场景空区多，所有 v 版本（v7/v8/v9）的 in/OOD matches/pair 比例都是 1.41-1.45×，差距 < 3% | **不需要修**，是物理特性。详见 [eloftr-v9-e2e §9 Q2](../eloftr-v9-e2e/SKILL.md)；要进一步排除"模型偏见"嫌疑，对比同 X 在两个数据集 `overall.txt` 的 `total_matches` / pair 数比例，应都在 1.4× 附近 |
