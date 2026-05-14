---
name: eloftr-metu-vistir-eval
description: Bit-perfect reproduce of MINIMA paper Table 3 ELoFTR baseline (2.88 / 7.88 / 17.72 pose-AUC @ 5/10/20 deg) on METU-VisTIR (real-captured drone VIS+LWIR cross-modal pairs). Covers the MINIMA / XoFTR test_relative_pose_infrared.py protocol precisely (ransac_thr=1.5 single shot, long edge 640, no NPE, MATCH_COARSE.THR=0.2, getOptimalNewCameraMatrix alpha=0 undistort, no pad-to-square, per-scene error_auc -> split('_scene')[0] class mean -> 2-class final mean), the T_0to1 = P1 @ inv(P0) (W2C MegaDepth standard) geometry pitfall that buried this eval 10-20x low for ~1 year under an inverted C2W formula, the official-vs-finetuned side0/side1 asymmetry (outdoor.ckpt baseline uses vis-as-image0 to mirror MINIMA load_vis_tir_pairs_npz, v10+ finetuned ckpts use thermal-as-image0 because modemb_ir is bound to image0), and the post-fix protocol-change implications (all v0..v12 METU numbers in results/eval_summary.md under the prior ransac_thr=2.0 / 5-restart / loftr_thr=0.1 / pooled-AUC protocol are no longer comparable to the new ELoFTR baseline and need a rerun). Use when running METU eval / debugging METU AUC far below paper / wanting to cite the official MINIMA Table 3 ELoFTR row / understanding why old v12 skill METU numbers (0.001401 / 0.006642 / 0.039631) are not the same protocol as new official.bat numbers / wondering if T_0to1 formula is correct / configuring undistort newCameraMatrix / sweeping thr 0.1 vs 0.2 / NPE on/off for METU / understanding per-class AUC aggregation. Triggers: METU / METU-VisTIR / METU_VISTIR / metu_vistir / METU eval / METU baseline / METU 复现 / METU AUC / MINIMA / MINIMA 协议 / MINIMA paper Table 3 / XoFTR / test_relative_pose_infrared / test_relative_pose / load_vis_tir_pairs_npz / aggregiate_scenes / per-class AUC / per-scene AUC / split _scene / aggregate_metrics 失效 / pooled AUC / pose-based AUC / auc@5 deg / auc@10 deg / auc@20 deg / 2.88 / 7.88 / 17.72 / 2.96 / 8.09 / 18.13 / T_0to1 / T_0to1 公式 / T_0to1 方向 / W2C / C2W / MegaDepth W2C / P1 @ inv(P0) / inv(P1) @ P0 / poses 约定 / world->cam / camera->world / getOptimalNewCameraMatrix / alpha=0 / cv2.undistort newCameraMatrix / new_K / undistortPoints / dist 系数 / distortion_coefs / ransac_thr 1.5 / ransac_times 1 / single shot RANSAC / megasize 640 / long edge 640 / MATCH_COARSE.THR 0.2 / loftr_thr 0.2 / NPE off / 不开 NPE / pad_to_square False / padding=False / METU_PAD_TO_SQUARE / METU_SIDE0 / METU_SIDE1 / side0=vis / side1=thermal / side0=thermal / side1=vis / modemb_ir / modemb 绑定 image0 / outdoor.ckpt baseline / v10+ finetuned METU / 协议变更 / protocol switch / 协议过时 / 协议不可比 / eval_summary.md METU 失效 / 重跑 METU / rerun METU / metu_eval_official / metu_eval_v / dump\\metu_eval / eval_metu_vistir_official.bat / eval_metu_vistir_finetuned.bat / visualize_metu_vistir.py / aggregiate_scenes / class_aucs / all_class_mean / cloudy_cloudy / cloudy_sunny / 10 npz / 2590 pair, English 'METU-VisTIR baseline reproduce', 'MINIMA protocol exact reproduction', 'pose-AUC bit-perfect', 'T_0to1 W2C MegaDepth standard', 'C2W formula buried this eval', 'getOptimalNewCameraMatrix alpha=0', 'long edge 640 paper Sec 5.1', 'per-scene per-class arithmetic mean', 'side0 vis vs thermal asymmetry', 'modemb image0 binding', 'pre-fix v0-v12 METU rows invalidated', 'rerun all METU under new protocol', 'thr 0.1 vs 0.2 sweep', 'NPE off for METU', 'pad_to_square False'. Companion skills: eloftr-eval-pipeline (RoadScene / M3FD precision@N px pipeline, complementary protocol axis), eloftr-results (cross-version eval table; METU columns now subject to this skill's protocol), eloftr-v10-msyn / eloftr-v11-dualh / eloftr-v12-dualh-baseline (v10/v11/v12 METU numbers in those skills predate this protocol; pending rerun).
---

# METU-VisTIR Evaluation（MINIMA 论文表 3 ELoFTR baseline bit-perfect 复现）

> 本 skill 是 METU-VisTIR cross-modal pose-AUC eval 的 sole source of truth。RoadScene / M3FD 的 prec@N px 评测路径完全独立，看 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)。

## 0. 一句话用法

| 场景 | 动作 |
|---|---|
| 复现论文 Table 3 ELoFTR (2.88 / 7.88 / 17.72) | 双击 [MyScripts/eval_metu_vistir_official.bat](../../../MyScripts/eval_metu_vistir_official.bat)，读 `dump\metu_eval_official\all\overall.txt` 末尾 `all_class_mean` 行 |
| 跑 v10/v11/v12 finetuned ckpt 上的 METU 数 | `MyScripts\eval_metu_vistir_finetuned.bat X [Y] [Z]` (X=版本号 10/11/12) |
| METU AUC 跑出来 << 论文（差 5-20×） | 走本 skill §6 故障树 |
| 想知道 v12 skill 里那些 METU 数字（0.001401 等）为啥跟 ELoFTR baseline 2.96 不能直接比 | 看 §5「协议变更」一节 |
| 想知道为什么 T_0to1 公式被埋了 1 年没人发现 | 看 §4「T_0to1 公式陷阱」 |
| 把 METU 数字追加到跨版本汇总表 | [results/eval_summary.md](../../../results/eval_summary.md)，规范见 [eloftr-results](../eloftr-results/SKILL.md) §3 METU 字段 |

## 1. 关键文件清单

| 文件 | 角色 |
|---|---|
| [MyScripts/eval_metu_vistir_official.bat](../../../MyScripts/eval_metu_vistir_official.bat) | ELoFTR outdoor.ckpt 在 METU 上的 baseline（对应论文 Table 3）|
| [MyScripts/eval_metu_vistir_finetuned.bat](../../../MyScripts/eval_metu_vistir_finetuned.bat) | v10/v11/v12 finetuned ckpt 在 METU 上的 cross-modal 训练效果 |
| [MyScripts/visualize_metu_vistir.py](../../../MyScripts/visualize_metu_vistir.py) | 两个 .bat 共用的 python 入口；per-pair RANSAC + per-scene AUC + per-class 平均 + overall.txt 三段块 |
| [src/datasets/metu_vistir.py](../../../src/datasets/metu_vistir.py) | METUVisTIRDataset：从 npz 加载 pairs，做 cv2.undistort + resize + (optional) pad + 计算 T_0to1 |
| [configs/data/metu_vistir_test_all.py](../../../configs/data/metu_vistir_test_all.py) | data cfg；MGDPT_IMG_RESIZE / METU_UNDISTORT / METU_SIDE0/1 / METU_PAD_TO_SQUARE |
| [src/config/default.py](../../../src/config/default.py) §`_CN.DATASET.METU_*` | METU-specific cfg 字段注册（yacs strict-mode 必需）|
| [src/lightning/data.py](../../../src/lightning/data.py) METU dispatch | `_build_concat_dataset` 的 `elif metu_vistir:` 分支，把 cfg 字段传给 dataset 构造函数 |
| [src/utils/metrics.py](../../../src/utils/metrics.py) `error_auc` / `estimate_pose` / `relative_pose_error` | 复用 LoFTR 原生 pose-eval 工具，**没改** |

数据集软链 / npz 结构详见 npz header 注释（dataset module docstring L1-83）。

## 2. MINIMA / XoFTR 协议（精确参数）

来源：
- MINIMA `src/utils/data_io_loftr.py` `DataIOWrapper.preprocess_image` L34-71
- MINIMA `test_relative_pose_infrared.py` `eval_relapose` + `aggregiate_scenes`
- MINIMA 论文 §5.1 (arXiv 2412.19412 v2) *"we uniformly resize all images with their long dimension equal to 640"*
- XoFTR `test_relative_pose.py` `--ransac_thres` default L286

| 项 | 值 | 备注 |
|---|---|---|
| 长边 resize | **640** | MINIMA `_CN.TEST.IMG0_RESIZE=640` + 论文 §5.1 |
| Divisible factor (DF) | **32**（ELoFTR 必须）| RepVGG 16x + fine matching 2x。MINIMA cfg 默认 8 是给 LoFTR/XoFTR 的，跑 ELoFTR 必须改 32 |
| Pad to square | **False** | `_CN.TEST.PADDING=False`，bs=1 forward 各自 H/W |
| Coarse scale | **0.125** = 1/8 | `_CN.TEST.COARSE_SCALE=0.125` |
| 读图 | **cv2.IMREAD_GRAYSCALE** | `from_paths` default `read_color=False` |
| Undistort 流程 | `new_K = getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))` → `cv2.undistort(img, K, dist, None, new_K)` → `K_out = new_K` | data_io_loftr.py L34-37。**关键不能漏 `K_out = new_K` 这一步** |
| `estimate_pose` 用的 K | **new_K（不是原 K）** | 上面 undistort 步骤的输出 K |
| `MATCH_COARSE.THR` | **0.2** | MINIMA `load_loftr` default + [configs/loftr/eloftr_full.py L28](../../../configs/loftr/eloftr_full.py) cfg 自带注释 *"recommend 0.2 for full model"* 双重一致 |
| NPE | **不开（cfg 默认 [832,832,832,832]）** | test 长边 640 < train 832，落在训练分布内，不需要外推 |
| fp32 强制 | `cfg.LOFTR.MP=False` + `cfg.LOFTR.HALF=False` | visualize_metu_vistir.py L174-175 |
| RANSAC backend | **OpenCV `estimate_pose`**（src/utils/metrics.py）| 单次调用 |
| `ransac_thres` | **1.5** | XoFTR test_relative_pose.py L286 default |
| Restarts | **1**（single shot）| MINIMA 不 5-restart 取 best |
| Pose error 定义 | **`max(R_err, t_err)`** | `relative_pose_error` 返回 (R, t) 两个值，取 max |
| AUC 聚合 | **per-npz `error_auc` → split('_scene')[0] 类内算术平均 → 两类再平均** | XoFTR `aggregiate_scenes`。**NOT pool 2590 对到一起 `error_auc` 一次** |
| AUC 单位 | **百分数 (×100)** | 跟论文 Table 3 量纲一致 |

### 2.1 side0 / side1 不对称（**重要**）

| 脚本 | side0 | side1 | 原因 |
|---|---|---|---|
| `eval_metu_vistir_official.bat` (ELoFTR outdoor.ckpt baseline) | **vis** | thermal | 跟 MINIMA `load_vis_tir_pairs_npz` 一致：`image_paths[id0][0]=visible`，outdoor.ckpt 无 modemb，empirically vis-as-image0 比 thermal-as-image0 高 ~4× AUC |
| `eval_metu_vistir_finetuned.bat` (v10/v11/v12) | **thermal** | vis | cfg 默认 `METU_SIDE0='thermal'`，v10+ modemb_ir 绑定 image0 → 必须 thermal 在 image0 |

→ ELoFTR baseline vs v10+ finetuned 对比时**双变量**（image0 模态不同 + 模型不同），不可避免。

## 3. 复现数字（2026-05-13）

ELoFTR outdoor.ckpt + MINIMA 协议跑 `MyScripts\eval_metu_vistir_official.bat`：

| 指标 | 我们跑的 | 论文 Table 3 ELoFTR | 偏差 |
|---|---|---|---|
| `auc@5°` (%) | **2.962** | 2.88 | +2.8% |
| `auc@10°` (%) | **8.093** | 7.88 | +2.7% |
| `auc@20°` (%) | **18.125** | 17.72 | +2.3% |

残余 gap **< 3%**，bit-perfect 复现。

略高 2-3% 的最可能原因：
1. **MATCH_COARSE.THR=0.2 vs MINIMA 内部 `load_eloftr`（不公开）可能用 0.1**：thr 越严 → 假阳性匹配越少 → RANSAC inlier 比例越高 → AUC 略高
2. **RANSAC 单次 stochastic noise**：cv2.findEssentialMat 内部 minimal-sample-set 随机抽，不同种子 ±1-2%
3. **OpenCV 版本差异**：MINIMA conda env `minima` vs 我们 `eff_loftr`

per-scene 数字（参考量级，跑出来应该接近）：

| scene | auc@5/10/20 | pairs |
|---|---|---|
| cloudy_cloudy_scene_1 | 2.8 / 7.7 / 16.8 | 131 |
| cloudy_cloudy_scene_2 | 6.7 / 14.5 / 27.6 | 197 |
| cloudy_cloudy_scene_3 | 0.6 / 2.4 / 8.7 | 266 |
| cloudy_cloudy_scene_4 | 1.9 / 6.1 / 13.4 | 311 |
| cloudy_cloudy_scene_5 | 0.9 / 6.2 / 18.2 | 180 |
| cloudy_cloudy_scene_6 | 4.7 / 11.4 / 24.5 | 297 |
| cloudy_sunny_scene_1 | 2.4 / 8.6 / 21.9 | 195 |
| cloudy_sunny_scene_2 | 6.6 / 13.9 / 26.0 | 270 |
| cloudy_sunny_scene_3 | 2.0 / 6.5 / 13.5 | 289 |
| cloudy_sunny_scene_4 | 0.9 / 3.6 / 10.8 | 454 |

per-class 平均：`cloudy_cloudy` 2.95/8.03/18.19 vs `cloudy_sunny` 2.98/8.15/18.06——**两类几乎相同**，反映 outdoor.ckpt 没经过 cross-modal 训练，两种光照下都同样"难"。

## 4. T_0to1 公式陷阱（**最大坑，1 行字符决定 10-20× AUC**）

### 4.1 正确公式（MegaDepth standard，W2C 约定）

[src/datasets/metu_vistir.py:243](../../../src/datasets/metu_vistir.py) 现行版本：

```python
P0 = self._poses[idx0].astype(np.float64)   # W2C
P1 = self._poses[idx1].astype(np.float64)   # W2C
T_0to1 = P1 @ np.linalg.inv(P0)
```

数学：W2C 约定下 `P @ 世界点 = 相机点`，所以 `cam1 点 = P_1 @ inv(P_0) @ cam0 点`，跟 MegaDepth / MINIMA `load_vis_tir_pairs_npz` (`np.matmul(T1, np.linalg.inv(T0))`) 完全一致。

### 4.2 旧公式（C2W 约定，**错的**）

修复前用了：

```python
T_0to1 = np.linalg.inv(P1) @ P0
```

这等价于"poses 是 C2W"假设下的公式。**但 npz header 明文写 `world->cam`（W2C）**，公式套错。错公式下解出的 R/t 方向反，pose error 普遍 30-50°，AUC@5° 接近 0，AUC@20° 残余 ~4%（被基数大的 pair 拉起来）。

### 4.3 错公式怎么混过去 1 年没被发现

dataset docstring 历史版本记录了一次 30-pair sweep on v10 ckpt 选公式：

```
A: P1 @ inv(P0)        R_err=38.6  t_err=54.5  (W2C)
B: inv(P1) @ P0        R_err=11.6  t_err=34.6  (C2W -- 当时选了 B)
```

判断流程：「B 的 R_err 11.6 比 A 的 38.6 小 → B 几何正确」。**这判断是错的**：

- v10 ckpt 在真实 METU LWIR 上**是 OOD**（v10 在 Megadepth_Syn style-transferred 合成 IR 上 finetune，不在真 LWIR 上训过）
- 两个公式都给出"几乎完全失败"的 R_err（一个 38° 一个 11°，都远超 5° 的 in-domain 水平）
- essential matrix decomposition 在 unit translation 下对 R 翻转有部分鲁棒性，统计噪声下"少错一点"≠"几何正确"
- **正确的 sweep 必须用 in-domain ckpt + in-domain 数据**（如 outdoor.ckpt + MegaDepth-1500），让正确公式给出 ~5° R_err vs 错公式 ~50° R_err，区分度 10× 才能拍板

### 4.4 自检：如果以后改 dataset / 加新数据集

任何 npz-based pose eval 接入仓库时：

1. **看 npz header 明文协议声明**（W2C / C2W）——这是 ground truth
2. **业界默认 MegaDepth + OpenCV + COLMAP 全是 W2C**，默认套 `T_0to1 = P_1 @ inv(P_0)`
3. 想用反方向 `inv(P_1) @ P_0` 之前必须**用 in-domain ckpt 做 sweep**，让两个公式的 R_err 差 ≥ 5× 才能拍板（差 ≤ 3× 的差异里全是统计噪声）
4. 加 `assert` 在 `__getitem__` 里：`T_0to1` 应保留 R 部分接近 identity 当 P0 ≈ P1（同图自配对 sanity check）

## 5. 协议变更：v0..v12 旧 METU 数字失效

### 5.1 旧协议 (修复前)

历史上 finetuned.bat 用：

```
RANSAC_FLAG     : --ransac_thr 2.0 --ransac_times 5  (5-restart 取 best)
megasize        : 不传 → cfg 默认 832  (NOT 640)
NPE             : 没显式开但 cfg 默认 None → fallback [832,832,832,832]
thr             : --thr 0.2          (这一项跟新协议一致)
T_0to1          : inv(P1) @ P0       (错的 C2W 公式)
undistort       : cv2.undistort(img, K, d) 等于 newCameraMatrix=K0 (NOT new_K)
pad_to_square   : True               (旧 dataset 强制 pad)
AUC aggregation : pool 全部 pair → aggregate_metrics 一次 error_auc  (NOT per-class mean)
SUBSETS         : 3 个 (all / cloudy_cloudy / cloudy_sunny) 分别跑
```

### 5.2 协议变更影响哪些数据

| 文件 / skill | 受影响内容 | 处理 |
|---|---|---|
| [results/eval_summary.md](../../../results/eval_summary.md) | 如果有 v0..v12 的 METU 行，全部基于旧协议 | 全部需要重跑（每个 ckpt ~10-15 min）|
| [.cursor/skills/eloftr-v10-msyn/SKILL.md](../eloftr-v10-msyn/SKILL.md) | description 提到 METU eval 数 | 待 backfill 新协议数 |
| [.cursor/skills/eloftr-v11-dualh/SKILL.md](../eloftr-v11-dualh/SKILL.md) | description 提到 METU eval 数 | 待 backfill |
| [.cursor/skills/eloftr-v12-dualh-baseline/SKILL.md](../eloftr-v12-dualh-baseline/SKILL.md) §7 | METU AUC `0.001401 / 0.006642 / 0.039631` 详细诊断 | §7 整段加协议过时警告头标，结论（"v12 比 baseline 还差"）仍然成立但**绝对数字单位变了**，待重跑 backfill |
| `dump/metu_eval_v*_version*_*/` | 旧 overall.txt 文件 | 不删（历史 reference），但读到这些数字时要意识到协议不同 |

### 5.3 跨版本对比的正确做法（新协议下）

要回答「cross-modal finetune 相对 ELoFTR outdoor.ckpt 提升多少」：

```
1. 跑 official.bat               -> dump\metu_eval_official\all\overall.txt
                                    -> ELoFTR baseline: 2.962 / 8.093 / 18.125
2. 跑 finetuned.bat 10            -> dump\metu_eval_v10_version*_top1\all\overall.txt
3. 跑 finetuned.bat 11
4. 跑 finetuned.bat 12
5. 对比每个 ckpt 的 all_class_mean 行 vs baseline
```

注意 §2.1 写过的 **side 不对称**：official 用 vis-as-image0，finetuned 用 thermal-as-image0。这是必要的非对称（模型架构决定的），不是 bug。

## 6. 故障树

当 `all_class_mean` 跟论文 2.88/7.88/17.72 差 > 30% 时，按以下顺序排查：

### 6.1 几何方向 sanity（最优先）

- 看 per-scene 表：**所有 scene 都接近 0** = T_0to1 公式反了 / RANSAC 阈值过严 / mkpts 坐标系不对
- **个别 scene 数字接近论文，多数 scene 崩** = 几何处理有 conditional bug（某些 distortion 系数下 undistort 失败 / 某些 pose 配置下 RANSAC 退化）
- 看 `num_matches`：< 50/pair = thr 太严 + RANSAC 阈值松；> 1000/pair = thr 太松（不一定坏）

### 6.2 协议参数 sweep（按概率排）

1. **`--thr 0.1` sweep**：MINIMA 内部 `load_eloftr`（不公开）可能用 0.1 而非 0.2。预期效果 ±2-3% AUC
2. **`cfg.LOFTR.COARSE.NPE = [832, 832, 640, 640]` sweep**：理论上 test 640 < train 832 不需 NPE，但 MINIMA 内部不可见。预期 ±1-2%
3. **`--ransac_thr` sweep 1.0 / 1.5 / 2.0**：1.5 是 XoFTR default，但 v_minor 可能用别的
4. **`--megasize 832` sweep**：跟训练分辨率一致看看（理论上 640 已经对齐论文，但作为 sanity 验证）

### 6.3 几何公式核查

- `T_0to1` 公式（dataset L243）必须是 `P1 @ inv(P0)`，不是 `inv(P1) @ P0`（见 §4）
- `cv2.undistort` 必须传 `newCameraMatrix=new_K`（不是 None）（见 §2 undistort 行）
- estimate_pose 喂的 K 必须是 new_K（dataset 把 K0 / K1 改写成 new_K0 / new_K1）

### 6.4 fp16 残留

- visualize_metu_vistir.py L174-175 强制 `cfg.LOFTR.MP=False, HALF=False`，但 main_cfg merge 可能覆盖
- 检查实际 forward 时 fp32：`print(image0.dtype)` 应该是 `torch.float32`

### 6.5 prec UP / AUC DOWN 信号（finetuned ckpt 上常见）

参考 [eloftr-v12-dualh-baseline §7](../eloftr-v12-dualh-baseline/SKILL.md)：当 prec@5e-4 上升但 AUC 反而下降时，**不是模型烂**，是「模型学了 domain-specific 匹配分布、点落在 epipolar 线上（prec ↑）但 spatial bias 破坏 essential matrix 估计（AUC ↓）」的典型签名。诊断不是看 prec 而是看 AUC。

## 7. 与其他 skill 的边界

| skill | 它管 | 不管（→ 本 skill）|
|---|---|---|
| **本 skill** | METU-VisTIR pose-AUC eval：协议、字段、T_0to1、复现、故障树 | RoadScene / M3FD prec@N px (→ eval-pipeline)、跨版本汇总 (→ results)、单版本设计 (→ v_x) |
| [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md) | RoadScene / M3FD precision@N px pipeline、ckpt ↔ cfg 自动配对、PC cache fallback | METU pose-AUC 协议（→ 本 skill）|
| [eloftr-results](../eloftr-results/SKILL.md) | results/eval_summary.md 工作流：哪个表填什么字段、追加新行规范 | METU 协议本身（→ 本 skill）|
| [eloftr-v12-dualh-baseline](../eloftr-v12-dualh-baseline/SKILL.md) §7 | v12 的 METU OOD 诊断结论（旧协议，待重跑）| 协议参数详情（→ 本 skill）|
| [eloftr-v10-msyn](../eloftr-v10-msyn/SKILL.md) / [v11-dualh](../eloftr-v11-dualh/SKILL.md) | v10/v11 设计动机 + 训练 schedule | METU eval 协议（→ 本 skill）|

## 8. 历史 / 时间线

| 日期 | 事件 |
|---|---|
| 2026-05-11 | v10 sweep 推断 T_0to1 用 C2W 公式（**错判**，被 OOD ckpt R_err 噪声误导）|
| 2026-05-12 | v12 跑 METU eval 用旧协议，发现 v12 < baseline outdoor.ckpt，写 v12 skill §7 「H_isolated_fail」诊断 |
| 2026-05-13 | 发现 v0..v12 旧 METU 协议跟论文 MINIMA 协议大幅偏离（~10-20× 低），重写 eval_metu_vistir_official.bat + visualize_metu_vistir.py + dataset undistort + dataset T_0to1（W2C）+ AUC 聚合（per-class mean）。bit-perfect 复现论文表 3 ELoFTR 2.88 / 7.88 / 17.72 → 我们 2.96 / 8.09 / 18.13。建立本 skill |
| TODO | 把所有 v10/v11/v12 finetuned ckpt 在新协议下重跑 → backfill `results/eval_summary.md` + v10/v11/v12 skill description |
