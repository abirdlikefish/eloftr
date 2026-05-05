"""判定一个 DATA_SOURCE 字符串是否属于 "对齐 IR-VIS" 的单一可信源。

"对齐 IR-VIS" 数据集的判定标准：
  - 训练时提供像素级对齐的 IR / VIS 图像对。
  - 提供 3x3 ``homography_0to1``（val/test 时为单位阵），而不是相机几何
    （没有 ``depth`` / ``K`` / ``T``）。
  - 走 ``RoadSceneDataset``（保留为统一的 I/O 类，``dataset_name`` 由
    config 注入，区分实际数据集来源）。
  - 触发 ``spvs_*_roadscene`` 监督函数与 ``_compute_roadscene_metrics``，
    而不是 ScanNet/MegaDepth 的 epipolar 路径。

未来新增数据集（如 MSRS / LLVIP / TNO）只需要：
  1. 把它的小写名加入下方 ``ALIGNED_IRVIS_SOURCES``。
  2. 写一份 ``configs/data/<name>_trainval.py``。
任何 dispatch 点都不必再改动 -- 现在所有 dispatch 都集中调用
``is_aligned_irvis(...)``，参见 ``src/lightning/data.py`` /
``src/loftr/utils/supervision.py`` / ``src/lightning/lightning_loftr.py`` /
``src/utils/plotting.py``。
"""
from __future__ import annotations

ALIGNED_IRVIS_SOURCES = frozenset({"roadscene", "m3fd"})


def is_aligned_irvis(name) -> bool:
    """Return True iff ``name`` (case-insensitive) is a known aligned IR-VIS data source.

    Accepts ``str`` or ``None``; returns ``False`` for ``None`` so callers can
    pass ``cfg.DATASET.TRAINVAL_DATA_SOURCE`` even before it is set.
    """
    if name is None:
        return False
    return str(name).lower() in ALIGNED_IRVIS_SOURCES
