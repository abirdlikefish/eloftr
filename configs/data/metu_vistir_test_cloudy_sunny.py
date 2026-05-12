"""METU_VISTIR cross-illumination subset (4 npz / 1208 pair).

Inherits the full cfg and only swaps the list path. Each pair links a
cloudy/ frame to a sunny/ frame -- the hardest robustness slice in
METU_VISTIR (cross-modal + cross-illumination).
"""
from configs.data.metu_vistir_test_all import cfg


cfg.DATASET.TEST_LIST_PATH = "assets/metu_vistir_test_lists/cloudy_sunny.txt"
