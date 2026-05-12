"""METU_VISTIR same-illumination subset (6 npz / 1382 pair).

Inherits the full cfg and only swaps the list path. Reports auc on the
"easier" same-illumination IR-VIS test slice; pair with cloudy_sunny
to see cross-illumination degradation.
"""
from configs.data.metu_vistir_test_all import cfg


cfg.DATASET.TEST_LIST_PATH = "assets/metu_vistir_test_lists/cloudy_cloudy.txt"
