"""v1 (Step 1): baseline + symmetric cross-modal contrastive loss.

Inherits everything from eloftr_full.py and only enables the new
contrastive-loss switches added in src/config/default.py. This guarantees
that any future change to the baseline config flows through automatically,
keeping the v1 vs baseline diff minimal and auditable.

Run with batch_size >= 2 so the cross-batch negatives in InfoNCE actually
contain cross-scene tokens. With batch_size == 1 the negative pool collapses
to in-image only, equivalent to dual_softmax's pool, defeating the purpose.
"""
from configs.loftr.eloftr_full import cfg

# Enable cross-modal contrastive loss (CLIP-style symmetric InfoNCE)
cfg.LOFTR.LOSS.USE_CONTRASTIVE = True
cfg.LOFTR.LOSS.CONTRASTIVE_WEIGHT = 0.01
cfg.LOFTR.LOSS.CONTRASTIVE_TEMP = 0.1
