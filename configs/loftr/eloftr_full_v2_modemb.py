"""v2 (Step 2): v1 (symmetric InfoNCE) + learnable modality embeddings.

Cumulative on top of v1: this config keeps the cross-modal contrastive loss
from Step 1 and additionally enables two learnable C-dim vectors that are
added to feat_c0 (IR) and feat_c1 (VIS) right before the coarse transformer.

Why "cumulative" rather than "modemb only":
    - Step 1 (InfoNCE) and Step 2 (modality emb) are complementary: InfoNCE
      pulls matched cross-modal features together while modemb gives the
      transformer an explicit modality signal so it can learn modality-aware
      attention. Most of the literature on cross-modal matching combines both.
    - Reusing v1's config means we only have to audit ONE diff vs the
      baseline at the end (v2 vs baseline), and the v1 vs v2 diff is exactly
      "with or without modemb" -- giving us a clean ablation in Step 3.

Init = 'zeros' so the very first training step is byte-identical to v1; the
modality embedding only starts diverging from 0 once the gradient signal
pushes it away. This guarantees that finetuning from the official ckpt cannot
"crash" performance at step 0 even if the embedding turns out unhelpful.
"""
from configs.loftr.eloftr_full_v1_contrast import cfg

# Enable learnable modality embedding (added to feat_c0=IR, feat_c1=VIS)
cfg.LOFTR.USE_MODALITY_EMB = True
cfg.LOFTR.MODALITY_EMB_INIT = 'zeros'   # safe finetune: step 0 == v1
