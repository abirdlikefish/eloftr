"""v3 (a.k.a. v3.1): v2 (InfoNCE + modemb) + anti-overfit setup.

Adds three orthogonal regularizations on top of v2 to fix the "val peaks
before epoch 4 then degrades" pattern observed in both v1 and v2:

    1. Freeze backbone (9.5M -> 0 trainable) + freeze BN
       Caps overfit capacity. Backbone learns generic edge / texture features
       on MegaDepth that transfer well to both VIS and IR; small RoadScene
       (~160 train pairs) cannot improve them and only risks corrupting them.

    2. EarlyStopping on val precision@3px (patience=4)
       Stops training automatically when the val metric stops improving.
       Removes the need to guess max_epochs.

    3. Aggressive LR schedule tailored to RoadScene's tiny dataset
       - max_epochs=20 (in the bat); EarlyStopping cuts it short in practice.
       - CANONICAL_LR=8e-3 (TRUE_LR = 5e-4 at bs=4). Empirically the prior
         "finetune-grade" 1e-3 (TRUE_LR ~ 6e-5) was *too small*: with backbone
         frozen the remaining 5.7M params actually need MORE LR, not less,
         to escape the pretrained init.
       - WARMUP_STEP=1 (effectively disabled). The original 1875 came from
         MegaDepth's bs=64 setup; even our previous WARMUP_STEP=10 gets
         scaled by train.py to ~160 steps = ~4 epochs at bs=4, which wastes
         the first 4 epochs ramping LR. With only 5.7M trainable params and
         a pretrained init, optimizer statistics stabilize within a few
         steps and warmup brings no benefit.
       - MSLR_MILESTONES=[5, 7] decays LR *before* the typical val plateau,
         so EarlyStopping at patience=4 still has a chance to see the post-
         decay curve before triggering. The default [8, 12, 16, 20, 24]
         (inherited from eloftr_full.py) was too late: with patience=4 the
         run dies around epoch 8-10 before MSLR ever fires.

The cumulative inheritance chain remains:
    eloftr_full -> v1_contrast -> v2_modemb -> v3_combined
so v3 still has step1 (loss_contrast) and step2 (modality_emb) active. v3 is
therefore a strict superset of v2 in terms of features, with extra
regularization on top.

Resulting effective TRUE LR trajectory at bs=4 (40 train batches/epoch):
    epoch 0      step   0   LR = 5e-4   (warmup essentially skipped)
    epoch 5  end step 200   LR = 2.5e-4 (MSLR step #1)
    epoch 7  end step 280   LR = 1.25e-4(MSLR step #2)
    EarlyStopping typically fires around epoch 8-10.
"""
from configs.loftr.eloftr_full_v2_modemb import cfg

# 1. Freeze backbone + BN to cap overfit capacity (16M -> ~5.7M trainable)
cfg.LOFTR.FREEZE_BACKBONE = True
cfg.LOFTR.FREEZE_BN = True

# 2. EarlyStopping on RoadScene's pixel precision@3px (mode=max, set in train.py)
cfg.TRAINER.EARLY_STOPPING = True
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 4

# 3. LR schedule tuned for RoadScene (~160 train pairs, 40 batches/epoch at bs=4).
#    See module docstring for the rationale behind each value.
cfg.TRAINER.CANONICAL_LR    = 8e-3   # TRUE_LR = 8e-3 * 4/64 = 5e-4 at bs=4
cfg.TRAINER.WARMUP_STEP     = 1      # effectively disabled (was 1875 -> 30000 scaled)
cfg.TRAINER.MSLR_MILESTONES = [5, 7] # was [8,12,16,20,24] – too late for patience=4
