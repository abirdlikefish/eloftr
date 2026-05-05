"""v6: M3FD finetune from v5 ckpt with v2-style slow LR. Target = lift in-domain p@1px.

==============================================================================
Why v6 = v5 ckpt resume (not retrain from pretrained, not switch to RoadScene)
------------------------------------------------------------------------------
v5 实测 (10 epoch on M3FD, eloftr-cross-modal-experiments §5.6):

  In-domain (M3FD test 210):
    p@1=0.4352  p@3=0.8072  p@5=0.8777  mpe=2.07
  OOD (RoadScene test 22):
    p@1=0.1858  p@3=0.5989  p@5=0.7825  mpe=3.39

vs v2 (RoadScene-trained, on RoadScene test 22):
    p@1=0.7567  p@3=0.8086  p@5=0.8151  mpe=1.97
vs v2 (RoadScene-trained, on M3FD test 210, OOD):
    p@1=0.1192  p@3=0.4971  p@5=0.6948  mpe=4.66

Key takeaways:
  1. v5 in-domain p@3 (0.807) ≈ v2 in-domain p@3 (0.809), p@5 even reverses
     (0.878 vs 0.815). v5 is competitive on the "easy" thresholds.
  2. v5 OOD beats v2 OOD on EVERY metric (p@1 +56%, p@3 +20%, p@5 +13%,
     mpe -27%). v5 learned a genuinely transferable IR-VIS representation;
     v2's high in-domain p@1 is RoadScene-22-pair memorisation overfit
     (collapses to 0.119 OOD).
  3. v5's only real gap is in-domain p@1 (0.435 vs v2's 0.757). Root cause
     is NOT BN un-convergence (v5 ran ~9450 step >> §5.4's 3000-step
     threshold and p@1 ALREADY climbed from v4 REV2's 0.329 to 0.418
     during training -- BN converged). Root cause is v5's high effective
     LR in epoch 0-3:

       v2 epoch 0-5k step:  effective LR ~2.08e-5 (still in 60k-step warmup ramp)
       v5 epoch 0-3 (~2835 step):  TRUE_LR=1.25e-4 hit instantly after WARMUP=320

     v5's LR is ~6x higher than v2's during the critical fine sub-pixel
     learning window. Fine refinement L2 sub-pixel offset target oscillates
     around the GT under high LR and never settles to <1px precision.

  4. Beyond LR, modemb is structurally invisible to fine path
     (eloftr-cross-modal-experiments §2.5: fine_preprocess 2 BN + fine_matching
     argmax both eat constant bias). This is a hard ceiling around p@1 ~0.55
     that v6 alone cannot break -- needs v7 MSBN-in-fine_preprocess (§6 path E).

==============================================================================
Design choice: resume vs retrain vs switch dataset
------------------------------------------------------------------------------
Three paths to lift in-domain p@1, ranked:

  (a) RETRAIN from pretrained with slow LR
      Cost: ~9000 step burned re-converging fine BN (already done by v5)
      Benefit: clean state, no ckpt-load surprises
      Verdict: wasteful

  (b) SWITCH to RoadScene + slow LR (replicate v2)
      Cost: 22 train pairs cannot sustain v5's general BN, will collapse
            back to memorisation mode and DESTROY OOD advantage
      Benefit: would likely hit v2's 0.74 in-domain p@1 on RoadScene
      Verdict: zero-sum with v5's main win

  (c) RESUME v5 ckpt + slow LR (this v6)  ← chosen
      Cost: must load ckpt cleanly + cannot change sampler/freeze/data
            (would invalidate ckpt)
      Benefit: starts at v5's converged BN + general backbone, every
            new step is pure refinement on top of v5's gains. Additive
            not subtractive.
      Verdict: best risk/reward

v6 is "v5 + v2-style slow LR continuation". Architecture identical to v5
(FREEZE_BACKBONE_BN=True / InfoNCE / modemb / M3FD sampler all inherited).
Only the LR schedule changes.

==============================================================================
Schedule changes (vs v5)
------------------------------------------------------------------------------
  CANONICAL_LR (override): 2e-3 -> 4e-4
        TRUE_LR = 4e-4 * 4/64 = 2.5e-5 at bs=4. Matches v2's effective peak
        LR (~2.67e-5) where v2's fine sub-pixel converged. 5x slower than
        v5 for the rest of training.

  WARMUP_STEP (override): 20 -> 50
        Resume scenario: ckpt already at v5-terminal state, AdamW second
        moments freshly re-init from PL trainer rebuild. 50 / (4/64) = 800
        absolute step warmup, ~0.85 epoch. Long enough for AdamW to settle
        without burning many steps.

  MSLR_MILESTONES (override): [3, 5, 7] -> [15, 25, 35]
        v5's schedule killed LR by epoch 7 (<2e-5 by ep7), too aggressive
        for slow refinement. v6 lets full LR run for 15 epoch (~14k step)
        before the first decay.

  EARLY_STOPPING_PATIENCE (override): 3 -> 12
        Slow LR refinement converges over many epochs with noisy p@1
        oscillation. v5's patience=3 would fire before any real improvement.

  max_epochs (in .bat): 10 -> 50
        Total training ~47k step (vs v5's 9.45k), giving fine refinement
        ~5x more updates at the slower LR.

==============================================================================
Performance overrides (v6-only opt-in via cfg, no architectural impact)
------------------------------------------------------------------------------
Both flags default to False/32 in src/config/default.py. v6 is the FIRST
config to opt in; v0-v5 keep defaults so retraining them stays bytes-
identical to history.

  PERSISTENT_WORKERS (override): False (default) -> True
        Keep DataLoader workers alive across all 50 epoch instead of
        re-spawning each epoch. Saves ~25min on v6 (50 epoch x ~30s
        spin-up at num_workers=6).

        Trade-off: RoadSceneDataset uses np.random.rand() in __getitem__
        (src/datasets/roadscene.py:249) and np.random.default_rng() to
        sample Homography params (roadscene.py:104). The project has NO
        worker_init_fn anywhere (grep verified), so PyTorch's default
        worker init only seeds torch.manual_seed -- NOT np.random.

        Consequence: with persistent_workers=True the per-worker numpy
        RNG state accumulates across epochs (a worker is forked once,
        never re-init'd). With persistent_workers=False (the historical
        default) the worker is re-spawned each epoch, so numpy state is
        re-fork'd from the main process each epoch. The two paths produce
        statistically equivalent but NOT bytes-identical augmentation
        sequences from epoch 1 onward.

        Acceptable for v6 because v6 is a single end-to-end run, not a
        bytes-identical reproducibility target. v0-v5 keep default False
        so any future retrain of v0-v5 reproduces history exactly.

        If we ever need v6 also bytes-identical, add a worker_init_fn in
        src/datasets/roadscene.py that explicitly seeds np.random per
        worker per epoch (deferred to a future v7+ if needed).

  N_VAL_PAIRS_TO_PLOT (override): 32 -> 1
        v6 has 50 epoch x ~35 val figures/epoch = 1750 figures total at
        default N_VAL_PAIRS_TO_PLOT=32. Each figure is a matplotlib +
        add_figure call, ~3-5s per figure. Reducing to 1 keeps a single
        sanity figure per epoch (50 total) and shrinks TB events file
        by ~70%, saves ~25min.

        ENABLE_PLOTTING (already False, set by configs/data/m3fd_trainval.py:73)
        controls TRAIN-time plotting in lightning_loftr.py:285.
        N_VAL_PAIRS_TO_PLOT controls VAL-time plotting in
        lightning_loftr.py:311. They are orthogonal -- both must be set
        to fully suppress figure overhead.

==============================================================================
Inherited unchanged (do NOT re-set these in v6 -- yacs merge order pitfall)
------------------------------------------------------------------------------
From v5_m3fd:
  N_SAMPLES_PER_SUBSET         = 3780      (M3FD full set per epoch)
  SB_SUBSET_SAMPLE_REPLACEMENT = False     (randperm, every pair seen once)
From v4 REVISION 2 (via v5):
  FREEZE_BACKBONE              = False     (9.5M backbone trainable)
  FREEZE_BN                    = False     (fine BN trainable)
  FREEZE_BACKBONE_BN           = True      (pin backbone BN to v5 stats)
From v2 (via v3 -> v4 -> v5):
  USE_MODALITY_EMB             = True
  MODALITY_EMB_INIT            = 'zeros'   (irrelevant: ckpt loads v5's value)
From v1 (via v2 -> ... -> v5):
  USE_CONTRASTIVE              = True
  CONTRASTIVE_WEIGHT           = 0.01
  CONTRASTIVE_TEMP             = 0.1

==============================================================================
How `--ckpt_path` actually works in this repo (NOT PL --resume_from_checkpoint)
------------------------------------------------------------------------------
This repo's `--ckpt_path` is a project-custom argument (see train.py:72 +
src/lightning/lightning_loftr.py:64-67) that does ONE thing only:

  state_dict = torch.load(pretrained_ckpt, ...)['state_dict']
  self.matcher.load_state_dict(state_dict, strict=False)

It loads ONLY model weights. It does NOT restore any of:
  - optimizer state (AdamW second moments) -- freshly re-initialized to 0
  - LR scheduler state (MSLR step count, last LR) -- starts from v6's config
  - epoch counter -- always starts at 0
  - global_step counter -- always starts at 0
  - PL trainer state (callbacks, EarlyStopping counter) -- freshly init

Consequences for v6:
  - max_epochs=50 means literally 50 epochs of new training (NOT 50-9=41)
  - WARMUP_STEP=50 / MSLR=[15,25,35] are executed from epoch 0 of v6
  - v5's ep9 momentum/scheduler state is fully discarded; only weights live on
  - Re-warming AdamW second moments from 0 takes ~50-200 step at small LR;
    that's why WARMUP_STEP=50 (-> 800 abs step) is set higher than v5's 20

This behaviour is intentional and is what we want -- it lets us cleanly swap
the LR schedule without inheriting v5's already-decayed 1.56e-5 LR.

==============================================================================
Validation gates for v6 (read these BEFORE letting it train >5 epoch)
------------------------------------------------------------------------------
1. Launch log MUST contain (within first 30 lines):
     "Load '...epoch=9-precision@1px=0.418-precision@3px=0.789-precision@5px=0.864.ckpt' as pretrained checkpoint"
   If absent, --ckpt_path was not picked up. Verify path spelling / quoting.

2. The same log will also print missing_keys / unexpected_keys from
   strict=False load. v5 ckpt was saved with the full v1+v2+v3+v4+v5 graph,
   so missing_keys should be EMPTY for v6 (which has identical architecture).
   If you see "modality_emb_ir / modality_emb_vis" missing, you accidentally
   pointed at a pre-v2 ckpt -- wrong path.

3. Epoch 0 val end: p@1px MUST be >= 0.4181 (v5 terminal).
   - First epoch trains under 0 -> 2.5e-5 warmup ramp, very gentle, weights
     barely move; val p@1 should land within +/- 0.005 of v5's 0.4181.
   - If MUCH lower (e.g. 0.30), ckpt loaded but graph mismatch silently
     re-initialised some weights -- check missing_keys list closely.

4. Epoch 0-5: p@1 should rise slowly (+0.005 ~ +0.02 per epoch). If flat
   or dropping, raise CANONICAL_LR to 6e-4 ~ 8e-4 in a v6.5 follow-up.

5. PL progress bar Epoch 0 must show "... 945/1155" (sampler override
   inherited from v5 via `from v5_m3fd import cfg`). If "50/260", the
   inheritance broke -- check yacs merge order in v6 config.

6. Epoch 15+ (after first MSLR decay at LR=1.25e-5): p@1 should be in the
   0.50-0.55 range. If still 0.42, the gap is structural fine-path
   modality-blindness, not LR -- jump to v7 MSBN (cross-modal-experiments
   §6 path E1) instead of pushing v6 further.

==============================================================================
Expected outcomes & next steps
------------------------------------------------------------------------------
Strong success (M3FD val p@1 >= 0.55 + RoadScene OOD p@1 >= 0.25):
  Slow-LR refinement alone closes the gap. Ship v6 as final cross-modal
  finetune. Optional: v7 MSBN to push p@1 toward 0.65+.

Weak success (M3FD val p@1 in 0.45-0.55 + OOD didn't improve):
  Slow LR helped but fine path's modality-blindness is the real ceiling.
  Move to v7 MSBN (eloftr-cross-modal-experiments §6 path E1) to break
  the BN-eats-bias mechanism in fine_preprocess.

Failure (p@1 stuck at 0.42 from epoch 0 onward):
  Either ckpt didn't load (gate #1 failed) or weights loaded but the
  full v6 schedule has zero learning effect (LR too low). Try v6.5 with
  CANONICAL_LR raised to 6e-4 ~ 8e-4.
"""
from configs.loftr.eloftr_full_v5_m3fd import cfg

cfg.TRAINER.CANONICAL_LR            = 4e-4         # v5=2e-3; TRUE_LR = 4e-4 * 4/64 = 2.5e-5 (matches v2 effective peak)
cfg.TRAINER.WARMUP_STEP             = 50           # v5=20; auto-scales to 800 abs step ~0.85 epoch
cfg.TRAINER.MSLR_MILESTONES         = [15, 25, 35] # v5=[3,5,7]; full LR for first 15 epoch
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 12           # v5=3; slow refinement needs higher tolerance

# Performance overrides (v6-only opt-in; v0-v5 keep defaults for bytes-identical retraining; see docstring "Performance overrides" section)
cfg.TRAINER.PERSISTENT_WORKERS      = True         # default False; keep workers alive across 50 epoch, saves ~25min
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT     = 1            # default 32; v6 50 epoch x 35 figs/epoch -> 1750 figs; keep 1 sanity, shrinks TB by ~70%
