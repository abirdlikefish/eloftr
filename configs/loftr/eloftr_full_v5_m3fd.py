"""v5: M3FD-only training. Inherit v4 REVISION 2 architecture, retune schedule.

==============================================================================
CRITICAL Sampler override: N_SAMPLES_PER_SUBSET = 3780 + replacement = False
------------------------------------------------------------------------------
LoFTR's RandomConcatSampler (src/datasets/sampler.py) is built for the
ScanNet/MegaDepth multi-scene structure (94/196 scenes per dataset, each
.npz = one subset). The default `_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200`
is a "per-scene quota" sized for that structure. RoadScene/M3FD wrap
their entire flat dataset as `ConcatDataset([ds])` (n_subset == 1, see
src/lightning/data.py:248), so the default 200 quietly degrades into a
"per-dataset cap of 200 samples/epoch":
  * RoadScene (177 < 200): with-replacement randint(0, 177, (200,))
    happens to cover ~63% of the set per epoch but 50 epochs of repeats
    eventually visit everything -- this is why v0..v4 worked despite the
    bug being latent.
  * M3FD (3780 >> 200): the cap silently throws away 95% of the new
    training data. Each epoch only sees 200 random samples, 10 epochs
    visit at most ~2000 unique pairs (with-replacement birthday-style),
    and the entire "21x more step density" premise of v5 collapses.

This v5 config explicitly overrides BOTH sampler knobs:
  * cfg.TRAINER.N_SAMPLES_PER_SUBSET = 3780
        Match M3FD train set size exactly.
  * cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False
        Switch sampler.py:53 branch from `torch.randint(low, high, (3780,))`
        (with-replacement, ~63% unique = ~2389 pairs/epoch) to
        `torch.randperm(3780)[:3780]` (full random permutation, exactly
        3780 unique pairs/epoch, every M3FD pair seen once per epoch).
        Cross-epoch shuffling is preserved because RandomConcatSampler is
        rebuilt every epoch by PL (see data.py:367 `re-init`), so the
        permutation is re-rolled with a fresh torch.randperm seed.

Without these two overrides, v5 trains on ~5% of M3FD per epoch and the
schedule numbers below (945 step/epoch, MSLR=[3,5,7], etc.) are all
off by ~19x. With them, the numbers in the comments are correct.

We do NOT change `_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200` in default.py
because that default is correct for the ScanNet/MegaDepth multi-scene
case it was designed for. Single-subset IR-VIS datasets must opt in
explicitly here.
==============================================================================

==============================================================================
Why M3FD-only is the cleanest first cross-dataset experiment
------------------------------------------------------------------------------
v2 v0 best (RoadScene): p@3px=0.800 at epoch 49, ~6400 grad steps
v4 REVISION 2 (RoadScene): p@3px=0.716 at epoch 9, ~640 grad steps
                         + p@1px=0.329 (fine BN running stats undertrained)

M3FD has 3780 train pairs vs RoadScene's 177 -> ~21x step density at bs=4
(945 step/epoch vs 44 step/epoch), but ONLY when the sampler overrides
above are in effect. Without them, n_samples_per_subset stays at the
default 200 and M3FD epochs collapse to 50 step/epoch (worse than
RoadScene). v5 inherits v4 REVISION 2's freeze strategy unchanged so any
improvement is attributable to "more steps + more diverse samples" alone
(controlled ablation against v4 REVISION 2).

If v5 reaches p@3px ~0.80 on M3FD val, the "BN running stats need 3000+
steps to converge" hypothesis from cross-modal-experiments skill SS5 is
confirmed. If it stays around 0.71 or drops, data scaling is NOT the
bottleneck and the next step is path A (slow LR + long epochs on
RoadScene) per cross-modal-experiments skill SS6.
==============================================================================

Schedule re-tuning (vs v4 REVISION 2):
  CANONICAL_LR=2e-3 (kept) -> TRUE_LR=1.25e-4 at bs=4. Kept identical to
        v4 to preserve "only data source changed" controlled experiment.
        Although more data theoretically supports a higher LR, raising
        LR here would confound BN-convergence vs LR-effect attribution.
  WARMUP_STEP (override): 2 -> 20. After train.py auto-scaling at bs=4
        (multiply by 16), this is 320 absolute steps = ~3.4% of total
        training (vs v4's 32 steps which was 73% of one RoadScene epoch
        but only 0.34% of total M3FD training -- effectively no warmup).
        320 steps lands in the standard 1-5% warmup sweet spot, lets
        AdamW second-moment estimates settle before backbone (now
        unfrozen) sees full LR, and protects fine_preprocess BN running
        stats from being polluted by bad-gradient first-step jitter.
  max_epochs (override via .bat --max_epochs=10): 30 -> 10. 10 * 945 =
        9450 grad steps, comfortably above v2's 6400 and well past the
        ~3000-step BN convergence threshold. Going higher risks
        overfitting once BN has converged.
  MSLR_MILESTONES (override): [10,15,20] -> [3,5,7]. Decays land at
        ~2800 / 4700 / 6600 steps. Final 3 epochs (7-10) at 1/8 LR for
        refinement.
  EARLY_STOPPING_PATIENCE (override): 8 -> 3. M3FD val=210 pairs is
        ~10x less noisy than RoadScene val=22 pairs, so 3-epoch patience
        is plenty to detect a real plateau.

Freeze inherited from v4 REVISION 2:
  FREEZE_BACKBONE=False, FREEZE_BN=False, FREEZE_BACKBONE_BN=True
  (pin backbone BN to MegaDepth pretrained stats, allow fine_preprocess BN
  to adapt to IR-VIS over the now-3780-pair training set).

Hardware constraint (NOT a free design choice):
  bs=4 is forced by 16GB VRAM. v4 RoadScene bs=4 already used ~24GB
  total (16GB dedicated + 8GB shared via PCIe), and M3FD per-sample
  memory is identical (same 480x480 padded canvas, same model). bs=8
  would need ~48GB and almost certainly OOM. So "control vs v4: only
  data source + schedule change" is hardware-forced, not a design
  preference -- which makes the bs=4 / LR=1.25e-4 attribution clean.

Padding note:
  M3FD originals are 93.5% 1024x768 (4:3) and 6.5% nine other smaller
  sizes. With ROAD_IMG_RESIZE=480 + ROAD_PAD_SIZE=480 (kept identical to
  v4), the dominant 1024x768 becomes 352x480 padded to 480x480 (~27%
  bottom-row padding waste). This is intentional: changing PAD_SIZE or
  RESIZE here would break the "v5 vs v4 only data source change"
  control logic. Padding optimisation candidates (rectangular pad,
  RESIZE=640) are deferred to v6+. Mask shape is more uniform than
  RoadScene (93.5% same top-352 / bot-128 pattern), which is a positive
  side-effect helping fine_preprocess BN running stats converge.

Naming: this is the first experiment in the m3fd_v_x series. Versioning
continues from v5 (not restarting at v1) because v5 inherits the full
v1+v2+v3+v4 cumulative feature stack and only swaps the data source +
schedule. Subsequent M3FD experiments (v6, v7, ...) follow the same
naming.
"""
from configs.loftr.eloftr_full_v4_combined import cfg

cfg.TRAINER.N_SAMPLES_PER_SUBSET         = 3780        # override default=200 (ScanNet per-scene quota); use full M3FD train set per epoch
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False       # override default=True; switch to randperm so every pair seen exactly once per epoch

cfg.TRAINER.WARMUP_STEP             = 20          # override v4=2; auto-scales to 320 abs step ~3.4% total training
cfg.TRAINER.MSLR_MILESTONES         = [3, 5, 7]   # override v4=[10,15,20]; M3FD epoch ~21x longer
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 3           # override v4=8; M3FD val=210 is ~10x less noisy
