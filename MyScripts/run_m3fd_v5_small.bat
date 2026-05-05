@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v5 SMALL: 100 train batches, 20 val batches, 2 epochs (smoke test).
REM
REM Note: M3FD has 3780 train pairs vs RoadScene 177. With v5 config's
REM   N_SAMPLES_PER_SUBSET=3780 + SB_SUBSET_SAMPLE_REPLACEMENT=False
REM   overrides in effect, 1 full M3FD epoch at bs=2 is 1890 step, ~21x
REM   more than RoadScene's ~88 step/epoch at bs=2. Here --limit_train_batches=100
REM   caps each epoch at 100 batches BEFORE the sampler is exhausted, so the
REM   sampler overrides don't change small-bat behaviour (they only matter
REM   for the unbounded combined.bat run). 2 epoch x 100 batch = 200 batch
REM   is plenty to confirm loss/metric direction; no need for 3+ epoch
REM   as in roadscene_v4_small.bat.
REM
REM Key things to watch in TensorBoard (compare against roadscene_v4_small):
REM   - val precision@3px curve: should rise faster on M3FD because data
REM     is more diverse and 6.5x more train batches per epoch even at
REM     bs=2 (88 RoadScene -> 100 M3FD limit-per-epoch).
REM   - mod_emb_*_norm growth: with bigger negative pool (more samples
REM     per epoch contributing to InfoNCE), modemb separation should
REM     stabilise sooner.
REM   - loss_contrast / loss_i2v / loss_v2i should all still appear
REM     (confirming v5 still inherits v1 InfoNCE on top of v2 modemb).
REM   - LR ramp: 0 -> 1.25e-4 over ~320 steps (WARMUP_STEP=20 auto-scaled
REM     to 320 at bs=4 but here bs=2 so warmup is even longer at
REM     ~640 steps -- you'll see LR still climbing throughout 100-batch
REM     epoch 0 and finally hitting full only mid epoch 1).
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v5_m3fd.py ^
  --exp_name=m3fd_v5_small ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=6 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=20 ^
  --limit_train_batches=100 ^
  --limit_val_batches=20 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=2 ^
  --disable_mp ^
  --thr 0.1

pause
