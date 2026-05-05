@echo off
setlocal
REM ============================================================================
REM  precompute_pc_edges.bat
REM  v7_pcclahe prerequisite: pre-compute Phase Congruency edge maps for
REM  IR/VIS images and cache them next to the source folders. Run this ONCE
REM  before any v7_pcclahe training; subsequent training runs read the cache
REM  from disk (no per-epoch phasecong cost).
REM
REM  Usage:
REM    precompute_pc_edges.bat                -> default: M3FD + RoadScene
REM    precompute_pc_edges.bat --dataset M3FD
REM    precompute_pc_edges.bat --dataset RoadScene --workers 4
REM    precompute_pc_edges.bat --dataset M3FD --overwrite
REM
REM  Outputs:
REM    data/M3FD_Detection/Ir_pc/*.png       (paired with Ir/)
REM    data/M3FD_Detection/Vis_pc/*.png      (paired with Vis/)
REM    data/RoadScene/cropinfrared_pc/*.png  (paired with cropinfrared/)
REM    data/RoadScene/crop_LR_visible_pc/*.png
REM
REM  Approximate runtime (workers=8, scipy fftpack fallback):
REM    M3FD       4200 imgs x 2 modalities = 8400 imgs   ~30-40 min
REM    RoadScene   222 imgs x 2 modalities =  444 imgs    ~1-2 min
REM
REM  Idempotent: re-running skips already-existing PC outputs unless
REM  --overwrite is passed. Safe to abort with Ctrl+C and resume later.
REM ============================================================================

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
if errorlevel 1 (
    echo [precompute_pc_edges] Failed to activate conda env "eff_loftr".
    pause
    exit /b 1
)

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ----------------------------------------------------------------------------
REM Argument routing:
REM   - If the user passed any args (e.g. --dataset M3FD or --workers 4), forward
REM     them verbatim to the python script.
REM   - Otherwise default to running BOTH datasets sequentially (the typical
REM     v7_pcclahe one-time prerequisite use case).
REM ----------------------------------------------------------------------------
if "%~1"=="" (
    echo [precompute_pc_edges] No args given - running default: M3FD + RoadScene
    python MyScripts\precompute_pc_edges.py --dataset M3FD RoadScene
) else (
    echo [precompute_pc_edges] Forwarding args: %*
    python MyScripts\precompute_pc_edges.py %*
)

set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE%==0 (
    echo [precompute_pc_edges] All done.
) else (
    echo [precompute_pc_edges] Exited with code %EXITCODE%.
)
echo.
pause
endlocal & exit /b %EXITCODE%
