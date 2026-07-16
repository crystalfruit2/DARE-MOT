# DARE-MOT fix ablation — meeting-brief-2026-07-16.
# Tests Fix #1 (foreground-focused appearance: center-weighted pooling + center-crop shrink)
# and Fix #3 (re-association age cap) as ISOLATED ablations vs the real-appearance baseline.
# Same detector/ckpt/params as YOLOX_outputs/_ablation_multiseq_fixed (the 655-IDSw baseline).
# baseline_refactor MUST reproduce that baseline exactly — it's the sanity gate proving the
# byte_tracker.py refactor changed nothing when the new knobs are at defaults.
# NOTE: must be "Continue", not "Stop" — track.py logs via loguru to stderr, which
# PowerShell 5.1 wraps as NativeCommandError; under "Stop" that aborts on the first
# log line before any tracking runs. Continue lets those records flow into the log.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$out  = "$dare\_fix_ablation"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare

# tag, POOL (mean|center), CROP_SHRINK (fraction), REASSOC_MAX (frames; -1 = off)
$configs = @(
  @{tag="abl_baseline_refactor"; POOL="mean";   CROP="0.0";  REASSOC="-1"},
  @{tag="abl_pool_center";       POOL="center"; CROP="0.0";  REASSOC="-1"},
  @{tag="abl_crop15";            POOL="mean";   CROP="0.15"; REASSOC="-1"},
  @{tag="abl_reassoc30";         POOL="mean";   CROP="0.0";  REASSOC="30"},
  @{tag="abl_combo";             POOL="center"; CROP="0.15"; REASSOC="30"}
)

foreach ($c in $configs) {
  $env:DARE_POOL=$c.POOL; $env:DARE_CROP_SHRINK=$c.CROP; $env:DARE_REASSOC_MAX=$c.REASSOC
  Write-Output "########## RUN $($c.tag)  POOL=$($c.POOL) CROP=$($c.CROP) REASSOC=$($c.REASSOC) ##########"
  & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse `
      --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
      -expn $($c.tag) *> "$out\log_$($c.tag).log"
  Write-Output "----- DONE $($c.tag) (exit $LASTEXITCODE) -----"
}
Write-Output "########## FIX ABLATION COMPLETE ##########"
