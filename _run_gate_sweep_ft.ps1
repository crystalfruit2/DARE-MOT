# DARE-MOT: scale-gated appearance fusion on the VISDRONE-FINE-TUNED OSNet-AIN embedding
# (Plan A, Task #4). Identical to _run_gate_sweep.ps1 EXCEPT DARE_REID_WEIGHTS points at
# the fine-tuned checkpoint and expn is prefixed "ft" — so only the embedding changes and
# the comparison isolates the effect of fine-tuning.
# Line to beat = ByteTrack: 173 ped / 552 all-class / IDF1 65.0 / MOTA 54.5.
# INTEGRITY: the fine-tune trained ONLY on VisDrone-MOT-train (disjoint from val7); the
# final headline threshold must be fixed by a principled rule / dev-split, not the eval best.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned weights: $ft"; exit 1 }
$out  = "$dare\_gate_sweep_ft"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"
$env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"

$thresholds = @(2500, 3500, 5000)
Write-Output "########## FT GATE SWEEP START (VisDrone-FT OSNet-AIN, hard size gate) ##########"
$t0 = Get-Date
foreach ($thr in $thresholds) {
  $env:DARE_GATE_LO = "$thr"; $env:DARE_GATE_HI = "0"
  $expn = "ftgate$thr"
  Write-Output "########## RUN $expn  GATE_LO=$thr  ($((Get-Date) - $t0) elapsed) ##########"
  & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --seed 0 `
      --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
      -expn $expn *> "$out\log_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
}
Write-Output "########## FT GATE SWEEP COMPLETE in $((Get-Date) - $t0) ##########"
