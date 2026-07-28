# DARE-MOT — Static-EMA gamma sweep (Prof Farzad, meeting 2026-07-28, item 3).
# =============================================================================================
# Runs the static-EMA control (DARE_STATIC_EMA, order-1: F^t = g*F^{t-1} + (1-g)*f^t) at
# gamma in {0.2, 0.3, 0.4, 0.5, 0.6}, holding every other knob at the validated DARE headline
# config (same as _run_mc_appnogate.ps1 / _run_mc_remeasure.ps1 Run 2), so the only variable
# is gamma. Each run is scored against mc_bytetrack and mc_dare (the dynamic-softmax headline)
# so the sweep shows both "vs no-appearance baseline" and "vs our own dynamic aggregation".
# =============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = if ($args.Count -ge 1) { $args[0] } else { "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" }
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"

Set-Location $dare
$env:PYTHONPATH = $dare
if (-not (Test-Path $ckpt)) { Write-Output "MISSING detector ckpt: $ckpt"; exit 1 }
if (-not (Test-Path $ft))   { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

$gammas = @("0.2","0.3","0.4","0.5","0.6")
$t0 = Get-Date
Write-Output "########## EMA GAMMA SWEEP START (detector = $ckpt) ##########"

foreach ($g in $gammas) {
  Clear-DareEnv
  # Headline config, unchanged, except DARE_STATIC_EMA replaces the dynamic softmax aggregation.
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
  $env:DARE_STATIC_EMA="$g"          # <-- the ONLY thing varying across this sweep
  $env:DARE_STATIC_GAMMAS=""

  $run = "mc_ema_g$($g -replace '\.','')"
  Write-Output "########## RUN $run (static-EMA gamma=$g) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $run *> "$dare\_mc_$run.log"
  Write-Output "----- DONE $run (exit $LASTEXITCODE) -----"
}
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring the sweep... ##########"
foreach ($g in $gammas) {
  $run = "mc_ema_g$($g -replace '\.','')"
  Write-Output "===== gamma=$g vs ByteTrack ====="
  & $py _score_multiclass.py $run mc_bytetrack
  Write-Output "===== gamma=$g vs DARE (dynamic softmax headline) ====="
  & $py _score_multiclass.py $run mc_dare
}
Write-Output "########## EMA GAMMA SWEEP COMPLETE in $((Get-Date) - $t0). ##########"
