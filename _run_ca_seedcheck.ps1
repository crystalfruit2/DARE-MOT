# DARE-MOT -- CA baseline repeat-seed check (2026-07-31 follow-up to the 07-29 CA-KF result).
# ============================================================================================
# The 07-29 CA-baseline win (-29 IDSw / +1.4% IDF1 / +1.1% MOTA macro vs the CV headline) is
# single-seed (seed=0). The existing CV/DARE pipeline was already proven seed-invariant in
# Move 0b (5 seeds, byte-identical, std=0.00) -- so the CV reference (mc_dare) does not need
# re-running. This script only re-runs the CA baseline (motion_model='ca', two-frame gate OFF,
# full headline config otherwise -- exact same config as the original mc_ca_baseline run) at
# seeds 1-3, to confirm the new CA code inherits the pipeline's seed-invariance rather than
# assuming it (it's new code, unit-tested but not yet seed-checked).
#
# Usage: _run_ca_seedcheck.ps1 [ckpt]
# ============================================================================================
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

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

function Run-CaBaseline($seed, $expn) {
  Clear-DareEnv
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
  $env:DARE_KF_MODEL="ca"
  $env:DARE_TWOFRAME_GATE="0"; $env:DARE_DIAG="1"
  $flags = @("-b","1","-d","1","--fp16","--fuse","--seed","$seed",
             "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
  Write-Output "########## RUN $expn (CA baseline, seed $seed) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $expn *> "$dare\_mc_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  Clear-DareEnv
}

$t0 = Get-Date
Write-Output "########## CA SEED CHECK START (detector = $ckpt) ##########"
Run-CaBaseline 1 "mc_ca_baseline_s1"
Run-CaBaseline 2 "mc_ca_baseline_s2"
Run-CaBaseline 3 "mc_ca_baseline_s3"
Write-Output "########## ALL RUNS DONE in $((Get-Date) - $t0). ##########"
