# DARE-MOT -- CA tuned-config (accel_noise=1/80) repeat-seed check (2026-08-03).
# ============================================================================================
# The 07-31 accel-noise sweep found DARE_KF_ACCEL_NOISE=0.0125 (1/80) beats the untuned 1/160
# reference on all three metrics (IDSw -39/-14%, IDF1 +1.1%, MOTA +1.9% vs CV headline). That
# tuned point (mc_ca_accel_80) is still only seed=0 -- the seed-invariance check done for the
# untuned reference (_run_ca_seedcheck.ps1, byte-identical across 4 seeds) was never repeated
# for the tuned value. This is the open item flagged to Prof. Farzad on 2026-07-31 and
# confirmed 2026-08-03 ("onu da tamamla") -- closing it before adoption.
#
# Same config as mc_ca_accel_80 otherwise (two-frame gate OFF, full headline IoU-gate +
# appearance config, DARE_KF_ACCEL_NOISE=0.0125), seeds 1-3 (seed 0 already exists as
# mc_ca_accel_80).
#
# Usage: _run_ca_accel80_seedcheck.ps1 [ckpt]
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

function Run-CaAccel80($seed, $expn) {
  Clear-DareEnv
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
  $env:DARE_KF_MODEL="ca"
  $env:DARE_KF_ACCEL_NOISE="0.0125"
  $env:DARE_TWOFRAME_GATE="0"; $env:DARE_DIAG="1"
  $flags = @("-b","1","-d","1","--fp16","--fuse","--seed","$seed",
             "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
  Write-Output "########## RUN $expn (CA accel_noise=0.0125, seed $seed) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $expn *> "$dare\_mc_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  Clear-DareEnv
}

$t0 = Get-Date
Write-Output "########## CA ACCEL_80 SEED CHECK START (detector = $ckpt) ##########"
Run-CaAccel80 1 "mc_ca_accel_80_s1"
Run-CaAccel80 2 "mc_ca_accel_80_s2"
Run-CaAccel80 3 "mc_ca_accel_80_s3"
Write-Output "########## ALL RUNS DONE in $((Get-Date) - $t0). ##########"
