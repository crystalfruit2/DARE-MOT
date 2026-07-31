# DARE-MOT -- density-gated fusion weight test (2026-07-31, "Dynamic Fusion Gate" narrowed
# to local density only after lit-checking the appearance-ambiguity half away).
# ============================================================================================
# Tests whether local per-detection crowding, composed on top of the existing size gate,
# improves on the current CV headline (mc_dare) -- and in which direction (boost = more
# appearance weight in dense scenes, suppress = less). Same config as mc_dare otherwise
# (Set-HeadlineEnv), plus DARE_DENSITY_GATE.
#
# Usage: _run_density_gate.ps1 [ckpt]
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

function Set-HeadlineEnv {
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
}

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

$t0 = Get-Date
Write-Output "########## DENSITY GATE SWEEP START (detector = $ckpt) ##########"

Clear-DareEnv; Set-HeadlineEnv
$env:DARE_DENSITY_GATE="boost"; $env:DARE_DENSITY_RADIUS="3.0"; $env:DARE_DENSITY_STRENGTH="2.0"
Write-Output "########## RUN mc_dare_density_boost ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_dare_density_boost *> "$dare\_mc_mc_dare_density_boost.log"
Write-Output "----- DONE mc_dare_density_boost (exit $LASTEXITCODE) -----"

Clear-DareEnv; Set-HeadlineEnv
$env:DARE_DENSITY_GATE="suppress"; $env:DARE_DENSITY_RADIUS="3.0"; $env:DARE_DENSITY_STRENGTH="2.0"
Write-Output "########## RUN mc_dare_density_suppress ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_dare_density_suppress *> "$dare\_mc_mc_dare_density_suppress.log"
Write-Output "----- DONE mc_dare_density_suppress (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## ALL RUNS DONE in $((Get-Date) - $t0). Scoring... ##########"
Write-Output "===== boost vs headline (mc_dare) ====="
& $py _score_multiclass.py mc_dare_density_boost mc_dare
Write-Output "===== suppress vs headline (mc_dare) ====="
& $py _score_multiclass.py mc_dare_density_suppress mc_dare
Write-Output "########## DENSITY GATE SWEEP COMPLETE in $((Get-Date) - $t0). ##########"
