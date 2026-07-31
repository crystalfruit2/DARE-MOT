# DARE-MOT -- Adaptive Weighting (AW) comparison arm (2026-07-31, same day as the density gate).
# ============================================================================================
# Round-2 novelty-brainstorm risk flag: Deep OC-SORT's AW (Sec 3.4) already measures per-
# instance appearance ambiguity directly (best-vs-second-best margin, in-frame) -- local
# density (this project's live thread) may just be a cruder proxy for the same thing.
# Tests, in order: (1) AW alone vs headline; (2) AW + the tuned density gate (r=3.0/s=1.0,
# the best single-axis point found in the radius/strength sweep) together. The density-alone
# point (mc_dare_density_s10, -14 IDSw) is already run/scored -- not re-run here.
#
# Usage: _run_aw_gate.ps1 [ckpt]
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
Write-Output "########## AW COMPARISON ARM START (detector = $ckpt) ##########"

Clear-DareEnv; Set-HeadlineEnv
$env:DARE_AW_GATE="boost"; $env:DARE_AW_STRENGTH="0.15"
Write-Output "########## RUN mc_dare_aw_only ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_dare_aw_only *> "$dare\_mc_mc_dare_aw_only.log"
Write-Output "----- DONE mc_dare_aw_only (exit $LASTEXITCODE) -----"

Clear-DareEnv; Set-HeadlineEnv
$env:DARE_AW_GATE="boost"; $env:DARE_AW_STRENGTH="0.15"
$env:DARE_DENSITY_GATE="boost"; $env:DARE_DENSITY_RADIUS="3.0"; $env:DARE_DENSITY_STRENGTH="1.0"
Write-Output "########## RUN mc_dare_aw_plus_density ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_dare_aw_plus_density *> "$dare\_mc_mc_dare_aw_plus_density.log"
Write-Output "----- DONE mc_dare_aw_plus_density (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## ALL RUNS DONE in $((Get-Date) - $t0). Scoring... ##########"
Write-Output "===== mc_dare_aw_only vs headline (mc_dare) ====="
& $py _score_multiclass.py mc_dare_aw_only mc_dare
Write-Output "===== mc_dare_aw_plus_density vs headline (mc_dare) ====="
& $py _score_multiclass.py mc_dare_aw_plus_density mc_dare
Write-Output "===== mc_dare_aw_plus_density vs density-alone (mc_dare_density_s10) ====="
& $py _score_multiclass.py mc_dare_aw_plus_density mc_dare_density_s10
Write-Output "########## AW COMPARISON COMPLETE in $((Get-Date) - $t0). ##########"
