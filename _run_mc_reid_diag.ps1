# DARE-MOT — Phase 4 ReID caveat diagnostic: is the truck/bus MOTA dip caused by the
# pedestrian-only fine-tune, or is it an intrinsic property of the appearance branch?
# =============================================================================================
# Phase 4 (2026-07-24) found DARE beats ByteTrack cleanly on IDSw (-22%, all 5 classes) but
# class-avg MOTA -1.0, driven entirely by truck/bus. Two confounds sit on those classes: (1) the
# ReID weights are fine-tuned on PEDESTRIAN crops only -> OOD for vehicles, (2) the size-gate
# (2500px) is untuned for vehicle box sizes. This script isolates confound (1) ONLY: swap the
# fine-tuned OSNet-AIN weights for the generic (never-fine-tuned) domain-generalized checkpoint
# that Plan A started from, keep every other DARE knob identical (size-gate, lambda, IoU-gate,
# lock), and re-score against the EXISTING mc_bytetrack results (appearance-off, unaffected by
# ReID weight choice -> no need to re-run it).
#
# Reads as a fast (~15 min, one tracking arm, no training) go/no-go for the Phase-4 caveat:
#   truck/bus MOTA dip shrinks/vanishes  -> OOD pedestrian fine-tune WAS the cause -> re-fine-tune
#                                            on all 5 classes (option c) is worth the investment.
#   truck/bus MOTA dip persists/worsens  -> generic weights aren't OOD-biased and still tank
#                                            vehicles -> the appearance branch itself (or the
#                                            size-gate) is the problem, not the fine-tune ->
#                                            re-fine-tuning on 5 classes likely won't fix it;
#                                            revisit the size-gate (confound 2) instead.
# =============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = if ($args.Count -ge 1) { $args[0] } else { "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" }
$genw = "$dare\reid_weights\osnet_ain_x1_0_dg_clean.pth"   # generic, never fine-tuned

Set-Location $dare
$env:PYTHONPATH = $dare
if (-not (Test-Path $ckpt)) { Write-Output "MISSING detector ckpt: $ckpt"; exit 1 }
if (-not (Test-Path $genw)) { Write-Output "MISSING generic ReID weights: $genw"; exit 1 }
if (-not (Test-Path "$dare\YOLOX_outputs\mc_bytetrack\track_results")) {
  Write-Output "MISSING mc_bytetrack results -- run _run_mc_remeasure.ps1 first"; exit 1
}

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

$t0 = Get-Date
Write-Output "########## MC REID DIAG START (detector = $ckpt, weights = generic dg_clean) ##########"

# ---- DARE headline, IDENTICAL to _run_mc_remeasure.ps1's Run 2, except ReID weights ----
Clear-DareEnv
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$genw
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
$dan = "mc_dare_genreid"
Write-Output "########## RUN $dan (DARE headline, generic ReID) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $dan *> "$dare\_mc_$dan.log"
Write-Output "----- DONE $dan (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring per-class vs existing mc_bytetrack... ##########"
& $py _score_multiclass.py $dan mc_bytetrack
Write-Output "########## MC REID DIAG COMPLETE in $((Get-Date) - $t0). ##########"
