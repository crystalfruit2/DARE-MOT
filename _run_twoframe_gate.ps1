# DARE-MOT — Two-frame dynamic IoU gate ablation (Prof Farzad, meeting 2026-07-28, item 8).
# =============================================================================================
# Confirmed NOT a novelty claim (STC-SORT, Applied Sciences 2026, already ships a strictly more
# general learned version of this over a much longer window -- see novelty-triage-2026-07-28 §6,
# STC-SORT-paper-notes). Run anyway for an honest before/after result, on top of the validated
# DARE headline config (same base as _run_mc_appnogate.ps1), varying ONLY the two-frame gate.
#   Row DARE            = headline (unchanged)                          -> have it: mc_dare
#   Row DARE+2F-dynamic = + two-frame dynamic IoU gate, softmax mode     -> THIS RUN
#   Row DARE+2F-static  = + two-frame dynamic IoU gate, static blend     -> THIS RUN
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

function Set-HeadlineEnv {
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
}

$t0 = Get-Date
Write-Output "########## TWO-FRAME GATE ABLATION START (detector = $ckpt) ##########"

Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_TWOFRAME_MODE="dynamic"; $env:DARE_TWOFRAME_TAU="0.5"
Write-Output "########## RUN mc_dare_twoframe_dyn (dynamic softmax mode) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_dare_twoframe_dyn *> "$dare\_mc_mc_dare_twoframe_dyn.log"
Write-Output "----- DONE mc_dare_twoframe_dyn (exit $LASTEXITCODE) -----"

Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_TWOFRAME_MODE="static"; $env:DARE_TWOFRAME_ALPHA="0.7"
Write-Output "########## RUN mc_dare_twoframe_static (static alpha=0.7 mode) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_dare_twoframe_static *> "$dare\_mc_mc_dare_twoframe_static.log"
Write-Output "----- DONE mc_dare_twoframe_static (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring... ##########"
Write-Output "===== DARE+2F-dynamic vs DARE (headline) ====="
& $py _score_multiclass.py mc_dare_twoframe_dyn mc_dare
Write-Output "===== DARE+2F-dynamic vs ByteTrack ====="
& $py _score_multiclass.py mc_dare_twoframe_dyn mc_bytetrack
Write-Output "===== DARE+2F-static vs DARE (headline) ====="
& $py _score_multiclass.py mc_dare_twoframe_static mc_dare
Write-Output "########## TWO-FRAME GATE ABLATION COMPLETE in $((Get-Date) - $t0). ##########"
