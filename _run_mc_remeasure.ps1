# DARE-MOT — MULTI-CLASS headline re-measure (Phase 4 go/no-go), on the clean 5-class detector.
# =============================================================================================
# This is the "one command" the whole multi-class migration is building toward. It runs the
# tracker TWICE on the val7 benchmark with the NEW 5-class YOLOX-X detector (fixes both the
# class-leak and the detector train/test leak), then scores per-class with _score_multiclass.py
# and prints the DARE-vs-ByteTrack delta split by class.
#
# PREREQ: the multi-class detector must have finished (or reached a usable epoch). Point $ckpt
# at YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar. Do NOT run while training still owns
# the GPU — one card, tracking inference will contend. (Pass a specific ckpt as arg 1.)
#
# >>> TWO PHASE-4 SCIENTIFIC DECISIONS ALP MUST CONFIRM before any number here is publishable <<<
#   1. ReID weights: the DARE headline below uses osnet_ain_x1_0_visdrone_ft.pth, which was
#      fine-tuned on PEDESTRIAN crops only (Plan A). It is out-of-distribution for car/van/
#      truck/bus. Options: (a) keep it (appearance helps peds, ~no-op on vehicles), (b) swap to
#      generic osnet_ain (no ped bias), (c) re-fine-tune on all 5 classes. Until decided, the
#      vehicle-class appearance numbers are provisional.
#   2. Size gate (DARE_GATE_LO=2500 area px): tuned so appearance only fires on large-enough
#      pedestrian boxes. Vehicles have a very different size distribution — this gate may
#      suppress or over-fire appearance on them. Revisit for the multi-class headline.
# The ByteTrack baseline (appearance OFF) has neither caveat — it is clean and class-agnostic.
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

# Shared track.py flags — identical to the single-class headline harness (_run_iou_gate_sweep.ps1).
$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

$t0 = Get-Date
Write-Output "########## MC RE-MEASURE START  (detector = $ckpt) ##########"

# ---- Run 1: ByteTrack baseline (appearance OFF, pure IoU; gate off, lock off) ----
Clear-DareEnv
$env:DARE_LAMBDA   = "0.0"     # appearance off  == ByteTrack matching
$env:DARE_IOU_GATE = "1.0"     # gate off (old bt baseline predates the gate)
$env:DARE_LOCK     = "0"
$btn = "mc_bytetrack"
Write-Output "########## RUN $btn (ByteTrack baseline) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $btn *> "$dare\_mc_$btn.log"
Write-Output "----- DONE $btn (exit $LASTEXITCODE) -----"

# ---- Run 2: DARE headline (osnet-ain FT, lambda 0.5, size-gate 2500, lock off, IoU-gate 0.95) ----
Clear-DareEnv
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
$dan = "mc_dare"
Write-Output "########## RUN $dan (DARE headline) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $dan *> "$dare\_mc_$dan.log"
Write-Output "----- DONE $dan (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring per-class... ##########"
& $py _score_multiclass.py $dan $btn   # prints per-class tables for both + DARE-minus-ByteTrack delta
Write-Output "########## MC RE-MEASURE COMPLETE in $((Get-Date) - $t0). ##########"
