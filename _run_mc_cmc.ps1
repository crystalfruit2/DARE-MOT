# DARE-MOT — Move 1: Camera Motion Compensation (CMC), on the clean 5-class detector.
# ============================================================================================
# Adds an inter-frame affine warp to each predicted track state before matching (DARE_CMC),
# to compensate VisDrone's heavy drone-camera motion (the KF assumes a static camera).
# Runs BOTH arms with CMC on -- DARE+CMC vs ByteTrack+CMC is the only honest comparison, since
# CMC helps any tracker. Identical flags/detector/seed to _run_mc_remeasure.ps1; the ONLY change
# vs the no-CMC headline is DARE_CMC. Scores against the existing no-CMC runs (mc_dare,
# mc_bytetrack) so we see both (i) does CMC help each arm, and (ii) the DARE-vs-ByteTrack gap
# with CMC. Method defaults to sparseOptFlow (BoT-SORT standard); pass 'ecc' as arg 2 for a
# fully-deterministic (RANSAC-free) variant.
# ============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = if ($args.Count -ge 1) { $args[0] } else { "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" }
$cmc  = if ($args.Count -ge 2) { $args[1] } else { "sparseOptFlow" }
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"

Set-Location $dare
$env:PYTHONPATH = $dare
if (-not (Test-Path $ckpt)) { Write-Output "MISSING detector ckpt: $ckpt"; exit 1 }

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

$t0 = Get-Date
Write-Output "########## MOVE 1 CMC START  (detector=$ckpt, cmc=$cmc) ##########"

# ---- Run 1: ByteTrack + CMC (appearance OFF, gate off, lock off; only CMC added) ----
Clear-DareEnv
$env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0"; $env:DARE_CMC=$cmc
$btn = "mc_bytetrack_cmc"
Write-Output "########## RUN $btn ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $btn *> "$dare\_mc_$btn.log"
Write-Output "----- DONE $btn (exit $LASTEXITCODE) -----"

# ---- Run 2: DARE + CMC (headline config + CMC) ----
Clear-DareEnv
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"; $env:DARE_CMC=$cmc
$dan = "mc_dare_cmc"
Write-Output "########## RUN $dan ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $dan *> "$dare\_mc_$dan.log"
Write-Output "----- DONE $dan (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring... ##########"
Write-Output "===== (i) DARE+CMC vs ByteTrack+CMC  (headline, with CMC) ====="
& $py _score_multiclass.py $dan $btn
Write-Output "===== (ii) DARE+CMC vs DARE (no CMC): does CMC help DARE? ====="
& $py _score_multiclass.py $dan mc_dare
Write-Output "===== (iii) ByteTrack+CMC vs ByteTrack (no CMC): sanity, CMC should help ====="
& $py _score_multiclass.py $btn mc_bytetrack
Write-Output "===== (iv) HOTA/AssA: DARE+CMC vs ByteTrack+CMC ====="
& $py _trackeval\_score_hota_mc.py $dan $btn
Write-Output "########## MOVE 1 CMC COMPLETE in $((Get-Date) - $t0). ##########"
