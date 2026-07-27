# DARE-MOT — Phase 4 class-gate diagnostic: does turning appearance OFF for truck/bus
# recover the MOTA those classes are losing?
# =============================================================================================
# The ReID-weight diagnostic (_run_mc_reid_diag.ps1, 2026-07-24) proved the truck/bus MOTA dip
# is NOT caused by the pedestrian-only ReID fine-tune (swapping to a generic embedding gave
# byte-identical truck/bus results). Read: appearance is firing on truck/bus (they're large, so
# the size-gate DARE_GATE_LO=2500 lets them through), but those classes have 0-1 true ID
# switches in EITHER tracker, so appearance has no identity problem to fix -- it can only
# perturb an already-correct IoU match into a stray FP, regardless of which embedding computes
# the appearance cost.
#
# This script tests the direct fix: force lambda=0 (pure IoU, i.e. ByteTrack matching) for
# truck+bus specifically (DARE_LAMBDA_CLASS_EXCLUDE=3,4 -- 0-indexed model head ids from
# convert_visdrone_mc.py: 0=ped,1=car,2=van,3=truck,4=bus), keep appearance ON for ped/car/van
# exactly as the headline, keep the pedestrian-FT ReID weights (already confirmed fine).
# Re-scored against the EXISTING mc_bytetrack results (unaffected -- appearance is off there
# for every class already).
#
# Expected if the read is right: truck/bus MOTA moves back toward (or to) ByteTrack parity,
# closing most/all of the class-avg -1.0 MOTA gap, while ped/car/van IDSw/IDF1 wins are
# untouched (their lambda is unchanged).
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
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }
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
Write-Output "########## MC CLASS-GATE DIAG START (detector = $ckpt, truck+bus appearance OFF) ##########"

# ---- DARE headline, IDENTICAL to _run_mc_remeasure.ps1's Run 2, plus class-exclude ----
Clear-DareEnv
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
$env:DARE_LAMBDA_CLASS_EXCLUDE="3,4"     # truck, bus -- appearance forced OFF for these only
$dan = "mc_dare_clsgate"
Write-Output "########## RUN $dan (DARE headline, truck+bus appearance off) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $dan *> "$dare\_mc_$dan.log"
Write-Output "----- DONE $dan (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring per-class vs existing mc_bytetrack... ##########"
& $py _score_multiclass.py $dan mc_bytetrack
Write-Output "########## MC CLASS-GATE DIAG COMPLETE in $((Get-Date) - $t0). ##########"
