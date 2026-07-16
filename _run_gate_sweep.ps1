# DARE-MOT: scale-gated appearance fusion (adaptive lambda) on the AIN-DG embedding.
# EXPLORATORY sweep to find the operating point + confirm the mechanism. Appearance is
# gated OFF (lambda=0, IoU only) for detections with box area < GATE_LO, full lambda above.
# Motivated by the ablation: appearance helps large targets, hurts tiny UAV crops.
# All seeded/deterministic. NOTE: final headline number must fix the threshold by an
# honest rule (principled / dev-split), not by picking the sweep's best on the eval set.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$out  = "$dare\_gate_sweep"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"
$env:DARE_REID_WEIGHTS="$dare\reid_weights\osnet_ain_x1_0_dg_clean.pth"
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"

$thresholds = @(2500, 3500, 5000)
Write-Output "########## GATE SWEEP START (AIN-DG, hard size gate) ##########"
$t0 = Get-Date
foreach ($thr in $thresholds) {
  $env:DARE_GATE_LO = "$thr"; $env:DARE_GATE_HI = "0"
  $expn = "gate$thr"
  Write-Output "########## RUN $expn  GATE_LO=$thr  ($((Get-Date) - $t0) elapsed) ##########"
  & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --seed 0 `
      --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
      -expn $expn *> "$out\log_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
}
Write-Output "########## GATE SWEEP COMPLETE in $((Get-Date) - $t0) ##########"
