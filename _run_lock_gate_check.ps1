# DARE-MOT — Gated lock ON/OFF cross-check (verifies ablation-ladder finding #1 under the HEADLINE config).
# Ladder finding #1 (ungated backdrop): hard KF lock is net-NEGATIVE (lock_off = lowest IDSw 460, highest MOTA 47.6).
# That was on the ungated appearance backdrop. The paper headline is the GATED, VisDrone-fine-tuned OSNet-AIN config.
# This run holds that headline config fixed and flips ONLY DARE_LOCK, across the same {2500,3500,5000} gate sweep,
# to see whether "the hard lock hurts" survives into the config that actually reaches Table III.
#
# Design: identical backdrop to _run_gate_sweep_ft.ps1 (FT OSNet-AIN embedding + hard size gate, lambda=0.5),
# seed 0 / deterministic (cudnn.benchmark=False under --seed, proven 7/7 byte-identical, class-leak §10-13).
# lock-ON side already exists on disk as ftgate{2500,3500,5000} (DARE_LOCK defaults to '1'); those are
# byte-reproducible by seed, so we only RUN the lock-OFF side here, then re-score BOTH sides with one harness.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned weights: $ft"; exit 1 }
$out  = "$dare\_lock_gate_check"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare

# ---- fixed HEADLINE backdrop (same as _run_gate_sweep_ft.ps1) ----
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"
$env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
# reset the ladder template-update knobs to the validated defaults (no bleed from prior runs)
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""

# ---- the ONE knob under test ----
$env:DARE_LOCK="0"   # hard KF lock OFF (lock-ON counterparts = existing ftgate{thr})

$thresholds = @(2500, 3500, 5000)
Write-Output "########## LOCK-GATE CHECK START (FT OSNet-AIN, hard size gate, LOCK OFF) ##########"
$t0 = Get-Date
foreach ($thr in $thresholds) {
  $env:DARE_GATE_LO = "$thr"; $env:DARE_GATE_HI = "0"
  $expn = "ftgate${thr}_lockoff"
  Write-Output "########## RUN $expn  GATE_LO=$thr  LOCK=0  ($((Get-Date) - $t0) elapsed) ##########"
  & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --seed 0 `
      --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
      -expn $expn *> "$out\log_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
}
Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring lock ON vs OFF... ##########"

# ---- score all 6 (existing lock-ON ftgate{thr} + new lock-OFF) both ways, one harness ----
$summary = "$out\_lock_gate_scores.txt"
Remove-Item $summary -ErrorAction SilentlyContinue
foreach ($thr in $thresholds) {
  foreach ($expn in @("ftgate$thr", "ftgate${thr}_lockoff")) {
    foreach ($mode in @("allclass","ped")) {
      & $py _score_ped.py $expn $mode *>> $summary
    }
  }
}
Write-Output "########## LOCK-GATE CHECK COMPLETE in $((Get-Date) - $t0). Scores -> $summary ##########"
Get-Content $summary
