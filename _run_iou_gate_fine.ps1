# DARE-MOT — IoU-feasibility gate sweep (branch: exp/iou-feasibility-gate).
# Gap decomposition (2026-07-20, _decompose_gap.py) showed the IDF1/MOTA deficit vs ByteTrack is
# a false-positive flood the appearance term manufactures: a low ReID distance pulls a
# geometrically-implausible (low-IoU) pair under match_thresh. ftgate2500_lockoff vs ByteTrack =
# FP +2045, FN +1218, IDSw -185 => -2.7 MOTA / -4.0 IDF1 despite the ID-switch WIN.
#
# Test: hold the exact ftgate2500_lockoff HEADLINE backdrop fixed and sweep ONLY DARE_IOU_GATE,
# which masks any appearance-eligible match whose raw 1-IoU exceeds the gate. Hypothesis: FP drops
# toward ByteTrack while the IDSw win (disambiguation among OVERLAPPING candidates) survives.
# Gate-off control = existing on-disk ftgate2500_lockoff (365 IDSw) — not re-run.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned weights: $ft"; exit 1 }
$out  = "$dare\_iou_gate_fine"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare

# ---- fixed HEADLINE backdrop = ftgate2500_lockoff (identical to _run_lock_gate_check.ps1) ----
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"
$env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"
$env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"   # lock OFF (headline)

# ---- the ONE knob under test ----
$gates = @{ "92" = "0.92"; "95" = "0.95"; "97" = "0.97" }   # finer knee between 0.90 and off

Write-Output "########## IOU-GATE SWEEP START (backdrop = ftgate2500_lockoff) ##########"
$t0 = Get-Date
foreach ($tag in @("92","95","97")) {
  $env:DARE_IOU_GATE = $gates[$tag]
  $expn = "ftg2500_lockoff_iou$tag"
  Write-Output "########## RUN $expn  DARE_IOU_GATE=$($gates[$tag])  ($((Get-Date) - $t0) elapsed) ##########"
  & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --seed 0 `
      --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
      -expn $expn *> "$out\log_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
}
Remove-Item Env:\DARE_IOU_GATE -ErrorAction SilentlyContinue
Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring + decomposing vs ByteTrack... ##########"

# ---- score (allclass + ped) and decompose each new config against the ByteTrack baseline ----
$summary = "$out\_iou_gate_scores.txt"
Remove-Item $summary -ErrorAction SilentlyContinue
foreach ($tag in @("92","95","97")) {
  $expn = "ftg2500_lockoff_iou$tag"
  foreach ($mode in @("allclass","ped")) { & $py _score_ped.py $expn $mode *>> $summary }
  & $py _decompose_gap.py bt_l0_s0_r1 $expn allclass *>> $summary
}
Write-Output "########## IOU-GATE SWEEP COMPLETE in $((Get-Date) - $t0). Scores -> $summary ##########"
Get-Content $summary
