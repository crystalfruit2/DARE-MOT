# DARE-MOT — Move 0c: ablation-ladder MIDDLE ROW (appearance ON, IoU-feasibility gate OFF).
# =============================================================================================
# The ablation ladder on the clean 5-class detector:
#   Row A = ByteTrack           (lambda 0, no gate)      -> have it: mc_bytetrack
#   Row B = +appearance, NO gate (lambda 0.5, gate 1.0)  -> THIS RUN (mc_dare_nogate)
#   Row C = +appearance +gate = DARE (lambda 0.5, 0.95)  -> have it: mc_dare
#
# A "lambda 0 + gate ON" config would be degenerate -- the IoU-feasibility gate only modulates
# the APPEARANCE term (it zeroes appearance cost for non-IoU-feasible pairs, the +2045-FP fix),
# so at lambda 0 it is a no-op and collapses to ByteTrack. The informative isolation is this one:
#   A->B  = what raw (ungated) appearance does (expect a big FP jump, incl. on truck/bus)
#   B->C  = what the gate buys back (expect FP recovered, IDSw win preserved)
# Config is IDENTICAL to _run_mc_remeasure.ps1 Run 2 (DARE headline) except DARE_IOU_GATE=1.0.
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

$t0 = Get-Date
Write-Output "########## MC APP-NO-GATE (Move 0c) START (detector = $ckpt) ##########"

Clear-DareEnv
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="1.0"     # <-- gate OFF; the ONLY change vs the DARE headline
$dan = "mc_dare_nogate"
Write-Output "########## RUN $dan (appearance ON, gate OFF) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $dan *> "$dare\_mc_$dan.log"
Write-Output "----- DONE $dan (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring the full ladder... ##########"
Write-Output "===== Row B (nogate) vs Row A (ByteTrack): what raw appearance costs ====="
& $py _score_multiclass.py $dan mc_bytetrack
Write-Output "===== Row C (DARE) vs Row B (nogate): what the gate buys ====="
& $py _score_multiclass.py mc_dare $dan
Write-Output "########## MC APP-NO-GATE (Move 0c) COMPLETE in $((Get-Date) - $t0). ##########"
