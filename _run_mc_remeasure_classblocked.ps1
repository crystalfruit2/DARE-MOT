# DARE-MOT -- class-BLOCKED re-measure of the headline (2026-07-29).
# =============================================================================================
# Validity check raised in novelty-triage-2026-07-28 (SS3 warning box): DARE-MOT's association
# has always been class-agnostic by design (byte_tracker.py STrack.cls comment). The field's
# actual default on multi-class benchmarks is class-BLOCKED (mmtracking cate_cost, PR #548;
# BoxMOT per_class=True) -- forbidding a track from matching a detection of a different
# predicted class, in every association stage. Both arms shared the agnostic setting before,
# so the delta was fair either way -- but a reviewer can attack the setup's standardness. This
# re-runs the exact same _run_mc_remeasure.ps1 headline configs with DARE_CLASS_BLOCK=1 added,
# nothing else changed, so the before/after isolates only this one knob.
#
# Identical to _run_mc_remeasure.ps1 except: DARE_CLASS_BLOCK=1 on both runs, tags suffixed _cb.
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

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

$t0 = Get-Date
Write-Output "########## MC RE-MEASURE (CLASS-BLOCKED) START  (detector = $ckpt) ##########"

# ---- Run 1: ByteTrack baseline, class-blocked ----
Clear-DareEnv
$env:DARE_LAMBDA   = "0.0"
$env:DARE_IOU_GATE = "1.0"
$env:DARE_LOCK     = "0"
$env:DARE_CLASS_BLOCK = "1"
$btn = "mc_bytetrack_cb"
Write-Output "########## RUN $btn (ByteTrack baseline, class-blocked) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $btn *> "$dare\_mc_$btn.log"
Write-Output "----- DONE $btn (exit $LASTEXITCODE) -----"

# ---- Run 2: DARE headline, class-blocked ----
Clear-DareEnv
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
$env:DARE_CLASS_BLOCK = "1"
$dan = "mc_dare_cb"
Write-Output "########## RUN $dan (DARE headline, class-blocked) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $dan *> "$dare\_mc_$dan.log"
Write-Output "----- DONE $dan (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring per-class... ##########"
& $py _score_multiclass.py $dan $btn
Write-Output "########## MC RE-MEASURE (CLASS-BLOCKED) COMPLETE in $((Get-Date) - $t0). ##########"
