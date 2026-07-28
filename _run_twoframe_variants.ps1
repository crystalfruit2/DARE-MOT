# DARE-MOT — Two-frame IoU gate, full variant sweep (2026-07-28 evening).
# =============================================================================================
# Follow-up to _run_twoframe_gate.ps1 (dynamic/static-0.7 both negative), after an Opus
# brainstorm pass found (a) a real bug -- coast_box_two_steps hard-coded a 2-step horizon
# regardless of actual track staleness, now fixed to use the true elapsed-frame count -- and
# (b) a prioritized list of combination variants closing the family properly. This script runs:
#   V1+V2: static-blend alpha sweep {1.0, 0.9, 0.8, 0.7, 0.5, 0.25, 0.0} (1.0=pure iou_t1 sanity
#          check, 0.0=pure iou_t2 signal-quality check) with DARE_DIAG=1 to also capture the
#          gate rejection-rate decomposition.
#   V3:    'resid' -- per-track hard fallback to iou_t2 only when the track's last correction
#          was itself a kinematic outlier (the only mode live in the theoretically-favorable regime).
#   V4:    'max' -- AND-gate (stricter), uncalibrated tau2 (defaults to DARE_IOU_GATE).
#   V5:    'min' -- OR-gate (more lenient) = "take the maximum IoU / most optimistic per pair".
#   V6:    'cov' -- inverse-KF-covariance-trace weighted blend.
#   V7:    'scale' -- only blend for tracks whose box area exceeds DARE_TWOFRAME_SCALE_MIN.
#   V8:    'age' -- only blend for tracks with tracklet_len >= DARE_TWOFRAME_AGE_MIN.
#   V10:   'disagree' -- veto pairs where iou_t1 and iou_t2 disagree by more than a delta.
# All on the validated DARE headline config, DARE_DIAG=1 throughout (prints coast-win-rate and
# gate-reject-rate diagnostics per sequence -- see print_diag_summary in byte_tracker.py).
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
  $env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
}

$t0 = Get-Date
Write-Output "########## TWO-FRAME VARIANT SWEEP START (detector = $ckpt) ##########"

# --- V1+V2: static alpha sweep, including the two boundary/sanity points ---
$alphas = @("1.0","0.9","0.8","0.7","0.5","0.25","0.0")
foreach ($a in $alphas) {
  Clear-DareEnv; Set-HeadlineEnv
  $env:DARE_TWOFRAME_MODE="static"; $env:DARE_TWOFRAME_ALPHA="$a"
  $run = "mc_2f_static_a$($a -replace '\.','')"
  Write-Output "########## RUN $run (static alpha=$a) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $run *> "$dare\_mc_$run.log"
  Write-Output "----- DONE $run (exit $LASTEXITCODE) -----"
}

# --- V3: resid (the theoretically-favorable regime) ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_MODE="resid"; $env:DARE_TWOFRAME_RESID_THRESH="0.3"
Write-Output "########## RUN mc_2f_resid ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_2f_resid *> "$dare\_mc_mc_2f_resid.log"
Write-Output "----- DONE mc_2f_resid (exit $LASTEXITCODE) -----"

# --- V4: max (AND-gate, uncalibrated) ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_MODE="max"
Write-Output "########## RUN mc_2f_max ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_2f_max *> "$dare\_mc_mc_2f_max.log"
Write-Output "----- DONE mc_2f_max (exit $LASTEXITCODE) -----"

# --- V5: min (OR-gate) ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_MODE="min"
Write-Output "########## RUN mc_2f_min ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_2f_min *> "$dare\_mc_mc_2f_min.log"
Write-Output "----- DONE mc_2f_min (exit $LASTEXITCODE) -----"

# --- V6: cov (inverse-covariance-trace weighted) ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_MODE="cov"
Write-Output "########## RUN mc_2f_cov ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_2f_cov *> "$dare\_mc_mc_2f_cov.log"
Write-Output "----- DONE mc_2f_cov (exit $LASTEXITCODE) -----"

# --- V7: scale-conditioned ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_MODE="scale"; $env:DARE_TWOFRAME_SCALE_MIN="2500"; $env:DARE_TWOFRAME_ALPHA="0.7"
Write-Output "########## RUN mc_2f_scale ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_2f_scale *> "$dare\_mc_mc_2f_scale.log"
Write-Output "----- DONE mc_2f_scale (exit $LASTEXITCODE) -----"

# --- V8: age-conditioned ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_MODE="age"; $env:DARE_TWOFRAME_AGE_MIN="10"; $env:DARE_TWOFRAME_ALPHA="0.7"
Write-Output "########## RUN mc_2f_age ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_2f_age *> "$dare\_mc_mc_2f_age.log"
Write-Output "----- DONE mc_2f_age (exit $LASTEXITCODE) -----"

# --- V10: disagreement veto ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_MODE="disagree"; $env:DARE_TWOFRAME_DISAGREE_DELTA="0.3"
Write-Output "########## RUN mc_2f_disagree ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_2f_disagree *> "$dare\_mc_mc_2f_disagree.log"
Write-Output "----- DONE mc_2f_disagree (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring all variants vs DARE headline... ##########"
$allRuns = @()
foreach ($a in $alphas) { $allRuns += "mc_2f_static_a$($a -replace '\.','')" }
$allRuns += @("mc_2f_resid","mc_2f_max","mc_2f_min","mc_2f_cov","mc_2f_scale","mc_2f_age","mc_2f_disagree")
foreach ($run in $allRuns) {
  Write-Output "===== $run vs DARE (headline) ====="
  & $py _score_multiclass.py $run mc_dare
}
Write-Output "########## TWO-FRAME VARIANT SWEEP COMPLETE in $((Get-Date) - $t0). ##########"
