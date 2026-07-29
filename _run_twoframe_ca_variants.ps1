# DARE-MOT — Two-frame IoU gate, CONSTANT-ACCELERATION KF variant (2026-07-29).
# =============================================================================================
# Prof Farzad follow-up after the 2026-07-28 negative result (_run_twoframe_variants.ps1, all 14
# CV-motion-model variants neutral-to-negative): did you try the t-2/t-1/t IoU gate under a
# constant-acceleration motion model instead of constant-velocity? The CV negative result's own
# theory doesn't cover this case (see prof-farzad-briefing-2026-07-28.md §9) -- under CV, t-1 is
# provably the MMSE-optimal 1-step prediction, so t-2 is redundant; under CA, the state has an
# acceleration DOF that a single prior point can't identify, so a third point genuinely carries
# information the filter can use. This is a real re-test, not a rerun of a settled question.
#
# Same sweep as the CV variant script, plus a CA-baseline run (twoframe gate OFF) to isolate
# whether switching the motion model alone -- independent of the two-frame gate question --
# changes anything. All runs add DARE_KF_MODEL=ca on top of Set-HeadlineEnv.
#   Run 0:      CA baseline, gate OFF                          -> mc_ca_baseline
#   V1+V2:      static-blend alpha sweep {1.0,0.9,0.8,0.7,0.5,0.25,0.0}
#   V3:         resid (hard fallback on kinematic outlier)
#   V4:         max (AND-gate)
#   V5:         min (OR-gate)
#   V6:         cov (inverse-KF-covariance-trace weighted)
#   V7:         scale-conditioned
#   V8:         age-conditioned
#   V10:        disagree (veto on signal disagreement)
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
  $env:DARE_KF_MODEL="ca"
}

$t0 = Get-Date
Write-Output "########## TWO-FRAME CA-MOTION VARIANT SWEEP START (detector = $ckpt) ##########"

# --- Run 0: CA baseline, gate OFF (isolates motion-model-alone effect) ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="0"; $env:DARE_DIAG="1"
Write-Output "########## RUN mc_ca_baseline (CA motion model, no two-frame gate) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_baseline *> "$dare\_mc_mc_ca_baseline.log"
Write-Output "----- DONE mc_ca_baseline (exit $LASTEXITCODE) -----"

# --- V1+V2: static alpha sweep, including the two boundary/sanity points ---
$alphas = @("1.0","0.9","0.8","0.7","0.5","0.25","0.0")
foreach ($a in $alphas) {
  Clear-DareEnv; Set-HeadlineEnv
  $env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
  $env:DARE_TWOFRAME_MODE="static"; $env:DARE_TWOFRAME_ALPHA="$a"
  $run = "mc_ca_2f_static_a$($a -replace '\.','')"
  Write-Output "########## RUN $run (CA, static alpha=$a) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $run *> "$dare\_mc_$run.log"
  Write-Output "----- DONE $run (exit $LASTEXITCODE) -----"
}

# --- V3: resid ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
$env:DARE_TWOFRAME_MODE="resid"; $env:DARE_TWOFRAME_RESID_THRESH="0.3"
Write-Output "########## RUN mc_ca_2f_resid ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_2f_resid *> "$dare\_mc_mc_ca_2f_resid.log"
Write-Output "----- DONE mc_ca_2f_resid (exit $LASTEXITCODE) -----"

# --- V4: max (AND-gate) ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
$env:DARE_TWOFRAME_MODE="max"
Write-Output "########## RUN mc_ca_2f_max ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_2f_max *> "$dare\_mc_mc_ca_2f_max.log"
Write-Output "----- DONE mc_ca_2f_max (exit $LASTEXITCODE) -----"

# --- V5: min (OR-gate) ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
$env:DARE_TWOFRAME_MODE="min"
Write-Output "########## RUN mc_ca_2f_min ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_2f_min *> "$dare\_mc_mc_ca_2f_min.log"
Write-Output "----- DONE mc_ca_2f_min (exit $LASTEXITCODE) -----"

# --- V6: cov ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
$env:DARE_TWOFRAME_MODE="cov"
Write-Output "########## RUN mc_ca_2f_cov ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_2f_cov *> "$dare\_mc_mc_ca_2f_cov.log"
Write-Output "----- DONE mc_ca_2f_cov (exit $LASTEXITCODE) -----"

# --- V7: scale-conditioned ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
$env:DARE_TWOFRAME_MODE="scale"; $env:DARE_TWOFRAME_SCALE_MIN="2500"; $env:DARE_TWOFRAME_ALPHA="0.7"
Write-Output "########## RUN mc_ca_2f_scale ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_2f_scale *> "$dare\_mc_mc_ca_2f_scale.log"
Write-Output "----- DONE mc_ca_2f_scale (exit $LASTEXITCODE) -----"

# --- V8: age-conditioned ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
$env:DARE_TWOFRAME_MODE="age"; $env:DARE_TWOFRAME_AGE_MIN="10"; $env:DARE_TWOFRAME_ALPHA="0.7"
Write-Output "########## RUN mc_ca_2f_age ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_2f_age *> "$dare\_mc_mc_ca_2f_age.log"
Write-Output "----- DONE mc_ca_2f_age (exit $LASTEXITCODE) -----"

# --- V10: disagreement veto ---
Clear-DareEnv; Set-HeadlineEnv
$env:DARE_TWOFRAME_GATE="1"; $env:DARE_DIAG="1"
$env:DARE_TWOFRAME_MODE="disagree"; $env:DARE_TWOFRAME_DISAGREE_DELTA="0.3"
Write-Output "########## RUN mc_ca_2f_disagree ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn mc_ca_2f_disagree *> "$dare\_mc_mc_ca_2f_disagree.log"
Write-Output "----- DONE mc_ca_2f_disagree (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring all CA variants vs CV headline and vs CA baseline... ##########"
$allRuns = @("mc_ca_baseline")
foreach ($a in $alphas) { $allRuns += "mc_ca_2f_static_a$($a -replace '\.','')" }
$allRuns += @("mc_ca_2f_resid","mc_ca_2f_max","mc_ca_2f_min","mc_ca_2f_cov","mc_ca_2f_scale","mc_ca_2f_age","mc_ca_2f_disagree")

Write-Output "===== mc_ca_baseline vs DARE (CV headline) ====="
& $py _score_multiclass.py mc_ca_baseline mc_dare

foreach ($run in $allRuns) {
  if ($run -eq "mc_ca_baseline") { continue }
  Write-Output "===== $run vs mc_ca_baseline (CA baseline, isolates the two-frame gate's own effect) ====="
  & $py _score_multiclass.py $run mc_ca_baseline
  Write-Output "===== $run vs DARE (CV headline, overall picture) ====="
  & $py _score_multiclass.py $run mc_dare
}
Write-Output "########## TWO-FRAME CA-MOTION VARIANT SWEEP COMPLETE in $((Get-Date) - $t0). ##########"
