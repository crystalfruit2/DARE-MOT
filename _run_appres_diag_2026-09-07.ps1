# DARE-MOT -- Round-3 candidate #1 cheap diagnostic: frame-level synchronized appearance shift.
# ============================================================================================
# Purpose: BEFORE writing any gate, test whether the mechanism the candidate assumes actually
# exists in the data. The claim is that per-track appearance-cost residuals are mutually
# independent under ordinary nuisance, but move TOGETHER when a scene-wide ground-sample-distance
# change (drone zoom / altitude / gimbal) hits every object in the same frame. If no sequence
# shows a shared-majority excursion of the cross-track median, the candidate is dead with zero
# tracker code written.
#
# This is ONE run of the existing CV headline config, unchanged except that DARE_APPRES_LOG is
# set. The logger (byte_tracker.py::_log_appearance_residuals) is write-only instrumentation --
# nothing it emits is read back by any association stage -- so the tracked output MUST come out
# byte-identical to the cached mc_dare_cv_rerun0803. That comparison is run at the end as a
# no-op proof; if it fails, the diagnostic is invalid and the logger has a bug.
#
# NOTE: no baseline rerun here on purpose -- this run reports NO metric and makes NO comparison
# claim. The fresh-baseline rule applies at FAZ 3, if the candidate survives this.
# ============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar"
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"

Set-Location $dare
$env:PYTHONPATH = $dare

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

$t0 = Get-Date
Write-Output "########## APPRES DIAGNOSTIC START (branch main @ $(git rev-parse --short HEAD)) ##########"

# ---- DARE headline, CV motion model (identical to mc_dare_cv_rerun0803) + residual logging ----
Clear-DareEnv
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
$env:DARE_APPRES_LOG="$dare\_appres_logs"
$n = "mc_dare_cv_appres"
Write-Output "########## RUN $n (CV headline + appearance-residual logging) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $n *> "$dare\_mc_$n.log"
Write-Output "----- DONE $n (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0) ##########"

# ---- No-op proof: tracked output must be byte-identical to the cached reference ----
Write-Output "########## NO-OP CHECK vs mc_dare_cv_rerun0803 ##########"
$a = "$dare\YOLOX_outputs\mc_dare_cv_appres\track_results"
$b = "$dare\YOLOX_outputs\mc_dare_cv_rerun0803\track_results"
$identical = $true
foreach ($f in (Get-ChildItem $b -Filter *.txt)) {
  $ha = (Get-FileHash "$a\$($f.Name)" -Algorithm SHA256).Hash
  $hb = (Get-FileHash "$b\$($f.Name)" -Algorithm SHA256).Hash
  if ($ha -eq $hb) { Write-Output "IDENTICAL  $($f.Name)" }
  else { Write-Output "*** DIFFERS *** $($f.Name)"; $identical = $false }
}
if ($identical) { Write-Output "NO-OP CONFIRMED: logger did not perturb tracking." }
else { Write-Output "NO-OP FAILED: logger perturbed tracking -- diagnostic invalid." }

Write-Output "########## APPRES DIAGNOSTIC COMPLETE in $((Get-Date) - $t0) ##########"
