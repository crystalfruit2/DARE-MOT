# DARE-MOT -- fresh 3-way re-run for Farzad's 2026-08-03 ask: ByteTrack vs CV-DARE vs CA-DARE(tuned).
# ============================================================================================
# Standing rule (Alp, 2026-08-03): any reported comparison re-runs the baseline fresh, never
# reuses a cached number. Runs all three configs back-to-back on current `main` (post
# multiclass-migration merge), identical protocol to _run_mc_remeasure.ps1 / _run_ca_accel_sweep.ps1
# (val7, 5-class detector, seed 0). New expn names (suffix _rerun0803) so the prior cached
# results under YOLOX_outputs/{mc_bytetrack,mc_dare,mc_ca_accel_80} are left untouched for diff.
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
Write-Output "########## FARZAD 3-WAY RERUN START (detector = $ckpt, branch main @ $(git rev-parse --short HEAD)) ##########"

# ---- Run 1: ByteTrack baseline (appearance OFF, pure IoU; gate off, lock off) ----
Clear-DareEnv
$env:DARE_LAMBDA   = "0.0"
$env:DARE_IOU_GATE = "1.0"
$env:DARE_LOCK     = "0"
$btn = "mc_bytetrack_rerun0803"
Write-Output "########## RUN $btn (ByteTrack baseline, fresh) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $btn *> "$dare\_mc_$btn.log"
Write-Output "----- DONE $btn (exit $LASTEXITCODE) -----"

# ---- Run 2: DARE headline, CV motion model (default) ----
Clear-DareEnv
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
$cvn = "mc_dare_cv_rerun0803"
Write-Output "########## RUN $cvn (DARE headline, CV, fresh) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $cvn *> "$dare\_mc_$cvn.log"
Write-Output "----- DONE $cvn (exit $LASTEXITCODE) -----"

# ---- Run 3: DARE headline, CA motion model, tuned accel-noise (1/80 = 0.0125) ----
Clear-DareEnv
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
$env:DARE_KF_MODEL="ca"; $env:DARE_KF_ACCEL_NOISE="0.0125"
$can = "mc_dare_ca_rerun0803"
Write-Output "########## RUN $can (DARE headline, CA tuned 1/80, fresh) ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $can *> "$dare\_mc_$can.log"
Write-Output "----- DONE $can (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring... ##########"
& $py _score_multiclass.py $cvn $btn
Write-Output "---- CA vs ByteTrack ----"
& $py _score_multiclass.py $can $btn
Write-Output "---- CA vs CV ----"
& $py _score_multiclass.py $can $cvn
Write-Output "########## FARZAD 3-WAY RERUN COMPLETE in $((Get-Date) - $t0). ##########"
