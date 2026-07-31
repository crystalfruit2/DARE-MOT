# DARE-MOT -- density gate joint radius+strength follow-up (2026-07-31, same day).
# ============================================================================================
# Stage-1 coordinate sweep found BOTH axes independently beat the untuned reference
# (r=3.0/s=2.0, -10 IDSw): larger radius (r=4-6 -> -13) and lower strength (s=1.0-1.5 -> -14),
# each plateauing at its own end of the tested range. Since they push the same direction and
# were never tried together, check whether they stack.
#
# Usage: _run_density_sweep2.ps1 [ckpt]
# ============================================================================================
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
}

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Run-Density($radius, $strength, $expn) {
  Clear-DareEnv; Set-HeadlineEnv
  $env:DARE_DENSITY_GATE="boost"; $env:DARE_DENSITY_RADIUS="$radius"; $env:DARE_DENSITY_STRENGTH="$strength"
  Write-Output "########## RUN $expn (radius=$radius, strength=$strength) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $expn *> "$dare\_mc_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  Clear-DareEnv
}

$t0 = Get-Date
Write-Output "########## DENSITY GATE JOINT FOLLOW-UP START (detector = $ckpt) ##########"
Run-Density "6.0" "1.0"  "mc_dare_density_r60s10"
Run-Density "8.0" "0.75" "mc_dare_density_r80s075"
Write-Output "########## ALL RUNS DONE in $((Get-Date) - $t0). Scoring... ##########"
foreach ($expn in @("mc_dare_density_r60s10","mc_dare_density_r80s075")) {
  Write-Output "===== $expn vs headline (mc_dare) ====="
  & $py _score_multiclass.py $expn mc_dare
}
Write-Output "########## FOLLOW-UP COMPLETE in $((Get-Date) - $t0). ##########"
