# DARE-MOT -- density gate radius/strength sweep (2026-07-31 follow-up).
# ============================================================================================
# The 07-31 first test (radius=3.0, strength=2.0, both untuned first guesses) gave a small,
# stratification-confirmed-real win in 'boost' mode: -10 IDSw / +0.9%/+0.6% IDF1 (macro/micro).
# 'suppress' was conclusively rejected same day (clearly harmful) -- not swept further.
# Same pattern as the CA accel-noise sweep: coordinate sweep from the untuned reference point,
# one parameter at a time. Reference (r=3.0, s=2.0) already run as mc_dare_density_boost --
# not re-run here.
#
# Stage 1: radius sweep, strength fixed at reference (2.0).
# Stage 2: strength sweep, radius fixed at reference (3.0).
# (A joint re-sweep around whichever axis wins can follow if warranted.)
#
# Usage: _run_density_sweep.ps1 [ckpt]
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
Write-Output "########## DENSITY GATE RADIUS/STRENGTH SWEEP START (detector = $ckpt) ##########"

Write-Output "===== STAGE 1: radius sweep (strength=2.0 fixed; reference r=3.0 already run) ====="
Run-Density "1.5" "2.0" "mc_dare_density_r15"
Run-Density "2.0" "2.0" "mc_dare_density_r20"
Run-Density "4.0" "2.0" "mc_dare_density_r40"
Run-Density "6.0" "2.0" "mc_dare_density_r60"

Write-Output "===== STAGE 2: strength sweep (radius=3.0 fixed; reference s=2.0 already run) ====="
Run-Density "3.0" "1.0" "mc_dare_density_s10"
Run-Density "3.0" "1.5" "mc_dare_density_s15"
Run-Density "3.0" "3.0" "mc_dare_density_s30"
Run-Density "3.0" "4.0" "mc_dare_density_s40"

Write-Output "########## ALL RUNS DONE in $((Get-Date) - $t0). Scoring... ##########"
foreach ($expn in @("mc_dare_density_r15","mc_dare_density_r20","mc_dare_density_r40","mc_dare_density_r60",
                     "mc_dare_density_s10","mc_dare_density_s15","mc_dare_density_s30","mc_dare_density_s40")) {
  Write-Output "===== $expn vs headline (mc_dare) ====="
  & $py _score_multiclass.py $expn mc_dare
}
Write-Output "########## DENSITY SWEEP COMPLETE in $((Get-Date) - $t0). ##########"
