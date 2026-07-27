# DARE-MOT — Move 0b: noise band. Run BOTH arms across N seeds so the -80 IDSw / +1.4 IDF1 /
# ~0 MOTA headline gets an error band, and the truck/bus single-run wobble can be separated from
# a real effect. Also runs DARE seed-0 a SECOND time (mc_nb_dare_s0_rep) as a determinism check.
# =============================================================================================
# Determinism: track.py sets cudnn.deterministic=True, cudnn.benchmark=False,
# use_deterministic_algorithms(True, warn_only=True) whenever --seed is passed. warn_only means a
# op with no deterministic kernel silently falls back, and --fp16 is on -> reproducibility is
# PLAUSIBLE but must be checked empirically, which the s0_rep run does.
#
# Configs are copied verbatim from _run_mc_remeasure.ps1 (ByteTrack Run 1 / DARE Run 2); ONLY the
# --seed value and the -expn change per iteration.
#
# Usage:  _run_mc_noiseband.ps1 [ckpt] [nseeds]      (default nseeds=5 -> seeds 0..4)
#         ~11 tracking runs (5 BT + 5 DARE + 1 rep) x 7 seqs. Long -- run in background.
# =============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = if ($args.Count -ge 1) { $args[0] } else { "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" }
$nseeds = if ($args.Count -ge 2) { [int]$args[1] } else { 5 }
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"

Set-Location $dare
$env:PYTHONPATH = $dare
if (-not (Test-Path $ckpt)) { Write-Output "MISSING detector ckpt: $ckpt"; exit 1 }
if (-not (Test-Path $ft))   { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

function Run-ByteTrack($seed, $expn) {
  Clear-DareEnv
  $env:DARE_LAMBDA = "0.0"; $env:DARE_IOU_GATE = "1.0"; $env:DARE_LOCK = "0"
  $flags = @("-b","1","-d","1","--fp16","--fuse","--seed","$seed",
             "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
  Write-Output "########## RUN $expn (ByteTrack, seed $seed) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $expn *> "$dare\_mc_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  Clear-DareEnv
}

function Run-Dare($seed, $expn) {
  Clear-DareEnv
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
  $flags = @("-b","1","-d","1","--fp16","--fuse","--seed","$seed",
             "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
  Write-Output "########## RUN $expn (DARE, seed $seed) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $expn *> "$dare\_mc_$expn.log"
  Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  Clear-DareEnv
}

$t0 = Get-Date
Write-Output "########## MC NOISE BAND START ($nseeds seeds, detector = $ckpt) ##########"
for ($s = 0; $s -lt $nseeds; $s++) {
  Run-ByteTrack $s "mc_nb_bt_s$s"
  Run-Dare      $s "mc_nb_dare_s$s"
}
# determinism check: DARE seed 0, run again
Run-Dare 0 "mc_nb_dare_s0_rep"

Write-Output "########## ALL RUNS DONE in $((Get-Date) - $t0). Aggregating... ##########"
$seedArgs = 0..($nseeds-1)
& $py _score_noiseband.py @seedArgs
Write-Output "########## MC NOISE BAND COMPLETE in $((Get-Date) - $t0). ##########"
