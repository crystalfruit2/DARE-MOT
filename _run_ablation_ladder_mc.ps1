# DARE-MOT -- professor-requested appearance-aggregation ablation ladder, RERUN on the CURRENT
# version (multi-class 5-class detector + IoU-feasibility gate), superseding the 2026-07-20 run
# which predates both. Isolates the appearance TEMPLATE-UPDATE rule only.
#
# What changed vs the original _run_ablation_ladder.ps1:
#   - 5-class detector (yolox_x_visdrone_mc_val7.py / YOLOX_outputs/yolox_x_visdrone_mc) instead
#     of the old single-class (class-leaked) detector.
#   - DARE_IOU_GATE=0.95 applied as a FIXED backdrop on every row (now a load-bearing default,
#     didn't exist on 2026-07-20). Not itself an ablation axis here -- Move 0c already covers that.
#   - ReID weights -> fine-tuned (osnet_ain_x1_0_visdrone_ft.pth), matching current production
#     (the pedestrian-OOD caveat was diagnosed and resolved as a non-issue).
#   - Reference row (lad_dare_full) uses LOCK=0, matching the current validated default (hard
#     KF-lock was found net-negative separately). The lock-ablation row is inverted to
#     "lad_lock_on" (tests turning lock BACK ON) since "off" is no longer a deviation from default.
#   - Size gate (DARE_LAMBDA_GATE) stays OFF on every row -- unchanged design intent: isolates the
#     aggregation rule cleanly, not muddied by the gate zeroing lambda on small boxes.
#   - Scored per-class with _score_multiclass.py (clean 5-class GT) instead of the old ped/allclass
#     dual score against the class-leaked GT (that fork question is now moot).
#
# All runs --seed 0 (pipeline verified fully deterministic + seed-invariant, 2026-07-27).
# Tags prefixed mc_lad_* so results land in new YOLOX_outputs dirs and don't clobber the
# 2026-07-20 single-class lad_* runs (kept for history).
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = if ($args.Count -ge 1) { $args[0] } else { "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" }
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$dare\_ablation_ladder_mc"
New-Item -ItemType Directory -Force -Path $out | Out-Null
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

# Per-row knobs. Empty string = leave at default. Reset ALL ablation knobs each row so no bleed.
#   AGG_ORDER (2=N=2 dynamic default, 1=order-1 dynamic t-1/t)
#   STATIC_EMA (-1=off, >=0 fixed gamma order-1 EMA)
#   STATIC_GAMMAS ("" = off, "g2,g1,g0" fixed-weight N=2)
#   LOCK (1=hard KF lock on, 0=off -- current validated default)
$configs = @(
  @{tag="mc_lad_dare_full"; ORDER="2"; SEMA="-1";   SG="";             LOCK="0"},  # reference: N=2 dynamic, lock off (current default)
  @{tag="mc_lad_no_filter"; ORDER="2"; SEMA="0";     SG="";             LOCK="0"},  # F^t=f^t, no EMA (professors' literal ablation)
  @{tag="mc_lad_ema_g090";  ORDER="2"; SEMA="0.9";   SG="";             LOCK="0"},  # order-1 static EMA, gamma const
  @{tag="mc_lad_ema_g070";  ORDER="2"; SEMA="0.7";   SG="";             LOCK="0"},  # gamma sweep
  @{tag="mc_lad_ema_g080";  ORDER="2"; SEMA="0.8";   SG="";             LOCK="0"},  # gamma sweep
  @{tag="mc_lad_ema_g095";  ORDER="2"; SEMA="0.95";  SG="";             LOCK="0"},  # gamma sweep
  @{tag="mc_lad_dyn_order1";ORDER="1"; SEMA="-1";    SG="";             LOCK="0"},  # order-1 dynamic window (t-1,t)
  @{tag="mc_lad_static_n2"; ORDER="2"; SEMA="-1";    SG="0.1,0.3,0.6";  LOCK="0"},  # fixed-weight N=2 EMA
  @{tag="mc_lad_lock_on";   ORDER="2"; SEMA="-1";    SG="";             LOCK="1"}   # reverse ablation: hard-lock back ON
)

Write-Output "########## MC ABLATION LADDER START ($($configs.Count) configs, detector = $ckpt) ##########"
$t0 = Get-Date
foreach ($c in $configs) {
  Clear-DareEnv
  # ---- fixed appearance backdrop (all rows): current production ReID, size gate OFF, IoU-gate ON ----
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="none"; $env:DARE_GATE_LO="0"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_IOU_GATE="0.95"
  # ---- per-row aggregation-rule knob ----
  $env:DARE_AGG_ORDER=$c.ORDER; $env:DARE_STATIC_EMA=$c.SEMA
  $env:DARE_STATIC_GAMMAS=$c.SG; $env:DARE_LOCK=$c.LOCK
  Write-Output "########## RUN $($c.tag)  ORDER=$($c.ORDER) SEMA=$($c.SEMA) SG='$($c.SG)' LOCK=$($c.LOCK)  ($((Get-Date) - $t0) elapsed) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $($c.tag) *> "$out\log_$($c.tag).log"
  Write-Output "----- DONE $($c.tag) (exit $LASTEXITCODE) -----"
}
Clear-DareEnv
Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring per-class... ##########"

# ---- score every row on its own, then each vs the ladder's own reference row ----
$summary = "$out\_ladder_scores_mc.txt"
Remove-Item $summary -ErrorAction SilentlyContinue
& $py _score_multiclass.py "mc_lad_dare_full" *>> $summary
foreach ($c in $configs) {
  if ($c.tag -eq "mc_lad_dare_full") { continue }
  & $py _score_multiclass.py $($c.tag) "mc_lad_dare_full" *>> $summary
}
Write-Output "########## MC LADDER COMPLETE in $((Get-Date) - $t0). Scores -> $summary ##########"
Get-Content $summary
