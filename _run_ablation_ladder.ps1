# DARE-MOT — professor-requested baseline/ablation ladder (meeting-notes-2026-07-16, due 2026-07-19).
# Isolates the appearance TEMPLATE-UPDATE rule. Backdrop held fixed across every row:
#   OSNet-AIN-DG embedding, lambda=0.5, NO size gate (LAMBDA_GATE=none) so appearance is ON
#   everywhere and the aggregation effect isn't muddied by the gate zeroing lambda on small boxes.
#   The gated headline is a separate axis (see _run_gate_sweep.ps1).
# Each row changes exactly ONE knob off the DARE-full reference (except no_filter = professors'
# "no gate/lock at all" definition, which flips EMA->raw AND lock off together).
# All runs --seed 0 (determinism verified 7/7). Each expn scored ped-only AND all-class (Fork A/B open).
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$out  = "$dare\_ablation_ladder"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare

# ---- fixed appearance backdrop (all rows) ----
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"
$env:DARE_REID_WEIGHTS="$dare\reid_weights\osnet_ain_x1_0_dg_clean.pth"
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="none"; $env:DARE_GATE_LO="0"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"

# Per-row knobs. Empty string = leave at default. Reset ALL ablation knobs each row so no bleed.
#   AGG_ORDER (2=N=2 dynamic default, 1=order-1 dynamic t-1/t)
#   STATIC_EMA (-1=off, >=0 fixed gamma order-1 EMA)
#   STATIC_GAMMAS ("" = off, "g2,g1,g0" fixed-weight N=2)
#   LOCK (1=hard KF lock on, 0=off)
$configs = @(
  @{tag="lad_dare_full";     ORDER="2"; SEMA="-1";   SG="";        LOCK="1"},  # reference: N=2 dynamic + lock
  @{tag="lad_no_filter";     ORDER="2"; SEMA="0";    SG="";        LOCK="0"},  # F^t=f^t, no lock
  @{tag="lad_ema_g090";      ORDER="2"; SEMA="0.9";  SG="";        LOCK="1"},  # order-1 static EMA, gamma const
  @{tag="lad_ema_g070";      ORDER="2"; SEMA="0.7";  SG="";        LOCK="1"},  # gamma sweep
  @{tag="lad_ema_g080";      ORDER="2"; SEMA="0.8";  SG="";        LOCK="1"},  # gamma sweep
  @{tag="lad_ema_g095";      ORDER="2"; SEMA="0.95"; SG="";        LOCK="1"},  # gamma sweep
  @{tag="lad_dyn_order1";    ORDER="1"; SEMA="-1";   SG="";        LOCK="1"},  # order-1 dynamic (t-1,t)
  @{tag="lad_static_n2";     ORDER="2"; SEMA="-1";   SG="0.1,0.3,0.6"; LOCK="1"},  # fixed-weight N=2
  @{tag="lad_lock_off";      ORDER="2"; SEMA="-1";   SG="";        LOCK="0"}   # hard-lock KF ablation
)

Write-Output "########## ABLATION LADDER START ($($configs.Count) configs) ##########"
$t0 = Get-Date
foreach ($c in $configs) {
  $env:DARE_AGG_ORDER=$c.ORDER; $env:DARE_STATIC_EMA=$c.SEMA
  $env:DARE_STATIC_GAMMAS=$c.SG; $env:DARE_LOCK=$c.LOCK
  Write-Output "########## RUN $($c.tag)  ORDER=$($c.ORDER) SEMA=$($c.SEMA) SG='$($c.SG)' LOCK=$($c.LOCK)  ($((Get-Date) - $t0) elapsed) ##########"
  & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --seed 0 `
      --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
      -expn $($c.tag) *> "$out\log_$($c.tag).log"
  Write-Output "----- DONE $($c.tag) (exit $LASTEXITCODE) -----"
}
Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring... ##########"

# ---- score every row both ways ----
$summary = "$out\_ladder_scores.txt"
Remove-Item $summary -ErrorAction SilentlyContinue
foreach ($c in $configs) {
  foreach ($mode in @("ped","allclass")) {
    & $py _score_ped.py $($c.tag) $mode *>> $summary
  }
}
Write-Output "########## LADDER COMPLETE in $((Get-Date) - $t0). Scores -> $summary ##########"
Get-Content $summary
