# DARE-MOT fix-ablation REPLICATION runner — bounds the run-to-run noise floor.
#
# WHY: the single-shot ablation (_run_fix_ablation.ps1) is NOT trustworthy. The YOLOX
# detector is non-deterministic under these args (no --seed => cudnn.deterministic stays
# False while cudnn.benchmark=True, plus --fp16). Two byte-identical baseline runs produced
# 655 vs 708 all-class IDSw and prediction files that differ by 215 lines (-133 on
# uav0000182 alone). Every fix delta from the single run (crop15 -43, combo -40, ...) sits
# INSIDE that spread, so none is quotable from one run.
#
# WHAT THIS DOES: re-runs each of the 5 ablation configs R times (default 3), ROUND-ROBIN
# (rep-major, not config-major) so any thermal/driver drift over the night is spread evenly
# across configs instead of confounding one of them. Each run gets a unique expn
# rep{r}_{tag}; predictions land in YOLOX_outputs/rep{r}_{tag}/track_results, logs in
# _fix_ablation_rep\. NO CODE CHANGES — pure inference. Safe to run unattended.
#
# AFTER IT FINISHES: score every rep all-class AND pedestrian-only (cat==1 GT), then report
# per-config mean +/- spread. A fix only counts if its mean beats baseline by MORE than the
# baseline's own run-to-run spread. Optionally add a DARE_LAMBDA=0 config here to finally get
# the missing ByteTrack (appearance-off) baseline in the same non-determinism regime.
#
# Continue (not Stop): track.py logs via loguru to stderr, which PS 5.1 wraps as
# NativeCommandError; Stop would abort on the first log line. See _run_fix_ablation.ps1.
param([int]$Reps = 3)

$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$out  = "$dare\_fix_ablation_rep"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare

# Same 5 rungs as the single-shot ablation. Defaults (mean/0.0/-1) reproduce the baseline
# path; the point is to sample each rung REPEATEDLY, not to change any knob.
$configs = @(
  @{tag="baseline_refactor"; POOL="mean";   CROP="0.0";  REASSOC="-1"},
  @{tag="pool_center";       POOL="center"; CROP="0.0";  REASSOC="-1"},
  @{tag="crop15";            POOL="mean";   CROP="0.15"; REASSOC="-1"},
  @{tag="reassoc30";         POOL="mean";   CROP="0.0";  REASSOC="30"},
  @{tag="combo";             POOL="center"; CROP="0.15"; REASSOC="30"}
)

Write-Output "########## REPLICATION START: $Reps reps x $($configs.Count) configs = $($Reps * $configs.Count) runs ##########"
$t0 = Get-Date
for ($r = 1; $r -le $Reps; $r++) {
  foreach ($c in $configs) {
    $expn = "rep${r}_$($c.tag)"
    $env:DARE_POOL=$c.POOL; $env:DARE_CROP_SHRINK=$c.CROP; $env:DARE_REASSOC_MAX=$c.REASSOC
    Write-Output "########## RUN $expn  POOL=$($c.POOL) CROP=$($c.CROP) REASSOC=$($c.REASSOC)  ($((Get-Date) - $t0) elapsed) ##########"
    & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse `
        --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
        -expn $expn *> "$out\log_$expn.log"
    Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  }
}
Write-Output "########## REPLICATION COMPLETE in $((Get-Date) - $t0) ##########"
