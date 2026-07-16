# DARE-MOT: seeded appearance-ON vs appearance-OFF, to get the missing ByteTrack
# pedestrian-only baseline that decides fork A vs B (see class-leak §8b / RESUME).
#
# Two configs, differing ONLY in DARE_LAMBDA (appearance weight in the fused cost,
# byte_tracker.py:398). Everything else at DARE-MOT defaults => a clean controlled
# ablation of the appearance branch, not a cross-repo ByteTrack comparison:
#   dare_l05  DARE_LAMBDA=0.5  appearance-ON  (DARE-MOT)
#   bt_l0     DARE_LAMBDA=0.0  appearance-OFF (pure IoU == ByteTrack matching)
#
# --seed 0 turns on cudnn.deterministic (track.py:132). Each config runs TWICE
# (r1/r2) so we can verify the predictions are byte-identical => seed actually
# removes the non-determinism that broke the earlier ablation. If r1 != r2 the
# seed isn't enough (cudnn.benchmark stays True) and we escalate to benchmark=False.
#
# Continue (not Stop): loguru->stderr is wrapped as NativeCommandError by PS 5.1.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$out  = "$dare\_bytetrack_ped_check"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare

# defaults for every appearance knob; only LAMBDA differs between configs
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"

$configs = @(
  @{tag="bt_l0";    LAMBDA="0.0"},
  @{tag="dare_l05"; LAMBDA="0.5"}
)

Write-Output "########## SEEDED PED-CHECK START (2 configs x 2 reps, seed 0) ##########"
$t0 = Get-Date
foreach ($c in $configs) {
  $env:DARE_LAMBDA = $c.LAMBDA
  for ($r = 1; $r -le 2; $r++) {
    $expn = "$($c.tag)_s0_r$r"
    Write-Output "########## RUN $expn  LAMBDA=$($c.LAMBDA)  ($((Get-Date) - $t0) elapsed) ##########"
    & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --seed 0 `
        --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
        -expn $expn *> "$out\log_$expn.log"
    Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  }
}
Write-Output "########## PED-CHECK COMPLETE in $((Get-Date) - $t0) ##########"
