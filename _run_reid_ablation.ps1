# DARE-MOT: does a REAL ReID embedding fix the appearance branch?
# Three configs, all seeded (determinism now fixed in track.py: cudnn.benchmark=False
# under --seed), 2 reps each to CONFIRM the fix gives byte-identical runs.
#   bt_l0    appearance OFF (DARE_LAMBDA=0)                 -> ByteTrack matching baseline
#   mob_l05  MobileNet ImageNet features (DARE_REID=mobilenet, LAMBDA=0.5) -> the known-bad
#   osn_l05  OSNet MSMT17 ReID embedding (DARE_REID=osnet,  LAMBDA=0.5)    -> the intended fix
# Verdict metric = pedestrian-only IDSw (score with _score_ped.py). osn_l05 < bt_l0 => a
# proper ReID embedding makes appearance HELP -> positive result.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
$out  = "$dare\_reid_ablation"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare
# appearance-map knobs at defaults (crop_shrink still applies to osnet; pool/center do not)
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"

$configs = @(
  @{tag="bt_l0";   REID="mobilenet"; LAMBDA="0.0"},
  @{tag="mob_l05"; REID="mobilenet"; LAMBDA="0.5"},
  @{tag="osn_l05"; REID="osnet";     LAMBDA="0.5"}
)

Write-Output "########## REID ABLATION START (3 configs x 2 reps, seeded, benchmark=False) ##########"
$t0 = Get-Date
foreach ($c in $configs) {
  $env:DARE_REID = $c.REID; $env:DARE_LAMBDA = $c.LAMBDA
  for ($r = 1; $r -le 2; $r++) {
    $expn = "$($c.tag)_r$r"
    Write-Output "########## RUN $expn  REID=$($c.REID) LAMBDA=$($c.LAMBDA)  ($((Get-Date) - $t0) elapsed) ##########"
    & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --seed 0 `
        --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 `
        -expn $expn *> "$out\log_$expn.log"
    Write-Output "----- DONE $expn (exit $LASTEXITCODE) -----"
  }
}
Write-Output "########## REID ABLATION COMPLETE in $((Get-Date) - $t0) ##########"
