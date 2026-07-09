# DARE-MOT ablation driver — Phase 0 (confirm A vs B) + Phase 1 (ablation ladder).
# Each config sets env knobs (read by yolox/tracker/byte_tracker.py), runs the tracker on the
# held-out sequence with IDENTICAL detector+params, renames the result, and scores with the
# shared eval_mot.py. Defaults reproduce the validated run.
$ErrorActionPreference = "Stop"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$ckpt = "$dare\YOLOX_outputs\yolox_x_visdrone_finetune\best_ckpt.pth.tar"
$exp  = "exps/example/mot/yolox_x_visdrone_finetune.py"
$res  = "$dare\YOLOX_outputs\yolox_x_visdrone_finetune\track_results\uav0000339_00001_v.txt"
$gt   = "C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val\uav0000339_00001_v\gt\gt.txt"
$eval = "C:\Users\User\Desktop\projects\ByteTrack-baseline\eval_mot.py"
$out  = "$dare\_ablation"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare

# tag, AGG, BETA, TAU, STATIC_EMA, LOCK, LAMBDA
$configs = @(
  @{tag="dare_A_lockon";  AGG="A"; BETA="4.0"; TAU="0.5"; SE="-1";  LOCK="1"; LAM="0.5"},
  @{tag="dare_A_lockoff"; AGG="A"; BETA="4.0"; TAU="0.5"; SE="-1";  LOCK="0"; LAM="0.5"},
  @{tag="static_ema_off"; AGG="A"; BETA="4.0"; TAU="0.5"; SE="0.9"; LOCK="0"; LAM="0.5"},
  @{tag="dare_lambda0";   AGG="A"; BETA="4.0"; TAU="0.5"; SE="-1";  LOCK="1"; LAM="0.0"}
)

foreach ($c in $configs) {
  $env:DARE_AGG=$c.AGG; $env:DARE_BETA=$c.BETA; $env:DARE_TAU=$c.TAU
  $env:DARE_STATIC_EMA=$c.SE; $env:DARE_LOCK=$c.LOCK; $env:DARE_LAMBDA=$c.LAM
  Write-Output "########## RUN $($c.tag)  AGG=$($c.AGG) LOCK=$($c.LOCK) SE=$($c.SE) LAM=$($c.LAM) ##########"
  & $py tools/track.py -f $exp -c $ckpt -b 1 -d 1 --fp16 --fuse --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 *> "$out\log_$($c.tag).log"
  Copy-Item $res "$out\$($c.tag).txt" -Force
  Write-Output "----- SCORE $($c.tag) -----"
  & $py $eval "$out\$($c.tag).txt" $gt $($c.tag)
}
Write-Output "########## ABLATION BATCH COMPLETE ##########"
