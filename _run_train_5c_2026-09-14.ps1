# DARE-MOT -- clean 5-class CONTROL detector (2026-09-14).
# Isolates the class set from the init: identical to the 09-10 10-class run in every other respect
# (same MOT17 init, 3 epochs, b=4, fp16, same multiscale range, same ignore-region handling).
# Exp:  exps/example/mot/yolox_x_visdrone_5c_ctrl.py   (data: _make_5c_train_json.py -> train_5c.json)
# Init: YOLOX_outputs\_init\yolox_x_10c_from_mot17.pth.tar  (bytetrack_x_mot17 minus cls_preds)
# Out:  YOLOX_outputs\<expn>\ (train_log.txt, best_ckpt = best val7 AP50 on the 5 evaluated classes)
# ETA:  ~10 h (the 10-class run took 9 h 49 min on the same schedule).
# Reference points for the printed AP50: clean 10c ep3 = 56.0 · old 5c ep2 = 60.7 · old 5c ep8 = 54.1.
param(
  [string]$Init = "C:\Users\User\Desktop\projects\DARE-MOT\YOLOX_outputs\_init\yolox_x_10c_from_mot17.pth.tar",
  [string]$Expn = "yolox_x_visdrone_5c_ctrl_mot17init",
  [int]$Epochs = 3
)
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
Set-Location $main

# don't fight a tracker queue or the per-sequence AP job for the GPU
Write-Output "waiting for GPU jobs to finish... ($(Get-Date -Format 'HH:mm'))"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like "*tools/track.py*" -or $_.CommandLine -like "*_det_ap_perseq*" }) {
  Start-Sleep -Seconds 30
}
Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
  ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
$env:DARE_5C_EPOCHS = "$Epochs"
Write-Output "########## TRAIN $Expn  init=$Init  epochs=$Epochs  start $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
& $py tools/train.py -f exps/example/mot/yolox_x_visdrone_5c_ctrl.py -d 1 -b 4 --fp16 -c $Init -expn $Expn
Write-Output "########## TRAIN DONE (exit $LASTEXITCODE) $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
