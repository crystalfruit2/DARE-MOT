# DARE-MOT -- 10-class + ignore-painted detector fine-tune (2026-09-10).
# Exp: exps/example/mot/yolox_x_visdrone_10c.py  (data: tools/convert_visdrone_10c.py -> train_10c.json / val7_10c.json)
# Init: built and CPU-verified by _make_10c_init.py (see experiment-log 2026-09-10 for which source and why).
# Waits until no tracker run is using the GPU (the CMC factorial queues), then trains alone.
# Output: YOLOX_outputs\<expn>\ (train_log.txt, best_ckpt = best val7 AP50 on the 5 evaluated classes).
param(
  [Parameter(Mandatory=$true)][string]$Init,
  [Parameter(Mandatory=$true)][string]$Expn,
  [int]$Epochs = 3
)
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
Set-Location $main

Write-Output "waiting for tracker queues to finish... ($(Get-Date -Format 'HH:mm'))"
while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
       Where-Object { $_.CommandLine -like "*_run_cmc_factorial*" -or $_.CommandLine -like "*_run_cmcfix_*" }) {
  Start-Sleep -Seconds 30
}
Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
  ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
$env:DARE_10C_EPOCHS = "$Epochs"
Write-Output "########## TRAIN $Expn  init=$Init  epochs=$Epochs  start $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
& $py tools/train.py -f exps/example/mot/yolox_x_visdrone_10c.py -d 1 -b 4 --fp16 -c $Init -expn $Expn
Write-Output "########## TRAIN DONE (exit $LASTEXITCODE) $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
