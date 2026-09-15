# DARE-MOT -- Path-1 follow-up: BoT-SORT WITHOUT ReID (GMC + BYTE only) on D3 (2026-09-15).
# Why (independent review of the Move-2 table): BoT-SORT-ReID's IDSw excess over ByteTrack + scale CMC sits
# almost entirely on uav0000305 (21 vs 6), and so does Deep OC-SORT's (20). Both use ungated ReID; both
# CMC-without-free-appearance arms get 5-6. With R ~ sI on 305 (|theta| ~ 1e-4) BoT-SORT's warp is
# numerically the scale-exact warp, so the warp is an unlikely cause. This arm separates the two:
#   305 drops to ~6  -> the excess is BoT-SORT's appearance term, not its CMC
#   305 stays ~20    -> the excess is in BoT-SORT's motion/CMC/KF path
# Same detector, thresholds and GMC as p1_botsort; only with_reid differs. Queued behind the v2 latency run.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$out  = "$main\_scratch\path1"

$gate = "$main\_scratch\latency_cmc_v2_master.log"
Write-Output "########## BOT-NOREID QUEUED $(Get-Date -Format 'HH:mm') -- waiting for LATENCY+CMC v2 DONE ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'LATENCY\+CMC v2 DONE' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20

Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } | ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
$env:DARE_TRACKER = "botsort"; $env:DARE_P1_BOT_REID = "0"; $env:DARE_MAX_CLASS = "4"
Set-Location $main
$env:PYTHONPATH = "$lapd;$main"
$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
Write-Output "########## RUN p1_botsort_noreid $(Get-Date -Format 'HH:mm') ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn p1_botsort_noreid *> "$out\p1_botsort_noreid.log"
$n = (Get-ChildItem "$main\YOLOX_outputs\p1_botsort_noreid\track_results" -Filter *.txt -ErrorAction SilentlyContinue).Count
Write-Output "----- DONE p1_botsort_noreid (exit $LASTEXITCODE, $n/7) $(Get-Date -Format 'HH:mm') -----"
Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } | ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
Write-Output "########## SCORE ##########"
& $py _score_path1_2026-09-15.py
Write-Output "########## BOT-NOREID DONE $(Get-Date -Format 'HH:mm') ##########"
