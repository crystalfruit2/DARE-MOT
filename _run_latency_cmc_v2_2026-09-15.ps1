# DARE-MOT -- latency WITH scale CMC, v2 instrumentation (2026-09-15).
# v1 (_run_latency_cmc_2026-09-15.ps1) timed only _extract_features_osnet. The ByteTrack arm runs with
# DARE_REID unset, so BYTETracker.update() still executes the legacy MobileNetV2 per-crop extractor (its
# features are unused at lambda = 0) and v1 charged that to "association" (369 ms/frame, invalid).
# v2 times the _extract_features dispatch. Both arms re-run: ByteTrack for the valid number, CV-DARE as a
# consistency check against v1 (its embedding path was timed correctly; numbers should agree within noise).
# Waits for the Path-1 GPU queue to finish so nothing shares the machine while timing.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\latency_cmc"

$gate = "$main\_scratch\path1_master.log"
Write-Output "########## LATENCY+CMC v2 QUEUED $(Get-Date -Format 'HH:mm') -- waiting for PATH1 DONE ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'PATH1 DONE' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20
Write-Output "########## LATENCY+CMC v2 START $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"

Set-Location $wt
$env:PYTHONPATH = "$lapd;$wt"
function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}
function Set-DareHeadline {
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
}
function Set-BT { $env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0" }
function Set-ScaleCMC { $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="scale" }

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
$runs = @(
  @{ name="lat10v2_bt_cmcscale_j";      ref="mc10_bt_cmcscale_j";      cfg={ Set-BT; Set-ScaleCMC } },
  @{ name="lat10v2_dare_cv_cmcscale_j"; ref="mc10_dare_cv_cmcscale_j"; cfg={ Set-DareHeadline; Set-ScaleCMC } }
)
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg; $env:DARE_MAX_CLASS = "4"
  Write-Output "########## ARM $($r.name) ##########"
  & $py _latency_cmc_2026-09-15.py --tag $r.name -- -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  Write-Output "----- DONE $($r.name) (exit $LASTEXITCODE) $(Get-Date -Format 'HH:mm') -----"
}
Clear-DareEnv
$diffs = 0; $n = 0
foreach ($r in $runs) {
  foreach ($f in Get-ChildItem "$wt\YOLOX_outputs\$($r.ref)\track_results" -Filter *.txt) {
    $n++
    $new = "$wt\YOLOX_outputs\$($r.name)\track_results\$($f.Name)"
    if (-not ((Test-Path $new) -and (Get-FileHash $f.FullName -Algorithm SHA256).Hash -eq (Get-FileHash $new -Algorithm SHA256).Hash)) { $diffs++; Write-Output "DIFF/MISSING $($r.name)\$($f.Name)" }
  }
}
Write-Output "########## IDENTITY: $($n - $diffs)/$n byte-identical ##########"
foreach ($r in $runs) { $j = "$out\latency_$($r.name).json"; if (Test-Path $j) { Write-Output "--- $($r.name) ---"; Get-Content $j } }
Write-Output "########## LATENCY+CMC v2 DONE $(Get-Date -Format 'HH:mm') ##########"
