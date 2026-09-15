# DARE-MOT -- night queue 2026-09-15: reviewer-proofing runs for the report (D3, val7, seed 0).
# Waits for the BoT-SORT no-ReID ablation to finish. Every arm = same detector, thresholds and harness as Path 1.
#   p1_deepocsort_mh1        Deep OC-SORT, min_hits 1 (upstream 3)            -> is its FN deficit the output rule?
#   p1_botsort_m08           BoT-SORT-ReID at upstream match_thresh 0.8        -> threshold sensitivity
#   p1_botsort_scalewarp     BoT-SORT-ReID with the scale-exact warp (xywh)    -> does our warp change BoT-SORT?
#   lat10_bt_cmcscale_j_ds4  ByteTrack + scale CMC, GMC downscale 4 (+latency) -> accuracy/latency trade
#   lat10_bt_cmcscale_j_ds8  ByteTrack + scale CMC, GMC downscale 8 (+latency)
# Then: _score_night_2026-09-15.py (headline CLEAR/IDF1, per seq, per class) -> _score_hota_path1_2026-09-15.py.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\night"
New-Item -ItemType Directory -Force $out | Out-Null

$gate = "$main\_scratch\path1_botnoreid_master.log"
Write-Output "########## NIGHT QUEUED $(Get-Date -Format 'HH:mm') -- waiting for BOT-NOREID DONE ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'BOT-NOREID DONE' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20
Write-Output "########## NIGHT START $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}
function Set-SharedReID { $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft; $env:DARE_CROP_SHRINK="0.0" }
function Set-BT { $env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0" }
function Set-ScaleCMC { $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="scale" }
$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

$runs = @(
  @{ name="p1_deepocsort_mh1";    tree=$main; lat=$false; cfg={ Set-SharedReID; $env:DARE_TRACKER="deepocsort"; $env:DARE_P1_DOC_MINHITS="1" } },
  @{ name="p1_botsort_m08";       tree=$main; lat=$false; cfg={ Set-SharedReID; $env:DARE_TRACKER="botsort"; $env:DARE_P1_BOT_MATCH="0.8" } },
  @{ name="p1_botsort_scalewarp"; tree=$main; lat=$false; cfg={ Set-SharedReID; $env:DARE_TRACKER="botsort"; $env:DARE_P1_BOT_WARP="scale" } },
  @{ name="lat10_bt_cmcscale_j_ds4"; tree=$wt; lat=$true; cfg={ Set-BT; Set-ScaleCMC; $env:DARE_CMC_DOWNSCALE="4" } },
  @{ name="lat10_bt_cmcscale_j_ds8"; tree=$wt; lat=$true; cfg={ Set-BT; Set-ScaleCMC; $env:DARE_CMC_DOWNSCALE="8" } }
)
$t0 = Get-Date
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg; $env:DARE_MAX_CLASS = "4"
  Set-Location $r.tree
  $env:PYTHONPATH = "$lapd;$($r.tree)"
  Write-Output "########## RUN $($r.name) $(Get-Date -Format 'HH:mm') ##########"
  if ($r.lat) {
    & $py _latency_cmc_2026-09-15.py --tag $r.name -- -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  } else {
    & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  }
  $code = $LASTEXITCODE
  $n = (Get-ChildItem "$($r.tree)\YOLOX_outputs\$($r.name)\track_results" -Filter *.txt -ErrorAction SilentlyContinue).Count
  Write-Output "----- DONE $($r.name) (exit $code, $n/7) after $((Get-Date) - $t0) -----"
  if ($code -ne 0 -or $n -ne 7) { Write-Output "!!! $($r.name) failed"; Get-Content "$out\$($r.name).log" -Tail 25 }
}
Clear-DareEnv
Set-Location $main
$env:PYTHONPATH = "$lapd;$main"
Write-Output "########## SCORE (CLEAR/IDF1) $(Get-Date -Format 'HH:mm') ##########"
& $py _score_night_2026-09-15.py
Write-Output "########## SCORE (HOTA) $(Get-Date -Format 'HH:mm') ##########"
& $py _score_hota_path1_2026-09-15.py
foreach ($t in "lat10_bt_cmcscale_j_ds4","lat10_bt_cmcscale_j_ds8") {
  $j = "$main\_scratch\latency_cmc\latency_$t.json"
  if (Test-Path $j) { Write-Output "--- latency $t ---"; Get-Content $j }
}
Write-Output "########## NIGHT DONE $(Get-Date -Format 'HH:mm') ##########"
