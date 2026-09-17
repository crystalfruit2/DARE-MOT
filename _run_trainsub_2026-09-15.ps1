# DARE-MOT -- out-of-val tracker-ranking check on 7 train sequences (2026-09-15).
# Pre-registered decision rules: vault experiment-log §2026-09-15 "Out-of-val tracker-ranking check".
# D3 was trained on these frames -> rankings only. Waits for the evening parity queue.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_trainsub7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\trainsub"
New-Item -ItemType Directory -Force $out | Out-Null

$gate = "$main\_scratch\evening_master.log"
Write-Output "########## TRAINSUB QUEUED $(Get-Date -Format 'HH:mm') -- waiting for EVENING DONE ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'EVENING DONE' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20
Write-Output "########## TRAINSUB START $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
Set-Location $main
& $py _make_trainsub_2026-09-15.py

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}
function Set-SharedReID { $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft; $env:DARE_CROP_SHRINK="0.0" }
function Set-DareHeadline {
  Set-SharedReID
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
}
function Set-BT { $env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0" }
function Set-CMC($mode) { $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX=$mode }
$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
$runs = @(
  @{ name="ts_bt";                 tree=$wt;   cfg={ Set-BT } },
  @{ name="ts_bt_cmcscale_j";      tree=$wt;   cfg={ Set-BT; Set-CMC "scale" } },
  @{ name="ts_bt_cmcparity_j";     tree=$wt;   cfg={ Set-BT; Set-CMC "parity" } },
  @{ name="ts_dare_cv_cmcscale_j"; tree=$wt;   cfg={ Set-DareHeadline; Set-CMC "scale" } },
  @{ name="ts_botsort";            tree=$main; cfg={ Set-SharedReID; $env:DARE_TRACKER="botsort" } },
  @{ name="ts_botsort_noreid";     tree=$main; cfg={ $env:DARE_TRACKER="botsort"; $env:DARE_P1_BOT_REID="0" } }
)
$t0 = Get-Date
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg; $env:DARE_MAX_CLASS = "4"
  Set-Location $r.tree
  $env:PYTHONPATH = "$lapd;$($r.tree)"
  Write-Output "########## RUN $($r.name) $(Get-Date -Format 'HH:mm') ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  $code = $LASTEXITCODE
  $n = (Get-ChildItem "$($r.tree)\YOLOX_outputs\$($r.name)\track_results" -Filter *.txt -ErrorAction SilentlyContinue).Count
  Write-Output "----- DONE $($r.name) (exit $code, $n/7) after $((Get-Date) - $t0) -----"
  if ($code -ne 0 -or $n -ne 7) { Write-Output "!!! $($r.name) failed"; Get-Content "$out\$($r.name).log" -Tail 25 }
}
Clear-DareEnv
Set-Location $main
$env:PYTHONPATH = "$lapd;$main"
Write-Output "########## SCORE $(Get-Date -Format 'HH:mm') ##########"
& $py _score_trainsub_2026-09-15.py
Write-Output "########## TRAINSUB DONE $(Get-Date -Format 'HH:mm') ##########"
