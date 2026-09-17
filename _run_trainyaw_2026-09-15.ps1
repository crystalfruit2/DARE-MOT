# DARE-MOT -- yaw-generalization check on the 9 yaw-containing train sequences (2026-09-15).
# Pre-registered rules Y1-Y4: vault experiment-log §2026-09-15 "Yaw generalization check".
# D3 was trained on these frames -> rankings and switch locations only. Waits for the out-of-val subset queue.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_trainyaw9.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\trainyaw"
New-Item -ItemType Directory -Force $out | Out-Null

$gate = "$main\_scratch\trainsub_master.log"
Write-Output "########## TRAINYAW QUEUED $(Get-Date -Format 'HH:mm') -- waiting for TRAINSUB DONE ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'TRAINSUB DONE' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20
Write-Output "########## TRAINYAW START $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
Set-Location $main
& $py _make_trainyaw_2026-09-15.py

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}
function Set-SharedReID { $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft; $env:DARE_CROP_SHRINK="0.0" }
function Set-BT { $env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0" }
function Set-CMC($mode) { $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX=$mode }
$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
$runs = @(
  @{ name="ty_bt";                tree=$wt;   cfg={ Set-BT } },
  @{ name="ty_bt_cmcparity_j";    tree=$wt;   cfg={ Set-BT; Set-CMC "parity" } },
  @{ name="ty_bt_cmcscale_j";     tree=$wt;   cfg={ Set-BT; Set-CMC "scale" } },
  @{ name="ty_botsort";           tree=$main; cfg={ Set-SharedReID; $env:DARE_TRACKER="botsort" } },
  @{ name="ty_botsort_scalewarp"; tree=$main; cfg={ Set-SharedReID; $env:DARE_TRACKER="botsort"; $env:DARE_P1_BOT_WARP="scale" } }
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
  Write-Output "----- DONE $($r.name) (exit $code, $n/9) after $((Get-Date) - $t0) -----"
  if ($code -ne 0 -or $n -ne 9) { Write-Output "!!! $($r.name) failed"; Get-Content "$out\$($r.name).log" -Tail 25 }
}
Clear-DareEnv
Set-Location $main
$env:PYTHONPATH = "$lapd;$main"
Write-Output "########## SCORE $(Get-Date -Format 'HH:mm') ##########"
& $py _score_trainyaw_2026-09-15.py
Write-Output "########## TRAINYAW DONE $(Get-Date -Format 'HH:mm') ##########"
