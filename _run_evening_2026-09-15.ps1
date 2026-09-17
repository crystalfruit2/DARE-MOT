# DARE-MOT -- evening queue 2026-09-15: parity vs scale CMC on the REPORTED detector D3.
# Why: the scale-exact warp was chosen on 2026-09-10 with the OLD detector after parity failed on uav0000305
# (in val7). The paper reports D3. Re-running parity on D3 shows whether that choice (and the 305 story)
# holds on the detector of record. Parity = BoT-SORT's warp recipe applied to DARE's xyah state.
#   mc10_bt_cmcparity_j       ByteTrack + parity CMC   (cf. p1ref_bt_cmcscale_j 57 / 66.14 / 47.92)
#   mc10_dare_cv_cmcparity_j  CV-DARE + parity CMC     (cf. p1ref_dare_cv_cmcscale_j 54 / 66.69 / 48.00)
# Waits for the night queue, then re-runs the night scorer (now listing these arms) and the HOTA scorer.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\evening"
New-Item -ItemType Directory -Force $out | Out-Null

$gate = "$main\_scratch\night_master.log"
Write-Output "########## EVENING QUEUED $(Get-Date -Format 'HH:mm') -- waiting for NIGHT DONE ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'NIGHT DONE' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20
Write-Output "########## EVENING START $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"

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
function Set-ParityCMC { $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="parity" }
$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
$runs = @(
  @{ name="mc10_bt_cmcparity_j";      cfg={ Set-BT; Set-ParityCMC } },
  @{ name="mc10_dare_cv_cmcparity_j"; cfg={ Set-DareHeadline; Set-ParityCMC } }
)
Set-Location $wt
$env:PYTHONPATH = "$lapd;$wt"
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg; $env:DARE_MAX_CLASS = "4"
  Write-Output "########## RUN $($r.name) $(Get-Date -Format 'HH:mm') ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  $n = (Get-ChildItem "$wt\YOLOX_outputs\$($r.name)\track_results" -Filter *.txt -ErrorAction SilentlyContinue).Count
  Write-Output "----- DONE $($r.name) (exit $LASTEXITCODE, $n/7) $(Get-Date -Format 'HH:mm') -----"
}
Clear-DareEnv
Set-Location $main
$env:PYTHONPATH = "$lapd;$main"
Write-Output "########## SCORE (CLEAR/IDF1, all arms) $(Get-Date -Format 'HH:mm') ##########"
& $py _score_night_2026-09-15.py
Write-Output "########## SCORE (HOTA, all arms) $(Get-Date -Format 'HH:mm') ##########"
& $py _score_hota_path1_2026-09-15.py
Write-Output "########## EVENING DONE $(Get-Date -Format 'HH:mm') ##########"
