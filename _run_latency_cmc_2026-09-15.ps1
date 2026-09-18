# DARE-MOT -- tracker latency WITH camera motion compensation, parts timed separately (2026-09-15).
# Closes paper limitation 4 on the desktop GPU (Jetson figure still outstanding).
# Detector = D3 (clean 10-class, the paper's provenance-clean detector), DARE_MAX_CLASS=4.
# Arms and env are copied verbatim from _run_10c_eval_2026-09-11.ps1, and each run's track_results are
# SHA-256 compared against that cached run: identical outputs prove the timed config IS the reported one.
#   lat10_dare_cv_cmcscale_j   CV-DARE + scale CMC   (best config)      ref mc10_dare_cv_cmcscale_j
#   lat10_bt_cmcscale_j        ByteTrack + scale CMC (no appearance)    ref mc10_bt_cmcscale_j
# Waits for the forced IoU repro (main worktree) to finish so the GPU/CPU are not shared while timing.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\latency_cmc"
New-Item -ItemType Directory -Force $out | Out-Null

$gate = "$main\_scratch\bboxnumpy_forced_master.log"
Write-Output "########## LATENCY+CMC QUEUED $(Get-Date -Format 'HH:mm') -- waiting for forced repro RESULT line ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'RESULT:|aborting' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20   # let the repro's python exit fully
Write-Output "########## LATENCY+CMC START $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"

Set-Location $wt
$env:PYTHONPATH = "$lapd;$wt"
& $py -c "import lap, yolox; print('solver:', lap.__file__); print('yolox:', yolox.__file__)"

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
  @{ name="lat10_dare_cv_cmcscale_j"; ref="mc10_dare_cv_cmcscale_j"; cfg={ Set-DareHeadline; Set-ScaleCMC } },
  @{ name="lat10_bt_cmcscale_j";      ref="mc10_bt_cmcscale_j";      cfg={ Set-BT; Set-ScaleCMC } }
)
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg; $env:DARE_MAX_CLASS = "4"
  Write-Output "########## ARM $($r.name) ##########"
  & $py _latency_cmc_2026-09-15.py --tag $r.name -- -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  Write-Output "----- DONE $($r.name) (exit $LASTEXITCODE) $(Get-Date -Format 'HH:mm') -----"
}
Clear-DareEnv

Write-Output "########## CONFIG IDENTITY: SHA-256 vs cached 09-11 eval runs ##########"
$diffs = 0; $n = 0
foreach ($r in $runs) {
  $refDir = @("$wt\YOLOX_outputs\$($r.ref)\track_results", "$main\YOLOX_outputs\$($r.ref)\track_results") | Where-Object { Test-Path $_ } | Select-Object -First 1
  $newDir = "$wt\YOLOX_outputs\$($r.name)\track_results"
  if (-not $refDir) { Write-Output "NOREF    $($r.ref) (cached run not found)"; $diffs++; continue }
  foreach ($f in Get-ChildItem $refDir -Filter *.txt) {
    $n++
    $new = Join-Path $newDir $f.Name
    if (-not (Test-Path $new)) { Write-Output "MISSING  $($r.name)\$($f.Name)"; $diffs++; continue }
    if ((Get-FileHash $f.FullName -Algorithm SHA256).Hash -eq (Get-FileHash $new -Algorithm SHA256).Hash) { Write-Output "SAME     $($r.name)\$($f.Name)" }
    else { Write-Output "DIFF     $($r.name)\$($f.Name)"; $diffs++ }
  }
}
Write-Output "########## IDENTITY: $($n - $diffs)/$n byte-identical ##########"
foreach ($r in $runs) {
  $j = "$out\latency_$($r.name).json"
  if (Test-Path $j) { Write-Output "--- $($r.name) ---"; Get-Content $j } else { Write-Output "!!! no json for $($r.name) -- see $out\$($r.name).log" }
}
Write-Output "########## LATENCY+CMC DONE $(Get-Date -Format 'HH:mm') ##########"
