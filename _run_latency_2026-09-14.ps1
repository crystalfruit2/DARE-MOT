# ============================================================================================
# Association-latency measurement: DARE headline arm vs static-EMA control.
#
# WHY: the report quotes "8.40 ms/frame vs a static-EMA control at 8.17 ms/frame, marginal
# 0.23 ms (~2.8%)". The 8.40 has a record; the control and the marginal cost do not appear in
# any logged run (audit 2026-09-14). This produces BOTH arms under one protocol.
#
# WHAT IS TIMED: t_assoc = t_update - t_embed, i.e. matching + KF predict/update + memory gate,
# with OSNet extraction subtracted, because that is the report's stated quantity. See the
# docstring of _latency_assoc_2026-09-14.py.
#
# CONFIG: camera motion compensation OFF in both arms, matching the condition under which the
# existing 8.40 figure was taken (and which the report already discloses). The two arms differ
# in exactly one knob: the template update rule.
#
# CAVEAT carried into the output: SAC blocks cython_bbox here, so IoU runs the NumPy fallback.
# Absolute ms is an upper bound; the MARGINAL cost is unaffected (common path, cancels).
# ============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar"
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$dare\_scratch\latency"

Set-Location $dare
New-Item -ItemType Directory -Force $out | Out-Null
$env:PYTHONPATH = "$lapd;$dare"

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

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

Write-Output "########## LATENCY START $(Get-Date -Format 'yyyy-MM-dd HH:mm') (main @ $(git rev-parse --short HEAD)) ##########"

# Arm 1: DARE, confidence-driven N=2 memory (the reported configuration).
Clear-DareEnv; Set-DareHeadline
Write-Output "########## ARM dare (dynamic N=2 memory gate) ##########"
& $py _latency_assoc_2026-09-14.py --tag dare -- -f $exp -c $ckpt @flags -expn lat_dare *> "$out\lat_dare.log"

# Arm 2: static-EMA control. ONE knob changes: the template update becomes a first-order EMA at
# gamma = 0.9, the canonical DeepSORT/FairMOT value and the control the report contrasts against.
Clear-DareEnv; Set-DareHeadline
$env:DARE_STATIC_EMA="0.9"; $env:DARE_AGG_ORDER="1"
Write-Output "########## ARM staticema (first-order EMA, gamma=0.9) ##########"
& $py _latency_assoc_2026-09-14.py --tag staticema -- -f $exp -c $ckpt @flags -expn lat_staticema *> "$out\lat_staticema.log"

Write-Output "########## LATENCY DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
foreach ($t in @("dare","staticema")) {
  $j = "$dare\_scratch\latency_$t.json"
  if (Test-Path $j) { Write-Output "--- $t ---"; Get-Content $j }
  else { Write-Output "!!! $t produced no json -- check $out\lat_$t.log" }
}
