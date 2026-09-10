# DARE-MOT -- solver-swap reproduction check (2026-09-10).
# ============================================================================================
# Smart App Control (enforced since ~2026-09-07) blocks lapx 0.9.4's _lapjv.pyd, so no tracker
# run has been possible. The official `lap` 0.5.12 wheel (same LAPJV algorithm, high-reputation
# binary -- ultralytics depends on it) DOES load under SAC. It is staged OUTSIDE the conda env at
# C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512 and put first on PYTHONPATH, so the env
# itself is untouched.
#
# Acceptance test: re-run the exact 2026-08-03 three-way protocol (_run_farzad_3way_2026-08-03.ps1)
# with ONLY the solver changed, then SHA-256 compare every track_results/*.txt against the cached
# *_rerun0803 outputs. Byte-identical on all 21 files => the swap is proven equivalent on this
# benchmark and every cached number stays comparable. Any diff => do NOT use lap 0.5.12 for
# reported numbers.
# ============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar"
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$dare\_scratch\lap0512_repro"

Set-Location $dare
New-Item -ItemType Directory -Force $out | Out-Null
$env:PYTHONPATH = "$lapd;$dare"
& $py -c "import lap; print('solver:', lap.__file__)"

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

$t0 = Get-Date
Write-Output "########## LAP 0.5.12 REPRO START (main @ $(git rev-parse --short HEAD)) ##########"

$runs = @(
  @{ name = "mc_bytetrack_lap0512"; ref = "mc_bytetrack_rerun0803"; cfg = { $env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0" } },
  @{ name = "mc_dare_cv_lap0512";   ref = "mc_dare_cv_rerun0803";   cfg = { Set-DareHeadline } },
  @{ name = "mc_dare_ca_lap0512";   ref = "mc_dare_ca_rerun0803";   cfg = { Set-DareHeadline; $env:DARE_KF_MODEL="ca"; $env:DARE_KF_ACCEL_NOISE="0.0125" } }
)

foreach ($r in $runs) {
  Clear-DareEnv
  & $r.cfg
  Write-Output "########## RUN $($r.name) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  Write-Output "----- DONE $($r.name) (exit $LASTEXITCODE) after $((Get-Date) - $t0) -----"
}
Clear-DareEnv

Write-Output "########## HASH COMPARE vs *_rerun0803 ##########"
$diffs = 0; $n = 0
foreach ($r in $runs) {
  $refDir = "$dare\YOLOX_outputs\$($r.ref)\track_results"
  $newDir = "$dare\YOLOX_outputs\$($r.name)\track_results"
  foreach ($f in Get-ChildItem $refDir -Filter *.txt) {
    $n++
    $new = Join-Path $newDir $f.Name
    if (-not (Test-Path $new)) { Write-Output "MISSING  $($r.name)\$($f.Name)"; $diffs++; continue }
    $a = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
    $b = (Get-FileHash $new -Algorithm SHA256).Hash
    if ($a -eq $b) { Write-Output "SAME     $($r.name)\$($f.Name)" } else { Write-Output "DIFF     $($r.name)\$($f.Name)"; $diffs++ }
  }
}
Write-Output "########## RESULT: $($n - $diffs)/$n files byte-identical, $diffs differ. Total $((Get-Date) - $t0). ##########"
