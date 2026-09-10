# DARE-MOT -- corrected-CMC runs (2026-09-10), code in worktree ..\DARE-MOT-cmcfix (branch exp/cmc-fixed,
# uncommitted -- Alp commits). Port of exp/cmc-move1's GMC onto current main with the xyah
# parameterisation bug fixed (STrack.multi_gmc, DARE_CMC_FIX = parity | scale | bug).
# Unit-checked by _cmcfix_unit.py (CV parity == 09-07 reference to 1e-13; bug reproduces -29.3% width).
#
# Gate: waits for _run_lap0512_repro.ps1 and runs ONLY if it printed 21/21 byte-identical.
# Fresh baselines = that same-day repro (mc_bytetrack_lap0512, mc_dare_ca_lap0512) -- DARE_CMC=none
# never imports GMC, so the worktree is byte-identical to main for those.
#
# Arms, priority order:
#   1 mc_dare_ca_cmcfix    CA-DARE + CMC (BoT-SORT parity)       -> does corrected CMC help DARE?
#   2 mc_dare_ca_cmcbug    CA-DARE + CMC (July bug)              -> isolates what the bug cost
#   3 mc_bt_cmcfix         ByteTrack + CMC (parity)              -> CMC must be offered to both arms
#   4 mc_bt_ca             ByteTrack + CA-KF, no CMC             -> fairness: is CA a generic upgrade?
#   5 mc_dare_ca_cmcscale  CA-DARE + CMC (scale-only)            -> second parameterisation arm
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_mc\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\cmcfix"
$gate = "$main\_scratch\lap0512_repro_master.log"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Output "waiting for repro result..."
while ($true) {
  $r = Select-String -Path $gate -Pattern "RESULT:" -ErrorAction SilentlyContinue
  if ($r) { break }
  Start-Sleep -Seconds 60
}
Write-Output $r.Line
if ($r.Line -notmatch "RESULT: 21/21 files byte-identical") {
  Write-Output "ABORT: solver swap not proven equivalent -- no CMC runs."
  exit 1
}

Set-Location $wt
$env:PYTHONPATH = "$lapd;$wt"
& $py -c "import lap, yolox; print('solver:', lap.__file__); print('yolox:', yolox.__file__)"

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}
function Set-DareCA {
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
  $env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"
  $env:DARE_KF_MODEL="ca"; $env:DARE_KF_ACCEL_NOISE="0.0125"
}
function Set-BT { $env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0" }

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
# v2 (2026-09-10 12:15): the first attempt crashed on both CMC arms. Cause, found by instrumented replay
# (_cmcfix_debug2.py): the warp leaves an ill-conditioned covariance and ByteTrack's P - K S K^T update
# then loses PSD (Cholesky crash ~frame 240). Every CMC arm now uses the Joseph-form update
# (DARE_KF_JOSEPH=1, default off), and the parity warp uses an exact Jacobian. Arm 2 is the control
# that shows Joseph alone is a numerical no-op (compare to mc_dare_ca_lap0512).
$runs = @(
  @{ name="mc_dare_ca_cmcfix_j";   cfg={ Set-DareCA; $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="parity" } },
  @{ name="mc_dare_ca_joseph";     cfg={ Set-DareCA; $env:DARE_KF_JOSEPH="1" } },
  @{ name="mc_dare_ca_cmcbug_j";   cfg={ Set-DareCA; $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="bug" } },
  @{ name="mc_bt_cmcfix_j";        cfg={ Set-BT;     $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="parity" } },
  @{ name="mc_bt_ca";              cfg={ Set-BT;     $env:DARE_KF_MODEL="ca"; $env:DARE_KF_ACCEL_NOISE="0.0125" } },
  @{ name="mc_dare_ca_cmcscale_j"; cfg={ Set-DareCA; $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="scale" } }
)
$t0 = Get-Date
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg
  Write-Output "########## RUN $($r.name) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  Write-Output "----- DONE $($r.name) (exit $LASTEXITCODE) after $((Get-Date) - $t0) -----"
}
Clear-DareEnv
Write-Output "########## CMCFIX RUNS COMPLETE in $((Get-Date) - $t0) ##########"
