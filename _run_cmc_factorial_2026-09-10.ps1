# DARE-MOT -- complete the {appearance x CA-KF x CMC} 2x2x2 factorial (2026-09-10).
# The cmcfix queue (_run_cmcfix_2026-09-10.ps1) found ByteTrack + fixed CMC (174 IDSw) beating
# CA-DARE + fixed CMC (209) -- but those differ in BOTH appearance and CA-KF. These two arms fill the
# missing cells so each component's effect is measured with the other two held fixed:
#   A0 C1 M1  mc_bt_ca_cmcfix_j     ByteTrack + CA-KF + CMC(parity)   -> does appearance help on top of CA+CMC?
#   A1 C0 M1  mc_dare_cv_cmcfix_j   CV-DARE   + CMC(parity)           -> does CA help on top of appearance+CMC?
# Same code/worktree/solver/flags as the cmcfix queue; Joseph update on (proven a no-op, control arm).
# Waits for the cmcfix queue (its powershell host) to exit so the GPU is free.
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_mc\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\cmcfix"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Output "waiting for cmcfix queue to finish..."
while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
       Where-Object { $_.CommandLine -like "*_run_cmcfix_2026-09-10.ps1*" }) { Start-Sleep -Seconds 30 }

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

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
$runs = @(
  @{ name="mc_bt_ca_cmcfix_j";   cfg={ Set-BT; $env:DARE_KF_MODEL="ca"; $env:DARE_KF_ACCEL_NOISE="0.0125"
                                       $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="parity" } },
  @{ name="mc_dare_cv_cmcfix_j"; cfg={ Set-DareHeadline
                                       $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="parity" } }
)
$t0 = Get-Date
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg
  Write-Output "########## RUN $($r.name) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  Write-Output "----- DONE $($r.name) (exit $LASTEXITCODE) after $((Get-Date) - $t0) -----"
}
Clear-DareEnv
Write-Output "########## FACTORIAL RUNS COMPLETE in $((Get-Date) - $t0) ##########"
