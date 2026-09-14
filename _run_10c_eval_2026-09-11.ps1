# DARE-MOT -- tracker evaluation of the clean 10-class detector (2026-09-11).
# Detector: YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt (MOT17 init, 10 classes, ignored regions
# painted, val7 AP50 56.0). Tracked with DARE_MAX_CLASS=4 (model ids 0..4 = the 5 evaluated classes).
# Same code/solver/flags as the 09-10 CMC factorial (worktree exp/cmc-fixed, lap 0.5.12, Joseph on CMC arms).
# Arms = the recommended baseline (CV + scale CMC) with and without appearance, plus the no-CMC pair:
#   mc10_bt                ByteTrack, no CMC               (fresh plain baseline on the new detector)
#   mc10_bt_cmcscale_j     ByteTrack + scale CMC            (recommended baseline)
#   mc10_dare_cv_cmcscale_j CV-DARE + scale CMC             (best config on the old detector)
#   mc10_dare_cv           CV-DARE, no CMC
# Score: python _score_cmcfix_2026-09-10.py (arms listed there).
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\eval10c"
New-Item -ItemType Directory -Force $out | Out-Null

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
  @{ name="mc10_bt";                 cfg={ Set-BT } },
  @{ name="mc10_bt_cmcscale_j";      cfg={ Set-BT; Set-ScaleCMC } },
  @{ name="mc10_dare_cv_cmcscale_j"; cfg={ Set-DareHeadline; Set-ScaleCMC } },
  @{ name="mc10_dare_cv";            cfg={ Set-DareHeadline } }
)
$t0 = Get-Date
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg; $env:DARE_MAX_CLASS = "4"
  Write-Output "########## RUN $($r.name) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  Write-Output "----- DONE $($r.name) (exit $LASTEXITCODE) after $((Get-Date) - $t0) -----"
}
Clear-DareEnv
Write-Output "########## 10C EVAL RUNS COMPLETE in $((Get-Date) - $t0) ##########"
