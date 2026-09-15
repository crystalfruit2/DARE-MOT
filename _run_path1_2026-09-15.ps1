# DARE-MOT -- Move 2 / Path 1: controlled baseline comparison on the paper's detector (2026-09-15).
# ============================================================================================
# Detector D3 (clean 10-class, DARE_MAX_CLASS=4), val7, seed 0, same flags for every arm. Only the
# association logic changes; BoT-SORT and Deep OC-SORT share DARE's OSNet-AIN embedding.
#
#   p1_ocsort                 OC-SORT (motion only, BYTE on)                       main tree
#   p1_botsort                BoT-SORT-ReID (GMC sparseOptFlow + shared OSNet)     main tree
#   p1_deepocsort             Deep OC-SORT (CMC = BoT-SORT GMC live, AW, shared OSNet) main tree
#   p1ref_bt                  ByteTrack                                            cmcfix worktree
#   p1ref_bt_cmcscale_j       ByteTrack + scale CMC (recommended baseline)          cmcfix worktree
#   p1ref_dare_cv_cmcscale_j  CV-DARE + scale CMC (best config)                     cmcfix worktree
# Reference arms are re-run fresh (standing rule) and SHA-256 compared with the cached 09-11 mc10_* runs.
# Waits for the detector opcheck (the GPU queue before it) to finish.
# Score: python _score_path1_2026-09-15.py
# ============================================================================================
$ErrorActionPreference = "Continue"
$main = "C:\Users\User\Desktop\projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$lapd = "C:\Users\User\Desktop\projects\DARE-MOT-pylibs\lap0512"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_10c_val7.py"
$ckpt = "$main\YOLOX_outputs\yolox_x_visdrone_10c_mot17init\best_ckpt.pth.tar"
$ft   = "$main\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$main\_scratch\path1"
New-Item -ItemType Directory -Force $out | Out-Null

$gate = "$main\_scratch\det_opcheck_master.log"
Write-Output "########## PATH1 QUEUED $(Get-Date -Format 'HH:mm') -- waiting for OPCHECK DONE ##########"
while (-not ((Test-Path $gate) -and (Select-String -Path $gate -Pattern 'OPCHECK DONE' -Quiet))) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 20
Write-Output "########## PATH1 START $(Get-Date -Format 'yyyy-MM-dd HH:mm') (main @ $(git -C $main rev-parse --short HEAD)) ##########"

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
function Set-ScaleCMC { $env:DARE_KF_JOSEPH="1"; $env:DARE_CMC="sparseOptFlow"; $env:DARE_CMC_FIX="scale" }

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")
$runs = @(
  @{ name="p1_ocsort";                tree=$main; ref=$null;                     cfg={ $env:DARE_TRACKER="ocsort" } },
  @{ name="p1_botsort";               tree=$main; ref=$null;                     cfg={ Set-SharedReID; $env:DARE_TRACKER="botsort" } },
  @{ name="p1_deepocsort";            tree=$main; ref=$null;                     cfg={ Set-SharedReID; $env:DARE_TRACKER="deepocsort" } },
  @{ name="p1ref_bt";                 tree=$wt;   ref="mc10_bt";                 cfg={ Set-BT } },
  @{ name="p1ref_bt_cmcscale_j";      tree=$wt;   ref="mc10_bt_cmcscale_j";      cfg={ Set-BT; Set-ScaleCMC } },
  @{ name="p1ref_dare_cv_cmcscale_j"; tree=$wt;   ref="mc10_dare_cv_cmcscale_j"; cfg={ Set-DareHeadline; Set-ScaleCMC } }
)
$t0 = Get-Date
foreach ($r in $runs) {
  Clear-DareEnv; & $r.cfg; $env:DARE_MAX_CLASS = "4"
  Set-Location $r.tree
  $env:PYTHONPATH = "$lapd;$($r.tree)"
  Write-Output "########## RUN $($r.name) (tree $($r.tree), DARE_TRACKER=$env:DARE_TRACKER) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $r.name *> "$out\$($r.name).log"
  $code = $LASTEXITCODE
  $n = (Get-ChildItem "$($r.tree)\YOLOX_outputs\$($r.name)\track_results" -Filter *.txt -ErrorAction SilentlyContinue).Count
  Write-Output "----- DONE $($r.name) (exit $code, $n/7 sequence files) after $((Get-Date) - $t0) -----"
  if ($code -ne 0 -or $n -ne 7) { Write-Output "!!! $($r.name) failed -- tail of log:"; Get-Content "$out\$($r.name).log" -Tail 25 }
}
Clear-DareEnv

Write-Output "########## REFERENCE DETERMINISM: SHA-256 vs cached 09-11 mc10_* ##########"
foreach ($r in $runs | Where-Object { $_.ref }) {
  $same = 0; $tot = 0
  foreach ($f in Get-ChildItem "$wt\YOLOX_outputs\$($r.ref)\track_results" -Filter *.txt) {
    $tot++
    $new = "$wt\YOLOX_outputs\$($r.name)\track_results\$($f.Name)"
    if ((Test-Path $new) -and (Get-FileHash $f.FullName -Algorithm SHA256).Hash -eq (Get-FileHash $new -Algorithm SHA256).Hash) { $same++ }
  }
  Write-Output "$($r.name) vs $($r.ref): $same/$tot byte-identical"
}
Set-Location $main
$env:PYTHONPATH = "$lapd;$main"
Write-Output "########## SCORE ##########"
& $py _score_path1_2026-09-15.py
Write-Output "########## PATH1 DONE $(Get-Date -Format 'HH:mm') ##########"
