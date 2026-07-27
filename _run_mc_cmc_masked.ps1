# DARE-MOT — Move 1 (masked CMC diagnostic). Same as _run_mc_cmc.ps1 but with DARE_CMC_MASK=1:
# detection regions are excluded from the sparseOptFlow motion estimate, so the warp reflects
# camera (background) motion, not the dense foreground that corrupted the unmasked run. Writes
# to NEW expn dirs (*_cmc_m) so the unmasked results (*_cmc) are preserved for comparison.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = if ($args.Count -ge 1) { $args[0] } else { "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" }
$cmc  = if ($args.Count -ge 2) { $args[1] } else { "sparseOptFlow" }
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"

Set-Location $dare
$env:PYTHONPATH = $dare
if (-not (Test-Path $ckpt)) { Write-Output "MISSING detector ckpt: $ckpt"; exit 1 }

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

$t0 = Get-Date
Write-Output "########## MOVE 1 MASKED CMC START  (cmc=$cmc, MASK=1) ##########"

# ---- Run 1: ByteTrack + masked CMC ----
Clear-DareEnv
$env:DARE_LAMBDA="0.0"; $env:DARE_IOU_GATE="1.0"; $env:DARE_LOCK="0"
$env:DARE_CMC=$cmc; $env:DARE_CMC_MASK="1"
$btn = "mc_bytetrack_cmc_m"
Write-Output "########## RUN $btn ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $btn *> "$dare\_mc_$btn.log"
Write-Output "----- DONE $btn (exit $LASTEXITCODE) -----"

# ---- Run 2: DARE + masked CMC ----
Clear-DareEnv
if (-not (Test-Path $ft)) { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }
$env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
$env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="size"; $env:DARE_GATE_LO="2500"; $env:DARE_GATE_HI="0"
$env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
$env:DARE_AGG_ORDER="2"; $env:DARE_STATIC_EMA="-1"; $env:DARE_STATIC_GAMMAS=""
$env:DARE_LOCK="0"; $env:DARE_IOU_GATE="0.95"; $env:DARE_CMC=$cmc; $env:DARE_CMC_MASK="1"
$dan = "mc_dare_cmc_m"
Write-Output "########## RUN $dan ##########"
& $py tools/track.py -f $exp -c $ckpt @flags -expn $dan *> "$dare\_mc_$dan.log"
Write-Output "----- DONE $dan (exit $LASTEXITCODE) -----"
Clear-DareEnv

Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring... ##########"
Write-Output "===== (i) DARE+maskedCMC vs ByteTrack+maskedCMC (headline) ====="
& $py _score_multiclass.py $dan $btn
Write-Output "===== (ii) DARE+maskedCMC vs DARE (no CMC): does masked CMC help? ====="
& $py _score_multiclass.py $dan mc_dare
Write-Output "===== (iii) DARE+maskedCMC vs DARE+CMC (unmasked): did masking fix it? ====="
& $py _score_multiclass.py $dan mc_dare_cmc
Write-Output "===== (iv) HOTA/AssA: DARE+maskedCMC vs ByteTrack+maskedCMC ====="
& $py _trackeval\_score_hota_mc.py $dan $btn
Write-Output "########## MOVE 1 MASKED CMC COMPLETE in $((Get-Date) - $t0). ##########"
