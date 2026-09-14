# DARE-MOT -- unattended queue for the night of 2026-09-14.
# ============================================================================================
# Waits for the 5-class control training (yolox_x_visdrone_5c_ctrl_mot17init) to exit, then runs,
# in order:
#   T1  per-sequence detection AP on all THREE detectors -- D2 (old 5-class, leak-inheriting),
#       D3 (clean 10-class) and D4 (tonight's clean 5-class control). D2/D3 come from cache, so
#       only D4 computes (~15 min). This is the decisive measurement: D2 vs D4 differ ONLY in
#       initialization (same class set), which is the cleanest leak test this project can run.
#   T2  the IoU-fallback byte-identical repro (~45 min), which gates every future tracker number
#       on this machine while cython_bbox stays blocked by Smart App Control.
#
# PRE-REGISTERED reading of D4, written before the number exists:
#   * D4 reproduces the -leaked / +uav0000339 pattern  -> leakage is confirmed independently of the
#     class set, and the class-set caveat in the paper's Table~\ref{tab:leaktest} paragraph can be
#     DELETED rather than softened.
#   * D4 pooled AP50 within ~1 pt of D3   -> the class set is not what cost recall, the init is.
#     Limitation (3) closes, D3 stays the paper's provenance-clean detector, nothing further runs.
#   * D4 pooled AP50 >= 2 pts above D3    -> the 10-way head cost the recall, and D4 becomes the
#     better candidate reported detector (clean AND no recall loss). Tracker arms on D4 then become
#     required -- but only after T2 passes.
# T3 (tracker arms on D4) is deliberately NOT queued here: the control's job is a detection-AP
# question, and queueing tracker runs speculatively would spend GPU on an unvalidated IoU path.
# ============================================================================================
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$out  = "$dare\_scratch"
Set-Location $dare

Write-Output "########## QUEUE ARMED $(Get-Date -Format 'yyyy-MM-dd HH:mm') -- waiting for the control training to exit ##########"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like "*yolox_x_visdrone_5c_ctrl*" }) {
  Start-Sleep -Seconds 60
}
Write-Output "########## TRAINING EXITED $(Get-Date -Format 'HH:mm') ##########"

$d4 = "$dare\YOLOX_outputs\yolox_x_visdrone_5c_ctrl_mot17init\best_ckpt.pth.tar"
if (-not (Test-Path $d4)) {
  Write-Output "!!! No best_ckpt for the control run -- training did not produce a checkpoint. T1 skipped, T2 still runs."
} else {
  Write-Output "########## T1: per-sequence AP, D2 vs D3 vs D4 $(Get-Date -Format 'HH:mm') ##########"
  & $py _det_ap_perseq.py `
      "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" `
      "YOLOX_outputs/yolox_x_visdrone_10c_mot17init/best_ckpt.pth.tar" `
      "YOLOX_outputs/yolox_x_visdrone_5c_ctrl_mot17init/best_ckpt.pth.tar" `
      *> "$out\det_ap_perseq\run_tonight_2026-09-14.log"
  Write-Output "----- T1 done (exit $LASTEXITCODE). Table: _scratch\det_ap_perseq\run_tonight_2026-09-14.log -----"
  Get-Content "$out\det_ap_perseq\run_tonight_2026-09-14.log" -Tail 12
}

Write-Output "########## T2: IoU-fallback byte-identical repro $(Get-Date -Format 'HH:mm') ##########"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$dare\_run_bboxnumpy_repro_2026-09-14.ps1" *> "$out\bboxnumpy_repro_master.log"
Write-Output "----- T2 done (exit $LASTEXITCODE) -----"
Get-Content "$out\bboxnumpy_repro_master.log" -Tail 6

Write-Output "########## QUEUE COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm') ##########"
