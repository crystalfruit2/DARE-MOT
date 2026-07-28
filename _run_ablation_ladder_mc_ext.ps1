# DARE-MOT -- extension to the 2026-07-27 professor-requested aggregation ladder, run 2026-07-28.
# Two questions the original 9-config ladder didn't isolate:
#   1. Option A (L1-normalized, beta=4) vs Option B (temperature-softmax, tau=0.5) -- the ladder's
#      "reference" row (mc_lad_dare_full) never set DARE_AGG, so it silently ran on the code default
#      (Option B / softmax) the whole time. This adds the missing head-to-head: Option A explicitly.
#   2. N=5 and N=10 dynamic memory order -- byte_tracker.py only supported N in {1,2} before
#      2026-07-28; generalized to arbitrary N (see STrack._calculate_gammas/update_features) so this
#      is now testable without further code changes.
# Same fixed backdrop as the original mc ladder (5-class detector, IoU-gate 0.95, fine-tuned ReID,
# lock off, size gate off), same held-out sequence, same scorer. Scored against the SAME reference
# row (mc_lad_dare_full) already produced by the 2026-07-27 run -- not re-run here.
$ErrorActionPreference = "Continue"
$dare = "C:\Users\User\Desktop\projects\DARE-MOT"
$py   = "C:\Users\User\miniconda3\envs\dare_mot\python.exe"
$exp  = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
$ckpt = if ($args.Count -ge 1) { $args[0] } else { "YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar" }
$ft   = "$dare\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
$out  = "$dare\_ablation_ladder_mc"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $dare
$env:PYTHONPATH = $dare
if (-not (Test-Path $ckpt)) { Write-Output "MISSING detector ckpt: $ckpt"; exit 1 }
if (-not (Test-Path $ft))   { Write-Output "MISSING fine-tuned ReID weights: $ft"; exit 1 }

$flags = @("-b","1","-d","1","--fp16","--fuse","--seed","0",
           "--track_thresh","0.6","--track_buffer","30","--match_thresh","0.9","--min-box-area","100")

function Clear-DareEnv {
  Get-ChildItem Env: | Where-Object { $_.Name -like "DARE_*" } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
}

# tag, ORDER, AGG (empty = leave at code default 'B')
$configs = @(
  @{tag="mc_lad_agg_A";     ORDER="2";  AGG="A"},  # same as reference but Option A (L1) instead of B (softmax)
  @{tag="mc_lad_order5";    ORDER="5";  AGG=""},   # dynamic softmax, N=5 memory window
  @{tag="mc_lad_order10";   ORDER="10"; AGG=""}    # dynamic softmax, N=10 memory window
)

Write-Output "########## MC ABLATION LADDER EXT START ($($configs.Count) configs, detector = $ckpt) ##########"
$t0 = Get-Date
foreach ($c in $configs) {
  Clear-DareEnv
  # ---- fixed appearance backdrop (all rows): current production ReID, size gate OFF, IoU-gate ON ----
  $env:DARE_REID="osnet"; $env:DARE_REID_MODEL="osnet_ain_x1_0"; $env:DARE_REID_WEIGHTS=$ft
  $env:DARE_LAMBDA="0.5"; $env:DARE_LAMBDA_GATE="none"; $env:DARE_GATE_LO="0"; $env:DARE_GATE_HI="0"
  $env:DARE_POOL="mean"; $env:DARE_CROP_SHRINK="0.0"; $env:DARE_REASSOC_MAX="-1"
  $env:DARE_IOU_GATE="0.95"
  $env:DARE_LOCK="0"
  # ---- per-row aggregation-rule knob ----
  $env:DARE_AGG_ORDER=$c.ORDER
  if ($c.AGG -ne "") { $env:DARE_AGG=$c.AGG }
  Write-Output "########## RUN $($c.tag)  ORDER=$($c.ORDER) AGG=$($c.AGG)  ($((Get-Date) - $t0) elapsed) ##########"
  & $py tools/track.py -f $exp -c $ckpt @flags -expn $($c.tag) *> "$out\log_$($c.tag).log"
  Write-Output "----- DONE $($c.tag) (exit $LASTEXITCODE) -----"
}
Clear-DareEnv
Write-Output "########## TRACKING COMPLETE in $((Get-Date) - $t0). Scoring per-class... ##########"

$summary = "$out\_ladder_scores_mc_ext.txt"
Remove-Item $summary -ErrorAction SilentlyContinue
foreach ($c in $configs) {
  & $py _score_multiclass.py $($c.tag) "mc_lad_dare_full" *>> $summary
}
Write-Output "########## MC LADDER EXT COMPLETE in $((Get-Date) - $t0). Scores -> $summary ##########"
Get-Content $summary
