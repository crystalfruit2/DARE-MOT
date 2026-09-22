# Standalone verdict checker for the 2026-09-18 post-merge confirmation run.
# Safe to run any time, any number of times; reads only, changes nothing.
#   & "C:\Users\User\Desktop\Projects\DARE-MOT\_check_mergeverify.ps1"
$main = "C:\Users\User\Desktop\Projects\DARE-MOT"
$wt   = "C:\Users\User\Desktop\Projects\DARE-MOT-cmcfix"
$new  = "$main\YOLOX_outputs\mergeverify_bt_cmcscale_j\track_results"
$ref  = "$wt\YOLOX_outputs\p1ref_bt_cmcscale_j\track_results"
$n = (Get-ChildItem $new -Filter *.txt -ErrorAction SilentlyContinue).Count
Write-Output "merged main : $(git -C $main rev-parse --short HEAD)"
Write-Output "sequences   : $n/7 written"
if ($n -lt 7) { Write-Output "STILL RUNNING (or died) -- tail of run log:"; Get-Content "$main\_scratch\mergeverify_bt_cmcscale_j.log" -Tail 5; return }
$same = 0; $tot = 0
foreach ($f in Get-ChildItem $ref -Filter *.txt) {
  $tot++
  $c = "$new\$($f.Name)"
  if ((Test-Path $c) -and (Get-FileHash $f.FullName -Algorithm SHA256).Hash -eq (Get-FileHash $c -Algorithm SHA256).Hash) {
    $same++; Write-Output "  OK    $($f.Name)"
  } else { Write-Output "  DIFF  $($f.Name)" }
}
Write-Output "RESULT: $same/$tot byte-identical"
if ($same -eq $tot -and $tot -eq 7) {
  Write-Output "VERDICT: PASS -- merge confirmed, cross-tree comparison valid."
  Write-Output "NEXT: tag paper-runs-2026-09 + archive/cmc-move1, then the branch deletes."
} else {
  Write-Output "VERDICT: FAIL -- merged main does NOT reproduce the cached run."
  Write-Output "NEXT: do NOT delete any branch. git revert -m 1 <merge sha> and return to the two-tree setup."
  Write-Output "      Prime suspect: yolox/tracker/matching.py (main's +39 vs cmcfix's base version)."
}
