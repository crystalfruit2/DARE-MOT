import json
from collections import defaultdict

SEQ = "uav0000086_00000_v"
WINDOW = 50

switches = json.load(open("_cmc_gate_decisive/switch_records.json"))
resid = json.load(open(f"_cmc_residual_logs/{SEQ}.json"))

resid_by_frame = {r["frame"]: r for r in resid}
max_frame = max(resid_by_frame.keys())

seq_switches = [s for s in switches if s["seq"] == SEQ]
print(f"total switch events in {SEQ}: {len(seq_switches)} (of 1740 pooled)")
print(f"max frame in residual log: {max_frame}")

# bin into WINDOW-frame windows
def window_of(frame):
    return frame // WINDOW

win_stats = defaultdict(lambda: {"added": 0, "removed": 0, "ped_added": 0, "ped_removed": 0})
for s in seq_switches:
    w = window_of(s["frame"])
    win_stats[w][s["label"]] += 1
    if s["cat"] == 1:  # pedestrian
        win_stats[w][f'ped_{s["label"]}'] += 1

# per-window residual/translation stats (pooled over frames in window, both pairs share the same warp log)
win_resid = defaultdict(list)
win_trans = defaultdict(list)
for f, r in resid_by_frame.items():
    w = window_of(f)
    if r["resid_px"] is not None:
        win_resid[w].append(r["resid_px"])
    if r["trans_px"] is not None:
        win_trans[w].append(r["trans_px"])

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def p90(xs):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = int(0.9 * (len(xs) - 1))
    return xs[idx]

print(f"\n{'window':>12} {'added':>6} {'removed':>8} {'net':>6} {'ped_net':>8} {'mean_resid':>11} {'p90_trans':>10}")
all_windows = sorted(set(list(win_stats.keys()) + list(win_resid.keys())))
for w in all_windows:
    st = win_stats[w]
    net = st["removed"] - st["added"]
    ped_net = st.get("ped_removed", 0) - st.get("ped_added", 0)
    mr = mean(win_resid[w])
    pt = p90(win_trans[w])
    frange = f"{w*WINDOW}-{w*WINDOW+WINDOW-1}"
    print(f"{frange:>12} {st['added']:>6} {st['removed']:>8} {net:>6} {ped_net:>8} {mr:>11.4f} {pt:>10.2f}")

total_added = sum(st["added"] for st in win_stats.values())
total_removed = sum(st["removed"] for st in win_stats.values())
print(f"\nTOTAL: added={total_added} removed={total_removed} net={total_removed-total_added}")

# concentration check: what fraction of net damage comes from the single worst window?
nets = [(w, win_stats[w]["removed"] - win_stats[w]["added"]) for w in win_stats]
nets.sort(key=lambda x: x[1])
worst = nets[0]
total_net = sum(n for _, n in nets)
pct = (100 * worst[1] / total_net) if total_net < 0 else float("nan")
print(f"worst window: {worst[0]*WINDOW}-{worst[0]*WINDOW+WINDOW-1}, net={worst[1]} ({pct:.1f}% of total net)")
