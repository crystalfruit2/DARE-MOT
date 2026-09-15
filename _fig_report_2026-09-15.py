"""Report figures from the 2026-09-15 measurements (static, print, light surface).

F1 fig-latency-breakdown   stacked horizontal bars: CMC / association / appearance ms per frame (v2 latency JSONs)
F2 fig-path1-idsw-heatmap  headline IDSw per tracker x sequence (sequential blue ramp, log colour scale, values in cells)
F3 fig-path1-idsw-vs-idf1  pooled IDSw (log x) vs IDF1, one series, direct labels (leader line where crowded)
Plus CSV table views of every plotted number. Palette + mark specs follow the dataviz reference instance
(categorical slots 1-3 validated: CVD worst adjacent dE 9.2, normal 27.6; aqua < 3:1 -> values labelled + CSV).
Output: vault Projects/Dare_Mot/figures/2026-09-15/
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\User\Desktop\Projects\second_brain\Projects\Dare_Mot\figures\2026-09-15"
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ F1 latency breakdown
lat = {}
for tag, label in (("lat10v2_bt_cmcscale_j", "ByteTrack + scale CMC"),
                   ("lat10v2_dare_cv_cmcscale_j", "CV-DARE + scale CMC")):
    j = json.load(open(os.path.join(HERE, "_scratch", "latency_cmc", f"latency_{tag}.json")))
    appearance = j["embed_ms"]["mean"] if j["env"].get("DARE_REID") == "osnet" else 0.0  # BT pays an unused pass
    lat[label] = {"CMC (estimate + warp)": j["gmc_ms"]["mean"] + j["warp_ms"]["mean"],
                  "Association": j["assoc_ms"]["mean"], "Appearance (OSNet-AIN)": appearance}
parts = [("CMC (estimate + warp)", S1), ("Association", S2), ("Appearance (OSNet-AIN)", S3)]
fig, ax = plt.subplots(figsize=(6.4, 2.0))
labels = list(lat)[::-1]
for yi, label in enumerate(labels):
    left = 0.0
    for part, color in parts:
        v = lat[label][part]
        if v <= 0:
            continue
        ax.barh(yi, v, left=left, height=0.42, color=color, edgecolor=SURFACE, linewidth=2)
        left += v
    ax.text(left + 3, yi, f"{left:.0f} ms", va="center", ha="left", color=INK, fontsize=9)
ax.axvline(50, color=MUTED, linewidth=1)
ax.text(51.5, 0.5, "50 ms edge budget", color=INK2, fontsize=8, ha="left", va="center")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Tracker time per frame, ms (mean, detector excluded)")
ax.set_xlim(0, 215)
ax.set_ylim(-0.5, len(labels) - 0.5)
ax.grid(axis="x", color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
ax.legend(handles, [p for p, _ in parts], ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02),
          fontsize=8, handlelength=1.0, columnspacing=1.2, borderaxespad=0)
save(fig, "fig-latency-breakdown")
write_csv("fig-latency-breakdown", ["method"] + [p for p, _ in parts] + ["total"],
          [[k] + [f"{v[p]:.2f}" for p, _ in parts] + [f"{sum(v.values()):.2f}"] for k, v in lat.items()])

# ------------------------------------------------------------------ Path-1 data
S = json.load(open(os.path.join(HERE, "_scratch", "path1_score", "path1_summary.json")))
ARMS = [("p1ref_bt", "ByteTrack"), ("p1_ocsort", "OC-SORT"), ("p1_deepocsort", "Deep OC-SORT"),
        ("p1_botsort", "BoT-SORT-ReID"), ("p1_botsort_noreid", "BoT-SORT, no ReID"),
        ("p1ref_bt_cmcscale_j", "ByteTrack + scale CMC"), ("p1ref_dare_cv_cmcscale_j", "CV-DARE + scale CMC")]
ARMS = [(e, l) for e, l in ARMS if e in S["head"]]
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]

# ------------------------------------------------------------------ F2 heatmap
M = [[S["head_seq"][e][s]["SUM_IDSw"] for s in SEQS] for e, _ in ARMS]
cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
norm = LogNorm(vmin=0.8, vmax=220)
fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(ARMS) + 0.9))
im = ax.imshow([[max(v, 0.8) for v in row] for row in M], cmap=cmap, norm=norm, aspect="auto")
for i, row in enumerate(M):
    for k, v in enumerate(row):
        rgba = cmap(norm(max(v, 0.8)))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(k, i, str(v), ha="center", va="center", fontsize=8.5, color="#ffffff" if lum < 0.5 else INK)
ax.set_xticks(range(len(SEQS)), [s[3:10] + ("*" if s.startswith("uav0000339") else "") for s in SEQS], fontsize=8)
ax.set_yticks(range(len(ARMS)), [l for _, l in ARMS])
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks([x - 0.5 for x in range(1, len(SEQS))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(ARMS))], minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("val7 sequence (* not in the old detector's training split); identity switches, headline protocol")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[1, 10, 100])
cb.ax.set_yticklabels(["1", "10", "100"])
cb.ax.yaxis.set_minor_formatter(NullFormatter())
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7, colors=MUTED)
cb.set_label("IDSw (log colour scale)", color=INK2, fontsize=8)
save(fig, "fig-path1-idsw-heatmap")
write_csv("fig-path1-idsw-heatmap", ["tracker"] + SEQS + ["total"],
          [[l] + M[i] + [S["head"][e]["SUM_IDSw"]] for i, (e, l) in enumerate(ARMS)])

# ------------------------------------------------------------------ F3 IDSw vs IDF1
pts = [(l, S["head"][e]["SUM_IDSw"], 100 * S["head"][e]["MICRO"]["idf1"], 100 * S["head"][e]["MICRO"]["mota"]) for e, l in ARMS]
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=64, color=S1, edgecolor=SURFACE, linewidth=2, zorder=3)
# (dx, dy, ha, leader?) in offset points
offsets = {"CV-DARE + scale CMC": (6, 6, "left", False), "ByteTrack + scale CMC": (34, -40, "left", True),
           "BoT-SORT-ReID": (7, -3, "left", False), "BoT-SORT, no ReID": (7, 6, "left", False),
           "Deep OC-SORT": (7, -3, "left", False), "OC-SORT": (-8, 6, "right", False),
           "ByteTrack": (-9, 0, "right", False)}
for l, x, y, _ in pts:
    dx, dy, ha, leader = offsets.get(l, (6, 4, "left", False))
    ax.annotate(l, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=5) if leader else None)
ax.set_xscale("log")
ax.set_xlim(35, 700)
ax.set_xticks([40, 60, 100, 200, 400], ["40", "60", "100", "200", "400"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Identity switches, pooled (log scale; lower is better)")
ax.set_ylabel("IDF1, pooled (%)")
ax.grid(color=GRID, linewidth=1)
ax.set_axisbelow(True)
save(fig, "fig-path1-idsw-vs-idf1")
write_csv("fig-path1-idsw-vs-idf1", ["tracker", "IDSw", "IDF1", "MOTA"], [[l, x, f"{y:.2f}", f"{m:.2f}"] for l, x, y, m in pts])
print("wrote", sorted(os.listdir(OUT)))
