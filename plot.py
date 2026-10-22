import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = json.load(open("results.json"))
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e4df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG})
names = list(res)
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

# A: discovery step by rank at final checkpoint
a = ax[0, 0]
for n, c in zip(names, COL):
    f = res[n]["results"][-1]
    a.plot(range(10), f["disc_median"], color=c, lw=2, marker="o", ms=5, label=n)
a.set(title="When is the rank-i result found? (final checkpoint)",
      xlabel="True rank i (0 = nearest)", ylabel="Median expansion step at discovery")
a.set_xticks(range(10))

# B: fraction of search elapsed when rank-0 found, over time
a = ax[0, 1]
for n, c in zip(names, COL):
    R = res[n]["results"]
    a.plot([r["batch"] for r in R], [r["disc_frac_of_search"][0] for r in R], color=c, lw=2, marker="o", ms=5)
a.set(title="Rank-0 found after this share of the full search",
      xlabel="Insert batches streamed", ylabel="Median (step found / total steps)")

# C: Ranked recall, Terminus vs full search
a = ax[1, 0]
for n, c in zip(names, COL):
    R = res[n]["results"]
    b = [r["batch"] for r in R]
    a.plot(b, [r["term"]["0.1"]["rrecall"] for r in R], color=c, lw=2, marker="o", ms=5)
    a.plot(b, [r["nat_rrecall"] for r in R], color=c, lw=1.2, ls="--", alpha=0.8)
a.set(title="Ranked Recall@10: Terminus ε=0.1 (solid) vs full search (dashed)",
      xlabel="Insert batches streamed", ylabel="Ranked Recall@10")

# D: steps saved
a = ax[1, 1]
for n, c in zip(names, COL):
    R = res[n]["results"]
    a.plot([r["batch"] for r in R], [100 * r["term"]["0.1"]["saved"] for r in R], color=c, lw=2, marker="o", ms=5)
a.set(title="Expansions saved by Terminus ε=0.1", xlabel="Insert batches streamed", ylabel="% of steps saved")

for a in ax.flat:
    a.title.set_color(INK); a.title.set_fontsize(11)
fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
           labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Does 'top results come early' survive streaming updates and drift?", color=INK, fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0.04, 1, 0.98))
fig.savefig("terminus_streaming.png", dpi=150)
print("saved")