"""Figure for template_tiers.py: SD of the estimate and 95% interval coverage by tier."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
s = pd.read_csv(f"{a.dir}/template_tiers_summary.csv")
tiers = ["100x1", "50x2", "20x5", "10x10"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, m in zip([1, 4, 16], ["o", "s", "^"], strict=True):
    r = s[(s.scheme == "iid") & (s.k == k)].set_index("tier").loc[tiers]
    ax[0].plot(tiers, 100 * r.sd, marker=m, label=f"pass@{k}")
ax[0].set(ylabel="SD of estimate across benchmarks (pp)", xlabel="templates x variants", title="Spread of the 100-problem score")
ax[0].legend(frameon=False)
for sc, col in [("iid", "C0"), ("stratified", "C1"), ("template_cluster", "C2")]:
    r = s[(s.scheme == sc) & (s.k == 1)].set_index("tier").loc[tiers]
    ax[1].plot(tiers, r.coverage, marker="o", color=col, label=sc)
ax[1].axhline(0.95, color="grey", ls="--", lw=1)
ax[1].set(ylim=(0.4, 1.0),
          ylabel="coverage of nominal 95% interval", xlabel="templates x variants", title="pass@1 interval coverage")
ax[1].legend(frameon=False)
for x in ax:
    x.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{a.dir}/fig_template_tiers.png", dpi=150)
