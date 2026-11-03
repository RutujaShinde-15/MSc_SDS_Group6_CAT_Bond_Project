# =============================================================================
# STEP 2  —  DESCRIBE THE TRIGGER INDICATORS (paper Sec. 3.1, Table 3, Fig. 2)
# =============================================================================
# The paper argues (Sec. 3.1) that "the averages of all three main indicators
# are far larger than their respective medians" and skewness/kurtosis are "far
# larger than 0 and 3", so the data are heavy-tailed and need extreme value
# theory (POT). We compute the same numbers on our data and put the paper's
# Table 3 beside them.
#
# Note on kurtosis: the paper compares kurtosis with 3, so it uses the
# "normal = 3" convention. We report that version (and the excess version).
#
# OUTPUT: outputs/tables/step2_table3_comparison.csv
#         outputs/figures/step2_qq.png   (our Figure 2)
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg

events = pd.read_csv(cfg.CLEAN_DIR / "events.csv")
team = pd.read_csv(cfg.CLEAN_DIR / "events_team279.csv")      # version 1, for comparison


def describe(x):
    return [x.max(), x.min(), x.mean(), np.median(x), stats.skew(x), stats.kurtosis(x, fisher=False)]


rows = []
for ind in cfg.INDICATORS:
    ours, old = describe(events[ind].values), describe(team[ind].values)
    paper = cfg.PAPER["table3"][ind]
    for name, p, o, t in zip(["Maximum", "Minimum", "Mean", "Median", "Skewness", "Kurtosis"],
                             paper, ours, old):
        rows.append({"Indicator": ind, "Statistic": name, "Paper (n=245)": p,
                     f"New compilation (n={len(events)})": o, "Team v1 (n=279)": t})
for (a, b) in [("AP", "CAA"), ("AP", "DEL"), ("CAA", "DEL")]:
    rows.append({"Indicator": f"{a}-{b}", "Statistic": "Spearman",
                 "Paper (n=245)": cfg.PAPER["spearman"][f"{a}-{b}"],
                 f"New compilation (n={len(events)})": stats.spearmanr(events[a], events[b])[0],
                 "Team v1 (n=279)": stats.spearmanr(team[a], team[b])[0]})
tab = pd.DataFrame(rows)
tab.to_csv(cfg.TABLES_DIR / "step2_table3_comparison.csv", index=False)

# Robustness: the same statistics after dropping our 10 largest floods by AP.
trim = events.nsmallest(len(events) - 10, "AP")
trim_rows = [{"Indicator": ind, "Mean": trim[ind].mean(), "Median": trim[ind].median(),
              "Max": trim[ind].max(), "Skewness": stats.skew(trim[ind]),
              "Kurtosis": stats.kurtosis(trim[ind], fisher=False)} for ind in cfg.INDICATORS]
pd.DataFrame(trim_rows).to_csv(cfg.TABLES_DIR / "step2_table3_without_top10.csv", index=False)

# ---- Figure 2: exponential QQ plots -----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, ind in zip(axes, cfg.INDICATORS):
    x = np.sort(events[ind].values)
    n = len(x)
    theo = -np.log(1 - np.arange(1, n + 1) / (n + 1))     # exponential quantiles
    ax.scatter(theo, x, s=12, facecolors="none", edgecolors="black", label="Observed")
    ax.plot(theo, x.mean() * theo, color="red", lw=1.4, label="Exponential reference")
    ax.set_title(cfg.INDICATOR_NAMES[ind])
    ax.set_xlabel("Exponential quantile"); ax.set_ylabel(cfg.UNITS[ind])
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Exponential QQ plots, China CFDID 2006-2020 (paper Fig. 2)")
fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "step2_qq.png", dpi=150); plt.close(fig)

print("================ STEP 2 REPORT (Table 3) ================")
pd.set_option("display.width", 200)
print(tab.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\nWithout our 10 largest floods (by AP):")
print(pd.DataFrame(trim_rows).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
