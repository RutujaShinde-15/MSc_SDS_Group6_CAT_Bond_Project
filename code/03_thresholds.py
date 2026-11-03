# =============================================================================
# STEP 3  —  CHOOSE THE POT THRESHOLDS FROM SCRATCH  (paper Sec. 2.3, 3.2,
#            Eq. 8, Figures 3 and 4)
# =============================================================================
# The paper picks u1 = 160 (AP), u2 = 12 (CAA), u3 = 15 (DEL) by:
#   (a) the mean residual life (MRL) plot, Eq. (8):  e_n(u) = mean of (x - u)
#       over floods above u. "The possible range of the threshold is detected
#       to be in the interval of (150, 200) according to the sample mean
#       residual plot with a linear trend" (Sec. 3.2);
#   (b) the parameter stability plots (Fig. 4): above a good u the GPD shape
#       xi and the modified scale sigma* = sigma - xi*u stay roughly constant.
#
# We REBUILD this choice on our data without looking at the paper's answer:
#   1. scan candidate thresholds (30%-95% quantiles);
#   2. draw both plots;
#   3. apply a written-down rule (below) to suggest u, then check by eye;
#   4. only THEN compare with (160, 12, 15).
#
# THE RULE (so the choice is reproducible, not just "by eye"):
#   Starting at the median, take the LOWEST candidate u such that, for every
#   higher candidate that still has at least STABILITY_MIN_NU floods above it,
#   the shape xi stays inside this u's 95% confidence band AND the modified
#   scale sigma* stays inside its 95% band. In words: "from this point up, the
#   tail model stops changing". (Standard reading of Fig. 4, Coles 2001, 4.3.)
#   Why the STABILITY_MIN_NU = 50 floor: the very highest thresholds leave
#   only 15-25 floods, and their estimates jump around (see the long error
#   bars on the right of Fig. 4). Demanding agreement with those noisy points
#   pushed the threshold up to the 90% quantile, which no-one would choose by
#   eye. 50 floods = about 18% of our sample. The choice is sensitive to this
#   floor (30 -> much higher thresholds), which we report as a limitation.
#
# OUTPUT: outputs/figures/step3_mrl.png (Fig. 3), step3_stability.png (Fig. 4)
#         outputs/tables/step3_threshold_scan.csv, step3_threshold_choice.csv
#         outputs/params/thresholds.json   <- used by step 4
# =============================================================================
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg

warnings.filterwarnings("ignore")
events = pd.read_csv(cfg.CLEAN_DIR / "events.csv")
QS = np.round(np.arange(cfg.CANDIDATE_Q_MIN, cfg.CANDIDATE_Q_MAX + 1e-9, cfg.CANDIDATE_Q_STEP), 4)


def fit_gpd(exc):
    """GPD maximum likelihood with large-sample standard errors
    (Coles 2001, p. 82): se(xi) = (1+xi)/sqrt(n), se(sigma) = sigma*sqrt(2(1+xi)/n)."""
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    n = len(exc)
    return xi, sigma, (1 + xi) / np.sqrt(n), sigma * np.sqrt(2 * (1 + xi) / n)


scan = []
for ind in cfg.INDICATORS:
    x = events[ind].values
    for q in QS:
        u = float(np.quantile(x, q))
        exc = x[x > u] - u
        row = {"indicator": ind, "q": q, "u": u, "n_u": len(exc),
               "mean_excess": exc.mean(),
               "me_ci": 1.96 * exc.std(ddof=1) / np.sqrt(len(exc))}
        if len(exc) >= cfg.MIN_EXCEED:
            xi, sig, se_xi, se_sig = fit_gpd(exc)
            row.update(xi=xi, xi_se=se_xi, sigma=sig, sigma_star=sig - xi * u, sigma_star_se=se_sig)
        scan.append(row)
scan = pd.DataFrame(scan)
scan.to_csv(cfg.TABLES_DIR / "step3_threshold_scan.csv", index=False)


def suggest(ind):
    d = scan[(scan.indicator == ind) & scan.xi.notna()].sort_values("q").reset_index(drop=True)
    for k, r in d.iterrows():
        if r.q < 0.5:              # below the median is not "extreme"
            continue
        higher = d.iloc[k + 1:]
        higher = higher[higher.n_u >= cfg.STABILITY_MIN_NU]
        if len(higher) == 0:
            break
        ok_xi = (abs(higher.xi - r.xi) <= 1.96 * r.xi_se).all()
        ok_ss = (abs(higher.sigma_star - r.sigma_star) <= 1.96 * r.sigma_star_se).all()
        if ok_xi and ok_ss:
            return r
    return None


choice_rows, chosen = [], {}
for ind in cfg.INDICATORS:
    x = events[ind].values
    r = suggest(ind)
    u_ours = cfg.OUR_U[ind] if cfg.OUR_U else float(r.u)
    u_paper = cfg.PAPER["u"][ind]
    q_paper = float(np.mean(x <= u_paper))       # where the paper's u sits in OUR data
    chosen[ind] = {"ours": u_ours, "paper": u_paper}
    choice_rows.append({
        "Indicator": ind, "Unit": cfg.UNITS[ind],
        "Our u (rule)": float(r.u), "quantile": float(r.q), "n above u": int(r.n_u),
        "xi at our u": r.xi,
        "Paper u": u_paper, "Paper u as quantile of our data": q_paper,
        "n above paper u (ours)": int((x > u_paper).sum()),
        "Paper u as quantile of paper data (approx)": None,
    })
choice = pd.DataFrame(choice_rows)
choice.to_csv(cfg.TABLES_DIR / "step3_threshold_choice.csv", index=False)
with open(cfg.PARAMS_DIR / "thresholds.json", "w") as f:
    json.dump(chosen, f, indent=2)

# ---- Figure 3: MRL plots ------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, ind in zip(axes, cfg.INDICATORS):
    d = scan[scan.indicator == ind]
    ax.plot(d.u, d.mean_excess, "k-", label="mean excess e(u)")
    ax.plot(d.u, d.mean_excess + d.me_ci, "b:", label="95% CI")
    ax.plot(d.u, d.mean_excess - d.me_ci, "b:")
    ax.axvline(chosen[ind]["ours"], color="red", ls="--", label=f"our u = {chosen[ind]['ours']:.1f}")
    ax.axvline(chosen[ind]["paper"], color="green", ls="-.", label=f"paper u = {chosen[ind]['paper']:g}")
    ax.set_xscale("log")
    ax.set_title(cfg.INDICATOR_NAMES[ind]); ax.set_xlabel(f"threshold u ({cfg.UNITS[ind]}, log scale)")
    ax.set_ylabel("mean excess"); ax.legend(fontsize=7)
fig.suptitle("Mean residual life plots (paper Fig. 3, Eq. 8)")
fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "step3_mrl.png", dpi=150); plt.close(fig)

# ---- Figure 4: stability plots ----------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
for j, ind in enumerate(cfg.INDICATORS):
    d = scan[(scan.indicator == ind) & scan.xi.notna()]
    for i, (col, se, lab) in enumerate([("sigma_star", "sigma_star_se", "modified scale sigma*"),
                                        ("xi", "xi_se", "shape xi")]):
        ax = axes[i, j]
        ax.errorbar(d.u, d[col], yerr=1.96 * d[se], fmt="o", ms=3, color="k", ecolor="grey", capsize=2)
        ax.axvline(chosen[ind]["ours"], color="red", ls="--")
        ax.axvline(chosen[ind]["paper"], color="green", ls="-.")
        ax.set_xscale("log"); ax.set_ylabel(lab); ax.set_xlabel(f"u ({cfg.UNITS[ind]}, log)")
        if i == 0:
            ax.set_title(cfg.INDICATOR_NAMES[ind] + "   (red = ours, green = paper)")
fig.suptitle("Parameter stability plots (paper Fig. 4)")
fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "step3_stability.png", dpi=150); plt.close(fig)

print("================ STEP 3 REPORT ================")
pd.set_option("display.width", 220)
print(choice.drop(columns=["Paper u as quantile of paper data (approx)"]).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\nShape xi at a few thresholds:")
for ind in cfg.INDICATORS:
    d = scan[(scan.indicator == ind) & scan.xi.notna()]
    print(ind, " ".join(f"q{r.q:.2f}:u={r.u:.1f},xi={r.xi:.2f}" for r in d.iloc[::6].itertuples()))
