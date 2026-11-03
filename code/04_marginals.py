# =============================================================================
# STEP 4  —  FIT EACH INDICATOR'S FULL DISTRIBUTION (paper Sec. 3.2, Eq. 12-13,
#            Table 4, Figure 5)
# =============================================================================
# Each indicator gets a two-piece model (paper Eq. 12):
#   below u: X* = (X - m)/(u - m) ~ Beta(a, b)    (m = sample minimum)
#   above u: X - u ~ GPD(shape xi, scale sigma)
# glued together by zeta = n_u / n (Eq. 7, 13). All by maximum likelihood.
# The formulas live in cat_model.py, PART A (with the Eq. 13 typo corrected
# so that F(x) joins up at u; see the note there).
#
# Goodness of fit, as the paper does: a Chi-square test (equal-probability
# bins, ~5+ floods per bin, df = bins - 1 - 2 fitted parameters). We also
# report KS p-values.
#
# We fit TWICE: at OUR thresholds (step 3 rule) and at the PAPER's thresholds
# (160, 12, 15), so Table 4 can be compared like for like.
#
# OUTPUT: outputs/tables/step4_table4_comparison.csv
#         outputs/params/marginal_params_ours.json, marginal_params_paper.json
#         outputs/figures/step4_pp_qq_<set>.png (our Figure 5)
# =============================================================================
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import cat_model as cm

warnings.filterwarnings("ignore")
events = pd.read_csv(cfg.CLEAN_DIR / "events.csv")
thresholds = cm.load_json(cfg.PARAMS_DIR / "thresholds.json")


def chi_square_p(data, cdf, n_params=2):
    n = len(data)
    k = min(10, n // 5)                         # about 5+ expected per bin
    df = k - 1 - n_params
    if df < 1:
        return np.nan
    obs = np.histogram(cdf(np.asarray(data)), bins=np.linspace(0, 1, k + 1))[0]
    exp = n / k
    return float(1 - stats.chi2.cdf(((obs - exp) ** 2 / exp).sum(), df))


rows = []
for tag in ["ours", "paper"]:
    params = {}
    fig, axes = plt.subplots(3, 2, figsize=(9, 11))
    for r_i, ind in enumerate(cfg.INDICATORS):
        x = events[ind].values
        u = thresholds[ind][tag]
        marg = cm.BetaGPDMarginal.fit(x, u)
        exc = x[x > u] - u
        n_u = len(exc)
        y = np.clip((x[x <= u] - marg.m) / (u - marg.m), marg.EPS, 1 - marg.EPS)
        se_xi = (1 + marg.xi) / np.sqrt(n_u)
        se_s = marg.sigma * np.sqrt(2 * (1 + marg.xi) / n_u)
        gpd_cdf = lambda e: stats.genpareto.cdf(e, marg.xi, 0, marg.sigma)
        beta_cdf = lambda v: stats.beta.cdf(v, marg.a, marg.b)
        row = {"Threshold set": tag, "Indicator": ind, "u": u, "n_u": n_u, "zeta=n_u/n": n_u / len(x),
               "sigma": marg.sigma, "sigma 95% CI": f"({marg.sigma-1.96*se_s:.2f}, {marg.sigma+1.96*se_s:.2f})",
               "xi": marg.xi, "xi 95% CI": f"({marg.xi-1.96*se_xi:.2f}, {marg.xi+1.96*se_xi:.2f})",
               "GPD Chi2 p": chi_square_p(exc, gpd_cdf), "GPD KS p": stats.kstest(exc, gpd_cdf).pvalue,
               "Beta a": marg.a, "Beta b": marg.b,
               "Beta Chi2 p": chi_square_p(y, beta_cdf), "Beta KS p": stats.kstest(y, beta_cdf).pvalue,
               "Full F KS p": stats.kstest(x, marg.cdf).pvalue}
        if tag == "paper":
            p = cfg.PAPER["table4"][ind]
            row.update({"paper sigma": p["sigma"], "paper xi": p["xi"], "paper a": p["a"], "paper b": p["b"]})
        rows.append(row)
        params[ind] = marg.to_dict()
        params[ind].update(n=int(len(x)), n_u=int(n_u))

        # Figure 5: PP and QQ plots of the GPD tail
        e = np.sort(exc)
        emp = np.arange(1, n_u + 1) / (n_u + 1)
        axes[r_i, 0].scatter(emp, gpd_cdf(e), s=14, facecolors="none", edgecolors="k")
        axes[r_i, 0].plot([0, 1], [0, 1], "r-")
        axes[r_i, 0].set_title(f"{ind}: PP plot (u = {u:.1f})")
        fq = stats.genpareto.ppf(emp, marg.xi, 0, marg.sigma)
        axes[r_i, 1].scatter(fq, e, s=14, facecolors="none", edgecolors="k")
        lim = max(fq.max(), e.max())
        axes[r_i, 1].plot([0, lim], [0, lim], "r-")
        axes[r_i, 1].set_title(f"{ind}: QQ plot ({cfg.UNITS[ind]})")
        axes[r_i, 1].set_xlabel("fitted excess"); axes[r_i, 1].set_ylabel("observed excess")
    fig.suptitle(f"GPD tail fit, {tag} thresholds (paper Fig. 5)")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / f"step4_pp_qq_{tag}.png", dpi=130); plt.close(fig)
    with open(cfg.PARAMS_DIR / f"marginal_params_{tag}.json", "w") as f:
        json.dump(params, f, indent=2, default=float)

tab = pd.DataFrame(rows)
tab.to_csv(cfg.TABLES_DIR / "step4_table4_comparison.csv", index=False)
print("================ STEP 4 REPORT (Table 4) ================")
pd.set_option("display.width", 250)
print(tab.T.to_string(header=False, float_format=lambda v: f"{v:,.3f}"))
