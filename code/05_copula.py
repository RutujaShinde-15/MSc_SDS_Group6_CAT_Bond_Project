# =============================================================================
# STEP 5  —  DEPENDENCE BETWEEN AP, CAA AND DEL: NESTED COPULA
#            (paper Sec. 3.3, Eq. 9 and 14, Table 5)
# =============================================================================
# What the paper does (Sec. 3.3):
#   1. Spearman rho for each pair: (AP,CAA) 0.771, (AP,DEL) 0.554,
#      (CAA,DEL) 0.515. The strongest pair goes in the INNER copula.
#   2. u_ij = F_i(x_ij) with the fitted Beta-GP margins (Eq. 13).
#   3. Nested copula C_outer(C_inner(u_AP, u_CAA; theta1), u_DEL; theta2) for
#      Gumbel, Clayton and Frank, by maximum likelihood (Table 5).
#   4. Choose the family with the smallest KS distance / largest p-value.
#
# We do the same. KS distance = largest gap between the data's empirical
# copula and the fitted copula at the data points. Its p-value comes from a
# parametric bootstrap: simulate GOF_BOOTSTRAP fake datasets from the fitted
# copula, refit each, and count how often the fake KS is at least as large
# as ours (the paper does not say how it got its p-values).
#
# EXTRA CHECK (not in the paper): Kendall's tau implied by each fitted theta,
# compared with the tau in the data. A copula with the right strength of
# dependence must reproduce the data's tau. This exposes a problem with the
# paper's Table 5 (see report).
#
# OUTPUT: outputs/tables/step5_spearman.csv, step5_table5_<set>.csv
#         outputs/params/copula_params_<set>.json
#         outputs/figures/step5_scatter_<set>.png
# =============================================================================
import json, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import cat_model as cm

warnings.filterwarnings("ignore")
rng = np.random.default_rng(cfg.SEED)
events = pd.read_csv(cfg.CLEAN_DIR / "events.csv")
X = events[cfg.INDICATORS].values
n = len(X)

# ---- 1. Spearman and Kendall for each pair ---------------------------------
pair_rows = []
for i, j in itertools.combinations(range(3), 2):
    name = f"{cfg.INDICATORS[i]}-{cfg.INDICATORS[j]}"
    rho, p = stats.spearmanr(X[:, i], X[:, j])
    tau = stats.kendalltau(X[:, i], X[:, j])[0]
    pair_rows.append({"pair": name, "i": i, "j": j, "Spearman rho (ours)": rho, "p-value": p,
                      "Spearman rho (paper)": cfg.PAPER["spearman"][name], "Kendall tau (ours)": tau})
pairs = pd.DataFrame(pair_rows)
pairs.drop(columns=["i", "j"]).to_csv(cfg.TABLES_DIR / "step5_spearman.csv", index=False)
best = pairs.loc[pairs["Spearman rho (ours)"].idxmax()]
inner = [int(best.i), int(best.j)]
outer = [k for k in range(3) if k not in inner][0]
print("================ STEP 5 REPORT ================")
print(pairs.drop(columns=["i", "j"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"Inner pair (strongest): {best.pair}; outer: {cfg.INDICATORS[outer]}  (paper: AP-CAA inner, DEL outer)")

# What the paper's own thetas imply (tau of a Gumbel with theta 44.68 is 0.98)
paper_tau_rows = []
for fam, (t1, t2, ks, p) in cfg.PAPER["table5"].items():
    paper_tau_rows.append({"family": fam, "paper theta1": t1, "implied tau inner": cm.kendall_tau(fam, t1),
                           "paper theta2": t2, "implied tau outer": cm.kendall_tau(fam, t2)})
paper_tau = pd.DataFrame(paper_tau_rows)
paper_tau.to_csv(cfg.TABLES_DIR / "step5_paper_theta_implied_tau.csv", index=False)
print("\nKendall tau implied by the PAPER's Table 5 thetas:")
print(paper_tau.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
# What thetas WOULD match the paper's own Spearman values? Found by simulating
# bivariate copulas and solving Spearman(theta) = target (400,000 draws):
#   Spearman 0.771 (inner)       -> Gumbel 2.40 (tau 0.58), Frank 7.18 (tau 0.57)
#   Spearman ~0.535 (outer avg.) -> Gumbel 1.61 (tau 0.38), Frank 3.77 (tau 0.37)
# The paper reports 44.68/22.80 (Gumbel) and 176.34/87.60 (Frank): about 20x
# too large to be consistent with its own correlations.
print("Thetas consistent with the paper's own Spearman: Gumbel 2.40 / 1.61, "
      "Frank 7.18 / 3.77 (tau 0.58 / 0.38).")

# ---- 2-4. Fit for both threshold sets ---------------------------------------
for tag in ["ours", "paper"]:
    margs = cm.load_marginals(cfg, tag)
    U = np.clip(np.column_stack([margs[k].cdf(X[:, k]) for k in range(3)]), 1e-6, 1 - 1e-6)
    tau_in_data = stats.kendalltau(U[:, inner[0]], U[:, inner[1]])[0]
    tau_out_data = np.mean([stats.kendalltau(U[:, k], U[:, outer])[0] for k in inner])
    rows, saved = [], {}
    for fam in cm.FAMILIES:
        cop = cm.NestedCopula.fit(U, fam, inner, outer)
        ks = cm.ks_distance(U, cop)
        fake = []
        for _ in range(cfg.GOF_BOOTSTRAP):
            Uf = cop.sample(n, rng)
            fake.append(cm.ks_distance(Uf, cm.NestedCopula.fit(Uf, fam, inner, outer, start=cop)))
        pval = float(np.mean(np.array(fake) >= ks))
        k_par = 2
        rows.append({"Copula": fam.capitalize(), "theta1 (inner)": cop.theta_inner,
                     "theta2 (outer)": cop.theta_outer,
                     "tau inner (model)": cm.kendall_tau(fam, cop.theta_inner),
                     "tau outer (model)": cm.kendall_tau(fam, cop.theta_outer),
                     "tau inner (data)": tau_in_data, "tau outer (data)": tau_out_data,
                     "log-lik": cop.loglik, "AIC": 2 * k_par - 2 * cop.loglik,
                     "KS": ks, "KS p (bootstrap)": pval,
                     "paper theta1": cfg.PAPER["table5"][fam][0], "paper theta2": cfg.PAPER["table5"][fam][1],
                     "paper KS": cfg.PAPER["table5"][fam][2], "paper p": cfg.PAPER["table5"][fam][3]})
        saved[fam] = {"theta_inner": cop.theta_inner, "theta_outer": cop.theta_outer,
                      "loglik": cop.loglik, "ks": ks, "ks_p": pval}
    tab = pd.DataFrame(rows)
    tab.to_csv(cfg.TABLES_DIR / f"step5_table5_{tag}.csv", index=False)
    # Selection rule of the paper: smallest KS (ties -> largest p)
    sel = tab.sort_values(["KS", "KS p (bootstrap)"], ascending=[True, False]).iloc[0]["Copula"].lower()
    sel_aic = tab.loc[tab.AIC.idxmin(), "Copula"].lower()
    with open(cfg.PARAMS_DIR / f"copula_params_{tag}.json", "w") as f:
        json.dump({"inner": inner, "outer": outer, "nested": saved,
                   "selected_family": sel, "selected_by_aic": sel_aic}, f, indent=2, default=float)
    print(f"\n--- Table 5, {tag} thresholds ---")
    print(tab.T.to_string(header=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Selected by KS (paper's rule): {sel};  by AIC: {sel_aic}")

    # Joint upper-tail check: share of floods with ALL three above their 90%
    # level, data vs each fitted copula (matters most for the bond).
    q = 0.9
    obs = np.mean(np.all(U > q, axis=1))
    msg = f"  P(all three in top 10%): data {obs:.3f}"
    for fam in cm.FAMILIES:
        s = saved[fam]
        Us = cm.NestedCopula(fam, s["theta_inner"], s["theta_outer"], inner, outer).sample(200_000, rng)
        msg += f", {fam} {np.mean(np.all(Us > q, axis=1)):.3f}"
    msg += f", independence {0.1**3:.3f}"
    print(msg)

    # Figure: data vs simulated from selected family
    s = saved[sel]
    Us = cm.NestedCopula(sel, s["theta_inner"], s["theta_outer"], inner, outer).sample(n, rng)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for c, (i, j) in enumerate(itertools.combinations(range(3), 2)):
        for r, (D, lab) in enumerate([(U, "data u = F(x)"), (Us, f"simulated nested {sel}")]):
            axes[r, c].scatter(D[:, i], D[:, j], s=8, alpha=0.6)
            axes[r, c].set_xlabel(cfg.INDICATORS[i]); axes[r, c].set_ylabel(cfg.INDICATORS[j])
            axes[r, c].set_title(lab)
    fig.suptitle(f"Copula scale: data vs fitted nested {sel} ({tag} thresholds)")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / f"step5_scatter_{tag}.png", dpi=120); plt.close(fig)
