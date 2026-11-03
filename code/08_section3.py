# =============================================================================
# STEP 8  —  SECTION 3 OF THE PAPER, REBUILT WITH OUR RESULTS
#            (Tables 3-7, Figures 2-8, numbered exactly as in Tang et al. 2023)
# =============================================================================
# Steps 0-7 did the analysis. This step collects the MAIN-CASE results into
# the paper's own tables and figures, so our Section 3 can be read side by
# side with theirs. Main case (a consistent chain, as in the paper):
#   - thresholds:  OUR rule-based choice from step 3 (196.9, 16.43, 27.32)
#   - margins:     Beta-GP fitted at those thresholds (step 4)
#   - copula:      nested Frank, selected by the KS rule as in the paper (step 5)
#   - frequency:   OUR ARMA(1,3) forecast Lambda = (15.05, 16.24, 15.49) (step 6)
#   - interest:    the paper's CIR model, Eq. (15)
#   - cashflows:   Eq. (4) exactly as printed
# Output: outputs/section3/table3.csv ... table7.csv, fig2.png ... fig8.png
#         and section3_numbers.json (every number quoted in the write-up)
# =============================================================================
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import config as cfg
import cat_model as cm

warnings.filterwarnings("ignore")
OUT = cfg.ROOT / "outputs" / "section3"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})
P = cfg.PAPER
NAMES = {"AP": "affected population (AP)", "CAA": "crop affected area (CAA)",
         "DEL": "direct economic loss (DEL)"}
UNIT = {"AP": "10,000 persons", "CAA": "10,000 ha", "DEL": "100 million CNY"}
LET = ["a", "b", "c"]
num = {}

events = pd.read_csv(cfg.CLEAN_DIR / "events.csv")
counts = pd.read_csv(cfg.CLEAN_DIR / "annual_counts.csv")
thr = cm.load_json(cfg.PARAMS_DIR / "thresholds.json")
margs = cm.load_marginals(cfg, "ours")
mpar = cm.load_json(cfg.PARAMS_DIR / "marginal_params_ours.json")
cop_p = cm.load_json(cfg.PARAMS_DIR / "copula_params_ours.json")
freq = cm.load_json(cfg.PARAMS_DIR / "frequency.json")
scan = pd.read_csv(cfg.TABLES_DIR / "step3_threshold_scan.csv")
X = events[cfg.INDICATORS].values
n = len(events)
num["n"] = n

# =============================================================================
# 3.1  Table 3 and Figure 2
# =============================================================================
t3 = []
for ind in cfg.INDICATORS:
    x = events[ind].values
    t3.append({"Trigger indicator": ind, "Maximum": x.max(), "Minimum": x.min(), "Mean": x.mean(),
               "Median": np.median(x), "Skewness": stats.skew(x),
               "Kurtosis": stats.kurtosis(x, fisher=False)})
t3 = pd.DataFrame(t3)
t3.to_csv(OUT / "table3.csv", index=False)
num["table3"] = t3.set_index("Trigger indicator").round(3).to_dict(orient="index")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for k, (ax, ind) in enumerate(zip(axes, cfg.INDICATORS)):
    x = np.sort(events[ind].values)
    theo = -np.log(1 - np.arange(1, n + 1) / (n + 1))
    ax.plot(theo, x, "k.", ms=4, label="observed (sorted)")
    ax.plot(theo, x.mean() * theo, "r-", lw=1.4, label="exponential reference")
    ax.set_title(f"({LET[k]}) {ind}")
    ax.set_xlabel("standard exponential quantile"); ax.set_ylabel(UNIT[ind])
    ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig2.png", dpi=150); plt.close(fig)

# =============================================================================
# 3.2  Figures 3-4, Table 4, Figure 5
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for k, (ax, ind) in enumerate(zip(axes, cfg.INDICATORS)):
    d = scan[scan.indicator == ind]
    ax.plot(d.u, d.mean_excess, "k-", lw=1.4, label="mean excess $e_n(u)$")
    ax.plot(d.u, d.mean_excess + d.me_ci, "b:", lw=1.2, label="95% CI")
    ax.plot(d.u, d.mean_excess - d.me_ci, "b:", lw=1.2)
    ax.axvline(thr[ind]["ours"], color="red", ls="--", lw=1.2, label=f"selected u = {thr[ind]['ours']:.1f}")
    ax.set_xscale("log"); ax.set_title(f"({LET[k]}) {ind}")
    ax.set_xlabel(f"threshold u ({UNIT[ind]}, log scale)"); ax.set_ylabel("mean excess")
    ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig3.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(11, 10))
labels = ["a", "b", "c", "d", "e", "f"]
for r, ind in enumerate(cfg.INDICATORS):
    d = scan[(scan.indicator == ind) & scan.xi.notna()]
    for c, (col, se, lab) in enumerate([("sigma_star", "sigma_star_se", r"modified scale $\sigma^*$"),
                                        ("xi", "xi_se", r"shape $\xi$")]):
        ax = axes[r, c]
        ax.errorbar(d.u, d[col], yerr=1.96 * d[se], fmt="o", ms=3, color="k", ecolor="grey", capsize=2)
        u0 = thr[ind]["ours"]
        sel = d.iloc[(d.u - u0).abs().argmin()]
        ax.add_patch(plt.Rectangle((sel.u * 0.96, sel[col] - 1.96 * sel[se] * 1.15),
                                   sel.u * 0.08, 2 * 1.96 * sel[se] * 1.15, fill=False, ec="red", lw=1.5))
        ax.set_xscale("log"); ax.set_ylabel(lab)
        ax.set_xlabel(f"threshold u ({UNIT[ind]}, log scale)")
        ax.set_title(f"({labels[2 * r + c]}) {ind}")
fig.tight_layout(); fig.savefig(OUT / "fig4.png", dpi=150); plt.close(fig)


def chi_p(data, cdf, npar=2):
    k = min(10, len(data) // 5); df = k - 1 - npar
    obs = np.histogram(cdf(np.asarray(data)), bins=np.linspace(0, 1, k + 1))[0]
    e = len(data) / k
    return float(1 - stats.chi2.cdf(((obs - e) ** 2 / e).sum(), df))


t4 = []
for ind, mg in zip(cfg.INDICATORS, margs):
    x = events[ind].values
    exc = x[x > mg.u] - mg.u
    nu = len(exc)
    se_s = mg.sigma * np.sqrt(2 * (1 + mg.xi) / nu); se_x = (1 + mg.xi) / np.sqrt(nu)
    y = np.clip((x[x <= mg.u] - mg.m) / (mg.u - mg.m), 1e-6, 1 - 1e-6)
    t4.append({"Indicator": ind, "u": mg.u, "n_u": nu, "m (minimum)": mg.m,
               "Scale sigma": mg.sigma, "sigma 95% CI": f"({mg.sigma - 1.96 * se_s:.2f}, {mg.sigma + 1.96 * se_s:.2f})",
               "Shape xi": mg.xi, "xi 95% CI": f"({mg.xi - 1.96 * se_x:.2f}, {mg.xi + 1.96 * se_x:.2f})",
               "GP p-value (Chi2)": chi_p(exc, lambda e: stats.genpareto.cdf(e, mg.xi, 0, mg.sigma)),
               "GP p-value (KS)": stats.kstest(exc, "genpareto", args=(mg.xi, 0, mg.sigma)).pvalue,
               "Beta a": mg.a, "Beta b": mg.b,
               "Beta p-value (Chi2)": chi_p(y, lambda v: stats.beta.cdf(v, mg.a, mg.b)),
               "Beta p-value (KS)": stats.kstest(y, "beta", args=(mg.a, mg.b)).pvalue})
t4 = pd.DataFrame(t4)
t4.to_csv(OUT / "table4.csv", index=False)
num["table4"] = t4.set_index("Indicator").to_dict(orient="index")

# Figure 5: PP and QQ plots of the GP tail, with 95% simulation envelopes
rng = np.random.default_rng(11)
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
for r, (ind, mg) in enumerate(zip(cfg.INDICATORS, margs)):
    x = events[ind].values
    e = np.sort(x[x > mg.u] - mg.u); nu = len(e)
    pemp = np.arange(1, nu + 1) / (nu + 1)
    sims = np.sort(stats.genpareto.rvs(mg.xi, 0, mg.sigma, size=(2000, nu), random_state=rng), axis=1)
    psims = stats.genpareto.cdf(sims, mg.xi, 0, mg.sigma)
    ax = axes[r, 0]
    ax.plot(pemp, stats.genpareto.cdf(e, mg.xi, 0, mg.sigma), "o", mfc="none", mec="k", ms=4)
    ax.plot([0, 1], [0, 1], "k-", lw=1)
    ax.plot(pemp, np.quantile(psims, 0.025, axis=0), "k:", lw=1)
    ax.plot(pemp, np.quantile(psims, 0.975, axis=0), "k:", lw=1)
    ax.set_xlabel("empirical probability"); ax.set_ylabel("fitted GP probability")
    ax.set_title(f"({LET[r]}) {ind}: PP plot")
    ax = axes[r, 1]
    fq = stats.genpareto.ppf(pemp, mg.xi, 0, mg.sigma)
    ax.plot(fq, e, "o", mfc="none", mec="k", ms=4)
    lim = max(fq.max(), e.max())
    ax.plot([0, lim], [0, lim], "k-", lw=1)
    ax.plot(fq, np.quantile(sims, 0.025, axis=0), "k:", lw=1)
    ax.plot(fq, np.quantile(sims, 0.975, axis=0), "k:", lw=1)
    ax.set_xlabel(f"fitted GP quantile ({UNIT[ind]})"); ax.set_ylabel(f"observed excess ({UNIT[ind]})")
    ax.set_title(f"({LET[r]}) {ind}: QQ plot")
fig.tight_layout(); fig.savefig(OUT / "fig5.png", dpi=150); plt.close(fig)

# =============================================================================
# 3.3  Spearman + Table 5
# =============================================================================
sp = pd.read_csv(cfg.TABLES_DIR / "step5_spearman.csv")
num["spearman"] = dict(zip(sp.pair, sp["Spearman rho (ours)"].round(3)))
t5 = pd.read_csv(cfg.TABLES_DIR / "step5_table5_ours.csv")[
    ["Copula", "theta1 (inner)", "theta2 (outer)", "KS", "KS p (bootstrap)", "log-lik", "AIC",
     "tau inner (model)", "tau outer (model)", "tau inner (data)", "tau outer (data)"]]
t5.to_csv(OUT / "table5.csv", index=False)
num["table5"] = t5.set_index("Copula").to_dict(orient="index")

# =============================================================================
# 3.4  Frequency: Figure 6, Figure 7, ARMA numbers
# =============================================================================
fit = counts[counts.year <= 2020]
y = fit.n_major.values.astype(float)
m13 = ARIMA(y, order=(1, 0, 3), trend="c").fit()
m00 = ARIMA(y, order=(0, 0, 0), trend="c").fit()
lr = 2 * (m13.llf - m00.llf)
num["arma"] = {"params": dict(zip(m13.param_names, np.round(m13.params, 3))),
               "LR_stat": lr, "LR_p": float(1 - stats.chi2.cdf(lr, 4)),
               "forecast": freq["ours_arma13"], "actual_2021_2023": freq["actual_2021_2023"]}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
plot_pacf(y, ax=axes[0], lags=10, method="ywm"); axes[0].set_title("(a) PACF")
plot_acf(y, ax=axes[1], lags=10); axes[1].set_title("(b) ACF")
for ax in axes:
    ax.set_xlabel("lag (years)")
fig.tight_layout(); fig.savefig(OUT / "fig6.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(fit.year, y, "k-o", ms=4, label="observed, 1994-2020")
ax.plot([2020, 2021, 2022, 2023], [y[-1]] + freq["ours_arma13"], "r--o", ms=4,
        label="ARMA(1,3) forecast, 2021-2023")
ax.plot(counts[counts.year > 2020].year, freq["actual_2021_2023"], "s", color="grey",
        label="recorded 2021-2023 (held out)")
ax.set_xlabel("year"); ax.set_ylabel("number of major rainstorms")
ax.legend(fontsize=9); ax.set_xticks(range(1994, 2024, 2)); ax.tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(OUT / "fig7.png", dpi=150); plt.close(fig)

# =============================================================================
# 3.5  Pricing: Table 6, Figure 8, Table 7
# =============================================================================
F, R = P["F"], P["R"]
s = cop_p["nested"]["frank"]
frank = cm.NestedCopula("frank", s["theta_inner"], s["theta_outer"], cop_p["inner"], cop_p["outer"])
LAM = freq["ours_arma13"]


def cir_integral(nsim, T, rng, spy=52):
    k, th, eps, r0 = P["cir"]["kappa"], P["cir"]["theta"], P["cir"]["eps"], P["cir"]["r0"]
    dt = 1 / spy; c = eps ** 2 * (1 - np.exp(-k * dt)) / (4 * k); df = 4 * k * th / eps ** 2
    r = np.full(nsim, r0); I = np.zeros((nsim, T + 1)); acc = np.zeros(nsim)
    for step in range(1, T * spy + 1):
        rn = c * rng.noncentral_chisquare(df, r * np.exp(-k * dt) / c)
        acc += 0.5 * (r + rn) * dt; r = rn
        if step % spy == 0:
            I[:, step // spy] = acc
    return I


def retentions(lams, d, nsim, seed, columns=None):
    rng = np.random.default_rng(seed); parts = []
    for st in range(0, nsim, 5000):
        parts.append(cm.simulate_retentions(min(5000, nsim - st), lams, frank, margs, d, rng, columns))
    return {k: np.vstack([p[k] for p in parts]) for k in parts[0]}


def price(res, how, t, T, I):
    a, b, g = res[f"alpha_{how}"], res[f"beta_{how}"], res[f"gamma_{how}"]
    pi = np.where(g < 1, b / 2, b)
    carried = np.ones(len(a)); pv = np.zeros(len(a))
    for s_ in range(t + 1, T + 1):
        pv += a[:, s_ - 1] * carried * F * R * np.exp(-(I[:, s_] - I[:, t]))
        carried = carried * pi[:, s_ - 1]
    pv += carried * F * np.exp(-(I[:, T] - I[:, t]))
    return pv.mean(), pv.std(ddof=1) / np.sqrt(len(pv))


N = cfg.N_SIMS
I = cir_integral(N, 3, np.random.default_rng(cfg.SEED + 1))
d90 = cm.trigger_levels(cfg, events, 0.90)
num["trigger_90"] = dict(zip(cfg.INDICATORS, d90.round(2)))
res = retentions(LAM, d90, N, cfg.SEED)
t6 = []
for T in (1, 2, 3):
    for t in range(T):
        p1, s1 = price(res, "avg", t, T, I); p2, s2 = price(res, "max", t, T, I)
        t6.append({"T": T, "t": t, "P1 (average)": p1, "se P1": s1, "P2 (maximum)": p2, "se P2": s2})
t6 = pd.DataFrame(t6); t6.to_csv(OUT / "table6.csv", index=False)
num["table6"] = t6.round(3).to_dict(orient="records")
# share of simulated years with principal halved / cut, and mean retentions
num["year1"] = {"P(gamma<1)": float((res["gamma_avg"][:, 0] < 1).mean()),
                "P(beta<1)": float((res["beta_avg"][:, 0] < 1).mean()),
                "mean alpha_avg": float(res["alpha_avg"][:, 0].mean()),
                "mean beta_avg": float(res["beta_avg"][:, 0].mean()),
                "mean alpha_max": float(res["alpha_max"][:, 0].mean()),
                "P(alpha_max<1)": float((res["alpha_max"][:, 0] < 1).mean())}

# Figure 8
n8 = 20_000
I8 = I[:n8]
qs = np.round(np.arange(0.80, 0.901, 0.02), 2)
lams = [10, 15, 20, 25, 30, 35, 40]
fa = [[price(retentions(LAM[:1], cm.trigger_levels(cfg, events, q), n8, cfg.SEED), how, 0, 1, I8)[0]
       for q in qs] for how in ("avg", "max")]
fb = [[price(retentions([L], d90, n8, cfg.SEED), how, 0, 1, I8)[0] for L in lams] for how in ("avg", "max")]
num["fig8a"] = {"q": qs.tolist(), "P1": np.round(fa[0], 2).tolist(), "P2": np.round(fa[1], 2).tolist()}
num["fig8b"] = {"Lambda1": lams, "P1": np.round(fb[0], 2).tolist(), "P2": np.round(fb[1], 2).tolist()}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(qs, fa[0], "k-o", label="$P^1$ (average retention)")
ax[0].plot(qs, fa[1], "b--s", label="$P^2$ (maximum retention)")
ax[0].set_xlabel("trigger level (sample quantile q)"); ax[0].set_ylabel("one-year bond price (CNY)")
ax[0].set_title(r"(a) trigger level, $\Lambda_1$ = %.2f" % LAM[0]); ax[0].legend(fontsize=9)
ax[1].plot(lams, fb[0], "k-o", label="$P^1$ (average retention)")
ax[1].plot(lams, fb[1], "b--s", label="$P^2$ (maximum retention)")
ax[1].set_xlabel(r"intensity measure $\Lambda_1$"); ax[1].set_ylabel("one-year bond price (CNY)")
ax[1].set_title("(b) intensity, trigger at 90% quantile"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig8.png", dpi=150); plt.close(fig)

# Table 7
t7 = []
for label, cols in [("AP-CAA-DEL", None), ("AP-CAA", [0, 1]), ("AP-DEL", [0, 2]), ("CAA-DEL", [1, 2])]:
    r7 = retentions(LAM[:1], d90, N, cfg.SEED, columns=cols)
    t7.append({"Trigger indicators": label, "P1": price(r7, "avg", 0, 1, I)[0],
               "P2": price(r7, "max", 0, 1, I)[0],
               "P(principal cut in year 1)": float((np.where(r7["gamma_avg"][:, 0] < 1, r7["beta_avg"][:, 0] / 2,
                                                            r7["beta_avg"][:, 0]) < 1).mean())})
t7 = pd.DataFrame(t7); t7.to_csv(OUT / "table7.csv", index=False)
num["table7"] = t7.round(4).to_dict(orient="records")

# Per-flood trigger probabilities under the fitted model
U = frank.sample(400_000, np.random.default_rng(3))
Xs = np.column_stack([margs[i].ppf(U[:, i]) for i in range(3)])
k = (Xs > d90).sum(1)
num["per_flood"] = {"P>=1": float((k >= 1).mean()), "P>=2": float((k >= 2).mean()), "P3": float((k == 3).mean())}

with open(OUT / "section3_numbers.json", "w") as f:
    json.dump(num, f, indent=1, default=float)
print(json.dumps(num, indent=1, default=float))
