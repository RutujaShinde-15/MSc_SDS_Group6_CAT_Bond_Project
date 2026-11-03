# =============================================================================
# STEP 6  —  ANNUAL FREQUENCY OF MAJOR RAINSTORMS (paper Sec. 3.4, Figs. 6-7,
#            Appendix A.2)
# =============================================================================
# What the paper does:
#   - series: annual number of major rainstorms in China, 1986-2020
#     (China Statistical Yearbook of Natural Disasters - licensed, not public);
#   - ADF test for stationarity (5% level);
#   - ARMA order from ACF / PACF cut-offs -> ARMA(1,3), MLE
#     (phi1, theta1, theta2, theta3) = (0.816, 0.268, 0.240, -0.748);
#   - residual checks: Jarque-Bera (p = 0.08), Ljung-Box (p = 0.502);
#   - forecast intensities Lambda for 2021-2023 = (41.86, 41.56, 39.39);
#   - Pearson correlation of the yearly count with yearly average AP, CAA,
#     DEL is near zero (0.08, -0.09, -0.1): supports frequency-severity
#     independence.
#
# What we can do with public data:
#   - CFDID counts of major province-level rainstorm records, 1994-2020
#     (27 years; CFDID starts in 1994), same rules as the severity file;
#   - same tests, ARMA(1,3) as in the paper AND the order chosen by AIC;
#   - forecast 2021-2023 and CHECK it against CFDID's actual 2021-2023
#     counts (which the paper could not do);
#   - the same Pearson independence check.
# IMPORTANT CAVEAT: CFDID's record count depends on which yearbooks cover
# which years (the meteorological yearbook starts in 2002), so the series
# mixes real frequency with reporting coverage. We therefore price with
# BOTH the paper's forecast (like-for-like replication) and our own.
#
# OUTPUT: outputs/tables/step6_*.csv, outputs/figures/step6_*.png,
#         outputs/params/frequency.json
# =============================================================================
import json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import config as cfg

warnings.filterwarnings("ignore")
counts = pd.read_csv(cfg.CLEAN_DIR / "annual_counts.csv")
fit_part = counts[counts.year <= 2020]
check_part = counts[counts.year > 2020]
y = fit_part.n_major.values.astype(float)

# ---- ADF stationarity test ---------------------------------------------------
adf_stat, adf_p, *_ = adfuller(y, autolag="AIC")

# ---- ARMA models ----------------------------------------------------------------
def fit_arma(p, q):
    return ARIMA(y, order=(p, 0, q), trend="c").fit()

grid = []
for p, q in itertools.product(range(0, 3), range(0, 4)):
    try:
        m = fit_arma(p, q)
        grid.append({"p": p, "q": q, "AIC": m.aic, "BIC": m.bic})
    except Exception:
        pass
grid = pd.DataFrame(grid).sort_values("AIC")
grid.to_csv(cfg.TABLES_DIR / "step6_arma_grid.csv", index=False)
best_p, best_q = int(grid.iloc[0].p), int(grid.iloc[0].q)

rows, forecasts = [], {}
for label, (p, q) in [("ARMA(1,3) as paper", (1, 3)), (f"ARMA({best_p},{best_q}) best AIC", (best_p, best_q))]:
    m = fit_arma(p, q)
    res = m.resid
    fc = m.get_forecast(3).predicted_mean
    forecasts[label] = [float(max(v, 0.1)) for v in fc]
    rows.append({"model": label, "params": "; ".join(f"{k}={v:.3f}" for k, v in zip(m.param_names, m.params)),
                 "AIC": m.aic, "Jarque-Bera p": stats.jarque_bera(res).pvalue,
                 "Ljung-Box p (lag 5)": float(acorr_ljungbox(res, lags=[5]).lb_pvalue.iloc[0]),
                 "forecast 2021": fc[0], "forecast 2022": fc[1], "forecast 2023": fc[2]})
tab = pd.DataFrame(rows)
tab.to_csv(cfg.TABLES_DIR / "step6_arma.csv", index=False)

# ---- Frequency-severity independence (Pearson) -------------------------------
events = pd.read_csv(cfg.CLEAN_DIR / "events.csv")
yearly = events.groupby("year")[cfg.INDICATORS].mean()
yearly["n"] = counts.set_index("year").loc[yearly.index, "n_major"]
pearson = {ind: stats.pearsonr(yearly.n, yearly[ind])[0] for ind in cfg.INDICATORS}

# ---- Poisson dispersion check (pricing step 1 assumes Poisson) ---------------
disp = y.var(ddof=1) / y.mean()

with open(cfg.PARAMS_DIR / "frequency.json", "w") as f:
    json.dump({"paper_lambda": cfg.PAPER["arma_forecast"],
               "ours_arma13": forecasts["ARMA(1,3) as paper"],
               "ours_best": forecasts[[k for k in forecasts if "best" in k][0]],
               "best_order": [best_p, best_q],
               "actual_2021_2023": check_part.n_major.tolist(),
               "mean_2006_2020": float(fit_part[fit_part.year >= 2006].n_major.mean())}, f, indent=2)

# ---- Figures 6 and 7 -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
plot_pacf(y, ax=axes[0], lags=10, method="ywm"); axes[0].set_title("(a) PACF")
plot_acf(y, ax=axes[1], lags=10); axes[1].set_title("(b) ACF")
fig.suptitle("Annual major rainstorm records, CFDID 1994-2020 (paper Fig. 6)")
fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "step6_acf_pacf.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(fit_part.year, y, "k-o", ms=4, label="CFDID major records (fit)")
ax.plot(check_part.year, check_part.n_major, "o", color="grey", label="CFDID 2021-2023 (held out)")
for lab, fc in forecasts.items():
    ax.plot([2021, 2022, 2023], fc, "--o", ms=4, label=f"forecast: {lab}")
ax.plot([2021, 2022, 2023], cfg.PAPER["arma_forecast"], "r--s", ms=4, label="paper forecast")
ax.set_xlabel("year"); ax.set_ylabel("events per year"); ax.legend(fontsize=8)
ax.set_title("Annual number of major rainstorms (paper Fig. 7)")
fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "step6_series.png", dpi=140); plt.close(fig)

print("================ STEP 6 REPORT ================")
print(f"Series 1994-2020: mean {y.mean():.1f}, variance/mean = {disp:.1f} (Poisson would be ~1)")
print(f"ADF p-value: {adf_p:.3f}  (paper: stationary at 5%)")
print("Best ARMA orders by AIC:"); print(grid.head(5).to_string(index=False))
pd.set_option("display.width", 250)
print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"Actual CFDID counts 2021-2023: {check_part.n_major.tolist()}   paper forecast: {cfg.PAPER['arma_forecast']}")
print("Pearson(count, yearly mean):", {k: round(v, 3) for k, v in pearson.items()},
      " paper: AP 0.08, CAA -0.09, DEL -0.1")
