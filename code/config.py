# =============================================================================
# config.py  —  ALL SETTINGS FOR THE CHINA REPLICATION, IN ONE PLACE
# =============================================================================
# Purpose of this project:
#   Our guide asked us to REBUILD Tang, Wen, Ling & Zhang (2023), "Pricing
#   Multi-Event-Triggered Catastrophe Bonds Based on a Copula-POT Model",
#   Risks 11: 151, from scratch on China data, to check that we understand
#   every step and can get the same (or similar) numbers before applying the
#   method to India.
#
#   Data: compiled by 00_compile.py from the raw China Flood Disaster Impact
#   Database (CFDID, Cui et al. 2026): province-level "major" rainstorm
#   floods (DEL >= CNY 100m), 2006-2020, reporting all three indicators.
#   (Version 1 used 279 parent-level events; see 00_compile.py for why we
#   changed.) The paper used 245
#   "major rainstorms" from the licensed China Statistical Yearbook of Natural
#   Disasters, so the two samples are NOT the same events. "Similar results"
#   is the realistic target, not identical numbers.
#
#   This is the same code structure as our India pipeline (same toolbox,
#   cat_model.py), with one difference: China has ALL THREE indicators, so the
#   nested 3-D copula (paper Eq. 9, 14) and the gamma_t "halving" term
#   (Eq. 3, 4) are now the main case, not an extension.
# =============================================================================
from pathlib import Path

# ---- Folders ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
PARAMS_DIR = ROOT / "outputs" / "params"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
for _d in (CLEAN_DIR, PARAMS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw input: the released CFDID (Cui et al. 2026), compiled by 00_compile.py

# ---- The three trigger indicators (paper Sec. 3, Table 3) --------------------
# Same order as the paper: X1 = AP, X2 = CAA, X3 = DEL.
INDICATORS = ["AP", "CAA", "DEL"]
INDICATOR_NAMES = {
    "AP": "Affected population (AP)",
    "CAA": "Crop affected area (CAA)",
    "DEL": "Direct economic loss (DEL)",
}
# Units: the paper's Table 3 labels say "millions / million ha / billion yuan",
# but its numbers (AP max 1510, threshold 160, etc.) only make sense in
# 10,000 persons / 10,000 ha / 100 million CNY, the units the Chinese
# yearbooks use. Our compilation guide reached the same conclusion, so we
# work in those units and the paper's thresholds (160, 12, 15) can be
# compared directly.
UNITS = {"AP": "10k persons", "CAA": "10k ha", "DEL": "100m CNY (2020 prices)"}

# ---- Inflation: put DEL on 2020 prices, like the paper ----------------------
# Paper Sec. 3: "direct economic loss (DEL) adjusted with the Consumer Price
# Index (CPI) in 2020". CFDID gives DEL in 2023 constant prices. The ratio
# (2023-price DEL) / (nominal DEL) for every 2020 event in the file is
# 1.031238, which equals China's CPI growth 2021-2023 (1.009 x 1.020 x 1.002).
# So:  DEL(2020 prices) = DEL(2023 prices) / 1.031238.
CPI_2023_OVER_2020 = 1.031238

# ---- The paper's own numbers, for side-by-side comparison --------------------
PAPER = {
    "n": 245,
    "table3": {  # Max, Min, Mean, Median, Skewness, Kurtosis
        "AP":  [1510.000, 1.600, 187.880, 116.750, 2.771, 12.678],
        "CAA": [135.200, 0.005, 12.492, 7.000, 3.689, 20.458],
        "DEL": [420.816, 1.011, 21.411, 8.083, 5.154, 37.823],
    },
    "u": {"AP": 160.0, "CAA": 12.0, "DEL": 15.0},   # Sec. 3.2 thresholds
    "table4": {  # sigma, xi, Beta a, Beta b
        "AP":  {"sigma": 173.369, "xi": 0.197, "a": 1.016, "b": 1.345},
        "CAA": {"sigma": 11.771, "xi": 0.341, "a": 1.076, "b": 1.687},
        "DEL": {"sigma": 23.538, "xi": 0.492, "a": 0.582, "b": 1.137},
    },
    "spearman": {"AP-CAA": 0.771, "AP-DEL": 0.554, "CAA-DEL": 0.515},
    "table5": {  # theta1 (inner), theta2 (outer), KS, p
        "gumbel":  [44.68, 22.80, 0.06, 0.12],
        "clayton": [9.58, 4.35, 0.12, 0.35],
        "frank":   [176.34, 87.60, 0.04, 0.61],
    },
    # Sec. 3.4 / 3.5 (frequency and pricing) - used once frequency data arrive
    "arma_forecast": [41.86, 41.56, 39.39],
    "cir": {"kappa": 0.2, "theta": 0.05, "eps": 0.05, "r0": 0.02962},
    "F": 100.0, "R": 0.035,
}

# ---- Threshold search (paper Sec. 2.3, 3.2, Figs. 3-4) ----------------------
# We scan candidate thresholds from the 30% to the 95% quantile of each
# indicator and draw the mean residual life and parameter stability plots.
CANDIDATE_Q_MIN, CANDIDATE_Q_MAX, CANDIDATE_Q_STEP = 0.30, 0.95, 0.0125
MIN_EXCEED = 15            # never fit a GPD to fewer than 15 floods above u
STABILITY_MIN_NU = 50      # stability rule only compares with fits on >= 50 floods

# The thresholds WE choose after reading our own plots (filled in after step 3;
# step 3 prints a data-driven suggestion). Values in the UNITS above.
OUR_U = None               # None -> step 3's suggestion is used

# Which threshold set is used downstream ("ours" or "paper").
THRESHOLD_SET = "ours"

# ---- Copula (paper Sec. 3.3) ------------------------------------------------
# "fitted": u_ij = F_i(x_ij) from our Beta-GPD margins (what the paper does)
COPULA_UNIFORMS = "fitted"
GOF_BOOTSTRAP = 200        # simulated datasets for the copula KS p-value

# ---- Pricing (paper Sec. 3.5) -----------------------------------------------
TRIGGER_Q = 0.90           # Table 6: trigger level = sample 90% quantile
N_SIMS = 50_000            # paper: m = 10^4; we use 5x10^4 for smaller noise
T_MAX = 3
SEED = 2023
