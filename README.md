# MSc_SDS_Group6_CAT_Bond_Project

# Pricing Multi-Event-Triggered Catastrophe Bonds

A from-scratch Python replication of the copula–POT pricing framework of **Tang, Wen, Ling & Zhang (2023)**, *Pricing multi-event-triggered catastrophe bonds based on a copula–POT model*, Risks 11(8): 151 ([doi:10.3390/risks11080151](https://doi.org/10.3390/risks11080151)), using major-rainstorm records for China, 2006–2020.

> Research Treatise I, Semester I, M.Sc. Statistics and Data Science
> School of Mathematics, Applied Statistics & Analytics, SVKM's NMIMS, Mumbai
> Project guide: Dr. Pradnya Khandeparkar

---

## Contents

1. [What this project does](#1-what-this-project-does)
2. [Repository structure](#2-repository-structure)
3. [Setup](#3-setup)
4. [Data](#4-data)
5. [Running the pipeline](#5-running-the-pipeline)
6. [Model settings](#6-model-settings)
7. [Expected results (checks for replication)](#7-expected-results-checks-for-replication)
8. [Implementation notes](#8-implementation-notes)
9. [Limitations](#9-limitations)
10. [Citation](#10-citation)
11. [Authors](#11-authors)

---

## 1. What this project does

A catastrophe (CAT) bond moves extreme disaster risk from an insurer or government to capital-market investors. In a **multi-event trigger**, a payout needs the disaster to be severe on several indicators at once. This lowers moral hazard and the chance of an accidental trigger. We use three indicators:

| Indicator | Description | Unit |
|---|---|---|
| **AP** | Affected population | 10,000 persons |
| **CAA** | Crop affected area | 10,000 hectares |
| **DEL** | Direct economic loss (CPI-adjusted to 2020 prices) | CNY 100 million |

The bond is priced in five stages:

| Stage | Model | Report section |
|---|---|---|
| 1. Severity of each indicator | Beta body + generalized Pareto (GP) tail (peaks over threshold) | 4.3, 5.1–5.2 |
| 2. Dependence between indicators | Nested Archimedean copula (Gumbel / Clayton / Frank) | 4.4, 5.3 |
| 3. Annual number of rainstorms | ARMA(1,3) on annual counts → Poisson intensities | 4.5, 5.4 |
| 4. Discounting | Cox–Ingersoll–Ross (CIR) short rate | 4.2 |
| 5. Price | Monte Carlo simulation of coupons and principal | 4.1, 4.5, 5.5 |

How the trigger works:
- If **one** indicator goes above its trigger, only the coupon is reduced.
- If **two** go above their triggers together, part of the principal is lost.
- If **all three** do, the principal is halved.

---

## 2. Repository structure

```
.
├── README.md
├── requirements.txt
├── config.py                  # All parameters and the random seed (Section 6)
├── data/
│   ├── raw/                   # Source files as collected
│   ├── rainstorm_events.csv   # 328 event records: AP, CAA, DEL (2006–2020)
│   └── annual_counts.csv      # Annual number of major rainstorms (1994–2020)
├── src/
│   ├── 01_descriptive.py      # Table 6, Figure 1
│   ├── 02_pot_marginals.py    # Thresholds, Beta–GP fits: Figures 2–4, Table 7
│   ├── 03_copula.py           # Spearman, nested copulas, bootstrap: Tables 8–9
│   ├── 04_frequency_arma.py   # ADF, ACF/PACF, ARMA(1,3), forecasts: Figure 5
│   ├── 05_pricing.py          # CIR + Monte Carlo pricing: Table 10, sensitivity
│   └── utils/
│       ├── beta_gp.py         # Piecewise Beta–GP CDF, inverse CDF, MLE
│       ├── nested_copula.py   # Nested Archimedean density, fit, Hofert sampler
│       ├── cir.py             # CIR path simulation and closed-form p(t,T)
│       └── triggers.py        # a_ij, s_j, s*_j, s**_j, alpha/beta/gamma, pi_k
├── outputs/
│   ├── figures/
│   └── tables/
└── report/
    └── Group_6_Pricing_Multi_Event_Trigger_Cat_Bonds_Report.pdf
```

Each script reads from `data/` and from the outputs of earlier scripts, and writes its tables and figures to `outputs/`. **Run the scripts in numbered order.**

---

## 3. Setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
statsmodels>=0.14
matplotlib>=3.7
```

No dedicated copula or EVT package is needed. The nested copulas, the Beta–GP marginals and the pricing simulation are written from scratch in `src/utils/`.

---

## 4. Data

### 4.1 `data/rainstorm_events.csv` (severity, n = 328)

There is one row for each major rainstorm in one province, 2006–2020. Only records that report all three indicators are kept.

| Column | Type | Description |
|---|---|---|
| `year` | int | Year of the event (2006–2020) |
| `province` | str | Province affected |
| `AP` | float | Affected population, 10,000 persons |
| `CAA` | float | Crop affected area, 10,000 hectares |
| `DEL` | float | Direct economic loss, CNY 100 million, **2020 prices** |

**Source:** compiled by the team from the China Flood Disaster Information Database (CFDID). DEL was converted to 2020 prices with China's consumer price index:
`DEL_2020 = DEL_nominal × CPI_2020 / CPI_year`.

### 4.2 `data/annual_counts.csv` (frequency, n = 27)

| Column | Type | Description |
|---|---|---|
| `year` | int | 1994–2020 |
| `n_rainstorms` | int | Number of major rainstorms in China that year |

### 4.3 Data notes

- Some source values are rounded (for example, CAA = 4.4). This affects the Chi-square test for the Beta body (Section 9).
- Our sample was compiled independently, so its coverage may differ from the licensed data used by Tang et al. (2023). Expect results close to the paper's, but not the same.

---

## 5. Running the pipeline

Run everything in order from the repository root:

```bash
python src/01_descriptive.py
python src/02_pot_marginals.py
python src/03_copula.py
python src/04_frequency_arma.py
python src/05_pricing.py
```

| Script | What it does | Main outputs |
|---|---|---|
| `01_descriptive.py` | Max, min, mean, median, skewness and kurtosis (normal = 3); exponential Q–Q plots | Table 6, Figure 1 |
| `02_pot_marginals.py` | Mean residual life and parameter-stability plots over a grid of thresholds; GP MLE above `u`; Beta MLE below `u` on `(x − m)/(u − m)`; Chi-square and KS tests; P–P and Q–Q plots with 95% simulation envelopes. Saves the fitted marginals for later scripts. | Figures 2–4, Table 7 |
| `03_copula.py` | Spearman ρ for each pair; pseudo-observations `F_i(x_ij)`; joint MLE of (θ₁, θ₂) with θ₁ ≥ θ₂ for nested Gumbel, Clayton and Frank (inner = AP, CAA; outer = DEL); KS distance to the empirical copula; parametric-bootstrap p-value (B = 200) | Tables 8–9 |
| `04_frequency_arma.py` | ADF test; ACF/PACF; AIC grid over p ≤ 2, q ≤ 3; ARMA(1,3) MLE; Jarque–Bera, Ljung–Box (lag 5) and likelihood-ratio test against a constant mean; 3-year forecasts Λ₁, Λ₂, Λ₃ | Figure 5, ARMA estimates, intensities |
| `05_pricing.py` | Simulates N_j ~ Poisson(Λ_j), copula draws → inverse marginals, CIR discount factors, cash flows from Eq. (7); prices for T = 1, 2, 3 and every purchase year t; sensitivity to trigger quantile and to Λ₁ | Table 10, sensitivity results |

**Runtime:** the copula bootstrap (step 3) and the pricing simulation (step 5) take the longest. To check the pipeline quickly, set `N_BOOT` and `N_SIM` lower in `config.py`. Keep in mind that this changes the p-values and prices slightly.

---

## 6. Model settings

All settings are in `config.py`. These are the values used in the report:

```python
SEED = <seed>                 # fixed seed; every result in the report is reproducible

# --- POT thresholds (selected from MRL and parameter-stability plots) ---
THRESHOLDS = {"AP": 196.9, "CAA": 16.4, "DEL": 27.3}

# --- Copula ---
INNER_PAIR   = ("AP", "CAA")  # strongest Spearman pair
OUTER_VAR    = "DEL"
PRICING_COPULA = "frank"      # selected by KS distance
N_BOOT       = 200            # parametric bootstrap replications

# --- Frequency ---
ARMA_ORDER   = (1, 0, 3)      # ARMA(1,3)
FORECAST_H   = 3

# --- Bond ---
FACE_VALUE   = 100            # F
COUPON_RATE  = 0.035          # R
TRIGGER_Q    = 0.90           # trigger u_i = sample 90% quantile of each indicator
MATURITIES   = [1, 2, 3]

# --- CIR (parameters from Tang et al. 2023, 2021 7-day SHIBOR) ---
CIR = {"kappa": 0.2, "theta": 0.05, "epsilon": 0.05, "r0": 0.02962}

# --- Monte Carlo ---
N_SIM        = 50_000         # M
```

**Two kinds of level are used. Keep them apart:**
- The **POT thresholds** `u` (196.9, 16.4, 27.3) separate the Beta body from the GP tail in the marginal model.
- The **trigger levels** of the bond are the sample 90% quantiles. They decide when the bond pays out.

---

## 7. Expected results (checks for replication)

If your setup is correct, you should get these numbers to the decimals shown. Monte Carlo prices can differ by about ±0.13 (one standard error) if a different seed is used.

**POT thresholds and GP fits (Table 7a)**

| Indicator | u | n_u | σ | ξ | χ² p | KS p |
|---|---|---|---|---|---|---|
| AP | 196.9 | 95 | 179.711 | 0.121 | 0.951 | 0.978 |
| CAA | 16.4 | 74 | 10.627 | 0.481 | 0.367 | 0.913 |
| DEL | 27.3 | 78 | 37.625 | 0.455 | 0.365 | 0.982 |

**Beta body (Table 7b):** AP (a, b) = (0.884, 1.634); CAA = (0.810, 1.586); DEL = (0.605, 1.380)

**Spearman ρ (Table 8):** AP–CAA 0.835, AP–DEL 0.668, CAA–DEL 0.691

**Nested copulas (Table 9)**

| Family | θ₁ (inner) | θ₂ (outer) | KS | p-value |
|---|---|---|---|---|
| Gumbel | 2.898 | 1.868 | 0.076 | 0.100 |
| Clayton | 2.489 | 0.942 | 0.120 | < 0.005 |
| **Frank** (selected) | 10.190 | 5.663 | 0.064 | 0.215 |

**ARMA(1,3):** φ₁ = 0.559, ϑ = (0.261, 0.026, 0.759), μ = 20.89, σ²ₑ = 73.27, AIC = 208.9; ADF p = 0.004; Jarque–Bera p = 0.544; Ljung–Box (lag 5) p = 0.998; LR χ² = 29.1 on 4 df.
**Forecast intensities:** (Λ₁, Λ₂, Λ₃) = (15.05, 16.24, 15.49)

**CIR check:** one-year discount factor p(0,1) = 0.969

**Bond prices (Table 10)** (F = 100, R = 3.5%, triggers at the 90% quantiles, nested Frank)

| | T=1, t=0 | T=2, t=0 | T=2, t=1 | T=3, t=0 | T=3, t=1 | T=3, t=2 |
|---|---|---|---|---|---|---|
| P_t | 84.095 | 70.197 | 82.819 | 59.286 | 69.354 | 83.156 |

**Sensitivity (one-year bond):**
- Trigger at the 80% quantile → 62.07; at the 90% quantile → 84.22
- Λ₁ = 10 → 88.36; Λ₁ = 40 → 68.79

---

## 8. Implementation notes

These are the points where a replication most often goes wrong.

1. **Beta–GP marginal (Eq. 15).** Below `u`, use `F(x) = (1 − n_u/n) · BetaCDF((x − m)/(u − m); a, b)`. Above `u`, use `F(x) = 1 − (n_u/n) · [1 − G_ξ,σ(x − u)]`. Weighting by the observed proportions makes `F` continuous at `u`. The inverse CDF used in simulation is defined piece by piece in the same way.
2. **Pseudo-observations.** The copula is fitted to `F_i(x_ij)` from the fitted Beta–GP marginals, not to empirical ranks. Clip values to (1e-10, 1 − 1e-10) before evaluating copula densities.
3. **Nesting constraint.** For a nested copula with the same family at both levels to be valid, θ₁ ≥ θ₂. Enforce this during optimisation, for example by fitting θ₁ = θ₂ + δ with δ ≥ 0.
4. **Copula sampling.** Nested draws use the exact algorithm of Hofert (2011): sample the outer frailty, then the inner frailty conditional on it. Do not sample the inner and outer copulas independently.
5. **Copula GOF p-value.** The parameters are estimated from the data, so the standard KS critical values do not apply. Use the parametric bootstrap: simulate from the fitted copula, refit, recompute KS, B = 200.
6. **Trigger arithmetic (Eqs. 1–6).**
   - `a_ij = max(x_ij − u_i, 0) / x_ij`
   - `s_j = Π(1 − a_ij)`; `s*_j` is the same product over pairs, and `s**_j` over the triple.
   - α_t, β_t and γ_t are yearly averages of these. If a year has no events, all three are 1.
   - `π_k = β_k / 2` if γ_k < 1, otherwise `β_k`.
7. **Cash flows (Eq. 7).**
   - The first coupon after purchase is `α_{t+1} · C₀`.
   - Each later coupon is `α_s · Π_{k=t+1}^{s−1} π_k · C₀`.
   - The principal is `Π_{k=t+1}^{T} π_k · F`.
   - Lost principal is never restored.
8. **Purchase year.** A bond bought at the end of year t is exposed only to disasters in years t+1, …, T. It uses the intensities Λ_{t+1}, …, Λ_T.
9. **Discounting.** Discount factors come from simulated CIR paths, and are checked against the closed form p(t,T) = A(t,T)·exp(−B(t,T)·r(t)). Under the given parameters, 2κθ ≥ ε² holds, so the rate stays positive.
10. **Likelihood-ratio test.** The paper uses an F-test for overall ARMA significance. We use an LR test, `2(ℓ₁ − ℓ₀) ~ χ²₄`.

---

## 9. Limitations

- **Data coverage:** the data were compiled independently and may differ from the licensed data used by Tang et al. (2023). Some values are rounded at the source.
- **Copula choice:** nested Frank is selected by KS, but nested Gumbel is not rejected (p = 0.100). Frank has no upper-tail dependence, so it may understate how often all three indicators are extreme together. Re-pricing with `PRICING_COPULA = "gumbel"` is a natural robustness check.
- **Beta body:** the Chi-square test rejects the Beta fit below the threshold for all three indicators. This affects only events well below the 90% trigger levels.
- **Interest rate:** the CIR parameters are taken from Tang et al. (2023), not estimated from our own data.
- **Independence assumptions:** interest rates, disaster severity and disaster frequency are assumed to be mutually independent.

**Next step:** apply the framework to multi-event-triggered flood CAT bonds for **India**, with CIR estimated from MIBOR.

---

## 10. Citation

If you use this code, please cite the original paper:

```bibtex
@article{tang2023pricing,
  title   = {Pricing Multi-Event-Triggered Catastrophe Bonds Based on a Copula--POT Model},
  author  = {Tang, Yifan and Wen, Conghua and Ling, Chengxiu and Zhang, Yuqing},
  journal = {Risks},
  volume  = {11},
  number  = {8},
  pages   = {151},
  year    = {2023},
  doi     = {10.3390/risks11080151}
}
```

Other references used:
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A theory of the term structure of interest rates. *Econometrica*, 53, 385–407.
- Hofert, M. (2011). Efficiently sampling nested Archimedean copulas. *Computational Statistics & Data Analysis*, 55, 57–70.

---

## 11. Authors

**Group 6**, M.Sc. Statistics and Data Science, SVKM's NMIMS (2026–27)

- Parayani Chaudhari
- Anushka Kumar
- Ekta Pandit
- Rutuja Shinde

Project guide: **Dr. Pradnya Khandeparkar**

## License

<Add a license, e.g. MIT for the code. Check that redistribution of the CFDID-derived data is permitted before publishing `data/`.>
