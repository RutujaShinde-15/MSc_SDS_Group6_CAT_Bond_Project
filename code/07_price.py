# =============================================================================
# STEP 7  —  PRICE THE CAT BOND BY MONTE CARLO (paper Sec. 2.1-2.2, 3.5,
#            Eq. 4, 5, 6, 10, 11, 15, Tables 6 and 7, Figure 8)
# =============================================================================
# The paper's simulation (Sec. 3.5), step by step:
#   (1) N_k ~ Poisson(Lambda_k), k = 1..T (Lambda from the ARMA forecast);
#   (2) N_k floods per year drawn from the nested FRANK copula + Beta-GP
#       margins: x_i = F_i^-1(u_i);
#   (3) interest rate: CIR  dr = 0.2(0.05 - r)dt + 0.05 sqrt(r) dW,
#       r(0) = 2.962% (Eq. 15), simulated to give the discount factors;
#   (4) trigger level u_i = sample 90% quantile of each indicator (Table 6);
#       cashflows by Eq. (4), price by Eq. (5), average over simulations.
#   Bond: F = 100, coupon rate R = 3.5%.
#
# Eq. (4) in words: coupon in year s = alpha_s x (principal factors of the
#   earlier years) x C0. Principal factor of year k = beta_k / 2 if gamma_k < 1
#   (all three indicators triggered by some flood), else beta_k.
#   alpha, beta, gamma = yearly AVERAGE (Eq. 10 -> price P1) or MAXIMUM
#   (Eq. 11 -> price P2) of the flood scores s, s*, s** (Eq. 1-3).
#   A year without floods keeps everything (alpha = beta = gamma = 1).
#
# Runs (all with the same random numbers, so differences are not noise):
#   A. paper's Lambda (41.86, 41.56, 39.39), nested Frank   <- like-for-like
#   B. paper's Lambda, nested Gumbel (better joint tail, step 5)
#   C. paper's Lambda, independence copula
#   D. our ARMA(1,3) Lambda from CFDID, nested Frank
#   Table 7: one-year bond, all three indicators vs each pair.
#   Figure 8: (a) trigger quantile 0.80-0.90, (b) Lambda_1 = 20..40.
#   DIAGNOSTIC: the same with the halving rule reversed (halve when gamma = 1),
#   to test a possible coding slip behind the paper's Table 7 (see report).
# =============================================================================
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import cat_model as cm

warnings.filterwarnings("ignore")
P = cfg.PAPER
F, R = P["F"], P["R"]
events = pd.read_csv(cfg.CLEAN_DIR / "events.csv")
freq = cm.load_json(cfg.PARAMS_DIR / "frequency.json")
margs = cm.load_marginals(cfg)                       # Beta-GP margins (step 4)
cop_p = cm.load_json(cfg.PARAMS_DIR / f"copula_params_{cfg.THRESHOLD_SET}.json")


def nested(fam):
    s = cop_p["nested"][fam]
    return cm.NestedCopula(fam, s["theta_inner"], s["theta_outer"], cop_p["inner"], cop_p["outer"])


COPULAS = {"frank": nested("frank"), "gumbel": nested("gumbel"),
           "independence": cm.IndependenceCopula(3)}


# ---- (3) CIR interest rate: exact simulation + pathwise discount factors -------
def cir_discounts(n, T, rng, steps_per_year=52):
    """Returns D[path, a, b] = exp(-integral of r from year a to year b),
    a < b in 0..T. Exact CIR transition (non-central chi-square), integral by
    the trapezoid rule on a weekly grid."""
    k, th, eps, r0 = P["cir"]["kappa"], P["cir"]["theta"], P["cir"]["eps"], P["cir"]["r0"]
    dt = 1 / steps_per_year
    c = eps ** 2 * (1 - np.exp(-k * dt)) / (4 * k)
    df = 4 * k * th / eps ** 2
    r = np.full(n, r0)
    integral = np.zeros((n, T + 1))
    acc = np.zeros(n)
    for step in range(1, T * steps_per_year + 1):
        lam = r * np.exp(-k * dt) / c
        r_new = c * rng.noncentral_chisquare(df, lam)
        acc += 0.5 * (r + r_new) * dt
        r = r_new
        if step % steps_per_year == 0:
            integral[:, step // steps_per_year] = acc
    return integral       # D(a,b) = exp(-(integral[:, b] - integral[:, a]))


def cir_closed_form(T):
    """p(0,T) = A e^{-B r0} (paper Sec. 2.2), to check the simulation."""
    k, th, eps, r0 = P["cir"]["kappa"], P["cir"]["theta"], P["cir"]["eps"], P["cir"]["r0"]
    eta = np.sqrt(k ** 2 + 2 * eps ** 2)
    den = (k + eta) * (np.exp(eta * T) - 1) + 2 * eta
    A = (2 * eta * np.exp((k + eta) * T / 2) / den) ** (2 * k * th / eps ** 2)
    B = 2 * (np.exp(eta * T) - 1) / den
    return A * np.exp(-B * r0)


# ---- (1)+(2) floods and yearly retentions, in chunks to save memory ------------
def retentions(lambdas, copula, d, seed, columns=None, n=None):
    n = n or cfg.N_SIMS
    rng = np.random.default_rng(seed)
    parts, chunk = [], 5000
    for start in range(0, n, chunk):
        parts.append(cm.simulate_retentions(min(chunk, n - start), lambdas, copula, margs, d, rng, columns))
    return {key: np.vstack([p[key] for p in parts]) for key in parts[0]}


# ---- (4) Eq. (4)-(5) with pathwise CIR discounting ---------------------------------
def price(res, how, t, T, I, flip=False):
    a, b, g = res[f"alpha_{how}"], res[f"beta_{how}"], res[f"gamma_{how}"]
    halve = (g >= 1) if flip else (g < 1)
    pi = np.where(halve, b / 2, b)
    C0 = F * R
    carried = np.ones(len(a))
    pv = np.zeros(len(a))
    for s in range(t + 1, T + 1):
        disc = np.exp(-(I[:, s] - I[:, t]))
        pv += a[:, s - 1] * carried * C0 * disc
        carried = carried * pi[:, s - 1]
    pv += carried * F * np.exp(-(I[:, T] - I[:, t]))
    return pv.mean(), pv.std(ddof=1) / np.sqrt(len(pv))


def table6(res, I, flip=False):
    rows = []
    for T in range(1, cfg.T_MAX + 1):
        for t in range(T):
            p1, se1 = price(res, "avg", t, T, I, flip)
            p2, se2 = price(res, "max", t, T, I, flip)
            rows.append({"T": T, "t": t, "P1 (average, Eq.10)": p1, "se1": se1,
                         "P2 (maximum, Eq.11)": p2, "se2": se2})
    return pd.DataFrame(rows)


SEED = cfg.SEED
d90 = cm.trigger_levels(cfg, events, 0.90)
I = cir_discounts(cfg.N_SIMS, cfg.T_MAX, np.random.default_rng(SEED + 1))
paper_tab6 = {(1, 0): (80.392, 83.259), (2, 0): (70.881, 73.211), (2, 1): (66.255, 68.450),
              (3, 0): (56.006, 58.812), (3, 1): (54.865, 57.649), (3, 2): (52.321, 53.984)}

runs = {"A paper Lambda, nested Frank": (P["arma_forecast"], "frank"),
        "B paper Lambda, nested Gumbel": (P["arma_forecast"], "gumbel"),
        "C paper Lambda, independence": (P["arma_forecast"], "independence"),
        "D our ARMA(1,3) Lambda, nested Frank": (freq["ours_arma13"], "frank")}
all6, cache = [], {}
for name, (lam, fam) in runs.items():
    res = retentions(lam, COPULAS[fam], d90, SEED)
    cache[name] = res
    for flip in [False, True]:
        t6 = table6(res, I, flip)
        t6.insert(0, "run", name)
        t6.insert(1, "halving rule", "reversed (diagnostic)" if flip else "as written (gamma<1)")
        all6.append(t6)
tab6 = pd.concat(all6, ignore_index=True)
tab6["paper P1"] = [paper_tab6[(T, t)][0] for T, t in zip(tab6["T"], tab6["t"])]
tab6["paper P2"] = [paper_tab6[(T, t)][1] for T, t in zip(tab6["T"], tab6["t"])]
tab6.to_csv(cfg.TABLES_DIR / "step7_table6.csv", index=False)

# Trigger probabilities per flood and per year (explains the prices)
resA = cache["A paper Lambda, nested Frank"]
rng = np.random.default_rng(5)
trig = []
for fam, cop in COPULAS.items():
    U = cop.sample(400_000, rng)
    X = np.column_stack([margs[i].ppf(U[:, i]) for i in range(3)])
    above = X > d90
    k = above.sum(1)
    p1, p2, p3 = (k >= 1).mean(), (k >= 2).mean(), (k == 3).mean()
    L = P["arma_forecast"][0]
    trig.append({"copula": fam, "P(>=1 indicator) per flood": p1, "P(>=2) per flood": p2,
                 "P(all 3) per flood": p3,
                 "P(year has a 3-trigger flood), Lambda=41.86": 1 - np.exp(-L * p3),
                 "P(year has a 2+ trigger flood), Lambda=41.86": 1 - np.exp(-L * p2)})
trig = pd.DataFrame(trig)
trig.to_csv(cfg.TABLES_DIR / "step7_trigger_probabilities.csv", index=False)

# ---- Table 7: one-year bond, triple vs pairs --------------------------------------
t7 = []
lam1 = P["arma_forecast"][:1]
for label, cols in [("AP-CAA-DEL", None), ("AP-CAA", [0, 1]), ("AP-DEL", [0, 2]), ("CAA-DEL", [1, 2])]:
    res = retentions(lam1, COPULAS["frank"], d90, SEED, columns=cols)
    row = {"trigger": label}
    for flip in [False, True]:
        tag = " reversed" if flip else ""
        row["P1" + tag] = price(res, "avg", 0, 1, I, flip)[0]
        row["P2" + tag] = price(res, "max", 0, 1, I, flip)[0]
    t7.append(row)
t7 = pd.DataFrame(t7)
t7["paper P1"] = [80.3922, 48.7558, 48.7561, 48.7564]
t7["paper P2"] = [83.2592, 48.7869, 48.7871, 48.7873]
t7.to_csv(cfg.TABLES_DIR / "step7_table7.csv", index=False)

# ---- Figure 8 -----------------------------------------------------------------
n8 = 20_000
qs = np.round(np.arange(0.80, 0.901, 0.02), 2)
lams = [20, 25, 30, 35, 40]
fa, fb = [], []
for flip in [False, True]:
    fa.append([price(retentions(lam1, COPULAS["frank"], cm.trigger_levels(cfg, events, q), SEED, n=n8),
                     "avg", 0, 1, I[:n8], flip)[0] for q in qs])
    fb.append([price(retentions([L], COPULAS["frank"], d90, SEED, n=n8), "avg", 0, 1, I[:n8], flip)[0]
               for L in lams])
pd.DataFrame({"q": qs, "P1": fa[0], "P1 reversed": fa[1]}).to_csv(cfg.TABLES_DIR / "step7_fig8a.csv", index=False)
pd.DataFrame({"Lambda1": lams, "P1": fb[0], "P1 reversed": fb[1]}).to_csv(cfg.TABLES_DIR / "step7_fig8b.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(qs, fa[0], "k-o", label="Eq. (4) as written"); ax[0].plot(qs, fa[1], "r--s", label="halving reversed")
ax[0].set_xlabel("trigger level (sample quantile)"); ax[0].set_ylabel("one-year price P1"); ax[0].legend()
ax[0].set_title("(a) price vs trigger level")
ax[1].plot(lams, fb[0], "k-o", label="Eq. (4) as written"); ax[1].plot(lams, fb[1], "r--s", label="halving reversed")
ax[1].set_xlabel("intensity Lambda_1"); ax[1].set_title("(b) price vs intensity"); ax[1].legend()
fig.suptitle("Price sensitivity of a one-year CAT bond (paper Fig. 8)")
fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "step7_fig8.png", dpi=140); plt.close(fig)

# ---- Report ---------------------------------------------------------------------
pd.set_option("display.width", 250)
print("================ STEP 7 REPORT ================")
print("Trigger levels (90% quantiles):", dict(zip(cfg.INDICATORS, np.round(d90, 2))))
print("CIR check, p(0,T) closed form vs simulated:",
      [(T, round(cir_closed_form(T), 4), round(np.exp(-I[:, T]).mean(), 4)) for T in (1, 2, 3)])
print("\nTrigger probabilities:"); print(trig.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nTable 6:"); print(tab6.drop(columns=["se1", "se2"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("max SE:", round(tab6[["se1", "se2"]].max().max(), 3))
print("\nTable 7:"); print(t7.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nFig 8a:", dict(zip(qs, np.round(fa[0], 2))), " reversed:", dict(zip(qs, np.round(fa[1], 2))))
print("Fig 8b:", dict(zip(lams, np.round(fb[0], 2))), " reversed:", dict(zip(lams, np.round(fb[1], 2))))
