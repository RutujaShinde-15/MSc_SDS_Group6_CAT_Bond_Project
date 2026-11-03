# Self-check of the copula toolbox: simulate from each nested family with
# known thetas, then (1) compare sample Kendall taus with theory and
# (2) refit by 3-D MLE and see that we recover the thetas.
import numpy as np, time
from scipy import stats, integrate
import cat_model as cm
rng = np.random.default_rng(1)
for fam, (t1, t0) in {"gumbel": (3.0, 1.8), "clayton": (3.0, 1.2), "frank": (9.0, 4.0)}.items():
    cop = cm.NestedCopula(fam, t1, t0, [0, 1], 2)
    t=time.time(); U = cop.sample(4000, rng); ts=time.time()-t
    tau01 = stats.kendalltau(U[:,0],U[:,1])[0]; tau02 = stats.kendalltau(U[:,0],U[:,2])[0]
    # density integrates to 1? Monte Carlo check: E_unif[c] = 1
    V = rng.random((200000,3)); m = np.exp(cop.logpdf(V)).mean()
    # empirical CDF vs model CDF
    ks = cm.ks_distance(U[:800], cop)
    t=time.time(); fit = cm.NestedCopula.fit(U[:800], fam, [0,1], 2); tf=time.time()-t
    print(f"{fam:8s} tau_in {tau01:.3f} (theory {cm.kendall_tau(fam,t1):.3f})  tau_out {tau02:.3f} (theory {cm.kendall_tau(fam,t0):.3f})"
          f"  E[c]={m:.3f}  KS={ks:.3f}  refit=({fit.theta_inner:.2f},{fit.theta_outer:.2f})  sample {ts:.2f}s fit {tf:.1f}s")
