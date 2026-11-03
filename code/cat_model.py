# =============================================================================
# cat_model.py  —  ALL THE PAPER'S FORMULAS, WRITTEN ONCE
# =============================================================================
#
# What this file is:
#   A "toolbox" file. It does not do anything when run on its own. The step
#   scripts (04_..., 05_..., 07_..., etc.) import it and use its functions.
#   Keeping every formula in ONE place means:
#     - each formula is written (and checked) only once;
#     - when the examiner asks "where is Eq. (4) in your code?", the answer is
#       always in this file, under a heading that names the equation.
#
# Contents (paper equation numbers in brackets):
#   PART A  The Beta-GPD marginal distribution of one indicator  (Eq. 7, 12, 13)
#   PART B  Archimedean copulas: Gumbel, Clayton, Frank          (Sec. 2.4, Table 2)
#   PART C  Nested Archimedean copula for 3 indicators           (Eq. 9, 14)
#   PART D  Coupon and principal retention                       (Eq. 1, 2, 3, 10, 11)
#   PART E  Cashflows and the bond price                         (Eq. 4, 5)
#   PART F  The Monte Carlo simulation                           (Sec. 3.5, Steps 1-4)
#   PART G  Small helpers to load saved results from earlier steps
# =============================================================================

import json
import itertools
import numpy as np
from scipy import stats, optimize, integrate


# =============================================================================
# PART A.  THE BETA-GPD MARGINAL DISTRIBUTION      (Paper Eq. 7, 12, 13)
# =============================================================================
# One indicator (say AP) is described by two pieces glued together at the
# threshold u chosen in step 3:
#   - ORDINARY floods (x <= u): rescaled to y = (x - m) / (u - m), which lies
#     between 0 and 1, and modelled with a Beta(a, b) distribution  (Eq. 12).
#     m = the smallest flood in the data (paper: "sample minima").
#   - EXTREME floods (x > u): the excess (x - u) follows a GPD with shape xi
#     and scale sigma  (Eq. 12, Sec. 2.3).
#   - zeta = n_u / n = share of floods above u (paper Eq. 7, "F(u) estimated
#     as n_u / n").
#
# The full distribution function (CDF) F(x) = P(flood value <= x):
#     x <= u :  F(x) = (1 - zeta) * BetaCDF( (x - m) / (u - m) )
#     x >  u :  F(x) = 1 - zeta * (1 - GPD_CDF(x - u))
#
# NOTE FOR THE WRITE-UP: paper Eq. (13) prints the body as plain BetaCDF and
# the tail as "1 - (n_u/n) G(x - u)". Taken literally, F would not join up at
# x = u and would go DOWN in the tail. The version above is the standard,
# correct form (it equals 1 - zeta exactly at u from both sides, and rises to
# 1). We state this as a correction of a typo, not a change of method.
# =============================================================================
class BetaGPDMarginal:
    """One indicator's fitted distribution (body = Beta, tail = GPD)."""

    EPS = 1e-6   # tiny number used to keep values strictly inside (0, 1)

    def __init__(self, m, u, zeta, a, b, xi, sigma):
        self.m, self.u, self.zeta = m, u, zeta      # minimum, threshold, n_u/n
        self.a, self.b = a, b                       # Beta parameters (body)
        self.xi, self.sigma = xi, sigma             # GPD shape and scale (tail)

    # ---- fitting by maximum likelihood (MLE), as in paper Table 4 ----------
    @classmethod
    def fit(cls, x, u):
        x = np.asarray(x, dtype=float)
        m = x.min()
        zeta = np.mean(x > u)                       # n_u / n
        # Body: rescale to (0,1). The smallest flood maps to exactly 0 and a
        # flood equal to u maps to exactly 1; the Beta likelihood is undefined
        # at those exact edges, so we nudge them in by EPS.
        y = np.clip((x[x <= u] - m) / (u - m), cls.EPS, 1 - cls.EPS)
        a, b, _, _ = stats.beta.fit(y, floc=0, fscale=1)
        # Tail: excesses over u; loc fixed at 0 because excesses start at 0.
        xi, _, sigma = stats.genpareto.fit(x[x > u] - u, floc=0)
        return cls(m, u, zeta, a, b, xi, sigma)

    # ---- F(x): probability that a flood is at most x  (Eq. 13) -------------
    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        y = np.clip((x - self.m) / (self.u - self.m), 0, 1)
        body = (1 - self.zeta) * stats.beta.cdf(y, self.a, self.b)
        tail = 1 - self.zeta * stats.genpareto.sf(np.maximum(x - self.u, 0),
                                                  self.xi, 0, self.sigma)
        return np.where(x <= self.u, body, tail)

    # ---- F^-1(p): turn a probability back into a flood size  (Sec. 3.5,
    #      Step 2: "x_ik = F_i^-1(u_ik)") ------------------------------------
    def ppf(self, p):
        p = np.clip(np.asarray(p, dtype=float), self.EPS, 1 - self.EPS)
        cut = 1 - self.zeta                          # F(u)
        body = self.m + (self.u - self.m) * stats.beta.ppf(
            np.clip(p / cut, 0, 1), self.a, self.b)
        tail = self.u + stats.genpareto.ppf(
            np.clip((p - cut) / self.zeta, 0, 1 - self.EPS), self.xi, 0, self.sigma)
        return np.where(p <= cut, body, tail)

    def to_dict(self):
        return dict(m=self.m, u=self.u, zeta=self.zeta, a=self.a, b=self.b,
                    xi=self.xi, sigma=self.sigma)

    @classmethod
    def from_dict(cls, d):
        return cls(d["m"], d["u"], d["zeta"], d["a"], d["b"], d["xi"], d["sigma"])


# =============================================================================
# PART B.  ARCHIMEDEAN COPULAS: GUMBEL, CLAYTON, FRANK   (Sec. 2.4, Table 2)
# =============================================================================
# A copula is the "glue" between indicators. It works on the 0-1 scale
# (probabilities), so it describes only HOW the indicators move together,
# not how big they are.
#
# Every Archimedean copula is built from a "generator" phi:
#     C(u1, ..., um) = psi( phi(u1) + ... + phi(um) ),  psi = inverse of phi
# (paper Sec. 2.4: "C(u1..um) = phi^-1( sum phi(ui) )").
#
#   Family   phi(t)                                   theta range  Tail behaviour
#   Gumbel   (-ln t)^theta                            >= 1         big floods big together (upper tail)
#   Clayton  t^(-theta) - 1                           > 0          small floods small together (lower tail)
#   Frank    -ln[(e^(-theta t)-1)/(e^(-theta)-1)]     != 0         symmetric, no tail dependence
# (Clayton's generator in Table 2 has an extra 1/theta factor; a constant
#  factor does not change the copula.)
# A larger theta means stronger dependence (paper, citing Charpentier & Segers).
# =============================================================================
FAMILIES = ["gumbel", "clayton", "frank"]


def phi(t, family, theta):
    """Generator phi(t) of each family (Table 2)."""
    t = np.clip(t, 1e-12, 1 - 1e-12)
    if family == "gumbel":
        return (-np.log(t)) ** theta
    if family == "clayton":
        return t ** (-theta) - 1
    if family == "frank":
        return -np.log(np.expm1(-theta * t) / np.expm1(-theta))
    raise ValueError(family)


def psi(s, family, theta):
    """Inverse generator psi = phi^-1."""
    s = np.maximum(s, 0)
    if family == "gumbel":
        return np.exp(-s ** (1 / theta))
    if family == "clayton":
        return (1 + s) ** (-1 / theta)
    if family == "frank":
        return -np.log1p(np.exp(-s) * np.expm1(-theta)) / theta
    raise ValueError(family)


def kendall_tau(family, theta):
    """Kendall's tau implied by theta (a 0-1 measure of dependence strength)."""
    if family == "gumbel":
        return 1 - 1 / theta
    if family == "clayton":
        return theta / (theta + 2)
    if family == "frank":   # uses the Debye function D1
        d1 = integrate.quad(lambda t: t / np.expm1(t), 0, theta)[0] / theta
        return 1 - 4 / theta * (1 - d1)
    raise ValueError(family)


def theta_from_tau(family, tau):
    """Inverse of kendall_tau: the theta that gives a target tau (tau > 0)."""
    tau = float(np.clip(tau, 1e-4, 0.95))
    if family == "gumbel":
        return 1 / (1 - tau)
    if family == "clayton":
        return 2 * tau / (1 - tau)
    return optimize.brentq(lambda th: kendall_tau("frank", th) - tau, 1e-4, 200)


THETA_BOUNDS = {"gumbel": (1.0, 60.0), "clayton": (1e-4, 60.0), "frank": (1e-4, 200.0)}


class BivariateCopula:
    """A 2-indicator Archimedean copula (used by Designs A and B)."""
    dim = 2

    def __init__(self, family, theta):
        self.family, self.theta = family, float(theta)

    # C(u, v): probability that BOTH indicators are below their u and v levels
    def cdf(self, U):
        U = np.atleast_2d(U)
        return psi(phi(U[:, 0], self.family, self.theta) +
                   phi(U[:, 1], self.family, self.theta), self.family, self.theta)

    # log of the copula density c(u, v): needed for maximum likelihood
    def logpdf(self, U):
        u = np.clip(U[:, 0], 1e-10, 1 - 1e-10)
        v = np.clip(U[:, 1], 1e-10, 1 - 1e-10)
        th = self.theta
        if self.family == "gumbel":
            x, y = -np.log(u), -np.log(v)
            S = x ** th + y ** th
            A = S ** (1 / th)
            # c = C/(uv) * (xy)^(theta-1) * S^(1/theta - 2) * (A + theta - 1)
            return (-A - np.log(u * v) + (th - 1) * np.log(x * y)
                    + (1 / th - 2) * np.log(S) + np.log(A + th - 1))
        if self.family == "clayton":
            return (np.log1p(th) - (th + 1) * np.log(u * v)
                    - (2 + 1 / th) * np.log(u ** -th + v ** -th - 1))
        if self.family == "frank":
            # c = theta (1-e^-theta) a b / ((1-e^-theta) a b - (a-1)(b-1))^2
            # with a = e^(theta u), b = e^(theta v). Written this way so the
            # computer never divides by a number that rounds to zero.
            e1 = -np.expm1(-th)                       # 1 - e^(-theta)
            la, lb = th * u, th * v                   # log a, log b
            D = e1 * np.exp(la + lb) - np.expm1(la) * np.expm1(lb)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.log(th * e1) + la + lb - 2 * np.log(D)
            return np.where(D > 0, out, -np.inf)
        raise ValueError(self.family)

    # Maximum likelihood fit: try thetas, keep the one with the highest
    # total log-density over our floods (paper Table 5: "maximum likelihood
    # estimates").
    @classmethod
    def fit(cls, U, family):
        lo, hi = THETA_BOUNDS[family]
        def nll(th):                       # "negative log-likelihood"
            val = -np.sum(cls(family, th).logpdf(U))
            return val if np.isfinite(val) else 1e12   # guard against overflow
        res = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded")
        cop = cls(family, res.x)
        cop.loglik = -res.fun
        return cop

    # Random draws from the copula (used in Monte Carlo, Sec. 3.5 Step 2)
    def sample(self, n, rng):
        th = self.theta
        if self.family == "frank":
            # "Conditional inversion": draw u, then draw v given u exactly.
            u, w = rng.random(n), rng.random(n)
            v = -np.log1p(w * np.expm1(-th) /
                          (w + (1 - w) * np.exp(-th * u))) / th
            return np.column_stack([u, v])
        # Gumbel and Clayton: "frailty" method (Marshall & Olkin 1988).
        # Draw one shared random shock V per flood; both indicators depend on
        # it, which is what makes them move together.
        V = frailty(self.family, th, n, rng)
        E = rng.exponential(size=(n, 2))
        return psi(E / V[:, None], self.family, th)

    def tau(self):
        return kendall_tau(self.family, self.theta)


class IndependenceCopula:
    """No dependence at all: C(u1..um) = u1 * ... * um. Used as the
    comparison case for the headline result (step 9)."""
    family, theta = "independence", None

    def __init__(self, dim):
        self.dim = dim

    def cdf(self, U):
        return np.prod(np.atleast_2d(U), axis=1)

    def sample(self, n, rng):
        return rng.random((n, self.dim))


# ---- frailty (shared shock) generators --------------------------------------
def positive_stable(alpha, n, rng):
    """Positive alpha-stable numbers with Laplace transform exp(-t^alpha),
    0 < alpha <= 1 (Kanter 1975 / Chambers-Mallows-Stuck method)."""
    if alpha >= 1:
        return np.ones(n)
    T = rng.uniform(0, np.pi, n)
    W = rng.exponential(size=n)
    return (np.sin(alpha * T) / np.sin(T) ** (1 / alpha)) * \
           (np.sin((1 - alpha) * T) / W) ** ((1 - alpha) / alpha)


def frailty(family, theta, n, rng):
    if family == "gumbel":
        return positive_stable(1 / theta, n, rng)
    if family == "clayton":
        return rng.gamma(1 / theta, 1.0, n)
    if family == "frank":
        # Frank's shared shock is a whole number from the "logarithmic
        # series" distribution with p = 1 - e^(-theta) (Marshall-Olkin 1988).
        return rng.logseries(-np.expm1(-theta), n).astype(float)
    raise ValueError(family)


_FRANK_TABLE_CACHE = {}


def frank_inner_frailty(t0, t1, V0, rng):
    """Inner shock for NESTED FRANK (Hofert 2011, Sec. 4).
    Given the outer shock V0 (a whole number), the inner shock is
        V01 = X_1 + ... + X_V0,   X_i independent, whole numbers >= 1,
        P(X = k) = binom(alpha, k) (-1)^(k-1) (1 - e^-t1)^k / (1 - e^-t0),
    with alpha = t0/t1. (Check: these probabilities add up to 1, because
    1 - (1 - (1 - e^-t1))^alpha = 1 - e^-t0.) We tabulate P(X <= k) once and
    draw X by looking up a uniform random number in that table."""
    key = (round(t0, 10), round(t1, 10))
    if key not in _FRANK_TABLE_CACHE:
        alpha, c1, c0 = t0 / t1, -np.expm1(-t1), -np.expm1(-t0)
        probs = []
        term = alpha * c1                       # k = 1
        total = 0.0
        k = 1
        while total < 1 - 1e-12 and k < 20_000_000:
            block = []
            for _ in range(100_000):
                block.append(term)
                term *= (k - alpha) / (k + 1) * c1
                k += 1
            block = np.array(block) / c0
            probs.append(block)
            total += block.sum()
        cdf = np.cumsum(np.concatenate(probs))
        cdf[-1] = max(cdf[-1], 1.0)
        _FRANK_TABLE_CACHE[key] = cdf
    cdf = _FRANK_TABLE_CACHE[key]
    counts = V0.astype(np.int64)
    owner = np.repeat(np.arange(len(V0)), counts)
    X = np.searchsorted(cdf, rng.random(len(owner)) * cdf[-1]) + 1
    return np.bincount(owner, weights=X, minlength=len(V0)).astype(float)


# =============================================================================
# PART C.  NESTED ARCHIMEDEAN COPULA FOR 3 INDICATORS     (Eq. 9, 14)
# =============================================================================
# Paper Eq. (9):  C(u1, u2, u3) = C_outer( C_inner(u1, u2; theta1), u3; theta2 )
# The two most strongly linked indicators go in the inner copula (stronger
# dependence, theta1), and the third joins through the outer copula (weaker,
# theta2). For this to be a valid copula (same family inside and outside),
# theta1 >= theta2 (paper Table 5 shows theta1 > theta2 for all families).
#
# CHINA VERSION (all three indicators available):
#   Estimation = full 3-D MAXIMUM LIKELIHOOD, as the paper does (Table 5:
#   "Maximum likelihood estimates of the parameters involved in the inner and
#   outer copulas"). The 3-D density c(u1,u2,u3) = d^3 C / du1 du2 du3 is
#   derived EXACTLY by the computer-algebra package sympy (no hand algebra,
#   no numerical differencing), once per family. Kendall-tau inversion
#   (the India A3 method) is kept as the starting value and as a check.
#   Simulation = exact Marshall-Olkin / Hofert (2011) algorithms for all three
#   families, including nested FRANK, which the paper uses for pricing
#   (Sec. 3.5, Step 2: "the nested Frank copula in Equation (14)").
# =============================================================================
_NESTED_LOGPDF_CACHE = {}


def _nested_logpdf_function(family):
    """Build (once) a fast numpy function for log c(u1,u2,u3; t1, t0) of the
    nested copula C0(C1(u1,u2; t1), u3; t0), by symbolic differentiation."""
    if family in _NESTED_LOGPDF_CACHE:
        return _NESTED_LOGPDF_CACHE[family]
    import sympy as sp
    u1, u2, u3 = sp.symbols("u1 u2 u3", positive=True)
    t1, t0 = sp.symbols("t1 t0", positive=True)

    def gen(t, th):          # generator phi (Table 2)
        if family == "gumbel":
            return (-sp.log(t)) ** th
        if family == "clayton":
            return t ** (-th) - 1
        return -sp.log((sp.exp(-th * t) - 1) / (sp.exp(-th) - 1))

    def inv(s, th):          # inverse generator psi
        if family == "gumbel":
            return sp.exp(-s ** (1 / th))
        if family == "clayton":
            return (1 + s) ** (-1 / th)
        return -sp.log(1 + sp.exp(-s) * (sp.exp(-th) - 1)) / th

    w = inv(gen(u1, t1) + gen(u2, t1), t1)          # inner copula C1(u1,u2)
    C = inv(gen(w, t0) + gen(u3, t0), t0)           # outer copula
    dens = sp.diff(C, u1, u2, u3)                   # exact 3-D density
    f = sp.lambdify((u1, u2, u3, t1, t0), dens, modules="numpy")
    _NESTED_LOGPDF_CACHE[family] = f
    return f


class NestedCopula:
    dim = 3

    def __init__(self, family, theta_inner, theta_outer, inner, outer):
        self.family = family
        self.theta_inner, self.theta_outer = float(theta_inner), float(theta_outer)
        self.inner = list(inner)      # positions of the inner pair, e.g. [0, 1] = AP, CAA
        self.outer = int(outer)       # position of the third indicator, e.g. 2 = DEL

    def cdf(self, U):
        U = np.atleast_2d(U)
        f, t1, t0 = self.family, self.theta_inner, self.theta_outer
        c_in = psi(phi(U[:, self.inner[0]], f, t1) + phi(U[:, self.inner[1]], f, t1), f, t1)
        return psi(phi(c_in, f, t0) + phi(U[:, self.outer], f, t0), f, t0)

    # ---- log-density for maximum likelihood (paper Table 5) ---------------
    def logpdf(self, U):
        U = np.clip(np.atleast_2d(U), 1e-10, 1 - 1e-10)
        f = _nested_logpdf_function(self.family)
        with np.errstate(all="ignore"):
            d = f(U[:, self.inner[0]], U[:, self.inner[1]], U[:, self.outer],
                  self.theta_inner, self.theta_outer)
            d = np.asarray(d, dtype=float) * np.ones(len(U))
            return np.where((d > 0) & np.isfinite(d), np.log(np.where(d > 0, d, 1)), -np.inf)

    @classmethod
    def fit(cls, U, family, inner, outer, start=None):
        """3-D maximum likelihood for (theta_inner, theta_outer) with the
        nesting rule theta_inner >= theta_outer. We optimise over
        theta_outer and the GAP g = theta_inner - theta_outer >= 0."""
        lo, hi = THETA_BOUNDS[family]
        if start is None:   # Kendall-tau inversion as the starting point
            start = cls.tau_inversion(U, family, inner, outer)
        t1s, t0s = start.theta_inner, start.theta_outer

        def nll(par):
            t0, g = par
            val = -np.sum(cls(family, t0 + g, t0, inner, outer).logpdf(U))
            return val if np.isfinite(val) else 1e12
        best = None
        for x0 in ([t0s, max(t1s - t0s, 1e-3)], [t0s * 0.8, max(t1s - t0s, 1e-3) * 1.5],
                   [t0s * 1.2, 1e-3]):
            x0 = [float(np.clip(x0[0], lo + 1e-3, hi - 1)), float(x0[1])]
            r = optimize.minimize(nll, x0, method="L-BFGS-B",
                                  bounds=[(lo, hi), (0.0, hi)])
            if best is None or r.fun < best.fun:
                best = r
        t0, g = best.x
        cop = cls(family, t0 + g, t0, inner, outer)
        cop.loglik = -best.fun
        return cop

    @classmethod
    def tau_inversion(cls, U, family, inner, outer):
        """theta1 from Kendall's tau of the inner pair; theta2 from the
        average tau of the two cross pairs (method used for India A3)."""
        t_in = stats.kendalltau(U[:, inner[0]], U[:, inner[1]])[0]
        t_out = np.mean([stats.kendalltau(U[:, k], U[:, outer])[0] for k in inner])
        th1, th0 = theta_from_tau(family, t_in), theta_from_tau(family, t_out)
        return cls(family, max(th1, th0), th0, inner, outer)

    def sample(self, n, rng):
        f, t1, t0 = self.family, self.theta_inner, self.theta_outer
        alpha = t0 / t1                       # between 0 and 1
        V0 = frailty(f, t0, n, rng)          # outer shared shock
        if f == "gumbel":
            # inner shock: V01 = V0^(t1/t0) * PositiveStable(alpha)
            V01 = V0 ** (1 / alpha) * positive_stable(alpha, n, rng)
        elif f == "clayton":
            V01 = tilted_stable(alpha, V0, rng)
        elif f == "frank":
            V01 = frank_inner_frailty(t0, t1, V0, rng)
        else:
            raise ValueError(f)
        E = rng.exponential(size=(n, 3))
        out = np.empty((n, 3))
        out[:, self.inner[0]] = psi(E[:, 0] / V01, f, t1)
        out[:, self.inner[1]] = psi(E[:, 1] / V01, f, t1)
        out[:, self.outer] = psi(E[:, 2] / V0, f, t0)
        return out


def tilted_stable(alpha, V0, rng):
    """Draws with Laplace transform exp(-V0 * ((1+t)^alpha - 1)), needed for
    nested Clayton (Hofert 2010). Rejection method, split into ceil(V0)
    pieces so that the acceptance rate stays high."""
    pieces = np.maximum(np.ceil(V0), 1).astype(int)
    owner = np.repeat(np.arange(len(V0)), pieces)       # which draw each piece belongs to
    c = np.repeat(V0 / pieces, pieces)                   # each piece's share of V0
    vals = np.zeros(len(c))
    todo = np.arange(len(c))                             # pieces not yet accepted
    while len(todo) > 0:                                 # all pieces at once
        s = c[todo] ** (1 / alpha) * positive_stable(alpha, len(todo), rng)
        ok = rng.random(len(todo)) <= np.exp(-s)         # accept with prob e^(-s)
        vals[todo[ok]] = s[ok]
        todo = todo[~ok]
    return np.bincount(owner, weights=vals, minlength=len(V0))


# ---- Empirical copula and a goodness-of-fit distance -------------------------
def empirical_copula(U, at):
    """C_n(points) = share of floods whose (u1..um) are all <= the point."""
    return np.array([np.mean(np.all(U <= p, axis=1)) for p in at])


def ks_distance(U, copula):
    """Kolmogorov-Smirnov style distance between the data's empirical copula
    and the fitted copula, measured at the data points (paper Table 5 uses a
    KS test to choose the copula). Smaller = better fit."""
    return np.max(np.abs(empirical_copula(U, U) - copula.cdf(U)))


# =============================================================================
# PART D.  COUPON AND PRINCIPAL RETENTION      (Eq. 1, 2, 3, 10, 11)
# =============================================================================
# For flood j and indicator i with trigger level d_i:
#     a_ij = (x_ij - d_i)+ / x_ij       "share of the flood above the line",
#                                       0 if below the line, close to 1 if far above.
#     s_j   = PRODUCT over i        of (1 - a_ij)                  Eq. (1)
#     s*_j  = PRODUCT over pairs    of (1 - a_i1 * a_i2)           Eq. (2)
#     s**_j = PRODUCT over triples  of (1 - a_i1 * a_i2 * a_i3)    Eq. (3)
#   s = 1 means nothing triggered; smaller = worse flood.
#   s* < 1 only if at least TWO indicators are above their lines at once.
#   s** < 1 only if THREE are (only possible in the A3 extension).
# Yearly values (Eq. 10 = average over the year's floods, Eq. 11 = maximum):
#     alpha_t from s_j  -> share of the COUPON kept
#     beta_t  from s*_j -> share of the PRINCIPAL kept
#     gamma_t from s**_j-> if gamma_t < 1, the principal factor is halved
# A year with NO flood keeps everything: alpha = beta = gamma = 1 (the paper
# leaves this case implicit; any other choice would penalise calm years).
# =============================================================================
def flood_scores(X, d):
    """X: floods x indicators (real units). d: trigger levels.
    Returns s, s*, s** for each flood."""
    A = np.clip(X - d, 0, None) / X                    # a_ij
    m = X.shape[1]
    s = np.prod(1 - A, axis=1)
    s_star = np.ones(len(X))
    for i1, i2 in itertools.combinations(range(m), 2):
        s_star *= 1 - A[:, i1] * A[:, i2]
    s_2star = np.ones(len(X))
    for i1, i2, i3 in itertools.combinations(range(m), 3):
        s_2star *= 1 - A[:, i1] * A[:, i2] * A[:, i3]
    return s, s_star, s_2star


def yearly_retention(scores, sim_id, n_sims, how):
    """Combine flood scores into one value per simulated year.
    how = "avg" (Eq. 10) or "max" (Eq. 11). Years with no flood -> 1."""
    if how == "avg":
        total = np.bincount(sim_id, weights=scores, minlength=n_sims)
        count = np.bincount(sim_id, minlength=n_sims)
        return np.where(count > 0, total / np.maximum(count, 1), 1.0)
    out = np.full(n_sims, -np.inf)
    np.maximum.at(out, sim_id, scores)
    return np.where(np.isinf(out), 1.0, out)


# =============================================================================
# PART E.  CASHFLOWS AND PRICE               (Eq. 4 and 5)
# =============================================================================
# Investor buys at the end of year t, bond matures at year T. C0 = F * R.
#   principal factor in year k:  pi_k = beta_k / 2  if gamma_k < 1
#                                     = beta_k      otherwise
#   Coupon in year s = t+1:      C = alpha_{t+1} * C0
#   Coupon in year s >= t+2:     C = alpha_s * (pi_{t+1} * ... * pi_{s-1}) * C0
#   Redemption at T:             F_T = (pi_{t+1} * ... * pi_T) * F          Eq. (4)
#   Price:  P_t = sum_s E[C_s] * p(t,s) + E[F_T] * p(t,T)                   Eq. (5)
#   Discount factor (our simplification of Sec. 2.2): p(t,s) = exp(-r (s-t))
# Arrays alpha, beta, gamma have shape (n_sims, number of years); column 0 is
# year 1.
# =============================================================================
def bond_price(alpha, beta, gamma, t, T, F, R, r):
    C0 = F * R
    pi = np.where(gamma < 1, beta / 2, beta)
    price_paths = np.zeros(alpha.shape[0])     # one price per simulated future
    carried = np.ones(alpha.shape[0])          # running product of pi_k
    for s in range(t + 1, T + 1):              # years t+1 ... T
        coupon = alpha[:, s - 1] * carried * C0
        price_paths += coupon * np.exp(-r * (s - t))
        carried = carried * pi[:, s - 1]       # principal factor after year s
    price_paths += carried * F * np.exp(-r * (T - t))
    # Average over all simulated futures = the expectation E{...} in Eq. (5).
    # Standard error tells us how precise that average is.
    return price_paths.mean(), price_paths.std(ddof=1) / np.sqrt(len(price_paths))


# =============================================================================
# PART F.  THE MONTE CARLO SIMULATION        (Sec. 3.5, Steps 1-4)
# =============================================================================
#  Step 1: number of floods in each year  N_k ~ Poisson(Lambda_k)
#  Step 2: for each flood, draw (u1..um) from the copula, then convert to real
#          sizes with x_i = F_i^-1(u_i)
#  Steps 3-4: compute retention, cashflows, discount (done in PART E).
# Returns alpha/beta/gamma arrays for both the "avg" (Eq. 10) and "max"
# (Eq. 11) versions.
# =============================================================================
def simulate_retentions(n_sims, lambdas, copula, marginals, d, rng, columns=None):
    """columns: optional list of indicator positions to USE in the trigger
    (for paper Table 7, where a 3-indicator draw is priced on a pair)."""
    T = len(lambdas)
    res = {k: np.ones((n_sims, T)) for k in
           ["alpha_avg", "beta_avg", "gamma_avg", "alpha_max", "beta_max", "gamma_max"]}
    for k, lam in enumerate(lambdas):
        N = rng.poisson(lam, n_sims)                         # Step 1
        total = N.sum()
        if total == 0:
            continue
        sim_id = np.repeat(np.arange(n_sims), N)             # which future each flood belongs to
        U = copula.sample(total, rng)                        # Step 2 (copula)
        X = np.column_stack([marginals[i].ppf(U[:, i]) for i in range(U.shape[1])])
        dd = np.asarray(d)
        if columns is not None:
            X, dd = X[:, columns], dd[columns]
        s, s1, s2 = flood_scores(X, dd)
        for how in ["avg", "max"]:
            res[f"alpha_{how}"][:, k] = yearly_retention(s, sim_id, n_sims, how)
            res[f"beta_{how}"][:, k] = yearly_retention(s1, sim_id, n_sims, how)
            res[f"gamma_{how}"][:, k] = yearly_retention(s2, sim_id, n_sims, how)
    return res


def price_table(res, T, F, R, r):
    """Prices for every maturity 1..T and purchase year t = 0..T'-1
    (paper Table 6), for both retention functions."""
    rows = []
    for Tm in range(1, T + 1):
        for t in range(Tm):
            row = {"maturity_T": Tm, "purchase_year_t": t}
            for how, label in [("avg", "P1_average_Eq10"), ("max", "P2_maximum_Eq11")]:
                p, se = bond_price(res[f"alpha_{how}"], res[f"beta_{how}"],
                                   res[f"gamma_{how}"], t, Tm, F, R, r)
                row[label], row[label + "_se"] = p, se
            rows.append(row)
    return rows


# =============================================================================
# PART G.  LOADING RESULTS SAVED BY EARLIER STEPS
# =============================================================================
def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_marginals(cfg, tag=None):
    """Fitted Beta-GPD margins saved by step 4 (tag = "ours" or "paper")."""
    tag = tag or cfg.THRESHOLD_SET
    p = load_json(cfg.PARAMS_DIR / f"marginal_params_{tag}.json")
    return [BetaGPDMarginal.from_dict(p[ind]) for ind in cfg.INDICATORS]


def load_copula(cfg, tag=None, family=None):
    """Nested copula saved by step 5. family=None -> the one chosen there."""
    tag = tag or cfg.THRESHOLD_SET
    p = load_json(cfg.PARAMS_DIR / f"copula_params_{tag}.json")
    fam = family or p["selected_family"]
    s = p["nested"][fam]
    return NestedCopula(fam, s["theta_inner"], s["theta_outer"], p["inner"], p["outer"])


def trigger_levels(cfg, events, q=None):
    """Trigger level d_i = sample quantile of each indicator (paper Table 6
    uses the 90% quantile)."""
    q = cfg.TRIGGER_Q if q is None else q
    return np.array([np.quantile(events[ind], q) for ind in cfg.INDICATORS])
