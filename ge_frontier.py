"""
ge_frontier.py -- V2 general-equilibrium extension (NEWLY CONSTRUCTED, not a pre-existing script).

This file is new in the v2-ge-extension branch. It does NOT touch main or verify.py's
frozen v1 behaviour. It builds an exact Armington CES general-equilibrium payoff layer
and re-evaluates the paper's four structural routes on exact real-income indirect utility,
replacing the reduced-form quadratic welfare of pipeline.py::NCountryTradeGame.

Dependencies: NumPy + SciPy (scipy.optimize) ONLY. No torch, no JAX, no new packages.

What it computes
    1. CESGeneralEquilibrium: a one-sector, one-factor Armington model. Each country i
       produces a differentiated good priced at its wage w_i, consumers have CES demand
       (elasticity sigma) over all origins, country i levies a uniform import tariff t_i
       whose revenue is rebated lump-sum. Market clearing (producer revenue = labour
       income) is a fixed point in wages, solved with scipy.optimize.root. The payoff is
       the EXACT real-income indirect utility U_i = E_i / P_i (nominal expenditure over
       the CES price index) -- not a second-order approximation.

    2. GE payoff surfaces over tariff grids, cached to data/processed/v2_ge_surfaces.npz:
       the bilateral US-China 2-D surface and the N=30 per-country best-response slices.

    3. The four routes on GE payoffs, saved to data/processed/v2_ge_routes.json:
         R1  LOLA transparency      -- Press-Dyson memory-one construction (colab/run_A2.py)
                                        fed GE-derived stage payoffs R/T/P/S.
         R2  evolutionary basins    -- replicator dynamics over {ALLC, ALLD, TFT}.
         R3  noisy-monitoring w*     -- optimal enforcement tolerance and delta*(sigma_noise).
         R4  N=30 network rankings  -- GE folk delta* and exposure rankings for 30 economies.

Modelling choices (confirmed 2026-11-20, documented in REGISTRY.md / docs/provenance.md):
    * CES elasticity sigma = 4.0 primary (Ossa 2014), sensitivity grid {2.5, 4.0, 6.0}.
    * The 30x30 bilateral structure is NOT in stage2_results.json, so it is
      gravity-reconstructed from the calibrated b and exposure: off-diagonal partner
      weights proportional to b_j, each row scaled so its foreign mass equals the
      economy's calibrated exposure e_i (home weight = 1 - e_i). An optional hook loads
      raw CEPII BACI CSVs if present locally; absent them the reconstruction is used.
    * sigma_noise (monitoring noise) is a DIFFERENT object from the CES sigma; it is
      always named sigma_noise in code to avoid collision.

Run:  python ge_frontier.py
"""
from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
from scipy import optimize

# ===================== CONFIG =====================
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
STAGE2 = PROC / "stage2_results.json"
SURFACES_NPZ = PROC / "v2_ge_surfaces.npz"
ROUTES_JSON = PROC / "v2_ge_routes.json"

SIGMA_GRID = [2.5, 4.0, 6.0]      # CES elasticity sensitivity grid
SIGMA_PRIMARY = 4.0               # headline structural baseline (Ossa 2014)

# Bilateral US-China calibration: the paper's two-country market-power vector (verify.py v1).
B_US, B_CN = 0.66, 0.56

# Tariff grids. Best responses are found by continuous optimisation on [0, TMAX];
# the cached surface is sampled on a coarser grid for inspection / plotting.
TMAX = 3.0
SURF_GRID = np.round(np.linspace(0.0, 1.2, 41), 6)      # bilateral 2-D and N=30 slices
REPLICATOR_DELTA = 0.9            # discount for the evolutionary route (matches LOLA run)

# Noisy-monitoring theorem constants (carried from verify.py v1 Section 6.3; the w* curve
# is monitoring geometry, payoff-independent -- the GE enters only through G and L below).
CX_OVER_CF = 0.9529
D_SIGNAL = 0.58                   # normalised cooperative-vs-defect signal gap (v1 §6.3)


# ===================== THE CES GENERAL-EQUILIBRIUM MODEL =====================
class CESGeneralEquilibrium:
    """One-sector, one-factor Armington CES general equilibrium with import tariffs.

    Primitives
        L     (N,)   labour endowments  (country "size")
        beta  (N,N)  CES preference weights; beta[i, j] = weight market i puts on origin j.
                     Rows sum to 1 (beta[i, i] is the home weight).
        sigma scalar CES elasticity of substitution (> 1).

    For a tariff vector t (N,), solve() returns the exact equilibrium: wages w clearing
    all goods markets (producer revenue = labour income, Walras drops one equation; the
    first country's wage is the numeraire), and the real-income indirect utility
    U_i = E_i / P_i for every country.
    """

    def __init__(self, L, beta, sigma):
        self.L = np.asarray(L, float)
        self.beta = np.asarray(beta, float)
        self.N = len(self.L)
        self.sigma = float(sigma)
        self._eye = np.eye(self.N, dtype=bool)
        self._wlast = None            # warm-start: last solved wage vector (consecutive
                                      # tariff profiles are close, so the root solver then
                                      # converges in a handful of residual evaluations)

    # ---- the equilibrium system, vectorised over the NxN origin-market grid ----
    def _allocate(self, w, t):
        """Given wages w (N,) and tariffs t (N,), return (E, P, lam, Rev).

        Pc[i, j]  consumer price in market i of a good from origin j (tariff on imports)
        lam[i, j] share of market-i expenditure falling on origin j  (CES)
        E[i]      total expenditure in market i  (labour income + rebated tariff revenue)
        P[i]      CES price index in market i
        Rev[j]    producer revenue accruing to country j  (border prices, net of tariff)
        """
        w = np.asarray(w, float)
        tcol = t[:, None]
        Pc = w[None, :] * (1.0 + np.where(self._eye, 0.0, tcol))   # no tariff on the diagonal
        A = self.beta * Pc ** (1.0 - self.sigma)
        Dsum = A.sum(axis=1)
        lam = A / Dsum[:, None]
        sM = 1.0 - np.diagonal(lam)                                # import expenditure share
        # E_i = w_i L_i + [t_i/(1+t_i)] * sM_i * E_i  (tariff revenue rebated lump-sum)
        E = (w * self.L) / (1.0 - (t / (1.0 + t)) * sM)
        P = Dsum ** (1.0 / (1.0 - self.sigma))
        # producer revenue: domestic at full price, imports net of the importer's tariff
        g = np.where(self._eye, 1.0, 1.0 / (1.0 + tcol))           # g[i, j] passthrough to producer j
        Rev = (lam * (E[:, None]) * g).sum(axis=0)
        return E, P, lam, Rev

    def _residual(self, logw_free, t):
        """Market-clearing residual for the N-1 non-numeraire wages (log-parameterised).
        Used only by the scipy.optimize.root fallback below."""
        w = np.empty(self.N)
        w[0] = 1.0
        w[1:] = np.exp(logw_free)
        E, P, lam, Rev = self._allocate(w, t)
        return (Rev - w * self.L)[1:]                              # drop country 0 (Walras)

    def solve(self, t, w0=None):
        """Solve the GE for tariff vector t with scipy.optimize.root (Powell hybrid) on the
        N-1 market-clearing equations (country 0's wage is the numeraire). Warm-started from
        the previous solve -- consecutive tariff profiles are close, so the solver converges
        in ~3 ms. Returns w, E, P, U (real income), lam."""
        t = np.asarray(t, float)
        if w0 is None:
            w0 = self._wlast
        guess = np.zeros(self.N - 1) if w0 is None else np.log(np.clip(np.asarray(w0, float)[1:], 1e-9, None))
        sol = optimize.root(self._residual, guess, args=(t,), method="hybr", tol=1e-11)
        if not sol.success:                                        # cold retry (rare)
            sol = optimize.root(self._residual, np.zeros(self.N - 1), args=(t,), method="hybr", tol=1e-11)
        w = np.empty(self.N)
        w[0] = 1.0
        w[1:] = np.exp(sol.x)
        self._wlast = w.copy()
        E, P, lam, Rev = self._allocate(w, t)
        return {"w": w, "E": E, "P": P, "U": E / P, "lam": lam, "ok": bool(sol.success)}

    def utility(self, t, w0=None):
        return self.solve(t, w0)["U"]

    # ---- best response and Nash in tariffs ----
    def best_response(self, i, t, xatol=1e-6):
        """Country i's welfare-maximising tariff given the others' tariffs in t."""
        t = np.asarray(t, float).copy()

        def neg(ti):
            tt = t.copy()
            tt[i] = ti
            return -self.utility(tt)[i]

        res = optimize.minimize_scalar(neg, bounds=(0.0, TMAX), method="bounded",
                                       options={"xatol": xatol})
        return float(res.x)

    def nash(self, iters=80, tol=1e-6):
        """Pure-strategy Nash tariff vector by iterated best response (Gauss-Seidel).
        Converges geometrically in ~10-20 sweeps for this game."""
        t = np.full(self.N, 0.2)
        for _ in range(iters):
            prev = t.copy()
            for i in range(self.N):
                t[i] = self.best_response(i, t, xatol=1e-5)
            if np.max(np.abs(t - prev)) < tol:
                break
        return t


# ===================== CALIBRATION =====================
def gravity_beta(b, exposure):
    """Reconstruct the CES preference matrix from calibrated b and exposure.

    beta[i, j] (j != i) = e_i * b_j / sum_{k != i} b_k      (off-diagonal ~ partner size b_j)
    beta[i, i]          = 1 - e_i                           (home weight)
    so each row sums to 1 and the foreign mass equals the economy's exposure e_i.
    """
    b = np.asarray(b, float)
    e = np.asarray(exposure, float)
    N = len(b)
    beta = np.zeros((N, N))
    for i in range(N):
        mask = np.arange(N) != i
        denom = b[mask].sum()
        beta[i, mask] = e[i] * b[mask] / denom
        beta[i, i] = 1.0 - e[i]
    return beta


def bilateral_model(sigma):
    """US-China two-country GE. Size ~ b; exposure mirrors the v1 row-stochastic W=[[0,1],[1,0]]
    (so e_US = b_CN, e_CN = b_US: each imports a mass equal to the partner's market power)."""
    b = np.array([B_US, B_CN])
    exposure = np.array([B_CN, B_US])      # partner market power = import preference mass
    L = b.copy()                           # labour endowment proportional to size
    beta = gravity_beta(b, exposure)
    return CESGeneralEquilibrium(L, beta, sigma), b, exposure


def network_model(sigma, stage2):
    """N=30 GE from stage2_results.json calibrated b and exposure."""
    b = np.array(stage2["b"], float)
    exposure = np.array(stage2["exposure"], float)
    L = b.copy()
    beta = gravity_beta(b, exposure)
    return CESGeneralEquilibrium(L, beta, sigma), b, exposure


# ===================== GE STAGE GAME (bilateral) =====================
def bilateral_stage_payoffs(sigma):
    """Compute the 2x2 GE stage game for US(0) and CN(1), folk-theorem convention.

       C = free trade (t=0). Temptation T = payoff from the OPTIMAL unilateral deviation
       (GE best response to a cooperating partner) -- matching v1's G = b^2/2k, which is the
       optimal deviation, not the Nash-level one. Punishment P = reversion to the stage
       Nash. Sucker S = cooperate while the partner takes its optimal deviation. (As usual
       when reducing a continuous game to a 2x2 PD, the "D" in DC/CD is the best deviation
       while the "D" in DD is the Nash action; this is the standard repeated-game mapping.)
       Returns per-player R,S,T,P in real-income units, the Nash and the deviation tariffs."""
    ge, b, exp = bilateral_model(sigma)
    nash = ge.nash()
    free = np.array([0.0, 0.0])
    brUS = ge.best_response(0, free)          # US optimal deviation vs cooperating CN
    brCN = ge.best_response(1, free)          # CN optimal deviation vs cooperating US
    u = lambda t: ge.utility(np.array(t, float))
    U_CC = u(free)
    U_DD = u(nash)
    U_devUS = u([brUS, 0.0])                   # US deviates, CN cooperates
    U_devCN = u([0.0, brCN])                   # CN deviates, US cooperates
    pay = {
        "nash_tariff": nash.tolist(),
        "deviation_tariff": {"US": float(brUS), "CN": float(brCN)},
        "US": {"R": float(U_CC[0]), "T": float(U_devUS[0]), "P": float(U_DD[0]), "S": float(U_devCN[0])},
        "CN": {"R": float(U_CC[1]), "T": float(U_devCN[1]), "P": float(U_DD[1]), "S": float(U_devUS[1])},
    }
    return pay, ge


# ===================== GE FOLK delta* (per country) =====================
def ge_folk_delta_star(ge, free=None, nash=None):
    """For each country: G = T - R (unilateral defection gain), L = R - P (punishment loss),
       delta* = G/(G+L). 'Defect' = GE best response while all others stay at free trade;
       'punish' = all-Nash. Real-income units. delta* = inf (hegemon) when L <= 0."""
    N = ge.N
    free = np.zeros(N) if free is None else np.asarray(free, float)
    U_free = ge.utility(free)                       # R_i : everyone cooperates (free trade)
    if nash is None:
        nash = ge.nash()
    U_nash = ge.utility(nash)                        # P_i : everyone defects (Nash)
    G = np.zeros(N); L = np.zeros(N); T = np.zeros(N)
    for i in range(N):
        bri = ge.best_response(i, free)              # i's optimal unilateral deviation
        t = free.copy(); t[i] = bri
        Ui = ge.utility(t)[i]                         # T_i
        T[i] = Ui
        G[i] = Ui - U_free[i]
        L[i] = U_free[i] - U_nash[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = np.where(L > 0, G / (G + L), np.inf)
    return {"delta_star": ds, "G": G, "L": L, "T": T,
            "U_free": U_free, "U_nash": U_nash, "nash_tariff": nash}


# ===================== ROUTE 1: LOLA TRANSPARENCY (Press-Dyson) =====================
def route_lola(stage_pay, delta=0.9, steps=400, seeds=6):
    """Re-run the pure-NumPy opponent-shaping construction (colab/run_A2.py) on GE payoffs.

    The memory-one value function is asymmetric here (US and CN have different GE payoffs),
    so we set A2.U1 and A2.U2 to the two players' GE stage-payoff vectors, normalised so the
    cooperative outcome R = 0 and rescaled by a common monotone SCALE to the reduced-form
    magnitude the run_A2_scaled hyperparameters were tuned for (order-preserving; divides
    reported payoffs back out). This is exactly the rescaling justified in run_A2_scaled.py.
    """
    sys.path.insert(0, str(ROOT / "colab"))
    import run_A2 as A2

    us, cn = stage_pay["US"], stage_pay["CN"]
    # joint-state order [CC, CD, DC, DD] from P1(=US) perspective (see run_A2.values)
    U1 = np.array([us["R"], us["S"], us["T"], us["P"]]) - us["R"]   # US payoff; CC normalised to 0
    U2 = np.array([cn["R"], cn["T"], cn["S"], cn["P"]]) - cn["R"]   # CN payoff; CC normalised to 0
    # Monotone order-preserving rescale to the magnitude the run_A2_scaled hyperparameters
    # (eta=40, alpha=0.08) were tuned for: the reduced-form game has max|payoff| = |S| = 0.30
    # and run_A2_scaled multiplies it by 12, so the tuned game has max|payoff| ~ 3.6. We map
    # the GE game's largest payoff to the same 3.6 (and divide reported payoffs back out).
    TARGET_MAG = 0.30 * 12.0
    mag = max(abs(U1).max(), abs(U2).max())
    SCALE = TARGET_MAG / mag if mag > 0 else 1.0
    A2.U1 = U1 * SCALE
    A2.U2 = U2 * SCALE

    out = {"delta": delta, "scale": float(SCALE),
           "stage_payoffs": {"US": us, "CN": cn}}
    for mode in ["naive", "lola"]:
        joints = []
        for s in range(seeds):
            th1, th2, _ = A2.train(mode, steps=steps, alpha=0.08, eta=40, seed=s)
            v1, v2 = A2.values(th1, th2, delta=delta)
            joints.append(((v1 + v2) / 2.0) / SCALE)
        out[mode + "_joint"] = [float(x) for x in joints]
        out[mode + "_joint_mean"] = float(np.mean(joints))
        out[mode + "_joint_sd"] = float(np.std(joints))
    # exploitability: a transparent learner against a fixed always-defector
    alld = np.full(5, -8.0)
    th1, _, _ = A2.train("lola", steps=steps, alpha=0.08, eta=40, seed=0, th2_fixed=alld)
    v1, v2 = A2.values(th1, alld, delta=delta)
    out["lola_vs_alld"] = {"V_lola": float(v1 / SCALE), "V_alld": float(v2 / SCALE),
                           "policy": A2.sig(th1).tolist()}
    # cooperative reference: CC joint payoff (both 0 after normalisation -> raw R average)
    out["cooperation_payoff_R"] = 0.0
    out["defection_payoff_P"] = float(((us["P"] - us["R"]) + (cn["P"] - cn["R"])) / 2.0)
    out["lola_crosses_gap"] = bool(out["lola_joint_mean"] > out["naive_joint_mean"] + 1e-6
                                   and out["lola_joint_mean"] > out["defection_payoff_P"])
    sys.path.pop(0)
    return out


# ===================== ROUTE 2: EVOLUTIONARY REPLICATOR BASINS =====================
def repeated_payoff_matrix(R, S, T, P, delta):
    """3x3 normalised discounted payoff matrix over strategies [ALLC, ALLD, TFT]
       in the infinitely repeated game (per-period units)."""
    ALLC, ALLD, TFT = 0, 1, 2
    M = np.zeros((3, 3))
    M[ALLC, ALLC] = R; M[ALLC, ALLD] = S;               M[ALLC, TFT] = R
    M[ALLD, ALLC] = T; M[ALLD, ALLD] = P;               M[ALLD, TFT] = (1 - delta) * T + delta * P
    M[TFT, ALLC] = R;  M[TFT, ALLD] = (1 - delta) * S + delta * P; M[TFT, TFT] = R
    return M


def route_evolution(R, S, T, P, delta=REPLICATOR_DELTA, n_starts=2000, dt=0.05, steps=3000, seed=0):
    """Replicator dynamics over {ALLC, ALLD, TFT}. Reports the basin of cooperation
       (fraction of random simplex starts that flow to the cooperative rest point, i.e. ALLD
       a minority at convergence) at `delta`, the basin as a function of delta, and the
       closed-form TFT-vs-ALLD invasion threshold delta_inv = (T-R)/(T-P) = G/(G+L).
       Dynamics are vectorised over all starts at once."""
    rng = np.random.default_rng(seed)
    starts = rng.dirichlet(np.ones(3), size=n_starts)      # fixed start cloud across deltas

    def basin(d):
        M = repeated_payoff_matrix(R, S, T, P, d)
        X = starts.copy()                                  # (n_starts, 3)
        for _ in range(steps):
            F = X @ M.T                                    # fitness of each strategy per start
            avg = np.sum(X * F, axis=1, keepdims=True)
            X = X + dt * X * (F - avg)
            np.clip(X, 0.0, None, out=X); X /= X.sum(axis=1, keepdims=True)
        return float(np.mean(X[:, 1] < 0.5))               # ALLD (index 1) a minority -> cooperation

    delta_inv = (T - R) / (T - P) if (T - P) != 0 else np.inf
    dgrid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    basin_by_delta = {f"{d:.2f}": float(basin(d)) for d in dgrid}
    return {"delta": delta, "basin_cooperation": float(basin(delta)),
            "basin_by_delta": basin_by_delta,
            "delta_inv_tft_vs_alld": float(delta_inv),
            "note": "basin = fraction of Dirichlet(1,1,1) starts flowing to cooperation (ALLD minority at convergence)"}


# ===================== ROUTE 3: NOISY-MONITORING OPTIMAL TOLERANCE =====================
def _K():
    import math
    return 1.0 / (math.sqrt(2 * math.pi) * CX_OVER_CF)


def w_star(sigma_noise):
    import math
    K = _K()
    arg = 2 * math.log(K / sigma_noise) if sigma_noise > 0 else -1
    return sigma_noise * math.sqrt(arg) if arg > 0 else 0.0


def _Phi(z):
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def m_of(sigma_noise, d=D_SIGNAL):
    w = w_star(sigma_noise)
    return _Phi((d - w) / sigma_noise) - _Phi(-w / sigma_noise)


def route_tolerance(G, L):
    """Enforcement tolerance under noisy monitoring, on GE G and L. The w*(sigma_noise)
       curve is monitoring geometry (payoff-independent, carried from v1 §6.3); the GE
       enters delta*(sigma_noise) = G/(G + L*m(sigma_noise)). sigma_noise != CES sigma."""
    import math
    K = _K()
    peak_sigma = K / math.sqrt(math.e)
    grid = [0.05, 0.10, 0.20, 0.30, 0.40]
    return {"K": float(K), "CX_OVER_CF": CX_OVER_CF,
            "peak_tolerance_sigma": float(peak_sigma),
            "w_star": {f"{s:.2f}": float(w_star(s)) for s in grid},
            "m": {f"{s:.2f}": float(m_of(s)) for s in grid},
            "G": float(G), "L": float(L),
            "delta_star_by_sigma_noise": {f"{s:.2f}": float(G / (G + L * m_of(s))) for s in grid},
            "delta_star_noiseless": float(G / (G + L)),
            "d_signal": D_SIGNAL}


# ===================== ROUTE 4: N=30 NETWORK RANKINGS =====================
def route_network(ge, b, exposure, iso, nash=None):
    fd = ge_folk_delta_star(ge, nash=nash)
    ds = fd["delta_star"]
    fin = np.isfinite(ds)
    b = np.asarray(b); exposure = np.asarray(exposure)

    def partial_corr(y, x, z):
        def resid(v):
            A = np.vstack([z, np.ones_like(z)]).T
            c, *_ = np.linalg.lstsq(A, v, rcond=None)
            return v - A @ c
        return float(np.corrcoef(resid(y), resid(x))[0, 1])

    order = np.argsort(-np.where(fin, ds, -np.inf))     # descending delta*
    ranking = [iso[k] for k in order]
    hegemons = [iso[i] for i in range(ge.N) if not fin[i]]
    binding = iso[int(np.nanargmax(np.where(fin, ds, -np.inf)))]
    return {
        "iso": iso,
        "delta_star": [None if not fin[i] else float(ds[i]) for i in range(ge.N)],
        "G": fd["G"].tolist(), "L": fd["L"].tolist(),
        "U_free": fd["U_free"].tolist(), "U_nash": fd["U_nash"].tolist(),
        "nash_tariff": fd["nash_tariff"].tolist(),
        "binding_country": binding,
        "binding_delta_star": float(np.nanmax(ds[fin])),
        "hegemons": hegemons,
        "ranking_desc": ranking,
        "MEX_rank": ranking.index("MEX") + 1, "CAN_rank": ranking.index("CAN") + 1,
        "corr_delta_b": float(np.corrcoef(b[fin], ds[fin])[0, 1]),
        "corr_delta_exposure": float(np.corrcoef(exposure[fin], ds[fin])[0, 1]),
        "partial_corr_exposure_given_b": partial_corr(ds[fin], exposure[fin], b[fin]),
    }


# ===================== SURFACE HELPERS =====================
def bilateral_surface(ge):
    """U_US and U_CN over the (t_US, t_CN) tariff grid."""
    n = len(SURF_GRID)
    U_us = np.zeros((n, n)); U_cn = np.zeros((n, n))
    for a, tu in enumerate(SURF_GRID):
        for c, tc in enumerate(SURF_GRID):
            sol = ge.solve(np.array([tu, tc]))
            U_us[a, c] = sol["U"][0]; U_cn[a, c] = sol["U"][1]
    return U_us, U_cn


def network_slices(gen, nash):
    """Per-country own real income vs own tariff, with others at free trade (cooperative
       reference) and at Nash (punishment reference)."""
    n = len(SURF_GRID); N = gen.N; free = np.zeros(N)
    slice_free = np.zeros((N, n)); slice_nash = np.zeros((N, n))
    for i in range(N):
        for g, ti in enumerate(SURF_GRID):
            tf = free.copy(); tf[i] = ti
            slice_free[i, g] = gen.utility(tf)[i]
            tn = nash.copy(); tn[i] = ti
            slice_nash[i, g] = gen.utility(tn)[i]
    return slice_free, slice_nash


# ===================== PER-SIGMA DRIVER =====================
def run_sigma(sigma, stage2, cache):
    """All four routes plus the cached surfaces at one CES elasticity. The expensive N=30
       Nash is solved once here and shared by route 4 and the slice cache."""
    iso = stage2["iso"]; tag = f"s{sigma:g}"

    # --- bilateral US-China ---
    stage_pay, ge_bilat = bilateral_stage_payoffs(sigma)
    us = stage_pay["US"]
    G_us = us["T"] - us["R"]; L_us = us["R"] - us["P"]
    U_us, U_cn = bilateral_surface(ge_bilat)
    cache[f"bilat_U_US_{tag}"] = U_us
    cache[f"bilat_U_CN_{tag}"] = U_cn

    lola = route_lola(stage_pay)
    evo = route_evolution(us["R"], us["S"], us["T"], us["P"])
    tol = route_tolerance(G_us, L_us)

    # --- N=30 network (one Nash) ---
    gen, b, exp = network_model(sigma, stage2)
    nash = gen.nash()
    net = route_network(gen, b, exp, iso, nash=nash)
    sf, sn = network_slices(gen, nash)
    cache[f"net_slice_free_{tag}"] = sf
    cache[f"net_slice_nash_{tag}"] = sn
    cache[f"net_nash_tariff_{tag}"] = nash

    return {
        "sigma": sigma,
        "bilateral_stage_game": stage_pay,
        "bilateral_GL": {"G_US": G_us, "L_US": L_us,
                         "delta_star_US": G_us / (G_us + L_us) if L_us > 0 else None},
        "route1_lola": lola,
        "route2_evolution": evo,
        "route3_tolerance": tol,
        "route4_network": net,
    }


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    stage2 = json.load(open(STAGE2))

    print("Evaluating four structural routes on exact CES GE payoffs (NumPy + SciPy only) ...")
    cache = {"surf_grid": SURF_GRID, "sigma_grid": np.array(SIGMA_GRID)}
    results = {"model": "Armington CES general equilibrium (ge_frontier.py, V2)",
               "provenance": "NEWLY CONSTRUCTED in branch v2-ge-extension; not a pre-existing script.",
               "dependencies": "numpy + scipy.optimize only",
               "sigma_primary": SIGMA_PRIMARY, "sigma_grid": SIGMA_GRID,
               "calibration": {"b_US": B_US, "b_CN": B_CN,
                               "W_reconstruction": "gravity from stage2 b/exposure (off-diag ~ b_j, row mass = exposure)"},
               "sensitivity": {}}
    for sigma in SIGMA_GRID:
        print(f"  sigma = {sigma} ...", flush=True)
        res = run_sigma(sigma, stage2, cache)
        results["sensitivity"][f"sigma_{sigma:g}"] = res
        if sigma == SIGMA_PRIMARY:
            results["primary"] = res

    np.savez_compressed(SURFACES_NPZ, **cache)
    print(f"  cached GE surfaces -> {SURFACES_NPZ}")
    json.dump(results, open(ROUTES_JSON, "w"), indent=1)
    print(f"  saved routes       -> {ROUTES_JSON}")

    p = results["primary"]
    net = p["route4_network"]
    print("\n==== V2 GE HEADLINES (sigma = {}) ====".format(SIGMA_PRIMARY))
    print(f"  bilateral GE Nash tariff (US, CN): {p['bilateral_stage_game']['nash_tariff']}")
    print(f"  bilateral GE delta*_US = {p['bilateral_GL']['delta_star_US']:.3f}")
    print(f"  R1 LOLA: naive joint {p['route1_lola']['naive_joint_mean']:+.4f} -> "
          f"LOLA joint {p['route1_lola']['lola_joint_mean']:+.4f}  crosses_gap={p['route1_lola']['lola_crosses_gap']}")
    print(f"  R2 evolution: basin(coop) @delta=0.9 = {p['route2_evolution']['basin_cooperation']:.3f}; "
          f"delta_inv = {p['route2_evolution']['delta_inv_tft_vs_alld']:.3f}")
    print(f"  R3 tolerance: peak sigma_noise = {p['route3_tolerance']['peak_tolerance_sigma']:.4f}; "
          f"delta*(0.20) = {p['route3_tolerance']['delta_star_by_sigma_noise']['0.20']:.3f}")
    print(f"  R4 network: binding {net['binding_country']} (delta*={net['binding_delta_star']:.3f}); "
          f"MEX rank {net['MEX_rank']}, CAN rank {net['CAN_rank']}; hegemons={net['hegemons'] or 'none'}")
    print(f"           corr(delta*,b)={net['corr_delta_b']:.3f}, "
          f"partial-corr(delta*,exposure|b)={net['partial_corr_exposure_given_b']:.3f}")


if __name__ == "__main__":
    main()
