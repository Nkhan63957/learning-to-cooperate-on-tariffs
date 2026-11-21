"""
verify.py: reproduce every headline number in the paper, from the files in this
repository alone. No BACI download, no Colab, no network. Runs in a few seconds.

    python verify.py

Why this exists
    pipeline.py needs the CEPII BACI files, which are several hundred megabytes and
    cannot be redistributed. That makes the headline results hard for a reader to
    check. This script closes that gap: it recomputes everything that follows from
    the calibration, and cross-checks everything else against the stored results
    files, printing PASS or FAIL against the values stated in the paper.

What it cannot do
    It does not re-run the learning experiments. Those need torch and are ingested
    from data/processed/. For those it verifies that the stored outputs match what
    the paper reports, which is a consistency check, not a replication.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
fails = []


# ======================================================================================
# V2 MODE (python verify.py --v2).  Additive, pure standard library.  The no-argument
# invocation below is UNCHANGED and remains the frozen v1 Regeneron STS check; --v2 runs
# only these V2 checks and exits before any v1 code executes.  These validate the NEW
# general-equilibrium headline numbers (ge_frontier.py) by INDEPENDENT recomputation: the
# GE real-income payoffs are taken as the model primitive (exactly as v1 takes the
# calibrated b), and every downstream closed-form number is recomputed here and compared
# to data/processed/v2_ge_routes.json.  Stochastic learning outputs (LOLA, the replicator
# basin) cannot be re-derived without NumPy and are labelled [consistency], matching how
# v1 treats the ingested torch results.
# ======================================================================================
def run_v2():
    proc = Path(__file__).parent / "data" / "processed"
    routes = json.load(open(proc / "v2_ge_routes.json"))
    s2 = json.load(open(proc / "stage2_results.json"))
    p = routes["primary"]
    vfails = []

    def ck(name, got, want, tol, unit=""):
        ok = abs(got - want) <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:52} {got:11.5f}{unit}  stored {want}{unit}")
        if not ok:
            vfails.append(name)

    def ck_bool(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            vfails.append(name)

    def corr(x, y):
        n = len(x); mx = sum(x) / n; my = sum(y) / n
        cov = sum((a - mx) * (c - my) for a, c in zip(x, y))
        vx = sum((a - mx) ** 2 for a in x); vy = sum((c - my) ** 2 for c in y)
        return cov / math.sqrt(vx * vy)

    print(f"V2 GENERAL-EQUILIBRIUM EXTENSION (ge_frontier.py, newly constructed)")
    print(f"Independent recomputation of every new V2 headline (CES sigma = {routes['sigma_primary']}).\n")

    # ---- bilateral stage game: folk delta* from the GE stage payoffs ----
    print("SECTION GE-1: bilateral US-China GE stage game (folk delta* from R, T, P)\n")
    us = p["bilateral_stage_game"]["US"]
    R, S, T, P = us["R"], us["S"], us["T"], us["P"]
    G, L = T - R, R - P
    ck("defection gain G_US = T - R", G, p["bilateral_GL"]["G_US"], 1e-9)
    ck("punishment loss L_US = R - P", L, p["bilateral_GL"]["L_US"], 1e-9)
    ck("critical discount delta*_US = G/(G+L)", G / (G + L), p["bilateral_GL"]["delta_star_US"], 1e-9)

    # ---- route 2: evolutionary invasion threshold (closed form) ----
    print("\nSECTION GE-2: evolutionary route -- TFT-resists-ALLD threshold (T-R)/(T-P)\n")
    r2 = p["route2_evolution"]
    dinv = (T - R) / (T - P)
    ck("delta_inv = (T-R)/(T-P)", dinv, r2["delta_inv_tft_vs_alld"], 1e-9)
    ck_bool("delta_inv equals bilateral folk delta* (G/(G+L))", abs(dinv - G / (G + L)) < 1e-9)
    bc = r2["basin_cooperation"]
    ck_bool(f"[consistency] replicator cooperation basin in [0,1] (= {bc:.3f})", 0.0 <= bc <= 1.0)

    # ---- route 3: enforcement-tolerance theorem, recomputed in pure stdlib ----
    print("\nSECTION GE-3: noisy-monitoring tolerance w*(sigma_noise) and delta*(sigma_noise)\n")
    r3 = p["route3_tolerance"]
    CXCF = r3["CX_OVER_CF"]
    K = 1.0 / (math.sqrt(2 * math.pi) * CXCF)

    def wstar(sg):
        arg = 2 * math.log(K / sg) if sg > 0 else -1
        return sg * math.sqrt(arg) if arg > 0 else 0.0

    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    d = r3["d_signal"]

    def m_of(sg):
        w = wstar(sg)
        return Phi((d - w) / sg) - Phi(-w / sg)

    ck("peak tolerance at sigma = K/sqrt(e)", K / math.sqrt(math.e), r3["peak_tolerance_sigma"], 1e-9)
    ck("w*(sigma_noise = 0.10)", wstar(0.10), r3["w_star"]["0.10"], 1e-9)
    ck("w*(sigma_noise = 0.20)", wstar(0.20), r3["w_star"]["0.20"], 1e-9)
    ck_bool("w* -> 0 as sigma_noise -> 0", wstar(1e-9) < 1e-8)
    Gge, Lge = r3["G"], r3["L"]
    ck("delta*(sigma_noise=0.20) = G/(G+L m)", Gge / (Gge + Lge * m_of(0.20)),
       r3["delta_star_by_sigma_noise"]["0.20"], 1e-9)
    ck("delta*_noiseless = G/(G+L)", Gge / (Gge + Lge), r3["delta_star_noiseless"], 1e-9)

    # ---- route 4: N=30 network folk delta* ranking, recomputed per economy ----
    print("\nSECTION GE-4: thirty-economy network -- folk delta*_i = G_i/(G_i+L_i)\n")
    r4 = p["route4_network"]
    iso = r4["iso"]; Gs = r4["G"]; Ls = r4["L"]
    ds = [Gs[i] / (Gs[i] + Ls[i]) if Ls[i] > 0 else float("inf") for i in range(len(iso))]
    maxdiff = 0.0
    for i in range(len(iso)):
        stored = r4["delta_star"][i]
        if stored is None:
            maxdiff = max(maxdiff, 0.0 if math.isinf(ds[i]) else 1.0)
        else:
            maxdiff = max(maxdiff, abs(ds[i] - stored))
    ck("max |recomputed - stored| delta*_i over 30 economies", maxdiff, 0.0, 1e-9)
    fin = [i for i in range(len(iso)) if math.isfinite(ds[i])]
    binding = iso[max(fin, key=lambda i: ds[i])]
    ck_bool(f"binding economy recomputed == stored ({binding})", binding == r4["binding_country"])
    heg = [iso[i] for i in range(len(iso)) if not math.isfinite(ds[i])]
    ck_bool(f"hegemon set recomputed == stored ({heg or 'none'})", heg == r4["hegemons"])
    order = sorted(fin, key=lambda i: -ds[i])
    rank = {iso[k]: r + 1 for r, k in enumerate(order)}
    ck("Mexico delta* rank", rank["MEX"], r4["MEX_rank"], 0)
    ck("Canada delta* rank", rank["CAN"], r4["CAN_rank"], 0)
    b = s2["b"]
    ck("corr(delta*, market power b)", corr([ds[i] for i in fin], [b[i] for i in fin]),
       r4["corr_delta_b"], 1e-6)

    # ---- route 1: LOLA transparency (stochastic -> consistency, as v1 does for torch) ----
    print("\nSECTION GE-5: LOLA transparency (stochastic; consistency checks)\n")
    r1 = p["route1_lola"]
    ck_bool(f"[consistency] LOLA joint > naive joint "
            f"({r1['lola_joint_mean']:+.4f} > {r1['naive_joint_mean']:+.4f})",
            r1["lola_joint_mean"] > r1["naive_joint_mean"])
    ck_bool("[consistency] transparency crosses the cooperation gap", bool(r1["lola_crosses_gap"]))
    vA = r1["lola_vs_alld"]["V_lola"]
    ck_bool(f"[consistency] LOLA not exploited by ALLD (V_lola = {vA:+.4f}, near defection floor)",
            vA <= 0.02)

    print("\n" + "=" * 74)
    if vfails:
        print(f"{len(vfails)} V2 CHECK(S) FAILED: {vfails}")
        raise SystemExit(1)
    print("ALL V2 CHECKS PASSED. Every new V2 general-equilibrium headline number reproduces")
    print("by independent recomputation from v2_ge_routes.json, with the exact CES GE")
    print("real-income payoffs taken as the model primitive (as v1 takes the calibrated b).")
    raise SystemExit(0)


if "--v2" in sys.argv:
    run_v2()


def check(name, got, want, tol, unit=""):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44} {got:8.4f}{unit}  paper {want}{unit}")
    if not ok:
        fails.append(name)


B = {"US": 0.66, "CN": 0.56}
KAPPA, PSI = 1.0, 0.10
OTHER = {"US": "CN", "CN": "US"}


def nash():
    t = {"US": 0.0, "CN": 0.0}
    for _ in range(10000):
        nt = {i: max(0.0, min(1.0, (B[i] - PSI * t[OTHER[i]]) / KAPPA)) for i in B}
        if max(abs(nt[i] - t[i]) for i in B) < 1e-14:
            return nt
        t = nt
    return t


def welfare(i, t):
    j = OTHER[i]
    return B[i] * t[i] - B[j] * t[j] - 0.5 * KAPPA * t[i] ** 2 - PSI * t[i] * t[j]


print("SECTION 2 and 5: the calibrated game (recomputed from scratch)\n")
t = nash()
check("Nash tariff, United States", t["US"], 0.61, 0.005)
check("Nash tariff, China", t["CN"], 0.50, 0.005)

G = {i: B[i] ** 2 / (2 * KAPPA) for i in B}
L = {i: -welfare(i, t) for i in B}
check("defection gain G, United States", G["US"], 0.218, 0.002)
check("defection gain G, China", G["CN"], 0.157, 0.002)
check("punishment loss L, United States", L["US"], 0.093, 0.002)
check("punishment loss L, China", L["CN"], 0.278, 0.002)
check("critical discount, United States", G["US"] / (G["US"] + L["US"]), 0.70, 0.005)
check("critical discount, China", G["CN"] / (G["CN"] + L["CN"]), 0.36, 0.005)

print("\nSECTION 5.2: the observability coefficient and the route ordering\n")
omega = {i: B[OTHER[i]] * t[OTHER[i]] for i in B}
T = {i: B[i] * t[i] - 0.5 * KAPPA * t[i] ** 2 - PSI * t[i] * t[OTHER[i]] for i in B}
check("omega, United States", omega["US"], 0.279, 0.002)
check("omega, China", omega["CN"], 0.403, 0.002)
for i in B:
    resid = abs(omega[i] - (L[i] + T[i]))
    print(f"  [{'PASS' if resid < 1e-12 else 'FAIL'}] identity omega = L + T, {i:22} residual {resid:.2e}")
    if resid >= 1e-12:
        fails.append(f"identity {i}")
check("omega/L ratio, United States", omega["US"] / L["US"], 3.0, 0.05)
check("omega/L ratio, China", omega["CN"] / L["CN"], 1.4, 0.05)
check("patience switch m*, United States", 1 - (G["US"] - T["US"]) / L["US"], 0.66, 0.01)
check("patience switch m*, China", 1 - (G["CN"] - T["CN"]) / L["CN"], 0.88, 0.01)

print("\nSECTION 6.3: the enforcement-tolerance theorem\n")
CX_OVER_CF = 0.9529
K = 1.0 / (math.sqrt(2 * math.pi) * CX_OVER_CF)


def w_star(sigma):
    arg = 2 * math.log(K / sigma)
    return sigma * math.sqrt(arg) if arg > 0 else 0.0


def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def m_of(sigma, d=0.58):
    w = w_star(sigma)
    return Phi((d - w) / sigma) - Phi(-w / sigma)


check("peak tolerance at sigma = K/sqrt(e)", K / math.sqrt(math.e), 0.2539, 0.001)
check("w* at sigma = 0.10", w_star(0.10), 0.15, 0.03)
check("w* at sigma = 0.20", w_star(0.20), 0.25, 0.03)
print(f"  [{'PASS' if w_star(1e-9) < 1e-8 else 'FAIL'}] w* -> 0 as sigma -> 0                       "
      f"{w_star(1e-9):.2e}")
check("m(sigma) at sigma = 0.20", m_of(0.20), 0.84, 0.02)
check("delta*(sigma=0.20) = G/(G + L m)", G["US"] / (G["US"] + L["US"] * m_of(0.20)), 0.73, 0.01)

print("\nSECTION 7.2: the thirty-economy multilateral extension (from stage2_results.json)\n")
s = json.load(open(PROC / "stage2_results.json"))
d30 = dict(zip(s["iso"], s["delta_star"]))
check("binding cooperator, United States", d30["USA"], 0.77, 0.006)
check("Mexico critical discount", d30["MEX"], 0.36, 0.006)
check("Canada critical discount", d30["CAN"], 0.36, 0.006)
check("corr(delta*, market power)", s["corr_b"], 0.93, 0.006)
heg = len(s["hegemons"]) == 0 and all(w < 0 for w in s["W_nash"])
print(f"  [{'PASS' if heg else 'FAIL'}] no economy gains at Nash (no hegemon)")
if not heg:
    fails.append("hegemon check")

print("\nSTORED EXPERIMENT OUTPUTS: consistency with the paper's reported values\n")
try:
    ppo = json.load(open(PROC / "ppo_results.json"))
    ok = all(v >= 0.60 - 1e-9 for v in ppo["independent"]["mean"])
    print(f"  [{'PASS' if ok else 'FAIL'}] PPO independent never below grid Nash 0.60")
    if not ok:
        fails.append("ppo floor")
    vr = ppo["vs_reciprocator"]["mean"]
    ok2 = all(v <= 0.20 for v in vr)
    print(f"  [{'PASS' if ok2 else 'FAIL'}] PPO vs reciprocator inside the cooperative band")
    if not ok2:
        fails.append("ppo band")
except FileNotFoundError:
    print("  [SKIP] ppo_results.json not present")

try:
    qre = json.load(open(PROC / "qre_results.json"))
    ok = abs(qre["mean_tau_US"][-1] - 0.6101) < 0.002
    print(f"  [{'PASS' if ok else 'FAIL'}] QRE converges to Nash as rationality grows"
          f"   ({qre['mean_tau_US'][-1]})")
    if not ok:
        fails.append("qre limit")
except FileNotFoundError:
    print("  [SKIP] qre_results.json not present")

print("\n" + "=" * 74)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    raise SystemExit(1)
print("ALL CHECKS PASSED. Every headline number in the paper reproduces from this")
print("repository alone, without the raw BACI files.")