"""
estimate_panel_v2.py — the estimation stage.

IMPORTANT: if you still have your original run_panel.py, USE THAT INSTEAD.
Comparing a 13-dyad result from one estimator against a 40-dyad result from a
different implementation confounds "more data" with "different code". This file
is a reimplementation for the case where the original is lost; if you use it,
run it on the OLD 13-dyad panel first and confirm it reproduces ATT +0.270,
TWFE +0.243 and placebo -0.049. If it does not, the two panels are not
comparable and that must be said in the paper.

What is computed
  1. Cohort event study  : for each treated cohort g and event time k, compare
                           cohort g against dyads NOT YET treated at that time.
                           Aggregating over cohorts gives ATT(k).
  2. ATT                 : mean of ATT(k) over k >= 0.
  3. Pre-trend           : mean of ATT(k) over k < 0. Should be ~0.
  4. Placebo             : assign false event years to never-treated dyads and
                           re-run. Should be ~0.
  5. Leave-one-out       : drop each treated dyad in turn, re-run.
  6. Naive TWFE          : two-way fixed effects for contrast. Expected to be
                           BIASED under staggered timing; the gap is the point.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PANEL = ROOT / "data/processed/panel_v2.csv"
RNG = np.random.default_rng(20260827)


def cohort_att(df, kmin=-5, kmax=5):
    """Callaway/Sant'Anna-style: each cohort against not-yet-treated units."""
    out = {}
    cohorts = sorted(df.loc[df.treated == 1, "cohort"].dropna().unique())
    for k in range(kmin, kmax + 1):
        num, den = 0.0, 0
        for g in cohorts:
            t = int(g) + k
            treat = df[(df.cohort == g) & (df.year == t)]["outcome"].dropna()
            base = df[(df.cohort == g) & (df.year == int(g) - 1)]["outcome"].dropna()
            # controls: never-treated, or treated later than t
            ctl_mask = ((df.treated == 0) | (df.cohort > t)) & (df.year == t)
            ctlb_mask = ((df.treated == 0) | (df.cohort > t)) & (df.year == int(g) - 1)
            ctl = df[ctl_mask]["outcome"].dropna()
            ctlb = df[ctlb_mask]["outcome"].dropna()
            if len(treat) and len(base) and len(ctl) and len(ctlb):
                did = (treat.mean() - base.mean()) - (ctl.mean() - ctlb.mean())
                num += did * len(treat); den += len(treat)
        out[k] = num / den if den else np.nan
    return pd.Series(out)


def naive_twfe(df):
    """Two-way fixed effects with a single post-treatment dummy."""
    d = df.dropna(subset=["outcome"]).copy()
    d["post"] = ((d.treated == 1) & (d.year >= d.treat_year)).astype(float)
    y = d["outcome"].values
    X = [d["post"].values]
    for col in ("dyad_id", "year"):
        dm = pd.get_dummies(d[col], drop_first=True).values.astype(float)
        X.append(dm.T)
    M = np.column_stack([X[0]] + [x for x in np.vstack([X[1], X[2]])])
    M = np.column_stack([np.ones(len(y)), M])
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return float(beta[1])


def run(df, label=""):
    e = cohort_att(df)
    att = e[e.index >= 0].mean()
    pre = e[e.index < 0].mean()
    print(f"  {label:24} ATT {att:+.3f}   pre-trend {pre:+.3f}")
    return att, pre, e


def main():
    df = pd.read_csv(PANEL)
    print(f"panel: {df.dyad_id.nunique()} dyads "
          f"({df.loc[df.treated==1,'dyad_id'].nunique()} treated), {len(df):,} dyad-years\n")

    print("MAIN")
    att, pre, ev = run(df, "full sample")
    print("\n  event-time path:")
    for k, v in ev.items():
        print(f"    k={k:+d}: {v:+.3f}")

    print("\nCONTRAST")
    print(f"  {'naive TWFE':24} {naive_twfe(df):+.3f}   (expected below the clean estimate)")

    print("\nPLACEBO  (false event years on never-treated dyads)")
    ctl_ids = sorted(df.loc[df.treated == 0, "dyad_id"].unique())
    fake = df[df.treated == 0].copy()
    assign = {d: RNG.choice([2018, 2019, 2020]) for d in ctl_ids}
    fake["treated"] = 1
    fake["treat_year"] = fake["dyad_id"].map(assign)
    fake["cohort"] = fake["treat_year"]
    fake["event_time"] = fake["year"] - fake["treat_year"]
    p_att, _, _ = run(fake, "placebo")

    print("\nLEAVE-ONE-DYAD-OUT")
    vals = []
    for d in sorted(df.loc[df.treated == 1, "dyad_id"].unique()):
        a, _, _ = run(df[df.dyad_id != d], f"without {d}")
        vals.append(a)
    print(f"  range [{min(vals):+.3f}, {max(vals):+.3f}]")

    print("\nCONTROL-GROUP ROBUSTNESS")
    ctl = df[df.treated == 0]["dyad_id"].unique()
    for n in (9, len(ctl) // 2, len(ctl)):
        keep = set(ctl[:n]) | set(df.loc[df.treated == 1, "dyad_id"])
        a, _, _ = run(df[df.dyad_id.isin(keep)], f"{n} controls")

    print("\nCOMPARISON TO THE PUBLISHED 13-DYAD PANEL")
    print("  published: ATT +0.270, pre-trend -0.003, placebo -0.049, TWFE +0.243")
    print(f"  this run : ATT {att:+.3f}, pre-trend {pre:+.3f}, placebo {p_att:+.3f}, "
          f"TWFE {naive_twfe(df):+.3f}")


if __name__ == "__main__":
    main()
