"""
check_panel_v2.py — run BEFORE estimating anything.

These checks catch the failure modes that would silently produce a wrong answer.
The most important is the US-China 2019 reproduction: if the rebuilt panel does not
recover roughly the 0.33 already reported in the paper, the reconstruction differs
from the original build and estimating on it would be meaningless.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data/processed/panel_v2.csv")
fails = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        fails.append(name)

print("PANEL SANITY CHECKS\n")

# 1. structure
dupes = df.duplicated(["dyad_id", "year"]).sum()
check("no duplicate dyad-year rows", dupes == 0, f"{dupes} duplicates")

# 2. treated set is exactly the four pre-specified dyads
tre = set(df.loc[df.treated == 1, "dyad_id"])
want = {"USA_CHN", "CHN_USA", "USA_IND", "CHN_AUS"}
check("treated set matches the prespec", tre == want, f"{sorted(tre)}")

# 3. controls sit near zero -- the core test of the cooperative baseline
ctl = df[df.treated == 0]["outcome"].dropna()
check("control outcomes centred near zero", abs(ctl.mean()) < 0.10,
      f"mean {ctl.mean():+.3f}, sd {ctl.std():.3f}, n {len(ctl):,}")

# 4. THE REPRODUCTION TEST
row = df.query("dyad_id == 'USA_CHN' and year == 2019")["outcome"]
if len(row):
    v = float(row.iloc[0])
    check("US-China 2019 reproduces the published ~0.33", abs(v - 0.33) < 0.06,
          f"got {v:.3f}")
else:
    check("US-China 2019 present", False, "row missing")

# 5. outcome in a sane range
bad = df["outcome"].dropna()
out_of_range = ((bad < -0.5) | (bad > 2.0)).sum()
check("outcomes within [-0.5, 2.0]", out_of_range == 0,
      f"{out_of_range} rows outside")

# 6. pre-period depth for the earliest cohort
pre = df[(df.treated == 1) & (df.event_time < 0)].groupby("dyad_id")["year"].count()
check("every treated dyad has >= 10 pre-periods", (pre >= 10).all(),
      dict(pre))

# 7. balance
print(f"\n  dyads {df.dyad_id.nunique()}   dyad-years {len(df):,}   "
      f"years {int(df.year.min())}-{int(df.year.max())}")
print(f"  treated {df.treated.sum():,} rows / controls {(df.treated==0).sum():,} rows")
miss = df["outcome"].isna().mean()
print(f"  outcome missing: {miss:.1%}")


# 8. rate-source composition -- Gate 1
if "rate_source" in df.columns:
    share = (df.rate_source == "AHS").mean()
    print(f"\n  AHS coverage: {share:.1%}  (MFN fallback {1-share:.1%})")
    if share < 0.50:
        print("  GATE 1: AHS covers under half the dyad-years. Tell Claude before")
        print("          proceeding; the primary indicator may need to be MFN,")
        print("          and the paper must say so.")

print("\n" + ("ALL CHECKS PASSED — safe to estimate" if not fails
              else f"{len(fails)} CHECK(S) FAILED — do not estimate: {fails}"))
