"""
build_panel_v2.py — assemble the expanded dyad-year panel.

Governed by PRESPEC_panel_v2.md. Do not change any parameter in the CONFIG block
after the tariff data have been seen; the pre-specification fixes them.

Inputs
    data/raw/wits_bilateral_2000_2025.csv   your WITS TRAINS pull
    data/processed/stage2_results.json      calibration (nash tariff per economy)
    fta_exclusions.csv                      treaty exclusions, VERIFY before use

Output
    data/processed/panel_v2.csv             one row per dyad-year
    build_log_v2.txt                        filter counts at every step

The WITS pull is expected to have at least these columns, case-insensitive; edit
COLMAP below if your export names them differently.
    reporter iso3, partner iso3, year, tariff rate
"""
import json, sys, itertools
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------- CONFIG (frozen by the prespec) -----------------
YEAR_MIN, YEAR_MAX      = 2000, 2025
TRADE_SHARE_FLOOR       = 0.001     # 0.1 percent of importer total imports
MIN_YEARS_COVERAGE      = 15        # of 26
MIN_CONTROL_DYADS       = 25        # below this, the floor may be loosened ONCE

TREATED = {                          # (importer, partner): first treated year
    ("USA", "CHN"): 2018,
    ("CHN", "USA"): 2018,
    ("USA", "IND"): 2019,
    ("CHN", "AUS"): 2020,
}

COLMAP = {          # substring to look for -> canonical name
    "reporter": "importer_iso",
    "partner":  "partner_iso",
    "year":     "year",
    "tariff":   "tariff_rate",
}
# The single WITS pull carries BOTH "Effectively Applied" (AHS) and "MFN Applied"
# rows, distinguished by a duty-type column. AHS is preferred; MFN is the fallback
# where AHS is missing. Set DUTYCOL to None if your export has no such column.
DUTYCOL_HINT = "duty"        # substring identifying the duty-type column
AHS_HINT     = "effectively" # substring identifying AHS rows
MFN_HINT     = "mfn"         # substring identifying MFN rows
# ------------------------------------------------------------------------------

ROOT = Path(__file__).parent
LOG  = []

def log(msg):
    print(msg)
    LOG.append(str(msg))


def load_calibration():
    s = json.load(open(ROOT / "data/processed/stage2_results.json"))
    return dict(zip(s["iso"], s["nash"])), list(s["iso"])


def load_wits():
    p = ROOT / "data/raw/wits_bilateral_2000_2025.csv"
    if not p.exists():
        sys.exit(f"missing {p}\nPull WITS TRAINS first. See PRESPEC_panel_v2.md.")
    df = pd.read_csv(p)
    lower = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for want, canon in COLMAP.items():
        hit = next((orig for low, orig in lower.items() if want in low), None)
        if hit is None:
            sys.exit(f"could not find a column matching {want!r} in {list(df.columns)}\n"
                     f"Edit COLMAP at the top of this script.")
        ren[hit] = canon
    raw = df.copy(); raw_cols = list(df.columns)
    df = df.rename(columns=ren)[list(COLMAP.values())].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["tariff_rate"] = pd.to_numeric(df["tariff_rate"], errors="coerce")
    dcol = next((c for c in raw_cols if DUTYCOL_HINT in c.lower()), None)
    if dcol is not None:
        df["_duty"] = raw[dcol].astype(str).str.lower()
        df["_pref"] = np.where(df["_duty"].str.contains(AHS_HINT), 0,
                       np.where(df["_duty"].str.contains(MFN_HINT), 1, 2))
    else:
        df["_pref"] = 0
    df = df.dropna(subset=["year", "tariff_rate"])
    df = df[(df.year >= YEAR_MIN) & (df.year <= YEAR_MAX)]
    # keep AHS where present, else MFN, one row per dyad-year
    df = (df.sort_values(["importer_iso", "partner_iso", "year", "_pref"])
            .drop_duplicates(["importer_iso", "partner_iso", "year"], keep="first"))
    df["rate_source"] = np.where(df["_pref"] == 0, "AHS", "MFN")
    return df.drop(columns=[c for c in ("_duty", "_pref") if c in df.columns])


def expand_fta_exclusions(isos):
    """Read fta_exclusions.csv and expand the group tokens into ordered pairs."""
    EU  = ["DEU", "FRA", "ITA", "ESP", "NLD", "POL", "SWE", "BEL"]
    GBR = ["GBR"]
    ASEAN = ["IDN", "THA", "VNM", "MYS"]
    RCEP  = []   # DECISION: RCEP is not an exclusion. See fta_exclusions.csv footer.
    groups = {"EU_INTERNAL": EU, "EU_MEMBER": EU + GBR, "ASEAN": ASEAN, "RCEP": RCEP}

    excl = set()
    raw = (ROOT / "fta_exclusions.csv").read_text().splitlines()
    for line in raw[1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        a, b = [x.strip() for x in line.split(",")[:2]]
        A = groups.get(a, [a]); B = groups.get(b, [b])
        for x, y in itertools.product(A, B):
            if x != y and x in isos and y in isos:
                excl.add((x, y)); excl.add((y, x))
    # EU internal and RCEP internal are all-pairs among members
    for grp in (EU + GBR, ASEAN):
        for x, y in itertools.permutations(grp, 2):
            if x in isos and y in isos:
                excl.add((x, y))
    return excl


def main():
    nash, isos = load_calibration()
    log(f"universe: {len(isos)} economies, {len(isos)*(len(isos)-1)} ordered pairs")

    wits = load_wits()
    log(f"WITS rows in window: {len(wits):,}")

    pairs = {(i, j) for i, j in itertools.permutations(isos, 2)}

    # filter 1: treated events other than our four are excluded by construction,
    # since any dyad appearing in fta_exclusions or in TREATED is handled below.
    # filter 2: treaty exclusions
    excl = expand_fta_exclusions(set(isos))
    pairs_2 = {p for p in pairs if p not in excl}
    log(f"after treaty exclusions:            {len(pairs_2):4d}  (dropped {len(pairs)-len(pairs_2)})")

    # filter 3: trade-share floor  -- requires bilateral flows
    flows_p = ROOT / "data/processed/bilateral_import_shares.csv"
    if flows_p.exists():
        fl = pd.read_csv(flows_p)   # importer_iso, partner_iso, import_share
        keep = {(r.importer_iso, r.partner_iso) for r in fl.itertuples()
                if r.import_share >= TRADE_SHARE_FLOOR}
        pairs_3 = {p for p in pairs_2 if p in keep}
        log(f"after trade-share floor {TRADE_SHARE_FLOOR:.3%}:     {len(pairs_3):4d}  (dropped {len(pairs_2)-len(pairs_3)})")
    else:
        pairs_3 = pairs_2
        log(f"trade-share floor SKIPPED: {flows_p.name} not found.")
        log("   Build it from BACI (importer_iso, partner_iso, import_share) and rerun.")

    # filter 4: coverage
    cov = (wits.groupby(["importer_iso", "partner_iso"])["year"]
                .nunique().rename("n_years").reset_index())
    ok = {(r.importer_iso, r.partner_iso) for r in cov.itertuples()
          if r.n_years >= MIN_YEARS_COVERAGE}
    pairs_4 = {p for p in pairs_3 if p in ok}
    log(f"after coverage >= {MIN_YEARS_COVERAGE} years:          {len(pairs_4):4d}  (dropped {len(pairs_3)-len(pairs_4)})")

    controls = sorted(pairs_4 - set(TREATED))
    log(f"\nTREATED dyads:  {len(TREATED)}")
    log(f"CONTROL dyads:  {len(controls)}")
    if len(controls) < MIN_CONTROL_DYADS:
        log(f"*** below the pre-specified minimum of {MIN_CONTROL_DYADS}. "
            f"The prespec permits loosening the trade-share floor ONCE. Do that "
            f"explicitly, record it, and rerun.")

    # assemble
    keep = set(controls) | set(TREATED)
    df = wits[[(r.importer_iso, r.partner_iso) in keep
               for r in wits.itertuples()]].copy()

    df["tau_nash"] = df["importer_iso"].map(nash) * 100      # calibration is a fraction
    base = (df[df.year < 2018]
              .groupby(["importer_iso", "partner_iso"])["tariff_rate"]
              .median().rename("tau_coop").reset_index())
    df = df.merge(base, on=["importer_iso", "partner_iso"], how="left")

    denom = df["tau_nash"] - df["tau_coop"]
    df["outcome"] = np.where(denom.abs() > 1e-9,
                             (df["tariff_rate"] - df["tau_coop"]) / denom, np.nan)

    df["treated"]    = [1 if (r.importer_iso, r.partner_iso) in TREATED else 0
                        for r in df.itertuples()]
    df["treat_year"] = [TREATED.get((r.importer_iso, r.partner_iso), np.nan)
                        for r in df.itertuples()]
    df["event_time"] = df["year"] - df["treat_year"]
    df["cohort"]     = df["treat_year"]
    df["dyad_id"]    = df["importer_iso"] + "_" + df["partner_iso"]

    cols = ["dyad_id", "importer_iso", "partner_iso", "year", "tariff_rate",
            "rate_source", "tau_coop", "tau_nash", "outcome", "treated",
            "treat_year", "event_time", "cohort"]
    out = ROOT / "data/processed/panel_v2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].sort_values(["dyad_id", "year"]).to_csv(out, index=False)

    log(f"\nwrote {out}")
    log(f"  dyads      : {df.dyad_id.nunique()}")
    log(f"  dyad-years : {len(df):,}")
    log(f"  years      : {int(df.year.min())}-{int(df.year.max())}")
    (ROOT / "build_log_v2.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
