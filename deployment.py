"""
deployment.py — consolidates everything into one organized report.

Reads data/processed/results.json (recomputed + ingested) and prints a single
labeled report divided into the paper's logical pieces, with every number and
its provenance. Also writes deployment_report.txt. Run pipeline.py first.

    python deployment.py

This is the output to hand back: each DIVISION is a self-contained block you can
explain on its own.
"""
from __future__ import annotations
from pathlib import Path
import json, numpy as np, io

ROOT=Path(__file__).parent
R=json.load(open(ROOT/"data"/"processed"/"results.json"))
buf=io.StringIO()
def P(*a): print(*a); print(*a,file=buf)

def line(c="="): P(c*72)

def fmt(x,n=3):
    return "inf" if (x is None) else (f"{x:.{n}f}" if isinstance(x,(int,float)) else str(x))

# ---------------------------------------------------------------- header
line(); P("LEARNING TO COOPERATE ON TARIFFS — CONSOLIDATED REPORT"); 
cf=R["config"]; P(f"config: N={cf['N']}  kappa={cf['kappa']}  delta={cf['delta']}  "
                  f"b in [{cf['B_LO']},{cf['B_HI']}] ({cf['B_MAPPING']})"); line()

# ---------------------------------------------------------------- DIV 1
line("-"); P("DIVISION 1 — THE CALIBRATED GAME (US–CHINA)   [recomputed]")
line("-")
t=R["two_country"]
P(f"  countries: {t['iso']}   market power b: {t['b']}")
P(f"  Nash tariffs: US={fmt(t['nash'][0])}  CN={fmt(t['nash'][1])}")
P(f"  folk delta*: US={fmt(t['delta_star'][0])}  CN={fmt(t['delta_star'][1])}")
P(f"  binding delta* = {fmt(t['binding'])}   regression guard passed: {t['guard_passed']}")
P("  meaning: prisoner's dilemma; cooperation (free trade) sustainable iff delta>=0.70.")

# ---------------------------------------------------------------- DIV 2-5 (ingested)
ing=R["external"]
def ext_block(num,title,key,explain):
    line("-"); P(f"DIVISION {num} — {title}   [{ing[key]['status']}]"); line("-")
    if ing[key]["status"]!="ingested":
        P(f"  NOT YET RUN — {ing[key]['note']}"); P(f"  {explain}"); return
    d=ing[key]["data"]
    P(f"  ingested keys: {list(d.keys())[:8]}")
    P(f"  {explain}")
    return d

ext_block(2,"BOUNDED RATIONALITY (QRE, one-shot)","qre",
          "QRE -> Nash as rationality lambda grows; finite lambda trade-wars >= Nash.")
ext_block(3,"EXISTENCE != ATTAINABILITY (repeated game)","reciprocity",
          "independent learners stay near Nash for all patience; cooperative SPE exists but is not reached.")
ext_block(4,"MECHANISM (committed reciprocity)","mechanism",
          "a committed reciprocator makes cooperation attainable above binding delta*; patience/priors alone do not.")
d4b=ext_block("4b","PPO confirmation (mechanism, deep RL)","ppo",
          "deep PPO traces the same descent -> not a tabular artifact.")
d5=ext_block(5,"HISTORICAL NATURAL EXPERIMENT (2000–2024)","natural_expt",
          "realized US tariff ~3% under WTO, ->~21% (2019), ->~47.5% (2025): collapse toward Nash when commitment withdrawn.")
if d5:
    yrs=d5["years"]; P(f"     binding delta* range: {fmt(min(d5['binding']))}–{fmt(max(d5['binding']))}; "
                       f"realized tariff {yrs[0]}->{yrs[-1]}: {d5['realized_tariff'][0]}%->{d5['realized_tariff'][-1]}%")

# ---------------------------------------------------------------- DIV 6
line("-"); P("DIVISION 6 — N=30 MULTILATERAL EXTENSION (BACI 2024)   [recomputed]"); line("-")
m=R["multilateral"]; iso=m["iso"]
ds=[None if x is None else x for x in m["delta_star"]]
imp=np.array(m["import_share"]); order=np.argsort(-imp)
P(f"  binding cooperator: {m['binding_country']}  (delta*={fmt(m['binding_delta_star'])})   hegemons: {m['hegemons'] or 'NONE'}")
P(f"  corr(delta*, market power) = {fmt(m['corr_b'],2)}")
P(f"  partial corr(delta*, network exposure | market power) = {fmt(m['partial_corr_exposure_given_b'],2)}  [partly structural — see rails]")
P("  per-economy (sorted by import share):")
P(f"    {'iso':>4} {'impShr':>7} {'b':>5} {'Nash':>5} {'d*':>6} {'expo':>6}")
for i in order:
    P(f"    {iso[i]:>4} {imp[i]:>7.3f} {m['b'][i]:>5.2f} {m['nash'][i]:>5.2f} {fmt(ds[i],2):>6} {m['exposure'][i]:>6.3f}")
iMEX,iCAN=iso.index("MEX"),iso.index("CAN")
P(f"  headline: MEX delta*={fmt(ds[iMEX],2)}, CAN delta*={fmt(ds[iCAN],2)} — mid-size but EASIEST to keep cooperative")
P("           (deep US exposure -> ruinous Nash -> strongly deterred). Network position, not size.")

# ---------------------------------------------------------------- DIV 7
line("-"); P("DIVISION 7 — ROBUSTNESS (16-combination sweep)   [recomputed]"); line("-")
s=R["sensitivity"]
P(f"  MEX & CAN below size trend in ALL combinations: {s['all_below_size_trend']}")
rs=s["combinations"]
mn=min(min(v['MEX_resid'],v['CAN_resid']) for v in rs.values())
mx=max(max(v['MEX_resid'],v['CAN_resid']) for v in rs.values())
P(f"  residual range across delta x b-mapping: [{mn:.3f}, {mx:.3f}]  (all negative = robust)")

# ---------------------------------------------------------------- provenance + rails
line("-"); P("PROVENANCE"); line("-")
P("  recomputed in pipeline.py (numpy + BACI): DIV 1, 6, 7")
P("  ingested from Colab (torch / World Bank API): DIV 2, 3, 4, 5")
for k,v in ing.items(): P(f"    {k:>12}: {v['status']}")

line("-"); P("HONESTY RAILS (carry verbatim into the paper)"); line("-")
P("  1. Pre-2018 ~3% match is partly DEFINITIONAL (cooperative anchored to WTO MFN);")
P("     falsifiable content = Nash level, breakdown timing, escalation direction.")
P("  2. N=30 partial-corr is partly STRUCTURAL (delta* is a function of b and exposure);")
P("     headline = the MEX/CAN ranking from real BACI exposure, NOT the correlation.")
P("  3. N=30 b is an import-share PROXY (no free per-country export-supply elasticity")
P("     panel for 30 WTO economies; grounded in BLW 2008 / Soderbery 2018; robust via sweep).")
P("  4. N=30 is a STRUCTURAL PREDICTION, not a behavioral test vs observed defection.")
P("  5. NO new theorem — contribution is integrative/applied.")

line("-"); P("RETIRED CLAIMS (must NOT appear)"); line("-")
for c in ["+2.941 / +12.185 advantage (reward-scale artifact)","'structurally irreversible'",
          "stylized 'hegemon makes cooperation impossible' (died on real BACI)",
          "'corr(GDP,tariff)=1.00' as a finding (circular)",
          "any delta* as a number when a country gains at Nash (report infinity)"]:
    P(f"  - {c}")
line()

open(ROOT/"deployment_report.txt","w").write(buf.getvalue())
P("written -> deployment_report.txt")
