# REGISTRY — the single crosswalk for the paper

Every figure and number in the paper traces to one row here. When drafting,
cite the **results.json key**, not a number typed from memory. Regenerate with
`python pipeline.py && python dashboard.py`.

## Reproducibility tiers
- **Recomputed** (numpy, from `config.py` + the BACI file): two-country, multilateral, sensitivity.
- **Ingested** (computed in Colab, loaded from `data/processed/*.json`): qre, reciprocity, mechanism, ppo, natural_expt.
  These need `torch` (deep PPO) or the World Bank API and are not recomputed in the pipeline.

## Crosswalk

*Coverage note: this table was written for the first version of the paper and covers the
calibration, QRE, reciprocity, mechanism, historical and multilateral results. The
two-sided learning contrast, the opponent-aware route, the evolutionary sweep, the
tolerance experiment, the general-equilibrium layer, the mean-field derivation and the
thirteen-dyad panel were added later and are not yet rowed here.*

| Result | pipeline fn | results.json key | figure | paper § | headline numbers |
|---|---|---|---|---|---|
| R1 PD game structure | (analytic, in game.py) | `two_country` | — | Model | reaction slope, Nash Pareto-dominated |
| R1 calibrated US-China | `exp_two_country` | `two_country.nash`, `.delta_star` | — | Model/Methods | Nash 0.610/0.499; binding δ*=0.700 (guarded) |
| R2 QRE (one-shot) | ingested | `external.qre` | qre_*.png (Colab) | Results | QRE→Nash as λ→∞; finite λ ≥ Nash hardness |
| R3 existence≠attainability | ingested | `external.reciprocity` | reciprocity_*.png | Results | independent learners stay near Nash for all γ |
| R4 mechanism (committed reciprocity) | ingested | `external.mechanism`, `external.ppo` | mechanism_*.png | Results | cooperation above binding δ*; PPO confirms |
| Natural experiment (Phase 3) | ingested | `external.natural_expt` | natural_experiment.png | Validation | realized ~3%→21%→47.5% vs Nash band; δ* 0.84→0.55 |
| R5 N=30 multilateral | `exp_multilateral` | `multilateral` | network_delta.png, nash_and_delta.png | Extension | binding USA δ*=0.77; no hegemon; partial-corr −0.99; MEX/CAN δ*≈0.36 |
| R5 robustness | `exp_sensitivity` | `sensitivity` | — | Extension | MEX/CAN below trend in all 16 combos |

## Honesty rails (carry verbatim into the paper)
- **two_country pre-2018 ≈3% match is partly definitional** (cooperative tariff anchored to WTO MFN). Falsifiable content = Nash level, breakdown timing, escalation direction.
- **N=30 partial-corr −0.99 is partly structural** (δ* is a function of b and exposure). Headline = the MEX/CAN ranking driven by real BACI exposure, NOT the correlation.
- **N=30 b is an import-share proxy** (no clean free per-country export-supply elasticity panel for 30 WTO economies; BLW 2008 covers 15 non-WTO, Soderbery 2018 is disaggregated). Grounded + robust (sweep), but a stated limitation.
- **N=30 is a structural prediction, not a behavioral test** against observed defection.
- **Two closed-form results are the paper's own** (added after this registry was first written):
  the optimal enforcement tolerance under noisy monitoring (§6.3) and the observability
  coefficient with the route-dominance identity ω = L + T (§5.2). Everything else is
  integrative and applied. Do not carry the older "no new theorem" wording.

## Retired claims (must NOT appear)
- The original "+2.941 advantage" / "+12.185" (reward-scale artifact).
- "structurally irreversible" framing.
- The stylized-demo "hegemon makes cooperation impossible" (did not survive real BACI calibration).
- "corr(GDP, tariff)=1.00" as a finding (circular — mechanical from the b-mapping).
- Any δ* reported as a number when the country gains at Nash (report ∞ / impossibility).
