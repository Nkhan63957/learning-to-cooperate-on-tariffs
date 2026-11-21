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

## V2 general-equilibrium extension crosswalk (branch `v2-ge-extension`)

New in the `v2-ge-extension` branch only; `main` and `python verify.py` (no arguments) are
frozen for the v1 submission. Generator: **`ge_frontier.py`** (NumPy + SciPy only, no BACI).
Outputs: **`data/processed/v2_ge_routes.json`** (routes) and **`data/processed/v2_ge_surfaces.npz`**
(cached GE real-income payoff surfaces). Every row below is re-derived in pure stdlib by
**`python verify.py --v2`**. These are **new V2 numbers**, not a reproduction of a pre-existing
GE result — `ge_frontier.py` is newly constructed (see `docs/provenance.md`). Model: one-sector
Armington CES GE, payoff = exact real-income indirect utility U = E/P. Headlines are the
**σ = 4.0** primary; the sensitivity grid is **σ ∈ {2.5, 4.0, 6.0}**.

| Result | ge_frontier fn | v2_ge_routes.json key | paper § | headline (σ=4; robust across σ grid) |
|---|---|---|---|---|
| GE bilateral stage game | `bilateral_stage_payoffs` | `primary.bilateral_stage_game`, `.bilateral_GL` | GE layer (§4.6) | GE Nash (US,CN)=0.469/0.436; folk δ*_US=0.619 (0.600→0.627 over σ grid) |
| R1 LOLA transparency on GE | `route_lola` (via `colab/run_A2.py`) | `primary.route1_lola` | §6.1 | naive joint −0.034 → LOLA joint −0.013; crosses gap, not exploited by ALLD; **true for all σ** |
| R2 evolutionary basins on GE | `route_evolution` | `primary.route2_evolution` | evolutionary route | coop basin @δ=0.9 = 0.48 (0.48–0.56 over σ); invasion threshold δ_inv=(T−R)/(T−P)=0.619=folk δ* |
| R3 noisy-monitoring tolerance on GE | `route_tolerance` | `primary.route3_tolerance` | §6.3 | peak tolerance at σ_noise=0.2539; δ*(σ_noise=0.20)=0.659 using GE G,L |
| R4 N=30 network δ* ranking on GE | `route_network` / `ge_folk_delta_star` | `primary.route4_network` | §7.2 | binding **CHN** δ*=0.421; **MEX/CAN remain last (ranks 30/29)**; no hegemon; partial-corr(δ*,exposure\|b)=−1.00; corr(δ*,b)=0.28 |
| GE payoff surfaces (cache) | `bilateral_surface`, `network_slices` | — (`v2_ge_surfaces.npz`) | GE layer | `bilat_U_{US,CN}_s{σ}` 41×41; `net_slice_{free,nash}_s{σ}` 30×41; `net_nash_tariff_s{σ}` |

## Honesty rails — V2 GE extension (carry verbatim into any V2 write-up)
- **`ge_frontier.py` is newly constructed, not recovered.** There was no pre-existing CES GE
  payoff layer to extend; these are new results. Do not describe them as reproducing a v1 GE number.
- **The 30×30 trade matrix is gravity-reconstructed** from `stage2` b and exposure (off-diagonal
  ∝ b_j, row foreign-mass = exposure), because bilateral BACI flows are not in the repo. Weaker
  provenance than the v1 BACI calibration; state it as a limitation.
- **σ is a chosen primitive, not estimated.** GE optimal tariffs ≈ 1/(σ−1); the headline uses
  σ=4 (Ossa 2014) and is checked across {2.5, 4, 6}.
- **What survives GE vs what does not (report both):** the MEX/CAN "hardest to sustain" ranking,
  the exposure partial-correlation (≈ −1), and "no hegemon" **survive**. The v1 "**USA** is the
  binding cooperator" and the strong corr(δ*, market power) do **not** survive — under GE the
  optimal-tariff compression narrows the δ* band (binding shifts to CHN; corr(δ*,b)≈0.28). This
  divergence is a finding, not a defect.
- **verify.py --v2 takes the GE real-income payoffs as the model primitive** (as v1 takes b) and
  recomputes every closed-form number; the stochastic LOLA/basin outputs are `[consistency]` checks.

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
