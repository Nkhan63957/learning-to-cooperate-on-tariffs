# Provenance

How every committed result was produced, where its generating code lives, and —
honestly — which generators did **not** survive into the published archive. This
file was assembled on 2026-11-20 from the committed repository and from the local
Colab/chat export under `_claude_export/` (git-ignored; working dir in Colab was
`/content/makoto`, the predecessor project the tariff paper was substituted into).

Locators of the form `conv#/msg#` point into
`_claude_export/tariff_conversations.json` (49 conversations; the tariff work is in
#17, #21, #23, #25, #27). They are provenance pointers, not runnable paths.

## Reproducibility tiers (restating README/REGISTRY, with generator status)

| Result | results.json key | Tier | Generator | Status |
|---|---|---|---|---|
| two-country US–China | `two_country` | Recomputed (numpy + BACI) | `pipeline.py::exp_two_country` | ✅ committed, runnable |
| N=30 multilateral | `multilateral`, `stage2_results.json` | Recomputed (numpy + BACI) | `pipeline.py::exp_multilateral` | ✅ committed, runnable |
| 16-combo sensitivity | `sensitivity` | Recomputed (numpy + BACI) | `pipeline.py::exp_sensitivity` | ✅ committed, runnable |
| §6.1 opponent-shaping (LOLA) | — (not ingested) | Recomputed (numpy) | `colab/run_A2.py`, `colab/run_A2_scaled.py` | ✅ committed, runnable |
| QRE | `qre_results.json` | Ingested (Colab) | logit-QRE solver | ❌ **source not in export** |
| reciprocity | `reciprocity_results.json` | Ingested (Colab) | two-sided learner sweep | ❌ **source not in export** |
| mechanism | `mechanism_results.json` | Ingested (Colab) | committed-reciprocator sweep | ❌ **source not in export** |
| PPO | `ppo_results.json` | Ingested (Colab, torch) | stable-baselines3 PPO | ❌ **source not in export** |
| natural experiment | `phase3_results.json` | Ingested (Colab + external data) | base series builder | ❌ base builder not in export; patch/figure cell archived |

## Git history of the committed artifacts

```
cd80629  Code and results for Learning to Cooperate on Tariffs   # pipeline.py, dashboard.py, deployment.py, all data/processed/*.json
7b017e6  Update CITATION.cff
6f95982  Update README.md
6520de9  Add attempted panel expansion and the measurement finding that stopped it   # analysis/
e946417  Link the analysis directory from the README
98c4ad2  Add ORCID iD to citation metadata
5dd483e  Add verify.py, reproducing every headline number without the raw data        # verify.py (source: conv27/msg360)
735cf84  Document verify.py in the README
90bcbb0  Document provenances in the README
6a9cf8a  Add the opponent-shaping implementation for Section 6.1                       # colab/run_A2.py, colab/run_A2_scaled.py
```

All six ingested learning JSONs entered the repo in the **initial commit `cd80629`**
as pre-computed files; their Colab generators were never committed.

## Exact configuration of each committed result

Values below are read directly from the committed files, not from memory.

**Calibration (shared).** κ = 1.0; δ (trade-contraction coupling / cross term ψ) = 0.10;
b_US = 0.66, b_CN = 0.56 ⇒ Nash 0.610 / 0.499, binding δ* = 0.700. Guarded in
`pipeline.py::exp_two_country` and recomputed from scratch in `verify.py`.

**N=30 multilateral / `stage2_results.json`.** 30 economies (USA…MYS); b mapping
linear on import share with B_LO = 0.50, B_HI = 0.68; same (κ, δ). Headlines:
corr(δ*, b) = 0.934, partial-corr(δ*, exposure | b) = −0.989, binding USA δ* = 0.767,
MEX/CAN δ* ≈ 0.36, hegemons = none. Verified in `verify.py` §7.2.

**Sensitivity.** 4 b-mappings {linear, rank, sqrt, log} × 4 δ {0.05, 0.10, 0.20, 0.30}
= 16 combinations; MEX and CAN below the size trend in all 16.

**§6.1 LOLA (`colab/run_A2_scaled.py`).** Payoffs R=0.0, T=0.218, P=−0.093, S=−0.30;
discount 0.9; SCALE = 12 (monotone, order-preserving); η = 40; α = 0.08; steps = 700;
**6 seeds**. Reproduces naive −0.0904 (sd 0.0001), LOLA −0.0329 (sd 0.0048), LOLA-vs-ALLD
−0.0965. (`run_A2.py` base config: steps = 600, α = 1.0, η = 1.0, 6 seeds.)

**QRE (`qre_results.json`).** λ grid = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
(12 points); deterministic logit solve, **no seeds**; converges to 0.6101 / 0.499.

**Reciprocity (`reciprocity_results.json`).** γ grid = [0.5, 0.55, …, 0.95] (10 points);
**n_seeds = 10**; conditions: independent, vs_reciprocator, coop_prior.

**Mechanism (`mechanism_results.json`).** γ grid = [0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
(8 points); seed count not recorded in the file; stated source `calibrated_arc.json`.

**PPO (`ppo_results.json`).** γ grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95] (6 points);
**n_seeds = 3**; algorithm "PPO (stable-baselines3)"; non-committed condition =
myopic best-response opponent (robust proxy).

**Natural experiment (`phase3_results.json`).** Years 2000–2024 **as committed**;
crisis_years [2009, 2020, 2022]; realized US-on-China tariff series ends 3% → 10% (2018)
→ 21% (2019) → 19.3% (2020–24). Sources: realized tariffs PIIE/Bown; macro aggregates
World Bank. ⚠️ **Discrepancy:** the REGISTRY headline "…→ 47.5%" refers to a 2025 point
that exists only in the *patched* Colab/figure version (see `legacy_generators/`), not
in the committed JSON. `verify.py` does not test the 47.5 figure.

## Archived legacy generators (`colab/legacy_generators/`)

Only code that exists **verbatim** in the export is archived here, each with a header
marking it as unmodified and non-runnable-in-repo:

- `legacy_pipeline_colab.py` — conv21/msg92. The monolithic Colab precursor of
  `pipeline.py`: recomputes two-country/multilateral/sensitivity from BACI and
  **ingests** the five learning JSONs as uploaded inputs. Superseded by the committed
  `pipeline.py`. This is the generator for the *recomputed* tier, in its original form.
- `phase3_add2025_and_fig3.py` — conv21/msg121. A patch + Figure-3 cell that mutates an
  **already-existing** `phase3_results.json` (appends 2025 → 47.5) and renders the plot.
  It does not build the base 2000–2024 series.

## Generators that are NOT recoverable (documented gap, not fabricated)

A full scan of the tariff conversations (457 Python blocks across #17/21/23/25/27)
found **no verbatim source** for the following. The repo ships their outputs; their
generators ran in earlier Colab sessions that did not survive into this export:

- **PPO training** — the only trace anywhere is a single line, `!pip install -q
  stable_baselines3 gymnasium` (conv21/msg80). No environment, no training loop.
- **QRE solver, reciprocity learner, mechanism sweep** — present only as ingested
  outputs and prose description; no computing code block exists.
- **`phase3` base-series builder** and **`stage2` BACI stage-1 builder** — only a
  patch/figure cell (phase3) and trivial save-to-path lines (conv21/msg720, msg722)
  survive; the stage-2 numbers are independently re-derivable via
  `pipeline.py::exp_multilateral` from the BACI files.

To reach literal 100% end-to-end provenance for the learning tier, the original Colab
notebooks (Google Drive `/content/makoto`) would need to be recovered and committed.
They are not in this export and have **not** been reconstructed here.

## Derivations backing the paper's closed-form results (prose locators)

Pointers into the transcript for the theorems `verify.py` checks numerically:

- **ω = L + T** (route-dominance identity, §5.2) — conv27/msg78, 86, 122, 166
  (sympy-verified; 4,048-point sweep, zero counterexamples).
- **Patience switch m\* = 1 − (G − T)/L** → 0.66 (US) / 0.88 (CN) — conv27/msg80, 170.
- **Enforcement-tolerance theorem (§6.3)** — conv21/msg789–794. Note: an exploratory
  branch here derived `T* = 8/15 ≈ 0.53` with a flagged fine-vs-netted constant
  mismatch; the committed `verify.py` uses the later `w*` / `CX_OVER_CF` formulation,
  which passes.
- **Mean-field attractor (§5)** — conv23/msg72–74, conv27/msg95–96. Continuous-time
  limit of independent Q-learning → coupled replicator ODEs; mutual Nash is the unique
  stable attractor.
- **Attainability index A = P/(P+N)** — conv27/msg2, conv23/msg68–74.

## Retired / abandoned workstreams

- **MAKOTO (predecessor)** — crisis-conditioned macro-RL, registered at IAI²O
  2026-06-15; audited and **abandoned** (headline was a reward-multiplier artifact;
  ablation reversed the conclusion). The tariff paper was substituted in its place.
  conv21/msg3, 841–942; conv27/msg68–116. Consistent with REGISTRY's retired "+2.941 /
  reward-scale" claim.
- **Dropped extensions** — coalition stability, sectoral/political-economy tariffs,
  leaky commitment: proposed, never reached the paper. conv23/msg266.
- **Panel expansion** — a larger bilateral panel attempted then stopped by a TRAINS
  measurement finding (TRAINS does not record discriminatory trade-war duties).
  conv23/msg31–152; committed as `analysis/` with `FINDINGS.md` (commit 6520de9).
