# Learning to Cooperate on Tariffs

Code and results for *Learning to Cooperate on Tariffs: The Attainability Frontier of
Decentralized Trade Policy* (Nubaid Khan, 2026).

The paper asks whether boundedly-rational learners ever reach the cooperative tariff
equilibrium that the folk theorem says exists. They do not, at any discount factor, across
two algorithm classes, with both sides learning, and under a general-equilibrium payoff as
well as a reduced-form one. Four structural forces close that gap, they enter one stability
condition additively, and every coefficient in that condition is closed form.

## What is here

```
pipeline.py     config, game, BACI calibration, experiments -> data/processed/results.json
dashboard.py    reads results.json -> figures/   (no hardcoded numbers)
deployment.py   reads results.json -> deployment_report.txt
REGISTRY.md     crosswalk: result -> function -> JSON key -> figure -> paper section
```

## Reproducibility tiers, stated plainly

Not everything in the paper is recomputed by this pipeline, and the split matters.

**Recomputed here** (numpy, from the BACI files): the two-country calibration, the
thirty-economy multilateral extension, and the sixteen-combination robustness sweep.

**Computed elsewhere and ingested** as JSON from `data/processed/`: the reinforcement
learning experiments, quantal response equilibrium, opponent-aware learning, the
evolutionary sweep, PPO, and the historical series. These need `torch` or the World Bank
API and were run in Colab. Running `pipeline.py` does **not** re-derive them.

So a clean run reproduces the calibration and the multilateral analysis from raw data, and
loads the rest. Anyone checking the learning results should read the JSONs in
`data/processed/` and the corresponding rows of `REGISTRY.md`.

## Calibration guard

`pipeline.py` asserts that the US-China calibration still yields Nash tariffs of 0.610 and
0.499 and a binding critical discount of 0.700. If calibration drifts, the run fails loudly
rather than letting a changed number reach the paper unnoticed. The guard covers the
two-country calibration only.

## Run order

```bash
# 1. place the CEPII BACI files in data/raw/ (see data/raw/README.txt; not committed)
# 2. the Colab outputs are already in data/processed/
python pipeline.py      # -> data/processed/results.json
python dashboard.py     # -> figures/
python deployment.py    # -> deployment_report.txt
```

Requires `numpy`, `pandas`, `scipy`, and `matplotlib`.

## Honesty rails

`REGISTRY.md` carries the constraints that govern how each result may be described,
including which correlations are partly structural, which fits are definitional, and a list
of claims retired during the project because they did not survive checking. The retired
list is kept deliberately: it records what was killed and why.

## Data sources

- Bilateral trade flows: CEPII BACI 2024 (Gaulier and Zignago, 2010)
- Bilateral applied tariffs: UNCTAD TRAINS via World Bank WITS
- Trade-weighted US-China war-era rates: Peterson Institute for International Economics
- Macroeconomic aggregates: World Bank national accounts
- Calibration targets: Ossa (2014); Broda and Weinstein (2006); Soderbery (2018)

## License

MIT. See `LICENSE`.
