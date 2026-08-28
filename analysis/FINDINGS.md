# Attempted panel expansion, and why it was abandoned

**2026-08-27.** This directory documents an extension that was pre-specified, executed, and
then not used. It is kept because the reason it failed is itself a result, and because the
pre-specification commits to reporting the outcome either way.

## What was attempted

The published panel in Section 7.3 has 13 dyads, 4 treated and 9 never-treated, over roughly
six years. The plan was to hold the treated set fixed and expand the control set and the time
dimension, using bilateral applied tariffs from UNCTAD TRAINS for all 30 economies in the
calibration, 2000 to 2025. `PRESPEC_panel_v2.md`, written before any data was pulled, fixes
the selection rules, the parameters, the stopping rules, and the commitment to report the
result whichever way it came out.

## What was found

The filters worked as designed. Of 870 ordered pairs, 548 survived treaty exclusions and 533
cleared a 0.1 percent trade-share floor, leaving roughly 530 candidate controls against the
published 9.

**The expansion failed on measurement, not on sample size.** UNCTAD TRAINS does not record
discriminatory trade-war duties. Across the four treated events its weighted-average applied
rates move as follows:

| Dyad | Event year | TRAINS pre | TRAINS post | Change |
|---|---|---|---|---|
| United States from China | 2018 | 2.78 | 2.60 | -0.17 pp |
| China from United States | 2018 | 6.19 | 5.25 | -0.94 pp |
| United States from India | 2019 | 2.50 | 2.90 | +0.40 pp |
| China from Australia | 2020 | 0.98 | 0.71 | -0.27 pp |

The actual United States rate on Chinese goods rose from 3.1 to 21.0 percent between 2017 and
2019 on the trade-weighted Bown/PIIE series. TRAINS reports 2.76 percent for 2019 and shows
no break at any of the four events. A panel built from it would show no treatment.

A second problem affects the control side. There are 803 dyad-year changes above 5 percentage
points in the 30 by 30 matrix, but they sit in very small dyads: Korea from Argentina moves
186 points in 2013, Switzerland from Mexico 156 points in 2006. These are import-composition
artifacts in weighted averages on thin trade flows, not policy changes. They would dominate
the control group as noise.

## What follows

The published panel stands. Its four treated dyads use hand-assembled trade-weighted rates
that capture the duties standard databases omit, which is why it is small: the binding
constraint is measurement, not the diligence of the search. Section 7.3 now states this and
cites the TRAINS figure.

## Files

    PRESPEC_panel_v2.md    pre-specification, written before the pull
    fta_exclusions.csv     treaty exclusions with per-row confidence and sources
    build_panel_v2.py      panel assembly and filter cascade
    check_panel_v2.py      sanity checks, including the reproduction gate
    estimate_panel_v2.py   cohort event study with placebo and leave-one-out
    data/                  the raw WITS pulls and the derived import shares

The scripts are complete and runnable. They were not used to produce any number in the paper.
