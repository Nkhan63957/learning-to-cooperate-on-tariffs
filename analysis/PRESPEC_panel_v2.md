# Pre-specification: expanded dyad panel (v2)

**Written 2026-08-27, before any tariff data for the expanded panel was retrieved.**

This document fixes the design of the expanded panel in advance. It is committed to the
repository before the WITS pull so that the selection rules cannot be adjusted after seeing
results. It supersedes nothing in the published paper; it governs an extension of §7.3.

## Purpose

The published panel has 13 dyads (4 treated, 9 never-treated) and 73 dyad-years. The
expansion adds control dyads and extends the time dimension. The treated set does not
change. The goal is a more credible parallel-trends assumption and a placebo distribution
rather than a placebo point estimate.

## What does not change

- **Treated dyads.** The same four: United States to China (2018), China to United States
  (2018), United States to India (2019), China to Australia (2020). The three previously
  examined and rejected candidates stay rejected, for the reasons already documented in
  §7.3. No candidate may be added after this document is written.
- **Estimator.** Cohort event study comparing each treated dyad only against not-yet-treated
  dyads.
- **Outcome.** Fraction of the structural Nash gap closed:
  `(tau_ij - tau_coop_ij) / (tau_nash_i - tau_coop_ij)`, with `tau_nash_i` taken from the
  calibration in `stage2_results.json` and `tau_coop_ij` the dyad's pre-treatment applied
  rate.
- **Diagnostics reported.** Pre-trend, placebo, leave-one-out, clean estimator against naive
  two-way fixed effects, and control-group robustness.

## Universe

The 30 economies in `stage2_results.json`, giving 870 ordered pairs.

## Control inclusion criteria

An ordered pair (i, j) enters the control set if and only if all four hold. Filters are
applied in this order and the surviving count is logged at each step.

1. **No documented economy-wide tariff action** by either side against the other, 2000-2025,
   other than the four treated events.
2. **Not inside a customs union or free trade agreement** with each other for the majority of
   the sample window. A dyad at zero tariffs by treaty has no variation and would inflate the
   control count without adding information.
3. **Bilateral trade above a floor** of **0.1 percent of the importer's total imports**, so
   the tariff series reflects a meaningful relationship rather than noise on trivial flows.
4. **WITS coverage for at least 15 of the 26 years** in the window.

## Parameters fixed in advance

| Parameter | Value |
|---|---|
| Trade-share floor | 0.1 percent of importer total imports |
| Minimum year coverage | 15 of 26 |
| Year range | 2000-2025 |
| Tariff indicator | AHS weighted average, MFN weighted average as fallback |
| US-China rates | Bown/PIIE trade-weighted, as in the published panel |

## Decisions recorded before the pull

**RCEP is not treated as an exclusion.** It entered force 2022-01-01 for Australia,
China, Japan, Thailand and Vietnam, 2022-02-01 for Korea, 2022-03-18 for Malaysia and
2023-01-02 for Indonesia, covering at most 4 of 26 years, with tariff elimination phased
over 10 to 20 years rather than immediate. Excluding on RCEP alone would remove up to 56
ordered pairs for coverage of under a sixth of the window.

**Both applied rate types are pulled in a single query.** Effectively Applied (AHS) and
MFN Applied are requested together with UNCTAD AVE estimation. AHS is used where present
and MFN as the fallback, with the source recorded per row in a `rate_source` column.

## Stopping rules

- If fewer than **25** control dyads survive, the trade-share floor may be loosened **once**,
  and the fact that it was loosened is reported.
- No other parameter may be adjusted after the data are seen.

## Commitment on reporting

**The result will be reported whichever way it comes out.** If the expanded panel weakens,
overturns, or fails to replicate the published ATT of 0.270, that is the finding and it goes
in the paper, with the 13-dyad and expanded results shown side by side. The extension is not
conditional on confirming the earlier estimate.

## Known issue to handle explicitly

China acceded to the WTO in December 2001. A series beginning in 2000 crosses that accession
for all China dyads. Either begin the window at 2002 or retain 2000 and note the accession,
as Figure 8 already does. The choice is made before estimation and stated in the paper.
