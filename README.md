# TESS Exoplanet Hunter

An independent pipeline for finding and vetting exoplanet candidates in NASA TESS light-curve data. Started as a fully manual workflow (MAST portal + Google Colab) and has since been rebuilt into two production-quality Python tools for automated, multi-sector transit detection.

## What this does

TESS (Transiting Exoplanet Survey Satellite) stares at patches of sky and records the brightness of hundreds of thousands of stars over time. If a planet passes in front of its star from our point of view, the star's brightness dips slightly and periodically. This project searches that brightness data (light curves) for those dips, separates genuine transit signals from the many things that mimic them, and documents the candidates that survive vetting.

## Tools

**`tess_exoplanet_hunter.py`** — batch pipeline
Processes many TESS targets automatically: downloads light curves from MAST, detrends them, runs a period search, flags candidates that pass initial thresholds, and logs results for later review.

**`tess_deep_dive.py`** — single-star analysis
Takes one TESS Input Catalog (TIC) ID and runs a full multi-sector investigation: phase-folded light curves, periodograms, per-sector comparison plots, and the diagnostics needed to confirm or reject a candidate by hand.

## Method

1. **Data retrieval** — SPOC 2-minute cadence light curves pulled from the NASA MAST archive via `lightkurve`, cross-referenced against the TESS Input Catalog and the NASA Exoplanet Archive TOI table.
2. **Detrending** — stellar and instrumental trends removed; flares clipped using MAD-based upward-only clipping so real transit dips (which are downward) aren't touched.
3. **Period search** — Box Least Squares (BLS) run first as a fast, coarse search. Promising periods are then narrowed and handed to Transit Least Squares (TLS) for a higher-fidelity check, since TLS alone is too slow to run blind on long multi-sector baselines.
4. **Vetting** — candidates are checked across every available sector (a signal that only shows up in one sector is usually an artifact of sparse coverage, not a real planet), and screened against known false-positive shapes: stellar rotation/starspot modulation, flares, instrumental noise, and eclipsing-binary aliasing.
5. **Logging** — every target processed, its verdict, and the reasoning is recorded in `search_summary.csv` / `search_summary_all.csv`, with `analyzed_tics.json` tracking what's already been run.

## Engineering notes

Building this surfaced a few non-obvious problems worth documenting:

- `lightkurve`'s built-in BLS wrapper generates an unworkably large search grid on long multi-sector baselines regardless of `frequency_factor` — replaced with `astropy.timeseries.BoxLeastSquares` and a manually controlled period grid.
- Running TLS without first narrowing the period range from BLS causes multi-hour hangs on multi-sector data; solved with BLS-guided narrowing plus a point-count cap with a `manual_bin()` fallback.
- `astropy`'s BLS implementation requires max transit duration to be strictly less than the minimum period being searched, or it raises a `ValueError` — durations are clipped to ~80% of `min_period` to stay safe.
- The `with`-block pattern for `ThreadPoolExecutor` timeouts silently fails to time out (it still blocks on `__exit__`); using `shutdown(wait=False)` outside the `with` block fixes this.
- Python 3.12+ needs a `setuptools` upgrade before `transitleastsquares`/`batman` will import, since they still depend on `distutils`.

## Status

The batch pipeline is complete and actively running against new targets. The deep-dive tool has been used for detailed candidate review, including a full multi-sector rejection of TIC 270415707 (confirmed as starspot-driven stellar variability, not a transit). A known issue — BLS power values clipping to an identical ceiling across many periods in deep-dive output — is under investigation, since it could in principle mask a genuine signal.

## Stack

`lightkurve`, `transitleastsquares`, `astropy`, `numpy`, `matplotlib`, `concurrent.futures`

## Data sources

NASA MAST archive, TESS Input Catalog (TIC), NASA Exoplanet Archive (TOI table), exo.MAST

---

Independent project by Muhammad Hashir Hussain.
