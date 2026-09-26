"""
tess_deep_dive.py

Deep-dive analysis of a SINGLE TIC target, complementary to tess_exoplanet_hunter.py.

Purpose
-------
The batch pipeline is tuned for throughput: it caps point counts, narrows the
TLS period search to +/-10% around the BLS best period, and applies hard
timeouts so one slow star can't stall the whole run. That's the right
tradeoff for scanning hundreds of stars, but it means the batch pipeline can
miss:
  - long-period / single or double-transit signals
  - shallow signals that need the full-resolution light curve (not binned)
  - a second planet in a multi-planet system (the primary signal dominates
    the periodogram and can mask a second one)

This script takes ONE TIC ID and throws much more compute + time at it:
  - downloads and stitches ALL available SPOC 2-min sectors (no sector cap)
  - runs BLS over a wide, fine period grid (no narrow bracket)
  - runs TLS at full resolution (no aggressive binning) with high
    oversampling, over the FULL period range BLS/TLS can support
  - after finding the best signal, masks it out and searches again, so a
    second candidate isn't hidden by the first
  - produces a much larger set of diagnostic plots per candidate
  - has generous (but still present) timeouts, since this is meant to be run
    interactively on a handful of stars you already care about, not in a
    batch loop

Usage
-----
    python tess_deep_dive.py 260647166
    python tess_deep_dive.py 260647166 --max-planets 2 --out-dir tess_results/deep_dive
    python tess_deep_dive.py 260647166 --min-period 0.5 --max-period 60 --no-bin

Output
------
    <out-dir>/<TIC>/
        full_light_curve.png
        per_sector_light_curves.png
        bls_periodogram.png
        candidate_1/
            tls_periodogram.png
            phase_fold.png
            odd_even.png
            secondary_eclipse_check.png
            transit_depths_by_sector.png
            summary.json
        candidate_2/            (if --max-planets > 1 and something was found)
            ...
        deep_dive_summary.json
"""

import argparse
import concurrent.futures
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightkurve as lk
from astropy.stats import mad_std
from astropy.timeseries import BoxLeastSquares
from transitleastsquares import transitleastsquares, transit_mask

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config defaults (all overridable via CLI)
# --------------------------------------------------------------------------
DEFAULT_MIN_PERIOD = 0.5      # days
DEFAULT_MAX_PERIOD = 100.0    # days -- much wider than the batch pipeline's narrow bracket
DEFAULT_BLS_DURATIONS = np.linspace(0.02, 0.5, 20)  # days, fine-but-bounded grid
DEFAULT_TLS_OVERSAMPLING = 5
DEFAULT_TLS_DURATION_GRID_STEP = 1.02   # finer than a coarse batch run
DEFAULT_MAST_TIMEOUT_SEC = 600          # generous, this is a single interactive star
DEFAULT_TLS_TIMEOUT_SEC = 3600          # up to an hour is fine for one star
DEFAULT_BIN_POINT_CAP = 400_000         # effectively "don't bin" for most targets


# --------------------------------------------------------------------------
# Timeout helper (same pattern as the batch pipeline: shutdown(wait=False)
# is required or the timeout won't actually return control)
# --------------------------------------------------------------------------
def run_with_timeout(fn, timeout_sec, *args, **kwargs):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_sec)
        executor.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        raise TimeoutError(f"{fn.__name__} exceeded {timeout_sec}s timeout")


# --------------------------------------------------------------------------
# Data download: ALL sectors, no cap
# --------------------------------------------------------------------------
def download_all_sectors(tic_id, mast_timeout_sec):
    def _search_and_download():
        search = lk.search_targetpixelfile(f"TIC {tic_id}", author="SPOC", exptime=120)
        if len(search) == 0:
            return None, None
        tpfs = search.download_all()
        return search, tpfs

    search, tpfs = run_with_timeout(_search_and_download, mast_timeout_sec)
    if tpfs is None or len(tpfs) == 0:
        return None, []

    lcs = []
    per_sector_meta = []
    for tpf in tpfs:
        try:
            lc = tpf.to_lightcurve(aperture_mask="pipeline")
            lc = lc.remove_nans()
            lcs.append(lc)
            per_sector_meta.append({
                "sector": int(tpf.sector),
                "n_points": len(lc.time),
                "cadence": "2min",
            })
        except Exception as e:
            print(f"  [warn] failed to build light curve for sector {tpf.sector}: {e}")

    return lcs, per_sector_meta


def upward_flare_clip(lc, sigma=5.0):
    """MAD-based upward-only clipping. Does not remove downward (transit) points."""
    flux = lc.flux.value
    med = np.nanmedian(flux)
    std = mad_std(flux, ignore_nan=True)
    upper_mask = flux < (med + sigma * std)  # keep points below the upper threshold
    return lc[upper_mask]


def stitch_light_curves(lcs):
    collection = lk.LightCurveCollection(lcs)
    stitched = collection.stitch()
    stitched = stitched.remove_nans()
    return stitched


# --------------------------------------------------------------------------
# BLS: wide, fine period grid (no narrow +/-10% bracket)
# --------------------------------------------------------------------------
def run_bls_wide(lc, min_period, max_period, durations):
    """Direct astropy BoxLeastSquares call, bypassing lightkurve's
    to_periodogram wrapper -- its internal auto-grid explodes on long
    multi-sector baselines regardless of what period/frequency_factor
    you pass it. This gives us full control over grid size."""
    time = lc.time.value
    flux = lc.flux.value

    # astropy requires every duration to be strictly shorter than the
    # shortest period tested -- clip the duration grid to leave headroom
    max_allowed_duration = min_period * 0.8
    safe_durations = durations[durations < max_allowed_duration]
    if len(safe_durations) == 0:
        # min_period is very small -- fall back to a tiny duration grid
        safe_durations = np.linspace(max_allowed_duration * 0.2, max_allowed_duration * 0.9, 5)

    model = BoxLeastSquares(time, flux)
    periods = np.linspace(min_period, max_period, 20000)
    result = model.power(periods, safe_durations)
    return result


# --------------------------------------------------------------------------
# TLS: full resolution, full period range, high oversampling
# --------------------------------------------------------------------------
def run_tls_full(time, flux, min_period, max_period, oversampling,
                  duration_grid_step, tls_timeout_sec, use_threads=True):
    def _run():
        model = transitleastsquares(time, flux)
        return model.power(
            period_min=min_period,
            period_max=max_period,
            oversampling_factor=oversampling,
            duration_grid_step=duration_grid_step,
            use_threads=os.cpu_count() if use_threads else 1,
        )
    return run_with_timeout(_run, tls_timeout_sec)


# --------------------------------------------------------------------------
# Vetting metrics for one TLS result
# --------------------------------------------------------------------------
def odd_even_depth_diff(results):
    try:
        odd = results.depth_mean_odd[0]
        even = results.depth_mean_even[0]
        odd_err = results.depth_mean_odd[1]
        even_err = results.depth_mean_even[1]
        diff = abs(odd - even)
        combined_err = np.sqrt(odd_err**2 + even_err**2)
        sigma = diff / combined_err if combined_err > 0 else np.nan
        return {
            "odd_depth": float(odd), "even_depth": float(even),
            "diff_sigma": float(sigma),
        }
    except Exception:
        return {"odd_depth": None, "even_depth": None, "diff_sigma": None}


def build_summary(tic_id, results, sector_meta, candidate_idx):
    return {
        "tic_id": tic_id,
        "candidate": candidate_idx,
        "period_days": float(results.period),
        "period_uncertainty": float(getattr(results, "period_uncertainty", np.nan)),
        "T0": float(results.T0),
        "duration_days": float(results.duration),
        "depth": float(results.depth),
        "SDE": float(results.SDE),
        "snr": float(getattr(results, "snr", np.nan)),
        "odd_even_transit_mismatch": odd_even_depth_diff(results),
        "n_transits": int(getattr(results, "transit_count", -1)),
        "sectors_used": sector_meta,
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_full_light_curve(lc, out_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(lc.time.value, lc.flux.value, s=1, color="black")
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized flux")
    ax.set_title("Full stitched light curve (all sectors)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_sector(lcs, out_path):
    n = len(lcs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, lc in zip(axes, lcs):
        sector = getattr(lc, "sector", "?")
        ax.scatter(lc.time.value, lc.flux.value, s=1, color="black")
        ax.set_ylabel(f"Sector {sector}")
    axes[-1].set_xlabel("Time (BTJD)")
    fig.suptitle("Per-sector light curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bls_periodogram(bls, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bls.period, bls.power, color="black", linewidth=0.6)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS power")
    ax.set_title("BLS periodogram (wide search)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tls_periodogram(results, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(results.periods, results.power, color="black", linewidth=0.6)
    ax.axvline(results.period, color="red", linestyle="--", alpha=0.7,
                label=f"Best period = {results.period:.4f} d")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("TLS power (SDE)")
    ax.set_title("TLS periodogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phase_fold(time, flux, results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    phase = ((time - results.T0 + 0.5 * results.period) % results.period) / results.period
    axes[0].scatter(phase, flux, s=2, color="black", alpha=0.5)
    axes[0].set_xlabel("Phase")
    axes[0].set_ylabel("Normalized flux")
    axes[0].set_title(f"Full phase fold, P = {results.period:.5f} d")

    zoom_width = 3 * (results.duration / results.period)
    zoom_mask = (phase > 0.5 - zoom_width) & (phase < 0.5 + zoom_width)
    axes[1].scatter(phase[zoom_mask], flux[zoom_mask], s=4, color="black", alpha=0.6)
    if hasattr(results, "model_folded_phase") and hasattr(results, "model_folded_model"):
        axes[1].plot(results.model_folded_phase, results.model_folded_model,
                      color="red", linewidth=1.5)
    axes[1].set_xlabel("Phase")
    axes[1].set_title("Zoom on transit + model")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_odd_even(time, flux, results, out_path):
    period, T0, duration = results.period, results.T0, results.duration
    epoch = np.round((time - T0) / period)
    is_odd = (epoch % 2 == 1)

    phase = ((time - T0 + 0.5 * period) % period) / period
    zoom_width = 3 * (duration / period)
    zoom_mask = (phase > 0.5 - zoom_width) & (phase < 0.5 + zoom_width)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(phase[zoom_mask & is_odd], flux[zoom_mask & is_odd],
               s=6, color="tab:blue", alpha=0.6, label="Odd transits")
    ax.scatter(phase[zoom_mask & ~is_odd], flux[zoom_mask & ~is_odd],
               s=6, color="tab:orange", alpha=0.6, label="Even transits")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized flux")
    ax.set_title("Odd vs even transit depth check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_secondary_eclipse_check(time, flux, results, out_path):
    """Look at phase 0 (secondary eclipse position) for a dip -- a real
    secondary eclipse of similar depth to the primary suggests an eclipsing
    binary rather than a planet."""
    period, T0, duration = results.period, results.T0, results.duration
    phase = ((time - T0) % period) / period  # secondary at phase 0/1 boundary, primary at 0.5
    phase_centered = np.where(phase > 0.5, phase - 1, phase)

    zoom_width = 3 * (duration / period)
    zoom_mask = np.abs(phase_centered) < zoom_width

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(phase_centered[zoom_mask], flux[zoom_mask], s=4, color="black", alpha=0.6)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Expected secondary position")
    ax.set_xlabel("Phase (centered on secondary position)")
    ax.set_ylabel("Normalized flux")
    ax.set_title("Secondary eclipse check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main per-candidate workflow
# --------------------------------------------------------------------------
def analyze_candidate(time, flux, sector_meta, min_period, max_period, args, out_dir, tic_id, idx):
    print(f"  Running TLS for candidate {idx} (period range {min_period:.2f}-{max_period:.2f} d)...")
    try:
        results = run_tls_full(
            time, flux, min_period, max_period,
            oversampling=args.tls_oversampling,
            duration_grid_step=args.tls_duration_grid_step,
            tls_timeout_sec=args.tls_timeout_sec,
            use_threads=args.tls_use_threads,
        )
    except TimeoutError as e:
        print(f"  [warn] {e} -- skipping candidate {idx}")
        return None, None

    cand_dir = out_dir / f"candidate_{idx}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    plot_tls_periodogram(results, cand_dir / "tls_periodogram.png")
    plot_phase_fold(time, flux, results, cand_dir / "phase_fold.png")
    plot_odd_even(time, flux, results, cand_dir / "odd_even.png")
    plot_secondary_eclipse_check(time, flux, results, cand_dir / "secondary_eclipse_check.png")

    summary = build_summary(tic_id, results, sector_meta, idx)
    with open(cand_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Candidate {idx}: P={results.period:.5f} d, SDE={results.SDE:.2f}, "
          f"depth={results.depth:.6f}, odd/even diff sigma="
          f"{summary['odd_even_transit_mismatch']['diff_sigma']}")

    return results, summary


def mask_out_signal(time, flux, results):
    """Remove the in-transit points of a found signal so a second search
    isn't dominated by the first (strongest) periodic signal."""
    mask = transit_mask(time, results.period, results.duration, results.T0)
    return time[~mask], flux[~mask]


# --------------------------------------------------------------------------
# CLI / orchestration
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Deep-dive analysis of a single TESS target")
    parser.add_argument("tic_id", type=int, help="TIC ID (integer, no 'TIC' prefix)")
    parser.add_argument("--out-dir", type=str, default="tess_results/deep_dive")
    parser.add_argument("--min-period", type=float, default=DEFAULT_MIN_PERIOD)
    parser.add_argument("--max-period", type=float, default=DEFAULT_MAX_PERIOD)
    parser.add_argument("--max-planets", type=int, default=1,
                         help="How many signals to search for (masks each found signal and re-searches)")
    parser.add_argument("--tls-oversampling", type=int, default=DEFAULT_TLS_OVERSAMPLING)
    parser.add_argument("--tls-duration-grid-step", type=float, default=DEFAULT_TLS_DURATION_GRID_STEP)
    parser.add_argument("--tls-timeout-sec", type=int, default=DEFAULT_TLS_TIMEOUT_SEC)
    parser.add_argument("--mast-timeout-sec", type=int, default=DEFAULT_MAST_TIMEOUT_SEC)
    parser.add_argument("--tls-use-threads", action="store_true", default=True)
    parser.add_argument("--no-bin", action="store_true",
                         help="Never bin the light curve before TLS, regardless of point count")
    parser.add_argument("--flare-clip-sigma", type=float, default=5.0)
    args = parser.parse_args()

    tic_id = args.tic_id
    out_dir = Path(args.out_dir) / f"TIC_{tic_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Deep dive: TIC {tic_id} ===")
    print("Downloading ALL available SPOC 2-min sectors (no sector cap)...")
    try:
        lcs, sector_meta = run_with_timeout(
            download_all_sectors, args.mast_timeout_sec, tic_id, args.mast_timeout_sec
        )
    except TimeoutError as e:
        print(f"[error] {e}")
        sys.exit(1)

    if not lcs:
        print("[error] No SPOC 2-minute light curves found for this TIC.")
        sys.exit(1)

    print(f"Found {len(lcs)} sector(s): {[m['sector'] for m in sector_meta]}")

    print("Applying upward-only MAD flare clipping per sector...")
    lcs = [upward_flare_clip(lc, sigma=args.flare_clip_sigma) for lc in lcs]

    print("Stitching sectors together...")
    stitched = stitch_light_curves(lcs)

    plot_full_light_curve(stitched, out_dir / "full_light_curve.png")
    plot_per_sector(lcs, out_dir / "per_sector_light_curves.png")

    time = stitched.time.value
    flux = stitched.flux.value
    n_points = len(time)
    print(f"Total points after stitching: {n_points}")

    if not args.no_bin and n_points > DEFAULT_BIN_POINT_CAP:
        bin_factor = int(np.ceil(n_points / DEFAULT_BIN_POINT_CAP))
        print(f"Point count exceeds cap; binning by factor {bin_factor} before BLS/TLS "
              f"(use --no-bin to force full resolution)")
        binned = stitched.bin(time_bin_size=bin_factor * np.median(np.diff(time)))
        time = binned.time.value
        flux = binned.flux.value
    else:
        print("Using full-resolution light curve for BLS/TLS (no binning).")

    print("Running wide-grid BLS as a first look...")
    bls = run_bls_wide(stitched, args.min_period, args.max_period, DEFAULT_BLS_DURATIONS)
    plot_bls_periodogram(bls, out_dir / "bls_periodogram.png")
    bls_best_period = bls.period[np.argmax(bls.power)]
    print(f"BLS best period (informational only, TLS below is NOT restricted to this): "
          f"{bls_best_period:.4f} d")

    all_summaries = []
    search_time, search_flux = time.copy(), flux.copy()
    for idx in range(1, args.max_planets + 1):
        results, summary = analyze_candidate(
            search_time, search_flux, sector_meta,
            args.min_period, args.max_period, args, out_dir, tic_id, idx
        )
        if results is None:
            break
        all_summaries.append(summary)
        if idx < args.max_planets:
            search_time, search_flux = mask_out_signal(search_time, search_flux, results)

    with open(out_dir / "deep_dive_summary.json", "w") as f:
        json.dump({
            "tic_id": tic_id,
            "n_sectors": len(lcs),
            "sectors": sector_meta,
            "bls_best_period_informational": float(bls_best_period),
            "candidates": all_summaries,
        }, f, indent=2)

    print(f"\nDone. Results saved to: {out_dir}")
    if not all_summaries:
        print("No signal survived the TLS timeout/search. Consider a longer --tls-timeout-sec "
              "or a narrower period range if you have a specific period in mind.")


if __name__ == "__main__":
    main()