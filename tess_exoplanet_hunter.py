"""
TESS Exoplanet Hunter
=====================
Combines the manual workflow (MAST catalog search -> exo.MAST check ->
target pixel file download -> BLS search) into one script.

WORKFLOW
  1. Pull candidate stars from the TESS Input Catalog (TIC) using your own
     size/mass/temperature/brightness criteria (astroquery.mast.Catalogs).
  2. Cross-check every candidate against the current TESS Objects of
     Interest (TOI) list from the NASA Exoplanet Archive, in one batch
     query, and drop any star that already has a known candidate/planet.
  3. For each remaining star, download every available SPOC 2-minute
     target pixel file (all sectors, not just one), stitch them into a
     single light curve, and search it with BLS (and TLS if installed).
  4. Save a summary CSV (sorted by TIC ID) plus diagnostic plots per star.

RUN IN GOOGLE COLAB
    !pip install lightkurve astroquery transitleastsquares --quiet
    !python tess_exoplanet_hunter.py
  (or paste the CONFIG block + function defs + main() call into separate
  cells if you'd rather run it interactively.)

NOTES ON WHAT CHANGED FROM THE ORIGINAL NOTEBOOK
  - author="TESS" is not a real pipeline name; TESS's official 2-minute
    pipeline is "SPOC". cadence="long" / quarter=... are Kepler-only
    keywords - TESS uses "sector". Mixing these is very likely why your
    manual search() calls were coming back empty and you had to fall back
    to hand-downloading zips.
  - lightkurve's search + .download()/.download_all() talks to MAST
    directly - there's no need for the manual zip/urlopen helper at all
    once the search call is built correctly.

RE-RUNNING THE SCRIPT
  Every TIC ID that gets analyzed (whether or not it produced a result)
  is recorded in <output_dir>/analyzed_tics.json. On the next run those
  IDs are skipped automatically, so re-running the script explores a
  fresh slice of stars instead of repeating the same ones. This file is
  also backfilled automatically from any TIC_<id> plot folders already
  sitting in output_dir - including ones from runs before this tracking
  feature existed - so nothing you've already downloaded gets redone.
  Delete analyzed_tics.json (or set skip_previously_analyzed=False in
  CONFIG) to start over from scratch.

  If star_criteria don't leave enough fresh stars after filtering out
  known TOIs and already-analyzed TICs, the script automatically widens
  Tmag/rad/Teff a bit and re-queries (up to max_widen_attempts times),
  so a run keeps finding new stars instead of just stopping.

  Results accumulate in <output_dir>/search_summary_all.csv across every
  run (this run's own results also land in search_summary.csv). Re-
  analyzing a TIC ID replaces its old row with the newer one.

  Flares: light curves are also passed through an upward-only outlier
  clip before the transit search, since bright upward spikes (flares)
  can't ever be transits and were generating false BLS/TLS peaks.
"""

import os
import json
import logging
import warnings
import concurrent.futures

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tess_hunter")

import lightkurve as lk
from astroquery.mast import Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

try:
    from transitleastsquares import transitleastsquares
    HAVE_TLS = True
except ImportError:
    HAVE_TLS = False
    log.info("transitleastsquares not installed - BLS only. `pip install transitleastsquares` to enable it.")


def with_timeout(func, timeout_sec, *args, **kwargs):
    """
    Runs func in a background thread with a hard wall-clock timeout, and
    raises TimeoutError if it's exceeded. astroquery/lightkurve don't
    expose a native timeout on their MAST calls, so a stalled connection
    (bad wifi, an overloaded archive server, etc.) would otherwise hang
    the script indefinitely with no way to recover automatically.

    Caveat worth knowing: Python can't force-kill a running thread, so if
    the underlying network call is genuinely stuck (not just slow), that
    background thread keeps running after this function returns control
    to you. It's harmless while the script keeps working through other
    stars, but a truly hung thread can make the script pause for a moment
    on final exit while Python waits for it. A timeout is still far
    better than the alternative of blocking forever mid-run.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        raise TimeoutError(f"{getattr(func, '__name__', 'call')} exceeded {timeout_sec}s timeout")
    executor.shutdown(wait=False)
    return result


# ----------------------------------------------------------------------
# 1. CONFIGURATION - edit this block to match what you're hunting for
# ----------------------------------------------------------------------
CONFIG = {
    # Criteria passed straight to astroquery.mast.Catalogs.query_criteria
    # (catalog="Tic"). This replaces the MAST portal's "Advanced Search".
    # Full column list: https://outerspace.stsci.edu/display/TESS/TIC+v8.2+and+CTL+v8.xx+Data+Release+Notes
    "star_criteria": {
        "objType": "STAR",
        "Tmag": [8, 11],       # brightness: bright enough for good S/N,
                                 # not so bright the pipeline saturates
        "rad": [0.1, 0.8],      # stellar radius (R_sun) -> K/M dwarfs;
                                 # small planets are much easier to see
                                 # transiting a small star
        "Teff": [3000, 5200],   # cool dwarfs
    },
    "max_candidate_stars": 40,   # how many stars to analyze in this run
    "catalog_pool_size": 500,    # how many stars to pull from the catalog
                                  # as a pool - larger than max_candidate_stars
                                  # so repeated runs always have a fresh batch
                                  # to draw from without widening criteria
    "sort_catalog_by": "Tmag",   # brightest first = best S/N per transit
    "skip_previously_analyzed": True,   # don't re-analyze TIC IDs already
                                         # tried in a previous run (tracked in
                                         # <output_dir>/analyzed_tics.json)
    "max_widen_attempts": 4,     # if too few fresh stars remain, widen
                                  # star_criteria and retry, this many times
    "widen_step_fraction": 0.25, # how much to widen each range per attempt
                                  # (as a fraction of its current width)
    "use_priority_ranking": True,  # rank fresh candidates by a detectability
                                    # score (brightness + small radius)
                                    # instead of brightness alone

    # Download / analysis settings
    "author": "SPOC",            # official TESS 2-min pipeline product
    "max_sectors_per_star": 6,   # cap sectors downloaded per star
    "search_timeout_sec": 120,   # hard cap on the MAST search call
    "download_timeout_sec": 300, # hard cap on the MAST download call -
                                  # without this a stalled connection can
                                  # hang the whole run indefinitely
    "flare_clip_sigma": 4,       # clip flux points more than this many MAD
                                  # above the median before searching - only
                                  # ever removes brightenings (flares), which
                                  # can never be transits, so it's safe
    "period_min": 0.5,           # days
    "period_max": 15.0,          # days
    "period_samples": 10000,
    "run_tls": True,             # also run Transit Least Squares (better
                                  # sensitivity to small planets than BLS)
    "max_points_for_tls": 100_000,  # TLS gets slow/memory-heavy on very
                                     # long multi-sector baselines; light
                                     # curves bigger than this get binned
                                     # down before TLS (BLS still runs on
                                     # the full-resolution data)
    "tls_timeout_sec": 900,       # hard cap per star on the TLS search
                                   # itself (separate from the download
                                   # timeouts above) - TLS is the slowest
                                   # single step in the whole pipeline
    "tls_narrow_around_bls": True,   # search TLS only near BLS's best
                                      # period instead of the full range -
                                      # much faster, but it means TLS is
                                      # refining/confirming BLS's answer
                                      # rather than doing a fully
                                      # independent search that could catch
                                      # a different period BLS missed. Set
                                      # False if you want TLS's full,
                                      # slower, independent search instead.
    "tls_narrow_fraction": 0.10,  # +/-10% around the BLS best period
    "tls_oversampling_factor": 3,
    "tls_duration_grid_step": 1.1,
    "tls_use_threads": min(os.cpu_count() or 4, 8),  # TLS spins up a real
                                   # multiprocessing pool for this - more
                                   # threads isn't always faster on a
                                   # laptop once you account for process
                                   # start-up overhead per star, so this
                                   # caps at 8 rather than using every core

    # Creates the results folder right next to this script, regardless of
    # which folder you happened to launch the terminal from.
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "tess_results"),
}


# ----------------------------------------------------------------------
# 2. STAR SELECTION  (replaces manual "Advanced Search" on the MAST portal)
# ----------------------------------------------------------------------
def find_candidate_stars(criteria, max_stars, sort_by):
    log.info(f"Querying the TIC with criteria: {criteria}")
    table = Catalogs.query_criteria(catalog="Tic", **criteria)
    df = table.to_pandas()
    log.info(f"{len(df)} stars in the TIC matched your criteria.")

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True)
    df = df.head(max_stars).reset_index(drop=True)

    keep_cols = [c for c in ["ID", "ra", "dec", "Tmag", "rad", "mass", "Teff"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["ID"])
    df["ID"] = df["ID"].astype(int)
    log.info(f"Keeping the {len(df)} brightest/best matches for analysis.")
    return df


def _widen_criteria(criteria, step_fraction=0.25):
    """
    Expands every [min, max] range in criteria outward by step_fraction of
    its current width, so the next catalog query covers more stars.
    Non-range entries (like objType) are left untouched.
    """
    widened = {}
    for key, val in criteria.items():
        if isinstance(val, list) and len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
            lo, hi = val
            span = hi - lo
            pad = span * step_fraction if span > 0 else max(abs(lo), 1) * step_fraction
            new_lo = round(max(0, lo - pad), 3)
            new_hi = round(hi + pad, 3)
            widened[key] = [new_lo, new_hi]
            log.info(f"  widening {key}: [{lo}, {hi}] -> [{new_lo}, {new_hi}]")
        else:
            widened[key] = val
    return widened


def rank_by_detection_priority(df):
    """
    Ranks candidate stars by a rough 'how likely is a transit here to
    actually be detectable' score, rather than brightness alone: it
    rewards both brightness (better S/N) and smaller stellar radius
    (a given planet blocks a bigger fraction of a smaller star's light,
    so the transit is deeper and easier to pull out of the noise).

    Worth knowing: this optimizes for *detectability*, which is what
    matters for this pipeline. It's not the same as geometric transit
    *probability*, which actually favors larger stars (bigger target to
    hit) - a genuine trade-off in transit surveys, not a simplification
    made here for convenience.
    """
    d = df.dropna(subset=["Tmag", "rad"]).copy()
    if d.empty:
        return d
    tmag_span = d["Tmag"].max() - d["Tmag"].min()
    rad_span = d["rad"].max() - d["rad"].min()
    d["tmag_score"] = 1 - (d["Tmag"] - d["Tmag"].min()) / (tmag_span if tmag_span > 0 else 1)
    d["radius_score"] = 1 - (d["rad"] - d["rad"].min()) / (rad_span if rad_span > 0 else 1)
    d["priority_score"] = 0.5 * d["tmag_score"] + 0.5 * d["radius_score"]
    return d.sort_values("priority_score", ascending=False).reset_index(drop=True)


def find_fresh_targets(config, known_ids, analyzed_ids):
    """
    Pulls candidate stars from the TIC, filters out known TOIs and
    already-analyzed TICs, and - if that doesn't leave enough fresh
    targets - automatically widens star_criteria and retries (up to
    max_widen_attempts times) instead of just running dry.
    """
    criteria = dict(config["star_criteria"])
    attempt = 0
    while True:
        pool = find_candidate_stars(criteria, config["catalog_pool_size"], config["sort_catalog_by"])
        pool = filter_new_targets(pool, known_ids)
        pool = pool[~pool["ID"].isin(analyzed_ids)].reset_index(drop=True)

        if len(pool) >= config["max_candidate_stars"] or attempt >= config.get("max_widen_attempts", 4):
            return pool

        attempt += 1
        log.info(
            f"Only {len(pool)} fresh star(s) available after filtering - "
            f"widening star_criteria (attempt {attempt}/{config.get('max_widen_attempts', 4)}) ..."
        )
        criteria = _widen_criteria(criteria, config.get("widen_step_fraction", 0.25))


# ----------------------------------------------------------------------
# 3. KNOWN-PLANET CHECK  (replaces manually pasting each TIC into exo.MAST)
# ----------------------------------------------------------------------
def load_known_tic_ids():
    """
    Downloads the full TESS Objects of Interest (TOI) table once and
    returns the set of TIC IDs that already have a candidate or confirmed
    planet on record. Checking everything in one batch query is both far
    faster and more reliable than looking up TIC IDs one at a time on the
    exo.MAST website.
    """
    log.info("Downloading current TOI list from the NASA Exoplanet Archive ...")
    toi = NasaExoplanetArchive.query_criteria(table="toi", select="tid,toi,tfopwg_disp")
    toi_df = toi.to_pandas()
    known_ids = set(toi_df["tid"].dropna().astype(int))
    log.info(f"{len(known_ids)} TIC IDs already have a TOI on record (candidate, false positive, or confirmed).")
    return known_ids


def filter_new_targets(star_df, known_ids):
    mask = ~star_df["ID"].isin(known_ids)
    new_df = star_df[mask].reset_index(drop=True)
    log.info(f"{len(new_df)} / {len(star_df)} candidate stars have no existing TOI - these are worth searching.")
    return new_df


# ----------------------------------------------------------------------
# 3b. ALREADY-ANALYZED TRACKING  (so re-running explores fresh stars)
# ----------------------------------------------------------------------
def _analyzed_tics_path(outdir):
    return os.path.join(outdir, "analyzed_tics.json")


def load_analyzed_tics(outdir):
    path = _analyzed_tics_path(outdir)
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_analyzed_tics(outdir, tic_ids):
    path = _analyzed_tics_path(outdir)
    combined = sorted(load_analyzed_tics(outdir) | {int(t) for t in tic_ids})
    with open(path, "w") as f:
        json.dump(combined, f)


def backfill_analyzed_from_output_dir(outdir):
    """
    Scans output_dir for TIC_<id> plot folders left behind by ANY previous
    run - including runs from before analyzed_tics.json existed at all -
    and folds their TIC IDs in. Safe to call every run: it only ever adds
    IDs, never removes any. This is what makes sure stars you already
    checked (in this script or by eye in a previous session) don't get
    re-downloaded and re-analyzed.
    """
    if not os.path.isdir(outdir):
        return
    found = []
    for name in os.listdir(outdir):
        if name.startswith("TIC_") and os.path.isdir(os.path.join(outdir, name)):
            try:
                found.append(int(name[len("TIC_"):]))
            except ValueError:
                continue
    if not found:
        return
    before = len(load_analyzed_tics(outdir))
    save_analyzed_tics(outdir, found)
    added = len(load_analyzed_tics(outdir)) - before
    if added:
        log.info(f"Backfilled {added} previously-analyzed star(s) found on disk into analyzed_tics.json.")


def mark_manually_reviewed(tic_ids, outdir=None):
    """
    Call this yourself for any TIC ID you've reviewed (e.g. by eye, in
    chat) that never got a TIC_<id> folder written - so it's skipped in
    future runs too. Example, from a Python prompt in this folder:
        import tess_exoplanet_hunter as m
        m.mark_manually_reviewed([147261632, 386838171, 126741967])
    """
    outdir = outdir or CONFIG["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    save_analyzed_tics(outdir, tic_ids)
    log.info(f"Marked {len(tic_ids)} TIC ID(s) as reviewed in {_analyzed_tics_path(outdir)}.")


# ----------------------------------------------------------------------
# 4. DOWNLOAD + ANALYZE  (replaces the manual zip download + "Analyze the star" cell)
# ----------------------------------------------------------------------
def odd_even_depth_check(time, flux, period, t0, duration):
    """
    Compares the depth of odd-numbered vs. even-numbered transits at the
    best-fit period. A large mismatch is a classic sign of an eclipsing
    binary (alternating primary/secondary eclipses) rather than a planet,
    where every transit should be the same depth.
    """
    try:
        cycle = np.floor((time - t0) / period + 0.5).astype(int)
        phase = (time - t0 + 0.5 * period) % period - 0.5 * period
        in_transit = np.abs(phase) < (duration / 2.0)

        odd = in_transit & (cycle % 2 != 0)
        even = in_transit & (cycle % 2 == 0)
        if odd.sum() < 3 or even.sum() < 3:
            return {"odd_depth": None, "even_depth": None, "odd_even_flag": None}

        odd_depth = float(1 - np.nanmedian(flux[odd]))
        even_depth = float(1 - np.nanmedian(flux[even]))
        biggest = max(abs(odd_depth), abs(even_depth), 1e-8)
        flag = abs(odd_depth - even_depth) > 0.5 * biggest
        return {"odd_depth": odd_depth, "even_depth": even_depth, "odd_even_flag": bool(flag)}
    except Exception:
        return {"odd_depth": None, "even_depth": None, "odd_even_flag": None}


def clip_upward_flares(lc, sigma=4):
    """
    Removes upward-only flux outliers (flares) before the transit search.
    A real transit only ever dims a star, so - unlike a normal symmetric
    sigma clip - this can never accidentally remove a genuine transit, only
    the brightening spikes that flares produce.
    """
    flux = lc.flux.value
    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med)) * 1.4826  # normal-consistent std estimate
    if not np.isfinite(mad) or mad == 0:
        return lc
    threshold = med + sigma * mad
    keep = flux < threshold
    n_clipped = int((~keep).sum())
    if n_clipped:
        log.info(f"  clipped {n_clipped} upward flare-like point(s) (> {sigma} MAD above median)")
    return lc[keep]


def manual_bin(time_arr, flux_arr, bin_factor):
    """
    Simple index-based downsampling: groups every bin_factor consecutive
    points (already time-sorted) and averages each group, returning plain
    numpy arrays rather than a LightCurve.

    Deliberately not using lightkurve's .bin() here - it delegates to
    astropy's aggregate_downsample, which has been observed to raise a
    ValueError ("shape mismatch") on some multi-sector light curves with
    large time gaps between sectors, depending on the installed astropy
    version. This sidesteps that code path entirely. We're only trying to
    reduce point count before TLS, not produce scientifically calibrated
    time bins, so simple index-based grouping is all that's needed here.
    """
    n = len(time_arr)
    n_bins = n // bin_factor
    trimmed = n_bins * bin_factor
    t = time_arr[:trimmed].reshape(n_bins, bin_factor).mean(axis=1)
    f = flux_arr[:trimmed].reshape(n_bins, bin_factor).mean(axis=1)
    if trimmed < n:  # keep any leftover points rather than silently dropping them
        t = np.concatenate([t, time_arr[trimmed:]])
        f = np.concatenate([f, flux_arr[trimmed:]])
        order = np.argsort(t)
        t, f = t[order], f[order]
    return t, f


def analyze_star(tic_id, config, outdir):
    """
    Downloads all available SPOC 2-min target pixel files for a TIC ID
    (all sectors, capped by max_sectors_per_star), stitches them into one
    light curve, and searches it for periodic transits with BLS (and TLS
    if available). Returns a result dict, or None if there was no usable
    data.
    """
    log.info(f"  TIC {tic_id}: stage=search")
    try:
        sr = with_timeout(
            lk.search_targetpixelfile, config.get("search_timeout_sec", 120),
            f"TIC {tic_id}", mission="TESS", author=config["author"],
        )
    except TimeoutError as e:
        log.warning(f"TIC {tic_id}: search timed out ({e}) - skipping.")
        return None
    except Exception as e:
        log.warning(f"TIC {tic_id}: search failed ({e})")
        return None

    if len(sr) == 0:
        log.info(f"TIC {tic_id}: no SPOC 2-min target pixel files available - skipping.")
        return None

    n = min(len(sr), config["max_sectors_per_star"])
    log.info(f"  TIC {tic_id}: stage=download ({n} sector(s))")
    try:
        tpfs = with_timeout(sr[:n].download_all, config.get("download_timeout_sec", 300))
    except TimeoutError as e:
        log.warning(f"TIC {tic_id}: download timed out ({e}) - skipping. "
                     f"(the background download thread may still finish in the background, "
                     f"harmlessly, since Python can't force-kill it - it just won't block the run)")
        return None
    except Exception as e:
        log.warning(f"TIC {tic_id}: download failed ({e})")
        return None
    log.info(f"  TIC {tic_id}: stage=download complete, extracting light curves")

    lcs = []
    for tpf in tpfs:
        try:
            lcs.append(tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask))
        except Exception:
            continue
    if not lcs:
        log.warning(f"TIC {tic_id}: could not extract a light curve from any TPF.")
        return None

    log.info(f"  TIC {tic_id}: stage=stitching {len(lcs)} sector(s)")
    # NOTE: deliberately no blanket remove_outliers() here - see
    # clip_upward_flares() below instead. A normal symmetric sigma clip
    # (e.g. sigma=5) will happily strip out your transit itself if the
    # star is bright/quiet enough that the transit depth is many times the
    # per-point noise - exactly the high-S/N targets this script prioritizes.
    lc = lk.LightCurveCollection(lcs).stitch().remove_nans()
    lc = clip_upward_flares(lc, config.get("flare_clip_sigma", 4))

    # flatten()'s smooth-trend division handles instrumental drift without
    # touching short dips, which is what we actually want here.
    flat_lc = lc.flatten(window_length=901)
    n_points = len(flat_lc.time.value)
    log.info(f"  TIC {tic_id}: stage=BLS ({n_points} points, {len(lcs)} sector(s))")

    period_grid = np.linspace(config["period_min"], config["period_max"], config["period_samples"])
    bls = flat_lc.to_periodogram(method="bls", period=period_grid, frequency_factor=500)
    log.info(f"  TIC {tic_id}: stage=BLS complete")

    result = {
        "TIC": tic_id,
        "n_sectors": len(lcs),
        "n_points": n_points,
        "baseline_days": float(flat_lc.time.value.max() - flat_lc.time.value.min()),
        "bls_period_d": float(bls.period_at_max_power.value),
        "bls_t0_btjd": float(bls.transit_time_at_max_power.value),
        "bls_duration_d": float(bls.duration_at_max_power.value),
        "bls_depth": float(bls.depth_at_max_power),
        "bls_power": float(bls.max_power.value),
    }

    result.update(
        odd_even_depth_check(
            flat_lc.time.value, flat_lc.flux.value,
            result["bls_period_d"], result["bls_t0_btjd"], result["bls_duration_d"],
        )
    )

    if config["run_tls"] and HAVE_TLS:
        # TLS gets slow (and memory-heavy) on very long multi-sector
        # baselines. Bin down to max_points_for_tls first if needed - BLS
        # above already ran on the full-resolution data, so nothing is
        # lost there. Using manual_bin() (plain numpy) rather than
        # lightkurve's .bin(), which can raise a shape-mismatch error from
        # astropy's aggregate_downsample on some gapped multi-sector light
        # curves depending on the installed astropy version.
        tls_time, tls_flux = flat_lc.time.value, flat_lc.flux.value
        max_points = config.get("max_points_for_tls", 100_000)
        if n_points > max_points:
            bin_factor = int(np.ceil(n_points / max_points))
            tls_time, tls_flux = manual_bin(tls_time, tls_flux, bin_factor)
            log.info(
                f"  TIC {tic_id}: {n_points} points exceeds max_points_for_tls "
                f"({max_points}) - binned to {len(tls_time)} points "
                f"(factor {bin_factor}) before TLS"
            )

        # Narrow TLS's search around BLS's best period rather than
        # re-searching the whole range - much faster, at the cost of TLS
        # refining/confirming what BLS already found instead of running a
        # fully independent search. Controlled by tls_narrow_around_bls.
        if config.get("tls_narrow_around_bls", True):
            frac = config.get("tls_narrow_fraction", 0.10)
            best_bls_period = result["bls_period_d"]
            tls_period_min = max(config["period_min"], best_bls_period * (1 - frac))
            tls_period_max = min(config["period_max"], best_bls_period * (1 + frac))
        else:
            tls_period_min = config["period_min"]
            tls_period_max = config["period_max"]

        log.info(
            f"  TIC {tic_id}: stage=TLS ({len(tls_time)} points, "
            f"period {tls_period_min:.2f}-{tls_period_max:.2f}d)"
        )
        try:
            model = transitleastsquares(tls_time, tls_flux)
            tls_res = with_timeout(
                model.power, config.get("tls_timeout_sec", 900),
                period_min=tls_period_min,
                period_max=tls_period_max,
                oversampling_factor=config.get("tls_oversampling_factor", 3),
                duration_grid_step=config.get("tls_duration_grid_step", 1.1),
                use_threads=config.get("tls_use_threads", 4),
                show_progress_bar=False,
            )
            result["tls_period_d"] = float(tls_res.period)
            result["tls_sde"] = float(tls_res.SDE)
            result["tls_depth"] = float(1 - tls_res.depth)
            log.info(f"  TIC {tic_id}: stage=TLS complete")
        except TimeoutError as e:
            log.warning(f"TIC {tic_id}: TLS timed out ({e}) - keeping BLS result only.")
        except Exception as e:
            log.warning(f"TIC {tic_id}: TLS failed ({e})")

    _save_plots(tic_id, flat_lc, bls, result, outdir)
    return result


def _save_plots(tic_id, flat_lc, bls, result, outdir):
    star_dir = os.path.join(outdir, f"TIC_{tic_id}")
    os.makedirs(star_dir, exist_ok=True)
    try:
        ax = flat_lc.plot()
        ax.figure.savefig(os.path.join(star_dir, "flattened_lightcurve.png"), dpi=120)
        ax.figure.clf()

        ax = bls.plot()
        ax.figure.savefig(os.path.join(star_dir, "bls_periodogram.png"), dpi=120)
        ax.figure.clf()

        ax = flat_lc.fold(period=result["bls_period_d"], epoch_time=result["bls_t0_btjd"]).scatter()
        ax.set_xlim(-0.3, 0.3)
        ax.figure.savefig(os.path.join(star_dir, "phase_folded.png"), dpi=120)
        ax.figure.clf()
    except Exception as e:
        log.warning(f"TIC {tic_id}: plotting failed ({e})")


def update_master_summary(outdir, new_rows_df):
    """
    Merges this run's results into search_summary_all.csv, which
    accumulates every star ever analyzed across every run. Re-analyzing a
    TIC ID (e.g. after a script change like adding flare-clipping)
    replaces its old row with the newer one rather than duplicating it.
    """
    master_path = os.path.join(outdir, "search_summary_all.csv")
    if os.path.exists(master_path):
        old = pd.read_csv(master_path)
        combined = pd.concat([old, new_rows_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="TIC", keep="last")
    else:
        combined = new_rows_df
    combined = combined.sort_values("TIC").reset_index(drop=True)
    combined.to_csv(master_path, index=False)
    return combined


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Pull in every star ever analyzed, including runs from before this
    # tracking system existed - anything with a TIC_<id> folder already
    # on disk gets marked as done so it's never redundantly re-downloaded.
    backfill_analyzed_from_output_dir(CONFIG["output_dir"])

    known_ids = load_known_tic_ids()
    analyzed_ids = load_analyzed_tics(CONFIG["output_dir"]) if CONFIG.get("skip_previously_analyzed", True) else set()

    pool = find_fresh_targets(CONFIG, known_ids, analyzed_ids)
    if pool.empty:
        log.info(
            "No fresh stars found even after widening star_criteria. Try raising "
            "catalog_pool_size or max_widen_attempts in CONFIG."
        )
        return

    if CONFIG.get("use_priority_ranking", True):
        pool = rank_by_detection_priority(pool)

    new_stars = pool.head(CONFIG["max_candidate_stars"]).reset_index(drop=True)

    results = []
    attempted_ids = []
    for i, row in new_stars.iterrows():
        tic_id = int(row["ID"])
        attempted_ids.append(tic_id)
        score_note = f", priority={row['priority_score']:.2f}" if "priority_score" in row else ""
        log.info(f"[{i + 1}/{len(new_stars)}] Analyzing TIC {tic_id} (Tmag={row.get('Tmag', float('nan')):.2f}{score_note}) ...")
        res = analyze_star(tic_id, CONFIG, CONFIG["output_dir"])
        if res:
            for col in ("Tmag", "rad", "mass", "Teff", "ra", "dec", "priority_score"):
                if col in row:
                    res[col] = row[col]
            results.append(res)

    if CONFIG.get("skip_previously_analyzed", True):
        save_analyzed_tics(CONFIG["output_dir"], attempted_ids)

    if not results:
        log.info("No light curves were successfully analyzed - try different star_criteria.")
        return

    out_df = pd.DataFrame(results).sort_values("TIC").reset_index(drop=True)
    csv_path = os.path.join(CONFIG["output_dir"], "search_summary.csv")
    out_df.to_csv(csv_path, index=False)
    log.info(f"Done. This run's summary for {len(out_df)} stars written to {csv_path}")

    master_df = update_master_summary(CONFIG["output_dir"], out_df)
    master_path = os.path.join(CONFIG["output_dir"], "search_summary_all.csv")
    log.info(f"Master summary across every run now covers {len(master_df)} stars: {master_path}")
    log.info(f"Per-star plots saved under {CONFIG['output_dir']}/TIC_<id>/")

    rank_col = "tls_sde" if "tls_sde" in out_df.columns else "bls_power"
    shortlist = out_df.sort_values(rank_col, ascending=False).head(10)
    print(f"\nTop candidates worth a closer look from this run, ranked by {rank_col}:")
    show_cols = [c for c in ["TIC", "bls_period_d", "bls_depth", "odd_even_flag", rank_col] if c in shortlist.columns]
    print(shortlist[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()