"""Reproduce active coefficients from corrected training acquisitions only.

The multipliers of the cooperative first stage are fitted on link medians formed as the engine forms them from the
scenarios the release draws (release_rule._rangings). An agent pair of a scenario carries both directions, each with one
reading on the one-sample track and up to ten on the ten-sample track, and the engine pools the admissible readings of
both directions into one link and takes their median (runtime/engine/src/cooperative/mod.rs pool_observations). An
anchor link carries one direction. So every unordered pair of a scan gives pooled medians and every directed pool gives
one-direction medians, SUBSAMPLES fixed-seed draws of each, and the fit weighs the two kinds equally on every track, as
the derived first stage weighs the agent and anchor blocks of its link fit (research/solid/first_stage.py)."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "datasets"))
sys.path.insert(0, str(HERE / "training"))
import evidence
import generate_training_scenarios as generator

TRAINING_SHA = "65c077bb1aa9b8793d9b206a406ecfd574047ad345fbafdbf93cfcdfad26fa21"
TRAINING_ROWS = 1_947_886
# the shared calibration of the engine, fitted on the same corrected training table
RANGE_SLOPE_CM_PER_NS = 28.512859035713586
RANGE_OFFSET_CM = 19.57939566609116
SCALE_SLOPE = 0.02635621175286423
SCALE_INTERCEPT_CM = 4.649664346296255
PRODUCTION = dict(range_slope_cm_per_ns=RANGE_SLOPE_CM_PER_NS, range_offset_cm=RANGE_OFFSET_CM, scale_slope=SCALE_SLOPE,
                  scale_intercept_cm=SCALE_INTERCEPT_CM, scale_min_cm=8.0, scale_max_cm=55.0)
# the largest sample of one direction of a link in a scenario (release_rule.link_samples), the fixed-seed draws of every
# link and sample count, and the raw timings the engine accepts (runtime/engine/src/model.rs MAX_TOF_NS)
TEN_PACKETS = 10
SUBSAMPLES = 16
MAX_TOF_NS = 200.0
KINDS = ("agents", "anchors")
SAMPLE_RULE = ("Link medians formed as the engine forms them from the release's scenarios: every unordered pair of a scan as an "
               "agent pair, one reading of each direction on the one-sample track and up to ten on the ten-sample track drawn "
               "from the direction's pool with the admissible ones of both directions pooled into one median, and every "
               "directed pool as an anchor link with one direction alone, 16 fixed-seed draws of each, the two kinds weighted "
               "equally on every track.")


def load_training(path):
    """The corrected responder table restricted to the eleven training scans."""
    frame = pd.read_pickle(path)
    frame = frame.loc[frame["Scan_ID"].isin(range(1, 12))].copy()
    if len(frame) != TRAINING_ROWS:
        raise ValueError("unexpected training row count")
    return frame


def asymmetric_fit(values, weights=None):
    """The multipliers of the asymmetric pseudo Huber law of standardized residuals. weights, when given, weigh the
    residuals in place of their plain mean, normalized to one."""
    values = np.asarray(values)
    share = None if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)

    def objective(log_scales):
        scales = np.exp(log_scales)
        positive = values >= 0
        z = values / np.where(positive, scales[0], scales[1])
        root = np.sqrt(1 + (z / 0.7) ** 2)
        gradient = scales / scales.sum()
        if share is None:
            value = np.mean(0.49 * (root - 1)) + np.log(scales.sum())
            gradient[0] -= np.sum(z[positive] ** 2 / root[positive]) / len(z)
            gradient[1] -= np.sum(z[~positive] ** 2 / root[~positive]) / len(z)
        else:
            value = share @ (0.49 * (root - 1)) + np.log(scales.sum())
            gradient[0] -= np.sum(share[positive] * z[positive] ** 2 / root[positive])
            gradient[1] -= np.sum(share[~positive] * z[~positive] ** 2 / root[~positive])
        return value, gradient
    result = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B",
                      bounds=[(-4, 4), (-4, 4)], options={"ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise ValueError(result.message)
    return np.exp(result.x).tolist()


def availability_fit(distances, observed):
    def objective(v):
        logit = v[0] - v[1] * distances
        residual = expit(logit) - observed
        return np.mean(np.logaddexp(0, logit) - observed * logit), np.array(
            [residual.mean(), -np.mean(residual * distances)])
    result = minimize(objective, [4., .2], jac=True, method="L-BFGS-B",
                      bounds=[(-20., 20.), (0., 10.)], options={"ftol": 1e-13, "gtol": 1e-9})
    if not result.success:
        raise ValueError(result.message)
    return result.x.tolist()


def fit_scans(held):
    if held not in range(12):
        raise ValueError("expected zero or a held training acquisition from 1 to 11")
    return [scan for scan in range(1, 12) if scan != held]


def admissible(values):
    """The raw timings the engine accepts, finite and within zero and MAX_TOF_NS."""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values >= 0) & (values <= MAX_TOF_NS)]


def drawn(pool, n, rng):
    """The readings of one direction of a link that the release draws for a scenario of n samples, one reading for the
    one-sample track and up to TEN_PACKETS for the ten-sample track, from the direction's whole pool
    (release_rule.link_samples)."""
    if pool is None or len(pool) == 0:
        return np.empty(0)
    return np.asarray(pool, dtype=float)[rng.choice(len(pool), min(n, len(pool)), replace=False)]


def link_medians(directions, n, rng, draws=SUBSAMPLES):
    """draws medians of the raw timings of one link as the engine forms them: the readings the release draws from the
    pool of every direction the link carries, those the engine accepts pooled into one sample and their median taken
    (pool_observations). NaN for a draw without an admissible reading."""
    medians = np.full(draws, np.nan)
    for draw in range(draws):
        pooled = admissible(np.concatenate([drawn(pool, n, rng) for pool in directions]))
        if len(pooled):
            medians[draw] = np.median(pooled)
    return medians


def link_statistics(frame):
    """The raw link medians of every training scan, {(scan, n): {kind: [(medians, horizontal distance)]}}, and the
    availability rows. Every unordered pair of a scan is an agent pair with both directions pooled, and every directed
    pool an anchor link with one direction alone, as the release's scenarios present them (link_medians)."""
    medians, availability = {}, []
    for scan in range(1, 12):
        subset = frame.loc[frame.Scan_ID == scan]
        positions, pools = generator.scan_positions(subset), generator.ranging_index(subset)
        ids, counts = sorted(positions), {}
        for n in (1, 10):
            medians[scan, n] = {kind: [] for kind in KINDS}
        for left in ids:
            for right in ids:
                if left == right:
                    continue
                pool = pools.get((left, right), np.array([]))
                counts[left, right] = len(admissible(pool))
                distance = float(np.linalg.norm((np.asarray(positions[left]) - positions[right])[:2]))
                for n in (1, 10):
                    medians[scan, n]["anchors"].append((link_medians([pool], n, np.random.default_rng([20260914, scan, left, right, n])), distance))
                    if left < right:
                        both = [pool, pools.get((right, left), np.array([]))]
                        rng = np.random.default_rng([20260914, scan, left, right, n, 2])
                        medians[scan, n]["agents"].append((link_medians(both, n, rng), distance))
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                availability.append((scan, generator.planar_distance(positions[left], positions[right]) / 100,
                                     max(counts[left, right], counts[right, left])))
        print(f"Prepared corrected training acquisition {scan}/11", flush=True)
    return medians, np.asarray(availability)


def residuals(entries, coefficients):
    """The standardized residuals of raw link medians, entries [(medians, distance)], under a calibration's range and
    scale laws, the draws without an admissible reading left out."""
    values = []
    for medians, distance in entries:
        medians = np.asarray(medians, dtype=float)
        medians = medians[np.isfinite(medians)]
        measured = coefficients["range_slope_cm_per_ns"] * medians + coefficients["range_offset_cm"]
        scale = np.clip(coefficients["scale_slope"] * distance + coefficients["scale_intercept_cm"], coefficients["scale_min_cm"],
                        coefficients["scale_max_cm"])
        values.append((measured - distance) / scale)
    return np.concatenate(values) if values else np.empty(0)


def first_stage_fit(medians, scans, n, coefficients):
    """The multipliers of the n-sample track from the link medians of the scans under a calibration, the residuals of
    the agent pairs and those of the anchor links each carrying half the weight."""
    parts = [residuals([entry for scan in scans for entry in medians[scan, n][kind]], coefficients) for kind in KINDS]
    if any(len(part) == 0 for part in parts):
        raise ValueError(f"the {n}-sample track holds no median of some link kind")
    weights = np.concatenate([np.full(len(part), 0.5 / len(part)) for part in parts])
    return asymmetric_fit(np.concatenate(parts), weights)


def active_profile(profile):
    if "range_error" in profile:
        return dict(free={n: dict(profile["range_error"][n], **profile["availability"][n]) for n in ("1", "10")},
                    anchored={n: {"2": dict(profile["range_error"][n])} for n in ("1", "10")})
    # Frozen prediction requests retain the original per-mode representation.
    return dict(free={str(n): {key: profile["free"][str(n)][key] for key in
                ("intercept", "slope", "positive_scale", "negative_scale")} for n in (1, 10)},
                anchored={str(n): {"2": {key: profile["anchored"][str(n)]["2"][key]
                for key in ("positive_scale", "negative_scale")}} for n in (1, 10)})


def fit(output):
    output.mkdir(parents=True, exist_ok=False)
    source = evidence.retained_input(TRAINING_SHA)
    frame = load_training(source)
    frame.sort_values(["Scan_ID", "Timestamp", "Listening_Period", "Poller_Pos_ID", "Responder_Pos_ID"],
                      kind="stable", inplace=True)
    medians, pairs = link_statistics(frame)
    for held in range(12):
        scans = fit_scans(held)
        profile = dict(fit_scans=scans, range_error={}, availability={}, sample=SAMPLE_RULE)
        if held:
            profile["held_scan"] = held
        mask = np.isin(pairs[:, 0], scans)
        for n in (1, 10):
            scales = first_stage_fit(medians, scans, n, PRODUCTION)
            intercept, slope = availability_fit(pairs[mask, 1], pairs[mask, 2] >= n)
            profile["range_error"][str(n)] = dict(positive_scale=scales[0], negative_scale=scales[1])
            profile["availability"][str(n)] = dict(intercept=intercept, slope=slope)
        evidence.write(output / (f"held-{held:02d}.json" if held else "all-training.json"), profile)
    paths = [source, Path(__file__), Path(generator.__file__)]
    evidence.write(output / "inputs.json", dict(training_rows=len(frame), training_scans=list(range(1, 12)),
        evaluation_rows_used=0, files={str(p): evidence.official_scorer.file_sha256(p) for p in paths}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    fit(parser.parse_args().output)
