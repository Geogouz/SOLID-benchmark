"""Fit the Bern link model with calibration isolated inside each training fold.

The default fits all eleven held training scans and the final all-training profile.
Every fit is resumable and records its raw inputs and calibration identity.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import fit_links
import method_constants

SCANS = common.DATA / "anc-tail-generalization"
SCAN_IDS = tuple(range(1, 12))
FULL_CALIBRATION = common.DATA / "corrected-training/full/fit.json"


def calibration(held):
    """The calibration of a fold and the record it comes from, the production calibration of all eleven scans for held
    0. A reader's data without that record takes the engine's law of raw timings with the production input filter,
    which the record holds (method_constants.CALIBRATION and TIMINGS_NS), and names no record."""
    if not held and common.READER and not FULL_CALIBRATION.exists():
        return dict(method_constants.CALIBRATION, **method_constants.TIMINGS_NS), None
    path = SCANS / f"scan-{held:02d}/fit.json" if held else FULL_CALIBRATION
    record = common.read_json(path)
    training = [scan for scan in SCAN_IDS if scan != held]
    if record["fit_scans"] != training or record.get("held_scan") != (held or None):
        raise ValueError(f"Wrong calibration fold in {path}")
    if record.get("evaluation_rows_used_for_fitting", record.get("evaluation_rows_used")) != 0:
        raise ValueError("Evaluation rows entered calibration")
    if held and record["held_rows_used_for_fitting"] != 0:
        raise ValueError("Held training rows entered calibration")
    coefficients = dict(record["coefficients"] if held else record["calibration"])
    # Match the production input filter, including valid timings below one ns.
    coefficients["runtime_tof_floor_ns"] = 0.0
    return coefficients, path


def calibrated_packets(value, coefficients):
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    packets = []
    for tof in raw:
        if not np.isfinite(tof) or not coefficients["runtime_tof_floor_ns"] <= tof <= coefficients["runtime_tof_max_ns"]:
            continue
        distance = max(0., coefficients["range_slope_cm_per_ns"] * tof + coefficients["range_offset_cm"])
        scale = min(coefficients["scale_max_cm"], max(coefficients["scale_min_cm"],
                    coefficients["scale_slope"] * distance + coefficients["scale_intercept_cm"]))
        packets.append(dict(distance_cm=distance, scale_cm=scale))
    return packets


def check_scenario(scenario, scan):
    metadata = scenario["Research Metadata"]
    if metadata["data_role"] != "training" or metadata["scan_id"] != scan:
        raise ValueError("Unexpected training scan")


def blocks_from_scan(scan, rng, scenarios, coefficients, samples=10):
    blocks = []
    with zipfile.ZipFile(SCANS / f"scan-{scan:02d}/scenarios.zip") as archive:
        names = sorted(archive.namelist())
        for name in rng.choice(names, min(scenarios, len(names)), replace=False):
            scenario = json.loads(archive.read(name))
            check_scenario(scenario, scan)
            anchors = {int(k): np.array([v["x"], v["y"], v["z"]]) for k, v in scenario["True Positions of Anchors"].items()}
            agents = {int(k): np.array([v["x"], v["y"], v["z"]]) for k, v in scenario["True Positions of Agents"].items()}
            for key, value in scenario["Iterations"][0]["Ranging samples to Anchors"].items():
                packets = calibrated_packets((value or {}).get(str(samples)), coefficients)
                if packets:
                    agent, anchor = map(int, key.split("-"))
                    blocks.append((np.array([p["distance_cm"] for p in packets]), np.array([p["scale_cm"] for p in packets]),
                                   float(np.linalg.norm(agents[agent] - anchors[anchor]))))
    return blocks


def fit_fold(held, per_scan, count):
    coefficients, calibration_path = calibration(held)
    training = [scan for scan in SCAN_IDS if scan != held]
    inputs = [SCANS / f"scan-{scan:02d}/scenarios.zip" for scan in training]
    identity = dict(held_scan=held or None, fit_scans=training, evaluation_rows_used=0, seed=11, per_scan=per_scan,
                    requested_blocks=count, calibration=coefficients, calibration_sha256=common.sha256(calibration_path) if calibration_path else None,
                    inputs={str(p.relative_to(common.ROOT) if p.is_relative_to(common.ROOT) else p): common.sha256(p) for p in inputs},
                    source={p.name: common.sha256(p) for p in (Path(__file__), Path(fit_links.__file__))})
    path = common.OUT / f"selection/anchor-only/fits/held-{held:02d}.json"
    manifest = path.with_suffix(".manifest.json")
    if common.cache_matches(manifest, identity, [path]):
        return held, common.read_json(path)
    rng = np.random.default_rng(11)
    pool = [block for scan in training for block in blocks_from_scan(scan, rng, per_scan, coefficients)]
    chosen = [pool[i] for i in rng.choice(len(pool), min(count, len(pool)), replace=False)]
    residuals = np.concatenate([block[0] - block[2] for block in chosen])
    fit = fit_links.fit4(chosen)
    if not fit["converged"]:
        raise ValueError(f"Link model fit did not converge for held scan {held}")
    result = dict(identity=identity, four=fit, blocks=len(chosen),
                  residual_quantiles=np.percentile(residuals, [5, 25, 50, 75, 95]).round(1).tolist())
    common.write_json(path, result)
    common.seal_cache(manifest, identity, [path])
    return held, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    results = {}
    progress = common.OUT / "selection/anchor-only/progress.json"
    common.write_json(progress, dict(phase="fitting", completed_scans=[], total_fits=12, complete=False))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_fold, held, 60 if held == 0 else 40, 4000 if held == 0 else 2500)
                   for held in (0, *SCAN_IDS)]
        for future in as_completed(futures):
            held, result = future.result()
            results[str(held)] = result
            print(f"held scan {held}: {result['blocks']} blocks, fit {result['four']}", flush=True)
            common.write_json(progress, dict(phase="fitting", completed_scans=sorted(map(int, results)), total_fits=12, complete=False))
    common.write_json(common.FITS / "bern_fold_fits.json", {str(scan): results[str(scan)] for scan in SCAN_IDS})
    common.write_json(common.FITS / "bern_production_fit.json", {"0": results["0"]})


if __name__ == "__main__":
    main()
