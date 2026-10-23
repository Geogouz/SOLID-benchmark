"""Fold nested cooperative reference profiles for the held-scan selection.

The frozen cooperative fit standardizes its training medians with the all-training calibration. For a fair
held-scan comparison the reference must be nested like the block model, so this script draws the same fixed-seed
subsample medians as fit_cooperative.py, converts and standardizes them with each fold's calibration from the
campaign's fold records, fits the multipliers and the availability law on the ten other scans, and writes one
profile per held scan. The medians are those of fit_cooperative.link_statistics, formed as the engine forms them from
the release's scenarios, every agent pair with both directions pooled and every anchor link with one direction, the two
kinds weighted equally (fit_cooperative.first_stage_fit). Held scan 0 uses the production calibration on all eleven
scans and must reproduce the committed production fit, which the script checks.

Usage: fit_bern_cooperative_folds.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/bern"))
import common  # noqa: E402
import fit_bern  # noqa: E402
import fit_cooperative as reference  # noqa: E402

OUT = common.OUT / "unify/fits"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = reference.evidence.retained_input(reference.TRAINING_SHA)
    frame = reference.load_training(source)
    frame.sort_values(["Scan_ID", "Timestamp", "Listening_Period", "Poller_Pos_ID", "Responder_Pos_ID"], kind="stable", inplace=True)
    medians, availability = reference.link_statistics(frame)
    for held in range(12):
        coefficients, calibration_path = fit_bern.calibration(held)
        scans = [scan for scan in range(1, 12) if scan != held]
        profile = dict(range_error={}, availability={}, sample=reference.SAMPLE_RULE)
        mask = np.isin(availability[:, 0], scans)
        for n in (1, 10):
            scales = reference.first_stage_fit(medians, scans, n, coefficients)
            intercept, slope = reference.availability_fit(availability[mask, 1], availability[mask, 2] >= n)
            profile["range_error"][str(n)] = dict(positive_scale=scales[0], negative_scale=scales[1])
            profile["availability"][str(n)] = dict(intercept=intercept, slope=slope)
        record = dict(identity=dict(held_scan=held or None, fit_scans=scans, calibration=coefficients,
                                    calibration_sha256=common.sha256(calibration_path), training_sha256=reference.TRAINING_SHA,
                                    training_rows=len(frame), evaluation_rows_used=0,
                                    source={p.name: common.sha256(p) for p in (Path(__file__), Path(reference.__file__))}),
                      **profile)
        common.write_json(OUT / f"cooperative-held-{held:02d}.json", record)
        print(f"held {held}: " + ", ".join(f"n={n} c+ {profile['range_error'][str(n)]['positive_scale']:.4f} c- {profile['range_error'][str(n)]['negative_scale']:.4f}"
                                          f" a {profile['availability'][str(n)]['intercept']:.3f} b {profile['availability'][str(n)]['slope']:.3f}" for n in (1, 10)), flush=True)
        if held == 0:
            committed = common.read_json(common.FITS / "bern_cooperative_fit.json")
            worst = 0.0
            for n in ("1", "10"):
                for key in ("positive_scale", "negative_scale"):
                    worst = max(worst, abs(profile["range_error"][n][key] - committed["range_error"][n][key]))
                for key in ("intercept", "slope"):
                    worst = max(worst, abs(profile["availability"][n][key] - committed["availability"][n][key]))
            print(f"parity with the committed production fit: largest absolute difference {worst:.2e}", flush=True)


if __name__ == "__main__":
    main()
