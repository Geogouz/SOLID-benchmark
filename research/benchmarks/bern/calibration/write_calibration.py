#!/usr/bin/env python3.12
"""Write the twelve Bern calibration records that research/solid/fit_bern.py reads.

Each law is the fit_corrected_training binary's fit on the rows of extract_rows.py, all eleven training scans or the
ten of a held scan's fold. A held scan's record also carries the fold's area, the bounding box of the positions of its
training scans widened by five percent of width and depth on each side (fit_profiles.area_bounds).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "ade_bp"))
import extract_rows  # noqa: E402
import fit_profiles  # noqa: E402

DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
TABLE = DATA / "inputs/65c077bb1aa9b879/Rangings at Responders.pkl"
BINARY = ROOT / "research/target/release/fit_corrected_training"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path, help="full-training.rows.bin of extract_rows.py")
    parser.add_argument("--binary", default=BINARY, type=Path)
    args = parser.parse_args()
    if extract_rows.file_digest(TABLE, "sha256") != extract_rows.INPUT_SHA256:
        raise ValueError("the responder table identity is not frozen")
    frame = pd.read_pickle(TABLE)
    frame = frame.loc[frame["Class"].eq("Train") & frame["Scan_ID"].isin(extract_rows.TRAINING_SCANS)]
    for held in (0, *extract_rows.TRAINING_SCANS):
        scans = [scan for scan in extract_rows.TRAINING_SCANS if scan != held]
        command = [args.binary, args.rows, *(["--held-scan", str(held)] if held else [])]
        law = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        rows = sum(extract_rows.SCAN_COUNTS[scan] for scan in scans)
        if held:
            area = fit_profiles.area_bounds(frame.loc[frame["Scan_ID"].isin(scans)])
            fit_profiles.write_json(DATA / f"anc-tail-generalization/scan-{held:02d}/fit.json", dict(
                held_scan=held, fit_scans=scans, training_rows=rows, evaluation_rows_used_for_fitting=0,
                held_rows_used_for_fitting=0, coefficients=law, profile=dict(area=area)))
        else:
            # The all-scan record holds the floor of the production input filter, which fit_bern.calibration sets for every fold.
            fit_profiles.write_json(DATA / "corrected-training/full/fit.json", dict(
                held_scan=None, fit_scans=scans, rows=rows, evaluation_rows_used=0,
                calibration=dict(law, runtime_tof_floor_ns=0.0)))
        print(f"held scan {held or None}: {rows} rows, {law}", flush=True)


if __name__ == "__main__":
    main()
