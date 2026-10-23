#!/usr/bin/env python3.12
"""Write the calibrated Bern evaluation archive that Jin et al. and STRONG solve.

Every timing pool of every scenario of the published evaluation archive becomes the calibrated samples of
fit_bern.calibrated_packets under the all-scan law fit_bern.calibration(0). Nothing else of a scenario changes.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "research/solid"))
import fit_bern  # noqa: E402

DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
SOURCE = DATA / "inputs/8617b54db218e099/TWR-CLOUD Bern Eval Scenarios.zip"
OUTPUT = DATA / "corrected-training/evaluation/calibrated-scenarios.zip"
LINKS = ("Ranging samples between Agents", "Ranging samples to Anchors")


def calibrated(member, coefficients):
    """The bytes of one calibrated scenario from the bytes of its published scenario."""
    scenario = json.loads(member)
    for iteration in scenario["Iterations"]:
        for field in LINKS:
            for link, pools in iteration[field].items():
                if pools is not None:
                    iteration[field][link] = {samples: None if pool is None else dict(
                        calibrated_samples=fit_bern.calibrated_packets(pool, coefficients)) for samples, pool in pools.items()}
    return json.dumps(scenario, separators=(",", ":")).encode()


def members(archive):
    return [name for name in archive.namelist() if name.endswith(".json") and not name.startswith("__MACOSX/")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, help="write only the first members, for a check")
    args = parser.parse_args()
    # fit_bern reads common.DATA, which does not follow SOLID_DATA
    fit_bern.FULL_CALIBRATION = DATA / "corrected-training/full/fit.json"
    coefficients, _ = fit_bern.calibration(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as source, zipfile.ZipFile(args.output, "x") as target:
        for name in members(source)[:args.limit]:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            target.writestr(info, calibrated(source.read(name), coefficients))


if __name__ == "__main__":
    main()
