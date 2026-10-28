"""One real TWR-CLOUD link for the paper's opening scene, ten readings that agree and still run long.

The script reads the anchor only training scenarios of the eleven training scans, the regenerable corpora that
research/benchmarks/bern/training/generate_heldscan_corpora.py anchor-only writes (common.BERN_ANCHOR_ONLY_CORPUS,
out/solid/unify/bern-anchor-only-scenarios-v2 unless SOLID_ANCHOR_ONLY_CAMPAIGN names another folder), converts every
reading with the calibration law of all eleven training scans exactly as research/solid/fit_bern.py does, and keeps
every block of ten admissible readings between an agent and an anchor, with its true distance in three dimensions. It
counts the blocks whose ten readings lie within 2 cm of each other and, among those, the blocks whose median reading
exceeds the true distance by 30 cm or more. The scene is the block of median excess among the latter, ties broken by
scan, scenario, agent and anchor, so the example is typical of such blocks and not the most extreme one. It solves and
fits nothing.

Usage: scene_link.py [--out research/artifacts/solid/figures/data/scene-link.json]
"""
import argparse
import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/solid"))
# the scene needs only the calibration and the conversion of fit_bern, not its fitter, so an empty stand-in for
# fit_links keeps scipy out of the plotting environment
sys.modules.setdefault("fit_links", types.ModuleType("fit_links"))
import common  # noqa: E402
import fit_bern  # noqa: E402

AGREE_CM = 2.0
LONG_CM = 30.0
# the anchor only training corpora of the held scans, scan-NN/scenarios.zip with the raw timings
CORPUS = common.BERN_ANCHOR_ONLY_CORPUS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def blocks(coefficients):
    for scan in fit_bern.SCAN_IDS:
        with zipfile.ZipFile(CORPUS / f"scan-{scan:02d}/scenarios.zip") as archive:
            for name in sorted(archive.namelist()):
                scenario = json.loads(archive.read(name))
                fit_bern.check_scenario(scenario, scan)
                anchors = {int(k): np.array([v["x"], v["y"], v["z"]]) for k, v in scenario["True Positions of Anchors"].items()}
                agents = {int(k): np.array([v["x"], v["y"], v["z"]]) for k, v in scenario["True Positions of Agents"].items()}
                for key, value in scenario["Iterations"][0]["Ranging samples to Anchors"].items():
                    packets = fit_bern.calibrated_packets((value or {}).get("10"), coefficients)
                    if len(packets) != 10:
                        continue
                    agent, anchor = map(int, key.split("-"))
                    readings = np.array([p["distance_cm"] for p in packets])
                    truth = float(np.linalg.norm(agents[agent] - anchors[anchor]))
                    yield dict(scan=scan, scenario_id=scenario["Research Metadata"]["scenario_id"], agent=agent, anchor=anchor,
                               true_distance_cm=truth, readings_cm=readings.tolist(),
                               scales_cm=[p["scale_cm"] for p in packets],
                               spread_cm=float(readings.max() - readings.min()),
                               median_error_cm=float(np.median(readings) - truth),
                               smallest_error_cm=float(readings.min() - truth))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "research/artifacts/solid/figures/data/scene-link.json")
    args = parser.parse_args()
    coefficients, calibration_path = fit_bern.calibration(None)
    total, agreeing, long_ones = 0, 0, []
    for block in blocks(coefficients):
        total += 1
        if block["spread_cm"] <= AGREE_CM:
            agreeing += 1
            if block["smallest_error_cm"] >= LONG_CM:
                long_ones.append(block)
    long_ones.sort(key=lambda b: (b["median_error_cm"], b["scan"], b["scenario_id"], b["agent"], b["anchor"]))
    scene = long_ones[len(long_ones) // 2]
    inputs = {relative(CORPUS / f"scan-{s:02d}/scenarios.zip"): sha256(CORPUS / f"scan-{s:02d}/scenarios.zip")
              for s in fit_bern.SCAN_IDS}
    record = dict(schema="solid.figures.scene-link.v1",
                  rule=f"blocks of ten admissible agent-to-anchor readings in the training scans, readings within {AGREE_CM} cm of "
                       f"each other, every reading at least {LONG_CM} cm above the true distance in three dimensions, the block "
                       "of median excess among them",
                  counts=dict(blocks_of_ten=total, agreeing_within_2_cm=agreeing, agreeing_and_all_at_least_30_cm_long=len(long_ones)),
                  scene={k: (round(v, 2) if isinstance(v, float) else [round(x, 2) for x in v] if isinstance(v, list) else v)
                         for k, v in scene.items()},
                  calibration=dict(path=relative(calibration_path), sha256=sha256(calibration_path), coefficients=coefficients),
                  inputs=inputs, code={str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__)),
                                       "research/solid/fit_bern.py": sha256(ROOT / "research/solid/fit_bern.py")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    s = record["scene"]
    print(f"{total} blocks of ten readings, {agreeing} within {AGREE_CM} cm, {len(long_ones)} of those all at least {LONG_CM} cm long")
    print(f"scene: scan {s['scan']}, scenario {s['scenario_id']}, agent {s['agent']}, anchor {s['anchor']}, true distance "
          f"{s['true_distance_cm']} cm, readings {min(s['readings_cm'])} to {max(s['readings_cm'])} cm, median error {s['median_error_cm']} cm")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
