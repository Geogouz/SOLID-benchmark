"""Training only selections on the TWR-CLOUD-Bern training scans.

anchor-only   All eleven held training scans, all 3,300 retained scenarios per scan, both iterations and sample
              counts. Calibration and link fitting exclude the held scan. The released scorer pools full unit
              distributions with equal weight per scan, anchor count and agent count cell.

Writes a record under research/artifacts/solid/selections. The shared range bias of the anchor aided mode, once
selected here, left the method with the unified block likelihood; its record stays as history. The anchor only
selection is a historical round too. One of its candidates sets the kept cells, which the engine's adaptive anchor
posterior no longer has, so the script refuses to run it again (RETIRED) and its record stays.

Usage: select_bern.py anchor-only [--workers 12]
"""
import argparse
import csv
import fcntl
import gzip
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_bern  # noqa: E402
import profiles  # noqa: E402

SCANS = common.DATA / "anc-tail-generalization"
TRAINING = common.DATA / "inputs/22e0c6fe88d9aba5/twr-cloud-bern-training-v1.zip"
SCORER = common.DATA / "inputs/0068d72b94db6b1c/score_twr_cloud_bern.py"
CLI = common.ROOT / "research/target/release/solid"
ARCHIVE_RUNNER = common.ROOT / "research/target/release/solid-archive"
PYTHON = sys.executable
WORK = common.OUT / "selection"

ANCHOR_ONLY_CANDIDATES = {
    "reference": {},
    "excess_0.05": dict(excess_probability=0.05),
    "excess_0.20": dict(excess_probability=0.20),
    "excess_fitted": "fitted",
    "support_384_96": dict(top_cells=384, joint_samples=96),
}
# the anchor only selection is a historical round that the current engine cannot run again, and its record stays
RETIRED = ("The anchor only selection of select_bern.py is a historical round. Its candidate support_384_96 sets top_cells, the cap on the "
           "kept cells that the engine's adaptive anchor posterior replaced, and the engine refuses every profile that holds it. The record "
           "research/artifacts/solid/selections/bern-anchor-only.json stays as the history of the round.")


def prepare_scan(scan, work):
    coefficients, calibration_path = fit_bern.calibration(scan)
    source = SCANS / f"scan-{scan:02d}/scenarios.zip"
    path = work / f"scan-{scan:02d}/calibrated.zip"
    manifest = path.with_suffix(".manifest.json")
    identity = dict(raw_sha256=common.sha256(source), calibration_sha256=common.sha256(calibration_path),
                    coefficients=coefficients, source_sha256=common.sha256(Path(fit_bern.__file__)))
    expected = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            scenario = json.loads(archive.read(name))
            fit_bern.check_scenario(scenario, scan)
            scenario_id = int(Path(name).stem)
            for iteration in scenario["Iterations"]:
                expected.update((scenario_id, int(iteration["ID"]), samples) for samples in (1, 10))
        if len(expected) != 3300 * 4:
            raise ValueError(f"Incomplete held scan {scan}")
        if not common.cache_matches(manifest, identity, [path]):
            temporary = path.with_suffix(".partial.zip")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
                for name in sorted(archive.namelist()):
                    scenario = json.loads(archive.read(name))
                    for iteration in scenario["Iterations"]:
                        for field in ("Ranging samples between Agents", "Ranging samples to Anchors"):
                            for values in iteration[field].values():
                                for samples, value in values.items():
                                    packets = fit_bern.calibrated_packets(value, coefficients)
                                    values[samples] = dict(calibrated_samples=packets) if packets else None
                    target.writestr(name, json.dumps(scenario, separators=(",", ":"), allow_nan=False))
            temporary.replace(path)
            common.seal_cache(manifest, identity, [path])
    return path, expected


def read_official_scores(path, expected):
    rows = {}
    with gzip.open(path, "rt") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["scenario_id"]), int(row["iteration"]), int(row["samples"]))
            error = float(row["mean_pair_error"])
            if key in rows or key not in expected or row["mode"] != "ANC" or int(row["q"]) != 2 or not math.isfinite(error):
                raise ValueError(f"Invalid or duplicate score in {path}")
            rows[key] = (error, int(row["anchors"]), int(row["agents"]))
    if rows.keys() != expected:
        raise ValueError(f"Missing held training units in {path}")
    return rows


def run_anchor_fold(directory, scenarios, expected, profile, workers):
    directory.mkdir(parents=True, exist_ok=True)
    common.write_json(directory / "profile.json", profile)
    identity = dict(binary_sha256=common.sha256(CLI), scenarios_sha256=common.sha256(scenarios), profile=profile,
                    workers=workers, samples=[1, 10], iterations=[1, 2], scorer_sha256=common.sha256(SCORER),
                    source_sha256=common.sha256(Path(__file__)))
    outputs = [directory / name for name in ("benchmark.json", "predictions.zip", "scenario_scores.csv.gz")]
    manifest = directory / "manifest.json"
    if not common.cache_matches(manifest, identity, outputs):
        with tempfile.TemporaryDirectory(prefix="run-", dir=directory) as temporary:
            stage = Path(temporary)
            command = [str(CLI), "benchmark", "--scenarios", str(scenarios), "--method", "solid",
                       "--deployment-profile", str(directory / "profile.json"), "--output", str(stage / "benchmark.json"),
                       "--predictions-output", str(stage / "predictions.zip"), "--mode", "anc", "--sample-counts", "1,10",
                       "--iterations", "1,2", "--include-cases", "--time-budget-ms", "0", "--workers", str(workers)]
            with (directory / "run.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            report = common.read_json(stage / "benchmark.json")
            case_keys = [(c["archive_scenario_id"], c["iteration"], c["samples"]) for c in report["cases"]]
            if report["solves"] != len(expected) or len(case_keys) != len(expected) or set(case_keys) != expected:
                raise ValueError("Incomplete held training solve")
            with zipfile.ZipFile(stage / "predictions.zip") as archive, (stage / "predictions.csv").open("wb") as target:
                with archive.open("predictions.csv") as source:
                    shutil.copyfileobj(source, target)
            scorer = common.load_scorer()
            predictions = scorer.load_predictions(stage / "predictions.csv")
            scores = scorer.score_prediction_scenarios(scenarios, predictions, ["ANC"], [1, 10], [2], False)
            native = {key: row for key, row in zip(case_keys, report["cases"])}
            for score in scores:
                case = native[score.scenario_id, score.iteration, score.samples]
                if not math.isclose(case["mean_pair_error_cm"], score.mean_pair_error, rel_tol=1e-9, abs_tol=1e-8):
                    raise ValueError("Native and released scenario scores disagree")
            scorer.write_scenario_scores(stage / "scenario_scores.csv.gz", scores)
            read_official_scores(stage / "scenario_scores.csv.gz", expected)
            if identity["binary_sha256"] != common.sha256(CLI):
                raise ValueError("Runner changed during selection")
            for path in outputs:
                (stage / path.name).replace(path)
        common.seal_cache(manifest, identity, outputs)
    return read_official_scores(outputs[-1], expected)


def summarize_tracks(scores, scorer):
    tracks = {}
    for samples in (1, 10):
        cells, levels = defaultdict(list), defaultdict(list)
        for (scan, scenario, iteration, q), (error, anchors, agents) in scores.items():
            if q == samples:
                cells[(scan, anchors, agents)].append(error)
                levels[anchors].append(error)
        tracks[str(samples)] = dict(units=sum(map(len, cells.values())), cells=len(cells),
                                    balanced=[scorer.balanced_percentile(cells, p) for p in (.5, .75, .9)],
                                    per_anchor_mean={str(k): float(np.mean(v)) for k, v in sorted(levels.items())})
    return tracks


def anchor_only(args):
    fold_fits = common.read_json(common.FITS / "bern_fold_fits.json")
    if set(fold_fits) != {str(scan) for scan in fit_bern.SCAN_IDS}:
        raise ValueError("All eleven training-only link fits are required")
    production = common.read_json(common.FITS / "bern_production_fit.json")
    raw_inputs = {str((SCANS / f"scan-{s:02d}/scenarios.zip").relative_to(common.ROOT)):
                  common.sha256(SCANS / f"scan-{s:02d}/scenarios.zip") for s in fit_bern.SCAN_IDS}
    fit_sources = {p.name: common.sha256(p) for p in (Path(fit_bern.__file__), Path(fit_bern.fit_links.__file__))}
    for held, result in {**production, **fold_fits}.items():
        identity = result["identity"]
        expected_scans = [s for s in fit_bern.SCAN_IDS if s != int(held)]
        expected_inputs = {str((SCANS / f"scan-{s:02d}/scenarios.zip").relative_to(common.ROOT)): raw_inputs[
                           str((SCANS / f"scan-{s:02d}/scenarios.zip").relative_to(common.ROOT))] for s in expected_scans}
        coefficients, calibration_path = fit_bern.calibration(int(held))
        if (identity["fit_scans"] != expected_scans or identity["inputs"] != expected_inputs
                or identity["source"] != fit_sources or identity["evaluation_rows_used"] != 0
                or identity["held_scan"] != (int(held) or None) or not result["four"]["converged"]
                or identity["calibration"] != coefficients or identity["calibration_sha256"] != common.sha256(calibration_path)):
            raise ValueError(f"Fit inputs or implementation changed for held scan {held}")
    work = WORK / "anchor-only"
    scorer = common.load_scorer()
    freeze = dict(schema="solid.selection.bern-anchor-only.v2", scans=list(fit_bern.SCAN_IDS), scenarios_per_scan=3300,
                  samples=[1, 10], iterations=[1, 2], candidates=ANCHOR_ONLY_CANDIDATES, workers=args.workers,
                  fold_fits_sha256=common.sha256(common.FITS / "bern_fold_fits.json"),
                  production_fit_sha256=common.sha256(common.FITS / "bern_production_fit.json"),
                  binary_sha256=common.sha256(CLI), scorer_sha256=common.sha256(SCORER),
                  rule="Lowest mean balanced S75 across both sample tracks. Each scan, anchor count and agent count cell has equal weight. Ties keep candidate order. No evaluation scores enter selection.",
                  source={p.name: common.sha256(p) for p in (Path(__file__), Path(profiles.__file__), Path(fit_bern.__file__))})
    freeze_path = work / "freeze.json"
    if freeze_path.exists() and common.read_json(freeze_path) != freeze:
        raise ValueError("Frozen selection changed. Review the differences before starting another campaign.")
    common.write_json(freeze_path, freeze)
    scores, completed = defaultdict(dict), []
    started = time.time()
    for scan in fit_bern.SCAN_IDS:
        fold = fold_fits[str(scan)]
        _, calibration_path = fit_bern.calibration(scan)
        scenarios, expected = prepare_scan(scan, work)
        scan_profile = common.read_json(calibration_path)["profile"]
        for name, change in ANCHOR_ONLY_CANDIDATES.items():
            profile = dict(area=scan_profile["area"], anchor_only=profiles.bern_anchor_only(fold["four"], change, held=scan))
            common.write_json(work / "progress.json", dict(phase="selection", active=dict(scan=scan, candidate=name),
                                                           completed=completed, total_jobs=99, complete=False))
            result = run_anchor_fold(work / f"scan-{scan:02d}" / name, scenarios, expected, profile, args.workers)
            scores[name].update({(scan, *key): value for key, value in result.items()})
            completed.append(dict(scan=scan, candidate=name, units=len(result)))
            common.write_json(work / "progress.json", dict(phase="selection", completed=completed, total_jobs=99, elapsed_seconds=time.time() - started,
                                                           complete=len(completed) == 99))
            print(f"{len(completed)}/99 scan {scan} {name} complete", flush=True)
    candidates = {}
    for name, change in ANCHOR_ONLY_CANDIDATES.items():
        tracks = summarize_tracks(scores[name], scorer)
        candidates[name] = dict(change=change, tracks=tracks, mean_s75=float(np.mean([tracks[q]["balanced"][1] for q in ("1", "10")])))
    selected = min(candidates, key=lambda name: candidates[name]["mean_s75"])
    record = dict(freeze, freeze_sha256=common.sha256(freeze_path), candidates=candidates, selected=selected, complete=True)
    common.write_json(common.SELECTIONS / "bern-anchor-only.json", record)
    print(f"Selected from all eleven training scans: {selected}", flush=True)


def scenario_only_training(work):
    """The released scorer reads every JSON member, so the training archive's manifest is left out."""
    path = work / "bern-training-scenarios.zip"
    if not path.exists():
        with zipfile.ZipFile(TRAINING) as source, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
            for member in source.infolist():
                if member.filename.endswith(".json") and "manifest" not in member.filename and not member.filename.startswith("__MACOSX"):
                    target.writestr(member, source.read(member))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selection", choices=["anchor-only"])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.selection == "anchor-only":
        raise SystemExit(RETIRED)
    work = WORK / args.selection
    work.mkdir(parents=True, exist_ok=True)
    with (work / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            anchor_only(args)
        except Exception as error:
            progress = common.read_json(work / "progress.json") if (work / "progress.json").exists() else {}
            common.write_json(work / "progress.json", dict(progress, phase="failed", error=str(error), complete=False))
            raise


if __name__ == "__main__":
    main()
