"""Locate retained inputs without changing frozen execution records."""

import json
import os
from pathlib import Path

import official_scorer


ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
EVIDENCE = ROOT / "research/artifacts/bern/retained-evidence.json"


def resolve(original):
    inventory = json.loads((DATA / "inventory.json").read_text())
    record = inventory["files"].get(str(original))
    if record is None:
        relative = str(Path(original).resolve().relative_to(DATA.resolve()))
        matches = [r for r in inventory["files"].values() if r["path"] == relative]
        if not matches or len({r["sha256"] for r in matches}) != 1:
            raise ValueError(f"unregistered or conflicting retained input {original}")
        record = matches[0]
    path = DATA / record["path"]
    official_scorer.require_hash(path, record["sha256"], "retained input")
    return path


def run_directory(method, mode, result):
    relative = Path(result.get("data_directory", f"runs/{method}/{mode.lower()}"))
    path = (DATA / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(DATA.resolve()):
        raise ValueError("execution directory must remain inside retained data")
    return path


def run_record(method, mode):
    evidence = json.loads(EVIDENCE.read_text())
    if "selection_sha256" in evidence:
        import selection

        selection.validate_evidence(evidence)
    row = next(row for row in evidence["methods"] if row["id"] == method)
    result = row["results"][mode.upper()]
    directory = run_directory(method, mode, result)
    if "refresh_freeze_sha256" in result:
        official_scorer.require_hash(directory.parents[2] / "freeze.json",
                                     result["refresh_freeze_sha256"], "refresh freeze")
    manifest = directory / "official/manifest.json"
    official_scorer.require_hash(manifest, result["manifest_sha256"], "execution manifest")
    official_scorer.require_hash(directory / "predictions.zip", result["prediction_sha256"], "predictions")
    official_scorer.require_hash(directory / "benchmark.json", result["benchmark_report_sha256"], "benchmark")
    record = json.loads(manifest.read_text())
    for name, expected in {**record["official_outputs"], **record["diagnostic_outputs"]}.items():
        official_scorer.require_hash(directory / "official" / name, expected, "score output")
    return record


if __name__ == "__main__":
    evidence = json.loads(EVIDENCE.read_text())
    count = 0
    for row in evidence["methods"]:
        for mode in row["results"]:
            run_record(row["id"], mode)
            count += 1
    print(f"Verified predictions, reports and score files for {count} method-mode runs")
