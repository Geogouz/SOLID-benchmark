#!/usr/bin/env python3
"""Run and verify the released TWR-CLOUD-Bern scorer."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROTOCOL = SCRIPT_DIRECTORY / "PROTOCOL.md"
VALID_OUTPUT_DIAGNOSTICS = SCRIPT_DIRECTORY / "valid_output_diagnostics.py"
SCENARIO_SHA256 = "8617b54db218e0999ebda2f6c36107e5c79f3f5e7e84eb1047ecf997551c110b"
SCORER_SHA256 = "0068d72b94db6b1c40414829390025a3b2b12e9b9d1fd7d939fff5fc97659073"
PREDICTION_COLUMNS = [
    "scenario_id",
    "iteration",
    "samples",
    "mode",
    "node_id",
    "q",
    "x",
    "y",
    "z",
]
SCENARIO_SCORE_COLUMNS = [
    "scenario_id",
    "iteration",
    "samples",
    "mode",
    "q",
    "anchors",
    "agents",
    "possible_pairs",
    "eligible_pairs",
    "scored_pair_coverage",
    "penalized_pairs",
    "penalized_pair_rate",
    "mean_pair_error",
]
MODE_COUNTS = {
    "AF": {"scenarios": 24_710, "track_units": 49_420, "units": 98_840},
    "AAC": {"scenarios": 123_550, "track_units": 247_100, "units": 494_200},
    "ANC": {"scenarios": 123_550, "track_units": 247_100, "units": 494_200},
}
OUTPUT_FILES = (
    "scenario_scores.csv.gz",
    "cell_summary.csv",
    "track_summary.csv",
    "balanced_cdf_points.csv",
    "score_manifest.json",
)
DIAGNOSTIC_FILES = (
    "diagnostics/valid_output_scenario_scores.csv.gz",
    "diagnostics/valid_output_summary.json",
)
REAL_SCENARIO_PREFIX = "TWR-CLOUD Bern Eval Scenarios/"
REAL_SCENARIO_COUNT = 148_260
LOCAL_FILE_HEADER = struct.Struct("<4s5H3L2H")


def load_valid_output_diagnostics():
    path = VALID_OUTPUT_DIAGNOSTICS
    specification = importlib.util.spec_from_file_location(
        "valid_output_diagnostics", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load valid-output diagnostics {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


valid_output_diagnostics = load_valid_output_diagnostics()
# The coordinate rule of every scorer (research/benchmarks/datasets/scoring.py), which the predictions pass before the
# released scorer and the valid-output diagnostic read them.
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent / "datasets"))
import scoring  # noqa: E402


@lru_cache(maxsize=None)
def _file_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _file_sha256(str(resolved), stat.st_size, stat.st_mtime_ns)


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 is {actual}, expected {expected}")


def _real_scenario_id(name: str) -> int | None:
    if not name.startswith(REAL_SCENARIO_PREFIX) or not name.endswith(".json"):
        return None
    stem = name[len(REAL_SCENARIO_PREFIX) : -len(".json")]
    return int(stem) if stem.isdigit() else None


@lru_cache(maxsize=None)
def _scenario_archive_identity(path: str, size: int, modified_ns: int) -> dict:
    del size, modified_ns
    path = Path(path)
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, path.open("rb") as raw:
        real = []
        unexpected_json = []
        for info in archive.infolist():
            scenario_id = _real_scenario_id(info.filename)
            if scenario_id is not None:
                real.append((scenario_id, info))
            elif info.filename.endswith(".json"):
                unexpected_json.append(info.filename)
        identifiers = [scenario_id for scenario_id, _ in real]
        if len(real) != REAL_SCENARIO_COUNT or len(set(identifiers)) != REAL_SCENARIO_COUNT:
            raise ValueError("scenario archive does not contain 148,260 unique real scenarios")

        for scenario_id, info in real:
            raw.seek(info.header_offset)
            fields = LOCAL_FILE_HEADER.unpack(raw.read(LOCAL_FILE_HEADER.size))
            if fields[0] != b"PK\x03\x04":
                raise ValueError(f"invalid ZIP member header for scenario {scenario_id}")
            name_length, extra_length = fields[-2:]
            raw.seek(name_length + extra_length, 1)
            digest.update(f"{scenario_id}:{info.filename}:{info.compress_size}:".encode())
            remaining = info.compress_size
            while remaining:
                block = raw.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"truncated ZIP member for scenario {scenario_id}")
                digest.update(block)
                remaining -= len(block)
    return {
        "real_scenarios": len(real),
        "unexpected_json_members": len(unexpected_json),
        "unexpected_json_examples": unexpected_json[:3],
        "compressed_scenario_fingerprint": digest.hexdigest(),
    }


def scenario_archive_identity(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return _scenario_archive_identity(
        str(resolved), stat.st_size, stat.st_mtime_ns
    ).copy()


def verify_scorer_scenarios(source: Path, scorer_input: Path) -> dict:
    source_identity = scenario_archive_identity(source)
    scorer_identity = scenario_archive_identity(scorer_input)
    if scorer_identity["unexpected_json_members"]:
        raise ValueError(
            "scorer input still contains non-scenario JSON members such as "
            f"{scorer_identity['unexpected_json_examples']}"
        )
    if (
        source_identity["compressed_scenario_fingerprint"]
        != scorer_identity["compressed_scenario_fingerprint"]
    ):
        raise ValueError("scorer input changed at least one real scenario member")
    return {
        "source": source_identity,
        "scorer_input": scorer_identity,
        "scorer_input_sha256": file_sha256(scorer_input),
    }


def parse_coordinate(value: str) -> float:
    if not value.strip():
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def scenario_fingerprint(scenario_ids: set[int]) -> str:
    digest = hashlib.sha256()
    for scenario_id in sorted(scenario_ids):
        digest.update(f"{scenario_id}\n".encode())
    return digest.hexdigest()


def inspect_predictions(
    path: Path, mode: str, expected_counts: dict | None = None
) -> dict:
    expected = MODE_COUNTS[mode] if expected_counts is None else expected_counts
    units = defaultdict(int)
    unit_scenarios = {
        (iteration, samples): set()
        for iteration in (1, 2)
        for samples in (1, 10)
    }
    records = 0
    missing = 0
    nonfinite = 0
    previous_key = None
    previous_unit = None
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["predictions.csv"]:
            raise ValueError("prediction archive must contain only predictions.csv")
        with archive.open("predictions.csv") as raw:
            reader = csv.DictReader(line.decode("utf-8") for line in raw)
            if reader.fieldnames != PREDICTION_COLUMNS:
                raise ValueError("prediction CSV does not use the released scorer schema")
            for row in reader:
                iteration = int(row["iteration"])
                samples = int(row["samples"])
                if iteration not in (1, 2) or samples not in (1, 10):
                    raise ValueError("prediction CSV contains an unexpected track")
                if row["mode"] != mode or int(row["q"]) != 2:
                    raise ValueError("prediction CSV contains a mismatched mode or dimension")
                scenario_id = int(row["scenario_id"])
                node_id = int(row["node_id"])
                key = scenario_id, iteration, samples, node_id
                if previous_key is not None and key <= previous_key:
                    raise ValueError("prediction keys must be unique and strictly ordered")
                previous_key = key
                unit = scenario_id, iteration, samples
                if unit != previous_unit:
                    units[iteration, samples] += 1
                    unit_scenarios[iteration, samples].add(scenario_id)
                    previous_unit = unit
                x = parse_coordinate(row["x"])
                y = parse_coordinate(row["y"])
                is_missing = not row["x"].strip() or not row["y"].strip()
                missing += int(is_missing)
                nonfinite += int(
                    not is_missing and (not math.isfinite(x) or not math.isfinite(y))
                )
                records += 1
    if not records:
        raise ValueError("prediction archive is empty")
    if set(units) != {(1, 1), (1, 10), (2, 1), (2, 10)}:
        raise ValueError("prediction archive does not contain all released tracks")
    if set(units.values()) != {expected["scenarios"]}:
        raise ValueError("prediction archive has incomplete scenario-unit coverage")
    fingerprints = {
        scenario_fingerprint(scenario_ids)
        for scenario_ids in unit_scenarios.values()
    }
    if len(fingerprints) != 1:
        raise ValueError("prediction tracks cover different physical scenarios")
    return {
        "records": records,
        "missing_coordinates": missing,
        "nonfinite_coordinates": nonfinite,
        "physical_scenarios": expected["scenarios"],
        "scenario_fingerprint": fingerprints.pop(),
    }


def inspect_scenario_scores(path: Path, mode: str) -> dict:
    expected = MODE_COUNTS[mode]
    counts = defaultdict(int)
    scenario_ids = defaultdict(set)
    penalized_pairs = 0
    eligible_pairs = 0
    non_scorable_units = 0
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SCENARIO_SCORE_COLUMNS:
            raise ValueError("released scorer produced unexpected scenario-score columns")
        for row in reader:
            iteration = int(row["iteration"])
            samples = int(row["samples"])
            scenario_id = int(row["scenario_id"])
            if row["mode"] != mode or int(row["q"]) != 2:
                raise ValueError("released scenario score has a mismatched mode or dimension")
            track = iteration, samples
            if scenario_id in scenario_ids[track]:
                raise ValueError("released scenario-score keys are not unique")
            scenario_ids[track].add(scenario_id)
            counts[track] += 1
            penalized_pairs += int(row["penalized_pairs"])
            eligible_pairs += int(row["eligible_pairs"])
            non_scorable_units += int(not math.isfinite(float(row["mean_pair_error"])))
    if sum(counts.values()) != expected["units"]:
        raise ValueError("released scorer produced an incomplete scenario population")
    if set(counts.values()) != {expected["scenarios"]}:
        raise ValueError("released scorer produced incomplete initialization tracks")
    track_fingerprints = {
        scenario_fingerprint(identifiers) for identifiers in scenario_ids.values()
    }
    if len(track_fingerprints) != 1:
        raise ValueError("released scorer used different scenarios across tracks")
    return {
        "scenario_units": sum(counts.values()),
        "scenario_fingerprint": track_fingerprints.pop(),
        "eligible_pairs": eligible_pairs,
        "penalized_pairs": penalized_pairs,
        "penalized_pair_rate": penalized_pairs / eligible_pairs if eligible_pairs else None,
        "non_scorable_units": non_scorable_units,
    }


def read_track_summary(path: Path, mode: str) -> dict[str, dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2:
        raise ValueError("released scorer must produce two sample tracks per mode")
    result = {}
    for row in rows:
        if row["mode"] != mode or int(row["q"]) != 2:
            raise ValueError("released track summary has a mismatched mode or dimension")
        samples = int(row["samples"])
        if samples not in (1, 10) or str(samples) in result:
            raise ValueError("released track summary has duplicate or unknown samples")
        if int(row["scenario_units"]) != MODE_COUNTS[mode]["track_units"]:
            raise ValueError("released track summary has incomplete coverage")
        result[str(samples)] = {
            "scenario_units": int(row["scenario_units"]),
            "scored_scenario_units": int(row["scored_scenario_units"]),
            "non_scorable_units": int(row["non_scorable_units"]),
            "eligible_pairs": int(row["eligible_pairs"]),
            "penalized_pairs": int(row["penalized_pairs"]),
            "penalized_pair_rate": float(row["penalized_pair_rate"]),
            "balanced_s50_cm": float(row["balanced_s50"]),
            "balanced_s75_cm": float(row["balanced_s75"]),
            "balanced_s90_cm": float(row["balanced_s90"]),
        }
    if set(result) != {"1", "10"}:
        raise ValueError("released track summary is incomplete")
    return result


def validate_outputs(directory: Path, mode: str) -> dict:
    missing = [name for name in OUTPUT_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"released scorer outputs are missing {missing}")
    scenario_summary = inspect_scenario_scores(
        directory / "scenario_scores.csv.gz", mode
    )
    tracks = read_track_summary(directory / "track_summary.csv", mode)
    if scenario_summary["non_scorable_units"] != sum(
        row["non_scorable_units"] for row in tracks.values()
    ):
        raise ValueError("released scenario and track summaries disagree")
    return {"scenario_scores": scenario_summary, "tracks": tracks}


def score(
    *,
    scorer: Path,
    scenarios: Path,
    scorer_scenarios: Path,
    predictions: Path,
    benchmark_report: Path,
    method: str,
    mode: str,
    output_directory: Path,
    binary: Path | None = None,
    command: list[str] | None = None,
    force: bool = False,
) -> dict:
    mode = mode.upper()
    if mode not in MODE_COUNTS:
        raise ValueError(f"unknown mode {mode}")
    require_hash(scenarios, SCENARIO_SHA256, "scenario archive")
    require_hash(scorer, SCORER_SHA256, "released scorer")
    scenario_input = verify_scorer_scenarios(scenarios, scorer_scenarios)
    prediction_summary = inspect_predictions(predictions, mode)
    report = json.loads(benchmark_report.read_text())
    archive_metadata = report.get("prediction_archive")
    if not isinstance(archive_metadata, dict):
        raise ValueError("benchmark report has no prediction archive metadata")
    for key in ("records", "missing_coordinates", "nonfinite_coordinates"):
        if archive_metadata.get(key) != prediction_summary[key]:
            raise ValueError(f"benchmark and prediction archive disagree about {key}")
    prediction_sha256 = file_sha256(predictions)
    benchmark_report_sha256 = file_sha256(benchmark_report)
    diagnostic_program_sha256 = file_sha256(VALID_OUTPUT_DIAGNOSTICS)

    if output_directory.exists():
        if not force:
            manifest_path = output_directory / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                same_inputs = (
                    manifest.get("method") == method
                    and manifest.get("mode") == mode
                    and manifest.get("scenario_sha256") == SCENARIO_SHA256
                    and manifest.get("scorer_scenarios_sha256")
                    == scenario_input["scorer_input_sha256"]
                    and manifest.get("scorer_sha256") == SCORER_SHA256
                    and manifest.get("prediction_sha256") == prediction_sha256
                    and manifest.get("benchmark_report_sha256")
                    == benchmark_report_sha256
                )
                if (
                    same_inputs
                    and manifest.get("protocol_sha256") == file_sha256(PROTOCOL)
                    and manifest.get("diagnostic_program_sha256")
                    == diagnostic_program_sha256
                    # a score of the predictions under the coordinate rule of every scorer
                    and (manifest.get("coordinate_bound") or {}).get("bound_cm")
                    == scoring.COORDINATE_BOUND_CM
                ):
                    validate_outputs(output_directory, mode)
                    output_hashes = manifest.get("official_outputs", {})
                    if set(output_hashes) != set(OUTPUT_FILES):
                        raise ValueError("official scorer output manifest is incomplete")
                    for name, expected in output_hashes.items():
                        path = output_directory / name
                        if not path.is_file() or file_sha256(path) != expected:
                            raise ValueError(f"changed official scorer output {name}")
                    diagnostic_hashes = manifest.get("diagnostic_outputs", {})
                    if set(diagnostic_hashes) != set(DIAGNOSTIC_FILES):
                        raise ValueError("valid-output diagnostic manifest is incomplete")
                    for name, expected in diagnostic_hashes.items():
                        path = output_directory / name
                        if not path.is_file() or file_sha256(path) != expected:
                            raise ValueError(f"changed diagnostic output {name}")
                    return manifest
                if not same_inputs:
                    raise ValueError(
                        f"existing score directory is not reusable {output_directory}"
                    )
            else:
                raise ValueError(
                    f"existing score directory is not reusable {output_directory}"
                )
        shutil.rmtree(output_directory)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.", dir=output_directory.parent
        )
    )
    try:
        prediction_csv = temporary / "predictions.csv"
        with zipfile.ZipFile(predictions) as archive:
            with archive.open("predictions.csv") as source, prediction_csv.open(
                "wb"
            ) as destination:
                shutil.copyfileobj(source, destination)
        # A finite coordinate beyond the bound is missing for every scorer, so its point is written as the CLI writes a
        # missing one before the released scorer reads the copy. The archive keeps what the method returned.
        points_made_missing = scoring.bound_prediction_csv(prediction_csv)
        scorer_output = temporary / "official"
        scorer_command = [
            sys.executable,
            str(scorer.resolve()),
            "score-predictions",
            "--scenarios",
            str(scorer_scenarios.resolve()),
            "--predictions",
            str(prediction_csv),
            "--output-dir",
            str(scorer_output),
            "--modes",
            mode,
            "--samples",
            "1,10",
            "--dimensions",
            "2",
        ]
        subprocess.run(scorer_command, check=True)
        verified = validate_outputs(scorer_output, mode)
        if (
            verified["scenario_scores"]["scenario_fingerprint"]
            != prediction_summary["scenario_fingerprint"]
        ):
            raise ValueError("prediction and scorer scenario populations differ")
        diagnostic_summary = valid_output_diagnostics.generate(
            scorer=scorer,
            scenarios=scorer_scenarios,
            predictions_csv=prediction_csv,
            mode=mode,
            output_directory=temporary / "diagnostics",
        )
        if any(
            row["scenario_units"] != MODE_COUNTS[mode]["track_units"]
            for row in diagnostic_summary["tracks"].values()
        ):
            raise ValueError("valid-output diagnostic has incomplete scenario coverage")
        diagnostic_penalized_pairs = sum(
            row["eligible_pairs"] - row["valid_eligible_pairs"]
            for row in diagnostic_summary["tracks"].values()
        )
        if diagnostic_penalized_pairs != verified["scenario_scores"][
            "penalized_pairs"
        ]:
            raise ValueError(
                "released scorer and valid-output diagnostic disagree about invalid pairs"
            )
        for name in OUTPUT_FILES:
            (scorer_output / name).replace(temporary / name)
        scorer_output.rmdir()
        prediction_csv.unlink()
        output_hashes = {
            name: file_sha256(temporary / name) for name in OUTPUT_FILES
        }
        diagnostic_hashes = {
            name: file_sha256(temporary / name) for name in DIAGNOSTIC_FILES
        }
        manifest = {
            "schema_version": "solid.twr-cloud-bern.official-score.v2",
            "protocol": str(PROTOCOL.relative_to(SCRIPT_DIRECTORY.parents[2])),
            "protocol_sha256": file_sha256(PROTOCOL),
            "method": method,
            "mode": mode,
            "scenario_sha256": SCENARIO_SHA256,
            "scenario_members": scenario_input,
            "scorer_scenarios_sha256": scenario_input["scorer_input_sha256"],
            "scorer_sha256": SCORER_SHA256,
            "prediction_archive": predictions.name,
            "prediction_sha256": prediction_sha256,
            "prediction_summary": prediction_summary,
            "coordinate_bound": {
                "bound_cm": scoring.COORDINATE_BOUND_CM,
                "points_made_missing": points_made_missing,
            },
            "benchmark_report": benchmark_report.name,
            "benchmark_report_sha256": benchmark_report_sha256,
            "binary_sha256": file_sha256(binary) if binary else None,
            "benchmark_command": command,
            "official_outputs": output_hashes,
            "official_summary": verified,
            "diagnostic_outputs": diagnostic_hashes,
            "diagnostic_program_sha256": diagnostic_program_sha256,
            "valid_output_diagnostic": diagnostic_summary,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temporary.replace(output_directory)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--scorer-scenarios", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mode", choices=("af", "aac", "anc"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    manifest = score(
        scorer=args.scorer,
        scenarios=args.scenarios,
        scorer_scenarios=args.scorer_scenarios,
        predictions=args.predictions,
        benchmark_report=args.benchmark_report,
        method=args.method,
        mode=args.mode,
        output_directory=args.output_directory,
        binary=args.binary,
        force=args.force,
    )
    print(json.dumps(manifest["official_summary"], indent=2))


if __name__ == "__main__":
    main()
