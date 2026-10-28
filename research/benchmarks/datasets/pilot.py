"""Frozen TIERS AF transfer pilot with prediction-only method dispatch, and the runner of the pilot adapters.

Usage: pilot.py prepare|execute
  prepare writes the pilot's inputs without running a method, the stage a reader's reproduction (SOLID_READER=1) runs;
  execute builds the TIERS adapter and runs the pilot on them.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import html
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

import audit

HERE = Path(__file__).resolve().parent
import evidence
import isolation
import scoring

ROOT = audit.ROOT
OUTPUT = audit.DATA / "tiers-af-pilot"
BINARY = ROOT / "research/target/release/tiers-adapter"
WINDOWS = 10
LABELS = {"jin-2021": "Jin 2021", "fnl": "FNL 2025"}
COLORS = {"jin-2021": "#e08b00", "fnl": "#7a7a7a", "strong": "#3b6fb6", "ade-bp-2015": "#6750a4"}
READER = audit.READER
# The settings a reader's reproduction (SOLID_READER=1) reads in place of the retained Bern evidence, its run records
# and the IDLab availability record, which the repository does not ship: extracts of exactly what those reads return.
SETTINGS = ROOT / "settings/data"


def identity(path):
    return evidence.official_scorer.file_sha256(path)


def shipped(name):
    """A settings extract of settings/data and its provenance, {path: SHA-256}."""
    path = SETTINGS / name
    return evidence.read(path), {str(path): identity(path)}


def document(path):
    """The SHA-256 of a protocol document or retired source that a record names, None for a reader whose copy does not
    ship it (SOLID_READER=1)."""
    return None if READER and not Path(path).exists() else identity(path)


def present(paths):
    """The paths of a record's sources, without the protocol documents and retired sources a reader's copy does not
    ship (SOLID_READER=1)."""
    return [p for p in paths if not READER or Path(p).exists()]


def construct(ranges, acquisition, next_id):
    """Use timestamps and measured ranges only. No pose argument is accepted."""
    nodes = sorted({n for pair in ranges for n in pair})
    streams = {p: np.asarray(ranges[p]).copy() for p in audit.combinations(nodes, 2)}
    for rows in streams.values():
        if not len(rows) or not np.isfinite(rows).all() or (rows[:, 1] <= 0).any():
            raise ValueError("invalid recorded range stream")
        # ROS bag timestamps can coincide for distinct publisher messages.
        rows[:, 0] = rows[:, 2] + rows[:, 3] / 1e9
        if len({tuple(r) for r in rows[:, 2:]}) != len(rows):
            raise ValueError("duplicate forward message identity")
        if (np.diff(rows[:, 0]) <= 0).any():
            raise ValueError("nonunique or unordered forward timestamps")
    start = max(rows[0, 0] for rows in streams.values())
    stop = min(rows[-1, 0] for rows in streams.values())
    boundaries = np.linspace(start, stop, WINDOWS + 1)
    units, metadata = [], []
    for window, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        selected = {}
        for pair, rows in streams.items():
            indices = np.flatnonzero((rows[:, 0] >= left) & (rows[:, 0] < right))[:10]
            if len(indices) != 10:
                raise ValueError("a fixed window has fewer than ten messages on a link")
            selected[pair] = (indices, rows[indices])
        sid = next_id + window
        metadata.append(dict(id=sid, acquisition=acquisition, window=window,
            interval_seconds=[float(left), float(right)],
            messages={f"{a}-{b}": dict(indices=idx.tolist(), reports=rows.tolist())
                      for (a, b), (idx, rows) in selected.items()}))
        for samples in (1, 10):
            observations = []
            for (a, b), (_, rows) in selected.items():
                cm = rows[:samples, 1] * 100.0
                # The supplied metric-input API fixes uncertainty per observation.
                scales = np.clip(0.026 * cm + 7.0, 8.0, 55.0)
                observations.append(dict(
                    **{"from": a, "to": b}, to_is_anchor=False, tof_ns=[],
                    calibrated_samples=[dict(distance_cm=float(d), scale_cm=float(s)) for d, s in zip(cm, scales)]))
            typical = float(np.median([v["distance_cm"] for o in observations for v in o["calibrated_samples"]]))
            rng = np.random.default_rng(20260914 + sid)
            initial = rng.uniform(-typical, typical, (len(nodes), 2))
            units.append(dict(id=sid, samples=samples, problem=dict(
                agent_ids=nodes, anchors=[], observations=observations,
                initial_positions=[dict(x=float(x), y=float(y)) for x, y in initial], mode="AnchorFree")))
    return units, metadata


def retained_settings():
    if READER:
        settings, sources = shipped("tiers-settings.json")
        return settings["af"]["profile"], settings["af"]["baselines"], sources
    roster = evidence.read(evidence.retained_data.EVIDENCE)
    baselines, sources = [], {}
    for method in ("jin-2021", "fnl"):
        evidence.retained_data.run_record(method, "AF")
        row = next(r for r in roster["methods"] if r["id"] == method)
        directory = evidence.retained_data.run_directory(method, "AF", row["results"]["AF"])
        path = directory / "benchmark.json"
        report = evidence.read(path)
        config = copy.deepcopy(report["config"])
        # Resolve the retained CLI all-planar mode policy, not a new dimension choice.
        config["dimension_policy"] = "planar"
        if config["max_time_micros"] != 0 or config["aggregation"] != "Median":
            raise ValueError("unexpected retained baseline configuration")
        baselines.append(dict(method=method, method_config=report["method_config"], solve_config=config))
        sources[str(path)] = identity(path)
    path = evidence.retained_data.DATA / "joint-information-evaluation/runs/af/request.json"
    profile = evidence.read(path)["profile"]
    if profile["fit_scans"] != list(range(1, 12)) or "held_scan" in profile:
        raise ValueError("expected all-training Bern profile")
    sources[str(path)] = identity(path)
    return profile, baselines, sources


def prepare():
    if OUTPUT.exists():
        raise ValueError("pilot already prepared, retained inputs must not be overwritten")
    audit_path = audit.DATA / "input-audit.json"
    checked = evidence.read(audit_path)
    profile, baselines, sources = retained_settings()
    units, metadata, references = [], [], {}
    for recording in checked["tiers"]:
        path = audit.DATA / recording["file"]
        if identity(path) != recording["sha256"]:
            raise ValueError("TIERS source changed")
        ranges, poses = audit.tiers_events(path)
        if sorted(poses) != list(range(1, 9)):
            raise ValueError("expected all eight tracked TIERS nodes")
        name = path.name.split("_")[0]
        new_units, new_meta = construct(ranges, name, len(metadata) + 1)
        if new_units[0]["problem"]["agent_ids"] != sorted(poses):
            raise ValueError("range and mocap node identities differ")
        units.extend(new_units)
        metadata.extend(new_meta)
        references[name] = {str(n): (100 * np.median(rows[:, 1:3], axis=0)).tolist() for n, rows in poses.items()}
        sources[str(path)] = identity(path)
    if len(units) != 60 or len(references) != 3:
        raise ValueError("expected three complete static acquisitions")
    OUTPUT.mkdir()
    evidence.write(OUTPUT / "request.json", dict(profile=profile, baselines=baselines, units=units))
    evidence.write(OUTPUT / "reference.json", dict(coordinates_cm=references, units=metadata))
    evidence.write(OUTPUT / "inputs.json", dict(
        schema="solid.external.tiers.af-pilot.v1", sources=sources,
        input_audit_sha256=identity(audit_path),
        request_sha256=identity(OUTPUT / "request.json"), reference_sha256=identity(OUTPUT / "reference.json"),
        profile_track="Frozen Bern shape and availability with raw TIERS metric ranges",
        uncertainty="clip(0.026 * measured_cm + 7, 8, 55), shared by every method",
        windows_per_acquisition=WINDOWS, samples=[1, 10], modes=["AF"],
        acquisition_count=3, units_per_method=60, models_fitted=0,
        measurements="First ten forward reports in each fixed common-time window. One sample is their first report.",
        report_columns=["header_seconds", "range_m", "header_sec", "header_nanosec", "publisher_sequence"],
        reference="Coordinatewise median of supplied static mocap x and y. No fitted antenna offset.",
        score="Released q=2 pair loss and diameter cap transferred to TIERS, balanced over acquisitions",
        methods=list(LABELS), unsupported_af=["ade-bp-2015", "strong"]))
    print("Prepared 60 AF solves per method without running a positioning method", flush=True)


def check_inputs(output=OUTPUT):
    manifest = evidence.read(output / "inputs.json")
    for name in ("request", "reference"):
        if identity(output / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError("prepared pilot input changed")
    for path, digest in manifest["sources"].items():
        if identity(Path(path)) != digest:
            raise ValueError("pilot source changed")
    return manifest


def build_predictor():
    if audit.uncommitted():
        raise ValueError("commit the source before the pilot")
    revision = audit.revision()
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLID_")}
    subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", "research/Cargo.toml",
                    "-p", "solid-cli", "--bin", "tiers-adapter"], cwd=ROOT, env=env, check=True)
    if subprocess.check_output([str(BINARY), "--identity"], text=True).strip() != isolation.source_identity():
        raise ValueError("binary source identity mismatch")
    return revision, env


def execute(output=OUTPUT):
    check_inputs(output)
    if (output / "freeze.json").exists():
        raise ValueError("execution already frozen, do not overwrite it")
    revision, env = build_predictor()
    evidence.write(output / "freeze.json", dict(source_revision=revision,
        binary_sha256=identity(BINARY), input_manifest_sha256=identity(output / "inputs.json"),
        protocol_sha256=document(HERE / "README.md"), scorer_sha256=evidence.official_scorer.SCORER_SHA256,
        host=platform.platform(), machine=platform.machine(), workers=1, no_warmup=True))
    started = time.perf_counter()
    with (output / "run.log").open("x") as log:
        subprocess.run([str(BINARY), str(output / "request.json"), str(output / "optimizer.jsonl")],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = [json.loads(line) for line in (output / "optimizer.jsonl").read_text().splitlines()]
    with (output / "predictions.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    evidence.write(output / "execution.json", dict(wall_seconds=time.perf_counter() - started,
        optimizer_runs=180, prediction_records=len(records),
        optimizer_sha256=identity(output / "optimizer.jsonl"), predictions_sha256=identity(output / "predictions.jsonl"),
        freeze_sha256=identity(output / "freeze.json")))
    print("Completed 180 optimizer runs and retained 240 method outputs", flush=True)


def run_pilot(requests, workers, folder, binary=None):
    """The records of every request, one adapter process per request and at most `workers` processes at a time.

    requests holds (key, request) pairs of the JSON adapters (the tiers-adapter by default). Every process runs with
    one thread for Rayon, OpenBLAS and OpenMP, and a request that crashes the adapter is solved again unit by unit and
    then method by method, the crashing solve recorded as its method's failure on its unit with a warning
    (isolation.solve_request, isolation.ABORT_RULE). Every adapter accepts each of those parts. The records of a request
    are cached under folder by the request, the binary and the thread settings, so a run resumes where it stopped.
    Returns {key: records}."""
    folder, binary = Path(folder), Path(binary or BINARY)
    folder.mkdir(parents=True, exist_ok=True)
    if not binary.exists():
        raise ValueError(f"build the adapter first: {binary}")
    binary_sha = identity(binary)

    def one(key, request):
        name = hashlib.sha256(isolation.encoded(dict(request=request, binary=binary_sha, threads=isolation.SINGLE_THREAD))).hexdigest()
        target = folder / f"{key}-{name[:12]}.jsonl"
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix=f"{key}-", dir=folder) as temporary:
                lines = isolation.solve_request(binary, request, Path(temporary) / "request")
                staged = Path(temporary) / "predictions.jsonl"
                staged.write_text("".join(line + "\n" for line in lines))
                staged.replace(target)
        records = [json.loads(line) for line in target.read_text().splitlines()]
        if len(records) != len(request["units"]) * len(request["baselines"]):
            raise ValueError(f"adapter output population changed for {key}")
        return key, records

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, key, request) for key, request in requests]
        try:
            for future in as_completed(futures):
                key, records = future.result()
                results[key] = records
        except BaseException:
            # without this the pool would finish every queued request before the error surfaces
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return {key: results[key] for key, _ in requests}


def score_records(request, reference, records, scorer, methods=LABELS):
    units = {(u["id"], u["samples"]): u for u in request["units"]}
    metadata = {r["id"]: r for r in reference["units"]}
    expected = {(sid, samples, method) for sid, samples in units for method in methods}
    actual = [(r["id"], r["samples"], r["method"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicated method population")
    scores = []
    for row in records:
        key = row["id"], row["samples"]
        agents = units[key]["problem"]["agent_ids"]
        if row["agent_ids"] != agents:
            raise ValueError("output agent identities changed")
        acquisition = metadata[key[0]]["acquisition"]
        reference_key = metadata[key[0]].get("reference", acquisition)
        truth = {int(n): tuple(p) for n, p in reference["coordinates_cm"][reference_key].items()}
        estimate = row["estimate"]
        if estimate is not None and estimate.get("agent_ids", agents) != agents:
            raise ValueError("estimator reordered the agents")
        positions = estimate["positions"] if estimate is not None else [None] * len(agents)
        if len(positions) != len(agents):
            raise ValueError("output agent count changed")
        # the coordinate rule of every scorer: a point with a coordinate that is not finite or lies beyond the bound is
        # missing, and so failed, and its pairs are charged and counted as penalized (scoring.scored_point)
        valid = {n: tuple(p) for n, p in zip(agents, positions) if scoring.scored_point(p) is not None}
        pairs = list(audit.combinations(agents, 2))
        eligible = [(a, b) for a, b in pairs if a in valid and b in valid]
        penalty = scorer.scenario_diameter(truth)
        total, penalized, error = scoring.score_pairs(scorer, truth, valid, pairs, penalty, 2)
        _, _, conditional = scoring.score_pairs(scorer, truth, valid, eligible, penalty, 2)
        pair_errors = [min(abs(math.dist(valid[a], valid[b]) - math.dist(truth[a], truth[b])), penalty)
                       if a in valid and b in valid else penalty for a, b in pairs]
        scores.append(dict(id=key[0], samples=key[1], acquisition=acquisition, method=row["method"],
            mean_pair_error_cm=error, conditional_error_cm=conditional if eligible else None,
            expected_agents=len(agents), failed_agents=len(agents) - len(valid),
            eligible_pairs=total, penalized_pairs=penalized, valid_pairs=len(eligible),
            pair_errors_cm=pair_errors, pairs=pairs, diameter_cm=penalty,
            elapsed_ms=row["elapsed_ns"] / 1e6, failure=row["failure"], starts=isolation.reported_starts(row),
            stayed_at_start=isolation.reported_stay(row)))
    return scores


def summarize(scores, scorer, labels=LABELS, sample_counts=(1, 10), mode="AF", colors=COLORS):
    methods = []
    for method, label in labels.items():
        tracks, curves, conditional_curves = {}, {}, {}
        for samples in sample_counts:
            rows = [r for r in scores if r["method"] == method and r["samples"] == samples]
            cells, successful = defaultdict(list), defaultdict(list)
            for row in rows:
                cells[row["acquisition"]].append(row["mean_pair_error_cm"])
                if row["conditional_error_cm"] is not None:
                    successful[row["acquisition"]].append(row["conditional_error_cm"])

            def curve(values):
                points, previous = [dict(error_cm=0, probability=0)], 0
                for _, probability, error in scorer.balanced_cdf_points(values):
                    points.extend([dict(error_cm=error, probability=previous), dict(error_cm=error, probability=probability)])
                    previous = probability
                return points if previous else []

            curves[samples], conditional_curves[samples] = curve(cells), curve(successful)
            tracks[str(samples)] = dict(
                **{f"s{q}": scorer.balanced_percentile(cells, q / 100) for q in (50, 75, 90)},
                **{f"conditional_s{q}": scorer.balanced_percentile(successful, q / 100) if successful else None for q in (50, 75, 90)},
                scenario_units=len(rows), expected_agents=sum(r["expected_agents"] for r in rows),
                failed_agents=sum(r["failed_agents"] for r in rows),
                failed_scenario_units=sum(r["failed_agents"] > 0 for r in rows),
                conditional_scored_scenario_units=sum(r["conditional_error_cm"] is not None for r in rows),
                valid_pair_coverage=sum(r["valid_pairs"] for r in rows) / sum(r["eligible_pairs"] for r in rows),
                mean_ms=float(np.mean([r["elapsed_ms"] for r in rows])),
                by_acquisition={a: {f"s{q}": scorer.percentile(sorted(v), q / 100) for q in (50, 75, 90)} for a, v in cells.items()})
        methods.append(dict(id=method, label=label, color=colors[method], results={mode: dict(
            status="measured", tracks=tracks, curves=curves, conditional_curves=conditional_curves, valid_output=dict(tracks=tracks))}))
    return dict(methods=methods, metric="TIERS transferred pairwise score", source="summary of retained predictions")


def register_files(paths):
    if READER:
        # The inventory of retained files belongs to the data collection of the research record, which a reader lacks.
        return
    inventory_path = evidence.retained_data.DATA / "inventory.json"
    inventory = evidence.read(inventory_path)
    for path in paths:
        inventory["files"][str(path)] = dict(path=str(path.relative_to(evidence.retained_data.DATA)),
                                            sha256=identity(path), bytes=path.stat().st_size)
    evidence.write(inventory_path, inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "execute"))
    {"prepare": prepare, "execute": execute}[parser.parse_args().command]()
