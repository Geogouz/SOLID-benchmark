"""Check STRONG's transferred radius using calibration-only synthetic inputs."""

import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

import aac_check
import aac_training
import pilot
import tiers_static_check
# The rule of every ranking of candidates and STRONG's same-layout candidates live in baseline_selection.py, so that a
# runner reads them without this module's design imports. They stay names of this module for its earlier users.
from baseline_selection import (DIMENSIONS, GRID_CM, NO_ROBUST_LOSS_CM, STARTS, TOLERANCE_CM, joint_candidates,  # noqa: F401
                                rank_candidates, tied_at_top)

OUTPUT = pilot.audit.DATA / "strong-threshold"
# the candidates of the retained threshold studies: 40 cm or a multiple of the calibration scale
RADII = {"retained-40": None, "sigma": 1., "1.5-sigma": 1.5, "2.5-sigma": 2.5}
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def variants(config, scale):
    if (config["robust_delta"] != 40 or config["max_time_micros"] != 0
            or config["dimension_policy"] != "spatial" or scale <= 0 or not np.isfinite(scale)):
        raise ValueError("unexpected retained STRONG configuration or scale")
    return [dict(name=name, solve_config=dict(copy.deepcopy(config),
                 robust_delta=40. if factor is None else factor * scale)) for name, factor in RADII.items()]


def tiers_units(source, reference, mode):
    expected = {(i, n) for i in range(1, 31) for n in (1, 10)}
    if (source["data_role"] != "synthetic-training" or mode not in ("AAC", "ANC")
            or len(source["units"]) != 60 or {(u["id"], u["samples"]) for u in source["units"]} != expected):
        raise ValueError("expected the retained sixty synthetic training inputs")
    units = copy.deepcopy(source["units"])
    for unit in units:
        p = unit["problem"]
        if p["mode"] != "AnchorAided":
            raise ValueError("unexpected training mode")
        if mode == "ANC":
            p["mode"] = "AnchorOnly"
            p["observations"] = [o for o in p["observations"] if o["to_is_anchor"]]
        unit.pop("area", None)
    return units, copy.deepcopy(reference)


def prepare():
    sources = {}

    def checked(folder, name, manifest="manifest.json"):
        path = folder / name
        expected = read(folder / manifest)["files"][name]
        pilot.evidence.official_scorer.require_hash(path, expected, "STRONG training source")
        sources[str(path)] = expected
        sources[str(folder / manifest)] = digest(folder / manifest)
        return read(path)

    source = checked(aac_training.OUTPUT, "request.json")
    reference = checked(aac_training.OUTPUT, "reference.json")
    fitted = checked(aac_training.OUTPUT, "profiles.json")["fitted"]
    jobs = []
    for mode in ("AAC", "ANC"):
        units, truth = tiers_units(source, reference, mode)
        settings, origin = tiers_static_check.anchored_baselines(mode, fitted)
        setting = next(s for s in settings if s["method"] == "strong")
        sources.update(origin)
        jobs.append(dict(name=f"tiers-{mode.lower()}", units=units, reference=truth,
                         method_config=setting["method_config"], variants=variants(setting["solve_config"], fitted["scale_cm"])))
    return jobs, sources


def summarize(request, reference, records, scorer):
    scores = pilot.score_records(request, reference, records, scorer, methods=RADII)
    metadata = {r["id"]: r for r in reference["units"]}
    outputs = {(r["id"], r["samples"], r["method"]): r for r in records}
    for row in scores:
        row["noiseless"] = metadata[row["id"]]["noiseless"]
        result = outputs[row["id"], row["samples"], row["method"]]["estimate"]
        row["converged"] = bool(result and result["metrics"]["converged"])
    groups = []
    for noiseless in (False, True):
        for samples in sorted({r["samples"] for r in scores}):
            for method in RADII:
                rows = [r for r in scores if (r["noiseless"], r["samples"], r["method"]) == (noiseless, samples, method)]
                cells = {}
                for row in rows:
                    cells.setdefault(row["acquisition"], []).append(row["mean_pair_error_cm"])
                groups.append(dict(noiseless=noiseless, samples=samples, method=method, units=len(rows),
                    s75=scorer.balanced_percentile(cells, .75), mean_cm=statistics.mean(r["mean_pair_error_cm"] for r in rows),
                    failed_agents=sum(r["failed_agents"] for r in rows), expected_agents=sum(r["expected_agents"] for r in rows),
                    converged=sum(r["converged"] for r in rows)))
    return scores, groups


def execute():
    if OUTPUT.exists():
        raise ValueError("STRONG threshold check already retained")
    revision, env, binary = aac_check.build_adapter_test()
    jobs, sources = prepare()
    write(OUTPUT / "freeze.json", dict(source_revision=revision, binary_sha256=digest(binary), sources=sources,
        protocol_sha256=digest(pilot.HERE / "README.md"), driver_sha256=digest(Path(__file__)),
        scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256, evaluation_rows_used=0,
        radii=RADII, workers=1, new_calibration_fits=0, selected_for_benchmark=False))
    results = []
    for job in jobs:
        result = solve(job, OUTPUT, env, binary)
        results.append(result)
        print(json.dumps(result), flush=True)
    write(OUTPUT / "summary.json", dict(jobs=results, evaluation_rows_used=0, selected_for_benchmark=False))
    paths = sorted(p for p in OUTPUT.rglob("*") if p.is_file())
    write(OUTPUT / "manifest.json", dict(files={str(p.relative_to(OUTPUT)): digest(p) for p in paths}))
    pilot.register_files(paths + [OUTPUT / "manifest.json"])


def solve(job, output, env, binary):
    folder = output / job["name"]
    request = dict(data_role="synthetic-training-strong-threshold", method_config=job["method_config"],
                   variants=job["variants"], units=job["units"])
    write(folder / "request.json", request)
    write(folder / "reference.json", job["reference"])
    write(folder / "inputs.json", {name: digest(folder / name) for name in ("request.json", "reference.json")})
    started = time.monotonic()
    with (folder / "run.log").open("x") as log:
        subprocess.run([str(binary), "strong_training::threshold_check", "--exact", "--ignored", "--nocapture"],
            env=dict(env, STRONG_TRAINING_INPUT=str(folder / "request.json"),
                     STRONG_TRAINING_OUTPUT=str(folder / "predictions.jsonl")),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    scores, groups = summarize(request, job["reference"], records, pilot.evidence.load_scorer())
    write(folder / "scores.json", scores)
    result = dict(name=job["name"], groups=groups,
        radii_cm={v["name"]: v["solve_config"]["robust_delta"] for v in job["variants"]},
        wall_seconds=time.monotonic() - started, outputs=len(records))
    write(folder / "summary.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    execute()
