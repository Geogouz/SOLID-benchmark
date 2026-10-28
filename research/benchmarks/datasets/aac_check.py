"""Run only the eighteen frozen TIERS AAC demonstrations with training-selected inputs."""

import argparse
import copy
import html
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import aac_preflight
import aac_training
import calibration
import pilot

OUTPUT = pilot.audit.DATA / "aac-check"
JIN_REVIEW = pilot.audit.DATA / "aac-jin-review"
LABELS = {"jin-2021": "Jin 2021", "ade-bp-2015": "Empirical ADE-BP 2015",
          "fnl": "FNL 2025", "strong": "STRONG 2021"}
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def construct(demonstrations, fitted, coordinates):
    expected = {(a, k, n) for a in ("army", "corner", "diagonal") for k in (1, 4, 6) for n in (1, 10)}
    keys = [(d["acquisition"], d["anchor_count"], d["samples"]) for d in demonstrations]
    if len(keys) != 18 or set(keys) != expected:
        raise ValueError("only the eighteen prepared demonstrations are authorized")
    units, metadata, references, identities = [], [], {}, {}
    for demo in demonstrations:
        acquisition, k, samples = demo["acquisition"], demo["anchor_count"], demo["samples"]
        key = acquisition, demo["source_unit"], k
        sid = identities.setdefault(key, len(identities) + 1)
        problem = copy.deepcopy(demo["problem"])
        if (problem["mode"] != "AnchorAided" or len(problem["anchors"]) != k
                or [a["id"] for a in problem["anchors"]] != list(range(1, k+1))
                or problem["agent_ids"] != list(range(k+1, 9))):
            raise ValueError("prepared node roles differ")
        for observation, source in zip(problem["observations"], demo["source_links"], strict=True):
            values = observation.pop("ranges_cm")
            original = demo["original_messages"][source]["reports"]
            if values != [100*r[1] for r in original] or len(values) != samples:
                raise ValueError("original message values changed")
            corrected = calibration.corrected(values, fitted["correction"])
            observation.update(tof_ns=[], calibrated_samples=[dict(distance_cm=float(v), scale_cm=fitted["scale_cm"])
                                                               for v in corrected])
        area = aac_preflight.operating_area(problem["anchors"])
        if area != demo["proposed_operating_area"]:
            raise ValueError("prepared operating area changed")
        units.append(dict(id=sid, samples=samples, problem=problem, area=area))
        if str(sid) not in references:
            references[str(sid)] = aac_preflight.scoring_reference(problem, coordinates[acquisition])
            metadata.append(dict(id=sid, reference=str(sid), acquisition=acquisition,
                                 source_unit=demo["source_unit"], anchors=k))
    if len(identities) != 9:
        raise ValueError("sample tracks must use the same nine layouts")
    paired = {(u["id"], u["samples"]): u for u in units}
    for sid in identities.values():
        one, ten = (paired[sid, n]["problem"]["observations"] for n in (1, 10))
        if any(a["calibrated_samples"] != b["calibrated_samples"][:1] for a, b in zip(one, ten, strict=True)):
            raise ValueError("one-sample values are not prefixes of ten-sample blocks")
    profile = dict(free={}, anchored={str(n): {"2": copy.deepcopy(fitted["range_error"])} for n in (1, 10)})
    return dict(profile=profile, units=units), dict(units=metadata, coordinates_cm=references)


def baseline_settings(fitted):
    roster = read(pilot.evidence.retained_data.EVIDENCE)
    baselines, sources = [], {}
    for method in LABELS:
        row = next(m for m in roster["methods"] if m["id"] == method)
        path = pilot.evidence.retained_data.run_directory(method, "AAC", row["results"]["AAC"]) / "benchmark.json"
        pilot.evidence.official_scorer.require_hash(path, row["results"]["AAC"]["benchmark_report_sha256"], "retained AAC baseline")
        record = read(path)
        config, options = copy.deepcopy(record["config"]), copy.deepcopy(record["method_config"])
        policy = "all-spatial" if method == "strong" else "all-planar"
        if (record["system_dimension_policy"] != policy or config["max_time_micros"] != 0
                or config["aggregation"] != "Median"):
            raise ValueError("unexpected retained AAC configuration")
        config["dimension_policy"] = policy.removeprefix("all-")
        if method == "ade-bp-2015":
            options.update(area=None, particles=500, iterations=10, importance_iterations=256,
                           ranging_model=fitted["ade_ranging_model"])
        baselines.append(dict(method=method, method_config=options, solve_config=config))
        sources[str(path)] = digest(path)
    return baselines, sources


def prepare():
    if OUTPUT.exists():
        raise ValueError("AAC check already retained, do not overwrite")
    preflight = aac_training.verify_preflight()
    for directory in (aac_training.OUTPUT, aac_training.ROUND_CHECK):
        for name, expected in read(directory / "manifest.json")["files"].items():
            pilot.evidence.official_scorer.require_hash(directory / name, expected, "training evidence")
    model = read(aac_training.OUTPUT / "profiles.json")["fitted"]
    original = pilot.OUTPUT / "reference.json"
    request, reference = construct(preflight["demonstrations"], model, read(original)["coordinates_cm"])
    request["baselines"], sources = baseline_settings(model)
    for path in (aac_preflight.OUTPUT, aac_training.OUTPUT / "profiles.json",
                 aac_training.ROUND_CHECK / "manifest.json", original,
                 *pilot.present([Path(__file__), Path(__file__).with_name("predict.rs"), Path(__file__).with_name("README.md")])):
        sources[str(path)] = digest(path)
    write(OUTPUT / "request.json", request)
    write(OUTPUT / "reference.json", reference)
    write(OUTPUT / "inputs.json", dict(schema="solid.external.aac-check.v1", sources=sources,
        request_sha256=digest(OUTPUT / "request.json"), reference_sha256=digest(OUTPUT / "reference.json"),
        units=18, methods=list(LABELS), solves=90, models_fitted=0, full_comparison_authorized=False,
        population="Previously prepared first window per acquisition, first 1, 4 or 6 node IDs as anchors, both sample tracks",
        scoring="Released Bern horizontal pairwise score, balanced over acquisition and anchor count",
        training="Frozen Case 1 calibration and likelihood fits, synthetic training selection of ADE-BP message rounds",
        calibration=model["correction"], common_sample_scale_cm=model["scale_cm"],
        internal_dimensions={m: 3 if m == "strong" else 2 for m in LABELS}))


def execute():
    if OUTPUT.exists():
        raise ValueError("AAC check already prepared or executed, do not overwrite")
    revision, env = pilot.build_predictor()
    prepare()
    pilot.check_inputs(OUTPUT)
    write(OUTPUT / "freeze.json", dict(source_revision=revision, binary_sha256=digest(pilot.BINARY),
        input_manifest_sha256=digest(OUTPUT / "inputs.json"), scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        host=platform.platform(), machine=platform.machine(), workers=1, no_warmup=True))
    started = time.perf_counter()
    with (OUTPUT / "run.log").open("x") as log:
        subprocess.run([str(pilot.BINARY), str(OUTPUT / "request.json"), str(OUTPUT / "predictions.jsonl")],
                       cwd=pilot.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(OUTPUT / "execution.json", dict(wall_seconds=time.perf_counter()-started,
        predictions_sha256=digest(OUTPUT / "predictions.jsonl"), freeze_sha256=digest(OUTPUT / "freeze.json")))
    report()


def report():
    inputs = read(OUTPUT / "inputs.json")
    freeze, execution = read(OUTPUT / "freeze.json"), read(OUTPUT / "execution.json")
    for name, expected in (("request.json", inputs["request_sha256"]), ("reference.json", inputs["reference_sha256"]),
                           ("inputs.json", freeze["input_manifest_sha256"]), ("freeze.json", execution["freeze_sha256"]),
                           ("predictions.jsonl", execution["predictions_sha256"])):
        pilot.evidence.official_scorer.require_hash(OUTPUT / name, expected, "AAC execution evidence")
    if freeze["scorer_sha256"] != pilot.evidence.official_scorer.SCORER_SHA256:
        raise ValueError("scoring contract changed")
    request, reference = read(OUTPUT / "request.json"), read(OUTPUT / "reference.json")
    records = [json.loads(line) for line in (OUTPUT / "predictions.jsonl").read_text().splitlines()]
    scorer = pilot.evidence.load_scorer()
    scores = pilot.score_records(request, reference, records, scorer, methods=LABELS)
    metadata = {m["id"]: m for m in reference["units"]}
    diagnostics, collapsed = [], []
    units = {(u["id"], u["samples"]): u for u in request["units"]}
    for row, record in zip(scores, records, strict=True):
        row["anchors"] = metadata[row["id"]]["anchors"]
        estimate = record["estimate"]
        metrics = (estimate.get("metrics", {k: estimate[k] for k in
            ("dimensions", "selected_start", "evaluations", "iterations", "converged_starts", "terminations")
            if k in estimate}) if estimate else {})
        dimensions = metrics.get("dimensions")
        if dimensions is not None and dimensions != inputs["internal_dimensions"][row["method"]]:
            raise ValueError("actual inference dimension differs from frozen settings")
        diagnostics.append(dict(id=row["id"], samples=row["samples"], method=row["method"],
                                anchors=row["anchors"], acquisition=row["acquisition"], metrics=metrics))
        if row["anchors"] == 1 and row["failed_agents"] == 0:
            anchor = units[row["id"], row["samples"]]["problem"]["anchors"][0]["position"]
            if all(abs(p[0]-anchor["x"]) < 1e-6 and abs(p[1]-anchor["y"]) < 1e-6 for p in estimate["positions"]):
                collapsed.append(dict(id=row["id"], samples=row["samples"], method=row["method"], acquisition=row["acquisition"]))
    balanced = [dict(r, acquisition=f'{r["acquisition"]}/anchors-{r["anchors"]}') for r in scores]
    result = pilot.summarize(balanced, scorer, labels=LABELS, mode="AAC")
    result.update(status="small-check-complete", units=18, solves=len(records), full_comparison_executed=False,
                  models_fitted=0, source_revision=freeze["source_revision"], diagnostics=diagnostics,
                  dimensions=inputs["internal_dimensions"], wall_seconds=execution["wall_seconds"],
                  one_anchor_collapses=collapsed)
    result["by_anchor_count"] = {str(k): pilot.summarize([r for r in scores if r["anchors"] == k], scorer,
                                                        labels=LABELS, mode="AAC") for k in (1, 4, 6)}
    write(OUTPUT / "scores.json", scores)
    write(OUTPUT / "summary.json", result)
    paths = [p for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "manifest.json"]
    write(OUTPUT / "manifest.json", dict(files={p.name: digest(p) for p in paths}))
    pilot.register_files(paths + [OUTPUT / "manifest.json"])
    print(f"Retained {len(records)} AAC outputs. Full comparison remains unexecuted.", flush=True)


def build_adapter_test():
    """The TIERS adapter's test binary, whose Jin control test replays the published decoder on noiseless inputs."""
    revision, env = pilot.build_predictor()
    build = subprocess.run(["cargo", "test", "--release", "--locked", "--manifest-path", "research/Cargo.toml",
        "-p", "solid-cli", "--bin", "tiers-adapter", "--no-run", "--message-format=json"],
        cwd=pilot.ROOT, env=env, text=True, capture_output=True, check=True)
    artifacts = [json.loads(line) for line in build.stdout.splitlines()]
    binary, = [Path(r["executable"]) for r in artifacts if r.get("executable") and r["target"]["name"] == "tiers-adapter"]
    return revision, env, binary


def review_jin():
    output = JIN_REVIEW
    if output.exists():
        raise ValueError("Jin review already retained, do not overwrite")
    revision, env, binary = build_adapter_test()
    sources = [Path(__file__), Path(__file__).with_name("jin_controls.rs"), Path(__file__).with_name("predict.rs"),
        pilot.ROOT / "research/methods/src/jin.rs", pilot.ROOT / "research/methods/src/geometry.rs",
        pilot.ROOT / "research/Cargo.lock"]
    identity = dict(source_revision=revision, binary_sha256=digest(binary),
        source_files={pilot.audit.name(p): digest(p) for p in sources},
        author_repository="https://gitlab.com/jindi.et/sdp_sparsity",
        author_revision="5ed57a0ba8d8e0527c50ad1ef7bc8a59cd6e8201",
        paper_sha256=pilot.document(pilot.ROOT / "research/papers/references/closest-current/2021-jin-yin-zoubir-so-sparse-ranging-biases.pdf"),
        host=platform.platform(), machine=platform.machine())
    test = "tests::jin_noiseless_anchor_geometry"
    completed = subprocess.run([str(binary), test, "--exact", "--nocapture"],
        cwd=pilot.ROOT, env=env, text=True, capture_output=True, check=True)
    prefix = "JIN_RECOVERY_DIAGNOSTIC "
    result, = [json.loads(line[len(prefix):]) for line in completed.stdout.splitlines() if line.startswith(prefix)]
    expected = {(name, n) for name in ("one", "one-translated", "two", "four-collinear", "three-noncollinear") for n in (1, 10)}
    if len(result["cases"]) != 10 or {(c["case"], c["samples"]) for c in result["cases"]} != expected:
        raise ValueError("unexpected synthetic control population")
    result.update(identity=identity, models_fitted=0, real_data_solves=0, author_solver_executed=False,
                  full_comparison_executed=False)
    path = output / "review.json"
    write(path, result)
    write(output / "manifest.json", dict(files={path.name: digest(path)}))
    pilot.register_files([path, output / "manifest.json"])
    print("Retained ten noiseless Jin controls. No real-data inference or baseline replacement.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "report", "review-jin"))
    {"run": execute, "report": report, "review-jin": review_jin}[parser.parse_args().action]()
