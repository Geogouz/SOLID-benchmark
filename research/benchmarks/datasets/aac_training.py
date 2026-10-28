"""Fit Case 1 AAC profiles and assess ADE-BP particle sensitivity on synthetic training graphs."""

import argparse
import copy
import hashlib
import html
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import aac_preflight
import calibration
import pilot

sys.path.insert(0, str(Path(__file__).parent.parent / "bern/ade_bp"))
from fit_profiles import ade_negative_log_likelihood, fit_ade_model

OUTPUT = pilot.audit.DATA / "aac-training"
BINARY = pilot.ROOT / "research/target/release/aac-training"
PREFLIGHT_REVISION = "83aec8a3317b62f0b8d6403709b95665689651ea"
PARTICLES = (500, 2000, 5000)
CONTROLS = pilot.audit.DATA / "aac-training-controls"
BUDGETS = ("500-4-256", "5000-4-256", "500-4-1024", "5000-4-1024", "500-10-256", "5000-10-256")
ROUND_CHECK = pilot.audit.DATA / "aac-training-rounds"
ROUND_BUDGETS = ("500-4-256", "500-10-256")
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def verify_preflight():
    record = read(aac_preflight.OUTPUT)
    for name, expected in record["inputs"].items():
        if name.endswith("/README.md"):
            if pilot.READER:
                # The approved protocol is a commit of the research history, which a reader's copy does not hold.
                continue
            content = subprocess.check_output(["git", "show", f"{PREFLIGHT_REVISION}:{name}"], cwd=pilot.ROOT)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("approved preflight protocol differs")
        else:
            pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "AAC preflight")
    return record


def aac_scale(reference):
    return np.clip(7 + .026*np.asarray(reference), 8, 55)


def fit_models(cells, correction):
    measured, reference, weights = calibration.pooled(cells)
    calibrated = calibration.corrected(measured, correction)
    residual = calibrated-reference
    shape = calibration.fit_shape(residual, aac_scale(reference), weights)
    ade, _ = fit_ade_model(calibrated, reference)
    return dict(correction=correction, scale_cm=float(np.sqrt(weights @ residual**2)),
                range_error=shape, ade_ranging_model=ade)


def held_scores(cells, model):
    measured, reference, weights = calibration.pooled(cells)
    calibrated = calibration.corrected(measured, model["correction"])
    residual = calibrated-reference
    shape = model["range_error"]
    scale = aac_scale(reference)
    width = np.where(residual >= 0, shape["positive_scale"], shape["negative_scale"])
    z = residual/(scale*width)
    # The constant normalizer cancels when comparing fits of this fixed AAC family.
    loss = .49*(np.sqrt(1+(z/.7)**2)-1) + np.log(scale*(shape["positive_scale"]+shape["negative_scale"])/2)
    p = model["ade_ranging_model"]
    parameters = [p[k] for k in ("positive_slope", "positive_intercept_cm", "negative_slope", "negative_intercept_cm")]
    return dict(aac_loss_up_to_constant=float(weights @ loss),
                ade_nll_per_report=ade_negative_log_likelihood(parameters, calibrated, residual)/len(residual),
                calibrated_mae_cm=float(weights @ np.abs(residual)))


def fit_case1(path, frozen):
    cells, excluded = calibration.load_case1(path)
    folds = []
    for family, held, training, validation in calibration.splits(cells):
        correction = calibration.fit_correction(training, frozen["selected"])
        model = fit_models(training, correction)
        folds.append(dict(family=family, held=list(held), training_conditions=len(training),
                          held_conditions=len(validation), parameters=model, scores=held_scores(validation, model)))
    fitted = fit_models(cells, frozen["correction"])
    return dict(source_sha256=digest(path), evaluation_rows_used=0, selected_family=frozen["selected"],
                eligible_rows=sum(len(c) for c in cells.values()), excluded_lines=excluded,
                folds=folds, fitted=fitted, sample_tracks_share_shape=True), cells


def synthetic(cells, model):
    """No test acquisition is accepted. Synthetic distances are corrupted by Case 1 residual draws."""
    units, metadata, coordinates = [], [], {}
    rng = np.random.default_rng(20260916)
    correction = model["correction"]
    for anchors_count in range(1, 7):
        for replicate in range(5):
            sid = len(metadata)+1
            points = rng.uniform([0, 0], [800, 900], size=(8, 2))
            known = {n: [*points[n-1], 0.] for n in range(1, anchors_count+1)}
            initial = rng.uniform([-900, -900], [900, 900], size=(8, 2))
            observations = []
            for a, b in combinations(range(1, 9), 2):
                distance = float(np.linalg.norm(points[a-1]-points[b-1]))
                nearest = min(abs(k[0]*100-distance) for k in cells)
                keys = [k for k in cells if abs(k[0]*100-distance) == nearest]
                rows = cells[keys[int(rng.integers(len(keys)))]]
                selected = rng.integers(len(rows), size=10)
                error = calibration.corrected(rows[selected, 1], correction)-rows[selected, 2]
                if replicate == 0:
                    error = np.zeros(10)
                values = np.maximum(.001, distance+error)
                observations.append(dict(**{"from": a, "to": b}, to_is_anchor=False, tof_ns=[],
                    calibrated_samples=[dict(distance_cm=float(v), scale_cm=model["scale_cm"]) for v in values]))
            source = dict(id=sid, samples=10, problem=dict(agent_ids=list(range(1, 9)), anchors=[],
                observations=observations, initial_positions=[dict(x=float(x), y=float(y)) for x, y in initial], mode="AnchorFree"))
            problem, _ = aac_preflight.construct(source, list(known), known)
            for samples in (1, 10):
                prepared = copy.deepcopy(problem)
                for observation in prepared["observations"]:
                    observation["calibrated_samples"] = [dict(distance_cm=d, scale_cm=model["scale_cm"])
                        for d in observation.pop("ranges_cm")[:samples]]
                    observation["tof_ns"] = []
                units.append(dict(id=sid, samples=samples, problem=prepared,
                                  area=aac_preflight.operating_area(prepared["anchors"])))
            key = str(sid)
            coordinates[key] = aac_preflight.scoring_reference(problem, {str(n): points[n-1].tolist() for n in range(1, 9)})
            metadata.append(dict(id=sid, reference=key, acquisition=f"anchors-{anchors_count}",
                                 anchors=anchors_count, noiseless=replicate == 0, replicate=replicate))
    return units, dict(units=metadata, coordinates_cm=coordinates)


def prepare():
    if OUTPUT.exists():
        raise ValueError("AAC training evidence already retained")
    verify_preflight()
    if pilot.audit.uncommitted():
        raise ValueError("commit the training protocol and code before fitting")
    source = pilot.audit.DATA / "sources/tiers/case_1/final_data.csv"
    frozen = read(calibration.OUTPUT / "calibration.json")
    pilot.evidence.official_scorer.require_hash(source, frozen["source_sha256"], "Case 1 training")
    if pilot.READER:
        # ADE-BP's AAC settings as the Bern run record holds them, from the extract of settings/data
        settings = pilot.SETTINGS / "tiers-settings.json"
        baseline = read(settings)["aac_ade_bp"]
    else:
        roster = read(pilot.evidence.retained_data.EVIDENCE)
        row = next(m for m in roster["methods"] if m["id"] == "ade-bp-2015")
        settings = pilot.evidence.retained_data.run_directory("ade-bp-2015", "AAC", row["results"]["AAC"]) / "benchmark.json"
        baseline = read(settings)
    write(OUTPUT / "freeze.json", dict(source_revision=pilot.audit.revision() if pilot.READER else
                                       subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        preflight_revision=PREFLIGHT_REVISION, preflight_sha256=digest(aac_preflight.OUTPUT),
        training_sha256=digest(source), previous_calibration_sha256=digest(calibration.OUTPUT / "calibration.json"),
        baseline_settings_sha256=digest(settings),
        particle_counts=list(PARTICLES), anchor_counts=list(range(1, 7)), synthetic_seed=20260916,
        graphs_per_anchor_count=5, noiseless_graphs_per_anchor_count=1, evaluation_rows_used=0,
        protocol="Fixed existing likelihood families. Leave out sixteen distances and twelve orientations. "
                 "Particle sensitivity uses 30 synthetic layouts with Case 1 residual resampling, not independent validation. "
                 "No automatic budget or method selection. Keep all finite outputs, failures and convergence diagnostics."))
    fit, cells = fit_case1(source, frozen)
    write(OUTPUT / "profiles.json", fit)
    units, reference = synthetic(cells, fit["fitted"])
    method = copy.deepcopy(baseline["method_config"])
    method.update(area=None, ranging_model=fit["fitted"]["ade_ranging_model"])
    write(OUTPUT / "request.json", dict(data_role="synthetic-training", method_config=method,
        solve_config=baseline["config"], units=units))
    write(OUTPUT / "reference.json", reference)
    write(OUTPUT / "inputs.json", {n: digest(OUTPUT / n) for n in ("freeze.json", "profiles.json", "request.json", "reference.json")})
    pilot.register_files(list(OUTPUT.glob("*.json")))
    print("Fitted two model families in 28 held-condition folds. Prepared 60 synthetic inputs, no Case 3 positioning.", flush=True)


def execute(output=OUTPUT, units=60, solves=180):
    for name, expected in read(output / "inputs.json").items():
        pilot.evidence.official_scorer.require_hash(output / name, expected, "AAC training input")
    if (output / "predictions.jsonl").exists():
        raise ValueError("particle study already started, do not overwrite predictions")
    revision = read(output / "freeze.json")["source_revision"]
    if not pilot.READER and (pilot.audit.uncommitted()
            or pilot.audit.revision() != revision):
        raise ValueError("execute from the clean frozen training revision")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLID_")}
    subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", "research/Cargo.toml",
                    "-p", "solid-cli", "--bin", "aac-training"], cwd=pilot.ROOT, env=env, check=True)
    if subprocess.check_output([str(BINARY), "--identity"], text=True).strip() != pilot.isolation.source_identity():
        raise ValueError("particle executable differs from frozen source")
    write(output / "execution.json", dict(source_revision=revision, binary_sha256=digest(BINARY),
                                          inputs_sha256=digest(output / "inputs.json"), units=units, solves=solves))
    subprocess.run([str(BINARY), str(output / "request.json"), str(output / "predictions.jsonl")],
                   cwd=pilot.ROOT, env=env, check=True)


def control_inputs(request, reference):
    """Relabel the same noiseless geometry to vary only the method's deterministic random stream."""
    units, metadata, coordinates = [], [], {}
    wanted = {u["id"]: u for u in reference["units"] if u["noiseless"] and u["anchors"] <= 3}
    for source in request["units"]:
        if source["samples"] != 1 or source["id"] not in wanted:
            continue
        for stream in range(5):
            unit = copy.deepcopy(source)
            unit["id"] = len(units) + 1
            problem, offset = unit["problem"], 100 * stream
            problem["agent_ids"] = [n + offset for n in problem["agent_ids"]]
            for anchor in problem["anchors"]:
                anchor["id"] += offset
            for observation in problem["observations"]:
                observation["from"] += offset
                observation["to"] += offset
            coordinates[str(unit["id"])] = {str(int(n)+offset): p for n, p in
                reference["coordinates_cm"][str(source["id"])].items()}
            metadata.append(dict(id=unit["id"], reference=str(unit["id"]), anchors=len(problem["anchors"]),
                acquisition=f"anchors-{len(problem['anchors'])}", source_id=source["id"], stream=stream))
            units.append(unit)
    if len(units) != 15:
        raise ValueError("expected five streams for each of three noiseless layouts")
    return dict(data_role="synthetic-budget-control", method_config=request["method_config"],
                solve_config=request["solve_config"], units=units), dict(units=metadata, coordinates_cm=coordinates)


def rotation_control(unit, coordinates):
    """Four equally plausible joint layouts have identical distances but coincident marginal means."""
    anchor = unit["problem"]["anchors"][0]["position"]
    center = np.array([anchor["x"], anchor["y"]])
    points = np.array([coordinates[str(n)] for n in unit["problem"]["agent_ids"]])
    layouts = [points]
    for _ in range(3):
        delta = layouts[-1] - center
        layouts.append(center + np.column_stack((-delta[:, 1], delta[:, 0])))
    d = lambda x: np.linalg.norm(x[:, None] - x[None, :], axis=2)
    area = unit["area"]
    all_points = np.concatenate(layouts)
    if any(np.any(all_points[:, i] < area[f"minimum_{axis}_cm"]) or
           np.any(all_points[:, i] > area[f"maximum_{axis}_cm"]) for i, axis in enumerate(("x", "y"))):
        raise ValueError("rotation control left the declared area")
    return dict(rotations=4, in_area=True,
                maximum_pair_distance_change_cm=float(max(np.max(np.abs(d(x)-d(points))) for x in layouts)),
                maximum_anchor_distance_change_cm=float(max(np.max(np.abs(np.linalg.norm(x-center, axis=1)
                    - np.linalg.norm(points-center, axis=1))) for x in layouts)),
                mean_layout_diameter_cm=float(np.max(d(np.mean(layouts, axis=0)))),
                original_layout_diameter_cm=float(np.max(d(points))))


def run_control(output, request, reference, budgets, protocol):
    if output.exists():
        raise ValueError("budget-control evidence already exists")
    if pilot.audit.uncommitted():
        raise ValueError("commit the control protocol before execution")
    for name, expected in read(OUTPUT / "manifest.json")["files"].items():
        pilot.evidence.official_scorer.require_hash(OUTPUT / name, expected, "retained training study")
    write(output / "freeze.json", dict(source_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=pilot.ROOT, text=True).strip(), budgets=list(budgets),
        previous_training_sha256=digest(OUTPUT / "manifest.json"), evaluation_rows_used=0,
        protocol=protocol))
    write(output / "request.json", request)
    write(output / "reference.json", reference)
    write(output / "inputs.json", {n: digest(output / n) for n in ("freeze.json", "request.json", "reference.json")})
    execute(output, units=len(request["units"]), solves=len(request["units"])*len(budgets))


def controls():
    request, reference = control_inputs(read(OUTPUT / "request.json"), read(OUTPUT / "reference.json"))
    run_control(CONTROLS, request, reference, BUDGETS,
                "Same three noiseless training layouts. Five ID relabellings change only the RNG. "
                "Particle, round and refinement checks. No seed or budget selected by evaluation outcomes.")
    report_controls()


def report_controls():
    for name, expected in read(CONTROLS / "inputs.json").items():
        pilot.evidence.official_scorer.require_hash(CONTROLS / name, expected, "budget control")
    request, reference = read(CONTROLS / "request.json"), read(CONTROLS / "reference.json")
    records = [json.loads(line) for line in (CONTROLS / "predictions.jsonl").read_text().splitlines()]
    scores = pilot.score_records(request, reference, scoring_records(records), pilot.evidence.load_scorer(), methods=list(BUDGETS))
    metadata = {u["id"]: u for u in reference["units"]}
    numerical = {(r["id"], r["method"]): r for r in records}
    rows = []
    for budget in BUDGETS:
        for k in (1, 2, 3):
            selected = [s for s in scores if s["method"] == budget and metadata[s["id"]]["anchors"] == k]
            errors = [s["mean_pair_error_cm"] for s in selected]
            details = [numerical[s["id"], budget] for s in selected]
            rows.append(dict(budget=budget, anchors=k, streams=len(errors), mean_cm=float(np.mean(errors)),
                minimum_cm=min(errors), maximum_cm=max(errors), failed_agents=sum(s["failed_agents"] for s in selected),
                converged=sum(bool((s["estimate"] or {}).get("metrics", {}).get("converged")) for s in details),
                mean_ms=float(np.mean([s["elapsed_ms"] for s in selected]))))
    ring = rotation_control(request["units"][0], reference["coordinates_cm"]["1"])
    write(CONTROLS / "scores.json", scores)
    write(CONTROLS / "summary.json", dict(case3_solves=0, solves=len(records), results=rows,
        rotation_control=ring, caveat="Five random streams on each of three noiseless layouts, not independent datasets."))
    seal_control(CONTROLS)


def seal_control(output):
    write(output / "manifest.json", dict(execution_revision=read(output / "execution.json")["source_revision"],
        report_source_sha256=digest(Path(__file__)), scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        files={p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}))
    pilot.register_files(list(output.iterdir()))


def rounds():
    request = read(OUTPUT / "request.json")
    request["data_role"] = "synthetic-round-control"
    run_control(ROUND_CHECK, request, read(OUTPUT / "reference.json"), ROUND_BUDGETS,
                "Four versus ten message rounds at 500 particles and cap 256. Same thirty training layouts "
                "and both sample tracks. No refitting, new geometry or evaluation outcome. "
                "A single target budget must serve all anchor counts and sample tracks.")
    report_rounds()


def report_rounds():
    for name, expected in read(ROUND_CHECK / "inputs.json").items():
        pilot.evidence.official_scorer.require_hash(ROUND_CHECK / name, expected, "round control")
    request, reference = read(ROUND_CHECK / "request.json"), read(ROUND_CHECK / "reference.json")
    records = [json.loads(line) for line in (ROUND_CHECK / "predictions.jsonl").read_text().splitlines()]
    scorer = pilot.evidence.load_scorer()
    scores = pilot.score_records(request, reference, scoring_records(records), scorer, methods=list(ROUND_BUDGETS))
    original = {(r["id"], r["samples"]): r for r in read(OUTPUT / "scores.json") if r["method"] == "500"}
    prior_predictions = {(r["id"], r["samples"]): r["estimate"] for r in
        map(json.loads, (OUTPUT / "predictions.jsonl").read_text().splitlines()) if r["method"] == "500"}
    for row in records:
        if row["method"] == ROUND_BUDGETS[0] and row["estimate"]["positions"] != prior_predictions[row["id"], row["samples"]]["positions"]:
            raise ValueError("unchanged four-round predictions did not reproduce exactly")
    for row in scores:
        if row["method"] == ROUND_BUDGETS[0] and row["pair_errors_cm"] != original[row["id"], row["samples"]]["pair_errors_cm"]:
            raise ValueError("unchanged four-round scores did not reproduce exactly")
    metadata = {u["id"]: u for u in reference["units"]}
    rows = []
    for budget in ROUND_BUDGETS:
        for samples in (1, 10):
            chosen = [s for s in scores if s["method"] == budget and s["samples"] == samples]
            before = [original[s["id"], samples]["mean_pair_error_cm"] for s in chosen]
            errors = [s["mean_pair_error_cm"] for s in chosen]
            rows.append(dict(budget=budget, samples=samples, units=len(chosen), mean_cm=float(np.mean(errors)),
                s75_cm=scorer.percentile(sorted(errors), .75), s90_cm=scorer.percentile(sorted(errors), .90),
                improved=sum(a < b-1e-9 for a, b in zip(errors, before)),
                worsened=sum(a > b+1e-9 for a, b in zip(errors, before)),
                maximum_increase_cm=max(a-b for a, b in zip(errors, before)),
                maximum_reduction_cm=max(b-a for a, b in zip(errors, before)),
                failed_agents=sum(s["failed_agents"] for s in chosen), mean_ms=float(np.mean([s["elapsed_ms"] for s in chosen])),
                converged=sum(bool((r["estimate"] or {}).get("metrics", {}).get("converged")) for r in records
                              if r["method"] == budget and r["samples"] == samples),
                by_anchor_count={str(k): float(np.mean([s["mean_pair_error_cm"] for s in chosen
                    if metadata[s["id"]]["anchors"] == k])) for k in range(1, 7)}))
    write(ROUND_CHECK / "scores.json", scores)
    write(ROUND_CHECK / "summary.json", dict(case3_solves=0, solves=len(records), results=rows,
        previous_four_round_scores_reproduced=True, previous_four_round_predictions_reproduced=True,
        caveat="Synthetic Case 1 residual-resampling study, not a real-data performance comparison."))
    seal_control(ROUND_CHECK)


def report():
    for name, expected in read(OUTPUT / "inputs.json").items():
        pilot.evidence.official_scorer.require_hash(OUTPUT / name, expected, "AAC training input")
    records = [json.loads(line) for line in (OUTPUT / "predictions.jsonl").read_text().splitlines()]
    reference, request = read(OUTPUT / "reference.json"), read(OUTPUT / "request.json")
    scorer = pilot.evidence.load_scorer()
    scores = pilot.score_records(request, reference, scoring_records(records), scorer, methods=list(map(str, PARTICLES)))
    metadata = {u["id"]: u for u in reference["units"]}
    summaries = []
    for particles in PARTICLES:
        for samples in (1, 10):
            rows = [r for r in scores if r["method"] == str(particles) and r["samples"] == samples]
            numerical = [r for r in records if r["method"] == str(particles) and r["samples"] == samples]
            summaries.append(dict(particles=particles, samples=samples, units=len(rows),
                s75_cm=scorer.percentile(sorted(r["mean_pair_error_cm"] for r in rows), .75),
                s90_cm=scorer.percentile(sorted(r["mean_pair_error_cm"] for r in rows), .90),
                failed_agents=sum(r["failed_agents"] for r in rows),
                converged=sum(bool((r.get("estimate") or {}).get("metrics", {}).get("converged")) for r in numerical),
                mean_ms=float(np.mean([r["elapsed_ms"] for r in rows])),
                by_anchor_count={str(k): dict(mean_error_cm=float(np.mean([r["mean_pair_error_cm"] for r in rows
                                    if metadata[r["id"]]["anchors"] == k]))) for k in range(1, 7)}))
    write(OUTPUT / "scores.json", scores)
    write(OUTPUT / "summary.json", dict(status="training-checks-complete", case3_solves=0, model_families_fitted=2,
        condition_folds=28, particle_results=summaries, selected_particle_count=None,
        caveat="Numerical sensitivity on synthetic training graphs, not external performance or proof of particle convergence."))
    write(OUTPUT / "manifest.json", dict(fitting_revision=read(OUTPUT / "freeze.json")["source_revision"],
        prediction_revision=read(OUTPUT / "execution.json")["source_revision"],
        report_source_sha256=digest(Path(__file__)), scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        files={p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != "manifest.json"}))
    pilot.register_files(list(OUTPUT.iterdir()))


def scoring_records(records):
    converted = copy.deepcopy(records)
    for row in converted:
        if row["estimate"] is not None:
            row["estimate"]["positions"] = [[p["x"], p["y"]] for p in row["estimate"]["positions"]]
    return converted


def section():
    text = '<details id="external-aac-preflight" class="report-document"><summary>TIERS AAC preparation and training checks</summary>'
    text += '<p>This section records training-only preparation completed before the real-input AAC check. '
    text += 'The eighteen checked input demonstrations and the proposed all-subset population are retained. Real-input results appear separately above.</p>'
    if (OUTPUT / "summary.json").exists():
        result = read(OUTPUT / "summary.json")
        fit = read(OUTPUT / "profiles.json")
        shape = fit["fitted"]["range_error"]
        text += '<p>Completed 28 held-condition profile folds and 180 synthetic ADE-BP solves. These are training diagnostics, not dataset benchmark scores. No particle budget has been selected automatically.</p>'
        text += f'<p>The fits use {fit["eligible_rows"]:,} Case 1 observations. AAC keeps its distance-dependent scale and now has target-fitted positive and negative multipliers of {shape["positive_scale"]:.6f} and {shape["negative_scale"]:.6f}. The same shape is supplied for both sample counts. These fits do not by themselves establish positioning accuracy.</p>'
        text += '<table><tr><th>Particles</th><th>Samples</th><th>Synthetic S75 / S90 cm</th><th>Converged solves</th><th>Failed agents</th><th>Mean ms</th></tr>'
        for r in result["particle_results"]:
            text += f'<tr><td>{r["particles"]:,}</td><td>{r["samples"]}</td><td>{r["s75_cm"]:.3f} / {r["s90_cm"]:.3f}</td><td>{r["converged"]}/{r["units"]}</td><td>{r["failed_agents"]}</td><td>{r["mean_ms"]:.3f}</td></tr>'
        text += '</table><p>All anchor counts remain included. Repeated residual draws are simulated noise, not additional real packets. More particles do not consistently reduce errors. A finite returned position does not mean the importance-refinement convergence test passed.</p>'
        text += '<details><summary>Per-anchor-count mean pair error</summary><p>Each cell shows one / ten simulated samples in cm. Five layouts contribute to each value, including one noiseless control. This small diagnostic cannot establish a generally optimal budget.</p><table><tr><th>Anchors</th><th>500 particles</th><th>2,000 particles</th><th>5,000 particles</th></tr>'
        by_key = {(r["particles"], r["samples"]): r for r in result["particle_results"]}
        for k in range(1, 7):
            text += f'<tr><td>{k}</td>'
            for n in PARTICLES:
                values = [by_key[n, samples]["by_anchor_count"][str(k)]["mean_error_cm"] for samples in (1, 10)]
                text += f'<td>{values[0]:.3f} / {values[1]:.3f}</td>'
            text += '</tr>'
        text += '</table><p>One anchor does not determine the network orientation from ranges alone. Marginal position averaging and incomplete iterative convergence are possible explanations for the sparse-anchor sensitivity, not yet isolated causes. Do not remove those cases or assume that a larger particle count fixes them.</p></details>'
        for name, label in (("profiles.json", "Case 1 fits and held-condition scores"), ("summary.json", "Particle sensitivity by anchor count"), ("predictions.jsonl", "Exact synthetic predictions")):
            text += f'<p><a href="{html.escape((OUTPUT / name).as_uri(), quote=True)}">{label}</a></p>'
        if (CONTROLS / "summary.json").exists():
            result = read(CONTROLS / "summary.json")
            text += '<h4>Sparse-anchor numerical controls</h4><p>These 90 additional solves use three noiseless layouts with one, two and three anchors. Five fixed node-ID relabellings per layout vary the deterministic random stream without changing ranges, geometry or priors. They are not five independent geometries.</p>'
            text += '<table><tr><th>Particles / rounds / refinement cap</th><th>Anchors</th><th>Mean pair error cm</th><th>Stream range cm</th><th>Converged</th><th>Mean ms</th></tr>'
            for r in result["results"]:
                text += f'<tr><td>{r["budget"].replace("-", " / ")}</td><td>{r["anchors"]}</td><td>{r["mean_cm"]:.3f}</td><td>{r["minimum_cm"]:.3f} to {r["maximum_cm"]:.3f}</td><td>{r["converged"]}/{r["streams"]}</td><td>{r["mean_ms"]:.3f}</td></tr>'
            text += '</table><p>A separate exact rotation control preserves every range in four joint layouts within the declared prior. Averaging those equally plausible layouts places all marginal means at the single anchor. Their pairwise distances collapse even though each original layout has the same correct relative geometry.</p>'
            text += '<p>This separates an ambiguity of marginal position means from a lack of ranging information about relative shape. It does not prove that every finite ADE-BP output equals the exact posterior mean. The paper uses Gaussian message approximations and iterative importance sampling.</p>'
            text += f'<p><a href="{html.escape((CONTROLS / "summary.json").as_uri(), quote=True)}">Exact numerical controls</a></p>'
        if (ROUND_CHECK / "summary.json").exists():
            result = read(ROUND_CHECK / "summary.json")
            text += '<h4>Message rounds on all synthetic training layouts</h4><p>Four and ten message rounds use the same 500 particles, 256-refinement cap, thirty layouts and paired sample tracks. Twenty-four layouts contain Case 1 residual draws and six are noiseless. The original four-round pair scores reproduce exactly.</p>'
            text += '<table><tr><th>Rounds</th><th>Samples</th><th>Mean pair error cm</th><th>S75 / S90 cm</th><th>Improved / worsened units</th><th>Converged</th><th>Mean ms</th></tr>'
            for r in result["results"]:
                text += f'<tr><td>{r["budget"].split("-")[1]}</td><td>{r["samples"]}</td><td>{r["mean_cm"]:.3f}</td><td>{r["s75_cm"]:.3f} / {r["s90_cm"]:.3f}</td><td>{r["improved"]} / {r["worsened"]}</td><td>{r["converged"]}/{r["units"]}</td><td>{r["mean_ms"]:.3f}</td></tr>'
            text += '</table><p>The training-supported budget for the small TIERS AAC check is 500 particles, ten message rounds and the existing 256-refinement cap. The same configuration applies to all anchor counts and both sample tracks. Bern retains its recorded four-round configuration and scores. These settings were selected before any real TIERS AAC result was calculated.</p>'
            losses = [r["maximum_increase_cm"] for r in result["results"] if r["budget"] == ROUND_BUDGETS[1]]
            text += f'<p>Changes are relative to four rounds on each identical input. Twenty units improve and ten worsen in each sample track. The largest increases in scenario mean error are {losses[0]:.3f} cm and {losses[1]:.3f} cm respectively. All 120 outputs are finite, but some inner updates remain iteration-limited. These are synthetic training results, not a TIERS Case 3 comparison or proof of a universally best numerical budget.</p>'
            text += f'<p><a href="{html.escape((ROUND_CHECK / "summary.json").as_uri(), quote=True)}">Round comparison and anchor-count breakdown</a></p>'
    else:
        text += '<p>Target profile fitting and particle sensitivity checks are in preparation. The full comparison remains disabled.</p>'
    return text + '<p><a href="#external-protocol-tiers-aac-comparison">Population, profiles and scoring protocol</a></p></details>'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "report", "controls", "report-controls", "rounds", "report-rounds"))
    {"prepare": prepare, "run": execute, "report": report,
     "controls": controls, "report-controls": report_controls,
     "rounds": rounds, "report-rounds": report_rounds}[parser.parse_args().action]()
