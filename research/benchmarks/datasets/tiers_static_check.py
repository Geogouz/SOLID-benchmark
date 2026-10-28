"""Bounded, outcome-independent TIERS checks before a full static campaign.

Usage: tiers_static_check.py [prepare|execute]
  prepare writes the check inputs and the ANC profiles without running a method, the stage a reader's reproduction
  (SOLID_READER=1) runs before tiers_campaign.py freeze; execute, the default, also runs and scores the checks.
"""

import argparse
import copy
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

import aac_check
import aac_preflight
import aac_training
import baseline_selection
import calibration
import pilot
import static_preflight

OUTPUT = pilot.audit.DATA / "tiers-static-check"
TRACKS = ("raw", "calibrated")
# The shared Bern calibration's uncertainty law, applied to published metric ranges of the raw track.
SCALE_SLOPE, SCALE_INTERCEPT_CM = 0.02635621175286423, 4.649664346296255
MODES = ("AF", "AAC", "ANC")
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def fit_anc(cells, correction):
    """Constant asymmetric-Laplace widths with equal weight per Case 1 condition."""
    measured, reference, weights = calibration.pooled(cells)
    residual = calibration.corrected(measured, correction) - reference
    roots = np.sqrt([weights @ np.maximum(residual, 0), weights @ np.maximum(-residual, 0)])
    widths = np.maximum(roots * roots.sum(), 1e-6)
    if not np.isfinite(widths).all():
        raise ValueError("invalid training residuals")
    return dict(positive_slope=0., positive_intercept_cm=float(widths[0]),
                negative_slope=0., negative_intercept_cm=float(widths[1]))


def first_block(ranges, sid):
    streams = static_preflight.tiers_streams(ranges)
    nodes = sorted({n for pair in streams for n in pair})
    if nodes != list(range(1, 9)):
        raise ValueError("expected eight TIERS nodes")
    selected = next(static_preflight.tiers_blocks(streams, 10))
    typical = float(np.median([100 * rows[0, 1] for rows in selected.values()]))
    initial = np.random.default_rng(20260916 + sid).uniform(-typical, typical, (8, 2))
    units = []
    for samples in (1, 10):
        observations = []
        for (a, b), rows in selected.items():
            cm = rows[:samples, 1] * 100
            observations.append(dict(**{"from": a, "to": b}, to_is_anchor=False, tof_ns=[],
                calibrated_samples=[dict(distance_cm=float(v), scale_cm=float(np.clip(SCALE_SLOPE*v+SCALE_INTERCEPT_CM, 8, 55))) for v in cm]))
        units.append(dict(id=sid, samples=samples, problem=dict(agent_ids=nodes, anchors=[],
            mode="AnchorFree", observations=observations,
            initial_positions=[dict(x=float(x), y=float(y)) for x, y in initial])))
    return units, {f"{a}-{b}": rows.tolist() for (a, b), rows in selected.items()}


def anchored_unit(unit, anchors, coordinates, mode, anc_profile):
    if mode not in ("AAC", "ANC"):
        raise ValueError("expected AAC or ANC")
    problem, links = aac_preflight.construct(unit, anchors, coordinates)
    if mode == "ANC":
        kept = [(o, link) for o, link in zip(problem["observations"], links, strict=True) if o["to_is_anchor"]]
        problem["observations"], links = map(list, zip(*kept))
        problem["mode"] = "AnchorOnly"
    area = aac_preflight.operating_area(problem["anchors"])
    # AF's start scale also includes ranges excluded after anchor roles are assigned.
    low = [area[f"minimum_{axis}_cm"] for axis in ("x", "y")]
    high = [area[f"maximum_{axis}_cm"] for axis in ("x", "y")]
    initial = np.random.default_rng(20260916 + unit["id"]).uniform(low, high, (len(problem["agent_ids"]), 2))
    problem["initial_positions"] = [dict(x=float(x), y=float(y)) for x, y in initial]
    result = dict(id=unit["id"] * 10 + len(anchors), samples=unit["samples"], problem=problem, area=area)
    if mode == "ANC":
        result["deployment_profile"] = dict(copy.deepcopy(anc_profile), area=copy.deepcopy(area))
    return result, links


def metric_input(unit, fitted):
    result = copy.deepcopy(unit)
    for observation in result["problem"]["observations"]:
        values = observation.pop("ranges_cm", None)
        if values is None:
            values = [s["distance_cm"] for s in observation["calibrated_samples"]]
        distances = calibration.corrected(values, fitted["correction"]) if fitted else values
        observation.update(tof_ns=[], calibrated_samples=[dict(distance_cm=float(v),
            scale_cm=fitted["scale_cm"] if fitted else float(np.clip(SCALE_SLOPE*v+SCALE_INTERCEPT_CM, 8, 55))) for v in distances])
    return result


def anchored_baselines(mode, fitted):
    """The anchored baselines of a mode from the declared base configurations with the options TWR-CLOUD-Bern chose
    (baseline_selection.declared), ADE-BP with the given ranging model or else Bern's, and the Bern evaluation run records
    those values were once read from, {path: SHA-256}, in the order of the baselines. The records are provenance only,
    and no setting is read from an evaluation run."""
    baselines = [baseline_selection.declared(method, mode, bern=True) for method in aac_check.LABELS]
    for entry in baselines:
        if entry["method"] == "ade-bp-2015" and fitted:
            entry["method_config"]["ranging_model"] = copy.deepcopy(fitted["ade_ranging_model"])
    data = pilot.evidence.retained_data.DATA
    provenance = baseline_selection.provenance(mode)
    return baselines, {str(data / name): provenance[name] for name in provenance}


def prepare():
    if OUTPUT.exists():
        raise ValueError("bounded evidence already exists, do not overwrite it")
    preflight = read(static_preflight.OUTPUT)
    for name, expected in preflight["sources"].items():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "input inventory")
    # The seals of the synthetic AAC training runs, which only the research checks of ADE-BP's budgets ran; a reader
    # needs only the profiles that aac_training.prepare fits
    for directory in () if pilot.READER else (aac_training.OUTPUT, aac_training.ROUND_CHECK):
        for name, expected in read(directory / "manifest.json")["files"].items():
            pilot.evidence.official_scorer.require_hash(directory / name, expected, "Case 1 profile")
    source = pilot.audit.DATA / "sources/tiers/case_1/final_data.csv"
    frozen = read(calibration.OUTPUT / "calibration.json")
    pilot.evidence.official_scorer.require_hash(source, frozen["source_sha256"], "Case 1 training")
    cells, excluded = calibration.load_case1(source)
    fitted = read(aac_training.OUTPUT / "profiles.json")["fitted"]
    if fitted["correction"] != frozen["correction"]:
        raise ValueError("frozen calibration differs")
    if pilot.READER:
        # The transferred Bern ANC profile as the runtime deployment profile of the retained evidence holds it
        anc_path = pilot.SETTINGS / "tiers-settings.json"
        transferred = read(anc_path)["anc"]
    else:
        roster = read(pilot.evidence.retained_data.EVIDENCE)
        anc_path = pilot.evidence.retained_data.resolve(roster["runtime_deployment_profile"])
        transferred = read(anc_path)
    profiles = {track: dict(ranging_model=copy.deepcopy(transferred["ranging_model"]),
                           detection_range_cm=transferred["detection_range_cm"]) for track in TRACKS}
    profiles["calibrated"]["ranging_model"] = fit_anc(cells, fitted["correction"])
    folds = []
    for family, held, train, validation in calibration.splits(cells):
        correction = calibration.fit_correction(train, frozen["selected"])
        model = fit_anc(train, correction)
        measured, reference, weights = calibration.pooled(validation)
        residual = calibration.corrected(measured, correction) - reference
        positive, negative = model["positive_intercept_cm"], model["negative_intercept_cm"]
        loss = np.log(positive + negative) + np.abs(residual) / np.where(residual >= 0, positive, negative)
        folds.append(dict(family=family, held=list(held), ranging_model=model,
                          validation_nll=float(weights @ loss), validation_mae_cm=float(weights @ np.abs(residual))))
    write(OUTPUT / "profiles.json", dict(anc=profiles, eligible_training_rows=sum(map(len, cells.values())),
        training_sha256=digest(source), excluded_without_reference_lines=excluded, evaluation_rows_used=0,
        correction=fitted["correction"], held_condition_folds=folds,
        weighting="Equal weight per Case 1 distance and orientation condition",
        range_model="Existing constant asymmetric Laplace family, same parameters for one and ten samples",
        availability="Bern detection range transferred unchanged in both tracks. Case 1 cannot identify it. It also weights observed links.",
        area="Known anchors and the published 8 m by 9 m arena only",
        raw="Bern range-error and availability parameters, uncorrected TIERS metric ranges",
        calibrated="Case 1 calibration and range-error widths, Bern detection range"))
    raw_profile, af_baselines, sources = pilot.retained_settings()
    cal_request = calibration.OUTPUT / "request.json"
    pilot.evidence.official_scorer.require_hash(cal_request,
        read(calibration.OUTPUT / "inputs.json")["request_sha256"], "calibrated AF profile")
    cal_profile = read(cal_request)["profile"]
    paths = pilot.present([source, static_preflight.OUTPUT, anc_path, cal_request, calibration.OUTPUT / "calibration.json",
             aac_training.OUTPUT / "profiles.json", aac_training.ROUND_CHECK / "manifest.json",
             Path(__file__), pilot.HERE / "predict.rs", pilot.HERE / "README.md"])
    batches = {}
    for track in TRACKS:
        for mode in MODES:
            baselines = copy.deepcopy(af_baselines)
            if mode != "AF":
                baselines, settings = anchored_baselines(mode, fitted if track == "calibrated" else None)
                sources.update(settings)
            profile = copy.deepcopy(raw_profile if track == "raw" else cal_profile)
            if mode == "AAC" and track == "calibrated":
                profile = dict(free={}, anchored={str(n): {"2": copy.deepcopy(fitted["range_error"])} for n in (1, 10)})
            batches[track, mode] = (dict(profile=None if mode == "ANC" else profile, baselines=baselines, units=[]),
                                   dict(units=[], coordinates_cm={}))
    messages = {}
    for sid, record in enumerate(read(pilot.audit.DATA / "input-audit.json")["tiers"], 1):
        path = pilot.audit.DATA / record["file"]
        pilot.evidence.official_scorer.require_hash(path, record["sha256"], "TIERS recording")
        sources[str(path)] = digest(path)
        acquisition = path.name.split("_")[0]
        ranges, _ = pilot.audit.tiers_events(path)
        units, messages[acquisition] = first_block(ranges, sid)
        poses = {int(n): [100*v for v in p] for n, p in record["pose_median_native"].items()}
        for track in TRACKS:
            for mode in MODES:
                request, reference = batches[track, mode]
                for unit in units:
                    for k in (range(1, 7) if mode != "AF" else (0,)):
                        item = unit
                        if k:
                            known = {n: poses[n] for n in range(1, k+1)}
                            item, _ = anchored_unit(unit, list(known), known, mode, profiles[track])
                        item = metric_input(item, fitted if track == "calibrated" else None)
                        request["units"].append(item)
                        key = str(item["id"])
                        if key not in reference["coordinates_cm"]:
                            reference["coordinates_cm"][key] = {str(n): poses[n][:2] for n in item["problem"]["agent_ids"]}
                            reference["units"].append(dict(id=item["id"], acquisition=acquisition, reference=key, anchors=k))
    for (track, mode), (request, reference) in batches.items():
        write(OUTPUT / f"{track}-{mode}/request.json", request)
        write(OUTPUT / f"{track}-{mode}/reference.json", reference)
    write(OUTPUT / "messages.json", messages)
    sources.update({str(p): digest(p) for p in paths})
    write(OUTPUT / "inputs.json", dict(sources=sources,
        files={str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.rglob("*.json")},
        population="First complete common-rank block per acquisition. First one through six node IDs become anchors. Both sample counts and both calibration tracks.",
        units_per_track=dict(AF=6, AAC=36, ANC=36), full_campaign=False,
        selection="Adapter check only, not model selection. No settings may be chosen from these outcomes.",
        scope="Native baselines with their declared configurations."))


def execute():
    if OUTPUT.exists():
        raise ValueError("bounded evidence already exists, do not overwrite it")
    revision, env = pilot.build_predictor()
    prepare()
    inputs = read(OUTPUT / "inputs.json")
    for name, expected in inputs["files"].items():
        pilot.evidence.official_scorer.require_hash(OUTPUT / name, expected, "frozen check input")
    write(OUTPUT / "freeze.json", dict(source_revision=revision, binary_sha256=digest(pilot.BINARY),
        inputs_sha256=digest(OUTPUT / "inputs.json"), scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        host=platform.platform(), workers=1, full_campaign=False))
    scorer, summaries = pilot.evidence.load_scorer(), []
    for track in TRACKS:
        for mode in MODES:
            folder = OUTPUT / f"{track}-{mode}"
            started = time.perf_counter()
            with (folder / "run.log").open("x") as log:
                subprocess.run([str(pilot.BINARY), str(folder / "request.json"), str(folder / "optimizer.jsonl")],
                               cwd=pilot.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            request, reference = read(folder / "request.json"), read(folder / "reference.json")
            records = [json.loads(line) for line in (folder / "optimizer.jsonl").read_text().splitlines()]
            labels = pilot.LABELS if mode == "AF" else aac_check.LABELS
            write(folder / "predictions.json", records)
            scores = pilot.score_records(request, reference, records, scorer, methods=labels)
            metadata = {u["id"]: u for u in reference["units"]}
            for row in scores:
                row["anchors"] = metadata[row["id"]]["anchors"]
            write(folder / "scores.json", scores)
            # Each acquisition has one input at every anchor count, so equal-cell
            # weighting and the acquisition-balanced helper give identical weights.
            result = pilot.summarize(scores, scorer, labels=labels, mode=mode)
            write(folder / "summary.json", result)
            summaries.append(dict(track=track, mode=mode, inputs=len(request["units"]), outputs=len(records),
                failed_agents=sum(r["failed_agents"] for r in scores), expected_agents=sum(r["expected_agents"] for r in scores),
                seconds=time.perf_counter()-started,
                collapsed_outputs=sum(r["estimate"] is not None and len({tuple(p) for p in r["estimate"]["positions"]}) == 1
                                      for r in records)))
            print(json.dumps(summaries[-1]), flush=True)
    write(OUTPUT / "summary.json", dict(status="bounded-check-complete", batches=summaries,
        expected_inputs=156, expected_outputs=768, full_campaign_started=False,
        note="Finite collapsed layouts are retained, not marked as missing estimates. No setting was selected from these checks."))
    files = sorted(p for p in OUTPUT.rglob("*") if p.is_file())
    write(OUTPUT / "manifest.json", dict(files={str(p.relative_to(OUTPUT)): digest(p) for p in files}))
    pilot.register_files([*files, OUTPUT / "manifest.json"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("prepare", "execute"), default="execute")
    {"prepare": prepare, "execute": execute}[parser.parse_args().command]()
