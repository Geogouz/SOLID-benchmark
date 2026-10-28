"""Select STRONG's native threshold on reserved IDLab radio configurations."""

from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

import aac_check
import calibration
import idlab_admission
import idlab_check
import idlab_population
import idlab_split
import idlab_training
import pilot
import static_preflight
import strong_threshold


OUTPUT = idlab_check.OUTPUT / "settings"
RADII = strong_threshold.RADII
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def fold_inputs(cells, held, correction, coordinates, index):
    training = {k: v for k, v in cells.items() if k[2:] != held}
    validation = {k: v for k, v in cells.items() if k[2:] == held}
    if not training or not validation or held[-1] != held[-2]:
        raise ValueError("invalid held radio configuration")
    measured, truth, weights = calibration.pooled(training)
    scale = float(np.sqrt(weights @ (calibration.corrected(measured, correction) - truth) ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid training noise scale")
    pools, values = defaultdict(list), {}
    settings = dict(zip(static_preflight.SETTINGS, held, strict=True))
    for (a, b, *_), rows in validation.items():
        for row in rows:
            identity = (a, int(row[0]))
            if identity in values:
                raise ValueError("duplicate training source identity")
            values[identity] = float(row[1])
            pools[tuple(sorted((a, b)))].append(list(identity))
    links = [dict(nodes=list(pair), records=sorted(rows, key=lambda row:
        idlab_population.digest([idlab_population.SEED, "records", settings, pair, row])))
        for pair, rows in sorted(pools.items())]
    panel = idlab_population.panel(dict(nodes=sorted(coordinates)), dict(settings=settings, links=links, panels=1), 0)
    fitted = dict(correction=correction, profiles=dict(scale_cm=scale))
    units, reference = [], dict(units=[], coordinates_cm={})
    for mode in ("AAC", "ANC"):
        for count in (1, 4, 8, 13):
            sid = index * 32 + count + (16 if mode == "ANC" else 0)
            known = {n: coordinates[n] / 10 for n in panel["anchor_order"][:count]}
            for samples in (1, 10):
                unit = idlab_check.unit(panel, sid, samples, "AAC", known, values, fitted, None)
                unit["area"] = None
                if mode == "ANC":
                    unit["problem"].update(mode="AnchorOnly", observations=[
                        o for o in unit["problem"]["observations"] if o["to_is_anchor"]])
                units.append(unit)
            reference["units"].append(dict(id=sid, acquisition=str(index), reference=str(sid), mode=mode, anchors=count))
            reference["coordinates_cm"][str(sid)] = {
                str(n): (coordinates[n][:2] / 10).tolist() for n in coordinates if n not in known}
    fit = dict(held=list(held), correction=correction, scale_cm=scale,
        training_cells=len(training), validation_cells=len(validation),
        training_reports=sum(map(len, training.values())), validation_reports=sum(map(len, validation.values())))
    return units, reference, panel, fit


def rank(scores, folds, scorer):
    counts = defaultdict(int)
    for row in scores:
        counts[row["acquisition"], row["mode"], row["samples"], row["method"]] += 1
    expected = {(str(i), mode, n, method) for i in range(folds)
                for mode in ("AAC", "ANC") for n in (1, 10) for method in RADII}
    if set(counts) != expected or any(value != 4 for value in counts.values()):
        raise ValueError("incomplete held-configuration comparison")
    ranking = []
    for method in RADII:
        groups = []
        for mode in ("AAC", "ANC"):
            for samples in (1, 10):
                rows = [r for r in scores if (r["method"], r["mode"], r["samples"]) == (method, mode, samples)]
                cells = defaultdict(list)
                for row in rows:
                    cells[row["acquisition"]].append(row["mean_pair_error_cm"])
                groups.append(dict(mode=mode, samples=samples, s75_cm=scorer.balanced_percentile(cells, .75),
                    failed_agents=sum(r["failed_agents"] for r in rows), agent_attempts=sum(r["expected_agents"] for r in rows),
                    converged=sum(r["converged"] for r in rows), solves=len(rows)))
        value = statistics.mean(r["s75_cm"] for r in groups)
        if not np.isfinite(value):
            raise ValueError("nonfinite training selection score")
        ranking.append(dict(method=method, mean_s75_cm=value, groups=groups))
    return sorted(ranking, key=lambda r: (round(r["mean_s75_cm"], 9), list(RADII).index(r["method"])))


def execute():
    if OUTPUT.exists():
        raise ValueError("IDLab settings check already retained")
    revision, env, binary = aac_check.build_adapter_test()
    source_files = [idlab_split.OUTPUT, idlab_training.OUTPUT, idlab_check.OUTPUT / "manifest.json"]
    for name, expected in read(idlab_check.OUTPUT / "manifest.json")["files"].items():
        pilot.evidence.official_scorer.require_hash(idlab_check.OUTPUT / name, expected, "IDLab retained check")
    split, fitted = read(idlab_split.OUTPUT), read(idlab_training.OUTPUT)
    for record in (read(idlab_admission.OUTPUT), split, fitted):
        for name, expected in record["sources"].items():
            pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab training source")
    coordinates = {int(r[0]): r[1:] for r in np.loadtxt(idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")}
    cells = idlab_training.load_training(split, idlab_admission.FOLDER, coordinates)
    folds = [r for r in fitted["folds"] if r["family"] == "setting" and r["model"] == fitted["selected"]]
    if len(folds) != 56 or {tuple(r["held"]) for r in folds} != {k[2:] for k in cells}:
        raise ValueError("expected every reserved training configuration")
    settings = [next(b for b in read(idlab_check.OUTPUT / mode / "request.json")["baselines"] if b["method"] == "strong")
                for mode in ("AAC", "ANC")]
    if settings[0] != settings[1]:
        raise ValueError("STRONG mode settings differ")
    setting = settings[0]
    sources = {pilot.audit.name(p): digest(p) for p in source_files}
    source_files = [Path(m.__file__).resolve() for m in (idlab_check, idlab_population, idlab_training, strong_threshold, calibration, pilot)]
    source_files += [Path(__file__).resolve(), pilot.HERE / "strong_training.rs"]
    write(OUTPUT / "freeze.json", dict(source_revision=revision, binary_sha256=digest(binary),
        sources=sources, source_hashes={pilot.audit.name(p): digest(p) for p in source_files},
        protocol_sha256=pilot.document(pilot.HERE / "README.md"), scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        radii=RADII, folds=56, inputs=896, training_solves=3584, evaluation_solves=0, workers=1,
        selection="Mean of AAC1, AAC10, ANC1 and ANC10 balanced S75. Balance configurations and the four anchor counts equally.",
        ties="Round the mean to nine decimals, then use the declared candidate order.",
        limitations="One identity-selected panel per training configuration and one assignment per anchor count. Same physical layout. Not independent environment validation.",
        full_campaign_started=False))
    scorer, scores = pilot.evidence.load_scorer(), []
    for index, fold in enumerate(folds):
        units, reference, panel, fit = fold_inputs(cells, tuple(fold["held"]), fold["parameters"], coordinates, index)
        if (fit["training_cells"], fit["validation_cells"]) != (fold["training_cells"], fold["validation_cells"]):
            raise ValueError("frozen calibration fold differs")
        folder = OUTPUT / f"fold-{index:02d}"
        request = dict(data_role="idlab-training-strong-threshold", method_config=setting["method_config"],
            variants=strong_threshold.variants(setting["solve_config"], fit["scale_cm"]), units=units)
        write(folder / "request.json", request)
        write(folder / "reference.json", reference)
        write(folder / "inputs.json", dict(fit=fit, records=panel))
        subprocess.run([str(binary), "strong_training::threshold_check", "--exact", "--ignored", "--nocapture"],
            cwd=pilot.ROOT, env=dict(env, STRONG_TRAINING_INPUT=str(folder / "request.json"),
                STRONG_TRAINING_OUTPUT=str(folder / "predictions.jsonl")), check=True, capture_output=True)
        records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
        rows = pilot.score_records(request, reference, records, scorer, methods=RADII)
        metadata = {r["id"]: r for r in reference["units"]}
        estimates = {(r["id"], r["samples"], r["method"]): r["estimate"] for r in records}
        for row in rows:
            result = estimates[row["id"], row["samples"], row["method"]]
            row.update(mode=metadata[row["id"]]["mode"], anchors=metadata[row["id"]]["anchors"],
                       converged=bool(result and result["metrics"]["converged"]))
        write(folder / "scores.json", rows)
        scores.extend(rows)
        print(f"STRONG training configuration {index + 1}/56 complete", flush=True)
    ranking = rank(scores, len(folds), scorer)
    selected = ranking[0]["method"]
    chosen = copy.deepcopy(setting)
    chosen["solve_config"]["robust_delta"] = 40. if RADII[selected] is None else RADII[selected] * fitted["profiles"]["scale_cm"]
    write(OUTPUT / "summary.json", dict(selected=selected, ranking=ranking, selected_setting=chosen,
        training_solves=len(scores), evaluation_solves=0, full_campaign_started=False,
        ade=dict(particles=500, importance_iterations=256, rounds=dict(AAC=10, ANC=4),
            status="Retained finite budget with disclosed sensitivity. Not a convergence or optimality claim."),
        jin="Published anchored coordinate decoder.",
        area="Unchanged known-anchor bounds using the published floor diagonal. No unknown coordinates define the search area."))
    files = sorted(p for p in OUTPUT.rglob("*") if p.is_file())
    write(OUTPUT / "manifest.json", dict(files={str(p.relative_to(OUTPUT)): digest(p) for p in files}))
    pilot.register_files([*files, OUTPUT / "manifest.json"])
    print(json.dumps(dict(selected=selected, radius_cm=chosen["solve_config"]["robust_delta"])), flush=True)


if __name__ == "__main__":
    execute()
