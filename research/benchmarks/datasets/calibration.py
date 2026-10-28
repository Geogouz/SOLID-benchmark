"""Select range calibration on TIERS Case 1 before replaying the fixed AF pilot."""

import argparse
from collections import Counter, defaultdict
import copy
import csv
import html
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.optimize import minimize

import pilot
import deployment_profile

OUTPUT = pilot.audit.DATA / "tiers-case1-calibrated"
MODELS = ("identity", "offset", "affine")


def load_case1(path):
    cells = defaultdict(list)
    excluded = []
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames != ["real", "plane", "angle", "serial", "error", "opti"]:
            raise ValueError("unexpected Case 1 columns")
        for line, row in enumerate(reader, 2):
            if row["opti"] in (None, ""):
                excluded.append(line)
                continue
            values = np.array([float(row[k]) for k in ("real", "angle", "serial", "error", "opti")])
            if not np.isfinite(values).all() or min(values[2], values[4]) <= 0:
                raise ValueError("invalid Case 1 measurement or reference")
            if abs(values[2] - values[4] - values[3]) > 1e-9:
                raise ValueError("Case 1 error column disagrees with its reference")
            key = (float(values[0]), row["plane"], float(values[1]))
            cells[key].append((line, 100 * values[2], 100 * values[4]))
    if not cells:
        raise ValueError("Case 1 has no referenced measurements")
    return {key: np.array(rows) for key, rows in sorted(cells.items())}, excluded


def pooled(cells):
    rows = np.concatenate(list(cells.values()))
    weights = np.concatenate([np.full(len(r), 1 / (len(cells) * len(r))) for r in cells.values()])
    return rows[:, 1], rows[:, 2], weights


def fit_correction(cells, model):
    measured, reference, weights = pooled(cells)
    if model == "identity":
        return dict(gain=1., offset_cm=0.)
    if model == "offset":
        return dict(gain=1., offset_cm=float(weights @ (reference - measured)))
    if model != "affine":
        raise ValueError("unknown calibration model")
    design = np.column_stack((measured, np.ones(len(measured))))
    gain, offset = np.linalg.lstsq(design * np.sqrt(weights[:, None]), reference * np.sqrt(weights), rcond=None)[0]
    if gain <= 0:
        raise ValueError("calibration must preserve range ordering")
    return dict(gain=float(gain), offset_cm=float(offset))


def corrected(measured, model):
    return np.maximum(0., model["gain"] * np.asarray(measured) + model["offset_cm"])


def splits(cells):
    for family in ("distance", "orientation"):
        group = (lambda key: key[:1]) if family == "distance" else (lambda key: key[1:])
        for held in sorted({group(key) for key in cells}):
            yield family, held, {k: v for k, v in cells.items() if group(k) != held}, \
                {k: v for k, v in cells.items() if group(k) == held}


def fit_shape(residuals, scale, weights):
    values = residuals / scale

    def objective(log_scales):
        scales = np.exp(log_scales)
        positive = values >= 0
        z = values / np.where(positive, scales[0], scales[1])
        root = np.sqrt(1 + (z / .7) ** 2)
        loss = weights @ (.49 * (root - 1)) + np.log(scales.sum())
        gradient = scales / scales.sum()
        for i, mask in enumerate((positive, ~positive)):
            gradient[i] -= np.sum(weights[mask] * z[mask] ** 2 / root[mask])
        return loss, gradient

    result = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B",
                      bounds=[(-4, 4), (-4, 4)], options={"ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise ValueError(result.message)
    positive, negative = np.exp(result.x)
    return dict(positive_scale=float(positive), negative_scale=float(negative))


def fit_case1(path):
    """No evaluation request, reference or score is accepted by the fitter."""
    cells, excluded = load_case1(path)
    folds = []
    for family, held, training, validation in splits(cells):
        for model in MODELS:
            parameters = fit_correction(training, model)
            scores = []
            for key, rows in validation.items():
                residual = corrected(rows[:, 1], parameters) - rows[:, 2]
                scores.append(dict(condition=list(key), rows=len(rows), mae_cm=float(np.mean(np.abs(residual)))))
            folds.append(dict(family=family, held=list(held), model=model, parameters=parameters,
                              training_conditions=len(training), validation=scores))
    results = []
    for model in MODELS:
        metrics = {family: float(np.mean([s["mae_cm"] for f in folds if f["family"] == family and f["model"] == model
                                         for s in f["validation"]])) for family in ("distance", "orientation")}
        results.append(dict(model=model, **metrics, selection_mae_cm=float(np.mean(list(metrics.values())))))
    selected = min(results, key=lambda row: (row["selection_mae_cm"], MODELS.index(row["model"])))
    parameters = fit_correction(cells, selected["model"])
    measured, reference, weights = pooled(cells)
    residuals = corrected(measured, parameters) - reference
    scale = float(np.sqrt(weights @ residuals ** 2))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("calibration has no identifiable positive residual scale")
    shape = fit_shape(residuals, scale, weights)
    return dict(source_sha256=pilot.identity(path), evaluation_rows_used=0,
                eligible_rows=len(measured), excluded_without_mocap_lines=excluded,
                conditions=[dict(condition=list(k), lines=r[:, 0].astype(int).tolist(),
                                 distinct_range_reference_rows=len({tuple(v) for v in r[:, 1:]})) for k, r in cells.items()],
                selection=results, selected=selected["model"], correction=parameters,
                scale_cm=scale, shape=shape, sample_tracks_share_shape=True, folds=folds)


def profile_package(fit, freeze, original, original_sha256):
    return deployment_profile.validate(dict(
        schema=deployment_profile.SCHEMA, name="TIERS Case 1 calibrated AF", method=deployment_profile.METHOD,
        inputs=copy.deepcopy(deployment_profile.INPUTS), calibration=copy.deepcopy(fit["correction"]),
        uncertainty=dict(constant_scale_cm=fit["scale_cm"]),
        range_error={n: copy.deepcopy(fit["shape"]) for n in ("1", "10")},
        availability={n: {k: original["profile"]["free"][n][k] for k in ("intercept", "slope")} for n in ("1", "10")},
        operating_area=None,
        provenance=dict(
            range_fit=dict(source="TIERS Case 1", training_sha256=fit["source_sha256"],
                           fit_sha256=freeze["calibration_sha256"], fitter_revision=freeze["source_revision"],
                           training_rows=fit["eligible_rows"], evaluation_rows_used=0),
            availability_fit=dict(source="Corrected Bern training, transferred unchanged",
                                  training_scans=original["profile"]["fit_scans"],
                                  retained_transfer_request_sha256=original_sha256, evaluation_rows_used=0),
            compatibility="TIERS acquisition hardware and deployment combination. Sensor-independent environment transfer is unverified. "
                          "Availability requires the declared pair-ranging opportunities. Both sample tracks share one range-error fit.")))


def raw_request(original):
    """Extract measured centimetres only from the hash-checked original transfer request."""
    raw = copy.deepcopy({k: original[k] for k in ("units", "baselines")})
    for unit in raw["units"]:
        for observation in unit["problem"]["observations"]:
            if observation.pop("tof_ns"):
                raise ValueError("expected original metric ranges without radio timing conversion")
            observation["ranges_cm"] = [s["distance_cm"] for s in observation.pop("calibrated_samples")]
    return raw


def prepare():
    if OUTPUT.exists():
        raise ValueError("calibration already prepared, do not overwrite its evidence")
    if pilot.audit.uncommitted():
        raise ValueError("commit the calibration source and protocol before fitting")
    path = pilot.audit.source(pilot.audit.TIERS + "case_1/final_data.csv", "tiers/case_1/final_data.csv")
    profile = fit_case1(path)
    pilot.evidence.write(OUTPUT / "calibration.json", profile)
    pilot.evidence.write(OUTPUT / "calibration-freeze.json", dict(
        source_revision=pilot.audit.revision(),
        training_source_sha256=pilot.identity(path), calibration_sha256=pilot.identity(OUTPUT / "calibration.json"),
        protocol_sha256=pilot.document(pilot.HERE / "README.md"), evaluation_rows_used=0))
    original = pilot.check_inputs()
    request_path = pilot.OUTPUT / "request.json"
    original_request = pilot.evidence.read(request_path)
    deployment = profile_package(profile, pilot.evidence.read(OUTPUT / "calibration-freeze.json"),
                                 original_request, pilot.identity(request_path))
    pilot.evidence.write(OUTPUT / "deployment-profile.json", deployment)
    request = deployment_profile.prepare_request(deployment, raw_request(original_request))
    pilot.evidence.write(OUTPUT / "request.json", request)
    (OUTPUT / "reference.json").write_bytes((pilot.OUTPUT / "reference.json").read_bytes())
    sources = dict(original["sources"])
    # The pilot's summary and manifest come from its run and report, which a reader's copy need not have made.
    for source in pilot.present([path, request_path, pilot.OUTPUT / "reference.json", pilot.OUTPUT / "summary.json",
                                 pilot.OUTPUT / "manifest.json", OUTPUT / "calibration.json", OUTPUT / "calibration-freeze.json"]):
        sources[str(source)] = pilot.identity(source)
    sources[str(OUTPUT / "deployment-profile.json")] = pilot.identity(OUTPUT / "deployment-profile.json")
    pilot.evidence.write(OUTPUT / "inputs.json", dict(original,
        schema="solid.external.tiers.case1-calibrated.v1", sources=sources,
        title="TIERS AF Case 1 calibration",
        request_sha256=pilot.identity(OUTPUT / "request.json"), reference_sha256=original["reference_sha256"],
        profile_track="Range correction, constant uncertainty and SOLID asymmetric shape use only Case 1. "
                      "All methods receive the same corrected ranges and uncertainty. Availability remains transferred from Bern. "
                      "Both sample tracks share the single-report shape. This is a retrospective Case 3 comparison, not independent confirmation.",
        uncertainty=f"Case 1 condition-balanced residual RMS of {profile['scale_cm']:.9f} cm, shared by every method",
        models_fitted=1, calibration_candidates=list(MODELS), calibration_sha256=pilot.identity(OUTPUT / "calibration.json")))
    print(json.dumps({k: profile[k] for k in ("eligible_rows", "selected", "selection", "correction", "scale_cm", "shape")}, indent=2))


def stable_record(value):
    if isinstance(value, dict):
        return {k: stable_record(v) for k, v in value.items() if k not in ("elapsed_ns", "elapsed_micros")}
    if isinstance(value, list):
        return [stable_record(v) for v in value]
    return value


def package():
    pilot.check_inputs()
    pilot.check_inputs(OUTPUT)
    original_path = pilot.OUTPUT / "request.json"
    deployment = profile_package(pilot.evidence.read(OUTPUT / "calibration.json"),
                                 pilot.evidence.read(OUTPUT / "calibration-freeze.json"),
                                 pilot.evidence.read(original_path), pilot.identity(original_path))
    path = OUTPUT / "deployment-profile.json"
    if path.exists() and deployment_profile.load(path) != deployment:
        raise ValueError("existing deployment profile differs from the frozen fit")
    request = deployment_profile.prepare_request(deployment, raw_request(pilot.evidence.read(original_path)))
    retained = pilot.evidence.read(OUTPUT / "request.json")
    expected = {n: {k: retained["profile"]["free"][n][k] for k in request["profile"]["free"][n]} for n in ("1", "10")}
    if request["units"] != retained["units"] or request["baselines"] != retained["baselines"] or request["profile"]["free"] != expected:
        raise ValueError("combined profile changed effective inference inputs")
    revision, env = pilot.build_predictor()
    with tempfile.TemporaryDirectory(prefix="solid-profile-parity-") as temporary:
        directory = Path(temporary)
        pilot.evidence.write(directory / "request.json", request)
        with (directory / "run.log").open("w") as log:
            subprocess.run([str(pilot.BINARY), str(directory / "request.json"), str(directory / "optimizer.jsonl")],
                           cwd=pilot.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        records = [json.loads(line) for line in (directory / "optimizer.jsonl").read_text().splitlines()]
        expected = [json.loads(line) for line in (OUTPUT / "predictions.jsonl").read_text().splitlines()]
        if stable_record(records) != stable_record(expected):
            raise ValueError("profile replay changed estimates or deterministic solver diagnostics")
    pilot.evidence.write(path, deployment)
    pilot.evidence.write(OUTPUT / "profile-verification.json", dict(
        source_revision=revision, binary_sha256=pilot.identity(pilot.BINARY),
        deployment_profile_sha256=pilot.identity(path), retained_request_sha256=pilot.identity(OUTPUT / "request.json"),
        retained_predictions_sha256=pilot.identity(OUTPUT / "predictions.jsonl"),
        units=len(request["units"]), matched_method_outputs=len(records),
        effective_inputs_identical=True, coordinates_and_solver_diagnostics_identical=True,
        new_models_fitted=0, timing_fields_excluded=["elapsed_ns", "elapsed_micros"]))
    print(f"Packaged profile reproduces all {len(records)} retained method outputs exactly", flush=True)


def comparison_section(profile, result, original):
    rows = []
    previous = {m["id"]: m for m in original["methods"]}
    for method in result["methods"]:
        values = []
        for n in ("1", "10"):
            old = previous[method["id"]]["results"]["AF"]["tracks"][n]["s75"]
            new = method["results"]["AF"]["tracks"][n]["s75"]
            values.extend((f"{old:.3f}", f"{new:.3f}", f"{new-old:+.3f}"))
        rows.append(f'<tr><th>{html.escape(method["label"])}</th>' + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    selection = "".join(f'<tr><th>{r["model"]}</th><td>{r["distance"]:.3f}</td><td>{r["orientation"]:.3f}</td><td>{r["selection_mae_cm"]:.3f}</td></tr>'
                        for r in profile["selection"])
    counts = Counter(len(c["lines"]) for c in profile["conditions"])
    return f'''<section><h2>Original transfer versus Case 1 calibration</h2>
<p>S75 in centimetres on exactly the same thirty units per sample column. Negative change is better. No Case 3 outcome selected the calibration. <a href="../tiers-af-pilot/index.html">Original frozen transfer results</a> remain unchanged.</p>
<div class="scroll"><table><thead><tr><th>Method</th><th>Original 1</th><th>Calibrated 1</th><th>Change 1</th><th>Original 10</th><th>Calibrated 10</th><th>Change 10</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><details><summary>Case 1 selection and limitations</summary>
<p>Mean absolute range errors in centimetres, not positioning scores. Whole nominal distances or whole orientations are withheld. The lowest mean of the two validation columns selects <b>{profile['selected']}</b>.</p>
<table><thead><tr><th>Correction</th><th>Distance holdout</th><th>Orientation holdout</th><th>Selection mean</th></tr></thead><tbody>{selection}</tbody></table>
<p>Corrected centimetres equal max(0, {profile['correction']['gain']:.9f} times measured centimetres {profile['correction']['offset_cm']:+.6f}). Shared scale is {profile['scale_cm']:.6f} cm. SOLID positive and negative shape multipliers are {profile['shape']['positive_scale']:.6f} and {profile['shape']['negative_scale']:.6f}.</p>
<p>Use {profile['eligible_rows']:,} referenced rows in {len(profile['conditions'])} conditions. Exclude {len(profile['excluded_without_mocap_lines']):,} rows with no motion-capture reference. Condition sizes are {html.escape(str(dict(sorted(counts.items()))))}. Repeated rows remain, but do not establish independence. No ten-report shrinkage is fitted. Availability remains Bern-derived because Case 1 has no failed-attempt labels.</p>
<p><a href="calibration.json">All calibration folds and coefficients</a> · <a href="calibration-freeze.json">Pre-replay calibration freeze</a></p></details></section>'''


def report():
    pilot.check_inputs(OUTPUT)
    execution = pilot.evidence.read(OUTPUT / "execution.json")
    if pilot.identity(OUTPUT / "predictions.jsonl") != execution["predictions_sha256"]:
        raise ValueError("retained predictions changed")
    original_manifest = pilot.evidence.read(pilot.OUTPUT / "manifest.json")
    if pilot.identity(pilot.OUTPUT / "summary.json") != original_manifest["files"]["summary.json"]:
        raise ValueError("original summary changed")
    records = [json.loads(line) for line in (OUTPUT / "predictions.jsonl").read_text().splitlines()]
    scorer = pilot.evidence.load_scorer()
    scores = pilot.score_records(pilot.evidence.read(OUTPUT / "request.json"), pilot.evidence.read(OUTPUT / "reference.json"), records, scorer)
    section = comparison_section(pilot.evidence.read(OUTPUT / "calibration.json"), pilot.summarize(scores, scorer),
                                 pilot.evidence.read(pilot.OUTPUT / "summary.json"))
    if (OUTPUT / "profile-verification.json").exists():
        verification = pilot.evidence.read(OUTPUT / "profile-verification.json")
        deployment_profile.load(OUTPUT / "deployment-profile.json")
        for name, key in (("deployment-profile.json", "deployment_profile_sha256"), ("request.json", "retained_request_sha256"),
                          ("predictions.jsonl", "retained_predictions_sha256")):
            if pilot.identity(OUTPUT / name) != verification[key]:
                raise ValueError("packaged profile verification is stale")
        section += '<section><details><summary>Single deployment profile</summary><p>One file contains calibration, uncertainty, error shape, transferred availability and training provenance. It supports AF with one or ten samples, not AAC or ANC. All 240 method outputs match the retained run exactly, excluding execution times. No model was refitted.</p><p><a href="deployment-profile.json">Loadable deployment profile</a> · <a href="profile-verification.json">Reproduction check</a></p></details></section>'
    pilot.report(OUTPUT, section)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "report", "package"))
    {"prepare": prepare, "run": lambda: pilot.execute(OUTPUT), "report": report, "package": package}[parser.parse_args().stage]()
