"""ADE-BP's ranging model fitted on the training data that SOLID's link model is fitted on.

Empirical ADE-BP (Li et al. 2015) scores a measured range r against the planar distance d between two position
hypotheses with an asymmetric Laplace density whose two scales grow linearly with the measured range. The baseline's
own procedure, fit_profiles.fit_ade_model, fits the four parameters by maximum likelihood over individual reports, with
the residual r - d taken against the planar distance because the method compares every range with a planar distance.
This script applies that procedure unchanged to the training data of every dataset, with the range conversion that
every method receives there.

| Dataset | Training data | Range conversion |
| --- | --- | --- |
| TIERS | Every report of the 28 links in the training blocks of the three recordings | The Case 1 affine correction |
| IDLab | Every valid reserved row of every directed link and radio setting | The affine correction fitted on the reserved rows |
| Bern, held scan h | Every row of the ten other training scans | The calibration law of fold h, which also converts the inputs of held scan h |
| Bern, evaluation | Every row of the eleven training scans | The law fitted on all eleven scans, which the engine applies to raw timings |

The training halves are those of research/artifacts/solid/splits/layout-split.json, the Bern folds those of
research/solid/fit_bern.py. No evaluation row enters a fit. The records go to
research/artifacts/baselines/<dataset>-ade-fit.json with the rows, the conversion, the previous parameters and the
hashes of every source, and the Bern evaluation configurations to research/benchmarks/bern/configs/
ade-bp-2015-refit-<mode>.json. The held-scan runner builds the configuration of a fold from the Bern record with
bern_config. With --check the fits are repeated and compared with the tracked records, which stay unchanged. A reader's
reproduction (SOLID_READER=1) compares the values alone, the fitted laws, their reports and rows, conversions, previous
laws and areas, and not the SHA-256 of the sources and the code, which name the reader's own files and build.

Usage: ade_training.py [--datasets tiers,idlab,bern] [--check]
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bern/ade_bp"))
import fit_profiles  # noqa: E402

# the data folder, SOLID_DATA or research/data
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
READER = os.environ.get("SOLID_READER") == "1"
EXTERNAL = DATA / "external-benchmarks"
RECORDS = ROOT / "research/artifacts/baselines"
BERN_CONFIGS = ROOT / "research/benchmarks/bern/configs"
BERN_TABLE = DATA / "inputs/65c077bb1aa9b879/Rangings at Responders.pkl"
BERN_TABLE_MD5 = "d5d71fe55c758c2a87d7efd812357c5b"
BERN_TRAINING_ROWS = 1_947_886
BERN_SCANS = tuple(range(1, 12))
ENGINE_CALIBRATION = ROOT / "runtime/engine/src/calibration.rs"
SCHEMA = "solid.baselines.ade-fit.v1"
MODES = ("aac", "anc")
# the method settings the Bern configurations keep, only the area and the ranging model come from the fit
BERN_METHOD = dict(method="ade-bp-2015", particles=500, iterations=4, importance_iterations=256)
PREVIOUS_BERN_CONFIG = BERN_CONFIGS / "ade-bp-2015-aac.json"
PROCEDURE = ("fit_profiles.fit_ade_model: maximum likelihood of the asymmetric Laplace density with scales "
             "slope times measured range plus intercept, over individual reports, Nelder-Mead from (0.02, 20, 0.01, 40) "
             "with nonnegative slopes and positive intercepts. Residual is the converted range minus the planar distance.")
# relative tolerance of --check for a repeated fit, far below any effect on the method and above the drift of a
# Nelder-Mead path under a different floating point library
CHECK_TOLERANCE = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def relative(path):
    """A path relative to the repository, without resolving the data link, so a record reads the same on every machine. A
    file of a data folder outside the repository (SOLID_DATA) is named by its place in research/data."""
    path = Path(path)
    if path.is_relative_to(DATA) and not path.is_relative_to(ROOT):
        return str(Path("research/data") / path.relative_to(DATA))
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def planar_cm(a, b):
    """The planar distance between two points given in centimetres, the distance ADE-BP scores a range against."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


PARAMETERS = ("positive_slope", "positive_intercept_cm", "negative_slope", "negative_intercept_cm")


def nll_per_report(model, measured, truth):
    """ADE-BP's negative log likelihood per report of a ranging model on paired ranges and distances."""
    return fit_profiles.ade_negative_log_likelihood(np.array([model[k] for k in PARAMETERS]), measured, measured - truth) / len(measured)


def fit(measured, truth, previous=None):
    """ADE-BP's own fit on paired converted ranges and planar distances, with the summary of its residuals and, for
    comparison, the likelihood per report of the previous model on the same reports."""
    measured, truth = np.asarray(measured, dtype=float), np.asarray(truth, dtype=float)
    if measured.shape != truth.shape or not len(measured) or not np.isfinite(measured).all() or not np.isfinite(truth).all():
        raise ValueError("ADE-BP needs paired finite ranges and distances")
    model, nll = fit_profiles.fit_ade_model(measured, truth)
    residual = measured - truth
    result = dict(ranging_model=model, negative_log_likelihood=nll, nll_per_report=nll / len(measured), reports=int(len(measured)),
                  residual_quantiles_cm=dict(zip(("5", "25", "50", "75", "95"), np.percentile(residual, [5, 25, 50, 75, 95]).round(3).tolist())))
    if previous is not None:
        result["previous_nll_per_report"] = nll_per_report(previous, measured, truth)
    return result


def gap(a, b):
    """The relative difference of two parameters, absolute for parameters at the zero bound."""
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 1e-9 else abs(a - b)


def header(dataset, rule, conversion, sources, previous):
    return dict(schema=SCHEMA, dataset=dataset, method="ade-bp-2015", rule=rule, procedure=PROCEDURE, distance="planar Euclidean, cm",
                conversion=conversion, evaluation_rows_used=0, previous=previous,
                sources={name: dict(path=relative(path), sha256=sha256(path)) for name, path in sources.items()},
                code={relative(p): sha256(p) for p in (Path(__file__).resolve(), Path(fit_profiles.__file__).resolve())},
                fitted_on=dict(host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                               scipy=__import__("scipy").__version__))


# ------------------------------------------------------------------------------------------------ TIERS
def tiers_reports(base, catalog, blocks):
    """(converted range, planar distance) of every report of every link in the given blocks, calibrated track.

    The AF units of a block carry every link. Its ten sample unit holds the ten reports of each link, and the one
    sample unit repeats the first of them, so the ten sample unit gives every report once."""
    import tiers_campaign as campaign
    measured, truth, links, used = [], [], 0, set()
    for batch in catalog:
        if batch["track"] != "calibrated" or batch["mode"] != "AF" or batch["block"] not in blocks:
            continue
        request, _ = campaign.batch_inputs(base, batch)
        coordinates = base["coordinates_cm"][batch["acquisition"]]
        unit = next(u for u in request["units"] if u["samples"] == 10)
        for observation in unit["problem"]["observations"]:
            samples = observation["calibrated_samples"]
            if len(samples) != 10:
                raise ValueError("a TIERS training link lacks its ten reports")
            distance = planar_cm(coordinates[str(observation["from"])], coordinates[str(observation["to"])])
            measured.extend(s["distance_cm"] for s in samples)
            truth.extend([distance] * len(samples))
            links += 1
        used.add(batch["block"])
    if used != set(blocks):
        raise ValueError("a TIERS training block has no AF batch")
    return np.asarray(measured), np.asarray(truth), dict(blocks=len(used), links=links)


def tiers():
    import layout_split
    folder = EXTERNAL / "tiers-static-campaign"
    base, catalog = read(folder / "base.json.gz"), read(folder / "catalog.json.gz")
    blocks = layout_split.tiers_blocks("training")
    measured, truth, counts = tiers_reports(base, catalog, blocks)
    previous = dict(source="Case 1 two device calibration table, tiers-static-campaign base fitted.ade_ranging_model",
                    ranging_model=base["fitted"]["ade_ranging_model"])
    record = header("tiers", "Every report of the 28 links of the training blocks of the three static recordings, calibrated track.",
                    dict(rule="max(0, gain * range + offset), the Case 1 affine correction every method receives",
                         correction=base["fitted"]["correction"]),
                    dict(base=folder / "base.json.gz", catalog=folder / "catalog.json.gz", split=layout_split.RECORD), previous)
    record.update(training_blocks=sorted(blocks), **counts, fit=fit(measured, truth, previous["ranging_model"]))
    return record


# ------------------------------------------------------------------------------------------------ IDLab
def idlab():
    import idlab_admission
    import idlab_split
    import idlab_training
    split, fitted = read(idlab_split.OUTPUT), read(idlab_training.OUTPUT)
    surveyed = np.loadtxt(idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")
    coordinates = {int(r[0]): r[1:] for r in surveyed}
    cells = idlab_training.load_training(split, idlab_admission.FOLDER, coordinates)
    measured, truth = [], []
    for (node, peer, *_), rows in cells.items():
        # surveyed coordinates are millimetres, the fit works in centimetres
        distance = planar_cm(coordinates[node][:2] / 10, coordinates[peer][:2] / 10)
        measured.append(np.maximum(0.0, fitted["correction"]["gain"] * rows[:, 1] + fitted["correction"]["offset_cm"]))
        truth.append(np.full(len(rows), distance))
    measured, truth = np.concatenate(measured), np.concatenate(truth)
    previous = dict(source="idlab-training.json profiles.ade_ranging_model, the same reserved rows against the surveyed 3D distance",
                    ranging_model=fitted["profiles"]["ade_ranging_model"])
    record = header("idlab", "Every valid reserved row of every directed link and radio setting, the IDLab training half.",
                    dict(rule="max(0, gain * range + offset), the affine correction fitted on the reserved rows",
                         correction=fitted["correction"]),
                    dict(split=idlab_split.OUTPUT, training=idlab_training.OUTPUT, surveyed=idlab_admission.FOLDER / "groundtruth9.csv"),
                    previous)
    record.update(link_settings=len(cells), fit=fit(measured, truth, previous["ranging_model"]),
                  source_note="The source of the frozen campaigns is confirmed. They fitted the same reserved rows with the same "
                              "correction, against the surveyed distance in three dimensions. The refit takes the planar "
                              "distance of the baseline's procedure. The node heights differ by less than ten centimetres.")
    return record


# ------------------------------------------------------------------------------------------------ Bern
def bern_table():
    import pandas as pd
    if md5(BERN_TABLE) != BERN_TABLE_MD5:
        raise ValueError("expected the corrected Bern responder table")
    frame = pd.read_pickle(BERN_TABLE)
    frame = frame.loc[frame["Scan_ID"].isin(BERN_SCANS)]
    if len(frame) != BERN_TRAINING_ROWS or set(frame["Class"].unique()) != {"Train"}:
        raise ValueError("incomplete or nontraining Bern fitting population")
    return frame


def converted(tof, coefficients):
    """The range every method receives for raw timings under a calibration law, and the kept rows. Timings outside
    the runtime interval carry no range, as fit_bern.calibrated_packets and the engine's raw path drop them."""
    tof = np.asarray(tof, dtype=float)
    kept = np.isfinite(tof) & (tof >= coefficients["runtime_tof_floor_ns"]) & (tof <= coefficients["runtime_tof_max_ns"])
    return np.maximum(0.0, coefficients["range_slope_cm_per_ns"] * tof[kept] + coefficients["range_offset_cm"]), kept


def engine_law():
    """The range conversion of the engine's default calibration, which ADE-BP applies to the raw evaluation timings."""
    source = ENGINE_CALIBRATION.read_text()

    def constant(name):
        match = re.search(rf"{name}:\s*([0-9_.]+)", source)
        if not match:
            raise ValueError(f"no {name} in {ENGINE_CALIBRATION}")
        return float(match.group(1).replace("_", ""))

    return dict(range_slope_cm_per_ns=constant("range_slope_cm_per_ns"), range_offset_cm=constant("range_offset_cm"))


def earlier_refit(held):
    """The fit of the same fold and conversion by the earlier corrected-training driver, whose code left the repository,
    and its record, None where the data collection lacks it or its record holds no such fit, as the calibration records
    of write_calibration.py hold none."""
    path = DATA / "corrected-training" / (f"scan-{held:02d}" if held else "full") / "fit.json"
    record = read(path) if path.exists() else {}
    return (record["ade"]["refit"]["ranging_model"], path) if "ade" in record else (None, path)


def bern_fold(frame, held, calibration, previous=None):
    """ADE-BP's fit on the training scans of one fold, held None for the evaluation fit on all eleven, with the
    likelihood of the previous model on the same reports and the agreement with the earlier refit of the fold."""
    scans = [s for s in BERN_SCANS if s != held]
    selected = frame.loc[frame["Scan_ID"].isin(scans)]
    coefficients, calibration_path = calibration(held)
    measured, kept = converted(selected["ToF"].to_numpy(dtype=float), coefficients)
    rows = selected.loc[kept]
    truth = 100.0 * np.hypot((rows["Poller_X"] - rows["Responder_X"]).to_numpy(dtype=float),
                             (rows["Poller_Y"] - rows["Responder_Y"]).to_numpy(dtype=float))
    result = fit(measured, truth, previous)
    area = fit_profiles.area_bounds(selected)
    earlier, earlier_path = earlier_refit(held)
    agreement = None if earlier is None else dict(record=relative(earlier_path), ranging_model=earlier,
                                                  largest_relative_difference=max(gap(earlier[k], result["ranging_model"][k]) for k in PARAMETERS))
    # a reader's data without the record of all eleven scans takes the engine's law, and names no record (fit_bern.calibration)
    return dict(held_scan=held, fit_scans=scans, rows=int(len(selected)), rows_outside_runtime_interval=int((~kept).sum()),
                calibration=dict(path=calibration_path and relative(calibration_path), sha256=calibration_path and sha256(calibration_path),
                                 coefficients=coefficients),
                area=area, **result, earlier_refit=agreement, config=dict(BERN_METHOD, area=area, ranging_model=result["ranging_model"]))


def bern():
    solid = str(ROOT / "research/solid")
    if solid not in sys.path:
        sys.path.insert(0, solid)
    import fit_bern
    frame = bern_table()
    previous_config = read(PREVIOUS_BERN_CONFIG)
    folds = {str(held): bern_fold(frame, held, fit_bern.calibration, previous_config["ranging_model"]) for held in BERN_SCANS}
    evaluation = bern_fold(frame, None, fit_bern.calibration, previous_config["ranging_model"])
    law = engine_law()
    if any(evaluation["calibration"]["coefficients"][k] != v for k, v in law.items()):
        raise ValueError("the all-scan calibration law differs from the engine's conversion of raw timings")
    previous = dict(source="configs/ade-bp-2015-aac.json and -anc.json, one fit on all eleven training scans under the earlier raw "
                           "timing conversion 28.50909085 cm/ns and 19.20725779 cm, used for the evaluation corpus and every held scan",
                    ranging_model=previous_config["ranging_model"], area=previous_config["area"],
                    sha256=sha256(PREVIOUS_BERN_CONFIG))
    record = header("bern", "Held scan h: every row of the ten other training scans of the corrected responder table, under the "
                            "calibration law of fold h. Evaluation: every row of the eleven training scans under the law fitted on "
                            "all eleven, the engine's conversion of the raw evaluation timings.",
                    dict(rule="max(0, slope * ToF + offset) for runtime_tof_floor_ns <= ToF <= runtime_tof_max_ns, the law of the fold "
                              "(fit_bern.calibration), exactly as the held-scan inputs and the engine convert raw timings",
                         engine=law),
                    dict(table=BERN_TABLE, full_calibration=fit_bern.FULL_CALIBRATION), previous)
    record.update(folds=folds, evaluation=evaluation, configs={mode: relative(bern_config_path(mode)) for mode in MODES})
    return record


def bern_config_path(mode):
    return BERN_CONFIGS / f"ade-bp-2015-refit-{mode}.json"


def bern_record(path=None):
    record = read(path or RECORDS / "bern-ade-fit.json")
    if record.get("schema") != SCHEMA or record.get("dataset") != "bern":
        raise ValueError("not a Bern ADE-BP fit record")
    return record


def bern_config(held, record=None):
    """ADE-BP's configuration for held scan `held`, fitted without it, or for the evaluation corpus with held None."""
    record = record or bern_record()
    entry = record["evaluation"] if held is None else record["folds"][str(int(held))]
    if entry["held_scan"] != (None if held is None else int(held)) or (held is not None and int(held) in entry["fit_scans"]):
        raise ValueError("the held scan entered its own ADE-BP fit")
    return copy.deepcopy(entry["config"])


def ranging_model(dataset, path=None):
    """The ranging model of a dataset's training half, from its tracked fit record."""
    record = read(path or RECORDS / f"{dataset}-ade-fit.json")
    if record.get("schema") != SCHEMA or record.get("dataset") != dataset:
        raise ValueError(f"not an ADE-BP fit record of {dataset}")
    return copy.deepcopy(record["fit"]["ranging_model"])


def with_ranging_model(baselines, model):
    """The baselines with ADE-BP's ranging model replaced, every other method and setting unchanged."""
    result = copy.deepcopy(baselines)
    for entry in result:
        if entry["method"] == "ade-bp-2015":
            if set(entry["method_config"]["ranging_model"]) != set(model):
                raise ValueError("the ranging model has other parameters")
            entry["method_config"]["ranging_model"] = copy.deepcopy(model)
    return result


# ------------------------------------------------------------------------------------------------ check and main
FITTERS = dict(tiers=tiers, idlab=idlab, bern=bern)


def models(record):
    if record["dataset"] != "bern":
        return {"fit": record["fit"]["ranging_model"]}
    return {**{f"fold {h}": e["ranging_model"] for h, e in record["folds"].items()}, "evaluation": record["evaluation"]["ranging_model"]}


def differences(tracked, fresh):
    """The largest relative difference of every fitted parameter, and the differences of anything else that must match.
    A reader's reproduction compares no SHA-256 of the sources, the code or a fold's calibration record, files of its own,
    but the values they gave: the reports and rows, the conversion, the previous law and a fold's calibration
    coefficients and area."""
    worst, problems = 0.0, []
    a, b = models(tracked), models(fresh)
    if set(a) != set(b):
        return math.inf, ["different fits"]
    for name in a:
        for key, value in a[name].items():
            other = b[name][key]
            worst = max(worst, gap(value, other))
    for key in ("conversion", "previous", "rule") if READER else ("sources", "code", "conversion", "previous", "rule"):
        if tracked.get(key) != fresh.get(key):
            problems.append(key)
    if tracked["dataset"] == "bern":
        for name in ("folds", "evaluation"):
            entries = tracked[name].values() if name == "folds" else [tracked[name]]
            others = fresh[name].values() if name == "folds" else [fresh[name]]
            for x, y in zip(entries, others):
                for key in ("fit_scans", "rows", "rows_outside_runtime_interval", "calibration", "area"):
                    kept, refit = x[key], y[key]
                    if READER and key == "calibration":
                        kept, refit = kept["coefficients"], refit["coefficients"]
                    if kept != refit:
                        problems.append(f"{name} {key}")
    elif tracked.get("fit", {}).get("reports") != fresh["fit"]["reports"]:
        problems.append("reports")
    return worst, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="tiers,idlab,bern")
    parser.add_argument("--check", action="store_true", help="refit and compare with the tracked records without writing them")
    args = parser.parse_args()
    failed = []
    for dataset in args.datasets.split(","):
        record = FITTERS[dataset]()
        path = RECORDS / f"{dataset}-ade-fit.json"
        for name, model in models(record).items():
            print(f"{dataset} {name}: " + ", ".join(f"{k} {v:.6g}" for k, v in model.items()), flush=True)
        if args.check:
            tracked = read(path)
            worst, problems = differences(tracked, record)
            if dataset == "bern":
                problems += [f"configuration {mode}" for mode in MODES if read(bern_config_path(mode)) != tracked["evaluation"]["config"]]
            status = "reproduced" if worst <= CHECK_TOLERANCE and not problems else "differs"
            print(f"{dataset}: {status}, largest relative parameter difference {worst:.3g}" + (f", other differences {problems}" if problems else ""), flush=True)
            if status != "reproduced":
                failed.append(dataset)
                write(ROOT / "out/baselines/ade-check" / path.name, record)
            continue
        write(path, record)
        if dataset == "bern":
            for mode in MODES:
                write(bern_config_path(mode), record["evaluation"]["config"])
        print(f"written {relative(path)}", flush=True)
    if failed:
        raise SystemExit(f"ADE-BP fits differ from the tracked records for {', '.join(failed)}, the refits are under out/baselines/ade-check")


if __name__ == "__main__":
    main()
