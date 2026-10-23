"""The cooperative first stage of IDLab on the reserved rows of each fold of the confirmatory round alone.

The first stage of IDLab is the pair of multipliers of the agent only and the anchor aided modes that
idlab_training.range_profiles fits on every reserved row. This script fits them by the same procedure on the rows of
each fold of the round alone, research/solid/folds.py, and writes them to fits/idlab_first_stage.json. The affine
correction and the packet scale stay those of the whole partition, the calibration on the same layout that every method
receives. The same procedure on every reserved row must give back the multipliers of idlab-training.json, and the round
positions the units of each fold with the first stage of the other.

The L-BFGS-B search of calibration.fit_shape can stop with an abnormal line search at the minimum itself, when rounding
leaves it nothing to gain, and calibration.fit_shape then raises. The objective is then minimized by BFGS from the same
start and its point kept where the gradient vanishes, the minimum of the convex objective, and the record notes how
every fit ended (fit_shape). calibration.fit_shape stays as it is for its other callers.

The record names what it was fitted from (identity): the training fit, the split of the reserved rows, the surveyed
coordinates, the fold assignment of IDLab and the sources of the procedure. A record that names the current ones is
kept as it is, so the stages that run side by side fit it once, and the round refuses a record that names others
(stale). This script replaces fit_idlab.py --first-stage, the same fit without the robustness to an abnormal stop.

Usage: fit_idlab_first_stage.py [--force]
       writes fits/idlab_first_stage.json unless it names the current inputs, and with --force in any case
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_idlab  # noqa: E402
import folds  # noqa: E402
import aac_training  # noqa: E402
import calibration  # noqa: E402
import idlab_admission  # noqa: E402
import idlab_split  # noqa: E402
import idlab_training  # noqa: E402

RECORD = "idlab_first_stage.json"
SCHEMA = "solid.fit.idlab-first-stage.v2"
MODES = ("af", "aac")
# the relative difference within which the same procedure on every reserved row gives back the training fit's multipliers
TOLERANCE = 1e-6
# the bound of calibration.fit_shape on the log of each multiplier and the settings of its L-BFGS-B search
SHAPE_LOG_BOUND = 4.0
SHAPE_OPTIONS = {"ftol": 1e-12, "gtol": 1e-8}
# the largest projected gradient of the shape objective at the point kept after an abnormal stop of that search, the
# tolerance of the first stage fit of research/solid/first_stage.py
SHAPE_GRADIENT_TOLERANCE = 1e-6
TRAINING = "idlab-training.json"
SURVEYED = "groundtruth9.csv"
RULE = ("The multipliers of the agent only and the anchor aided first stage of idlab_training.range_profiles, fitted on the reserved rows "
        "of the configurations of each fold of the round alone, with the affine correction and the packet scale of every reserved row. "
        "The round positions the units of each fold with the first stage of the other.")


def shape_objective(residuals, scale, weights):
    """The objective of calibration.fit_shape with its gradient in the logs of the two multipliers, written out as that
    function writes it: the pseudo Huber loss, delta 0.7, of every residual over the scale and the multiplier of its
    sign, weighted, and the log of the multipliers' sum. It is convex in the logs, so a point where its gradient
    vanishes is its minimum."""
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
    return objective


def projected_gradient(objective, point):
    """The largest component of the objective's gradient at a point, leaving out a component that pushes against a
    bound of calibration.fit_shape."""
    gradient = np.array(objective(point)[1], dtype=float)
    gradient[(point <= -SHAPE_LOG_BOUND) & (gradient > 0)] = 0.0
    gradient[(point >= SHAPE_LOG_BOUND) & (gradient < 0)] = 0.0
    return float(np.max(np.abs(gradient)))


def fit_shape(residuals, scale, weights, note=None):
    """The multipliers of calibration.fit_shape, robust to an abnormal stop of its L-BFGS-B search. A search that
    completes gives calibration.fit_shape's own numbers. That search can end with an abnormal line search at the minimum
    itself, when rounding leaves it nothing to gain, and calibration.fit_shape then raises. The search is then run again
    on the same objective to find where it stopped, and the objective is minimized by BFGS from the same start. Its
    point is kept when it lies inside the bounds and its gradient vanishes there within SHAPE_GRADIENT_TOLERANCE, the
    minimum of the convex objective, and the fit is refused otherwise. note, a dict, receives how the fit ended: the
    method, and after an abnormal stop the search's message, the gradients where it stopped and at the point kept, and
    the largest relative change of a multiplier between the two."""
    note = {} if note is None else note
    try:
        shape = calibration.fit_shape(residuals, scale, weights)
    except ValueError as error:
        objective = shape_objective(residuals, scale, weights)
        stopped = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B", bounds=[(-SHAPE_LOG_BOUND, SHAPE_LOG_BOUND)] * 2, options=SHAPE_OPTIONS)
        if stopped.success or str(stopped.message) != str(error):
            raise ValueError(f"the search of the written out objective ends otherwise than calibration.fit_shape's ({error})") from error
        polished = minimize(objective, np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-10})
        gradient = projected_gradient(objective, polished.x)
        if not (np.isfinite(polished.x).all() and np.all(np.abs(polished.x) <= SHAPE_LOG_BOUND) and gradient <= SHAPE_GRADIENT_TOLERANCE):
            raise ValueError(f"the shape fit stopped abnormally ({error}), and BFGS reaches no stationary point inside its bounds, "
                             f"gradient {gradient:.1e}") from error
        note.update(method="BFGS after an abnormal stop of L-BFGS-B", message=str(error), stopped_gradient=projected_gradient(objective, stopped.x),
                    gradient=gradient, change=float(np.max(np.abs(np.expm1(polished.x - stopped.x)))))
        positive, negative = np.exp(polished.x)
        return dict(positive_scale=float(positive), negative_scale=float(negative))
    note.update(method="L-BFGS-B")
    return shape


def first_stage(cells, correction, scale, notes=None):
    """The cooperative first stage fitted on the rows of the given link settings, as idlab_training.range_profiles fits
    it on every reserved row: the asymmetric shape of the corrected residuals over the packet scale for the agent only
    mode and over the anchor aided scale law for the anchor aided mode, every link setting with the same total weight
    and every row of it the same share, by fit_shape. The affine correction and the packet scale are given. notes, a
    dict, receives how the fit of each mode ended."""
    measured, reference, weights = calibration.pooled(cells)
    residual = calibration.corrected(measured, correction) - reference
    notes = {} if notes is None else notes
    return dict(af=fit_shape(residual, scale, weights, notes.setdefault("af", {})),
                aac=fit_shape(residual, aac_training.aac_scale(reference), weights, notes.setdefault("aac", {})))


def relative_difference(first, second):
    """The largest relative difference between the two multipliers of two first stages."""
    return max(abs(first[key] - second[key]) / abs(second[key]) for key in ("positive_scale", "negative_scale"))


def fold_rows(cells, fold):
    """The reserved rows of the link settings whose radio configuration lies in one fold of the round, the link
    settings fit_idlab.py --fold keeps."""
    kept = folds.idlab_configurations(fold)
    return {key: values for key, values in cells.items() if fit_idlab.configuration_of(key) in kept}


def first_stages(cells, training):
    """The fits of the record from the reserved rows and the IDLab training fit: the first stage of every reserved row,
    which must give back the training fit's multipliers within TOLERANCE, so the procedure is the training fit's, and
    the first stage of each fold of the round on the rows of its configurations alone, all with the affine correction
    and the packet scale of the whole partition. Every part records how the fit of each mode ended (fit_shape)."""
    correction, scale = training["correction"], training["profiles"]["scale_cm"]
    notes = {}
    whole = first_stage(cells, correction, scale, notes)
    recorded = {mode: training["profiles"][f"{mode}_range_error"] for mode in MODES}
    differences = {mode: relative_difference(whole[mode], recorded[mode]) for mode in MODES}
    if max(differences.values()) > TOLERANCE:
        raise ValueError(f"the first stage of every reserved row differs from the training fit's by {differences}, so the procedure is not the "
                         "training fit's")
    parts = {}
    for fold in folds.FOLDS:
        rows, fold_notes = fold_rows(cells, fold), {}
        fitted = first_stage(rows, correction, scale, fold_notes)
        parts[str(fold)] = dict(fitted, fold=folds.record("idlab", fold), link_settings=len(rows), rows=int(sum(len(values) for values in rows.values())),
                                optimizer=fold_notes)
    return dict(schema=SCHEMA, correction=correction, scale_cm=scale, whole=dict(whole, differences=differences, optimizer=notes), folds=parts, rule=RULE)


# ------------------------------------------------------------------------------------------- the record
def sources():
    """The files of the procedure whose SHA-256 the record names: this script, the configuration of a link setting
    (fit_idlab.py), the reading of the reserved rows and the fit of the shape. The fold assignment enters by its own
    identity."""
    return [Path(__file__).resolve(), Path(fit_idlab.__file__).resolve(), Path(idlab_training.__file__).resolve(),
            Path(calibration.__file__).resolve(), Path(aac_training.__file__).resolve()]


def identity():
    """What a record is fitted from: the IDLab training fit, the split of the reserved rows, the surveyed coordinates,
    the fold assignment of IDLab and the sources of the procedure. No evaluation row enters."""
    return dict(training_sha256=common.sha256(common.EXTERNAL / TRAINING), split_sha256=common.sha256(idlab_split.OUTPUT),
                surveyed_sha256=common.sha256(idlab_admission.FOLDER / SURVEYED), assignment_sha256=folds.identity("idlab"),
                evaluation_rows_used=0, sources={path.name: common.sha256(path) for path in sources()})


def stale(record):
    """Why a record no longer holds the first stage of the current inputs, empty when it does: another schema, another
    identity, or a fold entry of another assignment."""
    if record.get("schema") != SCHEMA:
        return [f"{RECORD} is of schema {record.get('schema')}, not {SCHEMA}"]
    reasons = []
    recorded, current = record.get("identity") or {}, identity()
    for key in sorted(set(recorded) | set(current)):
        if key == "sources":
            names = set(recorded.get(key) or {}) | set(current[key])
            changed = sorted(name for name in names if (recorded.get(key) or {}).get(name) != current[key].get(name))
            if changed:
                reasons.append(f"{RECORD} was fitted with other sources, {', '.join(changed)}")
        elif recorded.get(key) != current.get(key):
            reasons.append(f"{RECORD} names another {key}")
    for fold in folds.FOLDS:
        if ((record.get("folds") or {}).get(str(fold)) or {}).get("fold") != folds.record("idlab", fold):
            reasons.append(f"{RECORD} does not hold fold {fold} of the declared assignment")
    return reasons


def reserved_rows():
    """The reserved IDLab training rows by link setting and the IDLab training fit, read as fit_idlab.py reads them
    (fit_reduced.idlab_inputs)."""
    split = fit_idlab.pilot.evidence.read(idlab_split.OUTPUT)
    surveyed = np.loadtxt(idlab_admission.FOLDER / SURVEYED, delimiter=",")
    coordinates = {int(row[0]): row[1:] for row in surveyed}
    return idlab_training.load_training(split, idlab_admission.FOLDER, coordinates), common.read_json(common.EXTERNAL / TRAINING)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="fit again when the record names the current inputs")
    args = parser.parse_args()
    path = common.FITS / RECORD
    if path.exists() and not args.force:
        reasons = stale(common.read_json(path))
        if not reasons:
            print(f"{RECORD} names the current inputs and is kept", flush=True)
            return
        print(" ".join(reasons), flush=True)
    cells, training = reserved_rows()
    print(f"{len(cells)} link settings, {sum(map(len, cells.values()))} reserved ranges, scale {training['profiles']['scale_cm']:.2f} cm", flush=True)
    record = dict(first_stages(cells, training), identity=identity())
    for part, entry in [("every reserved row", record["whole"])] + [(f"fold {fold}", record["folds"][str(fold)]) for fold in folds.FOLDS]:
        print(part, {mode: {key: round(value, 6) for key, value in entry[mode].items()} for mode in MODES}, entry["optimizer"], flush=True)
    common.write_json(path, record)


if __name__ == "__main__":
    main()
