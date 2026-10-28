"""Fit shared IDLab range calibration from reserved training records only."""

from collections import defaultdict
import csv
from itertools import islice
from pathlib import Path
import subprocess

import numpy as np

import aac_training
import calibration
import idlab_admission
import idlab_split
import pilot
import static_preflight
import tiers_static_check


OUTPUT = idlab_split.OUTPUT.with_name("idlab-training.json")


def load_training(split, folder, coordinates):
    by_node = defaultdict(list)
    for group in split["groups"]:
        by_node[group["node"]].append(group)
    cells = {}
    for node, groups in sorted(by_node.items()):
        with (folder / f"office_lab_node_{node:02d}_short.csv").open(newline="") as stream:
            reader, consumed = csv.DictReader(stream), 0
            for group in sorted(groups, key=lambda g: g["calibration"]["row_start"]):
                part = group["calibration"]
                start, stop = part["row_start"], part["row_stop"]
                if start < consumed or stop < start or stop != group["evaluation"]["row_start"]:
                    raise ValueError("overlapping or unordered training intervals")
                peer = group["peer"]
                reference = float(np.linalg.norm(coordinates[node] - coordinates[peer]) / 10)
                if not np.isfinite(reference) or reference <= 0:
                    raise ValueError("invalid surveyed training distance")
                rows, invalid, count = [], [], 0
                # Skip held rows without parsing their ranges or computing their residuals.
                for index, row in enumerate(islice(reader, start - consumed, stop - consumed), start):
                    if int(row[""]) != index or int(row["anchor_ID_anchor"]) != peer:
                        raise ValueError("source row identity changed")
                    value = float(row["distance"])
                    if np.isfinite(value) and value > 0:
                        rows.append([index, value / 10, reference])
                    else:
                        invalid.append(index)
                    count += 1
                consumed = stop
                if (count != part["recorded_ranges"] or len(rows) != part["valid_ranges"]
                        or invalid != part["invalid_publisher_rows"]):
                    raise ValueError("reserved training counts changed")
                if rows:
                    key = (node, peer, *(group["settings"][k] for k in static_preflight.SETTINGS))
                    if key in cells:
                        raise ValueError("duplicate training link and setting")
                    cells[key] = np.asarray(rows, dtype=float)
    if not cells or sum(map(len, cells.values())) != split["totals"]["calibration"]["valid_ranges"]:
        raise ValueError("incomplete reserved training population")
    return cells


def folds(cells):
    for family in ("link", "setting"):
        group = (lambda k: tuple(sorted(k[:2]))) if family == "link" else (lambda k: k[2:])
        held_groups = sorted({group(k) for k in cells})
        if len(held_groups) < 2:
            raise ValueError("calibration needs multiple training links and settings")
        for held in held_groups:
            yield family, held, {k: v for k, v in cells.items() if group(k) != held}, \
                {k: v for k, v in cells.items() if group(k) == held}


def select_calibration(cells):
    records = []
    for family, held, training, validation in folds(cells):
        for model in calibration.MODELS:
            record = dict(family=family, held=list(held), model=model,
                          training_cells=len(training), validation_cells=len(validation))
            try:
                parameters = calibration.fit_correction(training, model)
                errors = [float(np.mean(np.abs(calibration.corrected(r[:, 1], parameters) - r[:, 2])))
                          for r in validation.values()]
                record.update(parameters=parameters, mae_cm=float(np.mean(errors)), failure=None)
            except ValueError as error:
                record.update(parameters=None, mae_cm=None, failure=str(error))
            records.append(record)
    ranking = []
    for model in calibration.MODELS:
        rows = [r for r in records if r["model"] == model]
        failed = sum(r["failure"] is not None for r in rows)
        metrics = {family: float(np.mean([r["mae_cm"] for r in rows if r["family"] == family]))
                   for family in ("link", "setting")} if not failed else {}
        ranking.append(dict(model=model, failed_folds=failed, **metrics,
                            selection_mae_cm=float(np.mean(list(metrics.values()))) if not failed else None))
    eligible = [r for r in ranking if not r["failed_folds"]]
    if not eligible:
        raise ValueError("no calibration candidate passed all training folds")
    selected = min(eligible, key=lambda r: (r["selection_mae_cm"], calibration.MODELS.index(r["model"])))
    return dict(selected=selected["model"], correction=calibration.fit_correction(cells, selected["model"]),
                selection=ranking, folds=records)


def range_profiles(cells, correction):
    measured, reference, weights = calibration.pooled(cells)
    corrected = calibration.corrected(measured, correction)
    residual = corrected - reference
    scale = float(np.sqrt(weights @ residual ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("no positive training residual scale")
    ade, nll = aac_training.fit_ade_model(corrected, reference)
    return dict(scale_cm=scale, corrected_zero_reports=int(np.sum(corrected == 0)),
        af_range_error=calibration.fit_shape(residual, scale, weights),
        aac_range_error=calibration.fit_shape(residual, aac_training.aac_scale(reference), weights),
        anc_ranging_model=tiers_static_check.fit_anc(cells, correction),
        ade_ranging_model=ade, ade_training_nll_per_report=nll / len(measured))


def prepare():
    if OUTPUT.exists():
        raise ValueError("IDLab training evidence already exists")
    if pilot.audit.uncommitted():
        raise ValueError("commit the fitting recipe before training")
    split = pilot.evidence.read(idlab_split.OUTPUT)
    checked = pilot.evidence.read(idlab_admission.OUTPUT)
    pilot.evidence.official_scorer.require_hash(idlab_admission.OUTPUT, split["admission_sha256"], "IDLab training admission")
    for record in (split, checked):
        for name, expected in record["sources"].items():
            pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab training source")
    surveyed = np.loadtxt(idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")
    if surveyed.shape != (15, 4) or not np.isfinite(surveyed).all() or surveyed[:, 0].tolist() != list(range(1, 16)):
        raise ValueError("unexpected surveyed IDLab coordinates")
    coordinates = {int(r[0]): r[1:] for r in surveyed}
    cells = load_training(split, idlab_admission.FOLDER, coordinates)
    print(f"Fitting from {sum(map(len, cells.values())):,} reserved ranges in {len(cells):,} link settings", flush=True)
    selected = select_calibration(cells)
    print(f"Training-only calibration choice {selected['selected']}", flush=True)
    profiles = range_profiles(cells, selected["correction"])
    import fit_profiles

    sources = [Path(m.__file__).resolve() for m in (calibration, aac_training, tiers_static_check, fit_profiles)]
    sources += [Path(__file__).resolve()]
    result = dict(schema="solid.external.idlab-training.v1", status="range-profiles-fitted-execution-not-frozen",
        source_revision=pilot.audit.revision(),
        split_sha256=pilot.identity(idlab_split.OUTPUT),
        sources={pilot.audit.name(p): pilot.identity(p) for p in sources},
        training_reports=sum(map(len, cells.values())), training_link_settings=len(cells),
        training_links=len({tuple(sorted(k[:2])) for k in cells}), training_settings=len({k[2:] for k in cells}),
        source_units="mm", fitting_units="cm", evaluation_rows_used_for_training=0, positioning_solves=0,
        correction_rule="One shared IDLab gain and offset. Project corrected ranges to nonnegative values identically for every method.",
        selection_rule="Identity, offset and positive-gain affine candidates. Hold out each unordered link across all settings and each setting across all links. Average the two families of held-fold MAEs equally. Failed candidates are ineligible. Exact ties prefer fewer parameters.",
        weighting="Equal weight per observed directed link and radio setting. Equal weight per record within a cell. ADE-BP retains its published independent-report likelihood fit.",
        profile_scope="Shared across radio settings. No node-specific, link-specific or position-lookup correction. AF and AAC retain their existing distinct scale functions. Both sample counts share fitted parameters.",
        availability=None, detection_range_cm=None,
        pending=["Verify attempted links before fitting AF availability and ANC detection range. No Bern parameters are substituted.",
                 "Freeze sample blocks, anchor assignments and scoring weights.",
                 "Finish method-specific numerical preparation and bounded adapter checks before full execution."],
        **selected, profiles=profiles)
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT])
    print(result["selection"], flush=True)


if __name__ == "__main__":
    prepare()
