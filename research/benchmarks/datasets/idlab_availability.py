"""Fit native availability families from verified IDLab training pools."""

from collections import Counter
import gzip
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

import idlab_admission
import idlab_attempts
import idlab_receivers
import idlab_split
import idlab_training
import pilot
import static_preflight
import fit_cooperative as joint_fit


OUTPUT = idlab_split.OUTPUT.with_name("idlab-availability.json")
# The 34 GB of original tag and receiver logs that this stage streams feed this record alone, so a reader's reproduction
# (SOLID_READER=1) does not run it and reads the record from settings/data, byte for byte the one this stage wrote.
SHIPPED = pilot.SETTINGS / "idlab-availability.json.gz"
SETTINGS = static_preflight.SETTINGS[:-1]


def training_labels(split, tags, coordinates):
    """Unknown reserved pools stay unknown, even when the other direction is absent."""
    nodes = sorted(split["nodes"])
    groups = {}
    for group in split["groups"]:
        settings = group["settings"]
        if settings["txpower_tag"] != settings["txpower_anchor"]:
            raise ValueError("opposite powers cannot share a physical-link label")
        key = (group["node"], group["peer"], *(settings[k] for k in SETTINGS))
        if key in groups:
            raise ValueError("duplicate training link and setting")
        groups[key] = group["calibration"]
    settings = sorted({key[2:] for key, part in groups.items() if part["valid_ranges"] > 0})
    if not settings or sorted(tags) != nodes or sorted(coordinates) != nodes:
        raise ValueError("incomplete training coordinates or acquisition schedule")
    opportunities = {}
    for node, tag in sorted(tags.items()):
        counts = Counter((row[1], row[3]) for row in tag["records"] if row[1] != node)
        for (peer, block), count in counts.items():
            setting = tag["blocks"][block]["settings"]
            key = (node, peer, *(setting[k] for k in SETTINGS))
            if peer not in nodes or key in opportunities:
                raise ValueError("unknown peer or repeated acquisition setting")
            # Reserve a nonempty prefix for blocks that supplied no published row.
            opportunities[key] = count // 2
    if set(groups) - set(opportunities):
        raise ValueError("published training pool lacks a verified acquisition opportunity")
    rows = []
    for setting in settings:
        for a, b in combinations(nodes, 2):
            parts, prefixes = [], []
            for left, right in ((a, b), (b, a)):
                key = (left, right, *setting)
                if opportunities.get(key, 0) < 1:
                    raise ValueError("missing reserved acquisition opportunity")
                part = groups.get(key)
                # A singleton assigned to evaluation is not a training failure.
                parts.append(True if part and part["valid_ranges"] > 0 else None if part else False)
                prefixes.append(opportunities[key] if part is None else None)
            observed = True if True in parts else None if None in parts else False
            distance = float(np.linalg.norm((coordinates[a] - coordinates[b])[:2]) / 10)
            if not np.isfinite(distance) or distance <= 0:
                raise ValueError("invalid horizontal training distance")
            rows.append(dict(nodes=[a, b], settings=dict(zip(SETTINGS, setting)), distance_cm=distance,
                observed=observed, directions=parts, absent_pool_prefix_records=prefixes))
    return rows


def fit_detection(distances_cm, observed):
    def objective(log_range):
        probability = np.clip(np.exp(-distances_cm ** 2 / (2 * np.exp(2 * log_range))), 1e-12, 1 - 1e-12)
        return float(-np.mean(np.where(observed, np.log(probability), np.log1p(-probability))))

    result = minimize_scalar(objective, bounds=(np.log(1.), np.log(100_000.)),
                             method="bounded", options={"xatol": 1e-12})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("native ANC detection fit failed")
    return float(np.exp(result.x))


def fit_models(rows):
    if not rows or any(row["observed"] is None for row in rows):
        raise ValueError("fit requires known training availability labels")
    distances = np.array([row["distance_cm"] for row in rows])
    observed = np.array([row["observed"] for row in rows], dtype=bool)
    if (not np.isfinite(distances).all() or (distances <= 0).any()
            or not observed.any() or observed.all()):
        raise ValueError("availability fitting needs finite distances and both outcomes")
    intercept, slope = joint_fit.availability_fit(distances / 100, observed)
    return dict(availability=dict(intercept=intercept, slope=slope),
                detection_range_cm=fit_detection(distances, observed),
                constant_probability=float(observed.mean()))


def scores(rows, fitted):
    distances = np.array([row["distance_cm"] for row in rows])
    observed = np.array([row["observed"] for row in rows], dtype=bool)
    parameters = fitted["availability"]
    probabilities = dict(af=expit(parameters["intercept"] - parameters["slope"] * distances / 100),
        anc=np.exp(-distances ** 2 / (2 * fitted["detection_range_cm"] ** 2)),
        constant=np.full(len(rows), fitted["constant_probability"]))
    result = {}
    for name, probability in probabilities.items():
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        result[name] = dict(log_loss=float(-np.mean(np.where(observed, np.log(probability), np.log1p(-probability)))),
                            brier=float(np.mean((probability - observed) ** 2)))
    return result


def validate_models(rows):
    known = [row for row in rows if row["observed"] is not None]
    folds = []
    for family in ("link", "setting"):
        key = (lambda r: tuple(r["nodes"])) if family == "link" else (lambda r: tuple(r["settings"].values()))
        for held in sorted({key(row) for row in known}):
            training = [row for row in known if key(row) != held]
            validation = [row for row in known if key(row) == held]
            fitted = fit_models(training)
            folds.append(dict(family=family, held=list(held), training_pools=len(training),
                validation_pools=len(validation), fitted=fitted, scores=scores(validation, fitted)))
    summary = {family: {model: {metric: float(np.mean([row["scores"][model][metric]
                for row in folds if row["family"] == family])) for metric in ("log_loss", "brier")}
                for model in ("af", "anc", "constant")} for family in ("link", "setting")}
    return dict(fitted=fit_models(known), validation=summary, folds=folds)


def record_path():
    """The availability record that the later stages read, the shipped one for a reader."""
    return SHIPPED if pilot.READER else OUTPUT


def record():
    path = record_path()
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix == ".gz" else pilot.evidence.read(path)


def prepare():
    if OUTPUT.exists():
        raise ValueError("IDLab availability evidence already exists")
    if pilot.audit.uncommitted():
        raise ValueError("commit the availability recipe before fitting")
    artifacts = (idlab_admission.OUTPUT, idlab_split.OUTPUT, idlab_training.OUTPUT,
                 idlab_attempts.OUTPUT, idlab_receivers.OUTPUT)
    records = {path: pilot.evidence.read(path) for path in artifacts}
    for record in records.values():
        for name, expected in record["sources"].items():
            pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab availability source")
    split, attempts, receivers = (records[path] for path in (idlab_split.OUTPUT, idlab_attempts.OUTPUT, idlab_receivers.OUTPUT))
    if (attempts["status"] != "complete-original-tag-audit" or receivers["status"] != "complete-receiver-audit"
            or attempts["split_sha256"] != pilot.identity(idlab_split.OUTPUT)
            or receivers["sender_audit_sha256"] != pilot.identity(idlab_attempts.OUTPUT)):
        raise ValueError("acquisition audit is incomplete or belongs to another partition")
    tags = {}
    for node in attempts["nodes"]:
        with gzip.open(pilot.audit.located(node["records_path"]), "rt") as stream:
            tags[node["node"]] = json.load(stream)
    surveyed = np.loadtxt(idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")
    rows = training_labels(split, tags, {int(row[0]): row[1:] for row in surveyed})
    fitted = validate_models(rows)
    sources = [Path(__file__).resolve(), Path(joint_fit.__file__).resolve()]
    result = dict(schema="solid.external.idlab-availability.v1", status="availability-fitted-execution-not-frozen",
        source_revision=pilot.audit.revision(),
        sources={pilot.audit.name(path): pilot.identity(path) for path in (*artifacts, *sources)},
        training_settings=len({tuple(row["settings"].values()) for row in rows}),
        known_pools=sum(row["observed"] is not None for row in rows),
        present_pools=sum(row["observed"] is True for row in rows),
        absent_pools=sum(row["observed"] is False for row in rows),
        unknown_pools=sum(row["observed"] is None for row in rows),
        evaluation_rows_used_for_training=0, positioning_solves=0,
        outcome="At least one valid published reading in the reserved physical-link training pool. Not per-packet radio detection.",
        eligibility="Use configurations with a valid reserved training report. Unknown reserved pools are excluded from fitting, never labelled failed. Evaluation inputs remain unchanged.",
        opportunity_rule="Both directions must have a verified nonempty acquisition prefix. For blocks absent from the compact source, reserve the first half of sender records as training opportunities. Counts are not likelihood weights or invented failed attempts.",
        weighting="Equal weight per known unordered physical link and configuration. No weighting by report count or evaluation panel count.",
        profile_scope="One shared logistic AF model and native radial ANC model. Both sample tracks retain short links and use the same availability parameters. AAC remains range-only.",
        validation_rule="Hold out each physical link across configurations and each configuration across links. Constant probability is a training-fitted diagnostic, not a solver alternative or model-selection rule.",
        limitations="The row split conditions on recorded successes. Availability concerns this publication and pool construction, including acquisition and processing omissions. It is not a causal distance-failure model or unseen-layout validation.",
        labels=rows, **fitted)
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT])
    print({key: result[key] for key in ("training_settings", "known_pools", "present_pools", "absent_pools", "unknown_pools", "fitted", "validation")}, flush=True)


if __name__ == "__main__":
    prepare()
