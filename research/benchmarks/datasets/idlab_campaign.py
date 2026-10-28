"""Freeze the prepared IDLab comparison without fitting or selecting estimators.

A reader's reproduction (SOLID_READER=1) freezes without the adapter checks, the selection of STRONG's threshold and the
numerics check, research runs of the IDLab adapters, and builds no adapter. The settings of every mode are those of the
paper's campaign, settings/data/idlab-campaign-settings.json, which must hold the settings of the check inputs that
idlab_check.py prepare writes (tiers_campaign.shipped_settings)."""

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess

import numpy as np

import idlab_check as checked
import idlab_population as population
import idlab_settings
import tiers_campaign

pilot = checked.pilot
OUTPUT = pilot.audit.DATA / "idlab-campaign"
WORKERS = 12
LABELS = dict(AF=pilot.LABELS, AAC=checked.aac_check.LABELS, ANC=checked.aac_check.LABELS)
read, write, digest = checked.read, checked.write, checked.digest
encoded, value_hash = tiers_campaign.encoded, tiers_campaign.value_hash


def catalog(base):
    for mode in ("AF", "ANC", "AAC"):
        for index, panel in enumerate(base["panels"]):
            for count in ((0,) if mode == "AF" else range(1, len(panel["nodes"]) - 1)):
                yield dict(key=f"{mode}-{index:04d}-{count:02d}", mode=mode, panel=index,
                    configuration=panel["configuration"], anchors=count,
                    units=2 if mode == "AF" else 2 * len(panel["nodes"]))


def batch_inputs(base, batch):
    panel = base["panels"][batch["panel"]]
    mode, count = batch["mode"], batch["anchors"]
    request = dict(base["settings"][mode], units=[])
    reference = dict(units=[], coordinates_cm={})
    roles = len(panel["nodes"]) * (len(panel["nodes"]) - 2)
    assignments = [(0, [])] if mode == "AF" else [
        (i + 1, anchors) for i, anchors in enumerate(population.anchor_subsets(panel["anchor_order"]))
        if len(anchors) == count]
    for role, anchors in assignments:
        sid = batch["panel"] * (roles + 1) + role + 1
        known = {n: base["coordinates_cm"][str(n)] for n in anchors}
        for samples in (1, 10):
            request["units"].append(checked.unit(panel, sid, samples, mode, known,
                base["values"], base["fitted"], base["availability"]))
        reference["units"].append(dict(id=sid, reference=str(sid),
            acquisition=f'{batch["configuration"]}/anchors-{count}', panel=batch["panel"],
            anchors=count, anchor_ids=anchors))
        reference["coordinates_cm"][str(sid)] = {str(n): base["coordinates_cm"][str(n)][:2]
            for n in panel["nodes"] if n not in known}
    if len(request["units"]) != batch["units"] or not 0 < batch["units"] <= 32:
        raise ValueError("IDLab batch population changed")
    return request, reference


def load_values(selected, folder):
    wanted = defaultdict(set)
    for configuration in selected["configurations"]:
        for link in configuration["links"]:
            for node, row in link["records"]:
                wanted[node].add(row)
    values = []
    for node, rows in sorted(wanted.items()):
        with (folder / f"office_lab_node_{node:02d}_short.csv").open(newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if index not in rows:
                    continue
                cm = float(row["distance"]) / 10
                if int(row[""]) != index or not np.isfinite(cm) or cm <= 0:
                    raise ValueError("frozen evaluation record changed")
                values.append([node, index, cm])
    if len(values) != sum(map(len, wanted.values())):
        raise ValueError("evaluation source records are missing")
    return values


def hydrate(base):
    result = dict(base)
    result["values"] = {(node, row): value for node, row, value in base["values"]}
    if len(result["values"]) != len(base["values"]):
        raise ValueError("duplicate evaluation source identity")
    return result


def source_hashes():
    names = ("idlab_campaign.py", "idlab_execution.py", "idlab_check.py", "idlab_population.py",
        "idlab_predict.rs", "pilot.py", "calibration.py", "tiers_campaign.py", "tiers_execution.py")
    paths = [pilot.HERE / name for name in names]
    paths += [pilot.HERE.parent / "bern/score_archives.py"]
    return {pilot.audit.name(p): digest(p) for p in pilot.present(paths)}


def freeze():
    if OUTPUT.exists():
        raise ValueError("IDLab campaign already frozen")
    if pilot.audit.uncommitted():
        raise ValueError("commit the campaign before freezing")
    revision = pilot.audit.revision()
    sources = {}
    # the retained adapter runs, which a reader's reproduction replaces by the check inputs of idlab_check.prepare
    for folder in () if pilot.READER else (checked.OUTPUT, idlab_settings.OUTPUT, checked.OUTPUT / "numerics"):
        sources[str(folder / "manifest.json")] = tiers_campaign.verify_manifest(folder)
    if pilot.READER:
        sources[str(checked.OUTPUT / "inputs.json")] = tiers_campaign.verify_inputs(checked.OUTPUT)
    else:
        previous = read(checked.OUTPUT / "freeze.json")
        for name, expected in read(checked.OUTPUT / "inputs.json")["sources"].items():
            path = pilot.audit.located(name)
            if path == pilot.HERE / "README.md":
                blob = subprocess.check_output(["git", "show", f'{previous["source_revision"]}:{path.relative_to(pilot.ROOT)}'],
                                               cwd=pilot.ROOT)
                if hashlib.sha256(blob).hexdigest() != expected:
                    raise ValueError("checked IDLab protocol changed")
            else:
                pilot.evidence.official_scorer.require_hash(path, expected, "checked IDLab input or source")
    for path in (checked.idlab_admission.OUTPUT, checked.idlab_split.OUTPUT,
                 checked.idlab_training.OUTPUT, checked.idlab_availability.record_path(), population.OUTPUT):
        sources[str(path)] = digest(path)
        if path == checked.idlab_availability.SHIPPED:
            continue
        for name, expected in read(path)["sources"].items():
            pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab source")
            sources[str(pilot.audit.located(name))] = expected
    selected = json.loads(gzip.decompress(population.INPUTS.read_bytes()))
    counts = population.summarize(selected)
    recorded = read(population.OUTPUT)
    if any(recorded[key] != json.loads(json.dumps(value)) for key, value in counts.items()):
        raise ValueError("selected population differs from its manifest")
    if not pilot.READER:
        choice = read(idlab_settings.OUTPUT / "summary.json")
        if choice["evaluation_solves"] != 0 or choice["training_solves"] != 3584:
            raise ValueError("native settings preparation is incomplete")
    settings = {mode: {k: v for k, v in read(checked.OUTPUT / mode / "request.json").items() if k != "units"}
                for mode in LABELS}
    if pilot.READER:
        # the settings of the paper's campaign, which hold those of the check inputs, in place of the STRONG and ADE-BP
        # settings that the training selection of idlab_settings.py confirmed
        settings, shipped = tiers_campaign.shipped_settings("idlab-campaign-settings.json", settings)
        sources.update(shipped)
    for mode in () if pilot.READER else ("AAC", "ANC"):
        strong = next(b for b in settings[mode]["baselines"] if b["method"] == "strong")
        if strong != choice["selected_setting"]:
            raise ValueError("checked STRONG settings differ from the training selection")
        ade = next(b["method_config"] for b in settings[mode]["baselines"] if b["method"] == "ade-bp-2015")
        if (ade["particles"] != choice["ade"]["particles"]
                or ade["importance_iterations"] != choice["ade"]["importance_iterations"]
                or ade["iterations"] != choice["ade"]["rounds"][mode]):
            raise ValueError("ADE numerical budget changed")
    panels = [dict(population.panel(selected, configuration, i), configuration=c)
              for c, configuration in enumerate(selected["configurations"])
              for i in range(configuration["panels"])]
    fitted = read(checked.idlab_training.OUTPUT)
    base = dict(panels=panels, settings=settings,
        fitted={key: fitted[key] for key in ("correction", "profiles")},
        availability=checked.idlab_availability.record(),
        coordinates_cm={str(int(r[0])): (r[1:] / 10).tolist()
            for r in np.loadtxt(checked.idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")},
        values=load_values(selected, checked.idlab_admission.FOLDER))
    batches = list(catalog(base))
    inputs = sum(b["units"] for b in batches)
    outputs = sum(b["units"] * len(LABELS[b["mode"]]) for b in batches)
    # the outputs of the paper's campaign count the methods its freeze names, SOLID's two among them, which LABELS no
    # longer lists, so a reader's reproduction checks the inputs and the batches of the population alone
    if (inputs, len(batches)) != (2117656, 73116) or not pilot.READER and outputs != 10582864:
        raise ValueError("full IDLab denominator changed")
    # the runs of a reader's reproduction take the adapters of its own build, so it builds and freezes no executable
    if not pilot.READER:
        env = {k: v for k, v in os.environ.items() if not k.startswith("SOLID_")}
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", "research/Cargo.toml",
                        "-p", "solid-cli", "--bin", "idlab-adapter"], cwd=pilot.ROOT, env=env, check=True)
        if subprocess.check_output([str(checked.BINARY), "--identity"], text=True).strip() != pilot.isolation.source_identity():
            raise ValueError("IDLab executable source identity differs")
    OUTPUT.mkdir()
    if not pilot.READER:
        (OUTPUT / "bin").mkdir()
        shutil.copy2(checked.BINARY, OUTPUT / "bin/idlab-adapter")
    for name, value in (("base.json.gz", base), ("catalog.json.gz", batches)):
        (OUTPUT / name).write_bytes(gzip.compress(encoded(value), mtime=0))
    # the planning estimate times the solves of the adapter checks, which a reader's reproduction does not run
    hours = None if pilot.READER else sum(2 * counts["inputs_per_sample_track"][r["mode"]] * r["mean_ms"] / 3_600_000
                                          for r in read(checked.OUTPUT / "summary.json")["methods"])
    write(OUTPUT / "freeze.json", dict(source_revision=revision, source_identity=pilot.isolation.source_identity(), sources=sources, source_hashes=source_hashes(),
        protocol_sha256=pilot.document(pilot.HERE / "README.md"), scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        methods={mode: list(labels) for mode, labels in LABELS.items()}, inputs=inputs, outputs=outputs,
        configurations=counts["radio_settings"], panels=counts["panels"],
        population_per_sample=counts["inputs_per_sample_track"], source_records=len(base["values"]),
        models_fitted=0, evaluation_rows_used_for_training=0, host=platform.platform(),
        construction="Hashed compact base and committed constructor determine every request and reference. Each checkpoint retains their derived hashes.",
        weights="Equal configuration and anchor-count cells. Equal panels and cyclic assignments within each cell.",
        initializations="Seed 20260918 plus the fixed panel and role ID. Both sample tracks and both anchored modes share starts.",
        budget=dict(workers=WORKERS, checkpoints=len(batches), one_worker_hours=hours,
                    ideal_parallel_hours=None if hours is None else hours / WORKERS, per_solve_wall_clock_cutoff=False)))
    paths = [p for p in OUTPUT.rglob("*") if p.is_file()]
    write(OUTPUT / "manifest.json", dict(files={str(p.relative_to(OUTPUT)): digest(p) for p in paths}))
    pilot.register_files(paths + [OUTPUT / "manifest.json"])
    print(json.dumps(dict(inputs=inputs, outputs=outputs, workers=WORKERS, batches=len(batches))), flush=True)


def verify():
    tiers_campaign.verify_manifest(OUTPUT)
    frozen = read(OUTPUT / "freeze.json")
    for name, expected in {**frozen["sources"], **frozen["source_hashes"]}.items():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab frozen source")
    if (frozen["methods"] != {m: list(labels) for m, labels in LABELS.items()}
            or frozen["budget"]["workers"] != WORKERS
            or frozen["scorer_sha256"] != pilot.evidence.official_scorer.SCORER_SHA256):
        raise ValueError("IDLab methods, workers or scorer changed")
    # a binary built before the source identity embeds its commit
    if subprocess.check_output([str(OUTPUT / "bin/idlab-adapter"), "--identity"], text=True).strip() != frozen.get("source_identity", frozen["source_revision"]):
        raise ValueError("frozen IDLab executable revision changed")
    base = hydrate(json.loads(gzip.decompress((OUTPUT / "base.json.gz").read_bytes())))
    batches = json.loads(gzip.decompress((OUTPUT / "catalog.json.gz").read_bytes()))
    if (batches != list(catalog(base)) or sum(b["units"] for b in batches) != frozen["inputs"]
            or sum(b["units"] * len(LABELS[b["mode"]]) for b in batches) != frozen["outputs"]):
        raise ValueError("IDLab execution population changed")
    return frozen, base, batches


if __name__ == "__main__":
    freeze()
