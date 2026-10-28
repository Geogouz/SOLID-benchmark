"""Training-only sparse IDLab adapter checks, not benchmark or model selection.

Usage: idlab_check.py [prepare|execute]
  prepare writes the check inputs without running a method, the stage a reader's reproduction (SOLID_READER=1) runs
  before idlab_campaign.py; execute, the default, builds the IDLab adapter and also runs and scores the checks.
"""

import argparse
from collections import defaultdict
import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

import aac_check
import calibration
import idlab_admission
import idlab_availability
import idlab_population
import idlab_split
import idlab_training
import pilot
import static_preflight
import tiers_static_check


OUTPUT = pilot.audit.DATA / "idlab-check"
BINARY = pilot.ROOT / "research/target/release/idlab-adapter"
MODES = dict(AF="AnchorFree", AAC="AnchorAided", ANC="AnchorOnly")
DIAMETER_CM = math.hypot(4100, 2600)
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def training_panels(cells, labels, nodes):
    groups = defaultdict(list)
    for label in labels:
        key = tuple(label["settings"][name] for name in static_preflight.SETTINGS[:-1])
        groups[key].append(label)
    eligible = sorted(key for key, rows in groups.items()
        if len(rows) == len(nodes) * (len(nodes) - 1) // 2
        and all(row["observed"] is not None for row in rows))
    if len(eligible) < 4:
        raise ValueError("need four configurations with known training availability")
    selected = [eligible[i * (len(eligible) - 1) // 3] for i in range(4)]
    pools, values = defaultdict(lambda: defaultdict(list)), {}
    for key, rows in cells.items():
        node, peer, *settings = key
        if settings[-1] != settings[-2]:
            raise ValueError("unequal power directions cannot be pooled")
        setting = tuple(settings[:-1])
        if setting not in selected:
            continue
        for row in rows:
            identity = (node, int(row[0]))
            if identity in values:
                raise ValueError("duplicate publisher identity")
            values[identity] = float(row[1])
            pools[setting][tuple(sorted((node, peer)))].append(list(identity))
    panels = []
    for key in selected:
        settings = dict(zip(static_preflight.SETTINGS, (*key, key[-1]), strict=True))
        expected = {tuple(r["nodes"]) for r in groups[key] if r["observed"]}
        if set(pools[key]) != expected:
            raise ValueError("reserved reports and training availability disagree")
        links = [dict(nodes=list(pair), records=sorted(rows, key=lambda row:
            idlab_population.digest([idlab_population.SEED, "records", settings, pair, row])))
            for pair, rows in sorted(pools[key].items())]
        panels.append(idlab_population.panel(dict(nodes=nodes),
            dict(settings=settings, links=links, panels=1), 0))
    return panels, values, len(eligible)


def operating_area(anchors):
    return {f"{bound}_{axis}_cm": operation(a["position"][axis] for a in anchors) + sign * DIAMETER_CM
            for axis in ("x", "y")
            for bound, operation, sign in (("minimum", max, -1), ("maximum", min, 1))}


def unit(panel, sid, samples, mode, known, values, fitted, availability):
    nodes = panel["nodes"]
    if mode not in MODES or samples not in (1, 10) or not set(known) <= set(nodes):
        raise ValueError("invalid check roles")
    if bool(known) != (mode != "AF") or len(nodes) - len(known) < 2:
        raise ValueError("invalid anchor count")
    anchors = [dict(id=n, position=dict(x=float(p[0]), y=float(p[1])), height_cm=float(p[2]))
               for n, p in sorted(known.items())]
    agents, observations = [n for n in nodes if n not in known], []
    for link in panel["links"]:
        a, b = link["nodes"]
        if (a in known and b in known) or (mode == "ANC" and a not in known and b not in known):
            continue
        if a in known:
            a, b = b, a
        identities = link["one" if samples == 1 else "up_to_ten"]
        if not identities:
            continue
        if len(identities) > samples or len(set(map(tuple, identities))) != len(identities):
            raise ValueError("invalid real-report allowance")
        distances = calibration.corrected([values[tuple(i)] for i in identities], fitted["correction"])
        observations.append(dict(**{"from": a, "to": b}, to_is_anchor=b in known, tof_ns=[],
            calibrated_samples=[dict(distance_cm=float(d), scale_cm=fitted["profiles"]["scale_cm"]) for d in distances]))
    area = operating_area(anchors) if known else None
    low = [area[f"minimum_{axis}_cm"] for axis in ("x", "y")] if area else [-DIAMETER_CM / 2] * 2
    high = [area[f"maximum_{axis}_cm"] for axis in ("x", "y")] if area else [DIAMETER_CM / 2] * 2
    initial = np.random.default_rng(20260918 + sid).uniform(low, high, (len(agents), 2))
    result = dict(id=sid, samples=samples, area=area, deployment_profile=None,
        problem=dict(agent_ids=agents, anchors=anchors, mode=MODES[mode], observations=observations,
                     initial_positions=[dict(x=float(x), y=float(y)) for x, y in initial]))
    if mode == "ANC":
        result["deployment_profile"] = dict(area=copy.deepcopy(area),
            ranging_model=copy.deepcopy(fitted["profiles"]["anc_ranging_model"]),
            detection_range_cm=availability["fitted"]["detection_range_cm"])
    return result


def af_baselines():
    if pilot.READER:
        settings, sources = pilot.shipped("idlab-settings.json")
        return settings["af_baselines"], sources
    roster = read(pilot.evidence.retained_data.EVIDENCE)
    settings, sources = [], {}
    for method in ("jin-2021", "fnl"):
        row = next(r for r in roster["methods"] if r["id"] == method)
        record = row["results"]["AF"]
        path = pilot.evidence.retained_data.run_directory(method, "AF", record) / "benchmark.json"
        pilot.evidence.official_scorer.require_hash(path, record["benchmark_report_sha256"], "native AF settings")
        report = read(path)
        config = dict(report["config"], dimension_policy="planar")
        if config["max_time_micros"] != 0 or config["aggregation"] != "Median":
            raise ValueError("unexpected retained baseline settings")
        settings.append(dict(method=method, method_config=report["method_config"], solve_config=config))
        sources[str(path)] = digest(path)
    return settings, sources


def prepare():
    if OUTPUT.exists():
        raise ValueError("IDLab check already retained")
    artifacts = [idlab_admission.OUTPUT, idlab_split.OUTPUT, idlab_training.OUTPUT, idlab_availability.record_path()]
    sources = {str(p): digest(p) for p in artifacts}
    for path in artifacts:
        # the shipped availability record names the private files it was fitted from, which a reader lacks
        if path == idlab_availability.SHIPPED:
            continue
        for name, expected in read(path)["sources"].items():
            pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab check source")
            sources[str(pilot.audit.located(name))] = expected
    split, fitted = map(read, artifacts[1:3])
    availability = idlab_availability.record()
    pilot.evidence.official_scorer.require_hash(idlab_admission.OUTPUT, split["admission_sha256"], "IDLab admission")
    pilot.evidence.official_scorer.require_hash(idlab_split.OUTPUT, fitted["split_sha256"], "IDLab training split")
    coordinates = {int(r[0]): r[1:] for r in np.loadtxt(idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")}
    cells = idlab_training.load_training(split, idlab_admission.FOLDER, coordinates)
    panels, values, eligible = training_panels(cells, availability["labels"], split["nodes"])
    for mode in MODES:
        baselines, native_sources = (af_baselines() if mode == "AF" else
                                    tiers_static_check.anchored_baselines(mode, fitted["profiles"]))
        sources.update(native_sources)
        profile = None if mode == "ANC" else dict(range_error=fitted["profiles"][f"{mode.lower()}_range_error"],
            availability=availability["fitted"]["availability"] if mode == "AF" else None)
        request = dict(profile=profile, scale_cm=fitted["profiles"]["scale_cm"], baselines=baselines, units=[])
        reference = dict(units=[], coordinates_cm={})
        for index, panel in enumerate(panels):
            for k in ((0,) if mode == "AF" else (1, 4, 8, 13)):
                sid = index * 16 + k + 1
                known = {n: coordinates[n] / 10 for n in panel["anchor_order"][:k]}
                for samples in (1, 10):
                    request["units"].append(unit(panel, sid, samples, mode, known, values, fitted, availability))
                reference["units"].append(dict(id=sid, acquisition=str(index), reference=str(sid), anchors=k))
                reference["coordinates_cm"][str(sid)] = {str(n): (coordinates[n][:2] / 10).tolist()
                    for n in split["nodes"] if n not in known}
        write(OUTPUT / mode / "request.json", request)
        write(OUTPUT / mode / "reference.json", reference)
    write(OUTPUT / "records.json", panels)
    for p in pilot.present([Path(__file__).resolve(), pilot.HERE / "idlab_predict.rs", pilot.HERE / "README.md",
              pilot.HERE / "idlab_population.py", pilot.HERE / "pilot.py"]):
        sources[str(p)] = digest(p)
    write(OUTPUT / "inputs.json", dict(sources=sources,
        files={str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.rglob("*.json")},
        scope="Training-only correctness and timing. Not held-out accuracy or parameter selection.",
        configurations=4, eligible_training_configurations=eligible, inputs=dict(AF=8, AAC=32, ANC=32),
        configuration_selection="Equally spaced sorted configurations with all 105 training pair labels known. No selection by residuals or position accuracy.",
        roles="First 1, 4, 8 or 13 nodes of the population identity-hash order. Same roles and starts in both report tracks.",
        area="Intersection of known-anchor coordinate bounds plus or minus the published 41 m by 26 m floor diagonal. Conservative declared support, not a measured wall outline.",
        training_overlap="Profiles and checks use reserved training records. Their accuracy is not a generalization estimate.",
        pending=["Training-only native parameter sensitivity and spatial support review.",
                 "Separate check of the retained Jin AAC component-recovery adaptation.",
                 "Final inputs, settings, execution budget and source license confirmation."],
        evaluation_positioning_inputs=0, full_campaign_started=False))


def timing_rows(scores, records, mode):
    rows = []
    for method in (pilot.LABELS if mode == "AF" else aac_check.LABELS):
        selected = [r for r in scores if r["method"] == method]
        estimates = [r["estimate"] for r in records if r["method"] == method]
        collapsed = sum(e is not None and all(p is not None and all(v is not None for v in p) for p in e["positions"])
                        and len({tuple(p) for p in e["positions"]}) == 1 for e in estimates)
        rows.append(dict(mode=mode, method=method, inputs=len(selected),
            failed_agents=sum(r["failed_agents"] for r in selected),
            expected_agents=sum(r["expected_agents"] for r in selected),
            inputs_with_missing_estimates=sum(r["failed_agents"] > 0 for r in selected),
            collapsed_outputs=collapsed,
            nonconverged_outputs=sum(e is not None and not e["metrics"]["converged"] for e in estimates),
            mean_ms=float(np.mean([r["elapsed_ms"] for r in selected]))))
    return rows


def execute():
    if OUTPUT.exists():
        raise ValueError("IDLab check already retained")
    if pilot.audit.uncommitted():
        raise ValueError("commit the adapter and check recipe before execution")
    revision = pilot.audit.revision()
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLID_")}
    subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", "research/Cargo.toml",
                    "-p", "solid-cli", "--bin", "idlab-adapter"], cwd=pilot.ROOT, env=env, check=True)
    if subprocess.check_output([str(BINARY), "--identity"], text=True).strip() != pilot.isolation.source_identity():
        raise ValueError("binary source identity mismatch")
    prepare()
    write(OUTPUT / "freeze.json", dict(source_revision=revision, binary_sha256=digest(BINARY),
        inputs_sha256=digest(OUTPUT / "inputs.json"), scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        host=platform.platform(), workers=1, no_warmup=True, full_campaign=False))
    scorer, summaries = pilot.evidence.load_scorer(), []
    for mode in MODES:
        folder = OUTPUT / mode
        with (folder / "run.log").open("x") as log:
            subprocess.run([str(BINARY), str(folder / "request.json"), str(folder / "optimizer.jsonl")],
                           cwd=pilot.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        request, reference = read(folder / "request.json"), read(folder / "reference.json")
        records = [json.loads(line) for line in (folder / "optimizer.jsonl").read_text().splitlines()]
        write(folder / "predictions.json", records)
        scores = pilot.score_records(request, reference, records, scorer,
                                     methods=pilot.LABELS if mode == "AF" else aac_check.LABELS)
        write(folder / "scores.json", scores)
        summaries.extend(timing_rows(scores, records, mode))
        print(json.dumps(dict(mode=mode, inputs=len(request["units"]), outputs=len(records))), flush=True)
    write(OUTPUT / "summary.json", dict(status="training-adapter-check-complete", methods=summaries,
        inputs=72, method_outputs=sum(r["inputs"] for r in summaries),
        evaluation_positioning_inputs=0, full_campaign_started=False,
        note="Timings are small training checks, not a speed ranking or held-out accuracy. No setting was selected from these outputs."))
    files = sorted(p for p in OUTPUT.rglob("*") if p.is_file())
    write(OUTPUT / "manifest.json", dict(files={str(p.relative_to(OUTPUT)): digest(p) for p in files}))
    pilot.register_files([*files, OUTPUT / "manifest.json"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("prepare", "execute"), default="execute")
    {"prepare": prepare, "execute": execute}[parser.parse_args().command]()
