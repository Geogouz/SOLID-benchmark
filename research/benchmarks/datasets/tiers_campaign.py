"""Check and freeze the complete common-block TIERS comparison.

A reader's reproduction (SOLID_READER=1) freezes without the bounded checks, research runs of the pilot adapter. The
check inputs that tiers_static_check.py prepare writes stand in for the checked demonstrations, and the settings of
every track and mode are those of the paper's campaign, settings/data/tiers-campaign-settings.json, which must hold the
settings of those inputs (shipped_settings)."""

import argparse
import copy
import gzip
import hashlib
from itertools import combinations
import json
import math
import platform
import shutil
import subprocess
import time

import aac_check
import aac_training
import pilot
import static_preflight
import tiers_static_check as checked

OUTPUT = pilot.audit.DATA / "tiers-static-campaign"
CHECKS = OUTPUT / "checks"
WORKERS = 4
MODES = checked.MODES
TRACKS = checked.TRACKS
ROLES = tuple(c for k in range(1, 7) for c in combinations(range(1, 9), k))
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def verify_manifest(folder):
    manifest = folder / "manifest.json"
    for name, expected in read(manifest)["files"].items():
        pilot.evidence.official_scorer.require_hash(folder / name, expected, "TIERS evidence")
    return digest(manifest)


def verify_inputs(folder):
    """The inputs of a check that its prepare stage wrote, against the hashes of its inputs.json, the record a reader's
    reproduction holds in place of the manifest of the check's runs."""
    inputs = folder / "inputs.json"
    for name, expected in read(inputs)["files"].items():
        pilot.evidence.official_scorer.require_hash(folder / name, expected, "check input")
    return digest(inputs)


def holds(paper, mine):
    """Whether the JSON value paper holds mine: equal values, lists of equal length whose items hold mine's, and every key
    of a mapping of mine in paper's with a value that holds it. paper may hold keys that mine lacks."""
    if isinstance(mine, dict):
        return isinstance(paper, dict) and all(key in paper and holds(paper[key], value) for key, value in mine.items())
    if isinstance(mine, list):
        return isinstance(paper, list) and len(paper) == len(mine) and all(map(holds, paper, mine))
    return type(paper) is type(mine) and paper == mine


def shipped_settings(name, prepared):
    """The settings of the paper's campaign base, those of every request of its checks without the units, from the
    extract settings/data/<name> that a reader's reproduction (SOLID_READER=1) reads in place of the checked requests,
    and the extract's provenance {path: SHA-256}. They must hold the settings of the reader's own check inputs, prepared,
    and may add only what the code that built the paper's campaign wrote and the current code does not."""
    extract, sources = pilot.shipped(name)
    settings = extract["settings"]
    if set(settings) != set(prepared) or not all(holds(settings[key], value) for key, value in prepared.items()):
        raise ValueError(f"the settings of the check inputs differ from those of the paper's campaign in settings/data/{name}")
    return settings, sources


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def corrected_anchored(track, mode):
    original = read(checked.OUTPUT / f"{track}-{mode}/request.json")
    source = read(checked.OUTPUT / f"{track}-AF/request.json")
    units = {(u["id"], u["samples"]): u for u in source["units"]}
    result = copy.deepcopy(original)
    for unit in result["units"]:
        seed_unit = units[unit["id"] // 10, unit["samples"]]
        known = {a["id"]: [a["position"]["x"], a["position"]["y"], a["height_cm"]]
                 for a in unit["problem"]["anchors"]}
        rebuilt, _ = checked.anchored_unit(seed_unit, list(known), known, mode, unit.get("deployment_profile"))
        unit["problem"]["initial_positions"] = rebuilt["problem"]["initial_positions"]
    return result


def summarize_batch(folder, request, reference, records, labels, track, mode, seconds):
    scores = pilot.score_records(request, reference, records, pilot.evidence.load_scorer(), labels)
    metadata = {r["id"]: r for r in reference["units"]}
    for row in scores:
        row["anchors"] = metadata[row["id"]]["anchors"]
    write(folder / "scores.json", scores)
    write(folder / "summary.json", pilot.summarize(scores, pilot.evidence.load_scorer(), labels,
        mode=mode, colors=pilot.COLORS))
    return dict(track=track, mode=mode, inputs=len(request["units"]), outputs=len(records),
        failed_agents=sum(r["failed_agents"] for r in scores), expected_agents=sum(r["expected_agents"] for r in scores),
        collapsed_outputs=sum(r["estimate"] is not None and len({tuple(p) for p in r["estimate"]["positions"]}) == 1 for r in records),
        seconds=seconds)


def check():
    if pilot.READER:
        raise ValueError("the bounded checks are research runs of the pilot adapter, and a reader's reproduction freezes the "
                         "campaign without them: run freeze")
    if CHECKS.exists():
        raise ValueError("pre-execution checks already retained")
    original_hash = verify_manifest(checked.OUTPUT)
    revision, env = pilot.build_predictor()
    for track in TRACKS:
        for mode in ("AAC", "ANC"):
            write(CHECKS / f"{track}-{mode}/request.json", corrected_anchored(track, mode))
    write(CHECKS / "freeze.json", dict(source_revision=revision, models_fitted=0, full_campaign_started=False,
        previous_check_sha256=original_hash, predictor_sha256=digest(pilot.BINARY),
        scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        inputs={str(p.relative_to(CHECKS)): digest(p) for p in CHECKS.glob("*/*request.json")},
        correction="AAC and ANC starts use only known anchors and the declared arena, not ranges excluded by the mode", workers=1))
    batches, paths = [], {}
    for track in TRACKS:
        for mode in MODES:
            old = checked.OUTPUT / f"{track}-{mode}"
            if mode == "AF":
                batches.append(next(r for r in read(checked.OUTPUT / "summary.json")["batches"] if (r["track"], r["mode"]) == (track, mode)))
                paths[f"{track}-{mode}"] = pilot.audit.name(old)
                continue
            folder = CHECKS / f"{track}-{mode}"
            reference = read(old / "reference.json")
            request = read(folder / "request.json")
            started = time.monotonic()
            with (folder / "run.log").open("x") as log:
                subprocess.run([str(pilot.BINARY), str(folder / "request.json"), str(folder / "predictions.jsonl")],
                    cwd=pilot.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                predictions = records(folder / "predictions.jsonl")
            batches.append(summarize_batch(folder, request, reference, predictions, aac_check.LABELS, track, mode, time.monotonic()-started))
            paths[f"{track}-{mode}"] = pilot.audit.name(folder)
            print(json.dumps(batches[-1]), flush=True)
    inputs, outputs = (sum(b[k] for b in batches) for k in ("inputs", "outputs"))
    if (inputs, outputs) != (156, 768):
        raise ValueError("incomplete bounded comparison")
    write(CHECKS / "summary.json", dict(status="checks-complete", batches=batches, summaries=paths,
        inputs=inputs, outputs=outputs, models_fitted=0, full_campaign_started=False,
        scope="AF predictions retained, AAC and ANC initialization corrected for every method"))
    files = sorted(p for p in CHECKS.rglob("*") if p.is_file())
    write(CHECKS / "manifest.json", dict(files={str(p.relative_to(CHECKS)): digest(p) for p in files}))
    pilot.register_files([*files, CHECKS / "manifest.json"])


def batch_inputs(base, batch):
    block = next(b for b in base["blocks"] if b["id"] == batch["block"])
    streams = {tuple(map(int, pair.split("-"))): rows for pair, rows in block["reports"].items()}
    units, _ = checked.first_block(streams, block["id"])
    track, mode, k = batch["track"], batch["mode"], batch["anchors"]
    request = copy.deepcopy(base["settings"][f"{track}-{mode}"])
    request["units"] = []
    reference = dict(units=[], coordinates_cm={})
    coordinates = base["coordinates_cm"][block["acquisition"]]
    assignments = ((0, ()),) if mode == "AF" else ((i+1, a) for i, a in enumerate(ROLES) if len(a) == k)
    for role, anchors in assignments:
        sid = block["id"] if mode == "AF" else (block["id"]-1)*len(ROLES) + role
        for unit in units:
            item = unit
            if anchors:
                known = {n: coordinates[str(n)] for n in anchors}
                item, _ = checked.anchored_unit(unit, anchors, known, mode, base["anc_profiles"][track])
            item = checked.metric_input(item, base["fitted"] if track == "calibrated" else None)
            item["id"] = sid
            request["units"].append(item)
            if unit["samples"] == 1:
                reference["units"].append(dict(id=sid, reference=str(sid), acquisition=block["acquisition"],
                    block=block["rank"], anchors=k, anchor_ids=list(anchors)))
                reference["coordinates_cm"][str(sid)] = {str(n): coordinates[str(n)][:2] for n in item["problem"]["agent_ids"]}
    return request, reference


def catalog(base):
    for track in TRACKS:
        for block in base["blocks"]:
            for mode in MODES:
                for k in ((0,) if mode == "AF" else range(1, 7)):
                    yield dict(key=f'{track}-{mode}-{block["id"]:03d}-{k}', track=track, mode=mode,
                        block=block["id"], anchors=k, acquisition=block["acquisition"],
                        units=2 if mode == "AF" else 2*math.comb(8, k))


def budget(batches):
    timings = {}
    for track in TRACKS:
        for mode in MODES:
            folder = (checked.OUTPUT if mode == "AF" else CHECKS) / f"{track}-{mode}"
            metadata = {u["id"]: (u["acquisition"], u["anchors"]) for u in read(checked.OUTPUT / f"{track}-{mode}/reference.json")["units"]}
            rows = read(folder / "predictions.json") if mode == "AF" else records(folder / "predictions.jsonl")
            for row in rows:
                acquisition, k = metadata[row["id"]]
                key = track, mode, acquisition, k
                timings[key] = timings.get(key, 0.) + row["elapsed_ns"] / 1e9
    per_track = {}
    for track in TRACKS:
        seconds = sum(timings[track, b["mode"], b["acquisition"], b["anchors"]] * b["units"] / 2
                      for b in batches if b["track"] == track)
        per_track[track] = seconds / 3600
    return dict(workers=WORKERS, maximum_continuous_hours=12, per_solve_wall_clock_cutoff=False,
        checkpoints=len(batches), one_worker_hours=per_track,
        ideal_parallel_hours=sum(per_track.values()) / WORKERS,
        caveat="Rough planning estimate from the checked first block and first anchor subset of each size. Other subsets, I/O and concurrency can differ substantially.",
        pause="Stop dispatching new batches after twelve hours. Finish running batches and resume without discarding completed results.")


def freeze():
    if (OUTPUT / "freeze.json").exists():
        raise ValueError("full campaign already frozen")
    if pilot.READER:
        # no bounded check ran and no adapter is built: the check inputs that tiers_static_check.prepare wrote
        check_hash, revision = verify_inputs(checked.OUTPUT), pilot.audit.revision()
    else:
        check_hash = verify_manifest(CHECKS)
        verify_manifest(checked.OUTPUT)
        summary = read(CHECKS / "summary.json")
        if summary["outputs"] != 768:
            raise ValueError("bounded check gate failed")
        revision, _ = pilot.build_predictor()
    settings = {}
    for track in TRACKS:
        for mode in MODES:
            request = read(checked.OUTPUT / f"{track}-{mode}/request.json")
            settings[f"{track}-{mode}"] = {k: v for k, v in request.items() if k != "units"}
    # a reader's reproduction takes the settings of the paper's campaign, which hold those of its check inputs
    settings, shipped = shipped_settings("tiers-campaign-settings.json", settings) if pilot.READER else (settings, {})
    base = dict(blocks=[], coordinates_cm={}, settings=settings,
        fitted=read(aac_training.OUTPUT / "profiles.json")["fitted"],
        anc_profiles=read(checked.OUTPUT / "profiles.json")["anc"])
    sources = ({str(checked.OUTPUT / "inputs.json"): check_hash} if pilot.READER else
               {str(checked.OUTPUT / "manifest.json"): digest(checked.OUTPUT / "manifest.json"), str(CHECKS / "manifest.json"): check_hash})
    sources.update({str(static_preflight.OUTPUT): digest(static_preflight.OUTPUT), **shipped})
    preflight = read(static_preflight.OUTPUT)
    for name, expected in preflight["sources"].items():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "static input")
    for acquisition_index, row in enumerate(read(pilot.audit.DATA / "input-audit.json")["tiers"], 1):
        path = pilot.audit.DATA / row["file"]
        pilot.evidence.official_scorer.require_hash(path, row["sha256"], "static recording")
        acquisition = path.name.split("_")[0]
        ranges, _ = pilot.audit.tiers_events(path)
        streams = static_preflight.tiers_streams(ranges)
        for rank, block in enumerate(static_preflight.tiers_blocks(streams, 10)):
            base["blocks"].append(dict(id=3*rank+acquisition_index, acquisition=acquisition, rank=rank,
                reports={f"{a}-{b}": rows.tolist() for (a, b), rows in block.items()}))
        base["coordinates_cm"][acquisition] = {n: [100*v for v in p] for n, p in row["pose_median_native"].items()}
        sources[str(path)] = digest(path)
    if len(base["blocks"]) != 139:
        raise ValueError("static input population changed")
    batches = list(catalog(base))
    checked_inputs = {}
    for track in TRACKS:
        for mode in MODES:
            path = (checked.OUTPUT if mode == "AF" else CHECKS) / f"{track}-{mode}/request.json"
            # without retained checks, the corrected anchored requests that check writes are built from the check inputs
            request = corrected_anchored(track, mode) if pilot.READER and mode != "AF" else read(path)
            for u in request["units"]:
                acquisition_id = u["id"] if mode == "AF" else u["id"] // 10
                checked_inputs[track, mode, acquisition_id, tuple(a["id"] for a in u["problem"]["anchors"]), u["samples"]] = u
    reproduced = 0
    for batch in batches:
        request, reference = batch_inputs(base, batch)
        if len(request["units"]) != batch["units"]:
            raise ValueError("wrong batch population")
        for unit in request["units"]:
            key = batch["track"], batch["mode"], batch["block"], tuple(a["id"] for a in unit["problem"]["anchors"]), unit["samples"]
            original = checked_inputs.get(key)
            if original:
                if {k: v for k, v in unit.items() if k != "id"} != {k: v for k, v in original.items() if k != "id"}:
                    raise ValueError("full input differs from checked demonstration")
                reproduced += 1
        batch.update(request_sha256=value_hash(request), reference_sha256=value_hash(reference))
    if reproduced != 156 or len(batches) != 3614:
        raise ValueError("demonstration parity or batch count changed")
    # the runs of a reader's reproduction take the adapter of its own build, so it freezes no executable
    frozen = ("base.json.gz", "catalog.json.gz") if pilot.READER else ("base.json.gz", "catalog.json.gz", "bin/external-pilot")
    OUTPUT.mkdir(exist_ok=True)
    for name, value in (("base.json.gz", base), ("catalog.json.gz", batches)):
        (OUTPUT / name).write_bytes(gzip.compress(encoded(value), mtime=0))
    if not pilot.READER:
        (OUTPUT / "bin").mkdir()
        shutil.copy2(pilot.BINARY, OUTPUT / "bin" / "external-pilot")
    for path in pilot.present([pilot.HERE / name for name in ("tiers_campaign.py", "tiers_static_check.py", "predict.rs", "README.md")]):
        sources[str(path)] = digest(path)
    write(OUTPUT / "freeze.json", dict(source_revision=revision, source_identity=pilot.isolation.source_identity(), state="frozen-not-started", models_fitted=0,
        files={name: digest(OUTPUT / name) for name in frozen},
        sources=sources, scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256,
        inputs=sum(b["units"] for b in batches), outputs=sum(b["units"] * len(dict(AF=pilot.LABELS, AAC=aac_check.LABELS, ANC=aac_check.LABELS)[b["mode"]]) for b in batches),
        methods=dict(AF=list(pilot.LABELS), AAC=list(aac_check.LABELS), ANC=list(aac_check.LABELS)),
        population_per_sample_and_track=preflight["tiers"]["proposed_inputs_per_track"]["10"],
        both_sample_counts=[1, 10], tracks=list(TRACKS), reproduced_demonstrations=reproduced,
        weighting="Equal acquisition weights, also equal anchor-count weights in AAC and ANC. Every unknown-agent pair is scored.",
        initializations="AF uses the first-report range scale. AAC and ANC use only the public arena and known anchors.",
        # the planning estimate times the solves of the bounded checks, which a reader's reproduction does not run
        budget=None if pilot.READER else budget(batches), host=platform.platform()))
    files = [OUTPUT / n for n in ("freeze.json", *frozen)]
    write(OUTPUT / "manifest.json", dict(files={str(p.relative_to(OUTPUT)): digest(p) for p in files}))
    pilot.register_files([*files, OUTPUT / "manifest.json"])
    print(json.dumps({k: read(OUTPUT / "freeze.json")[k] for k in ("state", "inputs", "outputs", "budget")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "freeze"))
    {"check": check, "freeze": freeze}[parser.parse_args().command]()
