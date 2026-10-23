"""Same-layout Bregar populations: the first block of ten reports of every link trains, the other two evaluate.

Each environment is one deployment: eight anchors, one tag carried over its surveyed positions, six channels and at
least thirty reports per link and channel. The frozen comparison held every environment out and calibrated it from
the other three. This protocol calibrates every environment from its own first block instead, the rule IDLab and
TIERS follow, so the training half shares the layout, the devices and the settings of the evaluation half and no
evaluation report enters a fit, a selection or a design decision:

- Offsets: per anchor and channel, the shortest line of sight link of the environment; the mean of its first ten
  reports minus the surveyed distance. The frozen rule restricted to the training block.
- Scale: per channel, the RMS residual of the corrected first-block reports over the environment's links, equal
  weight per link, as the frozen adapter profiles.
- Units: adapter.construct on the same streams, block 0 for training and blocks 1 and 2 for evaluation, identities as
  in the frozen population, 1 + 18 environment + 3 channel + block, both sample counts. Every link the radio answered
  in the block is an observation. Its inadmissible readings (corrected distance below zero) stay out of the samples,
  so a link whose every reading was inadmissible is present without samples.
- STRONG's radius: selected per environment on the training block among 40 cm and one, one and a half and two and a
  half channel RMS by the mean balanced S75 over the channel cells and the two sample counts, ties by declared order,
  the selection this protocol first made and that the one below replaced.
  The evaluation request carries the selected radius of its environment and channel.

The options of Jin 2021, FNL 2025 and STRONG 2021 that TWR-CLOUD-Bern's scans chose are now selected, which replaces
the radius selection of the list at run time and the spatial model every baseline once kept. Per environment on the
training block, each method on its own among its declared candidates (baseline_selection.py), Jin its model and the
aggregation of a link's readings, FNL its model, aggregation and start, and STRONG its Huber radius among 10, 20, 40,
80 and 160 cm and no robust loss, its model and its start, by the mean over the two sample counts of the balanced S75
over the channel cells, ties by declared order among the means within 0.01 cm of the lowest. Both halves run with the
declared settings of the candidate their environment selected (baseline_selection.chosen) and every other setting from
the declared base configurations (baseline-configurations.json). The first selection stays
in strong-selection.json and freeze.json with the radii it wrote into the evaluation request, and the joint STRONG
selection over the earlier grid of twelve candidates in strong-selection-v2.json.

A unit scores only the agent pairs that its measured links join, with every anchor counted as one node (the joined pair
set of scoring.py). A selection ranks the candidates under that pair set over the units of the anchor only headline,
three or more anchors, which every Bregar unit holds, and its record names the method, the rule, the candidates, the
pair set, the anchor rule, the declared base configurations and the rule of the shared starts (layout_inputs.record),
with the share of every candidate's solves that stayed at their start (isolation.STAYED). A run refuses a record whose
content differs from the declared values, so a record kept from an earlier rule, grid, base record or rule of the
starts runs nothing, and check-selection verifies the records. Every scored row keeps the mean
pair error and the pair count of every pair set of its unit (scoring.score_with_pair_sets) with the unit's environment
as its key and its anchor count, so rows pair with SOLID's on (key, id, samples). The summary of a half reports the
balanced percentiles under the joined pairs and the supplementary ranged pairs with the share of pairs and units each
scores.

Every run reads the half's request through half_request, which draws every method's starts inside SOLID's operating area
of the unit as SOLID's runner does (layout_inputs.levelled) and gives every unit that area, which ADE-BP receives. The
stored request keeps the starts of its build.

ADE-BP 2015 runs in the planar form of its paper beside the three, with the anchor only candidate and the fitted error
law of each environment, model 2 in environments 0, 2 and 3 and model 1 in environment 1, which the shipped settings
name (settings/baseline-options.json, baseline_selection.options). The runs of ADE-BP's selection and of its fits
belong to the design process. In reader mode (SOLID_READER=1) every method takes its final option of each environment
from the shipped settings instead of a selection record, and a run checks no input against a selection record.

Every adapter process runs with one thread for Rayon, OpenBLAS and OpenMP and solves one unit. A unit whose request
crashes the adapter is solved again method by method, which the adapter accepts as it accepts any nonempty subset of
Jin, FNL and STRONG, so a single solve that crashes is scored as that method's failure on that unit with the failure
penalty and a warning, and the other methods of the unit keep their own results (pilot.run_pilot,
isolation.ABORT_RULE). Every summary counts the crashed solves of every method beside listing them and prints the
counts. The selection record and the summaries name the machine they ran on, the workers that solved at once and the
timing rule of their solve times (isolation.execution), and the summaries the starts of every method
(isolation.STARTS). A half runs only with the build of the adapter that made every selection, so every method of Bregar
comes from one build (isolation.same_build).

Outputs under external-benchmarks/bregar-layout of the data folder (SOLID_DATA): calibration.json, population.json,
training/ and evaluation/ with request.json.gz, reference.json and selections.json.gz, <method>-selection-v3.json after
select, results/<split>/ after evaluate.

Usage: bregar_layout.py build
       bregar_layout.py select --method jin-2021|fnl|strong [--workers N]
       bregar_layout.py check-selection [--method jin-2021|fnl|strong]
       bregar_layout.py evaluate --split training|evaluation [--workers N]
"""
import argparse
import copy
import gzip
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import baseline_selection
import bregar
import bregar_adapter as adapter
import isolation
import layout_inputs
import layout_split
import pilot
import scoring

OUTPUT = bregar.audit.DATA / "bregar-layout"
BINARY = adapter.BINARY
BLOCKS = {"training": layout_split.BREGAR_TRAINING_BLOCKS, "evaluation": layout_split.BREGAR_EVALUATION_BLOCKS}
BLOCK_REPORTS = 10
SCHEMA = "solid.external.bregar-layout.v1"
# The roster of Bregar in the order of the final tables, the three baselines of the frozen comparison and ADE-BP 2015 in
# its planar form, and the baselines every built request names, the entries of the frozen comparison's request
# (bregar-evaluation/request.json.gz, SHA-256 FROZEN_REQUEST_SHA256), declared here so that a build reads no earlier
# campaign. A run replaces their settings by the declared base configurations and the chosen candidates (unit_request).
LABELS = {"jin-2021": "Jin 2021", "ade-bp-2015": "ADE-BP 2015", "fnl": "FNL 2025", "strong": "STRONG 2021"}
ENVIRONMENTS = tuple(f"environment{index}" for index in range(4))
FROZEN_REQUEST_SHA256 = "e5e2e3591f867e7ffe516726380943724088b866133b120f119cfe3a4532e0cb"
_FROZEN_SOLVE = dict(aggregation="Median", dimension_policy="spatial", gradient_tolerance=1e-05, history_size=10, max_iterations=10000,
                     max_time_micros=0, max_tof_ns=200.0, objective="AccuracyProfile", requested_starts=32,
                     research=dict(aac_decoder="projected_edm_median", accuracy_solver_policy=False, anc_cutoff_standardized=0.0,
                                   anc_initializer="production"))
BASELINES = [dict(method="jin-2021", method_config=dict(method="jin-2021", residual_multiplier=2.0), solve_config=dict(_FROZEN_SOLVE, robust_delta=0.7)),
             dict(method="fnl", method_config=dict(initialization="released", method="fnl"), solve_config=dict(_FROZEN_SOLVE, robust_delta=0.7)),
             dict(method="strong", method_config=dict(initialization="solid-geometry", method="strong"), solve_config=dict(_FROZEN_SOLVE, robust_delta=40.0))]
# the methods whose options every environment selects, the three baselines of the frozen comparison, and the schema of
# their records. ADE-BP takes its anchor only candidate and law of each environment from the shipped settings.
SELECTED = tuple(method for method in baseline_selection.SELECTED if method in LABELS and baseline_selection.selects(method, "ANC"))
SELECTION_SCHEMA = baseline_selection.SELECTION_SCHEMA
# The pair set a selection ranks under and the fewest anchors of the units it ranks over, those of the headline
PAIR_SET = scoring.PAIR_SET
HEADLINE_MIN_ANCHORS = scoring.HEADLINE_MIN_ANCHORS
RULE = ("Per environment on the training block, each of Jin 2021, FNL 2025 and STRONG 2021 on its own among the declared candidates "
        "of the options TWR-CLOUD-Bern chose for it (baseline_selection.py), by the lowest mean over the two sample counts of the "
        "balanced S75 over the six channel cells, ties by declared order among the means within 0.01 cm of the lowest. A unit's error "
        "is its mean capped pair error over the pairs its measured links join, and the units are those of the anchor only headline, "
        "three or more anchors. Every other setting is the declared base configuration.")
# The candidates a unit's environment chose, {method: solve and method settings}, carried beside the unit for its
# request and never sent inside it
SETTINGS = "baseline_settings"
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def write_gz(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(gzip.compress(encoded(value), mtime=0))


def environment_cells(cells, environment):
    return [r for r in cells if r["environment"] == environment and not (environment == "environment2" and r["anchor"] == "A6")]


def calibrate(cells, environment, hardware):
    """Offsets per anchor and channel and one scale per channel from the first block of the environment's own links."""
    rows = environment_cells(cells, environment)
    by_pair = defaultdict(list)
    for row in rows:
        values = np.asarray(row["ranges_m"], dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(row["distance_m"]) or row["distance_m"] < 0:
            raise ValueError("invalid calibration ranges or reference")
        if len(values) and len(values) < 3 * BLOCK_REPORTS:
            raise ValueError("a nonempty link lacks the three complete blocks of the frozen population")
        if len(values) and (row["anchor_ids"] != hardware["hardware_ids"][row["anchor"]] or row["tag_ids"] != hardware["tag_ids"]):
            raise ValueError("calibration hardware differs from verified identities")
        by_pair[row["anchor"], row["channel"]].append(row)
    expected = {(f"A{a}", ch) for a in range(1, 9) for ch in bregar.CHANNELS if not (environment == "environment2" and a == 6)}
    if set(by_pair) != expected:
        raise ValueError("calibration links differ from the published population")
    corrections, offsets = [], {}
    for key, links in sorted(by_pair.items()):
        candidates = [r for r in links if r["los"] and len(r["ranges_m"])]
        if not candidates:
            raise ValueError(f"no line of sight link for {environment} {key}")
        selected = min(candidates, key=lambda r: (r["distance_m"], r["position"]))
        first = np.asarray(selected["ranges_m"][:BLOCK_REPORTS], dtype=float)
        offsets[key] = float(np.mean(first) - selected["distance_m"])
        corrections.append(dict(anchor=key[0], channel=key[1], offset_m=offsets[key], source_member=selected["member"],
                                source_reports=BLOCK_REPORTS, los_links=len(candidates), reference_distance_m=selected["distance_m"],
                                mean_raw_m=float(np.mean(first))))
    profiles = []
    for channel in bregar.CHANNELS:
        errors, links, failed = [], 0, 0
        for row in sorted((r for r in rows if r["channel"] == channel and len(r["ranges_m"])), key=lambda r: r["member"]):
            corrected = adapter.calibrated(row["ranges_m"][:BLOCK_REPORTS], offsets[row["anchor"], channel])
            accepted = corrected >= 0
            residual = 100 * (corrected[accepted] - row["distance_m"])
            links += 1
            failed += int((~accepted).sum())
            if len(residual):
                errors.append(float(np.mean(residual ** 2)))
        scale = float(np.sqrt(np.mean(errors))) if errors else 0.0
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"no positive training range uncertainty for {environment} channel {channel}")
        profiles.append(dict(channel=channel, scale_cm=scale, training_links=links, accepted_links=len(errors),
                             training_reports=links * BLOCK_REPORTS, failed_reports=failed))
    return dict(environment=environment, corrections=corrections, profiles=profiles, block_reports=BLOCK_REPORTS,
                evaluation_rows_used=0)


def environment_streams(cells, environment, channel):
    selected = sorted((r for r in environment_cells(cells, environment) if r["channel"] == channel), key=lambda r: r["member"])
    positions = sorted({r["position"] for r in selected})
    identities = {p: i + 1 for i, p in enumerate(positions)}
    streams, members = defaultdict(dict), []
    for row in selected:
        agent, anchor = identities[row["position"]], int(row["anchor"][1:])
        if anchor in streams[agent]:
            raise ValueError("duplicate raw link")
        streams[agent][anchor] = row["ranges_m"]
        members.append(dict(agent=agent, anchor=anchor, member=row["member"]))
    return positions, identities, streams, members


def answered(units, records):
    """Every link the radio answered is an observation of its unit. Inadmissible readings (corrected distance below
    zero) stay out of the samples, so a link whose every reading was inadmissible is present with no sample, a
    successful exchange without a range. A link without any report has no observation."""
    for unit, record in zip(units, records, strict=True):
        if (unit["id"], unit["samples"]) != (record["id"], record["samples"]):
            raise ValueError("unit and selection records disagree")
        emitted = {(o["from"], o["to"]): o for o in unit["problem"]["observations"]}
        observations = []
        for link in record["links"]:
            if not link["raw_m"]:
                continue
            key = (link["agent"], link["anchor"] + 100)
            observations.append(emitted.pop(key, None) or {"from": key[0], "to": key[1], "to_is_anchor": True, "tof_ns": [], "calibrated_samples": []})
        if emitted:
            raise ValueError("an observation without an answered link")
        unit["problem"]["observations"] = observations
    return units


def construct(cells, environment, anchors_cm, calibration):
    """Both halves of one environment: units, reference metadata, coordinates, heights and the selected raw rows."""
    result = {side: dict(units=[], metadata=[], selections=[]) for side in BLOCKS}
    coordinates, heights, first_positions = None, None, None
    for channel in bregar.CHANNELS:
        positions, identities, streams, members = environment_streams(cells, environment, channel)
        if first_positions is None:
            first_positions = positions
            coordinates = {str(identities[p]): (np.array(p.split("_"), dtype=float)[:2] * 100).tolist() for p in positions}
            heights = {str(identities[p]): float(p.split("_")[2]) * 100 for p in positions}
        elif positions != first_positions:
            raise ValueError("the channels of an environment do not share the surveyed positions")
        offsets = {int(r["anchor"][1:]): r["offset_m"] for r in calibration["corrections"] if r["channel"] == channel}
        scale, = [p["scale_cm"] for p in calibration["profiles"] if p["channel"] == channel]
        for side, blocks in BLOCKS.items():
            for block in blocks:
                sid = layout_split.bregar_sid(int(environment[-1]), bregar.CHANNELS.index(channel), block)
                units, records = adapter.construct(anchors_cm, streams, offsets, scale, sid, block)
                for unit in answered(units, records):
                    unit["strong_radius_cm"] = scale
                result[side]["units"].extend(units)
                result[side]["selections"].extend(records)
                result[side]["metadata"].append(dict(id=sid, acquisition=f"{environment}-ch{channel}", reference=environment,
                                                     block=block, channel=channel, members=members))
    return result, coordinates, heights


def population(units, side):
    expected = {(sid, n) for sid in layout_split.bregar_split()[f"{side}_units"] for n in (1, 10)}
    keys = [(u["id"], u["samples"]) for u in units]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"incomplete Bregar {side} population")
    agents = sum(len(u["problem"]["agent_ids"]) for u in units)
    pairs = sum(len(u["problem"]["agent_ids"]) * (len(u["problem"]["agent_ids"]) - 1) // 2 for u in units)
    return dict(inputs=len(units), agent_attempts=agents, pair_attempts=pairs)


def build():
    if OUTPUT.exists():
        raise ValueError("Bregar layout populations already built")
    checked = read(bregar.RAW_AUDIT)
    pilot.evidence.official_scorer.require_hash(bregar.ARCHIVE, checked["source"]["sha256"], "Bregar raw archive")
    baselines = copy.deepcopy(BASELINES)
    calibrations, halves = [], {side: dict(units=[], metadata=[], selections=[], coordinates_cm={}, heights_cm={}) for side in BLOCKS}
    with zipfile.ZipFile(bregar.ARCHIVE) as archive:
        cells = bregar.calibration_cells(archive, checked)
        for index in range(4):
            environment = f"environment{index}"
            anchors_cm = bregar.coordinates(archive.read(f"data_set/{environment}/anchors.csv"), True) * 100
            calibration = calibrate(cells, environment, checked)
            calibrations.append(calibration)
            result, coordinates, heights = construct(cells, environment, anchors_cm, calibration)
            for side, part in result.items():
                halves[side]["units"].extend(part["units"])
                halves[side]["metadata"].extend(part["metadata"])
                halves[side]["selections"].extend(part["selections"])
                halves[side]["coordinates_cm"][environment] = coordinates
                halves[side]["heights_cm"][environment] = heights
            print(f"{environment}: {len(coordinates)} positions, scales " + ", ".join(f"ch{p['channel']} {p['scale_cm']:.1f}" for p in calibration["profiles"]), flush=True)
    counts = {}
    OUTPUT.mkdir(parents=True)
    for side, half in halves.items():
        counts[side] = population(half["units"], side)
        write_gz(OUTPUT / side / "request.json.gz", dict(baselines=baselines, units=half["units"]))
        write_gz(OUTPUT / side / "selections.json.gz", half["selections"])
        write(OUTPUT / side / "reference.json", dict(units=half["metadata"], coordinates_cm=half["coordinates_cm"], heights_cm=half["heights_cm"]))
    write(OUTPUT / "calibration.json", dict(schema=SCHEMA, rule="Per environment, from the first ten reports of its own links: shortest line of "
                                                                "sight offset per anchor and channel, RMS scale per channel with equal weight per link.",
                                             environments=calibrations, archive_sha256=checked["source"]["sha256"],
                                             raw_audit_sha256=digest(bregar.RAW_AUDIT), evaluation_rows_used_for_fitting=0))
    write(OUTPUT / "population.json", dict(schema=SCHEMA, split=layout_split.bregar_split()["rule"], counts=counts,
                                           baselines_source="bregar_layout.BASELINES, the entries of the frozen comparison's request "
                                                            "bregar-evaluation/request.json.gz", baselines_source_sha256=FROZEN_REQUEST_SHA256,
                                           files={str(p.relative_to(OUTPUT)): digest(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file()},
                                           source_sha256={name: digest(pilot.HERE / name) for name in ("bregar_layout.py", "bregar_adapter.py", "bregar.py", "layout_split.py")}))
    print(json.dumps(counts), flush=True)


def half_request(split):
    """The built request of a half with every method's starts drawn by SOLID's deployment rules (layout_inputs.levelled),
    as SOLID's runner takes them. The stored request keeps the starts of its build, and every run replaces them."""
    return layout_inputs.levelled(read_gz(OUTPUT / split / "request.json.gz"), "bregar", "ANC")


def unit_request(unit, roster):
    """The adapter request of one unit, every method of the roster from the declared base configurations with the
    candidate its environment chose, carried beside the unit, and ADE-BP with the law of its environment
    (baseline_selection.declared_baselines)."""
    unit = dict(unit)
    chosen = unit.pop(SETTINGS, {})
    law = (chosen.get("ade-bp-2015") or {}).get("ranging_model")
    return dict(baselines=baseline_selection.declared_baselines(roster, "ANC", chosen, law), units=[unit])


def run_adapter(units, roster, workers, folder):
    """The methods of the roster on every unit, one single-threaded adapter process per unit and at most `workers` at a
    time, a solve that crashes the adapter recorded as its method's failure (pilot.run_pilot). Cached per unit."""
    requests = [(f"{unit['id']:02d}-s{unit['samples']}", unit_request(unit, roster)) for unit in units]
    results = pilot.run_pilot(requests, workers, folder, BINARY)
    return [record for key, _ in requests for record in results[key]]


def aborted_solves(predictions):
    """The solves that crashed the adapter, per method."""
    failures = defaultdict(list)
    for row in predictions:
        if isolation.is_abort(row["failure"]):
            failures[row["method"]].append(dict(id=row["id"], samples=row["samples"], failure=row["failure"]))
    return dict(failures)


def score(request, reference, predictions, methods=tuple(LABELS)):
    """Scored rows of the methods, each with the mean pair error and the pair count of every pair set of its unit, its
    environment as the key and its acquisition as the cell that SOLID's rows carry, and its anchor count. The full
    roster also passes the adapter's population check, a subset the scorer's."""
    if set(methods) == set(LABELS):
        adapter.check_outputs(request, predictions, LABELS)
    converted = copy.deepcopy(predictions)
    for row in converted:
        if row["estimate"] is not None:
            row["estimate"]["positions"] = [None if p is None else [p["x"], p["y"]] for p in row["estimate"]["positions"]]
    metadata = {u["id"]: u for u in reference["units"]}
    anchors = {u["id"]: len(u["problem"]["anchors"]) for u in request["units"]}
    scores = scoring.score_with_pair_sets(request, reference, converted, pilot.evidence.load_scorer(), methods)
    for row in scores:
        row["environment"] = row["key"] = metadata[row["id"]]["reference"]
        row["cell"] = metadata[row["id"]]["acquisition"]
        row["block"] = metadata[row["id"]]["block"]
        row["channel"] = metadata[row["id"]]["channel"]
        row["anchors"] = anchors[row["id"]]
    return scores


def environment_of(unit):
    """The environment of a unit from its identity, 1 + 18 environment + 3 channel + block."""
    return f"environment{(unit['id'] - 1) // 18}"


def settings(entry):
    """What a unit carries of a candidate or of a selection entry, its solve and method settings, and ADE-BP's model and
    law where the entry holds them."""
    return dict(solve_config=copy.deepcopy(entry["solve_config"]), method_config=copy.deepcopy(entry.get("method_config") or {}),
                **{key: copy.deepcopy(entry[key]) for key in ("model", "ranging_model") if key in entry})


def settled(environment):
    """The final options of every method of the roster in an environment from the shipped settings, each as a selection
    entry with the declared settings of its candidate, ADE-BP's with its model and law (baseline_selection.options)."""
    result = {}
    for method, choice in baseline_selection.options("bregar", "ANC", environment).items():
        entry = baseline_selection.chosen(method, "ANC", choice)
        result[method] = dict(selected=entry["name"], **settings(entry))
    return result


def with_settings(units, settings_of):
    """Units carrying the candidates their environment chose, settings_of(environment) mapping a method to a candidate
    or a selection entry, beside the unit for its request. A unit carries the declared settings of the candidate the
    entry names, and an entry whose settings differ from them is refused (baseline_selection.chosen). STRONG's radius
    goes into the unit as well, where the adapter reads it."""
    result = []
    for unit in units:
        chosen = {}
        for method, entry in settings_of(environment_of(unit)).items():
            declared = baseline_selection.chosen(method, "ANC", entry)
            chosen[method] = dict(selected=declared["name"], **settings(declared))
        entry = {key: value for key, value in unit.items() if key != SETTINGS}
        if "strong" in chosen:
            entry["strong_radius_cm"] = chosen["strong"]["solve_config"]["robust_delta"]
        result.append(dict(entry, **{SETTINGS: chosen}))
    return result


def choose(scores, environments, scorer, method):
    """Per environment, the method's candidates ranked by the declared rule over its channel cells and the two sample
    counts. A unit's error is its mean pair error over the pairs its measured links join, the headline's pair set, and
    the units are those of the anchor only headline. A unit without a joined pair is left out, as the released scorer
    leaves out a scenario without eligible pairs."""
    candidates = baseline_selection.candidates(method, "ANC")
    result = {}
    for environment in environments:
        values, mean = {}, {}
        for candidate in candidates:
            values[candidate["name"]] = []
            for samples in (1, 10):
                cells = defaultdict(list)
                for row in scores:
                    if row["candidate"] == candidate["name"] and row["environment"] == environment and row["samples"] == samples and row["method"] == method:
                        error = row["pair_sets"][PAIR_SET]["mean_pair_error_cm"]
                        if error is not None and row["anchors"] >= HEADLINE_MIN_ANCHORS:
                            cells[row["acquisition"]].append(error)
                if len(cells) != len(bregar.CHANNELS):
                    raise ValueError(f"incomplete {method} selection population")
                values[candidate["name"]].append(scorer.balanced_percentile(cells, 0.75))
            mean[candidate["name"]] = float(np.mean(values[candidate["name"]]))
        ranking = [dict(candidate=c["name"], **settings(c), s75_by_samples=values[c["name"]], mean_s75=round(mean[c["name"]], 9))
                   for c in baseline_selection.rank_candidates(candidates, mean)]
        result[environment] = dict(selected=ranking[0]["candidate"], solve_config=ranking[0]["solve_config"], method_config=ranking[0]["method_config"],
                                   ranking=ranking)
    return result


def selection_path(method):
    """The Bregar selection record of a method."""
    return OUTPUT / f"{method}-selection-v3.json"


def selection(method):
    """The selection record of a method, refused unless it is Bregar's selection of that method, made by the declared
    rule over the method's declared candidates, ranking them under the headline's pair set over the units of the anchor
    only headline, from the declared base configurations and with the shared inputs of this run (layout_inputs.record),
    and made by the adapter that runs now. The record's content decides, so a record kept from an earlier rule, grid,
    base record or rule of the starts runs nothing."""
    path = selection_path(method)
    record = read(path)
    if record.get("schema") != SELECTION_SCHEMA or record.get("dataset") != "bregar":
        raise ValueError(f"{path} is not a selection record of Bregar of the declared schema {SELECTION_SCHEMA}")
    if record.get("method") != method:
        raise ValueError(f"{path} is not a selection of {method}")
    if record.get("rule") != RULE or record.get("candidates") != json.loads(json.dumps(baseline_selection.grid(method, ("ANC",)))):
        raise ValueError(f"{path} was made under another rule or candidate grid than the declared ones")
    if record.get("pair_set") != PAIR_SET:
        raise ValueError(f"{path} ranked {method}'s candidates under another pair set than the {PAIR_SET} pairs of the headline")
    if record.get("anchor_only_headline_min_anchors") != HEADLINE_MIN_ANCHORS:
        raise ValueError(f"{path} ranked the anchor only mode over other units than those of {HEADLINE_MIN_ANCHORS} or more anchors")
    if record.get("base_configurations_sha256") != digest(baseline_selection.BASE):
        raise ValueError(f"{path} was made with other declared base configurations than those of this run")
    if record.get("shared_inputs") != json.loads(json.dumps(layout_inputs.record("bregar"))):
        raise ValueError(f"{path} was made with other shared inputs than those of this run (layout_inputs.RULE)")
    return isolation.same_build(record, path, BINARY)


def select(method, workers):
    """A method's options per environment on the training block, the method solved alone under every candidate, since
    the other baselines do not change with its options and run once, in the runs of the halves."""
    path = selection_path(method)
    if path.exists():
        raise ValueError(f"{method}'s Bregar options already selected")
    request = half_request("training")
    reference = read(OUTPUT / "training" / "reference.json")
    candidates = baseline_selection.candidates(method, "ANC")
    scores, started = [], time.monotonic()
    for candidate in candidates:
        units = with_settings(request["units"], lambda environment: {method: candidate})
        predictions = run_adapter(units, [method], workers, OUTPUT / "work" / "training")
        for row in score(dict(units=units), reference, predictions, methods=(method,)):
            scores.append(dict(row, candidate=candidate["name"]))
        print(f"{method} candidate {candidate['name']} scored after {time.monotonic() - started:.0f}s", flush=True)
    environments = sorted({m["reference"] for m in reference["units"]})
    choice = choose(scores, environments, pilot.evidence.load_scorer(), method)
    training_scores = [dict((k, v) for k, v in row.items() if k not in ("pairs", "pair_errors_cm")) for row in scores]
    inputs = [OUTPUT / name for name in ("calibration.json", "population.json", "training/request.json.gz", "training/reference.json",
                                         "evaluation/request.json.gz", "evaluation/reference.json")]
    aborted = {name: isolation.abort_counts([r for r in scores if r["candidate"] == name], (method,))[method] for name in (c["name"] for c in candidates)}
    # per candidate, the solves over every environment that stayed at their start
    stayed = {c["name"]: isolation.stayed_counts([r for r in scores if r["candidate"] == c["name"]]).get(method) for c in candidates}
    write(path, dict(schema=SELECTION_SCHEMA, dataset="bregar", method=method, scope="environment", rule=RULE,
                     pair_set=PAIR_SET, anchor_only_headline_min_anchors=HEADLINE_MIN_ANCHORS,
                     candidates=baseline_selection.grid(method, ("ANC",)), selections=choice, training_solves=len(scores), evaluation_solves=0,
                     scores=training_scores, stayed_at_start=stayed, stayed_rule=isolation.STAYED,
                     inputs_sha256={str(p.relative_to(OUTPUT)): digest(p) for p in inputs}, workers=workers, **isolation.build(BINARY),
                     base_configurations_sha256=digest(baseline_selection.BASE), evaluation_rows_used_for_selection=0,
                     scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256, abort_rule=isolation.ABORT_RULE,
                     abort_counts=isolation.abort_counts(scores, (method,)), candidate_abort_counts=aborted,
                     shared_inputs=layout_inputs.record("bregar"), **isolation.execution(workers)))
    isolation.report_aborts(aborted, f"Bregar {method} candidates")
    print(json.dumps({environment: entry["selected"] for environment, entry in choice.items()}), flush=True)


def evaluate(split, workers):
    """Every baseline on one half, each with the selected candidate of its environment and every other setting from
    the declared base configurations, and ADE-BP with the candidate and law of its environment from the shipped
    settings. In reader mode every method takes its final option of each environment from the shipped settings, and no
    input is checked against a selection record."""
    settled_options = {environment: settled(environment) for environment in ENVIRONMENTS}
    if baseline_selection.READER:
        chosen = {method: {environment: settled_options[environment][method] for environment in ENVIRONMENTS} for method in LABELS}
    else:
        missing = [method for method in SELECTED if not selection_path(method).exists()]
        if missing:
            raise ValueError(f"select the options of {', '.join(missing)} before running a half")
        records = {method: selection(method) for method in SELECTED}
        for method, record in records.items():
            for name, expected in record["inputs_sha256"].items():
                if digest(OUTPUT / name) != expected:
                    raise ValueError(f"{name} changed after the selection of {method}")
        chosen = {method: record["selections"] for method, record in records.items()}
        chosen["ade-bp-2015"] = {environment: settled_options[environment]["ade-bp-2015"] for environment in ENVIRONMENTS}
    request = half_request(split)
    reference = read(OUTPUT / split / "reference.json")
    units = with_settings(request["units"], lambda environment: {method: chosen[method][environment] for method in chosen})
    predictions = run_adapter(units, list(LABELS), workers, OUTPUT / "work" / split)
    scores = score(dict(units=units), reference, predictions)
    results = OUTPUT / "results" / split
    results.mkdir(parents=True, exist_ok=True)
    (results / "predictions.jsonl.gz").write_bytes(gzip.compress(b"".join(encoded(r) + b"\n" for r in predictions), mtime=0))
    (results / "scores.jsonl.gz").write_bytes(gzip.compress(b"".join(encoded(r) + b"\n" for r in scores), mtime=0))
    # the tables of every method per sample track and reported group, under the pairs the measured links join and the
    # supplementary ranged pairs
    percentile = pilot.evidence.load_scorer().balanced_percentile
    tables = {method: scoring.summary([r for r in scores if r["method"] == method], "ANC", percentile) for method in LABELS}
    counts = isolation.abort_counts(predictions, LABELS)
    summary = dict(metric="Bregar transferred horizontal pairwise score over the pairs the measured links join", pair_set=PAIR_SET,
                   supplementary_pair_sets=list(scoring.SUPPLEMENTARY), anchor_only_headline_min_anchors=HEADLINE_MIN_ANCHORS,
                   methods=[dict(id=method, label=label, tables=tables[method], aborted=counts[method]) for method, label in LABELS.items()],
                   status="complete", split=split, inputs=len(units), workers=workers, **isolation.build(BINARY), commit=isolation.commit(),
                   selections={method: dict(record_sha256=(baseline_selection.options_sha256() if baseline_selection.READER or method not in SELECTED
                                                           else digest(selection_path(method))),
                                            environments={environment: dict(selected=entry["selected"], **settings(entry))
                                                          for environment, entry in chosen[method].items()}) for method in chosen},
                   base_configurations_sha256=digest(baseline_selection.BASE),
                   abort_rule=isolation.ABORT_RULE, abort_counts=counts, aborted_solves=aborted_solves(predictions),
                   starts=isolation.start_counts(scores), starts_rule=isolation.STARTS, stayed_at_start=isolation.stayed_counts(scores),
                   stayed_rule=isolation.STAYED, shared_inputs=layout_inputs.record("bregar"), **isolation.execution(workers))
    write(results / "summary.json", summary)
    for method, entries in tables.items():
        for samples, groups in entries.items():
            for label, entry in groups.items():
                for table in entry.values():
                    print(f"  {method:9s} {samples:>2s} {label}: {scoring.describe(table)}", flush=True)
    isolation.report_aborts(counts, f"Bregar {split}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "select", "check-selection", "evaluate"))
    parser.add_argument("--method", choices=SELECTED, help="the baseline whose options select or check-selection concerns")
    parser.add_argument("--split", choices=("training", "evaluation"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "select":
        if not args.method:
            parser.error("select needs --method")
        select(args.method, args.workers)
    elif args.command == "check-selection":
        for method in [args.method] if args.method else SELECTED:
            record = selection(method)
            print(f"Bregar {method} selection verified, " + ", ".join(f"{environment} {entry['selected']}" for environment, entry in record["selections"].items()),
                  flush=True)
    else:
        if not args.split:
            parser.error("evaluate needs --split")
        evaluate(args.split, args.workers)


if __name__ == "__main__":
    main()
