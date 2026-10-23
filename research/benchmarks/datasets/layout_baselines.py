"""The baselines on the same-layout halves of TIERS, with the options of Jin, FNL and STRONG selected on the training half.

The frozen campaigns ran every baseline over the whole recordings with settings that TWR-CLOUD-Bern's scans or
synthetic training inputs chose. Under the same-layout protocol the training half is real data of the deployment, so
every option of Jin 2021, FNL 2025 and STRONG 2021 that Bern chose is selected there, each method on its own among the
declared candidates of baseline_selection.py. Jin chooses its model and the aggregation of a link's readings, FNL its
model, aggregation and start, and STRONG its Huber radius, model and start. ADE-BP chooses its rounds of message passing
in the anchor aided mode between the four of its paper and the ten the campaign took from synthetic TIERS layouts. The
rule is the lowest mean over the sample tracks of the balanced S75 over the reporting cells, per mode wherever the
method runs, ties by declared order among the means within 0.01 cm of the lowest, and in the anchor free mode Jin and
FNL keep the planar model alone. The earlier selection of STRONG among 40 cm and multiples of the calibration scale kept
the largest radius, so its grid
never bracketed the optimum, and it held STRONG to the spatial model on the planar TIERS scene. Every other setting is
the declared base configuration of baseline_selection.py, never a value read from an evaluation run, and ADE-BP's
ranging model comes from the fit on the TIERS training half (ade_training.py, research/artifacts/baselines/
tiers-ade-fit.json) in place of the Case 1 fit of the campaign. Both halves then run on one machine with the pilot
binary. The training half gives the numbers the design round compares against, and the evaluation half the reported
ones.

A unit scores only the agent pairs that its measured links join, with every anchor counted as one node (the joined pair
set of scoring.py), and its error is the mean capped error over those pairs. A selection ranks the candidates under
that pair set, in the anchor only mode over the cells with three or more anchors of the headline, the only cells it
solves, and in the anchor free and anchor aided modes over every cell. Its record names the method, the rule, the
candidates of every mode, the pair set, the anchor rule, the declared base configurations and the shared inputs, and
counts per candidate the solves that stayed at their start (isolation.STAYED). A run refuses a record whose content
differs from the declared values or from the base configurations and shared inputs of the run, so a record kept from
an earlier rule, grid, base record or rule of the starts runs nothing, and it applies the declared settings of the
candidate the record names (baseline_selection.chosen). check-selection verifies the records.
Every scored row keeps the mean pair error and the pair count of every pair set of its unit
(scoring.score_with_pair_sets), so rows pair with SOLID's on (key, id, samples). Every summary reports the balanced
percentiles under the joined pairs and the supplementary ranged pairs with the share of pairs and units each scores,
the anchor only headline over three or more anchors and every smaller anchor count alone.

Every request passes through layout_inputs.levelled as SOLID's runner's does, so every method of a unit starts from the
same coordinates drawn inside SOLID's operating area of the unit, and ADE-BP receives that area.

Every adapter process runs with one thread for Rayon, OpenBLAS and OpenMP. A request that crashes the adapter is solved
again unit by unit and then method by method, which the adapter accepts as it accepts any nonempty subset of the
roster of the mode, so a single solve that crashes is scored as that method's failure on that unit with the failure
penalty and a warning, and every other solve of the request keeps its own result (pilot.run_pilot,
isolation.ABORT_RULE). Every run record counts the crashed solves of every method beside listing them, and every
summary prints the counts. Every record names the machine it ran on, the workers that solved at once and the timing
rule of its solve times (isolation.execution), and every run record the starts of every method (isolation.STARTS).

The selected options and every baseline of a half run with one build of the adapter. A run refuses a selection record
that another build made, so after a rebuild every selection is made again before any half runs (isolation.same_build).

Selection records go to research/artifacts/baselines/tiers-<method>-selection.json, and the earlier STRONG record
stays as tiers-strong-radius.json. Runs write per unit scores under out/baselines/<dataset>/<split>/<mode>, the
layout the SOLID runners use. In reader mode (SOLID_READER=1) a run takes the final options of every baseline and
ADE-BP's law of the mode from the shipped settings (baseline_selection.options) instead of selection records.

Usage: layout_baselines.py tiers select --method jin-2021|fnl|strong|ade-bp-2015 [--workers N]
       layout_baselines.py tiers check-selection [--method jin-2021|fnl|strong|ade-bp-2015]
       layout_baselines.py tiers run --split training|evaluation --mode af|aac|anc [--workers N]
"""
import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import ade_training
import baseline_selection
import isolation
import layout_inputs
import layout_split
import pilot
import scoring
import tiers_campaign as campaign

DATA = pilot.audit.DATA
ROOT = pilot.ROOT
BINARY = pilot.BINARY
OUT = ROOT / "out/baselines"
RECORDS = ROOT / "research/artifacts/baselines"
# STRONG's candidates, which the held-out runner reads, and the methods whose options every training half selects
CANDIDATES = baseline_selection.joint_candidates()
SELECTED = baseline_selection.SELECTED
SELECTION_SCHEMA = baseline_selection.SELECTION_SCHEMA
# The pair set every selection ranks under and the fewest anchors of the anchor only cells it ranks over, the headline's
PAIR_SET = scoring.PAIR_SET
HEADLINE_MIN_ANCHORS = scoring.HEADLINE_MIN_ANCHORS
RULE = ("Per mode wherever the method runs on the training half, each of Jin 2021, FNL 2025 and STRONG 2021 on its own among the "
        "declared candidates of the options TWR-CLOUD-Bern chose for it, and ADE-BP 2015 among four and ten rounds in the anchor aided "
        "mode (baseline_selection.py), by the lowest mean over the sample "
        "tracks of the balanced S75 over the reporting cells, in the anchor only mode the cells with three or more anchors of its "
        "headline and in the other modes every cell, ties by declared order among the means within 0.01 cm of the lowest. A unit's "
        "error is its mean capped pair error over the "
        "pairs its measured links join, and a unit without such a pair is left out. Every other setting is the declared base "
        "configuration.")
read, write, digest = pilot.evidence.read, pilot.evidence.write, pilot.identity


def read_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def record_path(method):
    """The TIERS selection record of a method."""
    return RECORDS / f"tiers-{method}-selection.json"


def selection_record(path):
    record = read(path)
    if record.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"{path} is not a selection record of the declared schema {SELECTION_SCHEMA}")
    return record


def ruled(record, path, rule, method, modes=None):
    """A selection record of a method, refused unless it was made by the declared rule over the method's declared
    candidates of the modes given, or of every mode that selects its options (baseline_selection.grid), ranking them
    under the headline's pair set, the joined pairs, and the anchor only mode over the headline cells of
    HEADLINE_MIN_ANCHORS or more anchors, from the declared base configurations as they are. The record's content
    decides, so a record kept from an earlier rule, grid or base record runs nothing."""
    if record.get("method") != method:
        raise ValueError(f"{path} is not a selection of {method}")
    if record.get("rule") != rule or record.get("candidates") != json.loads(json.dumps(baseline_selection.grid(method, modes))):
        raise ValueError(f"{path} was made under another rule or candidate grid than the declared ones")
    if record.get("base_configurations_sha256") != digest(baseline_selection.BASE):
        raise ValueError(f"{path} was made with other declared base configurations than those of this run")
    if record.get("pair_set") != PAIR_SET or any(entry.get("pair_set") != PAIR_SET for entry in (record.get("selections") or {}).values()):
        raise ValueError(f"{path} ranked {method}'s candidates under another pair set than the {PAIR_SET} pairs of the headline")
    if record.get("anchor_only_headline_min_anchors") != HEADLINE_MIN_ANCHORS:
        raise ValueError(f"{path} ranked the anchor only mode over other cells than those of {HEADLINE_MIN_ANCHORS} or more anchors")
    return record


def build():
    """The hash and the embedded source identity of the pilot binary."""
    return isolation.build(BINARY)


def same_build(record, path):
    """A selection record, refused unless the pilot binary that runs now made it (isolation.same_build)."""
    return isolation.same_build(record, path, BINARY)


def run_pilot(requests, workers, folder):
    """One single-threaded pilot process per request, at most `workers` at a time, with crashed solves isolated."""
    return pilot.run_pilot(requests, workers, folder, BINARY)


def aborted_solves(rows):
    """The solves that crashed the adapter, per method, from scored rows."""
    failures = defaultdict(list)
    for row in rows:
        if isolation.is_abort(row["failure"]):
            failures[row["method"]].append(dict(key=row["key"], id=row["id"], samples=row["samples"], failure=row["failure"]))
    return dict(failures)


def aborts(rows, methods=()):
    """What a run record keeps of the solves that crashed the adapter, the declared rule, their count per method with
    the methods given counted at zero, and every such solve per method."""
    return dict(abort_rule=isolation.ABORT_RULE, abort_counts=isolation.abort_counts(rows, methods), aborted_solves=aborted_solves(rows))


def scored_rows(key, request, reference, records, scorer, cell_of, extra):
    """The rows of one request's records, each with the mean pair error and the pair count of every pair set of its
    unit (scoring.score_with_pair_sets), as SOLID's runners keep them, and the starts its solver reports."""
    methods = [b["method"] for b in request["baselines"]]
    rows = []
    for s in scoring.score_with_pair_sets(request, reference, records, scorer, methods):
        rows.append(dict(key=key, id=s["id"], samples=s["samples"], method=s["method"], cell=cell_of(s), acquisition=s["acquisition"],
                         error_cm=s["mean_pair_error_cm"], elapsed_ms=s["elapsed_ms"], failure=s["failure"], starts=s["starts"],
                         failed_agents=s["failed_agents"], expected_agents=s["expected_agents"], pair_sets=s["pair_sets"],
                         stayed_at_start=s["stayed_at_start"], **extra))
    return rows


def ranked(rows, mode):
    """The rows a selection ranks, each with its error under the headline's pair set. The anchor only mode keeps the
    headline cells of HEADLINE_MIN_ANCHORS or more anchors, and a unit without a pair of the set is left out, as the
    released scorer leaves out a scenario without eligible pairs. The pair set follows from the inputs alone, so every
    candidate keeps the same units."""
    result = []
    for row in rows:
        error = row["pair_sets"][PAIR_SET]["mean_pair_error_cm"]
        if error is not None and (mode != "ANC" or row["anchors"] >= HEADLINE_MIN_ANCHORS):
            result.append(dict(row, error_cm=error))
    return result


def balanced(rows, scorer, method, samples):
    cells = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["samples"] == samples:
            cells[row["cell"]].append(row["error_cm"])
    return [float(scorer.balanced_percentile(cells, q)) for q in (0.5, 0.75, 0.9)]


def choose(rows_by_candidate, scorer, candidates=CANDIDATES, method="strong"):
    """A method's candidates ranked by the declared rule, each with its balanced percentiles per sample track and the
    solve and method settings it sets."""
    if set(rows_by_candidate) != {c["name"] for c in candidates}:
        raise ValueError(f"incomplete {method} selection population")
    tracks, mean = {}, {}
    for candidate in candidates:
        rows = rows_by_candidate[candidate["name"]]
        tracks[candidate["name"]] = {str(s): balanced(rows, scorer, method, s) for s in sorted({r["samples"] for r in rows})}
        mean[candidate["name"]] = float(np.mean([t[1] for t in tracks[candidate["name"]].values()]))
    return [dict(candidate=c["name"], solve_config=c["solve_config"], method_config=c.get("method_config") or {}, tracks=tracks[c["name"]],
                 mean_s75=round(mean[c["name"]], 9))
            for c in baseline_selection.rank_candidates(candidates, mean)]


def selection(ranking):
    return dict(selected=ranking[0]["candidate"], solve_config=ranking[0]["solve_config"], method_config=ranking[0].get("method_config") or {},
                ranking=ranking)


def write_run(directory, rows, split, mode, workers, **details):
    """scores.json and run.json of a run, the record with the build of the adapter, the commit of the checkout, the
    machine, the workers and the timing rule, the crashed solves of every method and the starts every method's solves
    report."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scores.json").write_text(json.dumps(rows) + "\n")
    methods = sorted({r["method"] for r in rows})
    write(directory / "run.json", dict(**build(), commit=isolation.commit(), split=split, mode=mode, workers=workers,
                                       units=len({(r["id"], r["samples"]) for r in rows}),
                                       methods=methods, split_record_sha256=digest(layout_split.RECORD), **aborts(rows, methods),
                                       starts=isolation.start_counts(rows), starts_rule=isolation.STARTS, stayed_at_start=isolation.stayed_counts(rows),
                                       stayed_rule=isolation.STAYED, **isolation.execution(workers), **details))


def summarize(rows, title, mode, groups=None):
    """Print per method, sample track and reported group the balanced S50, S75 and S90 under the joined pairs and the
    supplementary ranged pairs, each with the share of pairs and units it scores (scoring.summary). The anchor only mode
    reports the headline over three or more anchors and every smaller anchor count alone, unless groups splits the rows
    of a track otherwise. Last come the solves of every method that crashed the adapter, with a warning for any."""
    scorer = pilot.evidence.load_scorer()
    for method in sorted({r["method"] for r in rows}):
        tables = scoring.summary([r for r in rows if r["method"] == method], mode, scorer.balanced_percentile, groups)
        for samples, entries in tables.items():
            for label, entry in entries.items():
                for table in entry.values():
                    print(f"  {title} {method:12s} {samples:>2s} {label}: {scoring.describe(table)}", flush=True)
    isolation.report_aborts(isolation.abort_counts(rows), title)


# ------------------------------------------------------------------------------------------- TIERS
TIERS = DATA / "tiers-static-campaign"


def tiers_batches(mode, split, minimum_anchors=0):
    """The campaign base and the calibrated batches of one mode and half with at least minimum_anchors anchors."""
    base = read_gz(TIERS / "base.json.gz")
    catalog = read_gz(TIERS / "catalog.json.gz")
    blocks = layout_split.tiers_blocks(split)
    batches = [b for b in catalog if b["track"] == "calibrated" and b["mode"] == mode and b["block"] in blocks and b["anchors"] >= minimum_anchors]
    if not batches:
        raise ValueError("no TIERS batches for this mode and split")
    return base, batches


def tiers_cell(mode, batch):
    return (lambda s: s["acquisition"]) if mode == "AF" else (lambda s: f'{s["acquisition"]}/anchors-{batch["anchors"]}')


def tiers_rows(mode, split, workers, choices, folder, alone=None, ade=None, minimum_anchors=0):
    """Scored rows of the baselines on one half and mode, over the batches with at least minimum_anchors anchors. Every
    baseline of the campaign's roster takes its declared base configuration (baseline_selection.declared_baselines),
    choices the candidate of every selected method and ade ADE-BP's ranging model. alone keeps one method, the request of
    a selection run."""
    base, batches = tiers_batches(mode, split, minimum_anchors)
    built = {}
    for batch in batches:
        request, reference = campaign.batch_inputs(base, batch)
        # every method's starts and ADE-BP's area by SOLID's deployment rules, as SOLID's runner takes them
        request = layout_inputs.levelled(request, "tiers", mode)
        roster = [alone] if alone else [entry["method"] for entry in request["baselines"]]
        request["baselines"] = baseline_selection.declared_baselines(roster, mode, choices, ade)
        built[batch["key"]] = (batch, request, reference)
    records = run_pilot([(key, request) for key, (_, request, _) in built.items()], workers, folder)
    scorer = pilot.evidence.load_scorer()
    rows = []
    for key, (batch, request, reference) in built.items():
        rows.extend(scored_rows(key, request, reference, records[key], scorer, tiers_cell(mode, batch),
                                dict(level=batch["anchors"], block=batch["block"], anchors=batch["anchors"])))
    return rows


def selection_header(dataset, scope, workers, method, binary=None, rule=RULE, modes=None, **details):
    """The fields every selection record shares. method is the baseline whose options it selects over its grid of the
    modes given or of every mode (baseline_selection.grid), binary the adapter that solved the candidates, the pilot
    binary unless named, and rule the declared rule in words."""
    return dict(schema=SELECTION_SCHEMA, dataset=dataset, method=method, rule=rule, candidates=baseline_selection.grid(method, modes), scope=scope,
                split_record_sha256=digest(layout_split.RECORD), **isolation.build(BINARY if binary is None else binary), workers=workers,
                evaluation_rows_used=0, base_configurations_sha256=digest(baseline_selection.BASE), stayed_rule=isolation.STAYED,
                **isolation.execution(workers), **details)


def tiers_select(method, workers):
    """A method's options per mode on the training half, ranked under the joined pairs, the anchor only mode over its
    headline cells of three or more anchors, the only batches it solves, and the other modes over every cell. The record
    counts the solves of every candidate that crashed the adapter."""
    path = record_path(method)
    if path.exists():
        raise ValueError(f"TIERS {method} settings already selected")
    scorer = pilot.evidence.load_scorer()
    # ADE-BP runs with the ranging model of the training half, the one its runs take
    ade = ade_training.ranging_model("tiers", RECORDS / "tiers-ade-fit.json") if method == "ade-bp-2015" else None
    selections = {}
    for mode in baseline_selection.MODES[method]:
        minimum = HEADLINE_MIN_ANCHORS if mode == "ANC" else 0
        candidates = baseline_selection.candidates(method, mode)
        by_candidate, aborted, stayed = {}, {}, {}
        for candidate in candidates:
            rows = tiers_rows(mode, "training", workers, {method: candidate}, OUT / "tiers/work" / f"training-{mode.lower()}",
                              alone=method, ade=ade, minimum_anchors=minimum)
            aborted[candidate["name"]] = isolation.abort_counts(rows, (method,))[method]
            stayed[candidate["name"]] = isolation.stayed_counts(rows).get(method)
            by_candidate[candidate["name"]] = ranked(rows, mode)
            print(f"TIERS {mode} {method} candidate {candidate['name']} scored", flush=True)
        if len({len(rows) for rows in by_candidate.values()}) != 1:
            raise ValueError(f"the candidates of {method} in {mode} keep different units under the {PAIR_SET} pairs")
        ranking = choose(by_candidate, scorer, candidates, method)
        kept = by_candidate[candidates[0]["name"]]
        selections[mode] = dict(selection(ranking), pair_set=PAIR_SET, units=len(kept), cells=sorted({r["cell"] for r in kept}), abort_counts=aborted,
                                stayed_at_start=stayed)
        print(f"TIERS {mode} {method}: " + ", ".join(f"{r['candidate']} {r['mean_s75']:.1f}" for r in ranking) + f" -> {ranking[0]['candidate']}",
              flush=True)
        isolation.report_aborts(aborted, f"TIERS {mode} {method} candidates")
    RECORDS.mkdir(parents=True, exist_ok=True)
    write(path, dict(selection_header("tiers", "mode", workers, method, campaign_freeze_sha256=digest(TIERS / "freeze.json"),
                                      pair_set=PAIR_SET, anchor_only_headline_min_anchors=HEADLINE_MIN_ANCHORS, abort_rule=isolation.ABORT_RULE,
                                      shared_inputs=layout_inputs.record("tiers")), selections=selections))


def tiers_selection(method):
    """The TIERS selection record of a method, refused unless it is TIERS' own, made by the declared rule over the
    method's candidates under the headline's pair set from the declared base configurations (ruled), with the shared
    inputs of this run (layout_inputs.record) and by the adapter that runs now (same_build)."""
    path = record_path(method)
    record = selection_record(path)
    if record.get("dataset") != "tiers":
        raise ValueError(f"{path} is not a TIERS selection")
    ruled(record, path, RULE, method)
    if record.get("shared_inputs") != json.loads(json.dumps(layout_inputs.record("tiers"))):
        raise ValueError(f"{path} was made with other shared inputs than those of this run (layout_inputs.RULE)")
    return same_build(record, path)


def tiers_run(split, mode, workers):
    """Every baseline of the mode on one half, each method whose options the mode selects with its selected candidate
    and ADE-BP with the ranging model of the training half. In reader mode every method takes its final option of the
    mode and ADE-BP its law from the shipped settings (baseline_selection.options)."""
    ade_path = RECORDS / "tiers-ade-fit.json"
    if baseline_selection.READER:
        choices = baseline_selection.options("tiers", mode)
        ade = baseline_selection.law("tiers", mode) if "ade-bp-2015" in choices else None
        selections, ade_fit = baseline_selection.kept(choices, mode), None
    else:
        selected = [method for method in SELECTED if baseline_selection.selects(method, mode)]
        records = {method: tiers_selection(method) for method in selected}
        choices = {method: record["selections"][mode] for method, record in records.items()}
        ade = ade_training.ranging_model("tiers", ade_path)
        selections = {method: dict(selected=entry["selected"], solve_config=entry["solve_config"], method_config=entry["method_config"],
                                   record_sha256=digest(record_path(method))) for method, entry in choices.items()}
        ade_fit = digest(ade_path)
    rows = tiers_rows(mode, split, workers, choices, OUT / "tiers/work" / f"{split}-{mode.lower()}", ade=ade)
    write_run(OUT / "tiers" / split / mode.lower(), rows, split, mode, workers, dataset="tiers", track="calibrated",
              selections=selections, base_configurations_sha256=digest(baseline_selection.BASE),
              campaign_freeze_sha256=digest(TIERS / "freeze.json"), ade_ranging_model=ade if mode != "AF" else None, ade_fit_sha256=ade_fit,
              pair_set=PAIR_SET, shared_inputs=layout_inputs.record("tiers"))
    summarize(rows, f"TIERS {mode} {split}", mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("tiers",))
    parser.add_argument("command", choices=("select", "check-selection", "run"))
    parser.add_argument("--method", choices=SELECTED, help="the baseline whose options select or check-selection concerns")
    parser.add_argument("--split", choices=("training", "evaluation"))
    parser.add_argument("--mode", choices=("af", "aac", "anc"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    if args.command == "select":
        if not args.method:
            parser.error("tiers select needs --method")
        tiers_select(args.method, args.workers)
    elif args.command == "check-selection":
        for method in [args.method] if args.method else SELECTED:
            record = tiers_selection(method)
            print(f"TIERS {method} selection verified, " + ", ".join(f"{mode} {entry['selected']}" for mode, entry in record["selections"].items()),
                  flush=True)
    else:
        if not (args.split and args.mode):
            parser.error("tiers run needs --split and --mode")
        tiers_run(args.split, args.mode.upper(), args.workers)
    print(f"done in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
