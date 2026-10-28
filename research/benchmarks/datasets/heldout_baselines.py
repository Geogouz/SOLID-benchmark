"""The baselines on both halves of the held-out benchmarks HYMN and AAU Meshed UWB, anchor only.

Both constellations have real height spread. Jin 2021, FNL 2025 and STRONG 2021 run, and ADE-BP 2015, which has no
spatial implementation, runs in the planar form of its paper with its anchor only candidate and the error law fitted on
the training half, model 2 on both datasets, which the shipped settings name (settings/baseline-options.json,
baseline_selection.options). Every baseline takes its declared base configuration (baseline_selection.py,
baseline-configurations.json), never a value read from an evaluation run. The options that TWR-CLOUD-Bern's scans once
chose are selected on the training half of each dataset by the rule of the other datasets (RULE), each method on its
own among its declared candidates. Jin chooses its model and the aggregation of a link's readings, FNL its model,
aggregation and start, and STRONG its Huber radius, model and start. The ranking takes the headline cells, three or
more anchors, under the pairs the measured links join, the figure the methods are reported on, so only the training
units of those cells are solved for a selection, and every record names the cells, the pair set and the anchor rule.
A candidate's start is the start of its solve, the unit's starting coordinates for the provided start and the engine's
geometric layout of the unit for the geometry start, as on every other dataset. ADE-BP alone takes the operating area a
unit carries.

Every scored row keeps the mean pair error and the pair count of every pair set of its unit
(scoring.score_with_pair_sets), as SOLID's rows do, and pairs with them on (key, id, samples).

Every request goes to a single-threaded heldout-adapter process, and a solve that crashes the adapter is recorded as its
method's failure on that unit with the failure penalty (pilot.run_pilot, isolation.ABORT_RULE). Every selection record
and run record counts the crashed solves of every method or candidate and names the starts its solves report. The
adapter outputs are cached per request under out/baselines/<dataset>/work/<split>, so a run resumes where it stopped.
Every command runs only after the method freeze and the ledger allow a read of its half (heldout.require_freeze), and
every selection record must name the training inputs, the freeze, the adapter and the declared base configurations it
was selected with. A run of either half also needs the records of held-out stage one committed as the ledger closed
them (heldout.require_stage_one) and the committed ledger to open its dataset (heldout.require_run), so the ledger
records every run.

Selection records go to research/artifacts/baselines/<dataset>-<method>-selection.json. Runs write per unit scores of
every cell and a run record under out/baselines/<dataset>/<split>/anc, and print the headline and every separate cell
under the joined pairs and the supplementary ranged pairs with the share of pairs and units each scores, and the flights
the build excluded.

check-selection verifies that the selection records belong to this declaration, rule, candidate grid, pair set and
headline cells, to the training half as built now, to the method freeze, to the adapter as it is and to the declared
base configurations as they are, which every run also requires. In reader mode (SOLID_READER=1) a run takes the final
option of every method and ADE-BP's law from the shipped settings instead of selection records, and the method freeze,
the ledger and the build of our records bind nothing (heldout.require_freeze).

Usage: heldout_baselines.py hymn|aau select --method jin-2021|fnl|strong [--workers N]
       heldout_baselines.py hymn|aau check-selection [--method jin-2021|fnl|strong]
       heldout_baselines.py hymn|aau run --split training|evaluation [--workers N]
"""
import argparse
import json
import subprocess
import time

import baseline_selection
import heldout
import isolation
import layout_baselines
import pilot
import scoring

ROOT = heldout.ROOT
BINARY = ROOT / "research/target/release/heldout-adapter"
OUT = ROOT / "out/baselines"
RECORDS = ROOT / "research/artifacts/baselines"
# The baselines that run on scenes with real height spread, ADE-BP in its planar form, in the order of the final tables,
# and the methods whose options the training half selects
METHODS = ("jin-2021", "ade-bp-2015", "fnl", "strong")
MODE = "ANC"
SELECTED = tuple(method for method in METHODS if baseline_selection.selects(method, MODE))
CHUNK = 8
SELECTION_SCHEMA = layout_baselines.SELECTION_SCHEMA
RULE = ("On the training half of each held-out dataset, in the anchor only mode, each of Jin 2021, FNL 2025 and STRONG 2021 on its own "
        "among the declared candidates of the options TWR-CLOUD-Bern chose for it (baseline_selection.py), by the lowest mean over the "
        "two tracks of the balanced S75 over the headline cells of three or more anchors, ties by declared order among the means within "
        "0.01 cm of the lowest. A unit's error is its mean capped pair error over the pairs its measured links join, and a unit without "
        "such a pair is left out. Every other setting is the declared base configuration.")


def baselines(choices, methods=METHODS, law=None):
    """The baselines of a request in the order of the roster, every one from the declared base configurations, each
    selected method with its candidate in choices and ADE-BP with its law (baseline_selection.declared_baselines)."""
    return baseline_selection.declared_baselines(list(methods), MODE, choices, law)


def inputs(dataset, split):
    """The request, the reference and the identity of one built half, the evaluation half only after the freeze."""
    return heldout.built_half(dataset, split)


def requests(dataset, split, units, entries):
    chunks = [units[i:i + CHUNK] for i in range(0, len(units), CHUNK)]
    return [(f"{dataset}-{split}-{index:05d}", dict(baselines=entries, units=chunk)) for index, chunk in enumerate(chunks)]


def solve(dataset, split, request, reference, entries, workers, folder):
    """Scored rows of the baselines on every unit of a half, each with the mean pair error and the pair count of every
    pair set of its unit and the starts its solver reports."""
    built = requests(dataset, split, request["units"], entries)
    records = pilot.run_pilot(built, workers, folder, BINARY)
    scorer = pilot.evidence.load_scorer()
    methods = [b["method"] for b in entries]
    metadata = {m["id"]: m for m in reference["units"]}
    rows = []
    for key, part in built:
        for s in scoring.score_with_pair_sets(part, reference, records[key], scorer, methods):
            # The key of a unit is the one the SOLID runner writes, so the rows of every method pair on (key, id, samples).
            rows.append(dict(key=f"{dataset}-{s['id']}", request=key, id=s["id"], samples=s["samples"], method=s["method"],
                             cell=s["acquisition"], acquisition=s["acquisition"], anchors=metadata[s["id"]]["anchors"],
                             error_cm=s["mean_pair_error_cm"], elapsed_ms=s["elapsed_ms"], failure=s["failure"], starts=s["starts"],
                             stayed_at_start=s["stayed_at_start"], failed_agents=s["failed_agents"], expected_agents=s["expected_agents"],
                             pair_sets=s["pair_sets"]))
    return rows


def binary_identity():
    return dict(binary_sha256=pilot.identity(BINARY), binary_identity=subprocess.check_output([str(BINARY), "--identity"], text=True).strip())


def selection_path(dataset, method):
    """The selection record of a method on a held-out dataset."""
    return RECORDS / f"{dataset}-{method}-selection.json"


def select(dataset, method, workers):
    """A method's options on the training half of a dataset, every candidate solved alone over the units of the
    headline cells and ranked under the joined pairs by RULE. The record counts the solves of every candidate that
    crashed the adapter."""
    freeze = heldout.require_freeze("training", dataset=dataset)
    if method not in SELECTED:
        raise ValueError(f"{method} has no options to select on the held-out benchmarks")
    path = selection_path(dataset, method)
    if path.exists():
        raise ValueError(f"the {dataset} options of {method} are already selected")
    request, reference, identity = inputs(dataset, "training")
    # The candidates are ranked over the headline cells, the figure the methods are reported on, so only their units are solved.
    cells, separate = heldout.headline(dataset)
    metadata = {m["id"]: m for m in reference["units"]}
    units = [unit for unit in request["units"] if metadata[unit["id"]]["acquisition"] in cells]
    if {metadata[unit["id"]]["acquisition"] for unit in units} != set(cells):
        raise ValueError(f"the {dataset} training half lacks a headline cell")
    evaluation_rows = heldout.evaluation_units(dataset, [metadata[identity] for identity in sorted({unit["id"] for unit in units})])
    if evaluation_rows:
        raise ValueError(f"{evaluation_rows} units of the {dataset} training half read an evaluation point or flight")
    candidates = baseline_selection.candidates(method, MODE)
    scorer = pilot.evidence.load_scorer()
    by_candidate, aborted, stayed = {}, {}, {}
    for candidate in candidates:
        # the candidate's settings, its start among them, go into the request of the method alone
        rows = solve(dataset, "training", dict(units=units), reference, baselines({method: candidate}, (method,)), workers,
                     OUT / dataset / "work" / "training")
        aborted[candidate["name"]] = isolation.abort_counts(rows, (method,))[method]
        stayed[candidate["name"]] = isolation.stayed_counts(rows).get(method)
        # the headline's pair set, the pairs the measured links join, ranks the candidates
        by_candidate[candidate["name"]] = layout_baselines.ranked(rows, MODE)
        print(f"{dataset} {method} candidate {candidate['name']} scored", flush=True)
    if len({len(rows) for rows in by_candidate.values()}) != 1:
        raise ValueError(f"the candidates of {method} keep different units of {dataset} under the {layout_baselines.PAIR_SET} pairs")
    ranking = layout_baselines.choose(by_candidate, scorer, candidates, method)
    print(f"{dataset} {method}: " + ", ".join(f"{r['candidate']} {r['mean_s75']:.1f}" for r in ranking) + f" -> {ranking[0]['candidate']}", flush=True)
    isolation.report_aborts(aborted, f"{dataset} {method} candidates")
    RECORDS.mkdir(parents=True, exist_ok=True)
    pilot.evidence.write(path, dict(schema=SELECTION_SCHEMA, dataset=dataset, method=method, rule=RULE, candidates=baseline_selection.grid(method, (MODE,)),
                                    scope=(f"dataset, the anchor only mode, the headline cells of {heldout.HEADLINE_MINIMUM_ANCHORS} or "
                                           f"more anchors under the {layout_baselines.PAIR_SET} pairs"),
                                    pair_set=layout_baselines.PAIR_SET, anchor_only_headline_min_anchors=heldout.HEADLINE_MINIMUM_ANCHORS,
                                    cells=cells, cells_not_ranked=separate, selection_units=len({unit["id"] for unit in units}),
                                    declaration_sha256=heldout.sha256(heldout.RECORD), training=identity,
                                    base_configurations_sha256=heldout.sha256(baseline_selection.BASE), workers=workers,
                                    evaluation_rows_used=evaluation_rows, abort_rule=isolation.ABORT_RULE, stayed_rule=isolation.STAYED, **freeze,
                                    **binary_identity(), **isolation.execution(workers),
                                    selections={MODE: dict(layout_baselines.selection(ranking), pair_set=layout_baselines.PAIR_SET,
                                                           units=len(by_candidate[candidates[0]["name"]]), abort_counts=aborted,
                                                           stayed_at_start=stayed)}))


def training_identity(dataset):
    """The identity of the built training half as heldout.built_half gives it, every key of it, the identity select
    records, so that a selection is checked against the half as it is built now."""
    return heldout.built_half(dataset, "training")[2]


def selection(dataset, method):
    """The selection record of a method on a dataset, refused unless it was selected under this declaration, rule,
    candidate grid and pair set from the declared base configurations as they are (layout_baselines.ruled) over the
    headline cells on the training half as built now, under the current method freeze and with the adapter as it is."""
    path = selection_path(dataset, method)
    record = layout_baselines.selection_record(path)
    if record.get("dataset") != dataset or record.get("declaration_sha256") != heldout.sha256(heldout.RECORD):
        raise ValueError(f"{path.name} was not selected under this declaration")
    layout_baselines.ruled(record, path.name, RULE, method, (MODE,))
    if record.get("cells") != heldout.headline(dataset)[0]:
        raise ValueError(f"{path.name} was not selected over the headline cells")
    if record.get("training") != heldout.canonical(training_identity(dataset)):
        raise ValueError(f"the {dataset} training half changed after the selection of {method}")
    if not heldout.FREEZE.is_file() or record.get("freeze_record_sha256") != heldout.sha256(heldout.FREEZE):
        raise ValueError(f"{path.name} was selected under another method freeze than {heldout.relative(heldout.FREEZE)}")
    if {key: record.get(key) for key in ("binary_sha256", "binary_identity")} != binary_identity():
        raise ValueError(f"{path.name} was selected with another heldout-adapter than the one of this run")
    return record


def configured(dataset):
    """Jin, FNL and STRONG for a run, each with the candidate its selection chose, ADE-BP with its anchor only candidate
    and law from the shipped settings, and what the run record keeps of the choices. In reader mode every method takes
    its final option from the shipped settings (baseline_selection.options)."""
    settled = baseline_selection.options(dataset, MODE)
    if baseline_selection.READER:
        choices = settled
        kept = baseline_selection.kept(choices, MODE)
    else:
        records = {method: selection(dataset, method) for method in SELECTED}
        choices = {method: record["selections"][MODE] for method, record in records.items()}
        kept = {method: dict(selected=entry["selected"], solve_config=entry["solve_config"], method_config=entry.get("method_config") or {},
                             record_sha256=heldout.sha256(selection_path(dataset, method)), training_selected_on=records[method]["training"],
                             selected_with=records[method].get("binary_identity")) for method, entry in choices.items()}
        choices["ade-bp-2015"] = settled["ade-bp-2015"]
        kept.update(baseline_selection.kept({"ade-bp-2015": settled["ade-bp-2015"]}, MODE))
    law = baseline_selection.law(dataset, MODE)
    return baselines(choices, law=law), dict(selections=kept, base_configurations_sha256=heldout.sha256(baseline_selection.BASE))


def run(dataset, split, workers):
    freeze = heldout.require_run(dataset, split)
    heldout.require_stage_one()
    folder = OUT / dataset / split / "anc"
    entries, details = configured(dataset)
    request, reference, identity = inputs(dataset, split)
    started = time.monotonic()
    rows = solve(dataset, split, request, reference, entries, workers, OUT / dataset / "work" / split)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scores.json").write_text(json.dumps(rows) + "\n")
    methods = sorted({r["method"] for r in rows})
    pilot.evidence.write(folder / "run.json", dict(dataset=dataset, split=split, mode="ANC", workers=workers, units=len({(r["id"], r["samples"]) for r in rows}),
                                                   methods=methods, commit=isolation.commit(), declaration_sha256=heldout.sha256(heldout.RECORD),
                                                   abort_rule=isolation.ABORT_RULE, abort_counts=isolation.abort_counts(rows, methods),
                                                   aborted_solves=layout_baselines.aborted_solves(rows), starts=isolation.start_counts(rows),
                                                   starts_rule=isolation.STARTS, stayed_at_start=isolation.stayed_counts(rows),
                                                   stayed_rule=isolation.STAYED, seconds=round(time.monotonic() - started, 1),
                                                   headline=dict(zip(("cells", "separate_rows"), heldout.headline(dataset))),
                                                   pair_set=layout_baselines.PAIR_SET,
                                                   **identity, **freeze, **details, **binary_identity(), **isolation.execution(workers)))
    layout_baselines.summarize(rows, f"{dataset} ANC {split}", "ANC", groups=lambda part: heldout.report_groups(part, dataset))
    if identity["excluded_flights"]:
        print(f"{dataset} {split}: excluded by the relocation guard " + ", ".join(f"{flight} ({entry['change_cm']} cm, {entry['units']} units)"
                                                                            for flight, entry in identity["excluded_flights"].items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=heldout.DATASETS)
    parser.add_argument("command", choices=("select", "check-selection", "run"))
    parser.add_argument("--method", choices=SELECTED, help="the baseline whose options select or check-selection concerns")
    parser.add_argument("--split", choices=heldout.SPLITS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "run" and not args.split:
        parser.error("run needs --split")
    if args.command == "select" and not args.method:
        parser.error("select needs --method")
    # Every command waits for the committed method freeze and the ledger's opening of its step, for either half.
    heldout.require_freeze(args.split or "training", dataset=args.dataset)
    started = time.monotonic()
    if args.command == "select":
        select(args.dataset, args.method, args.workers)
    elif args.command == "check-selection":
        for method in [args.method] if args.method else SELECTED:
            record = selection(args.dataset, method)
            print(f"{args.dataset} {method} selection verified, {record['selections'][MODE]['selected']}", flush=True)
    else:
        run(args.dataset, args.split, args.workers)
    print(f"done in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
