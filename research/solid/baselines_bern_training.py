"""The baselines on the eleven held Bern training scans, scored like SOLID's held-scan rounds, and the selection of
the options of Jin, FNL and STRONG there.

Every baseline runs on the same inputs as SOLID's rounds, in the cooperative modes the generated training scenarios of
each held scan with every link calibrated by that fold's law and in the anchor only mode the campaign's fold calibrated
scenarios of the held scan. Every setting comes from the declared base configurations
(research/benchmarks/datasets/baseline-configurations.json), never from an evaluation run. The options of Jin 2021,
FNL 2025 and STRONG 2021 that TWR-CLOUD-Bern's scans once chose are selected here by the rule of the other datasets
(baseline_selection.BERN_RULE). select --method solves every declared candidate of the method (baseline_selection.py) in
every mode it runs on every held scan, ranks the candidates by the lowest mean over the two sample tracks of the
balanced S75 under the joined pairs over the cells of held scan, anchor count and agent count, the anchor only mode
over its headline units of three or more anchors, ties by declared order among the means within 0.01 cm of the
lowest, and writes research/artifacts/baselines/bern-<method>-selection.json with the crashed solves of every
candidate, the solves of every candidate that stayed at their start (isolation.STAYED) and the hash of the CLI. In the
anchor free mode Jin and FNL keep the planar model alone. The standing run of every method takes the declared settings
of the candidate its selection chose for each mode (baseline_selection.chosen), from the stored candidate runs, and
run_bern.py takes it on the evaluation corpus. A record of another rule, grid, pair set, base record or build of the
CLI runs nothing. ADE-BP keeps its fixed options and takes the ranging model and area fitted on the ten other training scans
under the same fold law (research/artifacts/baselines/bern-ade-fit.json, ade_training.bern_config), never a fit that
saw the held scan. The
corpora are solved by sharded.py in single-threaded CLI processes with every solve that crashes the CLI isolated and
scored as that method's failure. The released scorer scores every run and the balanced percentiles pool all held
scans with equal weight per scan, anchor count and agent count cell, the cells of SOLID's rounds. The released figure
is the benchmark's own. Beside it every run is scored from the same predictions over the pairs its measured links join,
with every anchor counted as one node, the pair set the rounds decide on (research/benchmarks/datasets/scoring.py),
and over the supplementary ranged pairs, those joined by links that carry a range. The record reports both figures
over the same cells with the share of pairs and units each scores. The record makes the training-side standing of
every method available to design decisions without touching evaluation data, and it names the machine of the run.
In reader mode (SOLID_READER=1) every method with selected options takes its final option of the mode from the shipped
settings (settings/baseline-options.json, baseline_selection.options) instead of a selection record, the options its
selection on these scans chose.

Usage: baselines_bern_training.py [run] [--workers 12] [--concurrent 1] [--methods strong,jin-2021,ade-bp-2015,fnl] [--freeze-only]
       baselines_bern_training.py select --method jin-2021|fnl|strong [--workers 12] [--concurrent 1]
       baselines_bern_training.py check-selection [--method jin-2021|fnl|strong]
"""
import argparse
import csv
import gzip
import json
import math
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_bern  # noqa: E402
import scoring  # noqa: E402

BENCH = common.ROOT / "research/benchmarks/bern"
sys.path.insert(0, str(BENCH))
import ade_training  # noqa: E402
import baseline_selection  # noqa: E402
import isolation  # noqa: E402
import sharded  # noqa: E402

SCHEMA = "solid.selection.bern-baselines-training.v4"
WORK = common.OUT / "unify/baselines-training-v3"
CLI = common.ROOT / "research/target/release/solid"
SETTINGS = json.loads((BENCH / "baselines.json").read_text())
MODES = ("af", "aac", "anc")
ADE_FITS = ade_training.RECORDS / "bern-ade-fit.json"
# the methods whose options the held scans select, their records and the rule, pair set and anchor count they rank by
SELECTED = baseline_selection.SELECTED
RECORDS = ade_training.RECORDS
PAIR_SET = scoring.PAIR_SET
HEADLINE_MIN_ANCHORS = scoring.HEADLINE_MIN_ANCHORS
# The inputs of SOLID's held-scan rounds, the generated training scenarios of the held scans
# (select_bern_cooperative.ARCHIVE) and the campaign's fold calibrated anchor only scenarios
# (ablate_bern_anchor_only.CAMPAIGN), and the released scorer (select_bern.SCORER), named here so that a run needs none
# of the design modules that name them too
ARCHIVE = common.OUT / "unify/bern-training-scenarios-v2.zip"
CAMPAIGN = common.BERN_ANCHOR_ONLY_CORPUS
SCORER = common.DATA / "inputs/0068d72b94db6b1c/score_twr_cloud_bern.py"


def fold_archive(scan):
    """The held scan's generated training scenarios with every link calibrated by the fold's law, ids scan * 10000 + n."""
    coefficients, calibration_path = fit_bern.calibration(scan)
    path = WORK / f"scan-{scan:02d}/calibrated.zip"
    manifest = path.with_suffix(".manifest.json")
    identity = dict(archive_sha256=common.sha256(ARCHIVE), calibration_sha256=common.sha256(calibration_path),
                    source_sha256=common.sha256(Path(fit_bern.__file__)), script_sha256=common.sha256(Path(__file__)))
    if common.cache_matches(manifest, identity, [path]):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.zip")
    with zipfile.ZipFile(ARCHIVE) as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
        names = sorted(n for n in source.namelist() if n.startswith(f"scan-{scan:02d}/") and n.endswith(".json"))
        for name in names:
            scenario = json.loads(source.read(name))
            fit_bern.check_scenario(scenario, scan)
            for iteration in scenario["Iterations"]:
                for field in ("Ranging samples between Agents", "Ranging samples to Anchors"):
                    links = iteration.get(field) or {}
                    for key, value in links.items():
                        links[key] = {samples: (dict(calibrated_samples=fit_bern.calibrated_packets((value or {}).get(samples), coefficients))
                                                if (value or {}).get(samples) is not None else None) for samples in ("1", "10")}
            number = int(Path(name).stem)
            target.writestr(f"{scan * 10000 + number}.json", json.dumps(scenario))
    temporary.replace(path)
    common.seal_cache(manifest, identity, [path])
    return path


def fold_config_path(scan):
    return WORK / f"scan-{scan:02d}/ade-bp-2015-fold.json"


def fold_config(scan):
    """ADE-BP's configuration for a held scan, fitted on the ten other training scans, written where the CLI reads it."""
    path = fold_config_path(scan)
    config = ade_training.bern_config(scan, ade_training.bern_record(ADE_FITS))
    if not path.exists() or common.read_json(path) != config:
        common.write_json(path, config)
    return path


def selection_path(method):
    """The TWR-CLOUD-Bern selection record of a method."""
    return RECORDS / f"bern-{method}-selection.json"


def selection(method):
    """The Bern selection record of a method, refused unless its content holds the declared rule, candidates, pair set
    and anchor rule and the CLI that runs now made it (baseline_selection.bern_record)."""
    path = selection_path(method)
    if not path.exists():
        raise ValueError(f"select the Bern options of {method} with baselines_bern_training.py select --method {method} first ({path})")
    return baseline_selection.bern_record(common.read_json(path), path, method, common.sha256(CLI), PAIR_SET, HEADLINE_MIN_ANCHORS)


def selected(method, mode):
    """The declared candidate a method's selection chose for a mode, refused when the entry stores other settings, in
    reader mode the final option of the shipped settings (baseline_selection.options)."""
    if baseline_selection.READER:
        return baseline_selection.chosen(method, mode.upper(), baseline_selection.options("bern", mode.upper())[method])
    return baseline_selection.chosen(method, mode.upper(), selection(method)["selections"][mode.upper()])


def summarize_tracks(scores, scorer):
    """The balanced S50, S75 and S90 of released scores per sample track over the cells of held scan, anchor count and
    agent count, with the mean per anchor count (select_bern.summarize_tracks, the rule of SOLID's rounds)."""
    tracks = {}
    for samples in (1, 10):
        cells, levels = defaultdict(list), defaultdict(list)
        for (scan, scenario, iteration, q), (error, anchors, agents) in scores.items():
            if q == samples:
                cells[(scan, anchors, agents)].append(error)
                levels[anchors].append(error)
        tracks[str(samples)] = dict(units=sum(map(len, cells.values())), cells=len(cells),
                                    balanced=[scorer.balanced_percentile(cells, p) for p in (.5, .75, .9)],
                                    per_anchor_mean={str(k): float(np.mean(v)) for k, v in sorted(levels.items())})
    return tracks


def selecting(method):
    """The modes, lower case, in which a baseline runs on Bern and its options are selected."""
    return [mode for mode in modes_of(method) if baseline_selection.selects(method, mode.upper())]


def method_flags(scan, method, mode, candidate=None):
    """The CLI's method flags of one baseline and mode on one held scan and the configuration file they name. A method
    with options to select in the mode runs the given candidate, or else the candidate its selection chose. Jin, FNL and
    STRONG take the declared base configurations, and ADE-BP the fold's fitted configuration with the rounds of its
    candidate in the anchor aided mode and the four of its paper in the anchor only mode. None for a mode the method
    does not run."""
    spec = SETTINGS["methods"][method]
    chosen = spec["modes"].get(mode.upper())
    if chosen is None:
        return None, None
    if baseline_selection.selects(method, mode.upper()):
        candidate = candidate or selected(method, mode)
    if method != "ade-bp-2015":
        entry, = baseline_selection.declared_baselines([method], mode.upper(), {method: candidate})
        config = baseline_selection.write_config(WORK / "configs" / f"{method}-{candidate['name']}.json", entry)
        return baseline_selection.cli_flags(entry, config), config
    flags = ["--method", method, "--solver-iterations", "10000", "--time-budget-ms", "0", "--dimensions", chosen["dimensions"],
             "--aggregation", spec["aggregation"]]
    config = fold_config_path(scan)
    if not config.exists():
        raise ValueError(f"write ADE-BP's configuration of held scan {scan} before its runs")
    if candidate is not None:
        fold = common.read_json(config)
        config = baseline_selection.write_config(config.with_name(f"{config.stem}-{candidate['name']}.json"),
                                                 dict(method_config=dict(fold, **candidate["method_config"])))
    return flags + ["--method-config", str(config)], config


def run(scan, method, mode, workers, scenarios, scorer, candidate=None):
    """The released scores of one baseline on one held scan and mode, {(scan, scenario, iteration, samples): (error,
    anchors, agents)}, the same units under the released rule and the joined and ranged pair sets from the same
    predictions, {key: dict(anchors, agents, released, joined, ranged)} (scoring.stored_run), and the timing record of
    the run with its crashed solves and starts. A method with options to select runs the given candidate, or else its
    selected one, below <method>-<mode>-<candidate>, so the standing run of the selected candidate is the candidate run
    of its selection. None for a mode the baseline does not run."""
    if baseline_selection.selects(method, mode.upper()):
        candidate = candidate or selected(method, mode)
    flags, config = method_flags(scan, method, mode, candidate)
    if flags is None:
        return None
    directory = WORK / f"scan-{scan:02d}" / (f"{method}-{mode}" if candidate is None else f"{method}-{mode}-{candidate['name']}")
    directory.mkdir(parents=True, exist_ok=True)
    # the results depend on the inputs alone, not on the number of processes, so the workers stay out of the identity
    identity = dict(binary_sha256=common.sha256(CLI), scenarios_sha256=common.sha256(scenarios), flags=flags, mode=mode,
                    config_sha256=common.sha256(config) if config else None, threads=isolation.SINGLE_THREAD,
                    scorer_sha256=common.sha256(SCORER), source_sha256=common.sha256(Path(__file__)),
                    runner_sha256=common.sha256(Path(sharded.__file__)), scoring_sha256=common.sha256(Path(scoring.__file__)))
    output, joined = directory / "scenario_scores.csv.gz", directory / "joined_scores.csv.gz"
    manifest = directory / "manifest.json"
    if not common.cache_matches(manifest, identity, [output, joined]):
        with tempfile.TemporaryDirectory(prefix="run-", dir=directory) as temporary:
            stage = Path(temporary)
            started = time.time()
            report = sharded.benchmark(CLI, scenarios, flags, mode, stage, workers, work=directory / "shards")
            with zipfile.ZipFile(stage / "predictions.zip") as archive:
                archive.extract("predictions.csv", stage)
            # the coordinate rule of every scorer on the extracted copy the released scorer reads (scoring.scored_point)
            scoring.bound_prediction_csv(stage / "predictions.csv")
            subprocess.run([sys.executable, str(SCORER), "score-predictions", "--scenarios", str(scenarios), "--predictions",
                            str(stage / "predictions.csv"), "--output-dir", str(stage), "--modes", mode.upper(), "--samples", "1,10"],
                           check=True, capture_output=True)
            units = scoring.stored_run(scenarios, stage / "predictions.csv", mode, scorer, stage / "scenario_scores.csv.gz")
            scoring.write_scenario_units(stage / "joined_scores.csv.gz", units)
            (stage / "scenario_scores.csv.gz").replace(output)
            (stage / "joined_scores.csv.gz").replace(joined)
            common.write_json(directory / "timing.json", dict(seconds=time.time() - started, workers=workers, solves=report["solves"],
                                                               aborted_solves=report["aborted_solves"], abort_counts=report["abort_counts"],
                                                               starts=report["starts"], stayed_at_start=(report.get("stayed_at_start") or {}).get(method),
                                                               **isolation.execution(workers)))
        common.seal_cache(manifest, identity, [output, joined])
    rows = {}
    with gzip.open(output, "rt") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] == mode.upper() and int(row["q"]) == 2:
                error = float(row["mean_pair_error"])
                if not math.isfinite(error):
                    raise ValueError(f"non finite score in {output}")
                rows[(scan, int(row["scenario_id"]), int(row["iteration"]), int(row["samples"]))] = (error, int(row["anchors"]), int(row["agents"]))
    units = {(scan, *key): unit for key, unit in scoring.read_scenario_units(joined).items()}
    if set(units) != set(rows):
        raise ValueError(f"{joined} does not hold the units of {output}")
    return rows, units, common.read_json(directory / "timing.json")


def modes_of(method):
    """The modes a baseline runs in on Bern, lower case."""
    return [mode for mode in MODES if SETTINGS["methods"][method]["modes"].get(mode.upper())]


def choose(by_candidate, candidates, mode, scorer):
    """The candidates of one mode ranked by BERN_RULE from their units on every held scan, {candidate: {(scan,
    scenario, iteration, samples): unit}}. A unit's error is its mean capped pair error over the joined pairs, a unit
    without such a pair is left out and the anchor only mode keeps its headline units of three or more anchors. The
    units enter the cells of held scan, anchor count and agent count in the order of their keys, so the ranking is a
    function of the units alone, and every candidate must keep the same units."""
    kept, tracks, mean = None, {}, {}
    for candidate in candidates:
        rows = [(key, unit) for key, unit in sorted(by_candidate[candidate["name"]].items())
                if unit[PAIR_SET][1] is not None and (mode != "anc" or unit["anchors"] >= HEADLINE_MIN_ANCHORS)]
        if kept is None:
            kept = [key for key, _ in rows]
        elif [key for key, _ in rows] != kept:
            raise ValueError(f"the candidates of {mode} keep different units under the {PAIR_SET} pairs")
        tracks[candidate["name"]] = {}
        for samples in (1, 10):
            cells = defaultdict(list)
            for key, unit in rows:
                if key[-1] == samples:
                    cells[(key[0], unit["anchors"], unit["agents"])].append(unit[PAIR_SET][1])
            if not cells:
                raise ValueError(f"no {PAIR_SET} units of {mode} with {samples} samples")
            tracks[candidate["name"]][str(samples)] = float(scorer.balanced_percentile(cells, 0.75))
        mean[candidate["name"]] = float(np.mean(list(tracks[candidate["name"]].values())))
    ranking = [dict(candidate=c["name"], solve_config=c["solve_config"], method_config=c.get("method_config") or {}, s75_by_samples=tracks[c["name"]],
                    mean_s75=round(mean[c["name"]], 9)) for c in baseline_selection.rank_candidates(candidates, mean)]
    units = {str(samples): sum(key[-1] == samples for key in kept) for samples in (1, 10)}
    return dict(selected=ranking[0]["candidate"], solve_config=ranking[0]["solve_config"], method_config=ranking[0]["method_config"],
                pair_set=PAIR_SET, units=units, ranking=ranking)


def inputs():
    """The hashes of every input of the held-scan runs, the archives, the anchor only scenarios of every held scan, the
    released scorer and the sources."""
    return dict(archive_sha256=common.sha256(ARCHIVE), scorer_sha256=common.sha256(SCORER),
                anchor_only_scenarios={f"scan-{s:02d}": common.sha256(CAMPAIGN / f"scan-{s:02d}/calibrated.zip") for s in fit_bern.SCAN_IDS},
                source={p.name: common.sha256(p) for p in (Path(__file__), Path(fit_bern.__file__), Path(sharded.__file__), Path(baseline_selection.__file__))})


def select(method, workers, concurrent=1):
    """A method's options per mode it runs in, every declared candidate of the mode on every held scan, ranked by
    BERN_RULE. The record counts the solves of every candidate that crashed the CLI and those that stayed at their
    start."""
    if method not in SELECTED:
        raise ValueError(f"{method} has no options to select")
    path = selection_path(method)
    if path.exists():
        raise ValueError(f"the Bern options of {method} are already selected ({path})")
    modes, per_run, scorer = selecting(method), max(1, workers // concurrent), common.load_scorer()
    candidates = {mode: baseline_selection.candidates(method, mode.upper()) for mode in modes}
    cli = common.sha256(CLI)
    units = {(mode, c["name"]): {} for mode in modes for c in candidates[mode]}
    aborts, stays = {key: 0 for key in units}, {key: [] for key in units}
    WORK.mkdir(parents=True, exist_ok=True)
    for scan in fit_bern.SCAN_IDS:
        scenarios = fold_archive(scan)
        if method == "ade-bp-2015":
            fold_config(scan)
        anchor_scenarios = CAMPAIGN / f"scan-{scan:02d}/calibrated.zip"
        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            futures = {(mode, c["name"]): pool.submit(run, scan, method, mode, per_run, anchor_scenarios if mode == "anc" else scenarios, scorer, c)
                       for mode in modes for c in candidates[mode]}
            for key, future in futures.items():
                _, scanned, timing = future.result()
                units[key].update(scanned)
                aborts[key] += len(timing["aborted_solves"])
                stays[key].append(timing.get("stayed_at_start"))
        print(f"scan {scan}: {len(futures)} candidate runs of {method} scored", flush=True)
    if common.sha256(CLI) != cli:
        raise ValueError("the CLI changed during the selection")
    selections = {}
    for mode in modes:
        entry = choose({c["name"]: units[(mode, c["name"])] for c in candidates[mode]}, candidates[mode], mode, scorer)
        selections[mode.upper()] = dict(entry, abort_counts={c["name"]: aborts[(mode, c["name"])] for c in candidates[mode]},
                                        stayed_at_start={c["name"]: isolation.summed_stays(stays[(mode, c["name"])]) for c in candidates[mode]})
        print(f"Bern {mode.upper()} {method}: " + ", ".join(f"{r['candidate']} {r['mean_s75']:.1f}" for r in entry["ranking"]) + f" -> {entry['selected']}",
              flush=True)
        isolation.report_aborts(selections[mode.upper()]["abort_counts"], f"Bern {mode.upper()} {method} candidates")
    common.write_json(path, dict(schema=baseline_selection.SELECTION_SCHEMA, dataset="bern", method=method, scope="mode", rule=baseline_selection.BERN_RULE,
                                 candidates=baseline_selection.grid(method, [mode.upper() for mode in modes]), pair_set=PAIR_SET,
                                 anchor_only_headline_min_anchors=HEADLINE_MIN_ANCHORS, scans=list(fit_bern.SCAN_IDS), stayed_rule=isolation.STAYED,
                                 samples=[1, 10], iterations=[1, 2], cli_sha256=cli, base_configurations_sha256=common.sha256(baseline_selection.BASE),
                                 evaluation_rows_used=0, inputs=inputs(), selections=selections, abort_rule=isolation.ABORT_RULE,
                                 workers=workers, concurrent=concurrent, workers_per_run=per_run, host=isolation.host(),
                                 execution=isolation.execution(workers)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=("run", "select", "check-selection"))
    parser.add_argument("--method", choices=SELECTED, help="the baseline whose options select or check-selection concerns")
    parser.add_argument("--workers", type=int, default=12, help="single-threaded CLI processes in total")
    parser.add_argument("--concurrent", type=int, default=1, help="runs positioned at the same time per scan")
    parser.add_argument("--methods", default=",".join(SETTINGS["methods"]))
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.command == "select":
        if not args.method:
            parser.error("select needs --method")
        select(args.method, args.workers, args.concurrent)
        return
    if args.command == "check-selection":
        for method in [args.method] if args.method else SELECTED:
            record = selection(method)
            print(f"Bern {method} selection verified, " + ", ".join(f"{mode} {entry['selected']}" for mode, entry in record["selections"].items()),
                  flush=True)
        return
    methods = args.methods.split(",")
    per_run = max(1, args.workers // args.concurrent)
    WORK.mkdir(parents=True, exist_ok=True)
    runs = [(method, mode) for method in methods for mode in modes_of(method)]
    # every method with options to select runs its selected candidate of each mode, in reader mode the final option of
    # the shipped settings
    chosen = {m: {mode: selected(m, mode)["name"] for mode in selecting(m)} for m in methods if m in SELECTED}
    source = {m: baseline_selection.OPTIONS if baseline_selection.READER else selection_path(m) for m in chosen}
    freeze = dict(schema=SCHEMA, scans=list(fit_bern.SCAN_IDS), samples=[1, 10], iterations=[1, 2], runs=[f"{m}-{mode}" for m, mode in runs],
                  settings={m: SETTINGS["methods"][m] for m in methods}, base_configurations_sha256=common.sha256(baseline_selection.BASE),
                  selections={m: dict(record=str(source[m].relative_to(common.ROOT) if source[m].is_relative_to(common.ROOT) else source[m]),
                                      sha256=common.sha256(source[m]), selected=chosen[m]) for m in chosen},
                  ade_fold_fits=dict(record=str(ADE_FITS.relative_to(common.ROOT)), sha256=common.sha256(ADE_FITS)) if "ade-bp-2015" in methods else None,
                  inputs=dict(cooperative="the held scan's generated training scenarios with every link calibrated by the fold's law",
                              anchor_only="the campaign's fold calibrated scenarios of the held scan"),
                  anchor_only_scenarios={f"scan-{s:02d}": common.sha256(CAMPAIGN / f"scan-{s:02d}/calibrated.zip") for s in fit_bern.SCAN_IDS},
                  archive_sha256=common.sha256(ARCHIVE), binary_sha256=common.sha256(CLI), scorer_sha256=common.sha256(SCORER),
                  concurrent=args.concurrent, workers_per_run=per_run, workers=args.workers, threads_per_process=1, host=isolation.host()["node"],
                  source={p.name: common.sha256(p) for p in (Path(__file__), Path(fit_bern.__file__), Path(sharded.__file__), Path(ade_training.__file__),
                                                             Path(baseline_selection.__file__))})
    freeze_path = WORK / "freeze.json"
    if freeze_path.exists() and common.read_json(freeze_path) != freeze:
        raise ValueError("Frozen run changed. Review the differences before starting another run.")
    common.write_json(freeze_path, freeze)
    if args.freeze_only:
        archive = fold_archive(1)
        print(f"freeze written to {freeze_path.relative_to(common.ROOT)}, fold 1 archive {archive.relative_to(common.ROOT)}")
        return
    scorer = common.load_scorer()
    scores, units = {key: {} for key in runs}, {key: {} for key in runs}
    aborts, starts, stays = {key: 0 for key in runs}, {key: {} for key in runs}, {key: [] for key in runs}
    done, total = 0, len(fit_bern.SCAN_IDS) * len(runs)
    for scan in fit_bern.SCAN_IDS:
        scenarios = fold_archive(scan)
        if "ade-bp-2015" in methods:
            fold_config(scan)
        anchor_scenarios = CAMPAIGN / f"scan-{scan:02d}/calibrated.zip"
        with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
            futures = {key: pool.submit(run, scan, key[0], key[1], per_run, anchor_scenarios if key[1] == "anc" else scenarios, scorer) for key in runs}
            for key, future in futures.items():
                rows, scanned, timing = future.result()
                scores[key].update(rows)
                units[key].update(scanned)
                aborts[key] += len(timing["aborted_solves"])
                for count, solves in ((timing.get("starts") or {}).get(key[0]) or {}).items():
                    starts[key][count] = starts[key].get(count, 0) + solves
                stays[key].append(timing.get("stayed_at_start"))
                done += 1
                common.write_json(WORK / "progress.json", dict(done=done, total=total, complete=done == total))
                print(f"{done}/{total} scan {scan} {key[0]} {key[1]}: {len(scores[key])} units so far", flush=True)
    results, pair_set_results = {}, {}
    for (method, mode), rows in scores.items():
        results.setdefault(method, {})[mode] = summarize_tracks(rows, scorer)
        # the same cells, scan, anchor count and agent count, under the joined and the supplementary ranged pair sets
        tables = scoring.scenario_tables(units[(method, mode)], scorer.balanced_percentile, cell=lambda key, unit: (key[0], unit["anchors"], unit["agents"]))
        pair_set_results.setdefault(method, {})[mode] = {samples: {name: entry[name] for name in scoring.REPORTED} for samples, entry in tables.items()}
    abort_counts, start_counts, stayed_counts = {}, {}, {}
    for (method, mode), count in aborts.items():
        abort_counts.setdefault(method, {})[mode] = count
        start_counts.setdefault(method, {})[mode] = dict(sorted(starts[(method, mode)].items(), key=lambda item: int(item[0])))
        stayed_counts.setdefault(method, {})[mode] = isolation.summed_stays(stays[(method, mode)])
    record = dict(freeze, freeze_sha256=common.sha256(freeze_path), results=results, pair_set=scoring.PAIR_SET,
                  supplementary_pair_sets=list(scoring.SUPPLEMENTARY), pair_set_results=pair_set_results, complete=True,
                  abort_rule=isolation.ABORT_RULE, abort_counts=abort_counts, starts=start_counts, starts_rule=isolation.STARTS,
                  stayed_at_start=stayed_counts, stayed_rule=isolation.STAYED, execution=isolation.execution(args.workers))
    common.write_json(common.SELECTIONS / "bern-baselines-training.json", record)
    for method, modes in results.items():
        for mode, tracks in modes.items():
            b1, b10 = tracks["1"]["balanced"], tracks["10"]["balanced"]
            print(f"  {method:12s} {mode:3s} one sample {b1[0]:6.1f}/{b1[1]:6.1f}/{b1[2]:6.1f}   ten samples {b10[0]:6.1f}/{b10[1]:6.1f}/{b10[2]:6.1f}")
            for samples, entry in pair_set_results[method][mode].items():
                for table in entry.values():
                    print(f"  {'':12s} {mode:3s} {samples}: {scoring.describe(table)}")
            print(f"  {'':12s} {mode:3s} {abort_counts[method][mode]} solves crashed the CLI and are scored as failures")


if __name__ == "__main__":
    main()
