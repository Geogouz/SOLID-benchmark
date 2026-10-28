"""Run one method and mode on the TWR-CLOUD-Bern evaluation corpus and score it with the released scorer.

Jin 2021, FNL 2025 and STRONG 2021 take the candidate of the mode that their selection on the eleven held training scans
chose (research/artifacts/baselines/bern-<method>-selection.json, written by research/solid/baselines_bern_training.py
select), with every other setting from the declared base configurations (research/benchmarks/datasets/
baseline-configurations.json), and the run refuses a record of another rule, grid, pair set, base record or build of
the CLI. A run applies the declared settings of the candidate the record names and refuses an entry that stores others
(baseline_selection.chosen). Their method settings go to method-config.json in the output directory, and command.json
keeps the record's SHA-256, the selected candidate and the SHA-256 of the declared base configurations. ADE-BP keeps
the fitted configuration of baselines.json, in the anchor aided mode with the rounds its selection chose, and every
baseline the input archive baselines.json names. SOLID takes its tracked profiles, which must hold the
composition of the method record research/artifacts/solid/selections/method.json once the confirmatory round has
confirmed the method, and of the pre-round fits before it. ADE-BP's configuration
holds the ranging model fitted on all eleven training scans under the engine's conversion of raw timings
(research/benchmarks/datasets/ade_training.py). The corpus is solved by sharded.py in single-threaded CLI processes, at
most --workers at a time, with every solve that crashes the CLI isolated and scored as that method's failure. The
output directory, RUNS/<method>/<mode> unless --output names another, receives benchmark.json, predictions.zip,
command.json with the machine of the run and the commit of the checkout, the shards under shards/ and the official
scorer outputs under official/. The final tables accept a run of the evaluation corpus only at a commit that holds the
declared analysis and the method freeze. In reader mode (SOLID_READER=1) every baseline takes its final option of the
mode from the shipped settings (settings/baseline-options.json, baseline_selection.options) instead of a selection
record, ADE-BP's configuration must hold the law and area those settings give, and SOLID's tracked profile is read as
it is written. The input archives lie in the data folder SOLID_DATA, research/data unless it names another.
The evaluation runs are written there, and research/solid/final_tables.py reads them from there. planned gives the command and the method settings of a run
without writing, the form the final tables' selection check compares a finished run with.

The released scorer's figure is the benchmark's own and stays the reported one. With --score the run is also scored
over the agent pairs its measured links join, with every anchor counted as one node, the pair set of every other
dataset (research/benchmarks/datasets/scoring.py), and over the supplementary ranged pairs, those joined by links that
carry a range. Both figures are reported beside the released one with the share of pairs and units each rule scores
(joined). In the anchor free mode the released and joined rules score the released components. These scores go to
joined/ beside official/.

Usage: run_bern.py --method solid|jin-2021|ade-bp-2015|fnl|strong --mode af|aac|anc [--output DIR] [--limit N --selection-seed S --workers W]
"""
import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "datasets"))
import official_scorer  # noqa: E402
import sharded  # noqa: E402
import evidence  # noqa: E402
import baseline_selection  # noqa: E402
import scoring  # noqa: E402

BINARY = ROOT / "research/target/release/solid"
SOLID = ROOT / "research/solid"
SETTINGS = json.loads((HERE / "baselines.json").read_text())
JOINED_SCHEMA = "solid.bern.joined-score.v1"
RECORDS = ROOT / "research/artifacts/baselines"
# the folder of the evaluation runs of every baseline, <RUNS>/<method>/<mode> below the repository, which stage 4 of the
# node pipeline writes and research/solid/final_tables.py reads
RUNS = "out/bern-baselines-v3"


def verified(path):
    """A tracked SOLID profile, refused when it no longer holds the composition of the fit records and deployment rules.
    A reader's reproduction reads the profile as it is written."""
    if baseline_selection.READER:
        return path
    if str(SOLID) not in sys.path:
        sys.path.insert(0, str(SOLID))
    import profiles
    profiles.resolve(path.name, path)
    return path


def solid_provenance(method, flags=()):
    """What the run record of SOLID keeps of the method: its anchor only budgets with the hash of their record, the
    method record with its hash, the digest of the sources the CLI is built from, which it embeds in its report, and
    the tracked profile the run solves with its SHA-256, the file the last of the CLI's flags names (planned)."""
    if method != "solid":
        return {}
    if str(SOLID) not in sys.path:
        sys.path.insert(0, str(SOLID))
    import deployment
    import profiles
    profile = Path(flags[-1])
    return dict(budget=deployment.budget_provenance(), method_record=profiles.method_provenance(),
                sources=sharded.isolation.source_identity(ROOT), profile=str(profile.relative_to(ROOT)),
                profile_sha256=official_scorer.file_sha256(profile))


def selection_path(method):
    """The TWR-CLOUD-Bern selection record of a method."""
    return RECORDS / f"bern-{method}-selection.json"


def selection(method):
    """The Bern selection record of a method, refused unless its content holds the declared rule, candidates, pair set
    and anchor rule and the CLI that runs now made it (baseline_selection.bern_record)."""
    path = selection_path(method)
    if not path.exists():
        raise ValueError(f"select the Bern options of {method} with research/solid/baselines_bern_training.py select --method {method} first ({path})")
    record = json.loads(path.read_text())
    return baseline_selection.bern_record(record, path, method, official_scorer.file_sha256(BINARY), scoring.PAIR_SET, scoring.HEADLINE_MIN_ANCHORS)


def planned(method, mode, output):
    """The corpus, the CLI's method flags of one method and mode, what the run record keeps of a selection and the
    method settings a method with selected options runs with from output/method-config.json, which its flags name, None
    for every other method. It writes nothing, so a check of a finished run compares the run's command and
    method-config.json with it. A method with selected options takes the declared candidate its record names
    (baseline_selection.chosen). The flag that names a SOLID profile comes last."""
    archives = SETTINGS["archives"]
    if method == "solid":
        spec = SETTINGS["solid"][mode]
        verified(ROOT / (spec["profile"] if mode == "ANC" else spec["profiles"]))
        scenarios = DATA / archives[spec["calibration"]]
        extra = ["--deployment-profile", str(ROOT / spec["profile"])] if mode == "ANC" else ["--cooperative-profiles", str(ROOT / spec["profiles"])]
        return scenarios, ["--method", method, "--solver-iterations", "10000", "--time-budget-ms", "0", *extra], None, None
    spec = SETTINGS["methods"][method]
    chosen = spec["modes"][mode]
    scenarios = DATA / archives[chosen["calibration"]]
    # in reader mode the final options of the mode from the shipped settings, in place of the selection records
    settled = baseline_selection.options("bern", mode) if baseline_selection.READER else None
    if baseline_selection.selects(method, mode):
        entry = settled[method] if settled is not None else selection(method)["selections"][mode]
        candidate = baseline_selection.chosen(method, mode, entry)
        path = baseline_selection.OPTIONS if settled is not None else selection_path(method)
        config = Path(output) / "method-config.json"
        chosen_by = dict(record=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), sha256=official_scorer.file_sha256(path),
                         selected=candidate["name"], base_configurations_sha256=official_scorer.file_sha256(baseline_selection.BASE))
        if method == "ade-bp-2015":
            # ADE-BP keeps its fitted configuration of baselines.json with the rounds its selection chose
            fitted = dict(json.loads((HERE / "configs" / chosen["config"]).read_text()), **candidate["method_config"])
            if settled is not None:
                fitted = settled_ade(fitted, mode, candidate)
            flags = ["--method", method, "--solver-iterations", "10000", "--time-budget-ms", "0", "--dimensions", chosen["dimensions"],
                     "--aggregation", spec["aggregation"], "--method-config", str(config)]
            return scenarios, flags, dict(chosen_by, method_config=fitted), fitted
        baseline, = baseline_selection.declared_baselines([method], mode, {method: entry})
        return scenarios, baseline_selection.cli_flags(baseline, config), dict(chosen_by, solve_config=baseline["solve_config"],
                                                                                 method_config=baseline["method_config"]), baseline["method_config"]
    extra = ["--dimensions", chosen["dimensions"], "--aggregation", spec["aggregation"]]
    if chosen.get("config"):
        if settled is not None and method == "ade-bp-2015":
            # the CLI reads the shipped configuration, which must hold the law, area and settings of the final option
            stored = json.loads((HERE / "configs" / chosen["config"]).read_text())
            candidate = baseline_selection.chosen(method, mode, settled[method])
            if settled_ade(stored, mode, candidate) != stored:
                raise ValueError(f"configs/{chosen['config']} holds other settings than ADE-BP's final option in {mode}")
        extra += ["--method-config", str(HERE / "configs" / chosen["config"])]
    if "robust_delta" in spec:
        extra += ["--robust-delta", str(spec["robust_delta"])]
    return scenarios, ["--method", method, "--solver-iterations", "10000", "--time-budget-ms", "0", *extra], None, None


def settled_ade(config, mode, candidate):
    """ADE-BP's configuration of the CLI in reader mode, its fitted law and its area those of the shipped settings for
    the mode (baseline_selection.law, bern_area) and its rounds those of its final option, refused unless the stored
    configuration holds the same law and area, so that the settings decide what runs."""
    law, area = baseline_selection.law("bern", mode), baseline_selection.bern_area()
    if config.get("ranging_model") != law or config.get("area") != area or spec_aggregation(candidate) != SETTINGS["methods"]["ade-bp-2015"]["aggregation"]:
        raise ValueError(f"ADE-BP's configuration in {mode} holds another law, area or aggregation than the shipped settings")
    return dict(config, ranging_model=law, area=area, **candidate["method_config"])


def spec_aggregation(candidate):
    """The CLI's flag of the aggregation an ADE-BP candidate sets, the median of the declared base configuration unless it
    names another."""
    return baseline_selection.AGGREGATION_FLAGS[candidate["solve_config"].get("aggregation", "Median")]


def arguments(method, mode, output):
    """The corpus, the CLI's method flags of one method and mode and what the run record keeps of a selection
    (planned), with the method settings of a method with selected options written to output/method-config.json, which
    its flags name."""
    scenarios, flags, chosen, settings = planned(method, mode, output)
    if settings is not None:
        baseline_selection.write_config(Path(output) / "method-config.json", dict(method_config=settings))
    return scenarios, flags, chosen


def joined(output, method, mode):
    """The run scored under the joined and ranged pair sets beside the released rule, from the predictions the released
    scorer read.

    Every unit is scored again by the released scorer's arithmetic over the agent pairs its measured links join and
    over those its ranged links join (scoring.score_scenarios). The released errors of the same predictions must equal
    the released scorer's scenario scores unit for unit and its balanced percentiles track for track, so every figure
    comes from one run. Writes output/joined/scenario_scores.csv.gz with every error of every unit and
    output/joined/summary.json with, per sample track, the balanced S50, S75 and S90 of every rule over the released
    reporting cells and the share of pairs and units each scores. Returns the summary."""
    official = output / "official"
    scenarios, scorer = DATA / SETTINGS["archives"]["scorer_scenarios"], evidence.load_scorer()
    with tempfile.TemporaryDirectory(prefix=".joined-", dir=output) as temporary:
        with zipfile.ZipFile(output / "predictions.zip") as archive:
            archive.extract("predictions.csv", temporary)
        units = scoring.stored_run(scenarios, Path(temporary) / "predictions.csv", mode, scorer, official / "scenario_scores.csv.gz")
    tracks = scoring.scenario_tables(units, scorer.balanced_percentile)
    released = official_scorer.read_track_summary(official / "track_summary.csv", mode)
    for samples, entry in tracks.items():
        if entry["released"]["balanced"] != [released[samples][f"balanced_s{q}_cm"] for q in (50, 75, 90)]:
            raise ValueError(f"the released tables of {output} differ from the released scorer's track summary")
    folder = output / "joined"
    scoring.write_scenario_units(folder / "scenario_scores.csv.gz", units)
    summary = dict(schema=JOINED_SCHEMA, method=method, mode=mode, pair_set=scoring.PAIR_SET, supplementary_pair_sets=list(scoring.SUPPLEMENTARY),
                   rule=("The released rule is the benchmark's own and the reported figure. The joined pair set scores the agent pairs "
                         "that the unit's measured links join, directly or through other agents or anchors, with every anchor counted "
                         "as one node, by the released scorer's arithmetic, and the supplementary ranged pair set those that links "
                         "carrying a range join. Units without such a pair are left out of the set's percentiles, and each rule "
                         "reports its pairs over every agent pair and its units with a pair over all units."),
                   tracks=tracks, predictions_sha256=official_scorer.file_sha256(output / "predictions.zip"),
                   scenarios_sha256=official_scorer.file_sha256(scenarios), scorer_sha256=official_scorer.SCORER_SHA256,
                   scoring_sha256=official_scorer.file_sha256(Path(scoring.__file__)),
                   scenario_scores_sha256=official_scorer.file_sha256(folder / "scenario_scores.csv.gz"))
    staged = folder / "summary.json.partial"
    staged.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    staged.replace(folder / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["solid", *SETTINGS["methods"]])
    parser.add_argument("--mode", required=True, choices=["af", "aac", "anc"])
    parser.add_argument("--output", type=Path, help=f"the output directory, {RUNS}/<method>/<mode> unless given")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8, help="single-threaded CLI processes at a time")
    parser.add_argument("--score", action="store_true", help="apply the released scorer after the run, complete corpus only")
    args = parser.parse_args()
    mode = args.mode.upper()
    args.output = args.output or ROOT / RUNS / args.method / args.mode
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios, flags, chosen = arguments(args.method, mode, args.output)
    command = [str(BINARY), "benchmark", "--scenarios", str(scenarios), "--mode", args.mode, "--sample-counts", "1,10", "--iterations", "1,2", *flags]
    (args.output / "command.json").write_text(json.dumps(dict(method=args.method, mode=mode, scenarios=str(scenarios), command=command,
                                                               runner="sharded.py", workers=args.workers, limit=args.limit,
                                                               selection_seed=args.selection_seed, commit=sharded.isolation.commit(ROOT),
                                                               **sharded.isolation.execution(args.workers),
                                                               **({"selection": chosen} if chosen else {}), **solid_provenance(args.method, flags)),
                                                          indent=1) + "\n")
    print(" ".join(command), flush=True)
    sharded.benchmark(BINARY, scenarios, flags, args.mode, args.output, args.workers, limit=args.limit, seed=args.selection_seed)
    if args.score:
        official_scorer.score(
            scorer=DATA / SETTINGS["archives"]["scorer"], scenarios=DATA / SETTINGS["archives"]["raw"],
            scorer_scenarios=DATA / SETTINGS["archives"]["scorer_scenarios"], predictions=args.output / "predictions.zip",
            benchmark_report=args.output / "benchmark.json", method=args.method, mode=mode,
            output_directory=args.output / "official", binary=BINARY, command=command, force=True)
        print((args.output / "official/track_summary.csv").read_text())
        for samples, entry in joined(args.output, args.method, mode)["tracks"].items():
            for table in entry.values():
                print(f"{args.method} {mode} samples={samples} {scoring.describe(table)}", flush=True)


if __name__ == "__main__":
    main()
