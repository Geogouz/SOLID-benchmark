"""The baselines on both IDLab halves, one machine and one binary for every method, with the options of Jin, FNL and
STRONG selected on the training half.

The evaluation half is the frozen campaign (idlab-campaign), the training half the campaign built from the reserved
calibration rows (idlab-training-campaign), with identical construction. Each batch's request is built by the
campaign's own batch_inputs, its roster restricted to the chosen baselines, and solved by the idlab-layout-adapter binary
(the frozen adapter with a subset roster). Every baseline takes its declared base configuration (baseline_selection.py,
baseline-configurations.json), never a value read from an evaluation run, and every method with selected options its
selected candidate. Every request passes through layout_inputs.levelled as SOLID's runner's does, so every method of a
unit starts from the same coordinates drawn inside SOLID's operating area of the unit, and ADE-BP receives that area.
ADE-BP's ranging model comes from its fit on the IDLab training half (ade_training.py,
research/artifacts/baselines/idlab-ade-fit.json) in place of the one the campaigns froze. Predictions are kept per
batch, compressed, so any scoring rule can be applied later without solving again. Beside them every run keeps the
capped error of every agent pair of every unit by the released scorer's arithmetic, and every row it writes carries the
unit's mean pair error under every pair set a SOLID row keeps (scoring.pair_sets), the declared ones and the
supplementary ranged pairs among them. Rows of runs made before these fields have neither. The summary of a run reports
the joined pairs of the headline and the ranged pairs with the share of pairs and units each scores.

Every batch is cached by its request, the binary's hash and the thread settings, so a run resumes where it stopped.
Every adapter runs with one thread for Rayon, OpenBLAS and OpenMP and the batches run in parallel processes.

Jin's SDP solver panics on a few IDLab units, and the release build aborts on a panic, which ends the adapter with the
whole batch. Such a batch is solved again unit by unit, then method by method, and the solve that aborts is recorded as
that method's failure on that unit with a warning, as the adapter records an error a method returns
(isolation.solve_request, isolation.ABORT_RULE). The scorer then charges it the failure penalty. Every other record is
the adapter's own output. run.json counts the solves of every method that aborted the adapter and lists them, and the
summary of a run prints the counts.

A run replaces the rows of its methods in scores.json and keeps those of the other methods, so a method whose inputs or
settings changed runs again alone. run.json holds one entry per method with its binary, settings, machine, workers, the
timing rule of its solve times (isolation.execution) and the starts its solves report (isolation.STARTS). Every method
of a half runs with one build of the adapter. A run keeps the rows of another method only when they come from the same
build, the hash and source identity of the adapter that runs, and it refuses any other before it solves, so a rebuild
sends every method back to run with the new build (kept). --fresh starts a half without the rows of the methods a run
leaves out, the first run after a rebuild. A run takes only selection records that the same build made.

The options of Jin 2021, FNL 2025 and STRONG 2021 that TWR-CLOUD-Bern chose, and ADE-BP's rounds of message passing
in the anchor aided mode, follow the rule of the other datasets (layout_baselines.py). Per mode wherever the method
runs, each method on its own among its declared candidates (baseline_selection.py), Jin its model and the aggregation of
a link's readings, FNL its model, aggregation and start, STRONG its Huber radius among 10, 20, 40, 80 and 160 cm and no
robust loss, its model and its start, and ADE-BP four or ten rounds in the anchor aided mode, is selected by the lowest
mean over the two sample tracks of the balanced S75 over the radio configuration and anchor count cells, ties by
declared order among the means within 0.01 cm of the lowest. In the anchor free mode Jin and FNL keep the planar model,
since without anchors the spatial one changes no solve. select --method solves the method alone under every candidate
over the batches whose key hash is zero modulo 8, about one eighth of the half spread over every configuration and
anchor count. A candidate run writes below <method>-<candidate>/ and its cache is keyed by the method and the
candidate. Every selection entry counts per candidate, and every run entry per method, the solves that stayed at their
start (isolation.STAYED).

The IDLab headline scores the pairs that the unit's measured links join, with every anchor counted as one node
(joined), and a unit's error is its mean capped error over those pairs. A selection ranks the candidates under both
pair sets from the stored per pair errors and records both rankings, every pair (all, the released scorer's rule in
the anchored modes) beside the joined pairs. The declared pair set PAIR_SET, the joined pairs of the headline, picks
the selected candidate. The anchor only headline counts the units with three or more anchors, so the anchor only
selection ranks the candidates over those cells alone.

A method with selected options always runs, without a named candidate, with the selected candidate of its mode, read
from research/artifacts/baselines/idlab-<method>-selection.json. run-selected runs a method so over every batch of each
half and writes its rows beside those of the other methods. A run takes only a selection made under the declared rule
and grid, from the declared base configurations and with the shared inputs of the run (layout_inputs.record), and
chosen under the headline's pair set, and it applies the declared settings of the candidate the selection names
(baseline_selection.chosen). select --pair-set all moves the recorded choice to the ranking under every
pair, a diagnostic that no run takes until the choice moves back to the joined pairs. check-selection verifies the
records. In reader mode (SOLID_READER=1) a run takes the final option of every method of each mode and ADE-BP's law of
the mode, model 2 in the anchor aided mode and model 1 in the anchor only mode, from the shipped settings
(baseline_selection.options) instead of selection records and the fit record.

Output: out/baselines/idlab/<split>/<mode>/{predictions/<key>-<hash>.jsonl.gz, pair-errors/<key>-<hash>.jsonl.gz,
scores.json, run.json}, the same below <method>-<candidate>/ for a candidate run.

Usage: idlab_layout_baselines.py --split training|evaluation --methods jin-2021,fnl,strong,ade-bp-2015
                                 [--modes af,aac,anc] [--workers 192] [--stride 1] [--candidate NAME] [--fresh]
       idlab_layout_baselines.py select --method jin-2021|fnl|strong|ade-bp-2015 [--workers 192] [--pair-set all|joined]
       idlab_layout_baselines.py run-selected --method jin-2021|fnl|strong|ade-bp-2015 [--splits training,evaluation] [--workers 192]
       idlab_layout_baselines.py check-selection [--method jin-2021|fnl|strong|ade-bp-2015]
"""
import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ade_training  # noqa: E402
import baseline_selection  # noqa: E402
import idlab_campaign as campaign  # noqa: E402
import isolation  # noqa: E402
import layout_baselines  # noqa: E402
import layout_inputs  # noqa: E402
import pilot  # noqa: E402
import scoring  # noqa: E402

DATA = pilot.audit.DATA
ROOT = pilot.ROOT
BINARY = ROOT / "research/target/release/idlab-layout-adapter"
OUT = ROOT / "out/baselines/idlab"
CAMPAIGNS = {"evaluation": "idlab-campaign", "training": "idlab-training-campaign"}
PERMITTED = {"AF": ("jin-2021", "fnl"), "AAC": ("jin-2021", "fnl", "strong", "ade-bp-2015"), "ANC": ("jin-2021", "fnl", "strong", "ade-bp-2015")}
RECORDS = ade_training.RECORDS
ADE_FIT = RECORDS / "idlab-ade-fit.json"
# The methods whose options the training half selects, each among its declared candidates in declared order, per mode
# it runs in, on the training batches whose key hash is zero modulo the stride.
SELECTED = baseline_selection.SELECTED
SELECTION_MODES = baseline_selection.MODES
SELECTION_STRIDE = 8
# The pair sets a unit's error can be taken over, in declared order. PAIR_SET, the headline's pair set, decides.
PAIR_SETS = {
    "all": "Every agent pair of the unit, the released scorer's rule in the anchor aided and anchor only modes.",
    "joined": ("The agent pairs that the unit's measured links join, directly or through other agents or anchors, with every "
               "anchor counted as one node. An observation joins its two ends whenever it exists. In the anchor only mode both "
               "agents need a measured anchor link."),
}
PAIR_SET = "joined"
# the anchor only headline counts the units with at least this many anchors, the cells the selection ranks over
HEADLINE_MIN_ANCHORS = 3
RULE = ("Per mode wherever the method runs on the IDLab training half, each of Jin 2021, FNL 2025 and STRONG 2021 on its own among "
        "the declared candidates of the options TWR-CLOUD-Bern chose for it, and ADE-BP 2015 among four and ten rounds in the anchor "
        "aided mode (baseline_selection.py), by the lowest mean over the sample tracks of the balanced S75 over the radio configuration "
        "and anchor count cells, ties by declared order among the means within 0.01 cm of the lowest. The batches are those whose key "
        "hash is zero modulo the stride, and the balance runs over the cells they hold, in the anchor only mode the cells with three or "
        "more anchors of its headline and in the other modes every cell. A unit's error is its mean capped pair error over the pairs "
        "the measured links join, the headline's pair set, and the ranking under every declared pair set is recorded. Every other "
        "setting is the declared base configuration.")
# The campaign base, loaded once in the parent and inherited by the forked workers.
STATE = {}
SINGLE_THREAD = isolation.SINGLE_THREAD
write = pilot.evidence.write


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


encoded = isolation.encoded
aborted = isolation.aborted


def in_stride(key, stride):
    return stride <= 1 or int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % stride == 0


def read_lines(path):
    return [json.loads(line) for line in gzip.decompress(Path(path).read_bytes()).decode().splitlines()]


def write_lines(path, values):
    """Compressed JSON lines, staged beside the target and moved in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.partial")
    staged.write_bytes(gzip.compress(b"".join(encoded(value) + b"\n" for value in values), mtime=0))
    staged.replace(path)


def replace_text(path, text):
    """A text file staged beside the target and moved in place, so a run stopped while writing leaves the earlier file
    whole. scores.json holds the rows of every method of a half, and another adapter no longer finds their cache."""
    staged = path.with_name(f"{path.name}.partial")
    staged.write_text(text)
    staged.replace(path)


def solved(request, path):
    """The adapter's prediction lines for a request, isolating any solve that aborts the adapter."""
    return isolation.solve_request(BINARY, request, path)


def selection_path(method):
    """The IDLab selection record of a method."""
    return RECORDS / f"idlab-{method}-selection.json"


def chosen(entries, mode, methods, ade_model=None, choices=None):
    """The chosen baselines as the requests carry them, in the order of the campaign's roster. Every one takes its
    declared base configuration, every method with selected options its candidate in choices and ADE-BP the ranging
    model of the training half (baseline_selection.declared_baselines)."""
    roster = [b["method"] for b in entries if b["method"] in methods]
    return baseline_selection.declared_baselines(roster, mode, choices or {}, ade_model)


def baselines(base_settings, mode, methods, ade_model, choices=None):
    """The chosen baselines of a mode as the requests carry them."""
    return chosen(base_settings[mode]["baselines"], mode, methods, ade_model, choices)


def batch_inputs(batch):
    """A batch's request and reference by the campaign's own construction, with every method's starts and ADE-BP's area
    by SOLID's deployment rules (layout_inputs.levelled), as SOLID's runner takes them."""
    built, reference = campaign.batch_inputs(STATE["base"], batch)
    return layout_inputs.levelled(built, "idlab", batch["mode"]), reference


def ade_of(mode):
    """ADE-BP's ranging model of a mode of the loaded half, the fit on the training half, or in reader mode the law of
    the mode from the shipped settings. None when the half was loaded without ADE-BP."""
    ade = STATE.get("ade")
    return ade.get(mode) if ade is not None and baseline_selection.READER else ade


def batch_request(built, mode, methods, choices=None):
    """A batch's request with the chosen baselines, from the campaign's own construction of the batch."""
    return dict(built, baselines=chosen(built["baselines"], mode, methods, ade_of(mode), choices))


def identity(request, binary_sha, candidate=None):
    """The cache key of a batch. The request, the binary and the thread settings, and the method and candidate of a
    candidate run, <method>-<candidate>."""
    key = dict(request=request, binary=binary_sha, threads=SINGLE_THREAD)
    if candidate is not None:
        key["candidate"] = candidate
    return hashlib.sha256(encoded(key)).hexdigest()[:16]


# ------------------------------------------------------------------------------------------- pair sets
def joined_pairs(problem):
    """The agent pairs that the unit's measured links join, in ascending order.

    Every agent is a node and all anchors together are one node. An observation joins its two ends whenever it exists,
    whatever its samples. Agent links join in the anchor free and anchor aided modes, anchor links in the anchor aided and
    anchor only modes. Two agents form a pair when a path of such links connects them. This is the rule of the joined
    pair set of scoring.py, and the tests compare both whenever that module provides it."""
    agents = sorted(problem["agent_ids"])
    index = {agent: i for i, agent in enumerate(agents)}
    anchors = {anchor["id"] for anchor in problem.get("anchors") or []}
    parent = list(range(len(agents) + 1))  # the last node stands for every anchor

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for observation in problem["observations"]:
        a, b = observation["from"], observation["to"]
        if a not in index:
            continue
        if observation["to_is_anchor"]:
            if problem["mode"] == "AnchorFree" or b not in anchors:
                continue
            other = len(agents)
        else:
            if problem["mode"] == "AnchorOnly" or b not in index or a == b:
                continue
            other = index[b]
        left, right = root(index[a]), root(other)
        if left != right:
            parent[max(left, right)] = min(left, right)
    groups = defaultdict(list)
    for agent in agents:
        groups[root(index[agent])].append(agent)
    return sorted(pair for group in groups.values() for pair in combinations(group, 2))


def pair_set(problem, name):
    """The agent pairs of one pair set of a unit, in ascending order, a declared one or another variant of scoring.py."""
    if name == "all":
        return list(combinations(sorted(problem["agent_ids"]), 2))
    if name == "joined":
        return joined_pairs(problem)
    if name in scoring.VARIANTS[scoring.mode_of(problem)]:
        return scoring.variant_pairs(problem, name)
    raise ValueError(f"no pair set {name}")


def pair_errors(request, reference, records, scorer):
    """The capped error of every agent pair of each record's unit, in the pair order of pilot.score_records.

    Every error is the released scorer's own, its score_pairs over the one pair (scoring.score_pairs), so a pair with an
    agent that the estimate lacks carries the penalty, and so does a pair with an agent whose coordinate the rule of
    scoring.scored_point makes missing. The mean over a unit's pairs in this order is its scored mean pair error bit for
    bit."""
    units = {(unit["id"], unit["samples"]): unit for unit in request["units"]}
    metadata = {entry["id"]: entry for entry in reference["units"]}
    result = []
    for record in records:
        agents = units[record["id"], record["samples"]]["problem"]["agent_ids"]
        entry = metadata[record["id"]]
        truth = {int(n): tuple(p) for n, p in reference["coordinates_cm"][entry.get("reference", entry["acquisition"])].items()}
        positions = record["estimate"]["positions"] if record["estimate"] is not None else [None] * len(agents)
        valid = {n: tuple(p) for n, p in zip(agents, positions) if scoring.scored_point(p) is not None}
        penalty = scorer.scenario_diameter(truth)
        result.append([scoring.score_pairs(scorer, truth, valid, [pair], penalty, 2)[2] for pair in combinations(agents, 2)])
    return result


def pair_set_means(problem, errors, names=None):
    """The unit's pair count and mean pair error under the named pair sets, from its per pair errors in the pair order
    of pilot.score_records. By default every pair set a SOLID row keeps (scoring.pair_sets), the declared ones and the
    other variants of the unit's mode, the supplementary ranged pairs among them. The mean is None when the set holds no
    pair of the unit."""
    pairs = [tuple(sorted(pair)) for pair in combinations(problem["agent_ids"], 2)]
    if len(pairs) != len(errors):
        raise ValueError("the per pair errors do not match the unit's agents")
    result = {}
    for name in names or ("all", *scoring.VARIANTS[scoring.mode_of(problem)]):
        members = set(pair_set(problem, name))
        kept = [error for pair, error in zip(pairs, errors) if pair in members]
        result[name] = dict(eligible_pairs=len(kept), mean_pair_error_cm=sum(kept) / len(kept) if kept else None)
    return result


# ------------------------------------------------------------------------------------------- runs
def solve_batch(batch, methods, folder, binary_sha, choices=None, candidate=None):
    """Solve one batch with the chosen baselines, cached by the request and binary identity, and store the per pair
    errors of its units beside the predictions. candidate names the candidate of a candidate run of the one method.
    Returns the scored rows."""
    if "scorer" not in STATE:
        STATE["scorer"] = pilot.evidence.load_scorer()
    built, reference = batch_inputs(batch)
    request = batch_request(built, batch["mode"], methods, choices)
    if not request["baselines"]:
        return []
    name = f"{batch['key']}-{identity(request, binary_sha, None if candidate is None else f'{methods[0]}-{candidate}')}.jsonl.gz"
    target = folder / "predictions" / name
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix=batch["key"] + "-", dir=folder) as temporary:
            lines = solved(request, Path(temporary) / "batch")
            staged = Path(temporary) / "predictions.jsonl.gz"
            staged.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))
            staged.replace(target)
    records = read_lines(target)
    labels = [b["method"] for b in request["baselines"]]
    if len(records) != len(request["units"]) * len(labels):
        raise ValueError(f"incomplete predictions for {batch['key']}")
    scores = pilot.score_records(request, reference, records, STATE["scorer"], labels)
    errors = pair_errors(request, reference, records, STATE["scorer"])
    # a pure function of the cached predictions, the unit inputs and the pinned scorer, so an existing file must agree
    stored = folder / "pair-errors" / name
    lines = [dict(id=r["id"], samples=r["samples"], method=r["method"], agent_ids=r["agent_ids"], errors=e) for r, e in zip(records, errors)]
    if not stored.exists():
        write_lines(stored, lines)
    elif read_lines(stored) != lines:
        raise ValueError(f"the stored per pair errors {stored.name} differ from those of its predictions")
    problems = {(unit["id"], unit["samples"]): unit["problem"] for unit in request["units"]}
    extra = {} if candidate is None else dict(candidate=candidate)
    rows = []
    for s, e in zip(scores, errors):
        means = pair_set_means(problems[s["id"], s["samples"]], e)
        if e and means["all"]["mean_pair_error_cm"] != s["mean_pair_error_cm"]:
            raise ValueError(f"the per pair errors of {batch['key']} unit {s['id']} do not give its scored error")
        rows.append(dict(key=batch["key"], id=s["id"], samples=s["samples"], method=s["method"], cell=s["acquisition"], acquisition=s["acquisition"],
                         anchors=batch["anchors"], configuration=batch["configuration"], panel=batch["panel"], error_cm=s["mean_pair_error_cm"],
                         elapsed_ms=s["elapsed_ms"], failure=s["failure"], starts=s["starts"], stayed_at_start=s["stayed_at_start"],
                         failed_agents=s["failed_agents"], expected_agents=s["expected_agents"], pair_sets=means, **extra))
    return rows


def kept(folder, methods, build, fresh=False):
    """The rows and run entries of the other methods of a half that a run of `methods` keeps, none when fresh.

    Every method of a half runs with one build of the adapter. The rows of another method stay only when its run entry
    names `build`, the hash and source identity of this run's adapter, and rows of another build or without an entry
    are refused, so that a rebuild sends every method of the half back to run with the new build, the first of them
    with fresh."""
    previous_rows, runs = [], {}
    if not fresh and (folder / "scores.json").exists():
        previous_rows = [r for r in json.loads((folder / "scores.json").read_text()) if r["method"] not in methods]
    if not fresh and (folder / "run.json").exists():
        record = json.loads((folder / "run.json").read_text())
        # a record of the earlier single-run layout describes every method it ran
        runs = record.get("runs") or {m: {k: v for k, v in record.items() if k != "methods"} for m in record.get("methods", [])}
    entries = {m: e for m, e in runs.items() if m not in methods and any(r["method"] == m for r in previous_rows)}
    other = sorted({r["method"] for r in previous_rows} - {m for m, e in entries.items() if {key: e.get(key) for key in build} == build})
    if other:
        raise ValueError(f"{folder} holds rows of {', '.join(other)} from another build of the adapter than this run's "
                         f"({build['binary_identity'][:12]}) or without a run entry. Every method of a half runs with one build, so run "
                         "every method of the half again with this build, the first run with --fresh")
    return previous_rows, entries


def merged(folder, methods, rows, entry, fresh=False):
    """scores.json and run.json with this run's methods replaced and every other method's rows and entry kept, when they
    come from the build of this run (kept), or none of them when fresh. The record counts the solves of every method
    that crashed the adapter, the starts its solves report and those that stayed at their start."""
    build = {key: entry[key] for key in ("binary_sha256", "binary_identity")}
    previous_rows, runs = kept(folder, methods, build, fresh)
    runs = dict(runs, **{m: dict(entry, aborted_solves=entry["aborted_solves"].get(m, []), starts=entry["starts"].get(m, {}),
                                 stayed_at_start=(entry.get("stayed_at_start") or {}).get(m)) for m in methods})
    return previous_rows + rows, dict(split=entry["split"], mode=entry["mode"], methods=sorted(runs), workers=entry["workers"], **build,
                                      abort_rule=isolation.ABORT_RULE,
                                      abort_counts={m: len(e.get("aborted_solves") or []) for m, e in sorted(runs.items())},
                                      starts={m: e.get("starts") for m, e in sorted(runs.items())}, starts_rule=isolation.STARTS,
                                      stayed_at_start={m: e.get("stayed_at_start") for m, e in sorted(runs.items())},
                                      stayed_rule=isolation.STAYED, runs=runs)


def load(split, ade=False):
    """The campaign of one half, loaded once in the parent for the forked workers, with ADE-BP's ranging model, in
    reader mode its law per mode from the shipped settings. Returns its folder."""
    D = DATA / CAMPAIGNS[split]
    base = json.loads(gzip.decompress((D / "base.json.gz").read_bytes()))
    base["values"] = {(v[0], v[1]): v[2] for v in base["values"]}
    model = None
    if ade:
        model = ({mode: baseline_selection.law("idlab", mode) for mode in ("AAC", "ANC")} if baseline_selection.READER
                 else ade_training.ranging_model("idlab", ADE_FIT))
    STATE.update(split=split, base=base, catalog=json.loads(gzip.decompress((D / "catalog.json.gz").read_bytes())), ade=model)
    return D


def in_parallel(function, tasks, workers, label):
    """function(*task) for every task in forked single-threaded processes. The results come in the order of the tasks,
    so no output depends on the order in which the processes finish."""
    started, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [pool.submit(function, *task) for task in tasks]
        try:
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 2000 == 0 or done == len(tasks):
                    print(f"{label}: {done}/{len(tasks)} batches after {time.time() - started:.0f}s", flush=True)
        except BaseException:
            # without this the pool would finish every queued batch before the error surfaces
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return [future.result() for future in futures]


def summarize(rows, methods, mode, split):
    """Balanced S50, S75 and S90 per method, sample track and reported group under the pairs the measured links join,
    the headline's pair set, and under the supplementary ranged pairs, each with the share of pairs and units it scores
    (layout_baselines.summarize). The anchor only mode reports the headline over three or more anchors and every
    smaller anchor count alone."""
    layout_baselines.summarize([r for r in rows if r["method"] in methods], f"IDLab {mode} {split}", mode)


def run_mode(split, mode, methods, workers, stride=1, choices=None, candidate=None, fresh=False, **details):
    """One mode of the loaded half over the batches the stride keeps, solved, scored and written to scores.json and
    run.json. choices holds the candidate of every method with selected options. A candidate run solves one method alone
    under a named candidate, writes below <method>-<candidate>/ and keys its cache by the method and candidate. The rows
    of the other methods of the half stay, when they come from the same build of the adapter, and fresh starts the half
    without them (kept). details go into the run entry. Returns the rows of this run."""
    if not BINARY.exists():
        raise ValueError(f"build the adapter first: {BINARY}")
    D = DATA / CAMPAIGNS[split]
    batches = [b for b in STATE["catalog"] if b["mode"] == mode and in_stride(b["key"], stride)]
    folder = OUT / split / mode.lower()
    if candidate is not None:
        folder = folder / f"{methods[0]}-{candidate}"
    build = isolation.build(BINARY)
    # a half holding rows of another build stops the run before any solve
    kept(folder, methods, build, fresh)
    for part in ("predictions", "pair-errors"):
        (folder / part).mkdir(parents=True, exist_ok=True)
    binary_sha, started = build["binary_sha256"], time.time()
    label = f"{split} {mode}" + ("" if candidate is None else f" {methods[0]}-{candidate}")
    tasks = [(batch, methods, folder, binary_sha, choices, candidate) for batch in batches]
    rows = [row for part in in_parallel(solve_batch, tasks, workers, label) for row in part]
    if digest(BINARY) != binary_sha:
        raise ValueError("the adapter changed during the run")
    failures = {}
    for r in rows:
        if isolation.is_abort(r["failure"]):
            failures.setdefault(r["method"], []).append(dict(key=r["key"], id=r["id"], samples=r["samples"], failure=r["failure"]))
    settings = baselines(STATE["base"]["settings"], mode, methods, ade_of(mode), choices)
    entry = dict(split=split, mode=mode, batches=len(batches), stride=stride, units=len({(r["key"], r["id"], r["samples"]) for r in rows}),
                 workers=workers, **build, commit=isolation.commit(), campaign_freeze_sha256=digest(D / "freeze.json"), settings=settings,
                 base_configurations_sha256=digest(baseline_selection.BASE),
                 ade_fit_sha256=digest(ADE_FIT) if STATE["ade"] is not None and not baseline_selection.READER else None,
                 pair_sets=["all", *scoring.VARIANTS[mode]],
                 pair_set=PAIR_SET, seconds=time.time() - started,
                 finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), aborted_solves=failures, starts=isolation.start_counts(rows),
                 stayed_at_start=isolation.stayed_counts(rows), stayed_rule=isolation.STAYED, shared_inputs=layout_inputs.record("idlab"),
                 **({} if candidate is None else dict(candidate=candidate)), **details, **isolation.execution(workers))
    all_rows, record = merged(folder, methods, rows, entry, fresh)
    replace_text(folder / "scores.json", json.dumps(all_rows) + "\n")
    replace_text(folder / "run.json", json.dumps(record, indent=1) + "\n")
    summarize(rows, methods, mode, split)
    return rows


def selection(method):
    """The IDLab selection record of a method, refused unless it was made by the declared selection of that method on
    the current training campaign, from the declared base configurations and with the shared inputs of this run
    (layout_inputs.record), so that no record of a trial with another stride, grid, base record or rule of the starts
    decides a run."""
    path = selection_path(method)
    if not path.exists():
        raise ValueError(f"select the IDLab options of {method} with select --method {method} first, or name a candidate with --candidate ({path})")
    record = layout_baselines.selection_record(path)
    D = DATA / CAMPAIGNS["training"]
    if record.get("dataset") != "idlab":
        raise ValueError(f"{path} is not an IDLab selection")
    if record.get("method") != method:
        raise ValueError(f"{path} is not a selection of {method}")
    if record.get("stride") != SELECTION_STRIDE or record.get("candidates") != json.loads(json.dumps(baseline_selection.grid(method, SELECTION_MODES[method]))):
        raise ValueError(f"{path} was made with another stride or other candidates than the declared ones")
    if record.get("base_configurations_sha256") != digest(baseline_selection.BASE):
        raise ValueError(f"{path} was made with other declared base configurations than those of this run")
    if record.get("shared_inputs") != json.loads(json.dumps(layout_inputs.record("idlab"))):
        raise ValueError(f"{path} was made with other shared inputs than those of this run (layout_inputs.RULE)")
    if record.get("rule") != RULE:
        raise ValueError(f"{path} was made under another rule than the declared one")
    if record.get("anchor_only_headline_min_anchors") != HEADLINE_MIN_ANCHORS:
        raise ValueError(f"{path} ranked the anchor only mode over other cells than those of {HEADLINE_MIN_ANCHORS} or more anchors")
    if (record.get("campaign_freeze_sha256"), record.get("campaign_base_sha256")) != (digest(D / "freeze.json"), digest(D / "base.json.gz")):
        raise ValueError(f"the IDLab training campaign changed after the selection of {method}")
    return record


def headline_selection(method):
    """The selection record of a method that a run takes, refused unless selection() accepts it, every mode's choice was
    made under the headline's pair set and the adapter that runs now made it (isolation.same_build), so that the
    selected options and every baseline of IDLab come from one build. A choice moved to another pair set by select
    --pair-set is a diagnostic and runs nothing."""
    record = selection(method)
    path = selection_path(method)
    chosen = {record.get("pair_set"), *(entry.get("pair_set") for entry in record["selections"].values())}
    if chosen != {PAIR_SET}:
        other = ", ".join(sorted(str(name) for name in chosen - {PAIR_SET}))
        raise ValueError(f"{path} chose the options of {method} under the pair set {other}, not the {PAIR_SET} pairs of the headline. Move "
                         f"the choice back with select --method {method} --pair-set {PAIR_SET} before a run")
    return isolation.same_build(record, path, BINARY)


def selected(mode, methods):
    """The selected candidate of every method among methods that has options to select and runs in the mode, and what
    a run entry keeps of each, its choice and the hash of its record. In reader mode the final option of every method
    among methods that the shipped settings name for the mode (baseline_selection.options)."""
    if baseline_selection.READER:
        choices = {method: entry for method, entry in baseline_selection.options("idlab", mode).items() if method in methods}
        return choices, baseline_selection.kept(choices, mode)
    choices, records = {}, {}
    for method in methods:
        if method in SELECTED and baseline_selection.selects(method, mode):
            entry = headline_selection(method)["selections"][mode]
            choices[method] = entry
            records[method] = dict(selected=entry["selected"], pair_set=entry["pair_set"], solve_config=entry["solve_config"],
                                   method_config=entry.get("method_config") or {}, record_sha256=digest(selection_path(method)))
    return choices, records


def run_split(split, methods, modes, workers, stride=1, candidate=None, fresh=False):
    """The chosen methods on the chosen modes of one half. A method with options to select runs alone under a named
    candidate below <method>-<candidate>/, or else every method with selected options takes the selected candidate of
    its mode. fresh starts every mode of the half without the rows of the methods the run leaves out."""
    if candidate is not None:
        if len(methods) != 1 or methods[0] not in SELECTED:
            raise ValueError("a candidate run solves one method with options to select alone")
        if not any(candidate in (c["name"] for c in baseline_selection.candidates(methods[0], mode)) for mode in modes
                   if baseline_selection.selects(methods[0], mode)):
            raise ValueError(f"{methods[0]} has no candidate {candidate} in the modes {', '.join(modes)}")
    unknown = set(methods) - set(PERMITTED["AAC"])
    if unknown:
        raise ValueError(f"no IDLab baseline {sorted(unknown)}")
    load(split, ade="ade-bp-2015" in methods)
    for mode in modes:
        allowed = [m for m in methods if m in PERMITTED[mode]]
        if candidate is not None:
            # a candidate runs only in the modes that select the method's options and hold the candidate, the planar
            # model alone in the anchor free mode
            allowed = [m for m in allowed if baseline_selection.selects(m, mode)
                       and candidate in (c["name"] for c in baseline_selection.candidates(m, mode))]
        if not allowed:
            continue
        if candidate is not None:
            run_mode(split, mode, allowed, workers, stride, choices={allowed[0]: baseline_selection.candidate(allowed[0], candidate, mode)},
                     candidate=candidate, fresh=fresh)
        else:
            choices, records = selected(mode, allowed)
            run_mode(split, mode, allowed, workers, stride, choices=choices, fresh=fresh, **({"selections": records} if records else {}))


def run_selected(method, splits, workers):
    """A method with the selected candidate of each mode over every batch of each half, its rows beside the other
    methods'."""
    if baseline_selection.READER:
        modes = [mode for mode in PERMITTED if method in baseline_selection.options("idlab", mode)]
    else:
        modes = list(headline_selection(method)["selections"])
    for split in splits:
        run_split(split, [method], modes, workers)


# ------------------------------------------------------------------------------------------- selection
def stored_units(batch, method, candidates, binary_sha):
    """Every unit of one training batch under each candidate of the method with its error under every declared pair set,
    from the per pair errors that the candidate runs stored. Reads files only and never starts the adapter. Returns
    (candidate, samples, cell, {pair set: error or None}) tuples."""
    built, reference = batch_inputs(batch)
    problems = {(unit["id"], unit["samples"]): unit["problem"] for unit in built["units"]}
    cells = {entry["id"]: entry["acquisition"] for entry in reference["units"]}
    result = []
    for entry in candidates:
        label = f"{method}-{entry['name']}"
        request = batch_request(built, batch["mode"], [method], {method: entry})
        path = (OUT / STATE["split"] / batch["mode"].lower() / label / "pair-errors"
                / f"{batch['key']}-{identity(request, binary_sha, label)}.jsonl.gz")
        if not path.exists():
            raise ValueError(f"{batch['key']} has no stored per pair errors under {label}, which select writes")
        lines = read_lines(path)
        if sorted((line["id"], line["samples"]) for line in lines) != sorted(problems) or {line["method"] for line in lines} != {method}:
            raise ValueError(f"the stored per pair errors {path.name} do not cover the batch")
        for line in lines:
            problem = problems[line["id"], line["samples"]]
            if line["agent_ids"] != problem["agent_ids"]:
                raise ValueError(f"the stored per pair errors {path.name} name other agents")
            means = pair_set_means(problem, line["errors"], PAIR_SETS)
            result.append((entry["name"], line["samples"], cells[line["id"]], {name: means[name]["mean_pair_error_cm"] for name in PAIR_SETS}))
    return result


def choice(mode, method, candidates, stride, binary_sha, pair_set, workers, scorer):
    """The candidates of a method in one mode ranked under every declared pair set from the stored per pair errors of the
    loaded training half, and the selection under the given pair set.

    A unit without any pair of a set is left out of that set's balance, as the released scorer leaves out a scenario
    without a finite mean. The pair sets follow from the inputs alone, so every candidate keeps the same units. The
    entry counts the candidates within the declared tolerance of the lowest mean S75 under each pair set
    (baseline_selection.tied_at_top), so a choice that the declared order made among tied candidates shows in the record.

    The released balanced percentile adds up the cells' fractions in the order the cells first appear. When the mixture
    sits exactly at a quantile level, that order decides the last bit and with it the percentile. The rows therefore
    enter in the order of their cell names, which makes every ranking a function of the stored errors alone."""
    batches = [b for b in STATE["catalog"] if b["mode"] == mode and in_stride(b["key"], stride)
               and (mode != "ANC" or b["anchors"] >= HEADLINE_MIN_ANCHORS)]
    tasks = [(batch, method, candidates, binary_sha) for batch in batches]
    units = [unit for part in in_parallel(stored_units, tasks, workers, f"{STATE['split']} {mode} {method} stored per pair errors") for unit in part]
    units.sort(key=lambda unit: unit[2])
    rankings, population = {}, {}
    for name in PAIR_SETS:
        rows = {entry["name"]: [] for entry in candidates}
        for label, samples, cell, errors in units:
            if errors[name] is not None:
                rows[label].append(dict(method=method, samples=samples, cell=cell, error_cm=errors[name]))
        if len({len(kept) for kept in rows.values()}) != 1:
            raise ValueError(f"the candidates of {method} in {mode} keep different units under the pair set {name}")
        ranking = layout_baselines.choose(rows, scorer, candidates, method)
        if not all(math.isfinite(entry["mean_s75"]) for entry in ranking):
            raise ValueError(f"no finite S75 of {method} in {mode} under the pair set {name}")
        rankings[name] = ranking
        kept = rows[candidates[0]["name"]]
        population[name] = {str(s): dict(units=sum(r["samples"] == s for r in kept), cells=len({r["cell"] for r in kept if r["samples"] == s}))
                            for s in sorted({r["samples"] for r in kept})}
    tied = {name: baseline_selection.tied_at_top(ranking) for name, ranking in rankings.items()}
    return dict(layout_baselines.selection(rankings[pair_set]), pair_set=pair_set, rankings=rankings, tied_at_top=tied,
                batches=len(batches), units=len(units) // len(candidates), population=population)


def report(method, mode, entry):
    for name, ranking in entry["rankings"].items():
        print(f"IDLab {mode} {method} under {name} pairs: " + ", ".join(f"{r['candidate']} {r['mean_s75']:.1f}" for r in ranking), flush=True)
    tied = entry["tied_at_top"][entry["pair_set"]]
    print(f"IDLab {mode} {method}: {entry['selected']} selected under {entry['pair_set']} pairs"
          + (f", by the declared order among {tied} candidates within {baseline_selection.TOLERANCE_CM} cm of the lowest mean S75" if tied > 1 else ""),
          flush=True)


def select(method, workers, pair_set=PAIR_SET):
    """A method's options per mode it runs in on the declared subsample of the training half, ranked under every
    declared pair set from the stored per pair errors and chosen under the given one. The entry of every mode counts the
    solves of every candidate that crashed the adapter and those that stayed at their start. With a selection already
    recorded under another pair set only the choice moves, from the stored per pair errors and without starting the
    adapter."""
    if pair_set not in PAIR_SETS:
        raise ValueError(f"no pair set {pair_set}")
    if method not in SELECTED:
        raise ValueError(f"{method} has no options to select")
    path = selection_path(method)
    if path.exists():
        return reselect(method, selection(method), pair_set, workers)
    # ADE-BP's candidates run with the ranging model of the training half
    D = load("training", ade=method == "ade-bp-2015")
    binary_sha, scorer = digest(BINARY), pilot.evidence.load_scorer()
    selections = {}
    for mode in SELECTION_MODES[method]:
        candidates = baseline_selection.candidates(method, mode)
        aborted, stayed = {}, {}
        for entry in candidates:
            rows = run_mode("training", mode, [method], workers, SELECTION_STRIDE, choices={method: entry}, candidate=entry["name"])
            aborted[entry["name"]] = isolation.abort_counts(rows, (method,))[method]
            stayed[entry["name"]] = isolation.stayed_counts(rows).get(method)
        selections[mode] = dict(choice(mode, method, candidates, SELECTION_STRIDE, binary_sha, pair_set, workers, scorer), abort_counts=aborted,
                                stayed_at_start=stayed)
        report(method, mode, selections[mode])
        isolation.report_aborts(aborted, f"IDLab {mode} {method} candidates")
    if digest(BINARY) != binary_sha:
        raise ValueError("the adapter changed during the selection")
    header = layout_baselines.selection_header("idlab", "mode", workers, method, binary=BINARY, rule=RULE, modes=SELECTION_MODES[method],
                                               campaign=CAMPAIGNS["training"], shared_inputs=layout_inputs.record("idlab"),
                                               campaign_freeze_sha256=digest(D / "freeze.json"), campaign_base_sha256=digest(D / "base.json.gz"),
                                               stride=SELECTION_STRIDE, pair_sets=PAIR_SETS, declared_pair_set=PAIR_SET, pair_set=pair_set,
                                               anchor_only_headline_min_anchors=HEADLINE_MIN_ANCHORS, abort_rule=isolation.ABORT_RULE,
                                               scorer_sha256=pilot.evidence.official_scorer.SCORER_SHA256)
    write(path, dict(header, selections=selections))


def reselect(method, record, pair_set, workers):
    """The recorded selection of a method with its choice moved to another declared pair set. Every ranking is
    recomputed from the stored per pair errors without starting the adapter and must equal the recorded one. record is
    checked by selection(). A choice under another pair set than the headline's is a diagnostic, and no run takes it
    (headline_selection)."""
    if record["pair_set"] == pair_set:
        raise ValueError(f"the IDLab options of {method} are already selected under the pair set {pair_set}")
    load("training", ade=method == "ade-bp-2015")
    scorer = pilot.evidence.load_scorer()
    selections = {}
    for mode, entry in record["selections"].items():
        again = choice(mode, method, record["candidates"][mode], record["stride"], record["binary_sha256"], pair_set, workers, scorer)
        if json.loads(json.dumps(again["rankings"])) != entry["rankings"]:
            raise ValueError(f"the stored per pair errors of {method} in {mode} no longer give the recorded rankings")
        selections[mode] = dict(again, abort_counts=entry.get("abort_counts"), stayed_at_start=entry.get("stayed_at_start"))
        report(method, mode, again)
    write(selection_path(method), dict(record, pair_set=pair_set, selections=selections, pair_set_changed=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=("run", "select", "run-selected", "check-selection"))
    parser.add_argument("--split", choices=sorted(CAMPAIGNS))
    parser.add_argument("--methods", help="comma separated subset of jin-2021, fnl, strong, ade-bp-2015")
    parser.add_argument("--method", choices=SELECTED, help="the baseline whose options select, run-selected or check-selection concerns")
    parser.add_argument("--modes", default="af,aac,anc")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--stride", type=int, default=1, help="keep batches whose key hash is zero modulo the stride")
    parser.add_argument("--candidate", help="the one method of --methods alone under this candidate of its options, below <method>-<candidate>/")
    parser.add_argument("--pair-set", choices=list(PAIR_SETS), default=PAIR_SET, help="the pair set that decides select")
    parser.add_argument("--splits", default="training,evaluation", help="the halves of run-selected in order")
    parser.add_argument("--fresh", action="store_true", help="start every mode of the half without the rows of the methods the run leaves out")
    args = parser.parse_args()
    started = time.monotonic()
    if args.command in ("select", "run-selected") and not args.method:
        parser.error(f"{args.command} needs --method")
    if args.command == "select":
        select(args.method, args.workers, args.pair_set)
    elif args.command == "run-selected":
        run_selected(args.method, args.splits.split(","), args.workers)
    elif args.command == "check-selection":
        for method in [args.method] if args.method else SELECTED:
            record = headline_selection(method)
            print(f"IDLab {method} selection verified, " + ", ".join(f"{mode} {entry['selected']}" for mode, entry in record["selections"].items()),
                  flush=True)
    else:
        if not (args.split and args.methods):
            parser.error("a run needs --split and --methods")
        run_split(args.split, args.methods.split(","), [m.upper() for m in args.modes.split(",")], args.workers, args.stride, args.candidate,
                  args.fresh)
    print(f"done in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
