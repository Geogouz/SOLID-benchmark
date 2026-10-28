"""Summarize the controlled timing of every method and mode on the TWR-CLOUD-Bern evaluation corpus.

The controlled runs (timing_bern.sh) solved the whole corpus once more with every method and mode, one after another,
90 single-threaded processes pinned to 90 physical cores (taskset -c 0-89), with the executables of the method freeze.
Their accuracy is not read. Every run keeps the evidence that the machine was idle. A run of timing_bern.sh has
conditions.json beside it, with the load and the busiest processes before and after. The runs of the session of
30 September, which waited for every other stage and started on an idle machine, are in the session's log,
records/timing-bern-launcher-2026-09-30.log, which --session names when it lies elsewhere and which a reader's timing
does not need, since every run of timing_bern.sh has its conditions.json. For every method and mode the summary keeps the
solves, the mean, median, 95th and 99th percentile and largest time of one solve, and the wall time of the run. It also
compares the run's predictions with those of the evaluation, which ran the same executables under other load, member
for member, so it records whether the rerun reproduced every estimate.

Usage: summarize_timing.py --timing TIMING --evaluation OUT --out FILE [--session LOG]
       TIMING is the bern folder of timing_bern.sh's runs (out/timing/bern unless SOLID_TIMING names another folder), OUT
       the folder of the evaluation's Bern runs (out/ of the checkout), FILE the summary, LOG the session log, read when it
       exists
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

SESSION = Path(__file__).resolve().parent / "records/timing-bern-launcher-2026-09-30.log"

EVALUATION = {"solid": "solid/bern", "jin-2021": "bern-baselines-v3/jin-2021", "fnl": "bern-baselines-v3/fnl", "strong": "bern-baselines-v3/strong",
              "ade-bp-2015": "bern-baselines-v3/ade-bp-2015"}


def contents(path):
    """The SHA-256 of every member of a prediction archive."""
    with zipfile.ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(archive.namelist())}


def session_runs(path=SESSION):
    """The runs of the session of 30 September, each with its start and end from the session's log, and the load the
    session started on, none without the log."""
    if not Path(path).is_file():
        return {}
    lines = Path(path).read_text().splitlines()
    idle = next(line for line in lines if "the node is idle" in line)
    load = [float(x) for x in idle.rsplit("load", 1)[1].split()]
    runs = {}
    for line in lines:
        match = re.match(r"(\S+ \S+) (\S+) (\S+) (starts|exit \d+)$", line)
        if match:
            stamp, method, mode, event = match.groups()
            runs.setdefault((method, mode), {})["started" if event == "starts" else "ended"] = stamp
    # a run whose end the log lacks, ADE-BP's AAC run after the launcher stopped, ends after its own wall time, and the
    # follow-up jobs waited for it (logs/after-timing.log on the node, "the timing ended" at 13:08)
    return {key: dict(source="the session log of 30 September", session_started_idle=True, session_start_load=load, **times)
            for key, times in runs.items()}


def conditions(folder, method, mode, session, wall_s):
    """The evidence that the node was idle during one run: its conditions.json, or else its place in the session log."""
    path = folder / "conditions.json"
    if path.exists():
        record = json.loads(path.read_text())
        idle = record["before"]["load"][0] < 2 and all(float(p["pcpu"]) <= 50 for p in record["before"]["busiest"])
        return dict(source="conditions.json", started=record["started"], ended=record["ended"], idle_at_start=idle,
                    load_before=record["before"]["load"], load_after=record["after"]["load"])
    entry = session.get((method, mode))
    if entry is None:
        return dict(source=None, idle_at_start=False)
    if "ended" not in entry:
        entry = dict(entry, ended_after_wall_s=round(wall_s))
    return dict(entry, idle_at_start=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session", type=Path, default=SESSION)
    args = parser.parse_args()
    rows = []
    session = session_runs(args.session)
    for path in sorted(args.timing.glob("*/*/benchmark.json")):
        method, mode = path.parent.parent.name, path.parent.name
        record = json.loads(path.read_text())
        command = json.loads((path.parent / "command.json").read_text())
        evaluated = args.evaluation / EVALUATION[method] / mode / "predictions.zip"
        same = evaluated.exists() and contents(path.parent / "predictions.zip") == contents(evaluated)
        rows.append(dict(method=method, mode=mode.upper(), solves=record["solves"], scenarios=record["applicable_scenario_count"],
                         mean_ms=record["solver_mean_micros"] / 1e3, median_ms=record["solver_p50_micros"] / 1e3,
                         p95_ms=record["solver_p95_micros"] / 1e3, p99_ms=record["solver_p99_micros"] / 1e3, max_ms=record["solver_max_micros"] / 1e3,
                         wall_s=record["wall_seconds"], workers=record["workers"], threads_per_process=record["threads_per_process"],
                         binary_sha256=record["binary_sha256"], implementation_revision=record["implementation_revision"], commit=command.get("commit"),
                         termination_counts=record["termination_counts"], failures=record["failure_counts"],
                         aborted=len(record["aborted_solves"]), reproduces_evaluation=same,
                         conditions=conditions(path.parent, method, mode, session, record["wall_seconds"])))
    summary = dict(schema="solid.timing.bern.v2", corpus="TWR-CLOUD-Bern evaluation scenarios, every applicable scenario, iteration and track",
                   placement="90 single-threaded processes pinned to CPUs 0 to 89, 90 of the node's 96 physical cores, their SMT siblings idle",
                   runs=rows, every_run_reproduces_evaluation=all(r["reproduces_evaluation"] for r in rows),
                   every_run_started_on_an_idle_node=all(r["conditions"]["idle_at_start"] for r in rows))
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    for r in rows:
        print(f"{r['method']:12s} {r['mode']:4s} solves {r['solves']:7d}  median {r['median_ms']:9.2f} ms  p95 {r['p95_ms']:9.2f} ms  "
              f"mean {r['mean_ms']:9.2f} ms  wall {r['wall_s'] / 60:6.1f} min  same predictions as the evaluation: {r['reproduces_evaluation']}")


if __name__ == "__main__":
    main()
