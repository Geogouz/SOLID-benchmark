"""The Bern benchmark CLI over a scenario corpus in single-threaded processes, with crashed solves isolated.

`solid benchmark` solves every applicable unit of a scenario corpus with one method. This runner splits the corpus into
shards of consecutive scenario identifiers and runs one CLI process per shard, with one worker and Rayon, OpenBLAS and
OpenMP limited to one thread (isolation.SINGLE_THREAD), at most `workers` processes at a time. It merges the shards
into the two outputs of the CLI, benchmark.json and predictions.zip with the released scorer's predictions.csv, so that
the released scorer and every reader of those files see one run.

A shard whose process crashes is solved again scenario by scenario, and a crashing scenario solve by solve, one
iteration and one sample count at a time. A single solve that still crashes is that method's failure on that unit, by
the rule of every method (isolation.ABORT_RULE), and it prints a warning. It is written as missing coordinates for
every agent of the unit, which the released scorer charges the diameter cap, listed under aborted_solves with the time
until the crash and counted under abort_counts. Any other failure of the CLI stops the run. Every piece is cached below
the work folder by its command, its identifiers, the executable, the corpus and the thread settings, so a run resumes
where it stopped. The merged report names the machine it ran on, the workers that solved at once and the timing rule
of its solve times (isolation.execution), and the starts its solves report (isolation.STARTS).
"""
import hashlib
import io
import json
import math
import subprocess
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "datasets"))
import isolation  # noqa: E402

HEADER = "scenario_id,iteration,samples,mode,node_id,q,x,y,z"
MODES = dict(af="AF", aac="AAC", anc="ANC")
MASK = (1 << 64) - 1
SCHEMA = "solid.benchmark.sharded.v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=indent, allow_nan=False) + "\n")
    temporary.replace(path)


# ------------------------------------------------------------------------------------------------ corpus
def corpus(path):
    """The scenario identifiers of a corpus with their members, as the CLI reads them: every member of an archive
    and every JSON file of a folder whose file stem is a number."""
    path = Path(path)
    if path.is_dir():
        names = [p.name for p in path.iterdir() if p.suffix == ".json"]
    else:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
    members = {}
    for name in names:
        stem = Path(name).stem
        if stem.isdigit():
            if int(stem) in members:
                raise ValueError(f"scenario {int(stem)} appears twice in {path}")
            members[int(stem)] = name
    if not members:
        raise ValueError(f"no scenarios in {path}")
    return dict(sorted(members.items()))


def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def select(ids, limit, seed):
    """The CLI's sampled selection of `limit` scenarios under a selection seed, all of them without a limit."""
    ids = sorted(ids)
    if not limit or limit >= len(ids):
        return ids
    return sorted(sorted(ids, key=lambda i: splitmix64((seed ^ i) & MASK))[:limit])


def shards(ids, workers, size=None):
    """Consecutive identifiers in shards, about four per worker and at most 256 scenarios each unless given."""
    size = size or max(1, min(256, math.ceil(len(ids) / (4 * max(1, workers)))))
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def agents_of(scenarios, members, sid):
    """The agents and the anchor count of one scenario, the units the CLI would write for it."""
    path = Path(scenarios)
    if path.is_dir():
        data = json.loads((path / members[sid]).read_text())
    else:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read(members[sid]))
    return sorted(int(a) for a in data["Agents"]), len(data["Anchors"])


def percentile(values, q):
    """The CLI's percentile of sorted integers, the value at the rounded index."""
    return values[round((len(values) - 1) * q)] if values else 0


# ------------------------------------------------------------------------------------------------ pieces
def parts(piece):
    """A crashed piece's parts: its scenarios one by one, then one scenario's solves one iteration and sample count
    at a time, none for a single solve."""
    if len(piece["ids"]) > 1:
        return [dict(piece, ids=[sid]) for sid in piece["ids"]]
    tracks = [(i, s) for i in piece["iterations"] for s in piece["samples"]]
    if len(tracks) > 1:
        return [dict(piece, iterations=[i], samples=[s]) for i, s in tracks]
    return []


class Run:
    """One method and mode over one corpus."""

    def __init__(self, binary, scenarios, arguments, mode, work, iterations=(1, 2), samples=(1, 10)):
        self.binary, self.scenarios, self.arguments = Path(binary), Path(scenarios), list(arguments)
        self.mode = MODES[mode.lower()]
        self.work = Path(work)
        self.iterations, self.samples = sorted(iterations), sorted(samples)
        if not self.binary.exists():
            raise ValueError(f"build the CLI first: {self.binary}")
        self.members = corpus(self.scenarios)
        self.identity = dict(binary=sha256(self.binary), scenarios=sha256(self.scenarios) if self.scenarios.is_file() else
                             hashlib.sha256(json.dumps(self.members).encode()).hexdigest(), arguments=self.arguments, mode=self.mode,
                             threads=isolation.SINGLE_THREAD)

    def argv(self, piece, folder):
        return [str(self.binary), "benchmark", "--scenarios", str(self.scenarios), "--mode", self.mode.lower(), *self.arguments,
                "--iterations", ",".join(map(str, piece["iterations"])), "--sample-counts", ",".join(map(str, piece["samples"])),
                "--scenario-ids", str(folder / "ids.txt"), "--workers", "1", "--include-cases",
                "--output", str(folder / "benchmark.json"), "--predictions-output", str(folder / "predictions.zip")]

    def run(self, piece, folder):
        """One CLI process for a piece, or its cached outcome: (returncode, outputs, stderr, elapsed_ns)."""
        identity = dict(self.identity, ids=piece["ids"], iterations=piece["iterations"], samples=piece["samples"])
        state = folder / "piece.json"
        if state.exists():
            saved = read(state)
            if saved["identity"] == identity and (saved["returncode"] != 0 or (folder / "predictions.zip").exists()):
                outputs = [dict(kind="piece", folder=str(folder))] if saved["returncode"] == 0 else None
                return saved["returncode"], outputs, saved["stderr"], saved["elapsed_ns"]
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("benchmark.json", "predictions.zip", "piece.json"):
            (folder / name).unlink(missing_ok=True)
        (folder / "ids.txt").write_text("".join(f"{sid}\n" for sid in piece["ids"]))
        started = time.monotonic_ns()
        process = subprocess.run(self.argv(piece, folder), text=True, capture_output=True, env=isolation.environment())
        elapsed = time.monotonic_ns() - started
        if process.returncode == 0 or isolation.crashed(process.returncode, process.stderr):
            write(state, dict(identity=identity, returncode=process.returncode, elapsed_ns=elapsed,
                              stderr=process.stderr[-4000:] if process.returncode else ""))
        outputs = [dict(kind="piece", folder=str(folder))] if process.returncode == 0 else None
        return process.returncode, outputs, process.stderr, elapsed

    def failed(self, piece, returncode, stderr, elapsed):
        """The outcome of a single solve that crashed the CLI, missing coordinates for every agent of its unit, with its
        warning."""
        sid, iteration, samples = piece["ids"][0], piece["iterations"][0], piece["samples"][0]
        agents, anchors = agents_of(self.scenarios, self.members, sid)
        if (self.mode == "AF") != (anchors == 0):
            raise isolation.AdapterError(f"scenario {sid} has no {self.mode} unit but crashed the CLI")
        failure = isolation.failure_message(returncode, stderr)
        method = self.arguments[self.arguments.index("--method") + 1] if "--method" in self.arguments else "the method"
        isolation.warn_crash(f"{method} on scenario {sid} iteration {iteration} with {samples} samples", failure)
        return [dict(kind="failure", scenario=sid, iteration=iteration, samples=samples, agents=agents, anchors=anchors,
                     elapsed_micros=elapsed // 1000, failure=failure)]

    def piece(self, ids, index):
        return dict(ids=ids, iterations=self.iterations, samples=self.samples), self.work / f"shard-{index:05d}-{ids[0]}-{ids[-1]}"


def execute(run, ids, workers, shard_size=None):
    """Every shard of the identifiers, at most `workers` processes at a time, the outcomes in identifier order."""
    pieces = [run.piece(chunk, index) for index, chunk in enumerate(shards(ids, workers, shard_size))]
    results = [None] * len(pieces)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(isolation.isolated, piece, run.run, parts, run.failed, folder): index for index, (piece, folder) in enumerate(pieces)}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                if done % 100 == 0 or done == len(pieces):
                    print(f"{done}/{len(pieces)} shards", flush=True)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return [outcome for result in results for outcome in result], len(pieces)


# ------------------------------------------------------------------------------------------------ merge
def merge(run, outcomes, output, include_cases=False):
    """predictions.zip in the released scorer's layout and the merged benchmark report, from the outcomes in order."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports, cases, aborted, applicable = [], [], [], set()
    records = missing = nonfinite = 0
    previous = None
    staged = output / ".predictions.zip.partial"
    with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive, archive.open("predictions.csv", "w") as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        stream.write(HEADER + "\n")

        def emit(line):
            nonlocal previous, records, missing, nonfinite
            fields = line.split(",")
            if len(fields) != 9 or fields[3] != run.mode or fields[5] != "2":
                raise ValueError(f"unexpected prediction row {line}")
            key = tuple(int(fields[i]) for i in (0, 1, 2, 4))
            if previous is not None and key <= previous:
                raise ValueError("merged prediction keys are not strictly ordered")
            previous = key
            applicable.add(key[0])
            absent = not fields[6].strip() or not fields[7].strip()
            missing += int(absent)
            nonfinite += int(not absent and not (math.isfinite(float(fields[6])) and math.isfinite(float(fields[7]))))
            records += 1
            stream.write(line + "\n")

        for outcome in outcomes:
            if outcome["kind"] == "failure":
                aborted.append({k: v for k, v in outcome.items() if k != "kind"})
                for node in outcome["agents"]:
                    emit(f"{outcome['scenario']},{outcome['iteration']},{outcome['samples']},{run.mode},{node},2,,,")
                continue
            folder = Path(outcome["folder"])
            report = read(folder / "benchmark.json")
            reports.append((folder, report))
            cases.extend(report.get("cases") or [])
            with zipfile.ZipFile(folder / "predictions.zip") as source, source.open("predictions.csv") as member:
                lines = io.TextIOWrapper(member, encoding="utf-8")
                if next(lines).rstrip("\n") != HEADER:
                    raise ValueError(f"unexpected prediction header in {folder}")
                for line in lines:
                    emit(line.rstrip("\n"))
        stream.flush()
        stream.detach()
    staged.replace(output / "predictions.zip")
    if not reports:
        raise ValueError("no shard solved any unit")
    first = reports[0][1]
    shared = ("implementation_revision", "method", "method_config", "deployment_profile", "cooperative_profiles", "baseline_config",
              "applicable_modes", "not_applicable_modes")
    for folder, report in reports:
        if any(report.get(key) != first.get(key) for key in shared):
            raise ValueError(f"shard {folder} ran another executable or configuration")
    elapsed = sorted([int(c["elapsed_micros"]) for c in cases] + [a["elapsed_micros"] for a in aborted])
    terminations, failures = Counter(), Counter()
    for _, report in reports:
        terminations.update(report.get("termination_counts") or {})
        failures.update(report.get("failure_counts") or {})
    if aborted:
        terminations["Aborted"] += len(aborted)
        failures.update(a["failure"] for a in aborted)
    method = first.get("method")
    # SOLID's anchor only solve starts no search, and the count its cases carry is that of its joint layout samples
    starts = None if (method, run.mode) == ("solid", "ANC") else {method: dict(sorted(Counter(
        str(c["starts"]) for c in cases if c.get("starts") is not None and not c.get("method_error")).items(), key=lambda item: int(item[0])))}
    # the solves of a baseline whose estimate stayed at its start, over those that report a start (isolation.STAYED)
    stays = isolation.stayed_counts([dict(c, method=method) for c in cases if not c.get("method_error")])
    return dict(
        {key: first.get(key) for key in shared}, schema_version=SCHEMA, scenarios=str(run.scenarios), mode_filter=run.mode,
        sample_counts=run.samples, iterations=run.iterations, applicable_scenario_count=len(applicable), solves=len(elapsed),
        solver_mean_micros=sum(elapsed) / max(1, len(elapsed)), solver_p50_micros=percentile(elapsed, 0.50),
        solver_p95_micros=percentile(elapsed, 0.95), solver_p99_micros=percentile(elapsed, 0.99), solver_max_micros=elapsed[-1] if elapsed else 0,
        termination_counts=dict(sorted(terminations.items())), failure_counts=dict(sorted(failures.items())), aborted_solves=aborted,
        abort_rule=isolation.ABORT_RULE, abort_counts={method: len(aborted)}, starts=starts, starts_rule=isolation.STARTS,
        stayed_at_start=stays or None, stayed_rule=isolation.STAYED,
        binary_sha256=run.identity["binary"],
        prediction_archive=dict(path="predictions.zip", member="predictions.csv", format="zip-compressed official prediction CSV",
                                records=records, missing_coordinates=missing, nonfinite_coordinates=nonfinite),
        scenario_fingerprint=hashlib.sha256("".join(f"{sid}\n" for sid in sorted(applicable)).encode()).hexdigest(),
        pieces={str(folder.relative_to(run.work)): sha256(folder / "benchmark.json") for folder, _ in reports},
        **({"cases": cases} if include_cases else {}))


def benchmark(binary, scenarios, arguments, mode, output, workers, work=None, limit=0, seed=1, include_cases=False, shard_size=None,
              iterations=(1, 2), samples=(1, 10)):
    """Solves a corpus with the CLI in single-threaded processes and writes output/benchmark.json and
    output/predictions.zip. arguments are the CLI's method flags (method, dimensions, aggregation, configuration,
    profiles, solver budget), the runner sets the corpus, the mode, the tracks, the workers and the outputs.
    Returns the merged report."""
    started = time.monotonic()
    run = Run(binary, scenarios, arguments, mode, work or Path(output) / "shards", iterations, samples)
    ids = select(run.members, limit, seed)
    outcomes, count = execute(run, ids, workers, shard_size)
    report = merge(run, outcomes, output, include_cases)
    report.update(scenario_count=len(ids), not_applicable_scenario_count=len(ids) - report["applicable_scenario_count"],
                  limit=limit, selection_seed=seed, workers=workers, shards=count, wall_seconds=time.monotonic() - started,
                  **isolation.execution(workers))
    write(Path(output) / "benchmark.json", report)
    isolation.report_aborts(report["abort_counts"], f"{run.mode} {Path(output).name}")
    print(f"benchmarked {len(ids)} scenarios and {report['solves']} solves in {report['wall_seconds']:.1f}s with {workers} processes", flush=True)
    return report
