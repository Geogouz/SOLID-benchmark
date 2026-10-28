"""One thread per solve, the machine and the timing of a run, and the isolation of solves that crash their process.

The adapters and the Bern CLI link OpenBLAS statically on Linux. The repository's cargo configuration
(.cargo/config.toml) builds it as a single-threaded library with locking whatever the environment of the build, where
a threaded build would start one thread per core in every process unless told otherwise. Threads that share a
reduction can change the last digits of a result with their number, and many processes that each start as many threads
as there are cores overload the machine. Every baseline runner therefore starts its executables with Rayon, OpenBLAS
and OpenMP limited to one thread and parallelizes across processes, so that a result depends on its inputs alone.
SOLID's runner solid-units takes the same settings and solves one unit per thread (research/solid/common.py). The
settings do not limit Apple Accelerate, which Clarabel links in place of OpenBLAS on macOS and which reads
VECLIB_MAXIMUM_THREADS instead. That matters on a Mac alone. Every solve of the evaluation, the selections and the fits
runs on the Linux node, where OpenBLAS takes the settings, and no solver, fit or selection runs on real data on a Mac.

The release build aborts on a panic, and Jin's SDP solver panics on a few inputs. The abort ends the process with
every other solve of its request. One rule, ABORT_RULE, holds for SOLID and every baseline alike. A request whose
process crashes is solved again in smaller parts, unit by unit and then method by method, and a single solve that
still crashes is recorded as that method's failure on that unit with the time until the crash, as an adapter records
an error that a method returns, and it prints a warning. The scorer charges it the failure penalty, and every run
record and summary counts such solves per method (abort_counts), however many there are. Any other nonzero exit, an
input the adapter refuses or a run stopped from outside, raises AdapterError and stops the run, so that no such exit
turns into failures of methods.

Every run record names its host, its threads, the workers that solved at once and how they ran in parallel
(execution), under the timing rule TIMING, and the starts of every method (start_counts, STARTS), since the solve times
of methods compare only between runs that solved alike.

Every executable of research/cli embeds the identity of the sources it is built from and prints it with --identity, a
digest of the files of its own source trees (research/cli/build.rs), so that a commit that changes records or documents
alone leaves every executable as it is. source_identity computes the same digest from the working tree, so a script
that builds an executable can check what it embeds.

Usage: isolation.py   prints the source identity of the working tree
"""
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# the thread settings of every solve process. They bind on the Linux node, where every solve runs. Apple Accelerate,
# which Clarabel links on macOS, reads VECLIB_MAXIMUM_THREADS and not these, so on a Mac Jin's SDP may take more threads.
SINGLE_THREAD = dict(RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
# the signals of a crash inside a solve, a panic under panic = "abort", a trap of the compiled code or a fault of
# native code
CRASH_SIGNALS = frozenset(int(s) for s in (signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL, signal.SIGFPE, signal.SIGTRAP))
# the exit status of a Rust panic that unwinds, in a build without panic = "abort"
PANIC_STATUS = 101
# the failures of solves that crashed their process, as the baseline runners and SOLID's runner record them
ABORT_PREFIXES = ("adapter aborted", "runner aborted")
ABORT_RULE = ("A solve that crashes its process is scored as that method's failure on that unit, with the failure penalty on every "
              "pair of the unit, for SOLID and every baseline alike and without a ceiling on their number. A crash is a panic, a trap "
              "or a fault of native code, the signals SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV and SIGTRAP or the exit status 101 of "
              "a panic that unwinds. A request that crashes is solved again in parts down to the single solves that crash, each of "
              "them prints a warning, and every run record and summary counts them per method. Any other exit that is not zero "
              "stops the run, an input the solver refuses as well as a process ended from outside or out of memory, whether the "
              "system's killer of processes that exhaust the memory ends it or its allocator aborts it after an allocation failed.")
# the message the Rust allocator prints before it aborts the process when an allocation fails, a process out of memory
ALLOCATION_FAILURE = re.compile(r"memory allocation of \d+ bytes failed")
# how the solve times of every run are taken, and what its starts count
PROCESSES, THREADS = "one single-threaded process per request", "one thread per unit in one process"
TIMING = ("A unit's time is the wall time of one method's solve of it, taken inside the solving process around the solve, with the "
          "solve on one thread and at most the run's workers solving at once on its host. The baselines, and SOLID on TWR-CLOUD-Bern, "
          "solve every request in a single-threaded process of its own, and SOLID on the other datasets in one solid-units process "
          "with one thread per unit. Every process takes Rayon, OpenBLAS and OpenMP at one thread. Times compare only between runs "
          "on the same host with the same workers, and every run record names both and how its solves ran in parallel.")
STARTS = ("The starts of a run count, per method and number of starts, the solves that started that many searches, as every solver "
          "reports them with its estimate. SOLID's cooperative solve starts from its start layout and from the jittered copies of it "
          "that its profile asks for. The start layout is the engine's geometric layout, except that with the block range model of "
          "the anchor aided mode an agent sits at the peak of its anchor posterior in place of its lateration when its block range links "
          "reach three horizontal anchor places or more, anchors at one horizontal position counting once, that do not all lie within "
          "1 m of one line. In the anchor aided mode every stage starts once more from the laterated agents when an agent ranges three "
          "anchors or more, after the peak start as after the geometric layout. Every baseline reports one start. SOLID's anchor only "
          "solve evaluates its posterior on a grid over the unit's area and starts no search, so its run records no starts. A failed "
          "solve reports none.")
STAYED = ("A solve stayed at its start when its estimate equals, agent for agent, the positions it started from, the unit's starting "
          "coordinates for the provided start and the engine's geometric layout of the unit for the geometry start. Every selection "
          "record counts such solves per candidate and every run record per method, over the solves of FNL 2025 and STRONG 2021 that "
          "return an estimate. Jin 2021's relaxation and ADE-BP 2015's particles take no start, and SOLID reports none.")
# progress lines of the adapters and of the Bern CLI, and the Rust hint that follows a panic
PROGRESS = re.compile(r"^(?:[A-Za-z]+ )?unit \d+ samples \d+ complete$|^processed \d+/\d+ scenarios in |^note: ")


class AdapterError(RuntimeError):
    """An adapter or the CLI ended with an exit that is not a crash of a solve."""


def environment(base=None):
    """The environment of an adapter process, the caller's with one thread for Rayon, OpenBLAS and OpenMP."""
    result = dict(os.environ if base is None else base)
    result.update(SINGLE_THREAD)
    return result


def host():
    """The machine a run executes on, for its run record."""
    return dict(node=platform.node(), platform=platform.platform(), machine=platform.machine(), processors=os.cpu_count())


def commit(root=None):
    """The commit the checkout stands at, which the run record of every baseline run and of every TWR-CLOUD-Bern run
    names, so that the final tables can show that a run of an evaluation half ran at a commit that holds the declared
    analysis and the method freeze. None outside a git checkout."""
    run = subprocess.run(["git", "-C", str(root or ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else None


def execution(workers=None, parallel=PROCESSES):
    """The execution settings every run record carries. They are its host, one thread per process and the thread
    settings of the environment, and under timing the rule of its solve times with the workers that solved at once and
    how they ran in parallel, PROCESSES for the baselines and THREADS for solid-units."""
    return dict(host=host(), threads_per_process=1, environment=dict(SINGLE_THREAD),
                timing=dict(rule=TIMING, workers=workers, parallel=parallel, threads_per_solve=1))


def crashed(returncode, stderr=""):
    """Whether a solve crashed its process, a crash signal as a panic of the release build or a fault of native code ends
    it, or the exit status of a panic that unwinds. A process whose allocator aborts it after an allocation failed ran
    out of memory, which stops the run as the system's killer does, and is no crash."""
    if ALLOCATION_FAILURE.search(stderr or ""):
        return False
    return (returncode < 0 and -returncode in CRASH_SIGNALS) or returncode == PANIC_STATUS


def aborted(stderr):
    """The panic message of a crashed process, without its progress lines and the toolchain's registry path."""
    lines = [line for line in (stderr or "").splitlines() if line.strip() and not PROGRESS.search(line)]
    return re.sub(r"\S*/registry/src/[^/]+/", "", " ".join(lines))[-400:]


def failure_message(returncode, stderr, process="adapter"):
    """The failure of a single solve that crashed its process, the adapter, the CLI or SOLID's runner."""
    return f"{process} aborted with exit status {returncode}: {aborted(stderr)}"


def is_abort(failure):
    """Whether a recorded failure is a solve that crashed its process."""
    return (failure or "").startswith(ABORT_PREFIXES)


def warn_crash(solve, failure):
    """The warning a single solve that crashed its process prints, on standard error so that no log misses it."""
    print(f"WARNING {solve} crashed its process and is scored as that method's failure on that unit. {failure}", file=sys.stderr, flush=True)


def abort_counts(rows, methods=()):
    """The solves of every method that crashed their process, from scored rows or records with their failure, the
    methods given counted at zero when they have none."""
    counts = dict.fromkeys(methods, 0)
    for row in rows:
        counts.setdefault(row["method"], 0)
        counts[row["method"]] += is_abort(row.get("failure"))
    return dict(sorted(counts.items()))


def report_aborts(counts, title):
    """Print the crashed solves of every method of a run for its summary, and a warning on standard error for every
    method with any."""
    for method, count in counts.items():
        print(f"  {title} {method:12s} {count} solves crashed their process and are scored as failures", flush=True)
        if count:
            print(f"WARNING {title}: {count} solves of {method} crashed their process and are scored as its failures", file=sys.stderr, flush=True)


def start_counts(rows):
    """Per method, the solves that reported each number of starts (STARTS), from scored rows or records that carry it."""
    counts = {}
    for row in rows:
        if row.get("starts") is not None:
            counts.setdefault(row["method"], Counter())[str(row["starts"])] += 1
    return {method: dict(sorted(counter.items(), key=lambda item: int(item[0]))) for method, counter in sorted(counts.items())}


def reported_starts(record):
    """The starts a solver reports with a record's estimate, or at the record's top level as solid-units does, None for
    a failed solve."""
    metrics = (record.get("estimate") or {}).get("metrics") or record.get("metrics") or {}
    return metrics.get("starts") if record.get("estimate") is not None else None


def reported_stay(record):
    """Whether a solve's estimate stayed at its start, as the adapters report it with the estimate (STAYED), None for a
    solve without a start or without an estimate."""
    return (record.get("estimate") or {}).get("stayed_at_start") if record.get("estimate") is not None else None


def stayed_counts(rows, key="method"):
    """Per method, or per the value of another field, the solves that report whether they stayed at their start, those
    that stayed and their share (STAYED), from scored rows or records that carry stayed_at_start. A field without such a
    solve is left out."""
    counts = {}
    for row in rows:
        stayed = row.get("stayed_at_start")
        if stayed is None:
            continue
        entry = counts.setdefault(row[key], dict(solves=0, stayed=0))
        entry["solves"] += 1
        entry["stayed"] += bool(stayed)
    for entry in counts.values():
        entry["share"] = entry["stayed"] / entry["solves"]
    return dict(sorted(counts.items()))


def summed_stays(entries):
    """The counts of stayed_counts entries added up, with their share, None without any solve."""
    solves = sum(entry["solves"] for entry in entries if entry)
    stayed = sum(entry["stayed"] for entry in entries if entry)
    return dict(solves=solves, stayed=stayed, share=stayed / solves) if solves else None


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def isolated(request, run, parts, failed, path):
    """The records of a request, solving the parts of a request whose process crashes.

    run(request, path) returns (returncode, records, stderr, elapsed_ns) with the records of a process that ended
    normally. parts(request) gives the smaller requests of a crashed one in the order of its records, none for a single
    solve, and failed(request, returncode, stderr, elapsed_ns) gives the records of a single solve that crashed. Every
    part runs in its own folder below path."""
    path = Path(path)
    returncode, records, stderr, elapsed = run(request, path)
    if returncode == 0:
        return records
    if not crashed(returncode, stderr):
        raise AdapterError(f"exit status {returncode}: {stderr.strip()[-2000:]}")
    pieces = parts(request)
    if not pieces:
        return failed(request, returncode, stderr, elapsed)
    return [record for index, piece in enumerate(pieces) for record in isolated(piece, run, parts, failed, path / f"part-{index}")]


def adapter_parts(request):
    """The parts of an adapter request: its units one by one, and a single unit's methods one by one."""
    units, baselines = request["units"], request["baselines"]
    if len(units) > 1:
        return [dict(request, units=[unit]) for unit in units]
    if len(baselines) > 1:
        return [dict(request, baselines=[entry]) for entry in baselines]
    return []


def adapter_failure(request, returncode, stderr, elapsed):
    """The record of a single solve that crashed the adapter, in the layout of the adapter's own records, with its
    warning."""
    unit, entry = request["units"][0], request["baselines"][0]
    failure = failure_message(returncode, stderr)
    warn_crash(f"{entry['method']} on unit {unit['id']} with {unit['samples']} samples", failure)
    return [json.dumps(dict(id=unit["id"], samples=unit["samples"], method=entry["method"], agent_ids=unit["problem"]["agent_ids"],
                            estimate=None, elapsed_ns=elapsed, failure=failure))]


def solve_request(binary, request, path):
    """The adapter's prediction lines for a JSON request, one single-threaded process per part, with every solve that
    crashes the adapter isolated and recorded as its method's failure. The lines keep the adapter's order."""

    def run(part, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "request.json").write_bytes(encoded(part))
        started = time.monotonic_ns()
        process = subprocess.run([str(binary), str(folder / "request.json"), str(folder / "predictions.jsonl")],
                                 text=True, capture_output=True, env=environment())
        elapsed = time.monotonic_ns() - started
        lines = (folder / "predictions.jsonl").read_text().splitlines() if process.returncode == 0 else None
        return process.returncode, lines, process.stderr, elapsed

    return isolated(request, run, adapter_parts, adapter_failure, path)


ROOT = Path(__file__).resolve().parents[3]
# the files the executables of research/cli are built from, as research/cli/build.rs lists them, an entry with
# suffixes standing for every file below its directory with one of them, symbolic links left out, and an entry without
# for one file. The cargo configuration fixes the build settings of OpenBLAS, so a change of them changes the identity.
SOURCES = (("rust-toolchain.toml", ()), (".cargo/config.toml", ()), ("runtime/Cargo.toml", ()), ("runtime/engine/Cargo.toml", ()),
           ("runtime/engine/src", ("rs",)), ("research/Cargo.toml", ()), ("research/Cargo.lock", ()), ("research/cli/Cargo.toml", ()),
           ("research/cli/build.rs", ()), ("research/cli/src", ("rs",)), ("research/evaluation/Cargo.toml", ()),
           ("research/evaluation/src", ("rs",)), ("research/io/Cargo.toml", ()), ("research/io/src", ("rs",)), ("research/methods/Cargo.toml", ()),
           ("research/methods/src", ("rs",)), ("research/benchmarks/datasets", ("rs",)), ("research/solid", ("rs",)))


def source_identity(root=None):
    """The identity of the sources the executables of research/cli are built from, the one each embeds: the SHA-256 of
    the lines "<SHA-256 of the file>  <path>" of every file of SOURCES in the order of their paths."""
    root = Path(root or ROOT)
    files = {}
    for name, suffixes in SOURCES:
        path = root / name
        if not suffixes:
            if path.is_file():
                files[name] = path
            continue
        for directory, _, names in os.walk(path):
            for file in names:
                found = Path(directory) / file
                if found.suffix[1:] in suffixes and not found.is_symlink():
                    files[found.relative_to(root).as_posix()] = found
    listing = "".join(f"{hashlib.sha256(files[name].read_bytes()).hexdigest()}  {name}\n" for name in sorted(files))
    return hashlib.sha256(listing.encode()).hexdigest()


def build(binary):
    """The build of an executable as a record names it, the SHA-256 of the file and the source identity it prints."""
    return dict(binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                binary_identity=subprocess.check_output([str(binary), "--identity"], text=True).strip())


def same_build(record, path, binary):
    """A record of runs or of a selection, refused unless the executable that runs now made it, the same hash and
    source identity, so that every method of a dataset and the settings selected for it come from one build."""
    if os.environ.get("SOLID_READER") == "1":
        return record  # a reader's build is never the one that made our records
    current = build(binary)
    if {key: record.get(key) for key in current} != current:
        raise ValueError(f"{path} was made by another build of {Path(binary).name} than the one that runs now ({current['binary_identity'][:12]}). "
                         "Every method of a dataset runs with one build, so run it again with this build, after moving a selection "
                         "record aside when it is one")
    return record


if __name__ == "__main__":
    print(source_identity())
