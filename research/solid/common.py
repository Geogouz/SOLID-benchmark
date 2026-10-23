"""Shared pieces of the SOLID dataset runners: paths, the runner binary, balanced percentiles and scoring."""
import bisect
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# A reader's reproduction (SOLID_READER=1) skips the gates that only the research process can satisfy, the records of
# the design rounds and of the method freeze among them, and takes the method of method_constants.py.
READER = os.environ.get("SOLID_READER") == "1"
# the data folder, research/data unless SOLID_DATA names another
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
EXTERNAL = DATA / "external-benchmarks"
ARTIFACTS = ROOT / "research/artifacts/solid"
PROFILES = ARTIFACTS / "profiles"
SELECTIONS = ARTIFACTS / "selections"
FITS = ARTIFACTS / "fits"
OUT = ROOT / "out/solid"
# the Bern held-scan anchor only corpora, built by the evaluation release's rules (generate_heldscan_corpora.py)
BERN_ANCHOR_ONLY_CORPUS = Path(os.environ.get("SOLID_ANCHOR_ONLY_CAMPAIGN", OUT / "unify/bern-anchor-only-scenarios-v2"))
BINARY = ROOT / "research/target/release/solid-units"
DATASETS = ROOT / "research/benchmarks/datasets"
if str(DATASETS) not in sys.path:
    sys.path.insert(0, str(DATASETS))
import isolation  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, indent=1):
    """Write a JSON file whole: to a temporary of this process alone, then in its place, so processes that write the
    same file at once each replace it with a complete file and never write into one temporary together."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=indent, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tracked_profile(path):
    """Whether a path names a file of the tracked profile directory research/artifacts/solid/profiles."""
    return Path(path).resolve().parent == PROFILES.resolve()


def verify_profile(path, composed):
    """The content of a tracked profile file, refused unless it equals the composition of the fit records."""
    stored = read_json(path)
    if fingerprint(stored) != fingerprint(composed):
        raise ValueError(f"The tracked profile {path} is stale. Its content hash {fingerprint(stored)[:12]} differs from the "
                         f"composition of the fit records and the deployment record, {fingerprint(composed)[:12]}. Regenerate it with "
                         "research/solid/profiles.py, or run without --profile to solve with the composition.")
    return stored


def cache_matches(manifest, identity, outputs):
    try:
        record = read_json(manifest)
        return record["identity"] == identity and record["outputs"] == {p.name: sha256(p) for p in outputs}
    except (OSError, ValueError, KeyError):
        return False


def seal_cache(manifest, identity, outputs, **extra):
    write_json(manifest, dict(identity=identity, outputs={p.name: sha256(p) for p in outputs}, **extra))


def prediction_records(path, units):
    expected = {unit["id"]: unit for unit in units}
    if len(expected) != len(units):
        raise ValueError("Duplicate requested unit IDs")
    records = {}
    with open(path) as stream:
        for line in stream:
            record = json.loads(line)
            key = record["id"]
            if key not in expected or key in records or record["samples"] != expected[key]["samples"]:
                raise ValueError(f"Unexpected or duplicate prediction identity in {path}")
            records[key] = record
    if records.keys() != expected.keys():
        raise ValueError(f"{path} holds {len(records)} records for {len(units)} units")
    return records


# the exit status of a runner that aborts, the engine's panic in a release build, one of the crashes of
# isolation.ABORT_RULE
ABORTED = -signal.SIGABRT
ABORT_PREFIX = "runner aborted"


def aborted_record(unit, elapsed_ns, returncode, stderr):
    """The record of a unit whose solve crashes the runner, no estimate and the crash as its failure, with its warning."""
    failure = isolation.failure_message(returncode, stderr, "runner")
    isolation.warn_crash(f"solid on unit {unit['id']} with {unit['samples']} samples" + (f" of {unit['label']}" if unit.get("label") else ""), failure)
    return dict(id=unit["id"], samples=unit["samples"], label=unit.get("label"), method="solid", agent_ids=unit["problem"]["agent_ids"],
                estimate=None, elapsed_ns=elapsed_ns, failure=failure)


def isolated_lines(units, folder, workers):
    """The prediction lines of a request solved whole, or split by split_lines when its solve crashes the runner
    (isolation.crashed). Any other failure of the runner raises. The runner takes the thread settings of every baseline
    process, and solves one unit per thread."""
    folder.mkdir(parents=True, exist_ok=True)
    request, predictions = folder / "request.json", folder / "predictions.jsonl"
    write_json(request, dict(units=units), indent=None)
    started = time.monotonic_ns()
    run = subprocess.run([str(BINARY), str(request), str(predictions), str(workers)], capture_output=True, text=True, env=isolation.environment())
    if run.returncode == 0:
        return predictions.read_text().splitlines()
    if not isolation.crashed(run.returncode, run.stderr):
        raise subprocess.CalledProcessError(run.returncode, run.args, run.stdout, run.stderr)
    return split_lines(units, folder, workers, time.monotonic_ns() - started, run.returncode, run.stderr)


def split_lines(units, folder, workers, elapsed_ns, returncode, stderr):
    """The prediction lines of a request whose solve just crashed the runner, the crash as the failure of a single
    unit, and otherwise its two halves each solved again, down to the units that crash."""
    if len(units) == 1:
        return [json.dumps(aborted_record(units[0], elapsed_ns, returncode, stderr))]
    half = len(units) // 2
    return isolated_lines(units[:half], folder / "a", workers) + isolated_lines(units[half:], folder / "b", workers)


def solve(units, path, workers):
    """Reuse predictions only when the inputs, executable and output bytes match.

    A request whose solve crashes the runner is solved again in halves down to the units that crash, as the baseline
    runners isolate the crashes of their adapters. Such a unit is recorded as its failure with a warning, and the scorer
    charges it the failure penalty, however many units crash, the rule of every method (isolation.ABORT_RULE). Every
    other record is the runner's own output, and any other failure of the runner raises, one the system's killer of
    processes that exhaust the memory causes among them. The manifest records the crashed units of a sealed request."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("Duplicate requested unit IDs")
    identity = dict(request_sha256=fingerprint(dict(units=units)), binary_sha256=sha256(BINARY), workers=workers)
    manifest = path.with_suffix(".manifest.json")
    if not cache_matches(manifest, identity, [path]):
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="solve-", dir=path.parent) as temporary:
            lines = isolated_lines(units, Path(temporary) / "isolated", workers)
            aborted = sum(1 for line in lines if isolation.is_abort(json.loads(line).get("failure")))
            predictions = Path(temporary) / "predictions.jsonl"
            predictions.write_text("".join(line + "\n" for line in lines))
            prediction_records(predictions, units)
            if sha256(BINARY) != identity["binary_sha256"]:
                raise ValueError("Runner changed during execution")
            predictions.replace(path)
        if aborted:
            print(f"WARNING {path.name}: {aborted} of {len(units)} solves crashed the runner and are scored as SOLID's failures",
                  file=sys.stderr, flush=True)
        seal_cache(manifest, identity, [path], **({"aborted_units": aborted} if aborted else {}))
        print(f"{path.name}: {len(units)} units in {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    return prediction_records(path, units)


def record_for_scoring(unit, record):
    """The campaign scorer's record layout for one solved unit."""
    return dict(id=unit["id"], samples=unit["samples"], method="solid", agent_ids=record["agent_ids"],
                estimate=record["estimate"], elapsed_ns=record["elapsed_ns"], failure=record["failure"])


def score(request, reference, records, scorer):
    """Score every unit of a request with the campaign scorer. records maps (id, samples) to a runner record. Every
    scored row also carries, under pair_sets, the mean pair error and the pair count of every pair set of its unit's
    inputs (research/benchmarks/datasets/scoring.py), so a round can decide on any declared pair set."""
    import scoring
    converted = [record_for_scoring(unit, records[(unit["id"], unit["samples"])]) for unit in request["units"]]
    return scoring.score_with_pair_sets(request, reference, converted, scorer, ("solid",))


SCORER_SHA256 = "0068d72b94db6b1c40414829390025a3b2b12e9b9d1fd7d939fff5fc97659073"


def load_scorer():
    """The released TWR-CLOUD-Bern scorer as a module, loaded from its frozen input file without the dataset tooling."""
    import importlib.util
    path = DATA / "inputs/0068d72b94db6b1c/score_twr_cloud_bern.py"
    if sha256(path) != SCORER_SHA256:
        raise ValueError("The released scorer identity changed")
    spec = importlib.util.spec_from_file_location("released_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def balanced(errors, cells):
    """Balanced S50, S75 and S90 with the released scorer's rule: every cell receives the same total weight."""
    grouped = {}
    for error, cell in zip(errors, cells):
        grouped.setdefault(cell, []).append(float(error))
    scorer = load_scorer()
    return [float(scorer.balanced_percentile(grouped, q)) for q in (0.5, 0.75, 0.9)]


def exact_level(alpha):
    """A percentile level as an exact fraction. A float is read as the decimal it prints as, so 0.9 is nine tenths."""
    if isinstance(alpha, float):
        alpha = repr(alpha)
    level = Fraction(alpha)
    if not 0 < level <= 1:
        raise ValueError(f"percentile level {alpha} outside (0, 1]")
    return level


def balanced_percentile_exact(cell_values, alpha):
    """The released scorer's balanced percentile with the balanced CDF in exact integer arithmetic.

    The balanced CDF at r is the mean over the nonempty cells of the share of the cell's values at or below r, and the
    percentile is the smallest value of any cell where it reaches alpha. The released scorer sums the shares in floating
    point, so where the CDF equals alpha exactly at a value its rounded sum can fall just below alpha and the scorer
    returns the next value. Here every share is counted in units of 1/(K L), with K the number of cells and L the least
    common multiple of their sizes, and the comparison with alpha is exact. The two agree everywhere else. Decisions
    use this function; the headline tables keep the released scorer.
    """
    level = exact_level(alpha)
    nonempty = [sorted(float(v) for v in values) for values in cell_values.values() if len(values)]
    if not nonempty:
        return math.nan
    if not all(math.isfinite(v) for values in nonempty for v in values):
        raise ValueError("balanced percentile of non finite values")
    common_size = math.lcm(*(len(values) for values in nonempty))
    shares = [common_size // len(values) for values in nonempty]
    target = level.numerator * len(nonempty) * common_size
    candidates = sorted(set().union(*nonempty))
    lo, hi, result = 0, len(candidates) - 1, candidates[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        count = sum(bisect.bisect_right(values, candidates[mid]) * share for values, share in zip(nonempty, shares))
        if count * level.denominator >= target:
            result, hi = candidates[mid], mid - 1
        else:
            lo = mid + 1
    return result


def balanced_exact(errors, cells, levels=(0.5, 0.75, 0.9)):
    """Balanced percentiles in exact arithmetic, S50, S75 and S90 by default, for decisions."""
    grouped = {}
    for error, cell in zip(errors, cells):
        grouped.setdefault(cell, []).append(float(error))
    return [float(balanced_percentile_exact(grouped, q)) for q in levels]


def abort_counts(rows):
    """The solves of SOLID that crashed the runner, from its scored rows, in the layout of every run record
    (isolation.abort_counts)."""
    return isolation.abort_counts([dict(method="solid", failure=row.get("failure")) for row in rows], ("solid",))


def summarize(rows, title, mode, groups=None):
    """Print per sample track and reported group the balanced S50, S75 and S90 under the joined pair set of
    research/benchmarks/datasets/scoring.py and the supplementary ranged pairs, each with the share of pairs and units
    it scores, and the mean joined error per level. The anchor only mode reports the headline over three or more anchors
    and every smaller anchor count alone, unless groups splits the rows of a track otherwise (scoring.summary). Last
    comes the count of solves that crashed the runner, with a warning for any (isolation.report_aborts)."""
    import scoring
    tables = scoring.summary(rows, mode, load_scorer().balanced_percentile, groups)
    for samples, entries in tables.items():
        track = [row for row in rows if row["samples"] == int(samples)]
        for label, entry in entries.items():
            for table in entry.values():
                print(f"{title} samples={samples} {label}: " + scoring.describe(table), flush=True)
        scored = [row for row in track if row["pair_sets"][scoring.PAIR_SET]["mean_pair_error_cm"] is not None]
        levels = sorted({row["level"] for row in scored}, key=lambda v: (isinstance(v, str), v))
        if 1 < len(levels) <= 16:
            print(f"   mean {scoring.PAIR_SET} error per level: " + ", ".join(
                f"{level}: {np.mean([r['pair_sets'][scoring.PAIR_SET]['mean_pair_error_cm'] for r in scored if r['level'] == level]):.1f}"
                for level in levels), flush=True)
    isolation.report_aborts(abort_counts(rows), title)


def source_identity():
    """The identity of the code a run executes: the identity the runner binary embeds and prints with --identity, the
    digest of the sources it is built from (research/cli/build.rs), the same digest of the working tree and whether the
    two agree, with the commit checked out as information."""

    def output(command):
        try:
            run = subprocess.run(command, capture_output=True, text=True)
        except OSError:
            return None
        return run.stdout.strip() if run.returncode == 0 else None
    embedded, sources = output([str(BINARY), "--identity"]), isolation.source_identity(ROOT)
    return dict(binary=embedded, sources=sources, binary_matches_sources=embedded == sources, commit=output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]))


def execution(workers):
    """The machine and the threads of a run, the host of isolation.execution, one solid-units process with `workers`
    threads that each solve one unit at a time, the thread settings every baseline process takes too, and the timing
    rule of its solve times with the workers that solved at once."""
    return dict(isolation.execution(workers, isolation.THREADS), threads_per_process=workers, processes=1)


def starts(manifests, records, mode):
    """The starts SOLID's solves of a run report, from its predictions (isolation.STARTS), none in the anchor only
    mode, whose solve evaluates a grid and whose metrics count its joint layout samples in that field."""
    if mode == "ANC":
        return None
    reported = []
    for path, record in zip(manifests, records):
        for name in record["outputs"]:
            with open(path.parent / name) as stream:
                reported.extend(dict(method="solid", starts=isolation.reported_starts(json.loads(line))) for line in stream)
    return isolation.start_counts(reported)


def write_run(directory, rows, workers=None, **details):
    """Write the per unit scores and a run record with the binary hash and the identity of the code it runs, the host
    and threads of the run, the worker count and the timing rule of its solve times, the solves that crashed the runner
    and the starts its solves report, and the method's budgets, their factor and the hash of the budget check record.
    The runners add the method record and its hash with the solved profile."""
    import deployment
    directory = Path(directory)
    manifests = sorted(directory.rglob("*.manifest.json"))
    if not manifests:
        raise ValueError("No verified prediction manifests for this run")
    records = [read_json(path) for path in manifests]
    for path, record in zip(manifests, records):
        if not cache_matches(path, record["identity"], [path.parent / name for name in record["outputs"]]):
            raise ValueError("Predictions changed after execution")
    identities = [record["identity"] for record in records]
    binary_sha = sha256(BINARY)
    if any(identity["binary_sha256"] != binary_sha or identity["workers"] != workers for identity in identities):
        raise ValueError("Prediction executable or worker count does not match this run")
    write_json(directory / "scores.json", rows, indent=None)
    write_json(directory / "run.json", dict(binary_sha256=identities[0]["binary_sha256"], units=len(rows), workers=workers,
                                            prediction_manifests={str(p.relative_to(directory)): sha256(p) for p in manifests},
                                            source_identity=source_identity(), execution=execution(workers), budget=deployment.budget_provenance(),
                                            abort_rule=isolation.ABORT_RULE, abort_counts=abort_counts(rows),
                                            starts=starts(manifests, records, details.get("mode")), starts_rule=isolation.STARTS,
                                            **details))
