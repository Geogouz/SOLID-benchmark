"""Held-out benchmarks HYMN and AAU Meshed UWB in anchor only mode, declared before any solve or fit.

Both datasets are evaluated once, after the method freeze, and never inform a design decision. This module writes and
checks the tracked declaration research/artifacts/solid/splits/heldout-benchmarks.json, which holds the archive
hashes, the extraction rule, the unit, split and calibration rules, the cells and a digest of every unit of both
halves. The unit plans follow from the declared rules and seeds, the published point and node lists and the reading
counts of the recordings. No range value enters a plan.

The declaration is the contract of two node stages, and both run only after the method freeze. Stage one builds the
training inputs, fits SOLID's link model and the deployment settings on them and selects the options of the baselines.
Its loaders parse the whole archive of each dataset, both halves, and from the evaluation half they compute only the
statistics the declaration states, the reading counts of its points and the report counts of its flights, so no
reading of an evaluation half enters a calibration, fit, selection or unit of stage one. Stage two checks every record
of stage one, builds the evaluation inputs and runs every method on both halves. Every check that could stop a step
after the freeze has the outcome the declaration states (rules.outcomes), since the freeze pins the sources and no
such stop could be fixed. Both baseline rosters follow the repository's rule for scenes with real height spread, so Jin
2021, FNL 2025 and STRONG 2021 run and ADE-BP, which has no spatial implementation, does not. A unit scores only the
agent pairs that its measured links join, the joined pair set of scoring.py and the rule of every dataset, and every
figure reports the share of pairs and units it scores. The headline of both datasets is the balanced percentiles over
the cells with three or more anchors, and the cells with one and two anchors are reported as separate rows (headline).

The method freeze. research/solid/method_freeze.py writes the record research/artifacts/solid/selections/method-freeze.json
once the confirmatory round has decided the method, and the record is committed. It holds the SHA-256 of the method
record and of the round's freeze, the identity of the sources (source_identity, every tracked file of the runtime and
the research tree without the records under research/artifacts, the papers and the Markdown documents), the identity
and SHA-256 of the solid-units and heldout-adapter executables (binaries), the Python environment, the SHA-256 of the
budget check record and, under FREEZE_KEY, the SHA-256 of this declaration and of the final analysis (freeze_pins).
Every builder, fitter, selection and runner of either half verifies all of them first (frozen), with the method record
and the budget check record committed and the likelihood of the fitter the one the round froze, so a change of any
code, executable, package or record of the method after the freeze runs nothing, and no link fit of another likelihood
can follow the opening of a step.

The ledger. The tracked record research/artifacts/solid/splits/heldout-evaluations.json records every step that reads
a held-out half after the freeze, and the stages commit each entry on a branch, never on a detached HEAD
(require_branch). An entry and its commit are one step of the stage, which leaves no entry behind when the commit
fails, tags the commit locally as heldout/<step>/<event> at once and logs its SHA with the UTC time outside the
repository. Stage one opens its entry before it reads any training half
and closes it with the SHA-256 of every record it writes, which every run requires (require_stage_one). Stage two
opens a dataset before the first read of its evaluation half, runs every method on both halves and closes the dataset
with the SHA-256 of its populations, requests, references, run and score files and executables. A dataset that the
ledger closes in any version that a commit of any branch, tag or side of a merge holds (ledger_history) is not
evaluated again, and an entry under another method freeze refuses the freeze, so the method is not frozen again once
a held-out half has been read (require_freeze, require_run).

No range statistic, calibration, fit, selection or score of either half is stored or displayed before the freeze
beyond the statistics that the disclosures and revisions of the declaration list.

Builders write under research/data/external-benchmarks/<dataset>-layout, and read the archives under
research/data/downloads, in the data folder that the variable SOLID_DATA names. The variable SOLID_HELDOUT_DATA moves
the output folder, for tests and scratch builds. The archives stay read only and members are read in memory, bags
through a temporary file.

Usage: heldout.py [--check]
       heldout.py --revise
       heldout.py --pins
       heldout.py --check-freeze
       heldout.py --check-stage-one
       heldout.py --open stage-one|hymn|aau
       heldout.py --complete stage-one|hymn|aau
       heldout.py --evaluated hymn|aau

--check rebuilds the declaration from the archives and compares it with the tracked record. It parses every reading of
both archives but computes from them only the reading counts of the plans, the failed exchanges of the training halves
and the sign check on seq1, the statistics the declaration states, so it runs before the freeze as well. --revise
rewrites the parts of the tracked declaration that the code alone builds, its role, revisions, disclosures, rules and
code hashes, and keeps both dataset sections, which the archives give, as they are (revise). It reads no archive, and
--check at held-out stage one rebuilds every part from the archives. --pins prints
the entry the freeze record holds under FREEZE_KEY, and --check-freeze verifies the committed freeze record. Stage two
runs --check-stage-one before it reads anything. --open and --complete append the opening and the closing entry of a
step to the ledger, and the node stage commits it. --evaluated exits with status 0 when the ledger closes the dataset
under this freeze and with status 1 otherwise.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import audit

ROOT = audit.ROOT
# A reader's reproduction (SOLID_READER=1) has no method freeze and no ledger, so their gates let it pass, while every
# check of a built half against its own records and the declaration stays.
READER = os.environ.get("SOLID_READER") == "1"
DOWNLOADS = audit.DATA_ROOT / "downloads"
OUTPUT = Path(os.environ.get("SOLID_HELDOUT_DATA") or audit.DATA)
RECORD = ROOT / "research/artifacts/solid/splits/heldout-benchmarks.json"
# The declared analysis of the paper's final tables (research/solid/final_tables.py), which the method freeze pins too.
ANALYSIS = ROOT / "research/artifacts/solid/splits/final-analysis.json"
FREEZE = ROOT / "research/artifacts/solid/selections/method-freeze.json"
LEDGER = ROOT / "research/artifacts/solid/splits/heldout-evaluations.json"
SCHEMA = "solid.split.heldout-benchmarks.v1"
FREEZE_SCHEMA = "solid.selection.method-freeze.v1"
LEDGER_SCHEMA = "solid.split.heldout-evaluations.v1"
DATASETS = ("hymn", "aau")
SPLITS = ("training", "evaluation")
TRACKS = (1, 10)
BLOCK_READINGS = 10
CODE = ("research/benchmarks/datasets/heldout.py", "research/benchmarks/datasets/hymn_layout.py", "research/benchmarks/datasets/aau_layout.py",
        "research/solid/deployment.py")
# The key of the held-out pins in the method freeze record.
FREEZE_KEY = "heldout_benchmarks"
# The records of the method that the freeze pins by their SHA-256, the method record of the confirmatory round
# (profiles.METHOD_RECORD) and its budget check record (deployment.BUDGET_RECORD).
METHOD_RECORD = ROOT / "research/artifacts/solid/selections/method.json"
BUDGET_RECORD = ROOT / "research/artifacts/solid/selections/budget-check.json"
# The sources the freeze identifies, the scope of the round's freeze (select_union_ablation.SOURCE_SCOPE). Every tracked
# file of the runtime and the research tree is code. The records under research/artifacts are pinned by their own
# hashes, and the papers and the Markdown documents change no result.
SOURCE_SCOPE = ("runtime", "research", ":(exclude)research/artifacts", ":(exclude)research/papers", ":(exclude,glob)**/*.md")
# The executables the method freeze pins: those of the held-out runs, SOLID's runner and the baselines' adapter, and
# those of the evaluation runs of the other datasets, the Bern CLI, which runs SOLID and every baseline on
# TWR-CLOUD-Bern, and the adapters of the baselines on TIERS, Bregar and IDLab, so that every run a table of an
# evaluation half reads is of an executable whose SHA-256 the freeze holds.
BINARIES = {name: ROOT / "research/target/release" / name
            for name in ("solid-units", "heldout-adapter", "solid", "tiers-adapter", "bregar-adapter", "idlab-layout-adapter")}
# The executables that embed the identity of their sources without printing it, the Bern CLI, whose reports name it
# (research/cli/build.rs).
EMBEDDED = ("solid",)
# The steps of the ledger, held-out stage one and the runs of each dataset in held-out stage two.
STAGE_ONE = "stage-one"
STEPS = (STAGE_ONE, *DATASETS)
# The records held-out stage one writes, which its closing entry of the ledger pins and every run requires.
STAGE_ONE_RECORDS = tuple(ROOT / name for name in (
    "research/artifacts/solid/fits/hymn_layout_fit.json", "research/artifacts/solid/fits/aau_layout_fit.json",
    "research/artifacts/solid/fits/heldout-deployment.json", "research/artifacts/solid/profiles/hymn-anchor-only.json",
    "research/artifacts/solid/profiles/aau-anchor-only.json",
    *(f"research/artifacts/baselines/{dataset}-{method}-selection.json" for dataset in DATASETS for method in ("jin-2021", "fnl", "strong"))))
# The folders of the runs of the baselines (heldout_baselines.OUT) and of SOLID (common.OUT).
RUNS = dict(baselines=ROOT / "out/baselines", solid=ROOT / "out/solid")
# The range of 200 ns of flight at the speed of light is the end of the interval of raw timings that the engine and the
# baselines accept (max_tof_ns of the retained configurations). A reading beyond it is a failed exchange.
MAXIMUM_RANGE_CM = 200.0 * 29.9792458
# The anchor only headline covers the cells with at least this many anchors. The cells with fewer are separate rows.
HEADLINE_MINIMUM_ANCHORS = 3
LEDGER_RULE = ("Every step that reads a held-out half after the method freeze is recorded here, and the node stage commits every entry on "
               "a branch, tags that commit locally as heldout/<step>/<event> at once and logs its SHA with the UTC time outside the "
               "repository. Held-out stage one opens its entry before it reads any training half and closes it with the SHA-256 of every "
               "record it writes. Held-out stage two opens a dataset before the first read of its evaluation half and closes it with the "
               "SHA-256 of the population, request and reference of both halves, of the run and score files of every method on both and "
               "of the executables. A dataset is evaluated once, so it is not opened again once the ledger closes it as it is or in any "
               "version that a commit of any branch, tag or side of a merge holds. Every entry names the method freeze it ran under, and "
               "a method freeze other than the one the entries name runs nothing.")


class FreezeMissing(RuntimeError):
    """A held-out half was requested without a committed method freeze record that holds the method, the code, the
    executables and the environment as they are."""


class LedgerRefusal(FreezeMissing):
    """A held-out half was requested without the committed entry of the ledger that its step needs, or the ledger
    refuses it, a second evaluation of a dataset or a step under another method freeze."""


# ---------------------------------------------------------------------------------------------------------------------
# Files and values


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical(value):
    """The value as it reads back from JSON, tuples as lists and keys as strings."""
    return json.loads(encoded(value))


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write_json(path, value):
    """Write JSON atomically, compressed without a timestamp when the name ends in .gz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.suffix == ".gz":
        temporary.write_bytes(gzip.compress(encoded(value), mtime=0))
    else:
        temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def keep_or_write(path, value):
    """Write a built file, or keep an identical one. A different existing file stops the build for review."""
    path = Path(path)
    if path.exists():
        if read_json(path) != canonical(value):
            raise ValueError(f"{relative(path)} exists with other content. Remove it after review to rebuild")
        return
    write_json(path, value)


def relative(path):
    """A path relative to the repository without resolving the data link, so records read the same on every machine, a
    file of the data folder under research/data wherever SOLID_DATA puts it (audit.name)."""
    path = Path(path)
    return audit.name(path) if path.is_relative_to(ROOT) or path.is_relative_to(audit.DATA_ROOT) else str(path)


def require_archive(path, expected_md5, expected_sha256):
    """The archive's SHA-256, refused unless both published digests match."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{relative(path)} is missing. Copy the downloaded archive there before building")
    found = sha256(path)
    if md5(path) != expected_md5 or found != expected_sha256:
        raise ValueError(f"{relative(path)} differs from the declared archive")
    return found


def hash_order(seed, *parts):
    """A deterministic integer from a declared seed and the identity of a draw, independent of any library's generator."""
    return int.from_bytes(hashlib.sha256(encoded([seed, *parts])).digest()[:8], "big")


def ordered(items, seed, *parts):
    """The items sorted by the hash of the seed, the draw and the item, ties impossible in practice and broken by the item."""
    return sorted(items, key=lambda item: (hash_order(seed, *parts, item), item))


# ---------------------------------------------------------------------------------------------------------------------
# The freeze


def inside(path):
    """Whether a path lies inside the repository, where a record counts only once it is committed."""
    return Path(path).resolve().is_relative_to(ROOT.resolve())


def git(*arguments):
    """The output of a git command in the repository."""
    return subprocess.run(["git", "-C", str(ROOT), *arguments], check=True, capture_output=True, text=True).stdout


def committed(path):
    """Whether a file of the repository is tracked by git and identical to its version at HEAD."""
    name = str(Path(path).relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", name], cwd=ROOT, capture_output=True)
    unchanged = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", name], cwd=ROOT, capture_output=True)
    return tracked.returncode == 0 and unchanged.returncode == 0


def freeze_pins():
    """The SHA-256 of this declaration and of the declared analysis of the paper's final tables (ANALYSIS) keyed by
    their paths in the repository, the entry the method freeze record holds under FREEZE_KEY. Every source of the
    held-out benchmarks is code, which the source identity of the record covers, and heldout.py --pins prints the
    entry."""
    return {relative(path): sha256(path) for path in (RECORD, ANALYSIS)}


def unpinned(content):
    """The held-out files whose pin in the content of a freeze record differs from their hash now, every file when the
    record pins none."""
    current = freeze_pins()
    pinned = content.get(FREEZE_KEY)
    if not isinstance(pinned, dict):
        return sorted(current)
    return sorted(name for name in set(current) | set(pinned) if pinned.get(name) != current.get(name))


def source_files():
    """The SHA-256 of every tracked file of the source scope keyed by its path, refused while the scope holds a change
    that is not committed, a modified, deleted or untracked file."""
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=all", "--", *SOURCE_SCOPE).splitlines() if line]
    if dirty:
        raise FreezeMissing(f"{len(dirty)} changes to code are not committed, for example {', '.join(dirty[:5])}. The method freeze "
                            "identifies committed code")
    names = [name for name in git("ls-files", "-z", "--", *SOURCE_SCOPE).split("\0") if name]
    return {name: sha256(ROOT / name) for name in sorted(names)}


def listing(files):
    """The digest of files given as {path: SHA-256}, the SHA-256 of the lines "<SHA-256>  <path>" in the order of the
    paths, the listing of sha256sum, which isolation.source_identity digests for the sources of the executables."""
    return sha256_bytes("".join(f"{files[name]}  {name}\n" for name in sorted(files)).encode())


def source_identity():
    """The identity of the sources, the scope with the SHA-256 of every tracked file of it and their digest, refused
    while the scope holds a change that is not committed (source_files)."""
    files = source_files()
    return dict(scope=list(SOURCE_SCOPE), sha256=listing(files), files=files)


def binaries():
    """The identity each executable the freeze pins prints with --identity, the digest of the sources it was built from,
    and its SHA-256, both None for an executable that is missing or does not answer. An executable of EMBEDDED prints
    no identity, and its identity is that of the sources as they are when it holds that digest (embedded)."""
    result = {}
    for name, path in BINARIES.items():
        identity = digest = None
        if path.is_file():
            if name in EMBEDDED:
                identity = embedded(path)
            else:
                answer = subprocess.run([str(path), "--identity"], capture_output=True, text=True)
                identity = answer.stdout.strip() if answer.returncode == 0 else None
            digest = sha256(path)
        result[name] = dict(path=relative(path), identity=identity, sha256=digest)
    return result


def embedded(path, root=None):
    """The identity of the sources as they are (isolation.source_identity) when the executable holds that digest, the
    one research/cli/build.rs embeds and its reports name, and None when it holds another, so an executable built from
    other sources has no identity."""
    import isolation
    identity = isolation.source_identity(root or ROOT)
    return identity if identity.encode() in Path(path).read_bytes() else None


def environment():
    """The Python, numpy and scipy versions, the environment of confirm_round.environment, which the round and the
    method freeze record."""
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def likelihood():
    """The SHA-256 of the link likelihood that SOLID's held-out link fits come from, the source of
    research/solid/link_likelihood.py that the fitter imports and stamps on every fit record
    (fit_links.likelihood_identity)."""
    solid = str(ROOT / "research/solid")
    if solid not in sys.path:
        sys.path.insert(0, solid)
    import fit_links
    return fit_links.likelihood_identity()["sha256"]


def changed_parts(content):
    """The parts of the content of a method freeze record that differ from the method, the code, the executables, the
    environment and the declaration as they are now, each named with what differs. The likelihood of the fitter must
    be the one the round froze, so the likelihood of a held-out fit is checked before the ledger opens any step."""
    changed = []
    for key, path in (("method_record", METHOD_RECORD), ("budget_record", BUDGET_RECORD)):
        if not path.is_file() or (content.get(key) or {}).get("sha256") != sha256(path):
            changed.append(f"{key} {relative(path)}")
    if ((content.get("round_freeze") or {}).get("likelihood") or {}).get("sha256") != likelihood():
        changed.append("likelihood")
    recorded, current = content.get("sources") or {}, source_identity()
    if (recorded.get("scope"), recorded.get("sha256")) != (current["scope"], current["sha256"]):
        files = recorded.get("files") or {}
        names = sorted(name for name in set(files) | set(current["files"]) if files.get(name) != current["files"].get(name))
        changed.append("sources" + (f" {', '.join(names[:5])}" if names else ""))
    recorded, current = content.get("binaries") or {}, binaries()
    changed += [f"executable {name}" for name in sorted(set(recorded) | set(current)) if recorded.get(name) != current.get(name)]
    if content.get("environment") != environment():
        changed.append("environment")
    return changed + [f"pin {name}" for name in unpinned(content)]


def verified(record=None):
    """The content of the method freeze record and its identity, the path and SHA-256 of the record that the records of
    a build or a run keep. Refused while the record is absent, not a method freeze record or, inside the repository, not
    committed, while the method record or the budget check record it pins is not committed, while a part of it differs
    from what it records (changed_parts), and when the ledger holds an entry under another freeze. A freeze is a
    commit, so an untracked or edited record freezes nothing. A change of code, executables, packages or records of the
    method after the freeze runs nothing, and the method is not frozen again once the ledger records a step."""
    if READER:
        return {}, {}
    record = Path(record or FREEZE)
    if not record.is_file():
        raise FreezeMissing(f"Both held-out halves are built, fitted and solved only after the method freeze. {relative(record)} does "
                            "not exist, and research/solid/method_freeze.py writes it once the confirmatory round has decided the method")
    try:
        content = json.loads(record.read_text())
    except (OSError, ValueError) as error:
        raise FreezeMissing(f"{relative(record)} is not a readable JSON record") from error
    if not isinstance(content, dict) or content.get("schema") != FREEZE_SCHEMA:
        raise FreezeMissing(f"{relative(record)} is not a method freeze record of schema {FREEZE_SCHEMA}, which research/solid/method_freeze.py "
                            "writes")
    if inside(record) and not committed(record):
        raise FreezeMissing(f"{relative(record)} is not committed. The method freeze is a commit")
    for path in (METHOD_RECORD, BUDGET_RECORD):
        if path.is_file() and inside(path) and not committed(path):
            raise FreezeMissing(f"{relative(path)} is not committed. The method freeze pins committed records of the method")
    changed = changed_parts(content)
    if changed:
        raise FreezeMissing(f"{relative(record)} does not hold the method, the code, the executables, the environment and the declaration "
                            f"as they are. What differs is {', '.join(changed[:6])}" + (f" and {len(changed) - 6} more" if len(changed) > 6 else ""))
    digest = sha256(record)
    others = sorted({str(entry.get("method_freeze_sha256")) for entry in ledger_history()} - {digest})
    if others:
        raise LedgerRefusal(f"{relative(LEDGER)} holds entries under another method freeze, {', '.join(value[:12] for value in others)}. "
                            "The method is not frozen again once a held-out half has been read")
    return content, dict(freeze_record=relative(record), freeze_record_sha256=digest)


def frozen(record=None):
    """The identity of the verified method freeze, the path and SHA-256 of its record (verified)."""
    return verified(record)[1]


def freeze_commit(record=None):
    """The commit that holds the method freeze record as it is, the last commit that changed it, None for a record
    outside the repository."""
    record = Path(record or FREEZE)
    if not inside(record):
        return None
    return git("log", "-1", "--format=%H", "--", relative(record)).strip() or None


# ---------------------------------------------------------------------------------------------------------------------
# The ledger


def ledger_content(text, where):
    """The entries of one version of the ledger."""
    try:
        record = json.loads(text)
    except ValueError as error:
        raise ValueError(f"{where} is not a readable ledger") from error
    if not isinstance(record, dict) or record.get("schema") != LEDGER_SCHEMA or not isinstance(record.get("entries"), list):
        raise ValueError(f"{where} is not a ledger of schema {LEDGER_SCHEMA}")
    return record["entries"]


def ledger_entries(path=None):
    """The entries of the ledger as it is, none while it does not exist."""
    path = Path(path or LEDGER)
    return ledger_content(path.read_text(), relative(path)) if path.exists() else []


def ledger_commits(path=None):
    """Every commit that changed the ledger in the history of any branch, tag or other reference of the repository
    (git rev-list --full-history --all). The full history follows both sides of every merge, so the side of a merge
    that kept the other parent's ledger still counts, and every reference counts, so a commit that a reset or a rebase
    left to the tag the node stage gave it still counts."""
    name = relative(Path(path or LEDGER))
    return git("rev-list", "--full-history", "--all", "--", name).split()


def ledger_history(path=None):
    """Every entry the ledger holds as it is or in any version that a commit of ledger_commits holds, each once, so
    that an entry a later commit removed, a merge dropped or a reset left behind still counts."""
    path = Path(path or LEDGER)
    entries = ledger_entries(path)
    if inside(path):
        name = relative(path)
        for commit in ledger_commits(path):
            shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{name}"], capture_output=True, text=True)
            if shown.returncode == 0:
                entries = entries + ledger_content(shown.stdout, f"{name} at {commit[:12]}")
    unique = {}
    for item in entries:
        unique.setdefault(fingerprint(item), item)
    return list(unique.values())


def committed_ledger(path=None):
    """The entries of the ledger, refused while the ledger inside the repository differs from its version at HEAD, so
    that every entry a step relies on is committed."""
    path = Path(path or LEDGER)
    if path.exists() and inside(path) and not committed(path):
        raise LedgerRefusal(f"{relative(path)} holds entries that are not committed. The node stage commits every entry before the step it "
                            "opens reads any input")
    return ledger_entries(path)


def require_branch(path=None):
    """Refused while the repository that holds the ledger is on a detached HEAD, whose commits no branch keeps. Every
    entry is committed on a branch, and the node stage tags that commit at once."""
    path = Path(path or LEDGER)
    if inside(path) and subprocess.run(["git", "-C", str(ROOT), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode != 0:
        raise LedgerRefusal(f"The repository is on a detached HEAD, and {relative(path)} is committed on a branch only. Check out the branch of "
                            "the held-out stages first")


def under(entries, freeze):
    """The entries of the freeze of a freeze identity."""
    return [item for item in entries if item.get("method_freeze_sha256") == freeze["freeze_record_sha256"]]


def latest(entries, step):
    """The last entry of a step, None without one."""
    return next((item for item in reversed(entries) if item.get("step") == step), None)


def closed(dataset, entries=None):
    """Whether the ledger, as it is or in any version a commit holds (ledger_history), closes a dataset, whose
    evaluation is then done."""
    entries = ledger_history() if entries is None else entries
    return any(item.get("step") == dataset and item.get("event") == "completed" for item in entries)


def ledger_entry(step, event, freeze, record=None, **details):
    """An entry of the ledger for a step under a freeze, with the commit of the freeze record and the time."""
    return dict(step=step, event=event, freeze_commit=freeze_commit(record), method_freeze_sha256=freeze["freeze_record_sha256"],
                time=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **details)


def write_ledger(entries, path=None):
    write_json(path or LEDGER, dict(schema=LEDGER_SCHEMA, rule=LEDGER_RULE, entries=entries))


def open_step(step, record=None):
    """Append the opening entry of a step to the ledger and return it, or the entry that already opens it under this
    freeze. Stage one opens once per freeze. A dataset opens only once stage one has closed under this freeze, and
    never again once the ledger closes it. The node stage commits the entry before the step reads any input, and a
    detached HEAD is refused (require_branch)."""
    if step not in STEPS:
        raise ValueError(f"unknown step {step}, one of {', '.join(STEPS)}")
    require_branch()
    freeze = frozen(record)
    if step != STAGE_ONE:
        require_stage_one(record)
        if closed(step):
            raise LedgerRefusal(f"{relative(LEDGER)} closes {step}, whose evaluation half was evaluated. A held-out half is evaluated once")
    entries = committed_ledger()
    last = latest(under(entries, freeze), step)
    if last is not None and (last["event"] == "opened" or step == STAGE_ONE):
        return last
    item = ledger_entry(step, "opened", freeze, record)
    write_ledger(entries + [item])
    return item


def dataset_files(dataset):
    """The files the closing entry of a dataset pins, the population, request and reference of both halves and the run
    and score files of every method on both."""
    files = []
    for split in SPLITS:
        half = OUTPUT / f"{dataset}-layout" / split
        files += [half / "population.json", half / "request.json.gz", half / "reference.json"]
        files += [runs / dataset / split / "anc" / name for runs in RUNS.values() for name in ("run.json", "scores.json")]
    return files


def complete_step(step, record=None):
    """Append the closing entry of an open step to the ledger with the SHA-256 of what the step wrote and of the
    executables, and return it. Stage one pins its records (STAGE_ONE_RECORDS) and a dataset the files of
    dataset_files, whose run records must name this freeze. A step that the ledger already closes with the same files
    keeps its entry. A detached HEAD is refused (require_branch)."""
    if step not in STEPS:
        raise ValueError(f"unknown step {step}, one of {', '.join(STEPS)}")
    require_branch()
    freeze = frozen(record)
    entries = committed_ledger()
    last = latest(under(entries, freeze), step)
    if last is None:
        raise LedgerRefusal(f"{step} is not open under this freeze. Open it with heldout.py --open {step}")
    paths = list(STAGE_ONE_RECORDS) if step == STAGE_ONE else dataset_files(step)
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{step} does not close without {', '.join(missing)}")
    other = [relative(path) for path in paths if path.name == "run.json" and read_json(path).get("freeze_record_sha256") != freeze["freeze_record_sha256"]]
    if other:
        raise ValueError(f"{', '.join(other)} ran under another freeze")
    key = "records" if step == STAGE_ONE else "files"
    files = {relative(path): sha256(path) for path in paths}
    if last["event"] == "completed":
        if last.get(key) != files:
            changed = sorted(name for name in files if files[name] != (last.get(key) or {}).get(name))
            raise LedgerRefusal(f"{relative(LEDGER)} closed {step} with other files, and {', '.join(changed)} changed since")
        return last
    item = ledger_entry(step, "completed", freeze, record, binaries=binaries(), **{key: files})
    write_ledger(entries + [item])
    return item


def require_stage_one(record=None):
    """The closing entry of stage one under the current freeze. Refused unless the committed ledger holds it with the
    executables as they are and every record of stage one is committed with the SHA-256 it names. Every run of either
    half requires it (heldout_runner and heldout_baselines)."""
    if READER:
        return {}
    freeze = frozen(record)
    last = latest(under(committed_ledger(), freeze), STAGE_ONE)
    if last is None or last["event"] != "completed":
        raise LedgerRefusal("Held-out stage one has not closed under this freeze in the committed ledger, and its records are the inputs of "
                            "every run")
    if last.get("binaries") != binaries():
        raise LedgerRefusal("Held-out stage one closed with other executables than those of this freeze")
    for path in STAGE_ONE_RECORDS:
        if not path.is_file() or not committed(path):
            raise LedgerRefusal(f"{relative(path)} of held-out stage one is not committed")
        if sha256(path) != (last.get("records") or {}).get(relative(path)):
            raise LedgerRefusal(f"{relative(path)} changed after held-out stage one closed")
    return last


def require_freeze(split, record=None, dataset=None):
    """The freeze of a held-out half. Either half needs the verified method freeze (frozen) and the committed opening of
    stage one under it in the ledger, so that no half is read before the ledger records a step. An evaluation half
    also needs the committed opening of its dataset and is refused once the ledger closes the dataset in any version a
    commit holds (closed), so that it is evaluated once."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    if split == "evaluation" and dataset not in DATASETS:
        raise ValueError("an evaluation half is read for its dataset, hymn or aau")
    if READER:
        return {}
    freeze = frozen(record)
    entries = under(committed_ledger(), freeze)
    if latest(entries, STAGE_ONE) is None:
        raise LedgerRefusal("No held-out half is read before held-out stage one opens under this freeze and the ledger commits its entry. "
                            "Open it with heldout.py --open stage-one")
    if split == "evaluation":
        if closed(dataset):
            raise LedgerRefusal(f"{relative(LEDGER)} closes {dataset}, whose evaluation half was evaluated. A held-out half is evaluated once")
        last = latest(entries, dataset)
        if last is None or last["event"] != "opened":
            raise LedgerRefusal(f"The evaluation half of {dataset} is read only after held-out stage two opens {dataset} under this freeze and "
                                f"the ledger commits its entry. Open it with heldout.py --open {dataset}")
    return freeze


def require_run(dataset, split, record=None):
    """The freeze of a run of SOLID or the baselines on a held-out half, which the ledger records. Beyond the freeze of
    the half (require_freeze), the committed ledger must open the dataset under this freeze, so every run falls inside
    an entry, and a dataset that the ledger closes runs no more."""
    if dataset not in DATASETS:
        raise ValueError(f"{dataset} is not a held-out dataset")
    if READER:
        return {}
    freeze = require_freeze(split, record, dataset)
    last = latest(under(committed_ledger(), freeze), dataset)
    if closed(dataset) or last is None or last["event"] != "opened":
        raise LedgerRefusal(f"A run of {dataset} starts only while held-out stage two opens {dataset} under this freeze in the committed "
                            f"ledger, and {relative(LEDGER)} does not")
    return freeze


def half_identity(dataset, split):
    """The identity of a built half that every run record keeps, from its population record alone. It holds the SHA-256
    of the population and the request, the flights the build left out of the half beyond the relocation guard, those it
    left out of the training half, which every half reports, and the pairs the guard could not compare for want of an
    admissible report, all of them AAU's alone."""
    folder = OUTPUT / f"{dataset}-layout" / split
    population = read_json(folder / "population.json")
    return dict(population_sha256=sha256(folder / "population.json"), request_sha256=population["files"]["request"],
                excluded_flights=population.get("excluded_flights", {}), training_excluded_flights=population.get("training_excluded_flights", {}),
                unguarded_pairs=population.get("unguarded_pairs", {}))


def built_half(dataset, split):
    """The request, the reference and the identity of a built half (half_identity), refused before the method freeze and
    the ledger allow its read (require_freeze), when a file changed after its build or when the half was built under
    another declaration."""
    if dataset not in DATASETS:
        raise ValueError(f"{dataset} is not a held-out dataset")
    require_freeze(split, dataset=dataset)
    folder = OUTPUT / f"{dataset}-layout" / split
    population = read_json(folder / "population.json")
    files = dict(request=folder / "request.json.gz", reference=folder / "reference.json")
    for name, path in files.items():
        if sha256(path) != population["files"][name]:
            raise ValueError(f"the {dataset} {split} {name} changed after its build")
    if population["declaration_sha256"] != sha256(RECORD):
        raise ValueError(f"the {dataset} {split} half was built under another declaration")
    return read_json(files["request"]), read_json(files["reference"]), half_identity(dataset, split)


def training_calibration(path, archive_sha256):
    """The calibration a builder wrote from its training half, refused before the method freeze and the ledger allow a
    read of the half and unless it came from the declared archive under the current declaration without an evaluation
    row, which the calibration counts."""
    require_freeze("training")
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{relative(path)} is missing. Build the training half first, it holds the calibration")
    record = read_json(path)
    if (record.get("archive_sha256"), record.get("declaration_sha256"), record.get("evaluation_rows_used")) != (archive_sha256, sha256(RECORD), 0):
        raise ValueError(f"{relative(path)} was not built from the training half of the declared archive under this declaration")
    return record


# ---------------------------------------------------------------------------------------------------------------------
# Units


def cell(anchors):
    """The reporting cell of a unit, its anchor count."""
    return f"anchors-{anchors}"


def cell_anchors(name):
    """The anchor count of a reporting cell."""
    prefix, _, count = str(name).partition("-")
    if prefix != "anchors" or not count.isdigit():
        raise ValueError(f"{name} is not an anchor count cell")
    return int(count)


def headline_split(cells):
    """Declared cells split into the headline cells, three or more anchors, and the cells reported as separate rows."""
    cells = list(cells)
    return ([c for c in cells if cell_anchors(c) >= HEADLINE_MINIMUM_ANCHORS],
            [c for c in cells if cell_anchors(c) < HEADLINE_MINIMUM_ANCHORS])


def headline(dataset, record=None):
    """The headline cells of a dataset and its cells reported as separate rows, as the tracked declaration states them."""
    if dataset not in DATASETS:
        raise ValueError(f"{dataset} is not a held-out dataset")
    rule = (record or load())["rules"]["headline"]
    return rule["cells"][dataset], rule["separate_rows"][dataset]


def report_groups(rows, dataset, record=None):
    """The scored rows of a run as they are reported, the headline over its cells first and then every separate cell
    alone, each as (label, rows). A row of an undeclared cell is refused."""
    cells, separate = headline(dataset, record)
    unknown = {row["cell"] for row in rows} - set(cells) - set(separate)
    if unknown:
        raise ValueError(f"rows of undeclared cells {sorted(unknown)}")
    groups = [(f"headline, {HEADLINE_MINIMUM_ANCHORS} or more anchors", [row for row in rows if row["cell"] in cells])]
    return groups + [(name, [row for row in rows if row["cell"] == name]) for name in separate]


def admissible(distances_cm):
    """Whether each calibrated reading is a range, from zero to the range of 200 ns of flight. A reading outside is a
    failed exchange, a value no radio distance can take."""
    values = np.asarray(distances_cm, dtype=float)
    return (values >= 0) & (values <= MAXIMUM_RANGE_CM)


def admissible_samples(distances_cm, scale_cm):
    """The calibrated samples of a link. An inadmissible reading stays out."""
    values = np.asarray(distances_cm, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not (math.isfinite(scale_cm) and scale_cm > 0):
        raise ValueError("invalid calibrated readings or scale")
    return [dict(distance_cm=float(d), scale_cm=float(scale_cm)) for d in values[admissible(values)]]


def observation(agent, anchor, distances_cm, scale_cm):
    """The observation of an agent to anchor link that carries readings, with its admissible readings as samples. A
    link whose every reading is inadmissible stays present without samples, an answered exchange without a range."""
    if not len(distances_cm):
        raise ValueError("a link without readings has no observation")
    return {"from": int(agent), "to": int(anchor), "to_is_anchor": True, "tof_ns": [],
            "calibrated_samples": admissible_samples(distances_cm, scale_cm)}


def problem(agents, anchors, observations, initial):
    """A prepared anchor only problem. anchors maps an identity to (x, y, z) in centimetres."""
    return dict(mode="AnchorOnly", agent_ids=[int(a) for a in agents],
                anchors=[dict(id=int(a), position=dict(x=float(p[0]), y=float(p[1])), height_cm=float(p[2])) for a, p in sorted(anchors.items())],
                observations=sorted(observations, key=lambda o: (o["from"], o["to"])),
                initial_positions=[dict(x=float(x), y=float(y)) for x, y in initial])


def shared_area(areas):
    """The area every one of the areas holds, their intersection, refused when they do not overlap. The starts of both
    tracks of a unit are drawn inside it on every dataset, so that every method starts inside SOLID's area on both."""
    result = dict(minimum_x_cm=max(a["minimum_x_cm"] for a in areas), maximum_x_cm=min(a["maximum_x_cm"] for a in areas),
                  minimum_y_cm=max(a["minimum_y_cm"] for a in areas), maximum_y_cm=min(a["maximum_y_cm"] for a in areas))
    if not (result["minimum_x_cm"] < result["maximum_x_cm"] and result["minimum_y_cm"] < result["maximum_y_cm"]):
        raise ValueError("the areas of a unit's tracks do not overlap")
    return result


def initial_positions(area, count, seed, unit_id):
    """Starting coordinates shared by every method and both tracks, uniform in the unit's operating area."""
    low = [area["minimum_x_cm"], area["minimum_y_cm"]]
    high = [area["maximum_x_cm"], area["maximum_y_cm"]]
    return np.random.default_rng(hash_order(seed, "initial", unit_id)).uniform(low, high, (count, 2))


def units_of(links, agents, anchors, unit_id, seed, scale_cm, area_rule, first=None):
    """Both tracks of one unit. links maps (agent, anchor) to the calibrated readings of its ten-reading block. first
    maps a link to the readings of its one-reading block, the reading of the block's first attempt, by default the
    first reading of its ten-reading block, the first report of an AAU block. A link that first leaves out, a HYMN link
    whose window's first epoch holds no reading, has no observation in the one-reading track, even when a later attempt
    answered. Agents without any link stay in the unit. The area both tracks share, the intersection of their operating
    areas (shared_area), places the starting coordinates of both tracks."""
    first = {key: values[:1] for key, values in links.items()} if first is None else first
    if any(len(values) > 1 or (values and values[0] != links.get(key, [None])[0]) for key, values in first.items()):
        raise ValueError("the one-reading block of a link is the first reading of its ten-reading block")
    blocks = {samples: [observation(g, a, values[:samples], scale_cm) for (g, a), values in sorted((first if samples == 1 else links).items()) if len(values)]
              for samples in TRACKS}
    for key in links:
        if key[0] not in agents or key[1] not in anchors:
            raise ValueError("a link outside the unit")
    empty = np.zeros((len(agents), 2))
    start = initial_positions(shared_area([area_rule(problem(agents, anchors, blocks[samples], empty)) for samples in TRACKS]),
                              len(agents), seed, unit_id)
    result = []
    for samples in TRACKS:
        prepared = problem(agents, anchors, blocks[samples], start)
        result.append(dict(id=int(unit_id), samples=samples, problem=prepared, area=area_rule(prepared)))
    return result


def population(units, metadata):
    """Counts of a built half, checked for one metadata entry per identity and both tracks per identity."""
    keys = [(u["id"], u["samples"]) for u in units]
    identities = {m["id"] for m in metadata}
    if len(keys) != len(set(keys)) or set(keys) != {(i, s) for i in identities for s in TRACKS} or len(identities) != len(metadata):
        raise ValueError("a built half lacks a track or an identity")
    per_cell = {}
    for entry in metadata:
        per_cell[entry["acquisition"]] = per_cell.get(entry["acquisition"], 0) + 1
    agents = sum(len(u["problem"]["agent_ids"]) for u in units)
    pairs = sum(len(u["problem"]["agent_ids"]) * (len(u["problem"]["agent_ids"]) - 1) // 2 for u in units)
    unranged = sum(1 for u in units for g in u["problem"]["agent_ids"]
                   if not any(o["from"] == g and o["calibrated_samples"] for o in u["problem"]["observations"]))
    return dict(identities=len(identities), inputs=len(units), units_per_cell=dict(sorted(per_cell.items(), key=lambda kv: int(kv[0].split("-")[1]))),
                agent_attempts=agents, pair_attempts=pairs, agents_without_anchor_sample=unranged)


def evaluation_units(dataset, metadata, record=None):
    """How many units of the reference metadata read a point or a flight of the declared evaluation half, the HYMN points
    of a unit's agents or the AAU flight of its block."""
    split = section(dataset, record)["split"]
    if dataset == "hymn":
        points = set(split["evaluation_points"])
        return sum(1 for entry in metadata if points & set(entry["points"]))
    flights = set(split["evaluation_flights"])
    return sum(1 for entry in metadata if entry["flight"] in flights)


# ---------------------------------------------------------------------------------------------------------------------
# The declaration


def headline_rule(cells):
    """The anchor only headline over the declared cells of every dataset, cells mapping a dataset to its cell names."""
    parts = {dataset: headline_split(names) for dataset, names in cells.items()}
    return dict(
        rule=("The anchor only headline of every dataset is the balanced S50, S75 and S90 per track over the cells with three or "
              "more anchors under the joined pair set of rules.score, S75 the headline figure. The cells with one and two anchors "
              "are reported as separate rows, the percentiles of each cell alone. Every row reports the share of pairs and units "
              "it scores. The cells stay as declared."),
        minimum_anchors=HEADLINE_MINIMUM_ANCHORS,
        cells={dataset: kept for dataset, (kept, _) in parts.items()},
        separate_rows={dataset: apart for dataset, (_, apart) in parts.items()})


def common_rules(cells):
    """The rules both datasets share, with the headline over the declared cells of each (headline_rule)."""
    return dict(
        mode="ANC. Only agent to anchor links enter a unit and no agent to agent range is supplied.",
        tracks=("Every unit has two tracks under one identity. The ten-reading track holds the readings of a link in the unit's "
                "block of ten, and the one-reading track the reading of the block's first attempt, the first report of an AAU "
                "block and the first epoch of a HYMN window. A link whose first attempt holds no reading has no observation in "
                "the one-reading track. A calibrated reading outside the interval from zero to the range of 200 ns of flight, the "
                "interval of raw timings the engine and the baselines accept, is a failed exchange. It is inadmissible and stays "
                "out of the samples, so a link whose every reading is inadmissible stays present without samples."),
        admissible_interval_cm=[0.0, MAXIMUM_RANGE_CM],
        initialization=("Every method receives the same starting coordinates, uniform in the area both tracks of the unit "
                        "share, the intersection of the operating areas of its one-reading and its ten-reading track by the "
                        "deployment rules, seeded by the dataset seed and the unit identity, as every other dataset draws "
                        "them. Both tracks share them, so every method starts inside SOLID's area on both tracks."),
        area=("The operating area of every unit and track follows the deployment rules of research/solid/deployment.py, the "
              "padded box of its known anchors and measured ranges bounded by a published site extent. A unit or track of a "
              "dataset without a published extent that holds an agent without any admissible anchor sample, or no admissible "
              "anchor sample at all, takes the box of its anchors padded by the end of the admissible interval, the longest range "
              "an admissible reading can take. Such an agent carries no range that bounds its place, and the box padded by the "
              "ranges of the other agents need not hold it, so the wider box keeps the check of the operating areas in stage one "
              "from stopping at it. SOLID receives that area, and no baseline of the roster takes one."),
        truth=("Published surveyed positions in the publisher's local frame, used for scoring and for fits on the training half "
               "only. The AAU halves share their node positions, see aau.scope."),
        score=("The released pairwise score of TWR-CLOUD-Bern transferred to horizontal pair distances and capped at the diameter "
               "of the unit's true agent positions, the scenario diameter of the released scorer, over the joined pair set of "
               "research/benchmarks/datasets/scoring.py, the rule of every dataset. A unit scores only the agent pairs that its "
               "measured links join, directly or through other agents or anchors with every anchor counted as one node. In the "
               "anchor only mode a pair is scored when both agents hold at least one answered anchor link, whether or not the link holds "
               "an admissible reading. A unit without such a pair is left out of the percentiles, as the released scorer leaves "
               "out a scenario without eligible pairs. Balanced S50, S75 and S90 per track over the headline cells of three or "
               "more anchors, S75 the headline, and the same percentiles of each cell with one or two anchors alone as a separate "
               "row. Every row reports the share of pairs scored, the joined pairs over every agent pair of its units, and the "
               "share of units scored, its units holding at least one joined pair over all its units. Both shares follow from the "
               "inputs alone and are the same for every method. The ranged pair set, the pairs joined by links that hold an "
               "admissible reading, is reported beside every figure as a supplementary one with its shares, and the headline "
               "stays joined. Before any scorer reads the predictions, a coordinate that is not finite or whose magnitude exceeds "
               "1e7 cm is treated as missing, for every method alike."),
        headline=headline_rule(cells),
        roster=("Both constellations have real height spread. HYMN's anchors stand 3.1 m to 4.2 m high over points about 2 m to "
                "3.3 m high, and AAU's nodes stand from 0.3 m to 3.3 m. The repository's rule for such scenes, the rule Bregar "
                "follows (bregar_predict.rs), therefore applies. Jin 2021, FNL 2025 and STRONG 2021 run, each with the planar or the "
                "spatial model that its selection on the training half chose. ADE-BP has no spatial implementation and is not "
                "applicable."),
        methods=dict(
            solid=("The frozen method's anchor only profile, composed from the method record of the confirmatory round, "
                   "research/artifacts/solid/selections/method.json, which the method freeze pins. Its link model is fitted by the "
                   "fitter of research/solid/fit_layout.py on at most 2000 blocks per track of the training half with the options of "
                   "the method's link model variant and refitted without the link components the method removed, as the round "
                   "refits them (research/solid/fit_heldout.py links). Every fit uses the round's likelihood, and a fit that does "
                   "not converge has the outcome of rules.outcomes. Its deployment settings come from the held-out deployment "
                   "record of the training half, its anchor only budgets from "
                   "the method's budget factor and its switches from the method record, and solid-units solves it. SOLID takes two "
                   "inputs that no baseline of the roster takes. One is the operating area of rules.area, bounded by the published "
                   "site extent where one exists, AAU's 5 m by 8 m area. The other is the height prior of its anchor links from the "
                   "true heights of the unknown nodes of the training half, which for AAU are the heights of the nine nodes that both "
                   "halves score (aau.scope)."),
            **{"jin-2021": ("The declared base configuration of research/benchmarks/datasets/baseline-configurations.json with the "
                            "planar or the spatial model and the median or the mean of a link's readings selected on the training "
                            "half by the lowest mean over the two tracks of the balanced S75 over the headline cells of three or more "
                            "anchors under the joined pair set, the figure it is reported on, ties by declared order among the "
                            "means within 0.01 cm of the lowest. Its semidefinite relaxation takes no starting coordinates."),
               "fnl": ("The declared base configuration with the planar or the spatial model, the median of a link's readings or "
                       "every reading as its own term and its start selected on the training half by the lowest mean over the two "
                       "tracks of the balanced S75 over the headline cells of three or more anchors under the joined pair set, the "
                       "figure it is reported on, ties by declared order among the means within 0.01 cm of the lowest. It starts "
                       "every agent at the unit's starting coordinates or at the engine's geometric layout of the unit, the start "
                       "its selection chose, and, with the spatial model, at the median height of the unit's known anchors."),
               "ade-bp-2015": "Not run. It has no spatial implementation, and both constellations have real height spread.",
               "strong": ("The declared base configuration with its Huber radius among 10, 20, 40, 80 and 160 cm and no robust "
                          "loss, its planar or spatial model and its start selected jointly on the training half by the lowest "
                          "mean over the two tracks of the balanced S75 over the headline cells of three or more anchors under the "
                          "joined pair set, the figure it is reported on, ties by declared order among the means within 0.01 cm of "
                          "the lowest. It starts every agent at the unit's starting coordinates or at the engine's geometric layout "
                          "of the unit, the start its selection chose, and, with the spatial model, at the median height of the "
                          "unit's known anchors.")}),
        execution=("Baselines run in single-threaded heldout-adapter processes with every solve that crashes an adapter recorded "
                   "as that method's failure on that unit. SOLID runs in solid-units."),
        freeze=("The method freeze record research/artifacts/solid/selections/method-freeze.json, which research/solid/method_freeze.py "
                "writes once the confirmatory round has decided the method, must be committed, and so must the method record and the "
                "budget check record it pins. It holds the SHA-256 of the method record and of the round's freeze, the identity of "
                "the sources, every tracked file of the runtime and the research tree without the records under research/artifacts, "
                "the papers and the Markdown documents, the identity and SHA-256 of the solid-units and heldout-adapter executables "
                "and of the executables of the evaluation runs of the other datasets, the TWR-CLOUD-Bern CLI and the adapters of "
                f"TIERS, Bregar and IDLab, the Python, numpy and scipy versions, the SHA-256 of the budget check record and, under "
                f"{FREEZE_KEY}, the SHA-256 of this declaration and of the declared analysis of the paper's final tables, "
                "research/artifacts/solid/splits/final-analysis.json. The sources and the packages must equal those of the round's "
                "freeze when it is written, "
                "solid-units must be byte for byte the executable the round froze, and the likelihood of the fitter must be the one "
                "the round froze. Every builder, fitter, selection and runner of either half refuses to run until then and while "
                "any of them differs. No range statistic, calibration, fit, selection or score of either half is stored or "
                "displayed before the freeze beyond the statistics that the disclosures and revisions list. These are the reading "
                "counts per point and anchor of all 48 HYMN points and the surveyed coordinates of all its points, which designed "
                "the split, the failed exchange counts of the training halves and the sign check on seq1 in this declaration, the "
                "failed readings of up to 306 m in HYMN and near 338 km in AAU, the largest median changes of 7.5 cm and 10.3 cm "
                "of the static ranges of the spiral_1to2_2m flights against seq1, the packet scales of 220.65 cm for HYMN and "
                "3.68 cm for AAU with the held-out deployment settings that follow from them, the smoke solves of eight training "
                "inputs of each dataset, whose outputs were checked for failures and finite estimates and whose baseline rows "
                "passed through the scorer without an error value printed or stored, and the inspection of the raw ZigPos logs of "
                "HYMN, the schema and the times of the first and last record of all fifty logs and the counts of the messages and "
                "of their statuses in the logs of the 24 training points. The first rows of the HYMN readings, of the evaluation "
                "point A13B6, were displayed before the split existed."),
        ledger=("The ledger research/artifacts/solid/splits/heldout-evaluations.json records every step that reads a held-out half "
                "after the method freeze, and the node stages commit every entry on a branch. An entry and its commit are one step "
                "of a stage, and a commit that fails leaves no entry behind. The stages refuse to start without a git identity or "
                "on a detached HEAD. Right after each commit of the ledger the stage tags it with the local annotated tag "
                "heldout/<step>/<event> and appends its SHA with the UTC time to logs/heldout-ledger-commits.log in the node's work "
                "folder, outside the repository, so that no reset, rebase or merge drops a commit unseen. Both stages and the stage "
                "of the method freeze refuse to start while that log names a commit the checkout lacks, as a fresh clone would. "
                "Nothing is pushed. Stage "
                "one opens its entry before it reads any training half and closes it with the SHA-256 of its records and of the "
                "executables. Stage two opens a dataset before the first read of its evaluation half and closes it with the "
                "SHA-256 of the population, request and reference of both halves, of the run and score files of every method on "
                "both and of the executables. Every run of SOLID and the baselines on either half needs the committed opening of "
                "its dataset and the closed stage one, and a training run takes the stage one profile into its own folder like an "
                "evaluation run. A dataset that the ledger closes in any version that a commit of the repository holds, on any "
                "branch or tag and on either side of a merge (git rev-list --full-history --all), is not evaluated again, and an "
                "entry under another method freeze in any such version refuses that freeze, so the method is not frozen again "
                "once a held-out half has been read. git log --full-history --all -p on the ledger shows every entry with the "
                "commit that added it, and the tags and the log name every such commit."),
        stages=dict(
            one=("After the method freeze only, with the freeze verified, the likelihood of the fitter among it. The ledger opens "
                 "the stage in a commit before any training half is read. Check this declaration, build the training inputs and the "
                 "calibration, fit SOLID's link model and the deployment settings, write the composed profiles and select the "
                 "options of Jin 2021, FNL 2025 and STRONG 2021. The loaders parse the whole archive of each dataset, both halves, "
                 "and check its structure, and from the evaluation half they compute only the statistics this declaration states, "
                 "the reading counts of its points and the report counts of its flights. No reading of an evaluation half enters a "
                 "calibration, fit, selection or unit of stage one. The ledger closes the stage with the SHA-256 of every record in "
                 "the commit of the records."),
            two=("After stage one, with the freeze verified again and every record of stage one committed as the ledger's closing "
                 "of stage one names it. Rebuild the training halves, rebuild SOLID's link fits, the held-out deployment record and "
                 "the composed profiles and compare each with its record, and verify the selection records of the baselines "
                 "against the declaration, the training half, the freeze and the adapter they name, without selecting again. For "
                 "each dataset the ledger opens it in a commit before its evaluation half is built, every method runs on both "
                 "halves, and the ledger closes it with the SHA-256 of its files in a further commit. Each dataset is evaluated on "
                 "its own, so a stop of one dataset's evaluation leaves the other's to go on. A dataset the ledger closes is not "
                 "evaluated again.")),
        outcomes=dict(
            rule=("The method freeze pins the sources, so a check that stopped a held-out step after the freeze could never be "
                  "fixed, and every such check has the outcome declared here. A data defect leaves out the affected unit, pair or "
                  "flight for every method alike and is reported with its count and reason, as an AAU flight beyond the relocation "
                  "guard is. A method defect counts against SOLID, and the tables mark it beside SOLID's rows. It is never grounds "
                  "for declaring a dataset not evaluable, and nothing is fitted or run again with other settings. A build defect, "
                  "which no reading can cause, stays a stop and is checked before the ledger opens the step it would stop. Stage "
                  "one closes both datasets with one entry, so no stop of it may depend on a reading, and stage two evaluates each "
                  "dataset on its own, so a stop of one dataset's evaluation leaves the other's to go on."),
            relocation=("An AAU flight whose static ranges moved from seq1 beyond the relocation guard, or that has no pair the guard "
                        "can compare, is a data defect, as aau.split.relocation_guard declares. It is left out of its half for every "
                        "method alike, whether it trains or evaluates, and a training flight out of the packet scale, SOLID's link fit, "
                        "the deployment settings and every selection as well. It never stops a stage. The calibration record, the "
                        "population records of both halves and every run record name each such flight with its change and its units, "
                        "and every table of AAU notes it for every method."),
            training_halves=("No reading can empty a training half. The declared counts leave 31,087 admissible readings of the "
                             "HYMN training points, and at least 4,118 admissible reports in seq1, the reference of the relocation "
                             "guard, which the guard therefore always keeps, since seq1 holds 13 blocks of all 72 ordered pairs and all "
                             "training flights together hold 5,242 failed exchanges. So every training half keeps readings for its "
                             "packet scale, SOLID's link fit, its deployment settings and every selection, and none of them stops for "
                             "want of a reading."),
            link_fit=("A link fit of SOLID on a training half, the full fit or the refit without the removed components, that ends "
                      "without converging is the method's link model as it stands. It is "
                      "never fitted again with other settings, "
                      "and every table of that dataset marks SOLID with it. A fit with a value that is not finite makes every solve of "
                      "SOLID with it fail, and every failure is scored as a failure with the failure penalty. A full fit with such a "
                      "value is not refitted, so the method's link model keeps no value."),
            likelihood=("A link fit from another likelihood than the one the round froze is a build defect and stops stage one. The "
                        "check of the freeze compares the likelihood of the fitter with the round's before the ledger opens stage "
                        "one, so no reading reaches a fit of another likelihood."),
            refit=("A refit that is not the full model minus the removed components is a build defect and stops stage one. Its "
                   "settings follow from the method's fit options and removed components alone, never from a reading, and the "
                   "round's own refits of the same components under the same sources passed the same check before the method record "
                   "was written."),
            nested_check=("The nested check of the refit, as the round runs it, is recorded with the refit, and a failure stops "
                          "nothing, since the refit is the method's link model. Every table of that dataset marks SOLID with it."),
            containment=("A true agent position of a training half outside its unit's operating area is counted in the held-out "
                         "deployment record and stops nothing. The area is SOLID's input alone, so such a position counts against "
                         "SOLID, and every table of that dataset marks SOLID with the count."),
            aau_pairs=("An ordered static pair of AAU without an admissible report in a flight, or in seq1, has no range for the "
                       "relocation guard to compare, a data defect. The guard compares that flight on its other pairs, and a flight "
                       "without any pair to compare counts as beyond the guard. The calibration record, the population record of the "
                       "half and every run record of it name each such pair with its flight, and its links stay present without "
                       "samples for every method alike, as rules.tracks declares for every link without an admissible reading. A pair "
                       "without an admissible report in any training flight stays out of the packet scale, and the calibration "
                       "record names it.")))


def build(sections=None):
    """The declaration. sections holds the sections of both datasets, which the archives give (hymn_layout.declaration
    and aau_layout.declaration), and they are read from the archives unless given. Every other part the code alone
    builds."""
    import aau_layout
    import hymn_layout
    sections = sections or dict(hymn=hymn_layout.declaration(), aau=aau_layout.declaration())
    return dict(
        schema=SCHEMA,
        role=("HYMN and AAU Meshed UWB are held-out benchmarks. No design decision of the method reads them. Both halves are "
              "built, fitted and solved only after the method freeze recorded in research/artifacts/solid/selections/method-freeze.json, "
              "and the evaluation halves are evaluated once."),
        freeze_record=relative(FREEZE),
        revisions=[("The first version took the sign of the AAU statistics file for the pair biases. Its declared check on seq1, a "
                    "training flight, stopped the first build, and the README's sign replaced it (aau.calibration.revision)."),
                   ("The first scratch builds of the training halves, before any link, ADE-BP or STRONG fit and before any solve, "
                    "found failed exchanges that the radios publish as ranges, AAU reports near 338 km in either sign and HYMN "
                    "readings of up to 306 m. Every calibrated reading of both datasets now has to lie from zero to the range of "
                    "200 ns of flight, the interval of raw timings the engine and the baselines accept, to be admissible. Each "
                    "dataset's calibration counts the failed exchanges of its training half."),
                   ("An independent review before any fit or solve on the node led to the third version. It scores a unit against "
                    "the diameter of its own agents as the released scorer does, fits ADE-BP on the training blocks SOLID's link "
                    "fit reads, declares the area of a unit without any admissible anchor sample, compares the AAU flights by "
                    "their admissible calibrated ranges, states that the AAU halves share their node positions, requires a "
                    "committed freeze record and has stage two check every record of stage one first. The plans and halves are "
                    "unchanged."),
                   ("A further review before any fit or solve on the node applied the repository's rule for scenes with real "
                    "height spread, the rule Bregar follows, to both datasets (rules.roster). Jin 2021 now infers "
                    "with the spatial model like FNL, STRONG keeps its planar or spatial selection on the training half, and ADE-BP, "
                    "which has no spatial implementation, is not run. The same review moved stage one after the method freeze, so "
                    "that no calibration, fit, selection or score of either half exists before it, and has the freeze record pin "
                    "this declaration and the held-out sources (rules.freeze). The plans and halves are unchanged."),
                   ("Before any fit or solve, the anchor only headline of every dataset became the balanced percentiles over the "
                    "cells with three or more anchors, and the cells with one and two anchors became separate rows "
                    "(rules.headline). The ranges of an agent to one or two anchors cannot fix its position, which stays "
                    "ambiguous along a circle for one anchor and between mirror candidates for two, so those cells are "
                    "underdetermined by their ranges and are reported apart from the headline. STRONG's selection on the "
                    "training half ranks its candidates over the headline cells, so that it is tuned for the figure it is "
                    "reported on. The cells, plans and halves are unchanged."),
                   ("Before any fit or solve, the score rule of every dataset was revised to the joined pair set, the agent pairs "
                    "that a unit's measured links join with every anchor counted as one node, and every figure reports the share of "
                    "pairs and units it scores, with the ranged pair set beside it as a supplementary figure (rules.score). STRONG's "
                    "selection ranks its candidates under the joined pair set, and the freeze pins the scoring code that defines it. "
                    "The cells, plans and halves are unchanged."),
                   ("Before any fit or solve, a review revised the freeze and the runs. The method freeze now holds the method record "
                    "of the confirmatory round, the identity of every source and of both executables, the Python environment and the "
                    "budget check record, and SOLID's link fit, budgets and switches follow that record (rules.freeze, rules.methods). "
                    "A tracked ledger records every step that reads a half, so a dataset is evaluated once, the method is not frozen "
                    "again after any read, and every run needs the closed stage one (rules.ledger, rules.stages). A HYMN unit with an "
                    "agent without an admissible anchor sample takes the anchor box padded by the end of the admissible interval "
                    "(rules.area), and an AAU evaluation flight beyond the relocation guard is excluded and reported "
                    "(aau.split.relocation_guard). The freeze rule names the statistics the disclosures list, and rules.methods "
                    "states the inputs that only SOLID receives and how every baseline starts. The cells and plans are unchanged."),
                   ("Before anything was built, HYMN's one-reading track was changed to the reading of the first epoch of each window "
                    "alone, epoch 10w, which a link carries only when that exchange answered. Before, it took the first reading anywhere "
                    "in the window, so a link counted as answered when any of its ten epochs answered and the one-reading track carried "
                    "the presence of ten attempts. A blank reading is a failed exchange of a polled anchor, as the inspection of the "
                    "raw logs in the disclosures shows (hymn.blank, hymn.units.rule, rules.tracks). The ten-reading track, the cells "
                    "and the plans are unchanged."),
                   ("Before any fit or selection, the baselines took the rule of every other dataset for the options that "
                    "TWR-CLOUD-Bern's scans once chose. Jin 2021, FNL 2025 and STRONG 2021 now start from the declared base "
                    "configuration of research/benchmarks/datasets/baseline-configurations.json, and each selects on the training "
                    "half its model and the aggregation of a link's readings, FNL and STRONG their start as well and STRONG its "
                    "Huber radius, by the rule STRONG followed over the headline cells under the joined pair set, with mean S75 "
                    "values within 0.01 cm of the lowest counted as tied and the tie going to the declared order, so that solver "
                    "noise cannot pick a candidate. A run applies the declared settings of the candidate its record names. Before, "
                    "Jin and FNL kept the spatial model and the retained Bern settings, and STRONG's start stayed the geometric "
                    "layout (rules.roster, rules.methods, rules.stages). The cells, plans and halves are unchanged."),
                   ("Before any fit or solve, a review declared the outcome of every check that could stop a held-out step after the "
                    "freeze, when it could never be fixed (rules.outcomes). A link fit that does not converge is used as it stands and "
                    "marked in the tables, a fit with a value that is not finite fails every solve of SOLID, a failed nested check of "
                    "the refit and true positions outside their operating area are marked and stop nothing, an AAU pair without an "
                    "admissible report is left out of the relocation guard and reported, and a fit of another likelihood or a refit "
                    "of other settings stays a stop that is checked before any step opens. The same review revised the ledger. Its "
                    "history covers every branch, tag and side of a merge, the stages commit it on a branch only and tag and log each "
                    "commit outside the repository at once, and a failed commit leaves no entry (rules.ledger). The freeze binds "
                    "solid-units to the round's build and needs the method record and the budget check record committed "
                    "(rules.freeze). The freeze rule names every statistic the disclosures list, stage one states what its loaders "
                    "parse, stage two states which records it rebuilds and which it verifies, and the role says that no design "
                    "decision of the method reads the datasets (role, rules.stages). An AAU training flight beyond the relocation "
                    "guard, which stopped the build for review before, is left out of the training half for every method alike and "
                    "reported, as an evaluation flight is, since nothing may be reviewed and changed after the freeze and stage one "
                    "closes both datasets together, no reading can empty a training half by the declared counts, and stage two "
                    "evaluates each dataset on its own, so that one dataset never blocks the other (aau.split.relocation_guard, "
                    "rules.outcomes). The cells, plans and halves are unchanged."),
                   ("Before any fit or solve, the starting coordinates of every held-out unit moved inside the area both tracks of "
                    "the unit share, the intersection of the operating areas of its one-reading and its ten-reading track, as every "
                    "other dataset draws them, so that every method starts inside SOLID's area on both tracks (rules.initialization). "
                    "SOLID's entry now points a fit that does not converge to its declared outcome (rules.methods). The cells, plans "
                    "and halves are unchanged."),
                   ("A further review before any fit or solve extended the method freeze. It now pins the declared analysis of the "
                    "paper's final tables beside this declaration, and the executables of the evaluation runs of the other datasets, "
                    "the TWR-CLOUD-Bern CLI and the adapters of TIERS, Bregar and IDLab, beside solid-units and heldout-adapter "
                    "(rules.freeze). Both stages and the stage of the method freeze refuse a checkout that lacks a commit the log of "
                    "the ledger names (rules.ledger). The outcome of a link fit that does not converge no longer refers back to "
                    "rules.methods.solid, which refers to it (rules.outcomes.link_fit). The cells, plans and halves are unchanged."),
                   ("Before any fit or solve, the score rule of every dataset declared the coordinate rule that every scorer applies. "
                    "Before any scorer reads the predictions, a coordinate that is not finite or whose magnitude exceeds 1e7 cm, the "
                    "engine's bound on a sane coordinate, is treated as missing, for every method alike, so a finite but enormous "
                    "coordinate is charged and counted as a missing one and cannot stop a scoring (rules.score). The cells, plans and "
                    "halves are unchanged.")],
        disclosures=[("Before the declaration a structural scan counted the published readings per point and anchor of all 48 HYMN "
                      "points, the surveyed coordinates of all points were displayed to design the split, and the median raw ranges "
                      "of the 72 static pairs of both spiral_1to2_2m flights were compared with those of seq1, largest changes 7.5 cm "
                      "and 10.3 cm, to confirm that the published constellation still held. None of these compared a reading with its "
                      "truth or ran a method."),
                     ("While inspecting the file format before any split existed, the first five rows of the HYMN UWB file, "
                      "readings of point A13B6, were displayed. A13B6 fell into the evaluation half."),
                     ("Tests committed in a82b26d compared the raw and the bias-corrected median residual of the pair 101 to 102 on "
                      "spiral_1to2_2m seq1, an evaluation flight, and pinned readings and the surveyed position of A13B6. They ran "
                      "locally three times, after both earlier revisions, and printed no value. They now read training data, and no "
                      "rule changed because of them."),
                     ("Before the fourth revision, which moved stage one after the freeze, scratch builds of both training halves "
                      "computed their packet scales, 220.65 cm for HYMN and 3.68 cm for AAU, and the held-out deployment settings "
                      "that follow from them. Smoke solves then ran the four baselines of the earlier roster and SOLID, with a "
                      "placeholder link model, on eight training inputs of each dataset. Their outputs were checked for failures and "
                      "finite estimates only, and the baseline rows passed through the scorer to check their structure with no error "
                      "value printed or stored. No link, ADE-BP or STRONG fit ran, and no evaluation input was built."),
                     ("To settle what a blank HYMN reading means, the raw ZigPos logs data/raw/uwb/log_zigpos_twr_*.sqlite of the "
                      "archive were inspected before anything was built. For all fifty logs the schema was read, a LOG table of record "
                      "time, topic and message and a RUN table of start and end times, and the times of the first and last record, "
                      "which match every log to the published time window of one point or of none. Only in the logs of the 24 "
                      "training points were the messages decoded in memory and counted, 7,425 messages of the one topic "
                      "RANGING_DISTANCES from two initiating addresses on beacon slots 71 and 75, each listing all ten anchors once "
                      "with a status, status 0 in all 74,250 entries. No range, accuracy, signal strength or position was stored or "
                      "displayed, no record time beyond the first and last of each log was displayed, and no message of a log of an "
                      "evaluation point was read.")],
        rules=common_rules({"hymn": [cell(k) for k in hymn_layout.CELLS], "aau": [cell(k) for k in aau_layout.CELLS]}),
        hymn=sections["hymn"],
        aau=sections["aau"],
        code={name: sha256(ROOT / name) for name in CODE})


def revise(path=None):
    """The tracked declaration with every part that the code alone builds rebuilt, its role, revisions, disclosures,
    rules and code hashes, and both dataset sections kept as they are but for the rules that a layout module declares as
    text alone (SECTION_TEXT, by their path in the section). It reads no archive. The sections hold what the archives
    give, and heldout.py --check rebuilds every part from the archives at held-out stage one."""
    import aau_layout
    import hymn_layout
    tracked = load(path)
    sections = {dataset: tracked[dataset] for dataset in DATASETS}
    for dataset, module in (("hymn", hymn_layout), ("aau", aau_layout)):
        for keys, text in getattr(module, "SECTION_TEXT", {}).items():
            holder = sections[dataset]
            for key in keys[:-1]:
                holder = holder[key]
            holder[keys[-1]] = text
    return build(sections)


def load(path=None):
    record = read_json(path or RECORD)
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{relative(path or RECORD)} is not a held-out declaration")
    return record


def differences(tracked, current):
    """The top level fields where a rebuilt declaration differs from the tracked one. The code hashes are informative
    and not compared. The builders compare every built half with its plan digest, and stage two rebuilds the training
    halves under the frozen code, so a code change that alters a built unit stops there."""
    keys = (set(tracked) | set(current)) - {"code"}
    return sorted(key for key in keys if tracked.get(key) != current.get(key))


def check(path=None):
    changed = differences(load(path), build())
    if changed:
        raise ValueError(f"The tracked held-out declaration differs from its rebuild in {changed}")
    return True


def section(dataset, record=None):
    """One dataset's section of the tracked declaration."""
    if dataset not in DATASETS:
        raise ValueError(f"{dataset} is not a held-out dataset")
    return (record or load())[dataset]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="rebuild the declaration and compare it with the tracked record")
    group.add_argument("--revise", action="store_true", help="rewrite the parts of the tracked declaration that the code alone builds")
    group.add_argument("--pins", action="store_true", help=f"print the pins the method freeze record holds under {FREEZE_KEY}")
    group.add_argument("--check-freeze", action="store_true", help="verify the committed method freeze record")
    group.add_argument("--check-stage-one", action="store_true", help="verify the records of held-out stage one against the ledger")
    group.add_argument("--open", choices=STEPS, help="append the opening entry of a step to the ledger")
    group.add_argument("--complete", choices=STEPS, help="append the closing entry of a step to the ledger")
    group.add_argument("--evaluated", choices=DATASETS, help="exit with status 0 when the ledger closes the dataset under this freeze")
    args = parser.parse_args()
    if args.pins:
        print(json.dumps({FREEZE_KEY: freeze_pins()}, indent=1), flush=True)
        return
    if args.check_freeze:
        freeze = frozen()
        print(f"method freeze verified, {freeze['freeze_record']} holds the method, the code, the executables, the environment and the "
              "declaration as they are", flush=True)
        return
    if args.check_stage_one:
        entry = require_stage_one()
        print(f"held-out stage one verified, its {len(entry['records'])} records as the ledger closed them at {entry['time']}", flush=True)
        return
    if args.open:
        entry = open_step(args.open)
        print(f"{args.open} opened at {entry['time']} under the method freeze, commit {relative(LEDGER)} before the step reads any input",
              flush=True)
        return
    if args.complete:
        entry = complete_step(args.complete)
        print(f"{args.complete} closed at {entry['time']}, commit {relative(LEDGER)}" + (" with the records of stage one" if args.complete == STAGE_ONE else ""),
              flush=True)
        return
    if args.evaluated:
        frozen()
        return 0 if closed(args.evaluated) else 1
    if args.check:
        check()
        print("held-out declaration verified", flush=True)
        return
    if args.revise:
        revised = revise()
        changed = differences(load(), revised)
        write_json(RECORD, revised)
        print(f"revised {relative(RECORD)}" + (f" in {changed}" if changed else ", no field but the code hashes changed") + ". Review and commit it",
              flush=True)
        return
    if RECORD.exists():
        raise SystemExit(f"{relative(RECORD)} exists. A declared held-out benchmark is changed only by a reviewed commit")
    record = build()
    write_json(RECORD, record)
    for dataset in DATASETS:
        plan = record[dataset]["plan"]
        print(dataset, json.dumps({split: dict(units=plan[split]["identities"], cells=plan[split]["identities_per_cell"]) for split in SPLITS}), flush=True)
    print(f"written {relative(RECORD)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
