#!/usr/bin/env python3
"""Held-scan scenario corpora of the eleven TWR-CLOUD-Bern training scans, built by the evaluation release's rules.

Every scenario is the one the release's generator writes for that training scan with the same number, read from the
corrected responder table (release_rule.py states the rules). The corpora keep a subset of each scan's 148,260
numbers. Per agent count they keep the middle group of each of k equal rank strata of the release's groups (sorted by
dispersion), paired with every anchor count of the corpus.

cooperative   One archive with members scan-XX/N.json for the cooperative held-scan rounds and the mixed block fits,
              8 agent groups per agent count and all six anchor counts, 288 scenarios per scan, N counting over the
              scans. The manifest sits next to the archive because the released scorer reads every JSON member.
anchor-only   One directory per held scan with scenarios.zip (raw timings), calibrated.zip (the fold's calibration
              applied exactly as select_bern.prepare_scan applies it) and cohort.json, 128 agent groups per agent
              count and the five anchored counts, 3,300 scenarios per scan, members scan * 100000 + n. With
              --campaign-profiles the fold profiles of an earlier campaign (scan-XX/excess_fitted/profile.json, which
              depend on the fold fits only) are copied, so the directory can serve as SOLID_ANCHOR_ONLY_CAMPAIGN. A
              reader's data (SOLID_READER=1) without the fold calibrations, records of the held-scan rounds, gives a
              scan without one its scenarios.zip alone, all that the evaluation path reads (deployment.bern_entry).
parity        Regenerates evaluation scenarios from the scan 12 rows and compares them byte for byte with the
              released archive, a sample by default, all 148,260 with --all.
statistics    Anchor levels, agent counts, anchor enclosure and ten packet track composition of the evaluation corpus
              and of both held-scan designs, from the groups and the table's packet counts.

Scenario content depends only on the scan and the scenario number, never on the worker count or the chunking. Each
worker process runs one thread.

Usage: generate_heldscan_corpora.py {cooperative,anchor-only,parity,statistics} [--workers N] [options]
"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import deque  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import release_rule as rule  # noqa: E402

# the data folder, research/data unless SOLID_DATA names another
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
TABLE = DATA / "inputs/65c077bb1aa9b879/Rangings at Responders.pkl"
EVALUATION = DATA / "inputs/8617b54db218e099/TWR-CLOUD Bern Eval Scenarios.zip"
EVALUATION_PREFIX = "TWR-CLOUD Bern Eval Scenarios/"
OUT = ROOT / "out/solid/unify"
COOPERATIVE = OUT / "bern-training-scenarios-v2.zip"
ANCHOR_ONLY = OUT / "bern-anchor-only-scenarios-v2"
PARITY = OUT / "bern-release-parity-v2.json"
STATISTICS = OUT / "bern-heldscan-statistics-v2.json"
SCHEMA = "solid.training.scenarios.v2"
COOPERATIVE_GROUPS = 8
ANCHOR_ONLY_GROUPS = 128
ANCHOR_ONLY_COUNTS = (4, 8, 12, 16, 20)
ANCHOR_ONLY_STRIDE = 100_000
CHUNK = 32
RULE = ("The TWR-CLOUD-Bern evaluation release's generation rules applied to each scan of the corrected responder table. "
        "Anchor superset of 20 by the release's max-min criterion with its tie-break, solved exactly over the scan's 40 "
        "positions. Anchor groups of 16, 12, 8 and 4 chosen independently from the superset by the same criterion. Agents "
        "are the other 20 positions, agent groups per count sorted by mean pairwise distance and thinned by rank to 5000. "
        "Release numbering and seeds, two iterations, one and up to ten packets per link without replacement, links with "
        "one to nine packets kept whole in the ten packet track. The corpus keeps, per agent count, the middle group of "
        "each of groups_per_cell equal rank strata of the release's groups, with every listed anchor count.")

_WORKER = {}


# ---- workers -------------------------------------------------------------------------------------------------------


def _init_worker(scans, evaluation):
    _WORKER.update(scans=scans, evaluation=evaluation, archive=None)


def _design(scan):
    return scan, rule.design(_WORKER["scans"][scan][0])


def _calibrated(data, coefficients, calibrate):
    """select_bern.prepare_scan's transform of one scenario."""
    for iteration in data["Iterations"]:
        for field in ("Ranging samples between Agents", "Ranging samples to Anchors"):
            for values in iteration[field].values():
                for samples, value in values.items():
                    packets = calibrate(value, coefficients)
                    values[samples] = dict(calibrated_samples=packets) if packets else None
    return data


def _generate(chunk):
    """Compact member bytes of every entry, and the calibrated bytes when the entry carries coefficients."""
    results = []
    calibrate = None
    for entry in chunk:
        layout, links = _WORKER["scans"][entry["scan"]]
        data = rule.scenario(entry["number"], entry["agents"], entry["anchors"], layout.coordinates, links)
        data = {"Research Metadata": entry["metadata"], **data}
        raw = json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")
        calibrated = None
        if entry.get("coefficients") is not None:
            if calibrate is None:
                calibrate = _fold_calibration_module().calibrated_packets
            calibrated = json.dumps(_calibrated(data, entry["coefficients"], calibrate), separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")
        results.append((entry["name"], raw, calibrated))
    return results


def _compare(numbers):
    """Regenerate evaluation scenarios and compare them with the released bytes."""
    layout, links = _WORKER["scans"][rule.EVALUATION_SCAN]
    design = _WORKER["design"]
    if _WORKER["archive"] is None:
        _WORKER["archive"] = zipfile.ZipFile(_WORKER["evaluation"])
    mismatches = []
    for number in numbers:
        _, _, _, agents, anchors = design.combination(number)
        mine = rule.release_bytes(rule.scenario(number, agents, anchors, layout.coordinates, links))
        try:
            released = _WORKER["archive"].read(f"{EVALUATION_PREFIX}{number}.json")
        except KeyError:
            released = None
        if mine != released:
            mismatches.append(number)
    return len(numbers), mismatches


def _init_parity(scans, evaluation, design):
    _init_worker(scans, evaluation)
    _WORKER["design"] = design


def _fold_calibration_module():
    solid = str(ROOT / "research/solid")
    if solid not in sys.path:
        sys.path.insert(0, solid)
    import fit_bern

    return fit_bern


# ---- shared pieces -------------------------------------------------------------------------------------------------


class _Done:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Serial:
    """The executor interface in this process, for one worker and for tests."""

    def __init__(self, initializer, initargs):
        initializer(*initargs)
        self._max_workers = 1

    def map(self, function, iterable):
        return map(function, iterable)

    def submit(self, function, *args):
        return _Done(function(*args))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def pool(workers, initializer, initargs):
    if workers <= 1:
        return _Serial(initializer, initargs)
    return ProcessPoolExecutor(workers, initializer=initializer, initargs=initargs)


def ordered(executor, function, items, window_per_worker=3):
    """Results in submission order with a bounded number of tasks in flight, so the writer never holds the corpus."""
    pending = deque()
    window = max(1, window_per_worker * executor._max_workers)
    for item in items:
        pending.append(executor.submit(function, item))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def zip_entry(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def sha256(path):
    return rule.file_digest(path, "sha256")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    import pandas

    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pandas.__version__)


def sources():
    return {path.name: sha256(path) for path in (HERE / "release_rule.py", Path(__file__).resolve())}


def load_scans(table, scans):
    frame = rule.load_table(table)
    return {scan: (rule.scan_layout(frame, scan), rule.scan_links(frame, scan)) for scan in scans}


def designs(executor, scans):
    return dict(executor.map(_design, scans))


def design_record(design):
    return dict(devices=list(design.layout.devices),
                anchor_groups={str(count): list(group) for count, group in design.anchors.items()},
                superset_min_distance_m=design.superset_min_distance(),
                agent_groups_per_count={str(count): len(groups) for count, groups in design.agents.items()},
                scenarios_in_full_corpus=design.scenario_count())


def entries_for(design, groups_per_cell, anchor_counts):
    """(number, agent count, anchor count, index among the kept groups, release rank, agents, anchors) in order."""
    kept = {count: design.kept_ranks(count, groups_per_cell) for count in design.agents}
    for number in design.numbers(groups_per_cell, anchor_counts):
        agent_count, anchor_count, rank, agents, anchors = design.combination(number)
        yield number, agent_count, anchor_count, kept[agent_count].index(rank), rank, agents, anchors


def metadata(scan, member, number, group_index, rank, agent_count, anchor_count):
    return {"schema_version": SCHEMA, "data_role": "training", "scan_id": scan, "scenario_id": member,
            "group_index": group_index, "sample_counts": list(rule.SAMPLE_COUNTS), "seed": number,
            "release_scenario": number, "agent_group_rank": rank, "agent_count": agent_count, "anchor_count": anchor_count}


def chunks(entries, size=None):
    size = size or CHUNK
    batch = []
    for entry in entries:
        batch.append(entry)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---- products ------------------------------------------------------------------------------------------------------


def cooperative(args):
    started = time.time()
    output = Path(args.output)
    scans = tuple(args.scans)
    data = load_scans(args.table, scans)
    with pool(args.workers, _init_worker, (data, None)) as executor:
        plans = designs(executor, scans)
        entries, member, record = [], 0, {}
        for scan in scans:
            cells = {}
            first = member + 1
            for number, agent_count, anchor_count, index, rank, agents, anchors in entries_for(plans[scan], args.groups_per_cell, rule.ANCHOR_COUNTS):
                member += 1
                entries.append(dict(scan=scan, number=number, agents=agents, anchors=anchors, name=f"scan-{scan:02d}/{member}.json",
                                    metadata=metadata(scan, member, number, index, rank, agent_count, anchor_count)))
                cells[f"{anchor_count},{agent_count}"] = cells.get(f"{anchor_count},{agent_count}", 0) + 1
            record[str(scan)] = dict(design_record(plans[scan]), members=[first, member], scenarios=member - first + 1, cells=cells)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        content = hashlib.sha256()
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for results in ordered(executor, _generate, chunks(entries)):
                for name, raw, _ in results:
                    archive.writestr(zip_entry(name), raw)
                    content.update(name.encode())
                    content.update(raw)
        temporary.replace(output)
    manifest = dict(schema_version=SCHEMA, product="cooperative held-scan archive", rule=RULE, table=str(Path(args.table).resolve()),
                    table_md5=rule.CORRECTED_TABLE_MD5, groups_per_cell=args.groups_per_cell, anchor_counts=list(rule.ANCHOR_COUNTS),
                    agent_counts=list(rule.AGENT_COUNTS), iterations=[1, 2], sample_counts=list(rule.SAMPLE_COUNTS), scans=record,
                    scenario_count=member, content_sha256=content.hexdigest(), archive=str(output.resolve()),
                    archive_sha256=sha256(output), sources=sources(), environment=environment(), workers=args.workers,
                    seconds=round(time.time() - started, 1))
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("scenario_count", "content_sha256", "archive_sha256", "seconds")}, indent=1))


def anchor_only(args):
    started = time.time()
    directory = Path(args.output_directory)
    scans = tuple(args.scans)
    fit_bern = _fold_calibration_module()
    # the fold calibration of every scan, none for a scan of a reader's data that holds none, which gets no
    # calibrated.zip
    reader = os.environ.get("SOLID_READER") == "1"
    folds = {scan: None if reader and not (fit_bern.SCANS / f"scan-{scan:02d}/fit.json").exists() else fit_bern.calibration(scan) for scan in scans}
    data = load_scans(args.table, scans)
    with pool(args.workers, _init_worker, (data, None)) as executor:
        plans = designs(executor, scans)
        entries, cohorts = [], {}
        for scan in scans:
            coefficients = folds[scan][0] if folds[scan] else None
            cells, count = {}, 0
            for number, agent_count, anchor_count, index, rank, agents, anchors in entries_for(plans[scan], args.groups_per_cell, ANCHOR_ONLY_COUNTS):
                count += 1
                member = scan * ANCHOR_ONLY_STRIDE + count
                entries.append(dict(scan=scan, number=number, agents=agents, anchors=anchors, name=f"{member}.json", coefficients=coefficients,
                                    metadata=metadata(scan, member, number, index, rank, agent_count, anchor_count)))
                cells[f"{anchor_count},{agent_count}"] = cells.get(f"{anchor_count},{agent_count}", 0) + 1
            cohorts[scan] = dict(scenario_count=count, cells=cells, sample_counts=list(rule.SAMPLE_COUNTS), iterations=[1, 2],
                                 groups_per_cell_cap=args.groups_per_cell, design=design_record(plans[scan]))
        scan_files, open_scan, archives, digests = {}, None, None, None

        def close(scan):
            for kind, archive in archives.items():
                archive.close()
                path = directory / f"scan-{scan:02d}/{kind}.zip"
                path.with_name(path.name + ".partial").replace(path)
            scan_files[scan] = {kind: dict(sha256=sha256(directory / f"scan-{scan:02d}/{kind}.zip"), content_sha256=digests[kind].hexdigest())
                                for kind in archives}

        for results in ordered(executor, _generate, chunks(entries)):
            for name, raw, calibrated in results:
                scan = int(name.split(".")[0]) // ANCHOR_ONLY_STRIDE
                if scan != open_scan:
                    if open_scan is not None:
                        close(open_scan)
                    (directory / f"scan-{scan:02d}").mkdir(parents=True, exist_ok=True)
                    archives = {kind: zipfile.ZipFile(directory / f"scan-{scan:02d}/{kind}.zip.partial", "w", zipfile.ZIP_DEFLATED)
                                for kind in (("scenarios", "calibrated") if folds[scan] else ("scenarios",))}
                    digests = {kind: hashlib.sha256() for kind in archives}
                    open_scan = scan
                for kind, payload in (("scenarios", raw), ("calibrated", calibrated)):
                    if kind not in archives:
                        continue
                    archives[kind].writestr(zip_entry(name), payload)
                    digests[kind].update(name.encode())
                    digests[kind].update(payload)
        if open_scan is not None:
            close(open_scan)
    record = {}
    for scan in scans:
        coefficients, calibration_path = folds[scan] or (None, None)
        folder = directory / f"scan-{scan:02d}"
        write_json(folder / "cohort.json", cohorts[scan])
        # The identity select_bern.prepare_scan seals, so the calibration of every fold can be audited the same way.
        if calibration_path is not None:
            write_json(folder / "calibrated.manifest.json", dict(identity=dict(raw_sha256=scan_files[scan]["scenarios"]["sha256"],
                       calibration_sha256=sha256(calibration_path), coefficients=coefficients, source_sha256=sha256(Path(fit_bern.__file__))),
                       outputs={"calibrated.zip": scan_files[scan]["calibrated"]["sha256"]}))
        copied = None
        if args.campaign_profiles:
            source = Path(args.campaign_profiles) / f"scan-{scan:02d}/excess_fitted/profile.json"
            target = folder / "excess_fitted/profile.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied = dict(source=str(source.resolve()), sha256=sha256(target))
        record[str(scan)] = dict(files=scan_files[scan], scenario_count=cohorts[scan]["scenario_count"],
                                 calibration=None if calibration_path is None else str(calibration_path.resolve()),
                                 calibration_sha256=None if calibration_path is None else sha256(calibration_path), coefficients=coefficients,
                                 copied_profile=copied)
    manifest = dict(schema_version=SCHEMA, product="anchor only held-scan corpora", rule=RULE, table=str(Path(args.table).resolve()),
                    table_md5=rule.CORRECTED_TABLE_MD5, groups_per_cell=args.groups_per_cell, anchor_counts=list(ANCHOR_ONLY_COUNTS),
                    agent_counts=list(rule.AGENT_COUNTS), iterations=[1, 2], sample_counts=list(rule.SAMPLE_COUNTS),
                    calibration="fit_bern.calibration(scan) and fit_bern.calibrated_packets on every packet, as select_bern.prepare_scan",
                    scans=record, sources=dict(sources(), **{"fit_bern.py": sha256(Path(fit_bern.__file__))}), environment=environment(),
                    workers=args.workers, seconds=round(time.time() - started, 1))
    write_json(directory / "manifest.json", manifest)
    print(json.dumps({scan: {kind: files["sha256"] for kind, files in entry["files"].items()} for scan, entry in record.items()}, indent=1))


def parity(args):
    started = time.time()
    data = load_scans(args.table, (rule.EVALUATION_SCAN,))
    if rule.file_digest(args.evaluation, "md5") != rule.EVALUATION_ARCHIVE_MD5 and not args.skip_archive_checksum:
        raise ValueError("the evaluation archive is not the published one")
    plan = rule.design(data[rule.EVALUATION_SCAN][0])
    superset = set(plan.anchors[rule.SUPERSET_SIZE])
    if args.all:
        numbers = list(range(1, plan.scenario_count() + 1))
    else:
        numbers = set()
        for _, _, first, groups in plan.cells():
            numbers.update((first, first + groups // 2, first + groups - 1))
        rng = np.random.default_rng(args.seed)
        numbers.update(int(n) for n in rng.choice(np.arange(1, plan.scenario_count() + 1), args.sample, replace=False))
        numbers.update(plan.numbers(COOPERATIVE_GROUPS, rule.ANCHOR_COUNTS))
        numbers = sorted(numbers)
    batches = [numbers[i:i + 64] for i in range(0, len(numbers), 64)]
    compared, mismatches = 0, []
    with pool(args.workers, _init_parity, (data, str(args.evaluation), plan)) as executor:
        for count, missing in ordered(executor, _compare, batches):
            compared += count
            mismatches.extend(missing)
    record = dict(schema_version="solid.training.release-parity.v2", table=str(Path(args.table).resolve()), table_md5=rule.CORRECTED_TABLE_MD5,
                  evaluation=str(Path(args.evaluation).resolve()), scan=rule.EVALUATION_SCAN, scenarios_in_release=plan.scenario_count(),
                  compared=compared, identical=compared - len(mismatches), mismatches=mismatches[:1000], mode="all" if args.all else "sample",
                  anchor_groups={str(count): list(group) for count, group in plan.anchors.items()},
                  release_pruning_holds=dict(forced_in_superset=set(rule.RELEASE_FORCED) <= superset,
                                             excluded_outside_superset=not (rule.RELEASE_EXCLUDED & superset)),
                  sources=sources(), environment=environment(), workers=args.workers, seconds=round(time.time() - started, 1))
    write_json(args.report, record)
    print(json.dumps({k: record[k] for k in ("compared", "identical", "mode", "release_pruning_holds", "seconds")}, indent=1))
    if mismatches:
        raise SystemExit(f"{len(mismatches)} regenerated scenarios differ from the release")


# ---- statistics ----------------------------------------------------------------------------------------------------


def statistics(args):
    import heldscan_statistics as stats

    started = time.time()
    scans = (*rule.TRAINING_SCANS, rule.EVALUATION_SCAN)
    data = load_scans(args.table, scans)
    with pool(args.workers, _init_worker, (data, None)) as executor:
        plans = designs(executor, scans)
    published_frame = rule.load_table(args.published_table, rule.PUBLISHED_TABLE_MD5)
    published = {scan: rule.scan_layout(published_frame, scan) for scan in scans}
    record = stats.summary(plans, data, published, COOPERATIVE_GROUPS, ANCHOR_ONLY_GROUPS, ANCHOR_ONLY_COUNTS,
                           old_cooperative=args.old_cooperative if args.old_cooperative and args.old_cooperative.exists() else None,
                           old_anchor_only=args.old_anchor_only if args.old_anchor_only and args.old_anchor_only.exists() else None)
    record.update(table=str(Path(args.table).resolve()), published_table=str(Path(args.published_table).resolve()),
                  old_cooperative=str(args.old_cooperative), old_anchor_only=str(args.old_anchor_only), sources=sources(),
                  seconds=round(time.time() - started, 1))
    write_json(args.output, record)
    print(stats.table_text(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("cooperative", "anchor-only", "parity", "statistics"):
        command = commands.add_parser(name)
        command.add_argument("--table", type=Path, default=TABLE)
        command.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
        if name in ("cooperative", "anchor-only"):
            command.add_argument("--scans", type=int, nargs="+", default=list(rule.TRAINING_SCANS))
        if name == "cooperative":
            command.add_argument("--output", type=Path, default=COOPERATIVE)
            command.add_argument("--groups-per-cell", type=int, default=COOPERATIVE_GROUPS)
        if name == "anchor-only":
            command.add_argument("--output-directory", type=Path, default=ANCHOR_ONLY)
            command.add_argument("--groups-per-cell", type=int, default=ANCHOR_ONLY_GROUPS)
            command.add_argument("--campaign-profiles", type=Path, default=None,
                                 help="earlier campaign directory whose scan-XX/excess_fitted/profile.json files are copied")
        if name == "parity":
            command.add_argument("--evaluation", type=Path, default=EVALUATION)
            command.add_argument("--all", action="store_true", help="compare all 148,260 scenarios")
            command.add_argument("--sample", type=int, default=300, help="random scenarios added to the per cell ones")
            command.add_argument("--seed", type=int, default=20260925)
            command.add_argument("--report", type=Path, default=PARITY)
            command.add_argument("--skip-archive-checksum", action="store_true")
        if name == "statistics":
            command.add_argument("--published-table", type=Path, default=DATA / "inputs/2048c6e056a6484e/Rangings at Responders.pkl")
            command.add_argument("--old-cooperative", type=Path, default=OUT / "bern-training-scenarios.zip",
                                 help="the old cooperative archive, counted from its members when it exists")
            command.add_argument("--old-anchor-only", type=Path, default=DATA / "anc-tail-generalization",
                                 help="the old anchor only corpora (scan-XX/scenarios.zip), counted when they exist")
            command.add_argument("--output", type=Path, default=STATISTICS)
    args = parser.parse_args()
    if any(scan not in rule.TRAINING_SCANS for scan in getattr(args, "scans", ())):
        raise ValueError("held-scan corpora take training scans 1 to 11 only")
    {"cooperative": cooperative, "anchor-only": anchor_only, "parity": parity, "statistics": statistics}[args.command](args)


if __name__ == "__main__":
    main()
