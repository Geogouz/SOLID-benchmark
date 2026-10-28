"""Audit Bregar v2 and check training-only calibration without positioning."""

import argparse
import copy
import csv
import hashlib
import io
import json
import re
import subprocess
from collections import defaultdict
import zipfile

import numpy as np

import audit
import pilot

OUTPUT = audit.DATA / "bregar-preflight.json"
SPLIT = audit.DATA / "bregar-split.json"
RAW_AUDIT = audit.DATA / "bregar-raw-audit.json"
CALIBRATION = audit.DATA / "bregar-calibration.json"
ARCHIVE = audit.DATA / "sources/bregar/8280736/data_set.zip"
RELEASE = "https://zenodo.org/api/records/8280736"
CODE = "ef1c7906344ce54ab1b0de9c3e7b4e1e5f4dc4cc"
CHANNELS = (1, 2, 3, 4, 5, 7)
# the anchor only comparison runs these baselines
LABELS = {"jin-2021": "Jin 2021", "fnl": "FNL 2025", "strong": "STRONG 2021"}
RAW = re.compile(r"data_set/raw_data/(environment[0-3])/data/([^/]+)/ch([1-7])_(A[1-8])\.csv")
HEADER = ("TAG_ID,ANCHOR_ID,X_TAG,Y_TAG,Z_TAG,X_ANCHOR,Y_ANCHOR,Z_ANCHOR,NLOS,RANGE,"
          "FP_INDEX,RSS,RSS_FP,FP_POINT1,FP_POINT2,FP_POINT3,STDEV_NOISE,CIR_POWER,MAX_NOISE,"
          "RXPACC,CHANNEL_NUMBER,FRAME_LENGTH,PREAMBLE_LENGTH,BITRATE,PRFR,PREAMBLE_CODE,CIR...")


def baseline_settings():
    """The retained Bern anchor only settings of every baseline with the spatial model, and the hashes of their sources.
    A reader (SOLID_READER=1) reads them from settings/data/bregar-baselines.json, an extract of what this returns."""
    if pilot.READER:
        settings, sources = pilot.shipped("bregar-baselines.json")
        return settings["baselines"], sources
    baselines, sources = [], {}
    roster = pilot.evidence.read(pilot.evidence.retained_data.EVIDENCE)
    for method in LABELS:
        row = next(m for m in roster["methods"] if m["id"] == method)
        path = pilot.evidence.retained_data.run_directory(method, "ANC", row["results"]["ANC"]) / "benchmark.json"
        pilot.evidence.official_scorer.require_hash(path, row["results"]["ANC"]["benchmark_report_sha256"], "retained baseline")
        record = pilot.evidence.read(path)
        config = copy.deepcopy(record["config"])
        if config["aggregation"] != "Median" or config["max_time_micros"] != 0:
            raise ValueError("unexpected retained baseline settings")
        config["dimension_policy"] = "spatial"
        method_config = record.get("method_config") or dict(method="fnl", initialization="released")
        baselines.append(dict(method=method, method_config=method_config, solve_config=config))
        sources[str(path)] = pilot.identity(path)
    return baselines, sources


def coordinates(payload, anchors=False):
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    values = np.array([r[1:] if anchors else r for r in rows[1:]], dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid surveyed coordinates")
    if anchors and [r[0] for r in rows[1:]] != [f"A{i}" for i in range(1, 9)]:
        raise ValueError("unexpected anchor identities")
    return values


def range_record(payload, position, anchor, channel):
    lines = payload.decode().splitlines()
    if len(lines) < 2 or lines[1] not in (HEADER, HEADER.replace(",RANGE", "")):
        raise ValueError("unexpected raw data headers")
    rows = list(csv.reader(lines[2:]))
    if rows and min(map(len, rows)) < 27:
        raise ValueError("incomplete raw range row")
    values = np.array([[0 if i in (0, 1, 8) else r[i] for i in range(26)] for r in rows], dtype=float).reshape(-1, 26)
    if not np.isfinite(values).all() or (rows and set(values[:, 20]) != {channel}):
        raise ValueError("invalid raw values or channel")
    # The publisher's index legend includes range at column 9 even when its header is omitted.
    return dict(rows=len(rows), range_header_omitted="RANGE" not in lines[1],
        minimum_columns=min(map(len, rows), default=0), maximum_columns=max(map(len, rows), default=0),
        distinct_full_rows=len({tuple(r) for r in rows}), has_timestamp_or_sequence_header=any(
            key in lines[1].lower() for key in ("timestamp", "sequence")),
        tag_ids=sorted({r[0] for r in rows}), anchor_ids=sorted({r[1] for r in rows}),
        tag_reference_matches_folder=bool(np.allclose(values[:, 2:5], position, atol=.005, rtol=0)),
        maximum_tag_label_difference_m=float(np.max(np.abs(values[:, 2:5] - position))) if rows else None,
        anchor_reference_matches_table=bool(np.allclose(values[:, 5:8], anchor, atol=.005, rtol=0)),
        invalid_ranges=int(np.sum(values[:, 9] <= 0)),
        recorded_nlos_labels=sorted({r[8] for r in rows}),
        ten_report_blocks=len(rows) // 10)


def inspect():
    if OUTPUT.exists():
        raise ValueError("Bregar source audit already retained")
    record_path = audit.source(RELEASE, "bregar/8280736/record.json", 100000)
    record = json.loads(record_path.read_text())
    release_file, = record["files"]
    if (record["id"] != 8280736 or record["metadata"]["version"] != "2.0"
            or release_file["key"] != "data_set.zip" or release_file["size"] != 4441862175):
        raise ValueError("unexpected pinned release")
    sources = [record_path]
    tree = audit.source("https://api.github.com/repos/KlemenBr/uwb_positioning/git/trees/v1.0.0?recursive=1",
                        "bregar/code-v1.0.0/tree.json", 100000)
    if json.loads(tree.read_text())["sha"] != CODE:
        raise ValueError("publisher code tag changed")
    sources.append(tree)
    for name in ("preprocess.sh", "preprocess/fix_position_data.py", "preprocess/fix_outliers.py",
                 "preprocess/fix_offset.py", "preprocess/fix_final_loc2_A6.py", "preprocess/save_raw_data_to_json.py"):
        sources.append(audit.source(f"https://raw.githubusercontent.com/KlemenBr/uwb_positioning/{CODE}/{name}",
                                    f"bregar/code-v1.0.0/{name}", 100000))
    url = "https://zenodo.org/records/8280736/files/data_set.zip"
    # A reader reads the members from the archive that tools/fetch_data.py downloaded, not over HTTP ranges.
    remote = ARCHIVE.open("rb") if pilot.READER else audit.RangeFile(url, release_file["size"], budget=40_000_000)
    members = []

    def retained(archive, name):
        info = archive.getinfo(name)
        if info.file_size > 10_000_000:
            raise ValueError("member exceeds bounded source inspection")
        path = audit.DATA / "sources/bregar/8280736/members" / name
        sidecar = path.with_name(path.name + ".source.json")
        if path.exists():
            payload = path.read_bytes()
            previous = json.loads(sidecar.read_text())
            if previous["sha256"] != audit.sha256(payload) or previous["zip_crc32"] != info.CRC:
                raise ValueError("retained member identity mismatch")
        else:
            payload = archive.read(info)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            pilot.evidence.write(sidecar, dict(url=url, member=name, bytes=len(payload),
                sha256=audit.sha256(payload), zip_crc32=info.CRC, archive_checksum=release_file["checksum"]))
        members.append(dict(member=name, bytes=len(payload), sha256=audit.sha256(payload), zip_crc32=info.CRC))
        sources.extend((path, sidecar))
        return payload

    results = []
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        retained(archive, "data_set/README")
        for env in (f"environment{i}" for i in range(4)):
            anchor = coordinates(retained(archive, f"data_set/{env}/anchors.csv"), anchors=True)
            positions = coordinates(retained(archive, f"data_set/{env}/walking_path.csv"))
            expected = {"_".join(f"{x:.2f}" for x in p) for p in positions}
            files = [m for n in names if (m := RAW.fullmatch(n)) and m[1] == env]
            actual = {(m[2], int(m[3]), m[4]) for m in files}
            population = {(p, ch, f"A{a}") for p in expected for ch in CHANNELS for a in range(1, 9)}
            if actual != population:
                raise ValueError(f"raw file population differs in {env}")
            checks = []
            for p, ch, a in ((min(expected), 1, 1), (max(expected), 7, 8)):
                name = f"data_set/raw_data/{env}/data/{p}/ch{ch}_A{a}.csv"
                checks.append(dict(member=name, **range_record(retained(archive, name),
                    np.array(p.split("_"), dtype=float), anchor[a - 1], ch)))
            results.append(dict(environment=env, positions=len(positions), distinct_positions=len(expected),
                repeated_path_positions=len(positions) - len(expected), anchors=len(anchor),
                anchor_heights_m=sorted(set(anchor[:, 2].tolist())), tag_heights_m=sorted(set(positions[:, 2].tolist())),
                channels=list(CHANNELS), raw_files=len(files), checked_ranges=checks))
            print(json.dumps(results[-1]), flush=True)
    sources += [p.with_name(p.name + ".source.json") for p in list(sources) if not p.name.endswith(".source.json")
                and p.with_name(p.name + ".source.json") not in sources]
    result = dict(release=RELEASE, version="2.0", license=record["metadata"]["license"]["id"],
        archive_bytes=release_file["size"], archive_checksum=release_file["checksum"],
        full_archive_checksum_verified=False, downloaded_bytes=0 if pilot.READER else remote.downloaded, zip_members=len(names),
        member_integrity="ZIP CRC checked on extraction and retained SHA-256 per member. Whole ZIP not downloaded.",
        publisher_code=CODE, environments=results, members=members,
        models_fitted=0, positioning_solves=0, accuracy_results_inspected=0,
        decision="Suitable in principle for derived ANC. Raw-data split and calibration protocol required before execution.",
        sources={audit.name(p): pilot.identity(p) for p in sources})
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files(sources + [OUTPUT])


def split_positions(positions):
    if set(positions) != {f"environment{i}" for i in range(4)}:
        raise ValueError("expected all four environments")
    population = {}
    for environment, values in sorted(positions.items()):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
            raise ValueError("invalid split positions")
        ids = [environment + "/" + "_".join(f"{x:.2f}" for x in p) for p in values]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate or rounded-colliding position identifiers")
        population[environment] = sorted(ids)
    return [dict(held_environment=held, training_environments=[e for e in population if e != held],
                 training_positions=[p for e, ids in population.items() if e != held for p in ids],
                 evaluation_positions=ids)
            for held, ids in population.items()]


def freeze_split():
    if SPLIT.exists():
        raise ValueError("Bregar split already frozen")
    if audit.uncommitted():
        raise ValueError("commit the split source and protocol before freezing")
    checked = pilot.evidence.read(OUTPUT)
    for name, expected in checked["sources"].items():
        pilot.evidence.official_scorer.require_hash(audit.located(name), expected, "Bregar split source")
    if checked["version"] != "2.0" or checked["publisher_code"] != CODE:
        raise ValueError("unexpected Bregar release or preprocessing reference")
    positions = {}
    for row in checked["environments"]:
        env = row["environment"]
        path = audit.DATA / "sources/bregar/8280736/members/data_set" / env / "walking_path.csv"
        if audit.name(path) not in checked["sources"]:
            raise ValueError("survey path is not in the verified source inventory")
        positions[env] = coordinates(path.read_bytes())
        if len(positions[env]) != row["distinct_positions"]:
            raise ValueError("survey population changed")
    result = dict(source_revision=audit.revision(),
        preflight_sha256=pilot.identity(OUTPUT), protocol_sha256=pilot.document(pilot.HERE / "README.md"),
        status="split-frozen-execution-not-frozen", folds=split_positions(positions),
        channels=list(CHANNELS), samples=[1, 10],
        calibration=dict(kind="shortest-training-LOS-offset-per-hardware-pair-and-channel",
            fitting_environments="Other three environments in each fold. No held rows.",
            formula="Corrected range equals raw range minus mean training range excess at the shortest training LOS position.",
            ties="Sort equal-distance cells by environment and position identifier.",
            missing_training_LOS="Stop preparation. Never borrow held-environment calibration.",
            uncertainty_and_method_profiles="Fit only on corrected training rows. Freeze each method before evaluation."),
        publisher_link_exclusions=[dict(environment="environment2", anchor="A6",
            reason="Publisher reports erratic ranges and excludes this anchor in its released preprocessing.")],
        input_policy=dict(raw_ranges=True, publisher_position_labels=True, outcome_based_filtering=False,
            group="All surveyed positions in an environment, separately for each channel.",
            sampling="All complete common ten-row blocks over nonempty eligible links. One report uses the same block's first row.",
            empty_links="Keep as missing without dropping their agents.",
            incomplete_blocks="Count unused rows. Stop preparation if a channel has no common complete block.",
            unknown_coordinates="Scoring references only. No tag heights or held LOS labels enter inference."),
        pending=["Verify the complete raw archive and every row.",
                 "Verify hardware identities across environments before sharing offsets.",
                 "Freeze report block counts, method scope, fitted profiles and solver settings."],
        models_fitted=0, positioning_solves=0, evaluation_rows_used_for_training=0)
    pilot.evidence.write(SPLIT, result)
    pilot.register_files([SPLIT])
    print(json.dumps(dict(path=str(SPLIT), folds=[dict(held=r["held_environment"],
        training_positions=len(r["training_positions"]), evaluation_positions=len(r["evaluation_positions"]))
        for r in result["folds"]])), flush=True)


def raw_summary(records):
    hardware = defaultdict(set)
    tags = set()
    issues = []
    groups = []
    for row in records:
        if "error" in row:
            issues.append(dict(member=row["member"], reason=row["error"]))
            continue
        hardware[row["anchor"]].update(row["anchor_ids"])
        tags.update(row["tag_ids"])
        if row["invalid_ranges"]:
            issues.append(dict(member=row["member"], reason="Nonpositive raw range readings"))
        if not row["anchor_reference_matches_table"]:
            issues.append(dict(member=row["member"], reason="Inconsistent anchor coordinates"))
        if set(row["recorded_nlos_labels"]) - {"LOS", "NLOS"}:
            issues.append(dict(member=row["member"], reason="Unrecognized LOS labels"))
    hardware_ok = (set(hardware) == {f"A{i}" for i in range(1, 9)}
                   and all(len(ids) == 1 for ids in hardware.values())
                   and len(set().union(*hardware.values())) == 8 and len(tags) == 1
                   and not (tags & set().union(*hardware.values())))
    for env in (f"environment{i}" for i in range(4)):
        for channel in CHANNELS:
            rows = [r for r in records if (r["environment"], r["channel"]) == (env, channel)]
            if any("error" in r for r in rows):
                continue
            eligible = [r for r in rows if not (env == "environment2" and r["anchor"] == "A6")]
            nonempty = [r for r in eligible if r["rows"]]
            blocks = min((r["rows"] // 10 for r in nonempty), default=0)
            groups.append(dict(environment=env, channel=channel,
                agents=len({r["position"] for r in rows}), anchors=7 if env == "environment2" else 8,
                raw_rows=sum(r["rows"] for r in rows), eligible_rows=sum(r["rows"] for r in eligible),
                publisher_excluded_rows=sum(r["rows"] for r in rows if env == "environment2" and r["anchor"] == "A6"),
                empty_links=len(eligible) - len(nonempty), common_ten_report_blocks=blocks,
                ten_report_rows_used=10 * blocks * len(nonempty), one_report_rows_used=blocks * len(nonempty),
                unused_eligible_rows=sum(r["rows"] for r in eligible) - 10 * blocks * len(nonempty)))
            if blocks == 0:
                issues.append(dict(environment=env, channel=channel, reason="No complete common ten-report block"))
    return dict(groups=groups, hardware_ids={a: sorted(ids) for a, ids in sorted(hardware.items())},
        tag_ids=sorted(tags), consistent_hardware=hardware_ok, issues=issues,
        ready_for_calibration=hardware_ok and not issues and len(groups) == 24)


def full_audit():
    if audit.uncommitted():
        raise ValueError("commit the audit source before execution")
    checked, split = pilot.evidence.read(OUTPUT), pilot.evidence.read(SPLIT)
    pilot.evidence.official_scorer.require_hash(OUTPUT, split["preflight_sha256"], "frozen Bregar split source")
    path = ARCHIVE if ARCHIVE.exists() else ARCHIVE.with_suffix(".zip.part")
    with path.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if path.stat().st_size != checked["archive_bytes"] or "md5:" + md5 != checked["archive_checksum"]:
        raise ValueError("complete Bregar archive does not match the published checksum and size")
    if path != ARCHIVE:
        path.rename(ARCHIVE)
    source = dict(url="https://zenodo.org/api/records/8280736/files/data_set.zip/content",
        bytes=ARCHIVE.stat().st_size, md5=md5, sha256=pilot.identity(ARCHIVE))
    if RAW_AUDIT.exists():
        previous = pilot.evidence.read(RAW_AUDIT)
        if previous["source"] != source or previous["split_sha256"] != pilot.identity(SPLIT):
            raise ValueError("cannot refresh an audit against changed raw inputs or split")
    sidecar = ARCHIVE.with_name(ARCHIVE.name + ".source.json")
    pilot.evidence.write(sidecar, source)
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in checked["members"]:
            if audit.sha256(archive.read(member["member"])) != member["sha256"]:
                raise ValueError("complete archive differs from retained source inspection")
        actual = sorted(n for n in archive.namelist() if RAW.fullmatch(n))
        expected = sorted(f"data_set/raw_data/{p.split('/')[0]}/data/{p.split('/')[1]}/ch{ch}_A{a}.csv"
            for fold in split["folds"] for p in fold["evaluation_positions"] for ch in CHANNELS for a in range(1, 9))
        if actual != expected:
            raise ValueError("raw file population differs from frozen split")
        anchors = {f"environment{i}": coordinates(archive.read(f"data_set/environment{i}/anchors.csv"), True)
                   for i in range(4)}
        for index, name in enumerate(actual, 1):
            match = RAW.fullmatch(name)
            env, position, channel, anchor = match[1], match[2], int(match[3]), match[4]
            payload = archive.read(name)
            row = dict(member=name, environment=env, position=position, channel=channel, anchor=anchor,
                       sha256=audit.sha256(payload), zip_crc32=archive.getinfo(name).CRC)
            try:
                row.update(range_record(payload, np.array(position.split("_"), dtype=float),
                                        anchors[env][int(anchor[1:]) - 1], channel))
            except (ValueError, UnicodeError) as error:
                row["error"] = str(error)
            records.append(row)
            if index % 1000 == 0:
                print(f"Checked {index:,} / {len(actual):,} raw files", flush=True)
    result = dict(source_revision=audit.revision(), source=source, split_sha256=pilot.identity(SPLIT),
        preflight_sha256=pilot.identity(OUTPUT), records=records, **raw_summary(records),
        models_fitted=0, positioning_solves=0, evaluation_rows_used_for_training=0)
    pilot.evidence.write(RAW_AUDIT, result)
    pilot.register_files([ARCHIVE, sidecar, RAW_AUDIT])
    print(json.dumps(dict(files=len(records), rows=sum(r.get("rows", 0) for r in records),
        issues=len(result["issues"]), consistent_hardware=result["consistent_hardware"],
        ready_for_calibration=result["ready_for_calibration"])), flush=True)


def calibration_cells(archive, checked):
    if not checked["consistent_hardware"] or any(
            r["reason"] != "Nonpositive raw range readings" for r in checked["issues"]):
        raise ValueError("resolve structural raw-data errors before calibration")
    anchors = {f"environment{i}": coordinates(archive.read(f"data_set/environment{i}/anchors.csv"), True)
               for i in range(4)}
    cells = []
    for row in checked["records"]:
        if row["environment"] == "environment2" and row["anchor"] == "A6":
            continue
        payload = archive.read(row["member"])
        if audit.sha256(payload) != row["sha256"]:
            raise ValueError("raw calibration member differs from its audit")
        # The audited prefix holds the range. CIR columns need not be decoded again.
        ranges = np.array([float(line.split(",", 10)[9]) for line in payload.decode().splitlines()[2:]])
        if len(ranges) != row["rows"] or not np.isfinite(ranges).all():
            raise ValueError("calibration range population differs from its audit")
        if ranges.size and row["recorded_nlos_labels"] not in (["LOS"], ["NLOS"]):
            raise ValueError("calibration requires one propagation label per surveyed link")
        position = np.array(row["position"].split("_"), dtype=float)
        cells.append(dict(member=row["member"], environment=row["environment"], position=row["position"],
            channel=row["channel"], anchor=row["anchor"], tag_ids=row["tag_ids"], anchor_ids=row["anchor_ids"],
            los=row["recorded_nlos_labels"] == ["LOS"], ranges_m=ranges,
            distance_m=float(np.linalg.norm(position - anchors[row["environment"]][int(row["anchor"][1:]) - 1]))))
    return cells


def training_environments(fold):
    population = {f"environment{i}" for i in range(4)}
    excluded = {fold["held_environment"]}
    if "outer_held_environment" in fold:
        outer = fold["outer_held_environment"]
        if outer in excluded:
            raise ValueError("inner and outer held environments must differ")
        excluded.add(outer)
    environments = set(fold["training_environments"])
    if not excluded <= population or environments != population - excluded:
        raise ValueError("invalid held-environment training environments")
    return environments


def calibration_fold(cells, fold, hardware):
    held, environments = fold["held_environment"], training_environments(fold)
    positions = set(fold["training_positions"])
    if positions & set(fold["evaluation_positions"]):
        raise ValueError("invalid held-environment training split")
    training = sorted((r for r in cells if r["environment"] in environments
        and not (r["environment"] == "environment2" and r["anchor"] == "A6")), key=lambda r: r["member"])
    expected = {(p, ch, f"A{a}") for p in positions for ch in CHANNELS for a in range(1, 9)
                if not (p.startswith("environment2/") and a == 6)}
    actual = {(r["environment"] + "/" + r["position"], r["channel"], r["anchor"]) for r in training}
    if actual != expected or len(actual) != len(training):
        raise ValueError("calibration training population differs from the frozen split")
    by_pair = defaultdict(list)
    for row in training:
        values = np.asarray(row["ranges_m"])
        if not np.isfinite(values).all() or not np.isfinite(row["distance_m"]) or row["distance_m"] < 0:
            raise ValueError("invalid calibration ranges or reference")
        if len(values) and (row["anchor_ids"] != hardware["hardware_ids"][row["anchor"]]
                            or row["tag_ids"] != hardware["tag_ids"]):
            raise ValueError("calibration hardware differs from verified identities")
        by_pair[row["anchor"], row["channel"]].append(row)
    corrections = []
    offsets = {}
    for key, rows in sorted(by_pair.items()):
        candidates = [r for r in rows if r["los"] and len(r["ranges_m"])]
        if not candidates:
            raise ValueError(f"no training LOS cell for {held} {key}")
        selected = min(candidates, key=lambda r: (r["distance_m"], r["environment"], r["position"]))
        mean = float(np.mean(selected["ranges_m"]))
        offsets[key] = mean - selected["distance_m"]
        corrections.append(dict(anchor=key[0], channel=key[1], tag_id=hardware["tag_ids"][0],
            anchor_id=hardware["hardware_ids"][key[0]][0], offset_m=offsets[key],
            source_member=selected["member"], source_reports=len(selected["ranges_m"]),
            training_los_cells=len(candidates), reference_distance_m=selected["distance_m"], mean_raw_m=mean))
    raw, corrected, references = [], [], []
    flags = []
    for row in training:
        before = np.asarray(row["ranges_m"])
        after = before - offsets[row["anchor"], row["channel"]]
        raw.append(before)
        corrected.append(after)
        references.append(np.full(len(before), row["distance_m"]))
        if np.any(before <= 0) or np.any(after <= 0):
            flags.append(dict(member=row["member"], minimum_raw_m=float(before.min()),
                minimum_corrected_m=float(after.min()),
                raw_nonpositive_lines=(np.flatnonzero(before <= 0) + 3).tolist(),
                corrected_nonpositive_lines=(np.flatnonzero(after <= 0) + 3).tolist()))

    def summarize(indices):
        before, after, truth = (np.concatenate([values[i] for i in indices])
                                for values in (raw, corrected, references))
        if not len(before):
            raise ValueError("no training reports for calibration diagnostics")
        return dict(reports=len(before), raw_nonpositive=int(np.sum(before <= 0)),
            corrected_nonpositive=int(np.sum(after <= 0)),
            resolved_nonpositive=int(np.sum((before <= 0) & (after > 0))),
            introduced_nonpositive=int(np.sum((before > 0) & (after <= 0))),
            minimum_raw_m=float(before.min()), minimum_corrected_m=float(after.min()),
            raw_mae_m=float(np.mean(np.abs(before - truth))), corrected_mae_m=float(np.mean(np.abs(after - truth))),
            raw_rmse_m=float(np.sqrt(np.mean((before - truth) ** 2))),
            corrected_rmse_m=float(np.sqrt(np.mean((after - truth) ** 2))))
    nested = {"outer_held_environment": fold["outer_held_environment"]} if "outer_held_environment" in fold else {}
    return dict(**nested, held_environment=held, training_environments=sorted(environments), training_positions=len(positions),
        training_files=len(training), empty_training_links=sum(not len(r["ranges_m"]) for r in training),
        corrections=corrections, training=summarize(range(len(training))),
        training_by_environment=[dict(environment=env, **summarize(
            [i for i, r in enumerate(training) if r["environment"] == env])) for env in sorted(environments)],
        flagged_training_links=flags, held_environment_rows_used_for_fitting=0,
        held_environment_rows_used_for_diagnostics=0)


def check_calibration():
    if CALIBRATION.exists():
        raise ValueError("Bregar calibration check already retained")
    if audit.uncommitted():
        raise ValueError("commit the calibration source and protocol before fitting")
    checked, split = pilot.evidence.read(RAW_AUDIT), pilot.evidence.read(SPLIT)
    pilot.evidence.official_scorer.require_hash(SPLIT, checked["split_sha256"], "Bregar calibration split")
    pilot.evidence.official_scorer.require_hash(OUTPUT, split["preflight_sha256"], "Bregar calibration preflight")
    pilot.evidence.official_scorer.require_hash(ARCHIVE, checked["source"]["sha256"], "Bregar calibration archive")
    if split["calibration"]["kind"] != "shortest-training-LOS-offset-per-hardware-pair-and-channel":
        raise ValueError("unexpected frozen Bregar calibration rule")
    with zipfile.ZipFile(ARCHIVE) as archive:
        cells = calibration_cells(archive, checked)
    folds = []
    for fold in split["folds"]:
        result = calibration_fold(cells, fold, checked)
        folds.append(result)
        print(json.dumps(dict(held=fold["held_environment"], **result["training"])), flush=True)
    result = dict(source_revision=audit.revision(), split_sha256=pilot.identity(SPLIT),
        raw_audit_sha256=pilot.identity(RAW_AUDIT), protocol_sha256=pilot.document(pilot.HERE / "README.md"),
        archive_sha256=checked["source"]["sha256"], calibration_rule=split["calibration"], folds=folds,
        scope="Training residual diagnostics only. Calibration cells are included. This is not held-out positioning accuracy.",
        units="metres", calibration_parameters_fitted=sum(len(r["corrections"]) for r in folds),
        method_profiles_fitted=0, positioning_solves=0, raw_files_modified=0,
        additional_rows_removed=0, ranges_clamped=0,
        requires_nonpositive_input_policy=any(r["training"]["corrected_nonpositive"] for r in folds))
    pilot.evidence.write(CALIBRATION, result)
    pilot.register_files([CALIBRATION])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--freeze-split", action="store_true")
    actions.add_argument("--audit-raw", action="store_true")
    actions.add_argument("--check-calibration", action="store_true")
    args = parser.parse_args()
    if args.check_calibration:
        check_calibration()
    else:
        full_audit() if args.audit_raw else freeze_split() if args.freeze_split else inspect()
