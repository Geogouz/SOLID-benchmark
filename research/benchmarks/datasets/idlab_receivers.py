"""Reconcile original receiver records with the published IDLab ranges."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request
from urllib.parse import urlencode

import audit
import idlab_admission as admission
import idlab_attempts as attempts
import pilot


OUTPUT = admission.OUTPUT.with_name("idlab-receivers.json")
INVENTORY = admission.FOLDER / "receiver-sources.json"
HEADER = re.compile(rb"^(\d+), \d+, (\d+), [0-9a-f]+, [0-9a-f]+, [0-9a-f]+, ")
COMPLETE = re.compile(rb"\t((?:\d+, ){9})(-?\d+), ((?:\d+, ){4})0x([0-9a-f]{8}), \*{5}\s*$")
COLUMNS = ["slot_ID_anchor", "byte_offset", "RAWT2", "RAWT3", "RAWT6", "T1", "T2", "T3",
           "T4", "T5", "T6", "trailer_flag", "distance", "channel", "bitrate", "preamble", "prf", "txpower_anchor"]


def read_receiver(stream, expected_bytes, peer):
    digest, size, offset, pending = hashlib.sha256(), 0, 0, b""
    records, counters, errors = [], Counter(), Counter()
    last_slot, gaps, rollovers = None, 0, 0

    def consume(raw):
        nonlocal offset, last_slot, gaps, rollovers
        errors.update(code.decode() for code in re.findall(rb"\[ERROR (\d+)\]", raw))
        header = HEADER.match(raw)
        full = COMPLETE.search(raw)
        if header:
            slot, node = map(int, header.groups())
            if node != peer:
                raise ValueError("receiver ID differs from its source file")
            counters["slot_headers"] += 1
            if last_slot is not None:
                gaps += slot != (last_slot + 1) % 65536
                rollovers += last_slot == 65535 and slot == 0
            last_slot = slot
            counters["headers_with_error_codes"] += b"[ERROR " in raw
            counters["headers_without_complete_range"] += not bool(full)
        if full:
            prefix = raw.split(b", CIR:", 1)[0].split(b", ")
            if not header or len(prefix) < 8 or not prefix[7].isdigit():
                raise ValueError("complete receiver range has no raw receive timestamp")
            first, distance, settings, power = full.groups()
            values = [int(value) for value in first.split(b", ")[:-1]]
            radio = [int(value) for value in settings.split(b", ")[:-1]]
            records.append([slot, offset, int(prefix[7]), *values, int(distance), *radio, power.decode()])
            counters["complete_ranges"] += 1
        elif b"\t" in raw and raw.rstrip().endswith(b"*****") and b"[ERROR " not in raw:
            counters["unparsed_nonerror_trailers"] += 1
        offset += len(raw) + 1

    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        if size > expected_bytes:
            raise ValueError("receiver log exceeds its declared byte length")
        digest.update(chunk)
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for raw in lines:
            consume(raw)
    # These published files can end during a printed slot. Preserve that fact.
    trailing = pending.decode("ascii", errors="replace")[:120] if pending else None
    if pending:
        consume(pending)
    if size != expected_bytes or not counters["slot_headers"]:
        raise ValueError("incomplete receiver download or no receiver records")
    return dict(bytes=size, sha256=digest.hexdigest(), columns=COLUMNS, records=records,
                counts=dict(counters), error_codes=dict(errors), slot_gaps=gaps,
                slot_rollovers=rollovers, trailing_fragment=trailing)


def compare_node(node, receiver_logs, tag, rows):
    compact = defaultdict(list)
    for row in rows:
        compact[int(row["anchor_ID_anchor"])].append(row)
    matched, mismatches, missing, extra, duplicate_keys = 0, [], [], [], []
    absent_diagnostics = Counter()
    for source in receiver_logs:
        peer, receiver = source["peer"], source["data"]
        index = {}
        for values in receiver["records"]:
            record = dict(zip(COLUMNS, values, strict=True))
            key = tuple(record[k] for k in ("RAWT3", "RAWT6", "T1", "T5"))
            if key in index:
                duplicate_keys.append(dict(peer=peer, timestamps=key))
            index[key] = record
        used = set()
        for row in compact[peer]:
            key = tuple(int(row[k]) for k in ("RAWT3", "RAWT6", "T1", "T5"))
            record = index.get(key)
            if record is None:
                missing.append(dict(peer=peer, publisher_row=int(row[""]), timestamps=key))
                continue
            used.add(key)
            differences = {}
            for field in COLUMNS:
                if field in ("byte_offset", "trailer_flag"):
                    continue
                if field == "RAWT2" and row[field] == "":
                    absent_diagnostics[field] += 1
                    continue
                expected = row[field] if field == "txpower_anchor" else int(float(row[field]))
                if field == "slot_ID_anchor":
                    expected %= 65536
                if record[field] != expected:
                    differences[field] = dict(receiver=record[field], compact=expected)
            if differences:
                mismatches.append(dict(peer=peer, publisher_row=int(row[""]), differences=differences))
            else:
                matched += 1
        extra.extend(dict(peer=peer, **record) for key, record in index.items() if key not in used)
    # Corrected transmit timestamps share the antenna delay. Receive corrections vary.
    delays = {(int(row["T1"]) - int(row["RAWT1"])) % 2**40 for row in rows}
    delays.update((int(row["T5"]) - int(row["RAWT5"])) % 2**40 for row in rows)
    if len(delays) != 1:
        raise ValueError("sender transmit delay is not constant across the compact file")
    delay = delays.pop()
    published_slots = {int(row["slot_ID_tag"]) for row in rows}
    checked_peers = {source["peer"] for source in receiver_logs}
    completed = defaultdict(list)
    for record in extra:
        key = tuple(record[k] for k in ("peer", "T1", "T5", "channel", "bitrate", "preamble", "prf", "txpower_anchor"))
        completed[key].append(record)
    orphans = []
    for slot, record in enumerate(tag["records"], 1):
        _, peer, t1, block, _, t4, t5 = record
        if peer == node or peer not in checked_peers or t4 is None or slot in published_slots:
            continue
        settings = tag["blocks"][block]["settings"]
        key = (peer, (t1 + delay) % 2**40, (t5 + delay) % 2**40,
               *(settings[k] for k in ("channel", "bitrate", "preamble", "prf", "txpower_tag")))
        matches = completed.get(key, [])
        if len(matches) > 1:
            raise ValueError("omitted sender range has ambiguous receiver matches")
        receiver = matches[0] if matches else None
        orphans.append(dict(sender_slot=slot, peer=peer, block=block,
                            receiver_complete=receiver is not None,
                            receiver_byte_offset=receiver["byte_offset"] if receiver else None))
    return dict(compact_rows=sum(len(compact[p]) for p in checked_peers), matched_compact_rows=matched,
                missing_receiver_records=missing, field_mismatches=mismatches,
                duplicate_timestamp_keys=duplicate_keys, complete_ranges_without_compact_row=extra,
                missing_compact_diagnostics=dict(absent_diagnostics),
                sender_closing_without_compact=orphans, transmit_delay_ticks=delay)


def inspect_source(node, name, peer, cached):
    url = attempts.preflight.IDLAB_SHARE + "/download?" + urlencode(dict(path=f"/Raw data/Node {node}", files=name))
    if cached:
        if cached["url"] != url or cached["peer"] != peer or cached["data"]["columns"] != COLUMNS:
            raise ValueError("retained receiver source identity changed")
        return cached
    with urllib.request.urlopen(url, context=audit.CONTEXT, timeout=40) as response:
        size = int(response.headers.get("Content-Length", "0"))
        if response.status != 200 or not 0 < size <= 1_000_000_000:
            raise ValueError("unexpected receiver response or byte length")
        result = read_receiver(response, size, peer)
    if node == 1 and peer == 2 and result["sha256"] != "967e196ddd1b3d37b679fe74a2844b143bd0a34592ccf536d1aab4ecc59b3354":
        raise ValueError("previously inspected receiver source changed")
    print(f"Checked sender {node}, receiver {peer}, {len(result['records']):,} complete ranges", flush=True)
    return dict(name=name, peer=peer, url=url, data=result)


def save_progress(receivers, tags, revision, failures, summaries):
    sources = {str(Path(__file__).resolve().relative_to(pilot.ROOT)): pilot.identity(Path(__file__)),
                          str(INVENTORY.relative_to(pilot.ROOT)): pilot.identity(INVENTORY)}
    for node, logs in sorted(receivers.items()):
        path = OUTPUT.with_name(f"idlab-node{node}-receiver-records.json.gz")
        file_hash = pilot.identity(path)
        sources[pilot.audit.name(path)] = file_hash
        if node in summaries and summaries[node]["records_sha256"] == file_hash:
            continue
        with gzip.open(pilot.audit.located(tags[node]["records_path"]), "rt") as stream:
            tag = json.load(stream)
        with (admission.FOLDER / f"office_lab_node_{node:02d}_short.csv").open(newline="") as stream:
            comparison = compare_node(node, logs, tag, list(csv.DictReader(stream)))
        summaries[node] = dict(node=node, checked_peers=sorted(log["peer"] for log in logs), comparison=comparison,
                          records_sha256=file_hash,
                          records_path=pilot.audit.name(path), sources=[dict(name=log["name"], peer=log["peer"],
                          url=log["url"], bytes=log["data"]["bytes"], sha256=log["data"]["sha256"],
                          counts=log["data"]["counts"], slot_gaps=log["data"]["slot_gaps"],
                          trailing_fragment=log["data"]["trailing_fragment"]) for log in logs])
    nodes = [summaries[node] for node in sorted(summaries)]
    checked = sum(len(row["checked_peers"]) for row in nodes)
    problems = sum(len(row["comparison"][key]) for row in nodes
                   for key in ("missing_receiver_records", "field_mismatches", "duplicate_timestamp_keys"))
    result = dict(schema="solid.external.idlab-receivers.v1", source_revision=revision, sources=sources,
                  sender_audit_sha256=pilot.identity(attempts.OUTPUT), nodes=nodes, failures=failures,
                  checked_files=checked, expected_files=210,
                  status="complete-receiver-audit" if checked == 210 and not problems and not failures else "receiver-audit-in-progress",
                  matched_compact_rows=sum(row["comparison"]["matched_compact_rows"] for row in nodes),
                  original_bytes=sum(s["bytes"] for row in nodes for s in row["sources"]),
                  positioning_solves=0, models_fitted=0,
                  interpretation="Completed receiver exchanges are checked against the compact CSV. Absence of a completed receiver record does not identify the physical cause of failure.")
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT, INVENTORY, *(pilot.audit.located(row["records_path"]) for row in nodes)])
    print(f"Saved {checked} / 210 receiver logs. {result['matched_compact_rows']:,} compact rows matched.", flush=True)


def prepare():
    if pilot.audit.uncommitted():
        raise ValueError("commit receiver audit source before execution")
    sender = pilot.evidence.read(attempts.OUTPUT)
    if sender["status"] != "complete-original-tag-audit":
        raise ValueError("complete sender audit is required")
    for name, expected in sender["sources"].items():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab sender audit")
    checked = pilot.evidence.read(admission.OUTPUT)
    for name, expected in checked["sources"].items():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab compact source")
    inventory = pilot.evidence.read(INVENTORY)
    if sorted(map(int, inventory["nodes"])) != list(range(1, 16)):
        raise ValueError("receiver inventory must include every sender")
    hosts = {name.split("_", 1)[1].removeprefix("nuc"): i for i, (name, _) in enumerate(attempts.TAG_LOGS, 1)}
    targets = []
    for node, names in inventory["nodes"].items():
        peers = [hosts[name.split("_", 1)[1]] for name in names]
        if sorted(peers) != [i for i in range(1, 16) if i != int(node)]:
            raise ValueError("receiver inventory omits or duplicates a peer")
        targets.extend((int(node), name, peer) for name, peer in zip(names, peers))
    previous = pilot.evidence.read(OUTPUT) if OUTPUT.exists() else {}
    receivers = defaultdict(list)
    if previous:
        pilot.evidence.official_scorer.require_hash(attempts.OUTPUT, previous["sender_audit_sha256"], "retained sender audit")
        for path in (Path(__file__).resolve(), INVENTORY):
            pilot.evidence.official_scorer.require_hash(path, previous["sources"][pilot.audit.name(path)], "receiver audit source")
        for node in previous["nodes"]:
            path = pilot.audit.located(node["records_path"])
            pilot.evidence.official_scorer.require_hash(path, previous["sources"][node["records_path"]], "retained receiver records")
            with gzip.open(path, "rt") as stream:
                receivers[node["node"]] = json.load(stream)
    retained = {(node, log["name"]): log for node, logs in receivers.items() for log in logs}
    revision = pilot.audit.revision()
    tags = {row["node"]: row for row in sender["nodes"]}
    failures, summaries = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(inspect_source, node, name, peer, retained.get((node, name))): (node, name)
                for node, name, peer in targets}
        for future in as_completed(jobs):
            node, name = jobs[future]
            try:
                source = future.result()
                if (node, name) not in retained:
                    receivers[node].append(source)
                receivers[node].sort(key=lambda log: log["peer"])
                path = OUTPUT.with_name(f"idlab-node{node}-receiver-records.json.gz")
                temporary = path.with_suffix(path.suffix + ".partial")
                with gzip.GzipFile(filename=str(temporary), mode="wb", mtime=0) as stream:
                    stream.write(json.dumps(receivers[node], separators=(",", ":")).encode())
                temporary.replace(path)
            except Exception as error:
                failures[f"{node}/{name}"] = str(error)
                print(f"Receiver source check failed. {node}/{name}. {error}", flush=True)
            save_progress(receivers, tags, revision, failures, summaries)
    if pilot.evidence.read(OUTPUT)["status"] != "complete-receiver-audit":
        raise ValueError("receiver checks need review before availability fitting")


if __name__ == "__main__":
    prepare()
