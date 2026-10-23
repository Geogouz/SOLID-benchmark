"""Check original IDLab tag records against the compact file before using missingness."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.request
from urllib.parse import urlencode

import audit
import idlab_admission as admission
import idlab_split
import pilot
import static_preflight as preflight


OUTPUT = admission.OUTPUT.with_name("idlab-attempts.json")
TAG_LOGS = [
    ("tag20200115-132104_9-35_USB1.txt", 951246836),
    ("tag20200116-120118_9-14_USB1.txt", 1248044018),
    ("tag20200116-225121_9-1_USB1.txt", 2084943638),
    ("tag20200117-143705_9-37_USB1.txt", 754814934),
    ("tag20200117-200423_9-16_USB1.txt", 1501690899),
    ("tag20200118-065155_9-3_USB1.txt", 1914038709),
    ("tag20200118-203559_9-17_USB1.txt", 1324389340),
    ("tag20200119-060738_9-18_USB1.txt", 1206236674),
    ("tag20200119-144858_9-5_USB1.txt", 2017822359),
    ("tag20200120-051825_9-6_USB1.txt", 1626350402),
    ("tag20200120-170042_nuc9-20_USB0.txt", 1220948056),
    ("tag20200121-082952_9-21_USB1.txt", 1077421807),
    ("tag20200121-161538_9-22_USB1.txt", 1047959880),
    ("tag20200121-234830_9-23_USB1.txt", 1001554278),
    ("tag20200207-113347_9-24_USB1.txt", 770865306),
]
SETTING = re.compile(r"UWB \(ch(\d+)\) \| @ (\d+) kbps \| +(\d+) pre \| (\d+) MHz \| 0x([0-9a-f]{8})")
RECORD = re.compile(r"^(\d+), \d+, (\d+), [0-9a-f]+, [0-9a-f]+, [0-9a-f]+, \d+, (\d+)(?:,|$)")


def slot_continuity(records):
    slots = [row[0] for row in records]
    return dict(consecutive_from_one=slots == list(range(1, len(slots) + 1)),
                consecutive_modulo_65536=all(slot == i % 65536 for i, slot in enumerate(slots, 1)),
                rollovers=sum(a == 65535 and b == 0 for a, b in zip(slots, slots[1:])),
                duplicate_slots=len(slots) - len(set(slots)))


def read_tag(stream, expected_bytes):
    digest, size, pending, offset = hashlib.sha256(), 0, b"", 0
    blocks, records, ended = [], [], False
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        if size > expected_bytes:
            raise ValueError("tag log exceeds its declared byte length")
        digest.update(chunk)
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for raw in lines:
            prefix = raw.split(b", CIR:", 1)[0].decode("ascii", errors="replace").strip()
            setting = SETTING.fullmatch(prefix)
            record = RECORD.match(prefix)
            if setting:
                channel, bitrate, preamble, prf, power = setting.groups()
                blocks.append(dict(settings=dict(channel=int(channel), bitrate=int(bitrate),
                    preamble=int(preamble), prf=int(prf), txpower_tag=power),
                    first_byte=offset, records=0, peers=[], termination=None))
            elif record:
                if not blocks:
                    raise ValueError("tag record has no radio setting header")
                slot, peer, raw_t1 = map(int, record.groups())
                if not 1 <= peer <= 15:
                    raise ValueError("tag record has an unknown peer")
                closing = re.search(rb"\t(\d+), (\d+),", raw)
                raw_t4, raw_t5 = map(int, closing.groups()) if closing else (None, None)
                records.append([slot, peer, raw_t1, len(blocks) - 1, offset, raw_t4, raw_t5])
                blocks[-1]["records"] += 1
                if peer not in blocks[-1]["peers"]:
                    blocks[-1]["peers"].append(peer)
            elif prefix in ("no range received, switching", "Maximum ranges reached, switching"):
                if not blocks or blocks[-1]["termination"] is not None:
                    raise ValueError("unexpected tag setting termination")
                blocks[-1]["termination"] = prefix
            elif prefix == "END":
                ended = True
            offset += len(raw) + 1
        if size % (128 * 1024 * 1024) == 0:
            print(f"Read {size / 1_000_000:.0f} MB of the original tag log", flush=True)
    if pending.strip() == b"END":
        ended = True
    elif pending.strip():
        raise ValueError("tag log has an incomplete final line")
    if size != expected_bytes or not ended or not records:
        raise ValueError("incomplete tag log")
    return dict(bytes=size, sha256=digest.hexdigest(), blocks=blocks, records=records,
                slot_continuity=slot_continuity(records))


def compare_compact(tag, rows):
    # Publisher IDs unwrap the raw 16-bit counter across the complete recording.
    slots = dict(enumerate(tag["records"], 1))
    matched, mismatches, identities, seen = 0, [], [], set()
    for row in rows:
        slot = int(row["slot_ID_tag"])
        if slot in seen:
            raise ValueError("duplicate compact tag slot")
        seen.add(slot)
        raw = slots.get(slot)
        expected = tuple(int(row[k]) for k in ("anchor_ID_tag", "RAWT1", "RAWT4", "RAWT5"))
        if raw is None or raw[0] != slot % 65536 or (raw[1], raw[2], raw[5], raw[6]) != expected:
            identities.append(dict(publisher_row=int(row[""]), slot=slot, tag_record=raw))
            continue
        settings = tag["blocks"][raw[3]]["settings"]
        differences = {key: dict(tag=value, compact=row[key]) for key, value in settings.items()
                       if str(value) != row[key]}
        if differences:
            mismatches.append(dict(publisher_row=int(row[""]), slot=slot, differences=differences))
        matched += 1
    return dict(compact_rows=len(seen), matched_compact_rows=matched,
                identity_mismatches=identities, setting_mismatches=mismatches,
                tag_records_without_compact_row=len(set(slots) - seen))


def population_counts(split):
    settings = defaultdict(list)
    for group in split["groups"]:
        settings[tuple(sorted(group["settings"].items()))].append(group["evaluation"]["valid_ranges"])
    common = sum(min(counts) // 10 for counts in settings.values())
    complete = sum(max(counts) // 10 for counts in settings.values())
    anchor_subsets = sum(math.comb(len(split["nodes"]), count) for count in range(1, 7))
    return dict(observed_settings=len(settings), common_ten_report_panels=common,
        settings_lost_by_common_rule=sum(min(counts) < 10 for counts in settings.values()),
        common_rule_used_reports=sum((min(counts) // 10) * 10 * len(counts) for counts in settings.values()),
        complete_rank_panels=complete, settings_without_any_complete_ten_report_link=sum(max(c) < 10 for c in settings.values()),
        panels_including_partial_last_block=sum(math.ceil(max(c) / 10) for c in settings.values()),
        complete_ten_report_link_blocks=sum(n // 10 for counts in settings.values() for n in counts),
        incomplete_block_reports=sum(n % 10 for counts in settings.values() for n in counts),
        anchor_subsets_one_through_six=anchor_subsets,
        exhaustive_role_input_count=2 * complete * (1 + 2 * anchor_subsets))


def inspect_node(node, source_revision, cached):
    name, expected_bytes = TAG_LOGS[node - 1]
    url = preflight.IDLAB_SHARE + "/download?" + urlencode(dict(path=f"/Raw data/Node {node}", files=name))
    print(f"Checking original tag log for node {node}", flush=True)
    if cached:
        with gzip.open(pilot.audit.located(cached["records_path"]), "rt") as stream:
            tag = json.load(stream)
        if cached["original_source"] != dict(url=url, bytes=tag["bytes"], sha256=tag["sha256"]) or tag["bytes"] != expected_bytes:
            raise ValueError(f"node {node} retained source identity changed")
        tag["slot_continuity"] = slot_continuity(tag["records"])
    else:
        with urllib.request.urlopen(url, context=audit.CONTEXT, timeout=40) as response:
            if response.status != 200 or int(response.headers.get("Content-Length", "0")) != expected_bytes:
                raise ValueError(f"node {node} original tag log size changed")
            tag = read_tag(response, expected_bytes)
    if node == 1 and tag["sha256"] != "3a57cbd3d361f95f27a3bb6de3a29348f2e3cd6bb9ab50960407ea21e65ca69c":
        raise ValueError("previously checked Node 1 source changed")
    with (admission.FOLDER / f"office_lab_node_{node:02d}_short.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    comparison = compare_compact(tag, rows)
    populated = [block for block in tag["blocks"] if block["records"]]
    identities = [(tuple(block["peers"]), tuple(sorted(block["settings"].items()))) for block in populated]
    errors = []
    if comparison["identity_mismatches"] or comparison["setting_mismatches"]:
        errors.append("compact-file identities or radio settings differ")
    if not tag["slot_continuity"]["consecutive_modulo_65536"]:
        errors.append("tag slots are not consecutive after counter unwrapping")
    if any(len(block["peers"]) != 1 for block in populated) or len(set(identities)) != len(identities):
        errors.append("setting blocks mix peers or repeat")
    peer_counts = dict(sorted(Counter(peer for block in populated for peer in block["peers"]).items()))
    if peer_counts != {n: 72 for n in range(1, 16)}:
        errors.append("the complete 15 by 72 schedule is not verified")
    path = OUTPUT.with_name(f"idlab-node{node}-tag-records.json.gz")
    temporary = path.with_suffix(path.suffix + ".partial")
    with gzip.GzipFile(filename=str(temporary), mode="wb", mtime=0) as stream:
        stream.write(json.dumps(tag, separators=(",", ":")).encode())
    temporary.replace(path)
    slots = {int(row["slot_ID_tag"]) for row in rows}
    return dict(node=node, source_revision=source_revision,
        original_source=dict(url=url, bytes=tag["bytes"], sha256=tag["sha256"]),
        records_path=pilot.audit.name(path), records_sha256=pilot.identity(path),
        tag_records=len(tag["records"]), self_records=sum(row[1] == node for row in tag["records"]),
        comparison=comparison, validation_errors=errors, slot_continuity=tag["slot_continuity"],
        peer_setting_blocks=peer_counts,
        setting_blocks=len(tag["blocks"]),
        terminations=dict(Counter(block["termination"] or "none" for block in tag["blocks"])),
        closing_exchange_without_compact_row=sum(r[1] != node and r[5] is not None and i not in slots
                                                for i, r in enumerate(tag["records"], 1)))


def save_progress(nodes, split, revision, failures):
    nodes = sorted(nodes, key=lambda row: row["node"])
    comparison = {key: sum(node["comparison"][key] for node in nodes)
                  for key in ("compact_rows", "matched_compact_rows", "tag_records_without_compact_row")}
    comparison.update({key: [dict(node=row["node"], **mismatch) for row in nodes for mismatch in row["comparison"][key]]
                       for key in ("identity_mismatches", "setting_mismatches")})
    checked_nodes = [row["node"] for row in nodes if not row["validation_errors"]]
    result = dict(schema="solid.external.idlab-attempts.v2",
        status="complete-original-tag-audit" if len(checked_nodes) == 15 else "partial-original-log-audit",
        source_revision=revision, split_sha256=pilot.identity(idlab_split.OUTPUT),
        sources={str(Path(__file__).resolve().relative_to(pilot.ROOT)): pilot.identity(Path(__file__))},
        checked_tag_nodes=checked_nodes, pending_tag_nodes=[n for n in range(1, 16) if n not in checked_nodes],
        nodes=nodes, failures=failures, comparison=comparison,
        tag_records=sum(row["tag_records"] for row in nodes), self_records=sum(row["self_records"] for row in nodes),
        original_bytes=sum(row["original_source"]["bytes"] for row in nodes),
        interpretation="Each verified node matches unwrapped compact slots, peers and three exchange timestamps. Missing compact records are not automatically radio failures. Receiver-side reconstruction remains unchecked.",
        population_candidates=population_counts(split),
        positioning_solves=0, models_fitted=0, availability=None, detection_range_cm=None)
    result["sources"].update({row["records_path"]: row["records_sha256"] for row in nodes})
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT, *(pilot.audit.located(row["records_path"]) for row in nodes)])
    print(f"Saved {len(nodes)} / 15 original tag logs. Verified {len(checked_nodes)}.", flush=True)


def prepare():
    if pilot.audit.uncommitted():
        raise ValueError("commit the audit source before execution")
    checked = pilot.evidence.read(admission.OUTPUT)
    for name, expected in checked["sources"].items():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab source")
    split = pilot.evidence.read(idlab_split.OUTPUT)
    pilot.evidence.official_scorer.require_hash(admission.OUTPUT, split["admission_sha256"], "IDLab split source")
    revision = pilot.audit.revision()
    previous = pilot.evidence.read(OUTPUT) if OUTPUT.exists() else {}
    retained = {row["node"]: row for row in previous.get("nodes", [])}
    if previous and previous["split_sha256"] != pilot.identity(idlab_split.OUTPUT):
        raise ValueError("retained tag audit belongs to a different partition")
    for node in retained.values():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(node["records_path"]), node["records_sha256"], "retained tag records")
    nodes = []
    failures = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(inspect_node, node, revision, retained.get(node)): node for node in range(1, 16)}
        for future in as_completed(jobs):
            node = jobs[future]
            try:
                nodes.append(future.result())
            except Exception as error:
                failures[str(node)] = str(error)
                print(f"Node {node} check failed. {error}", flush=True)
            save_progress(nodes, split, revision, failures)
    review = [row["node"] for row in nodes if row["validation_errors"]]
    if failures or review:
        raise ValueError(f"original tag audit needs review. Download failures {sorted(failures)}, data checks {review}")


if __name__ == "__main__":
    prepare()
