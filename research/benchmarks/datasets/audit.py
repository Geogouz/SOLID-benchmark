"""Inspect external inputs without fitting or running a positioning method."""

import hashlib
import html
import io
import json
import os
import re
import ssl
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader


ROOT = Path(__file__).resolve().parents[3]
# The data folder, research/data under the repository unless the variable SOLID_DATA names another one in the same layout.
DATA_ROOT = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
DATA = DATA_ROOT / "external-benchmarks"
# A reader's reproduction (SOLID_READER=1) skips the gates that only prove the research process: clean-tree refusals,
# commits of our history, the private data inventory and the records of the Bern evaluation runs.
READER = os.environ.get("SOLID_READER") == "1"
TIERS_REVISION = "5dceab32c4d2b1dc32016a80a5b2406d973cea2d"
MILUV_REVISION = "399ad46f4e6e88c3c98b40aae33fe8e2c6719cff"
TIERS = f"https://raw.githubusercontent.com/TIERS/uwb-relative-localization-dataset/{TIERS_REVISION}/"
MILUV = f"https://raw.githubusercontent.com/decargroup/miluv/{MILUV_REVISION}/"
RELEASE = "https://api.figshare.com/v2/articles/28386041/versions/1"
CONTEXT = ssl.create_default_context(
    cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def name(path):
    """The name a record keeps for a file, its path in the repository, and for a file of the data folder its path under
    research/data wherever SOLID_DATA puts the folder, so that records read the same on every machine."""
    path = Path(path)
    if not path.is_relative_to(ROOT) and path.is_relative_to(DATA_ROOT):
        return str("research/data" / path.relative_to(DATA_ROOT))
    return str(path.relative_to(ROOT))


def located(name):
    """The file of a name that a record keeps (name)."""
    parts = Path(name).parts
    return DATA_ROOT.joinpath(*parts[2:]) if parts[:2] == ("research", "data") else ROOT / name


def revision():
    """The commit of the checkout. A reader's copy (SOLID_READER=1) has another history or none, so it records the
    commit git names or None."""
    if not READER:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    answer = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return answer.stdout.strip() if answer.returncode == 0 else None


def uncommitted():
    """Whether the checkout holds changes that are not committed, which a step of the research process refuses. A
    reader's copy (SOLID_READER=1) is not a commit of the research record, and nothing is refused."""
    return not READER and bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())


def fetch(url, limit, byte_range=None):
    headers = {"User-Agent": "SOLID external input audit"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), context=CONTEXT, timeout=40
    ) as response:
        if byte_range is not None:
            expected = f"bytes {byte_range[0]}-{byte_range[1]}/"
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise ValueError("Server did not honor the requested byte range")
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Download exceeds the audit byte limit")
    if byte_range is not None and len(payload) != byte_range[1] - byte_range[0] + 1:
        raise ValueError("Incomplete byte range")
    return payload


def source(url, relative, limit=32_000_000):
    path = DATA / "sources" / relative
    manifest = path.with_name(path.name + ".source.json")
    if path.exists():
        record = json.loads(manifest.read_text())
        if record["url"] != url or sha256(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Source identity mismatch for {relative}")
    else:
        payload = fetch(url, limit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest.write_text(json.dumps({"url": url, "bytes": len(payload), "sha256": sha256(payload)}, indent=2) + "\n")
    return path


class RangeFile(io.RawIOBase):
    """Read ZIP members without downloading unrelated images and ROS bags."""

    def __init__(self, url, size, budget=100_000_000):
        self.url, self.size, self.position = url, size, 0
        self.budget, self.downloaded = budget, 0
        self.blocks = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: self.size}
        if whence not in bases or bases[whence] + offset < 0:
            raise ValueError("Invalid seek")
        self.position = bases[whence] + offset
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if self.position >= end:
            return b""
        key = (self.position, end - 1)
        if key not in self.blocks:
            count = end - self.position
            if self.downloaded + count > self.budget:
                raise ValueError("Remote ZIP exceeds the audit byte budget")
            self.blocks[key] = fetch(self.url, count, key)
            self.downloaded += count
        self.position = end
        return self.blocks[key]


def tiers_events(path):
    ranges, poses = defaultdict(list), defaultdict(list)
    with AnyReader([path]) as reader:
        connections = [c for c in reader.connections if
                       re.fullmatch(r"/uwb/tof/n_\d+/n_\d+/distance", c.topic)
                       or re.fullmatch(r"/vrpn_client_node/turtlebot\d+/pose", c.topic)]
        for connection, timestamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            time = timestamp / 1e9
            if connection.msgtype == "sensor_msgs/msg/Range":
                pair = tuple(map(int, re.findall(r"n_(\d+)", connection.topic)))
                stamp = message.header.stamp
                ranges[pair].append((time, float(message.range), stamp.sec, stamp.nanosec, message.header.seq))
            else:
                node = int(re.search(r"turtlebot(\d+)", connection.topic)[1])
                p = message.pose.position
                poses[node].append((time, p.x, p.y, p.z))
    return {p: np.asarray(rows) for p, rows in ranges.items()}, {
        n: np.asarray(rows) for n, rows in poses.items()}


def summarize_tiers(path):
    ranges, poses = tiers_events(path)
    counts = {f"{a}-{b}": len(rows) for (a, b), rows in sorted(ranges.items())}
    coordinates = {str(n): np.median(rows[:, 1:], axis=0).tolist() for n, rows in sorted(poses.items())}
    spans = np.concatenate([np.sort(r[:, 0])[9:] - np.sort(r[:, 0])[:-9] for r in ranges.values()])
    unordered = {tuple(sorted(p)) for p in ranges}
    reverse_matches, reverse_comparisons = 0, 0
    for a, b in sorted(unordered):
        if (a, b) not in ranges or (b, a) not in ranges:
            continue
        left, right = ranges[(a, b)], ranges[(b, a)]
        idx = np.searchsorted(right[:, 0], left[:, 0])
        for row, nearest in zip(left, idx):
            candidates = right[max(0, nearest - 1):nearest + 1]
            if len(candidates):
                match = candidates[np.argmin(np.abs(candidates[:, 0] - row[0]))]
                reverse_matches += int(abs(match[0] - row[0]) <= 0.002 and match[1] == row[1])
                reverse_comparisons += 1
    positions = np.asarray(list(coordinates.values()))
    horizontal = [np.linalg.norm(positions[a, :2] - positions[b, :2]) for a, b in combinations(range(len(positions)), 2)]
    spatial = [np.linalg.norm(positions[a] - positions[b]) for a, b in combinations(range(len(positions)), 2)]
    return {
        "file": str(path.relative_to(DATA)), "sha256": sha256(path.read_bytes()),
        "range_nodes": sorted({n for pair in ranges for n in pair}),
        "pose_nodes": sorted(poses), "directed_link_counts": counts,
        "range_messages": sum(counts.values()),
        "range_median_native": float(np.median(np.concatenate(list(ranges.values()))[:, 1])),
        "pose_median_native": coordinates,
        "pose_span_native": {str(n): np.ptp(rows[:, 1:], axis=0).tolist() for n, rows in sorted(poses.items())},
        "duration_seconds": float(max(r[:, 0].max() for r in ranges.values()) - min(r[:, 0].min() for r in ranges.values())),
        "unordered_links": len(unordered),
        "ten_messages_span_seconds": quantiles(spans),
        "equal_reverse_range_within_2ms": reverse_matches,
        "reverse_comparisons": reverse_comparisons,
        "max_spatial_minus_horizontal_reference_cm": float(100 * np.max(np.asarray(spatial) - horizontal)),
    }


EVENT_KEY = ["timestamp", "from_id", "to_id", "tx1_raw", "rx1_raw"]
RANGE_COLUMNS = EVENT_KEY + ["range_raw", "tx2_raw", "tx3_raw", "rx2_raw", "rx3_raw"]


def miluv_ranges(paths):
    frames = [pd.read_csv(path, usecols=RANGE_COLUMNS) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    if not np.isfinite(merged[EVENT_KEY].to_numpy()).all():
        raise ValueError("Missing physical event identity")
    unique = merged.drop_duplicates(EVENT_KEY)
    if len(merged.drop_duplicates()) != len(unique):
        raise ValueError("Conflicting copies of the same MILUV ranging event")
    return unique.sort_values("timestamp", kind="stable").reset_index(drop=True), [len(f) for f in frames]


def quantiles(values):
    values = np.asarray(values)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite audit distribution")
    return {key: float(value) for key, value in zip(
        ("min", "median", "p90", "max"), np.quantile(values, [0, 0.5, 0.9, 1]))}


def window_counts(frame, width, links):
    start, stop = float(frame.timestamp.min()), float(frame.timestamp.max())
    windows = max(1, int(np.ceil((stop - start) / width)))
    counts = np.zeros((windows, len(links)), dtype=int)
    for j, (a, b) in enumerate(links):
        times = frame.loc[(frame.from_id == a) & (frame.to_id == b), "timestamp"].to_numpy()
        bins = np.minimum(((times - start) / width).astype(int), windows - 1)
        np.add.at(counts[:, j], bins, 1)
    return {"window_seconds": width, "windows": windows, "directed_links": len(links),
            "link_windows": int(counts.size),
            "with_one_or_more": int((counts >= 1).sum()),
            "with_ten_or_more": int((counts >= 10).sum()),
            "missing_link_windows": int((counts == 0).sum())}


def summarize_miluv():
    root = DATA / "sources/miluv/default_3_random2_0"
    paths = [root / robot / "uwb_range.csv" for robot in ("ifo001", "ifo002", "ifo003")]
    frame, source_counts = miluv_ranges(paths)
    counts = frame.groupby(["from_id", "to_id"]).size()
    links = list(counts.index)
    tags = sorted(set(frame.from_id))
    observed_peer_pairs = {tuple(sorted((a, b))) for a, b in links if b in tags}
    spans = []
    for _, group in frame.groupby(["from_id", "to_id"]):
        times = group.timestamp.to_numpy()
        spans.extend(times[9:] - times[:-9])
    motion = {}
    for robot in ("ifo001", "ifo002", "ifo003"):
        poses = pd.read_csv(root / robot / "mocap.csv")
        xyz = poses[[f"pose.position.{axis}" for axis in "xyz"]].to_numpy()
        times = poses.timestamp.to_numpy()
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(times)
        xyz, times = xyz[valid], times[valid]
        motion[robot] = {"body_z_span_m": float(np.ptp(xyz[:, 2]))}
        for interval in (1.0, float(np.median(spans))):
            endpoints = np.searchsorted(times, times + interval)
            usable = endpoints < len(times)
            displacement = 100 * np.linalg.norm(xyz[endpoints[usable]] - xyz[usable], axis=1)
            motion[robot][f"{interval:.3f}_seconds"] = quantiles(displacement)
    header = pd.read_csv(paths[0], nrows=0).columns.tolist()
    return {
        "recording": "default_3_random2_0", "source_rows": source_counts,
        "rows_before_deduplication": sum(source_counts), "physical_events": len(frame),
        "duplicate_copies": sum(source_counts) - len(frame),
        "event_identity": EVENT_KEY,
        "invalid_raw_range_events": int((~np.isfinite(frame.range_raw) | (frame.range_raw <= 0)).sum()),
        "tag_ids": [int(n) for n in tags],
        "anchor_ids": sorted(int(n) for n in set(frame.to_id) - set(tags)),
        "directed_link_counts": {f"{a}-{b}": int(n) for (a, b), n in counts.items()},
        "observed_peer_pairs": [list(map(int, p)) for p in sorted(observed_peer_pairs)],
        "absent_peer_pairs": [list(map(int, p)) for p in combinations(tags, 2) if p not in observed_peer_pairs],
        "ten_samples_span_seconds": quantiles(spans),
        "window_availability": [window_counts(frame, width, links) for width in (1.0, 2.0)],
        "motion_diagnostic_only": motion,
        "excluded_columns": [c for c in header if c not in RANGE_COLUMNS],
        "predictor_columns": ["timestamp", "from_id", "to_id", "range_raw"],
    }


def coverage_section(bern_complete):
    tiers = DATA / "tiers-static-campaign/completion.json"
    bregar = DATA / "bregar-evaluation/results/summary.json"
    complete = {
        "TWR-CLOUD-Bern": bern_complete,
        "TIERS": tiers.exists(),
        "Bregar": bregar.exists() and json.loads(bregar.read_text())["status"] == "complete",
    }
    rows = [
        ("TWR-CLOUD-Bern", "AF, AAC and ANC with 1 and 10 samples.",
         "Full compatible corpus. Retrospective evaluation.", "bern-results"),
        ("TIERS", "139 common static blocks. All three modes with 1 and 10 samples.",
         "Close the preparation audit. Moving recordings need a separate protocol.", "tiers-matrix"),
        ("Bregar", "Four held environments. ANC with 1 and 10 samples.",
         "No peer measurements for AF or AAC.", "bregar-benchmark"),
        ("IDLab PHY", "All 15 compact node files checked. No positioning run.",
         "Fix the training split, invalid-input rule and attempt accounting.", "idlab-benchmark"),
        ("MILUV", "Moving nodes. Input audit only.",
         "Define a temporal experiment before execution.", "miluv-benchmark"),
        ("CTTC", "One documented tag position.",
         "Cannot test inter-agent pairwise geometry.", "cttc-benchmark"),
    ]
    states = {name: "Complete" if value else "Incomplete" for name, value in complete.items()}
    states.update({"IDLab PHY": "Admission pending", "MILUV": "Separate protocol needed", "CTTC": "Not applicable"})
    text = '<h2 id="external-coverage">Benchmark coverage</h2>'
    text += '<p>Complete refers to the declared input population, not every possible use of an archive. '
    text += 'No benchmark is excluded because a competitor wins.</p>'
    text += '<table class="coverage-table"><thead><tr><th>Dataset</th><th>Execution</th><th>Scope</th><th>Boundary or remaining work</th></tr></thead><tbody>'
    for name, scope, remaining, anchor in rows:
        text += (f'<tr><td><a href="#{anchor}">{name}</a></td><td>{states[name]}</td>'
                 f'<td>{scope}</td><td>{remaining}</td></tr>')
    return text + '</tbody></table>'


def input_section(report):
    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(
                f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows) + "</tbody></table>"

    tiers_rows = []
    for row in report["tiers"]:
        tiers_rows.append([
            Path(row["file"]).stem.split("_")[0], f'{row["range_messages"]:,}',
            row["unordered_links"], f'{row["ten_messages_span_seconds"]["median"]:.3f} s',
            f'{row["max_spatial_minus_horizontal_reference_cm"]:.3f} cm',
            f'{100 * row["equal_reverse_range_within_2ms"] / row["reverse_comparisons"]:.1f}%'
        ])
    miluv = report["miluv"]
    motion_rows = []
    for robot, values in miluv["motion_diagnostic_only"].items():
        key = next(k for k in values if k.endswith("_seconds") and k != "1.000_seconds")
        motion_rows.append([robot, f'{values["1.000_seconds"]["median"]:.1f} cm',
                            f'{values[key]["median"]:.1f} cm'])
    content = '<p>These are completed raw-input checks. They fit no model and are not accuracy experiments. See the <a href="#external-coverage">coverage table</a> for current results and remaining work.</p>'
    content += "<h2>TIERS raw-input checks</h2><p>Eight tracked nodes and all 28 unordered links occur in each acquisition. The last column checks how often the reverse stream reports the identical range within 2 ms. It is evidence of mirrored reports, not proof of a shared physical packet ID.</p>"
    content += table(["Configuration", "Range messages", "Unordered links", "Median span of ten messages", "Largest spatial versus horizontal reference gap", "Matching reverse reports"], tiers_rows)
    content += "<p>No configuration was removed for geometry or observability. Ten samples can be constructed from one reporting stream per pair without counting the reverse report as an additional sample. The input audit did not run either AF estimator.</p>"
    content += f'<h2>MILUV pilot checks</h2><p>The previously examined recording <code>default_3_random2_0</code> contains {miluv["rows_before_deduplication"]:,} rows across three robot files. These represent {miluv["physical_events"]:,} distinct recorded events. The {miluv["duplicate_copies"]:,} duplicate copies must not increase sample counts.</p>'
    content += f'<p>Ten consecutive samples on a directed link span a median of <strong>{miluv["ten_samples_span_seconds"]["median"]:.3f} seconds</strong>. No one-second or two-second link window in this pilot contains ten samples.</p>'
    content += table(["Robot", "Median movement over one second", "Median movement over the ten-sample interval"], motion_rows)
    content += "<p>The absent peer links are 10–11, 20–21 and 30–31. Each connects two tags mounted on the same robot. Their absence must not be interpreted as evidence that they are far apart. The raw-range adapter excludes the published ground-truth range, bias and pose columns.</p>"
    content += "<h2>Training and confirmation boundary</h2><p>The calibrated TIERS comparison uses Case 1 for fitting and all three Case 3 acquisitions for assessment. Case 4 recordings and several MILUV recordings were previously examined. MILUV confirmation recordings were not downloaded by this audit. Metadata inspection alone does not establish that a recording was never used in earlier project work.</p>"
    content += "<p>Fit target-specific profiles only after disjoint training and test acquisitions are fixed. Freeze the raw-input adapter, dimensional information, failure accounting and current method identities before measuring performance. Unknown range opportunities must not become negative evidence.</p>"
    content += '<p><a href="input-audit.json">Machine-readable input audit</a> · <a href="sources/miluv/default_3_random2_0.index.json">MILUV archive member inventory</a></p>'
    content += '<p>Sources include the <a href="https://arxiv.org/html/2305.08532v1">TIERS paper</a>, <a href="https://doi.org/10.25452/figshare.plus.28386041.v1">MILUV release</a>, <a href="https://zenodo.org/records/5996710">CTTC v2 release</a> and <a href="https://doi.org/10.1038/s41597-023-02639-5">Bregar paper</a>.</p>'
    return content


def inspect_sources():
    recordings = ["army_2023-03-24-20-09-29_v2", "corner_2023-03-24-20-18-25_v2", "diagonal_2023-03-24-20-13-40_v2"]
    reports = []
    for recording in recordings:
        relative = f"case_3/ros1/{recording}.bag"
        path = source(TIERS + relative, "tiers/" + relative)
        reports.append(summarize_tiers(path))
        print(recording, reports[-1]["range_messages"], "range messages", flush=True)
    for relative in ["LICENSE", "README.md", "case_1/final_data.csv"]:
        source(TIERS + relative, "tiers/" + relative)
    # MILUV is no dataset of the paper and the fetch script pins none of it, so a reader's audit covers TIERS alone.
    archive, entries, members, downloaded = (None, [], [], 0) if READER else miluv_pilot()
    roster = ["jin-2021", "ade-bp-2015", "fnl", "strong"]
    report = {"tiers_revision": TIERS_REVISION, "miluv_revision": MILUV_REVISION,
              "tiers": reports, "miluv_pilot_archive": archive,
              "miluv_member_count": len(entries), "miluv_extracted": members, "miluv_bytes_read": downloaded,
              "miluv": None if READER else summarize_miluv(),
              "method_roster": roster,
              "source_revision": revision(),
              "audit_source_sha256": sha256(Path(__file__).read_bytes()),
              "packages": {name: version(name) for name in ("rosbags", "numpy", "pandas", "scipy")},
              "positioning_methods_executed": 0, "models_fitted": 0}
    DATA.mkdir(parents=True, exist_ok=True)
    if not READER:
        (DATA / "sources/miluv/default_3_random2_0.index.json").write_text(json.dumps(entries, indent=2) + "\n")
    (DATA / "input-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not READER:
        print("MILUV member count", len(entries), "bytes read", downloaded, flush=True)
    for entry in entries:
        if entry["name"].endswith(".csv") and any(s in entry["name"] for s in ("uwb_range", "mocap", "time")):
            print(entry, flush=True)


def miluv_pilot():
    """The MILUV pilot archive's release entry, its member index and the extracted members, and the bytes it read."""
    for relative in ["LICENSE", "README.md", "config/experiments.csv", "config/uwb/anchors.yaml", "config/uwb/tags.yaml", "preprocess/process_uwb.py", "paper/calibrate_uwb.py"]:
        source(MILUV + relative, "miluv/devkit/" + relative)
    release = json.loads(source(RELEASE, "miluv/release-v1.json").read_text())
    # This recording was already used during earlier development.
    archive = next(f for f in release["files"] if f["name"] == "default_3_random2_0.zip")
    remote = RangeFile(archive["download_url"], archive["size"])
    with zipfile.ZipFile(remote) as zf:
        entries = [{"name": f.filename, "size": f.file_size, "compressed_size": f.compress_size,
                    "crc32": f"{f.CRC:08x}"} for f in zf.infolist()]
        members = []
        for robot in ("ifo001", "ifo002", "ifo003"):
            for name in ("uwb_range.csv", "mocap.csv"):
                member = f"default_3_random2_0/{robot}/{name}"
                info = zf.getinfo(member)
                path = DATA / "sources/miluv" / member
                record_path = path.with_name(path.name + ".source.json")
                if path.exists():
                    record = json.loads(record_path.read_text())
                    if record["sha256"] != sha256(path.read_bytes()) or record["crc32"] != info.CRC:
                        raise ValueError("Extracted MILUV member identity mismatch")
                else:
                    if info.file_size > 16_000_000:
                        raise ValueError("MILUV member exceeds the pilot size limit")
                    payload = zf.read(member)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                    record = {"url": archive["download_url"], "member": member,
                              "archive_md5": archive["computed_md5"], "crc32": info.CRC,
                              "sha256": sha256(payload), "bytes": len(payload)}
                    record_path.write_text(json.dumps(record, indent=2) + "\n")
                members.append(record)
    return archive, entries, members, remote.downloaded


if __name__ == "__main__":
    inspect_sources()
