"""Check all published compact IDLab range files without fitting an estimator."""

import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

import audit
import pilot
import static_preflight as preflight


OUTPUT = audit.DATA / "idlab-admission.json"
FOLDER = audit.DATA / "sources/idlab"


def timestamp_ranges_mm(frame):
    times = {i: frame[f"T{i}"].to_numpy(dtype=float) for i in range(1, 7)}
    if not all(np.isfinite(values).all() for values in times.values()):
        raise ValueError("nonfinite IDLab timestamps")
    ra = (times[4] - times[1]) % 2**40
    rb = (times[6] - times[3]) % 2**40
    da = (times[5] - times[4]) % 2**40
    db = (times[3] - times[2]) % 2**40
    if ((ra + rb + da + db) == 0).any():
        raise ValueError("zero IDLab ranging interval")
    return (ra * rb - da * db) / (ra + rb + da + db) * 299792458000 / (499200000 * 128)


def inspect_file(frame, node):
    result = preflight.idlab_counts(frame, node)
    positive = np.isfinite(frame.distance) & (frame.distance > 0)
    delta = timestamp_ranges_mm(frame)[positive] - frame.loc[positive, "distance"].to_numpy()
    if not len(delta) or (delta < 0).any() or (delta >= 1).any():
        raise ValueError("positive IDLab ranges do not reproduce as truncated millimetres")
    invalid = []
    for offset in np.flatnonzero(~positive):
        row = frame.iloc[offset]
        invalid.append(dict(csv_line=int(offset) + 2, publisher_row=int(row["Unnamed: 0"]),
            peer=int(row.anchor_ID_anchor), distance_native=float(row.distance) if np.isfinite(row.distance) else None))
    return dict(**result, positive_ranges=len(delta), timestamp_mm_delta_bounds=[float(delta.min()), float(delta.max())],
                invalid_ranges=invalid)


def check_publisher(frames, aggregate, coordinates):
    populated = aggregate.dropna(subset=["Average distance error anchor"])
    coverage = {node: np.zeros(len(frame), dtype=np.int32) for node, frame in frames.items()}
    powers, settings, mean_deltas, std_deltas = set(), set(), [], []
    identity_columns = dict(anchor_ID_anchor="anchor", prf="PRF", channel="channel",
                            bitrate="data rate", preamble="preamble")
    for _, row in populated.iterrows():
        node, peer = int(row.tag), int(row.anchor)
        start, stop = int(row.startindex), int(row.stopindex)
        if node not in frames or start < 0 or stop < start or stop >= len(frames[node]):
            raise ValueError("invalid publisher row interval")
        coverage[node][start:stop + 1] += 1
        records = frames[node].iloc[start:stop + 1]
        if any(not (records[key] == row[value]).all() for key, value in identity_columns.items()):
            raise ValueError("publisher interval mixes link settings")
        if len(records[preflight.SETTINGS].drop_duplicates()) != 1:
            raise ValueError("publisher interval mixes power settings")
        true_distance = np.linalg.norm(coordinates.loc[node].to_numpy() - coordinates.loc[peer].to_numpy())
        errors = abs(records.distance - true_distance)
        mean_delta = abs(float(errors.mean() - row["Average distance error anchor"]))
        std = errors.std(ddof=1)
        expected = row["Stde distance error anchor"]
        if pd.isna(std) != pd.isna(expected):
            raise ValueError("publisher standard deviation availability differs")
        std_delta = 0.0 if pd.isna(std) else abs(float(std - expected))
        if mean_delta > 1e-6 or std_delta > 1e-6:
            raise ValueError("publisher absolute range-error statistics do not reproduce")
        mean_deltas.append(mean_delta)
        std_deltas.append(std_delta)
        powers.add((int(row.channel), int(row.PRF), str(records.txpower_tag.iloc[0]),
                    str(records.txpower_anchor.iloc[0]), float(row.TXpower)))
        settings.add((int(row.channel), int(row.PRF), int(row["data rate"]), int(row.preamble), float(row.TXpower)))
    if not len(populated) or any(not (counts == 1).all() for counts in coverage.values()):
        raise ValueError("publisher intervals must cover every compact-file row exactly once")
    return dict(populated_summaries=len(populated), covered_rows=sum(len(f) for f in frames.values()),
        maximum_mean_difference=max(mean_deltas), maximum_standard_deviation_difference=max(std_deltas),
        observed_configurations=len(settings), power_mapping=[dict(channel=c, prf=p, tag_code=t,
            anchor_code=a, publisher_gain_db=g) for c, p, t, a, g in sorted(powers)],
        interpretation="Publisher means and sample standard deviations use absolute ranging errors, including the invalid rows. They are not positioning results.")


def prepare():
    if OUTPUT.exists():
        raise ValueError("retained IDLab admission evidence must not be overwritten")
    sources, frames = [], {}
    for node in range(1, 16):
        name = f"office_lab_node_{node:02d}_short.csv"
        url = preflight.IDLAB_SHARE + "/download?" + urlencode(dict(path="/Processed files (csv)", files=name))
        path = audit.source(url, "idlab/" + name, 40_000_000)
        sources += [path, path.with_name(path.name + ".source.json")]
        frames[node] = pd.read_csv(path)
    for name in ("groundtruth9.csv", "Experiment description.md", "dataset-card.md",
                 "publisher-aggregate/README.md", "publisher-aggregate/groundtruth9.csv",
                 "publisher-aggregate/office_lab_allsettings2.csv"):
        path = FOLDER / name
        manifest = path.with_name(path.name + ".source.json")
        pilot.evidence.official_scorer.require_hash(path, json.loads(manifest.read_text())["sha256"], "IDLab source")
        sources += [path, manifest]
    sources += [Path(__file__), Path(preflight.__file__), Path(audit.__file__)]
    coordinates = pd.read_csv(FOLDER / "groundtruth9.csv", header=None)
    if (coordinates.shape != (15, 4) or not np.isfinite(coordinates).all().all()
            or coordinates[0].tolist() != list(range(1, 16))):
        raise ValueError("invalid IDLab coordinates")
    if (FOLDER / "groundtruth9.csv").read_bytes() != (FOLDER / "publisher-aggregate/groundtruth9.csv").read_bytes():
        raise ValueError("publisher coordinate copies differ")
    files = [inspect_file(frame, node) for node, frame in frames.items()]
    aggregate = pd.read_csv(FOLDER / "publisher-aggregate/office_lab_allsettings2.csv")
    published = check_publisher(frames, aggregate, coordinates.set_index(0))
    grouped = pd.concat([frame.assign(node=node) for node, frame in frames.items()], ignore_index=True)
    counts = grouped.groupby(["node", "anchor_ID_anchor", *preflight.SETTINGS], dropna=False).size()
    report = dict(schema="solid.external.idlab-admission.v1", status="protocol-decision-required",
        models_fitted=0, positioning_methods_executed=0, raw_nodes_checked=len(files),
        recorded_ranges=sum(row["rows"] for row in files), valid_ranges=sum(row["positive_ranges"] for row in files),
        invalid_ranges=sum(row["invalid_distance_rows"] for row in files), files=files, publisher_check=published,
        range_units="mm", coordinate_units="mm",
        unit_evidence="Every positive range equals truncated ADS-TWR millimetres from T1 to T6. All publisher absolute-error summaries reproduce from the unchanged coordinate CSV in the same units.",
        timestamp_reference="https://forum.qorvo.com/uploads/short-url/x34DrF7EW5fQP9wY3aNESqPKz8z.pdf",
        paper_reference="https://arxiv.org/html/2210.15498#S5.SS2",
        acquisition_scope="All 15 compact CSVs. Full CIR CSVs, original serial logs and processing exclusions are not audited.",
        recorded_directed_links=len(grouped[["node", "anchor_ID_anchor"]].drop_duplicates()),
        recorded_link_settings=len(counts), link_settings_with_ten_records=int((counts >= 10).sum()),
        largest_link_setting_count=int(counts.max()), link_settings_above_500_records=int((counts > 500).sum()),
        preparation_rules=["Keep original files unchanged. Any invalid-range exclusion must be shared by every method and counted.",
            "Use actual publisher-mapped power gains. Do not relabel every high-power setting as 10.5 dB.",
            "Do not derive failed attempts or a detection probability by subtracting successful rows from 500.",
            "No evaluation position or measurement may enter its fitted profile."],
        remaining_decisions=["Choose disjoint measurement blocks for a same-layout test or device-disjoint folds with smaller networks.",
            "Verify original-log processing and attempt identities before fitting availability or interpreting absent links.",
            "Confirm the declared dataset-card license covers separately hosted files before redistribution.",
            "Freeze invalid-input accounting, populations, profiles and method demonstrations before a positioning campaign."],
        sources={audit.name(p): pilot.identity(p) for p in sources})
    pilot.evidence.write(OUTPUT, report)
    pilot.register_files([OUTPUT, *[p for p in sources if p.is_relative_to(audit.DATA)]])
    print(json.dumps({key: report[key] for key in ("status", "raw_nodes_checked", "recorded_ranges", "invalid_ranges")}, indent=2))


if __name__ == "__main__":
    prepare()
