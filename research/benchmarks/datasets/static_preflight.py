"""Inventory static benchmark inputs without fitting or executing an estimator."""

from importlib.metadata import version
import json
import math
import subprocess
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

import audit
import pilot


OUTPUT = audit.DATA / "static-preflight.json"
IDLAB_REVISION = "d813ce72783241142102723b77c09c0f5fa165d1"
IDLAB_SHARE = "https://cloud.ilabt.imec.be/index.php/s/rFWMHfD5WaD6HWy"
SETTINGS = ["channel", "bitrate", "preamble", "prf", "txpower_tag", "txpower_anchor"]
EVENT_KEY = ["anchor_ID_anchor", *SETTINGS, "slot_ID_tag", "slot_ID_anchor", *[f"T{i}" for i in range(1, 7)]]


def tiers_streams(ranges):
    nodes = sorted({n for pair in ranges for n in pair})
    pairs = list(audit.combinations(nodes, 2))
    if not pairs or any(p not in ranges for p in pairs):
        raise ValueError("expected one forward stream for every unordered pair")
    streams = {}
    for pair in pairs:
        rows = np.asarray(ranges[pair])
        if rows.ndim != 2 or rows.shape[1] != 5 or not len(rows):
            raise ValueError("invalid TIERS stream shape")
        if not np.isfinite(rows).all() or (rows[:, 1] <= 0).any():
            raise ValueError("invalid TIERS report requires an explicit input policy")
        if len(np.unique(rows[:, 2:], axis=0)) != len(rows):
            raise ValueError("duplicate forward message identity")
        times = rows[:, 2] + rows[:, 3] / 1e9
        if (np.diff(times) <= 0).any():
            raise ValueError("unordered forward timestamps")
        streams[pair] = rows
    return streams


def tiers_blocks(streams, samples):
    """One sample is the first report of the matching ten-report static block."""
    if samples not in (1, 10):
        raise ValueError("expected one or ten real reports")
    for block in range(min(map(len, streams.values())) // 10):
        yield {pair: rows[block * 10:block * 10 + samples] for pair, rows in streams.items()}


def tiers_inventory(ranges):
    streams = tiers_streams(ranges)
    counts = {f"{a}-{b}": len(rows) for (a, b), rows in streams.items()}
    tracks = {}
    blocks = min(counts.values()) // 10
    for samples in (1, 10):
        used = blocks * samples
        tracks[str(samples)] = dict(
            blocks=blocks, used_reports=used * len(streams),
            unused_reports=sum(counts.values()) - used * len(streams),
            unused_by_link={pair: count - used for pair, count in counts.items()},
            block_stride=10, samples_per_block=samples)
    return dict(forward_reports=sum(counts.values()), reverse_reports=sum(len(r) for p, r in ranges.items() if p[0] > p[1]),
                forward_counts=counts, tracks=tracks)


def idlab_counts(frame, node):
    required = set(EVENT_KEY) | {"anchor_ID_tag", "distance"}
    if required - set(frame.columns) or frame[EVENT_KEY].isna().any().any():
        raise ValueError("missing IDLab identity columns or values")
    if not frame.anchor_ID_tag.equals(frame.anchor_ID_anchor):
        raise ValueError("IDLab endpoint columns disagree")
    peers = sorted(int(v) for v in frame.anchor_ID_anchor.unique())
    if node in peers or not set(peers) <= set(range(1, 16)):
        raise ValueError("invalid IDLab peer identity")
    unique = frame.drop_duplicates(EVENT_KEY)
    if len(frame[EVENT_KEY + ["distance"]].drop_duplicates()) != len(unique):
        raise ValueError("conflicting ranges for one recorded IDLab event")
    counts = frame.groupby(["anchor_ID_anchor", *SETTINGS], dropna=False).size()
    return dict(node=node, rows=len(frame), columns=list(frame.columns), peers=peers,
                absent_peer_ids=sorted(set(range(1, 16)) - {node} - set(peers)),
                settings_with_records=len(frame[SETTINGS].drop_duplicates()), link_settings_with_records=len(counts),
                rows_per_present_link_setting=dict(min=int(counts.min()), median=float(counts.median()), max=int(counts.max())),
                duplicate_event_keys=len(frame) - len(unique),
                invalid_distance_rows=int((~np.isfinite(frame.distance) | (frame.distance <= 0)).sum()),
                missing_values={key: int(n) for key, n in frame.isna().sum().items() if n},
                explicit_failure_column_present=any("fail" in c.lower() or "status" in c.lower() for c in frame.columns))


def workbook_inventory(path):
    # The publisher's identity sheet is small. Read cells without changing the workbook.
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", ns)
        if len(sheets) != 1 or sheets[0].attrib["name"] != "OfficeLab Jan-2020":
            raise ValueError("unexpected IDLab identity workbook")
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    cells = {c.attrib["r"]: c for c in sheet.findall(".//s:sheetData/s:row/s:c", ns)}
    ids = [int(cells[f"B{r}"].findtext("s:v", namespaces=ns)) for r in range(2, 17)]
    if ids != list(range(1, 16)):
        raise ValueError("unexpected IDLab workbook node identities")
    populated = [f"{c}{r}" for c in "DEF" for r in range(2, 17)
                 if f"{c}{r}" in cells and list(cells[f"{c}{r}"])]
    return dict(sheet=sheets[0].attrib["name"], identity_range="B2:B16", node_ids=ids,
                coordinate_range="D2:F16", populated_coordinate_cells=populated)


def inspect_idlab():
    sources = []
    readme = audit.source(f"https://huggingface.co/datasets/iwine-ugent/uwb-positioning-datasets/raw/{IDLAB_REVISION}/README.md",
                          "idlab/dataset-card.md", 100_000)
    sources.append(readme)
    for name in ("Experiment description.md", "Readme.md", "9thFloorAnchorsLayout.xlsx", "groundtruth9.csv",
                 "office_lab_node_01_short.csv", "office_lab_node_02_short.csv"):
        directory = "/Processed files (csv)" if name.startswith("office_") else "/"
        sources.append(audit.source(IDLAB_SHARE + "/download?" + urlencode(dict(path=directory, files=name)),
                                    "idlab/" + name, 20_000_000 if directory != "/" else 100_000))
    folder = audit.DATA / "sources/idlab"
    coordinates = pd.read_csv(folder / "groundtruth9.csv", header=None).to_numpy()
    if coordinates.shape != (15, 4) or not np.isfinite(coordinates).all() or not np.array_equal(coordinates[:, 0], np.arange(1, 16)):
        raise ValueError("invalid IDLab coordinate CSV")
    description = (folder / "Experiment description.md").read_text()
    if "maximumtries = 505" not in description or "maximumerrors = 20" not in description:
        raise ValueError("IDLab collection settings changed")
    report = dict(card_revision=IDLAB_REVISION, share_url=IDLAB_SHARE,
        scope="First two node short CSVs by identifier. No full-CSV or raw-log audit.",
        models_fitted=0, positioning_methods_executed=0, raw_nodes_checked=2, documented_nodes=15,
        documented_radio_configurations=72, maximumtries=505, maximumerrors=20,
        files=[idlab_counts(pd.read_csv(folder / f"office_lab_node_{n:02d}_short.csv"), n) for n in (1, 2)],
        workbook=workbook_inventory(folder / "9thFloorAnchorsLayout.xlsx"), coordinate_csv_rows=len(coordinates),
        coordinate_csv_units_confirmed=False, distance_units_confirmed=False,
        decision="Promising static network candidate. Not admitted for positioning or missingness fitting.",
        remaining_checks=["Verify range and coordinate units and PHY power-code mapping from publisher processing code",
            "Verify event identity, processing exclusions and all 15 node files",
            "Recover attempt and early-stop records before interpreting missing links",
            "Fix independent training or an explicitly transferred-profile track",
            "Confirm the dataset-card license also covers the separately hosted raw files"])
    return report, sources + [p.with_name(p.name + ".source.json") for p in sources]


def prepare():
    if OUTPUT.exists():
        raise ValueError("retained static preflight must not be overwritten")
    checked_path = audit.DATA / "input-audit.json"
    sources = [checked_path, pilot.HERE / "static_preflight.py"]
    records = []
    for record in pilot.evidence.read(checked_path)["tiers"]:
        path = audit.DATA / record["file"]
        pilot.evidence.official_scorer.require_hash(path, record["sha256"], "static TIERS input")
        ranges, _ = audit.tiers_events(path)
        result = tiers_inventory(ranges)
        records.append(dict(file=record["file"], **result))
        sources.append(path)
    roles = sum(math.comb(8, k) for k in range(1, 7))
    population = {str(n): dict(AF=sum(r["tracks"][str(n)]["blocks"] for r in records)) for n in (1, 10)}
    for row in population.values():
        row.update(AAC=row["AF"] * roles, ANC=row["AF"] * roles)
    idlab, paths = inspect_idlab()
    sources += paths
    result = dict(schema="solid.external.static-preflight.v1", source_revision=audit.revision(), models_fitted=0, positioning_methods_executed=0,
        protocol_status="Input inventory complete. Proposed population is not an execution freeze.",
        tiers=dict(revision=audit.TIERS_REVISION, recordings=records, proposed_inputs_per_track=population,
            roles_per_block=roles, sample_block_rule="Complete disjoint ten-report blocks up to the shortest stream. The one-sample track takes each block's first report.",
            timing="Static common-rank blocks. Not synchronized physical ranging rounds.",
            unused_policy="Faster-link remainder and incomplete blocks are counted, not interpreted as failed ranges",
            complete_raw_utilization=False, independent_acquisitions=3,
            run_gates=["Approve the complete-block population and disclosed unused reports",
                "Prepare explicit TIERS ANC profiles",
                "Verify all modes on bounded demonstrations with identical input information for each method",
                "Freeze profiles, predictor, scorer, roster, input manifest and execution budget"]),
        idlab=idlab, sources={audit.name(p): pilot.identity(p) for p in sources},
        packages={name: version(name) for name in ("numpy", "pandas", "rosbags")})
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT, *[p for p in sources if p.is_relative_to(audit.DATA)]])
    print(json.dumps(dict(path=str(OUTPUT), proposed_inputs_per_track=population,
                         idlab_checked_rows=sum(r["rows"] for r in idlab["files"])), indent=2))


if __name__ == "__main__":
    prepare()
