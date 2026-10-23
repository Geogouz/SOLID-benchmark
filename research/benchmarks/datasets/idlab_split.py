"""Reserve disjoint IDLab records for a same-layout calibration and evaluation."""

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import idlab_admission as admission
import pilot
import static_preflight as preflight


OUTPUT = admission.OUTPUT.with_name("idlab-split.json")
INTERVAL_COLUMNS = ["tag", "anchor", "PRF", "channel", "data rate", "preamble",
                    "TXpower", "startindex", "stopindex"]


def partition_records(frames, intervals):
    if (intervals.startindex.isna() != intervals.stopindex.isna()).any():
        raise ValueError("incomplete publisher interval")
    coverage = {node: np.zeros(len(frame), dtype=np.int32) for node, frame in frames.items()}
    for frame in frames.values():
        if not np.array_equal(frame["Unnamed: 0"].to_numpy(), np.arange(len(frame))):
            raise ValueError("publisher row identity does not match source order")
    groups, identities = [], set()
    for _, row in intervals.dropna(subset=["startindex", "stopindex"]).sort_values(["tag", "startindex"]).iterrows():
        values = [row.tag, row.anchor, row.startindex, row.stopindex]
        if not all(np.isfinite(v) and v == int(v) for v in values):
            raise ValueError("noninteger publisher interval identity")
        node, peer, start, last = map(int, values)
        if node not in frames or peer not in frames or peer == node or start < 0 or last < start or last >= len(frames[node]):
            raise ValueError("invalid publisher interval")
        stop = last + 1
        records = frames[node].iloc[start:stop]
        for column, expected in (("anchor_ID_anchor", peer), ("anchor_ID_tag", peer),
                                 ("prf", row.PRF), ("channel", row.channel),
                                 ("bitrate", row["data rate"]), ("preamble", row.preamble)):
            if not records[column].eq(expected).all():
                raise ValueError("publisher interval mixes link settings")
        if len(records[preflight.SETTINGS].drop_duplicates()) != 1:
            raise ValueError("publisher interval mixes power settings")
        settings = records[preflight.SETTINGS].iloc[0].to_dict()
        identity = (node, peer, *settings.values())
        if identity in identities:
            raise ValueError("link setting occurs in separate publisher intervals")
        identities.add(identity)
        coverage[node][start:stop] += 1
        middle = start + (stop - start) // 2
        group = dict(node=node, peer=peer, settings=settings, publisher_gain_db=float(row.TXpower))
        for name, left, right in (("calibration", start, middle), ("evaluation", middle, stop)):
            # Assignment precedes validity filtering so exclusions cannot shift the boundary.
            part = frames[node].iloc[left:right]
            valid = np.isfinite(part.distance) & part.distance.gt(0)
            group[name] = dict(row_start=left, row_stop=right, recorded_ranges=len(part),
                               valid_ranges=int(valid.sum()),
                               invalid_publisher_rows=part.loc[~valid, "Unnamed: 0"].astype(int).tolist())
        groups.append(group)
    if not groups or any(not (counts == 1).all() for counts in coverage.values()):
        raise ValueError("publisher intervals must cover every source row exactly once")
    totals = {}
    for name in ("calibration", "evaluation"):
        parts = [group[name] for group in groups]
        totals[name] = dict(recorded_ranges=sum(p["recorded_ranges"] for p in parts),
            valid_ranges=sum(p["valid_ranges"] for p in parts),
            invalid_ranges=sum(len(p["invalid_publisher_rows"]) for p in parts),
            link_settings_with_valid_ranges=sum(p["valid_ranges"] > 0 for p in parts),
            link_settings_with_ten_valid_ranges=sum(p["valid_ranges"] >= 10 for p in parts))
    return dict(nodes=sorted(frames), recorded_ranges=sum(len(f) for f in frames.values()),
                recorded_link_settings=len(groups), overlap_rows=0, unassigned_rows=0,
                totals=totals, groups=groups)


def freeze_split():
    if OUTPUT.exists():
        raise ValueError("IDLab split already frozen")
    if pilot.audit.uncommitted():
        raise ValueError("commit the split source and protocol before freezing")
    checked = pilot.evidence.read(admission.OUTPUT)
    for name, expected in checked["sources"].items():
        pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab split source")
    frames = {node: pd.read_csv(admission.FOLDER / f"office_lab_node_{node:02d}_short.csv",
                               usecols=["Unnamed: 0", "anchor_ID_tag", "anchor_ID_anchor", "distance", *preflight.SETTINGS])
              for node in range(1, 16)}
    intervals = pd.read_csv(admission.FOLDER / "publisher-aggregate/office_lab_allsettings2.csv",
                            usecols=INTERVAL_COLUMNS)
    split = partition_records(frames, intervals)
    if (split["recorded_ranges"] != checked["recorded_ranges"]
            or split["recorded_link_settings"] != checked["recorded_link_settings"]
            or sum(p["invalid_ranges"] for p in split["totals"].values()) != checked["invalid_ranges"]):
        raise ValueError("split accounting differs from the complete source audit")
    result = dict(schema="solid.external.idlab-split.v1", status="split-frozen-execution-not-frozen",
        source_revision=pilot.audit.revision(),
        admission_sha256=pilot.identity(admission.OUTPUT),
        sources={pilot.audit.name(p): pilot.identity(p)
                 for p in (Path(__file__).resolve(), Path(pilot.__file__).resolve())},
        scope="Same 15-node layout and radio configurations. Disjoint records do not establish unseen-site generalization.",
        units="mm", row_intervals="Zero-based publisher row IDs, inclusive start and exclusive stop.",
        split_rule="First floor(n/2) recorded rows of each directed link and radio setting are calibration. Remaining rows are evaluation.",
        invalid_input_rule="Exclude only nonfinite or nonpositive ranges after partition assignment, equally for all methods. Preserve and count their source rows.",
        population_rule="Keep all 15 nodes. Do not drop nodes or links because of missing or invalid measurements.",
        fitting_rule="Fit only on reserved calibration records. Do not use evaluation residuals or per-position lookup corrections.",
        missingness_rule="Absent settings, invalid records and unknown attempts are not evidence of distance. Verify acquisition opportunities before fitting availability.",
        pending=["Freeze calibration, uncertainty and method profile recipes using reserved calibration records only.",
                 "Resolve original-log attempt accounting before using network missingness.",
                 "Freeze sample blocks, anchor assignments, scoring weights and execution budget.",
                 "Verify bounded examples for every applicable method and mode before full execution."],
        models_fitted=0, positioning_solves=0, evaluation_rows_used_for_training=0, **split)
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT])
    print(result["totals"], flush=True)


if __name__ == "__main__":
    freeze_split()
