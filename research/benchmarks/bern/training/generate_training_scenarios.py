#!/usr/bin/env python3
"""Generate deterministic benchmark-style scenarios from training scans only."""

import argparse
import hashlib
import itertools
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = "solid.training.scenarios.v1"
TRAINING_SCANS = tuple(range(1, 12))
AGENT_COUNTS = (4, 7, 10, 13, 16, 19)
ANCHOR_COUNTS = (0, 4, 8, 12, 16, 20)
SAMPLE_COUNTS = (1, 5, 10, 15, 20)
INITIALIZATION_COUNT = 2
EXPECTED_INPUT_MD5 = "ddca1cc9de31e1e525c6597519f040a2"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--groups-per-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*values):
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_training_data(path):
    if file_digest(path, "md5") != EXPECTED_INPUT_MD5:
        raise ValueError("the responder table checksum does not match the official release")
    frame = pd.read_pickle(path)
    frame = frame[frame["Scan_ID"].isin(TRAINING_SCANS)].copy()
    if set(frame["Class"].unique()) != {"Train"}:
        raise ValueError("the selected rows are not exclusively marked as training data")
    frame.sort_values(
        ["Scan_ID", "Timestamp", "Listening_Period", "Poller_Pos_ID", "Responder_Pos_ID"],
        inplace=True,
        kind="stable",
    )
    return frame


def scan_positions(frame):
    positions = {}
    for prefix in ("Poller", "Responder"):
        columns = [f"{prefix}_Pos_ID", f"{prefix}_X", f"{prefix}_Y", f"{prefix}_Z"]
        for position_id, x, y, z in frame[columns].drop_duplicates().itertuples(index=False):
            point = (float(x) * 100.0, float(y) * 100.0, float(z) * 100.0)
            position_id = int(position_id)
            existing = positions.get(position_id)
            if existing is not None and any(abs(left - right) > 1e-6 for left, right in zip(existing, point)):
                raise ValueError(f"position {position_id} has inconsistent coordinates")
            positions[position_id] = point
    if len(positions) != 40:
        raise ValueError(f"expected 40 positions in a scan and found {len(positions)}")
    return positions


def planar_distance(left, right):
    return math.hypot(left[0] - right[0], left[1] - right[1])


def nested_anchor_groups(positions):
    ids = sorted(positions)
    farthest_pair = max(
        itertools.combinations(ids, 2),
        key=lambda pair: (planar_distance(positions[pair[0]], positions[pair[1]]), -pair[0], -pair[1]),
    )
    selected = list(farthest_pair)
    while len(selected) < 20:
        remaining = [position_id for position_id in ids if position_id not in selected]
        next_id = max(
            remaining,
            key=lambda candidate: (
                min(planar_distance(positions[candidate], positions[current]) for current in selected),
                -candidate,
            ),
        )
        selected.append(next_id)
    return {count: tuple(sorted(selected[:count])) for count in ANCHOR_COUNTS if count > 0}


def mean_pair_distance(group, positions):
    distances = [
        planar_distance(positions[left], positions[right])
        for left, right in itertools.combinations(group, 2)
    ]
    return sum(distances) / len(distances)


def sampled_combinations(pool, count, candidate_limit, rng):
    total = math.comb(len(pool), count)
    if total <= candidate_limit:
        return list(itertools.combinations(pool, count))
    selected = set()
    while len(selected) < candidate_limit:
        selected.add(tuple(sorted(rng.choice(pool, size=count, replace=False).tolist())))
    return sorted(selected)


def balanced_agent_groups(pool, count, positions, groups_per_cell, seed, scan):
    rng = np.random.default_rng(stable_seed(seed, scan, "agents", count))
    candidates = sampled_combinations(pool, count, 5000, rng)
    candidates.sort(key=lambda group: (mean_pair_distance(group, positions), group))
    indexes = np.linspace(0, len(candidates) - 1, groups_per_cell).round().astype(int)
    groups = [candidates[index] for index in indexes]
    if len(set(groups)) != groups_per_cell:
        raise ValueError("not enough distinct agent groups for the requested cell depth")
    return groups


def ranging_index(frame):
    return {
        (int(poller), int(responder)): group["ToF"].to_numpy(dtype=float)
        for (poller, responder), group in frame.groupby(
            ["Poller_Pos_ID", "Responder_Pos_ID"], sort=True
        )
    }


def nested_samples(values, seed, scan, scenario_id, iteration, link_kind, left, right):
    if values is None or len(values) == 0:
        return {str(count): None for count in SAMPLE_COUNTS}
    rng = np.random.default_rng(
        stable_seed(seed, scan, scenario_id, iteration, link_kind, left, right)
    )
    order = rng.permutation(len(values))
    output = {}
    for count in SAMPLE_COUNTS:
        if len(values) < count:
            output[str(count)] = None
            continue
        selected = [float(values[index]) for index in order[:count]]
        output[str(count)] = selected[0] if count == 1 else selected
    return output


def initialization(agents, scenario_id, iteration, seed):
    if iteration == 1:
        return {str(agent): {"x": 0.0, "y": 0.0, "z": 0.0} for agent in agents}
    rng = np.random.default_rng(stable_seed(seed, scenario_id, iteration, "initialization"))
    return {
        str(agent): {
            "x": float(rng.uniform(0.0, 3000.0)),
            "y": float(rng.uniform(0.0, 1500.0)),
            "z": float(rng.uniform(0.0, 300.0)),
        }
        for agent in agents
    }


def true_positions(ids, positions):
    return {
        str(position_id): {
            "x": positions[position_id][0],
            "y": positions[position_id][1],
            "z": positions[position_id][2],
        }
        for position_id in ids
    }


def scenario_json(
    scenario_id,
    scan,
    group_index,
    agents,
    anchors,
    positions,
    rangings,
    seed,
):
    iterations = []
    for iteration in range(1, INITIALIZATION_COUNT + 1):
        agent_links = {}
        for left in agents:
            for right in agents:
                if left == right:
                    continue
                agent_links[f"{left}-{right}"] = nested_samples(
                    rangings.get((left, right)),
                    seed,
                    scan,
                    scenario_id,
                    iteration,
                    "agent",
                    left,
                    right,
                )
        anchor_links = {}
        for agent in agents:
            for anchor in anchors:
                anchor_links[f"{agent}-{anchor}"] = nested_samples(
                    rangings.get((anchor, agent)),
                    seed,
                    scan,
                    scenario_id,
                    iteration,
                    "anchor",
                    agent,
                    anchor,
                )
        iterations.append(
            {
                "ID": iteration,
                "Initial positions of Agents": initialization(agents, scenario_id, iteration, seed),
                "Ranging samples between Agents": agent_links,
                "Ranging samples to Anchors": anchor_links,
            }
        )
    return {
        "Research Metadata": {
            "schema_version": SCHEMA_VERSION,
            "data_role": "training",
            "scan_id": scan,
            "scenario_id": scenario_id,
            "group_index": group_index,
            "sample_counts": list(SAMPLE_COUNTS),
            "seed": seed,
        },
        "Agents": list(agents),
        "Anchors": list(anchors),
        "True Positions of Agents": true_positions(agents, positions),
        "True Positions of Anchors": true_positions(anchors, positions),
        "Iterations": iterations,
    }


def deterministic_zip_entry(name, contents):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info, contents


def generate(frame, output, groups_per_cell, seed):
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort_digest = hashlib.sha256()
    cell_counts = {}
    scenario_id = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for scan in TRAINING_SCANS:
            scan_frame = frame[frame["Scan_ID"] == scan]
            positions = scan_positions(scan_frame)
            anchor_groups = nested_anchor_groups(positions)
            agent_pool = tuple(sorted(set(positions) - set(anchor_groups[20])))
            rangings = ranging_index(scan_frame)
            for agent_count in AGENT_COUNTS:
                agent_groups = balanced_agent_groups(
                    agent_pool,
                    agent_count,
                    positions,
                    groups_per_cell,
                    seed,
                    scan,
                )
                for anchor_count in ANCHOR_COUNTS:
                    anchors = () if anchor_count == 0 else anchor_groups[anchor_count]
                    for group_index, agents in enumerate(agent_groups):
                        scenario_id += 1
                        data = scenario_json(
                            scenario_id,
                            scan,
                            group_index,
                            agents,
                            anchors,
                            positions,
                            rangings,
                            seed,
                        )
                        encoded = json.dumps(
                            data,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                        name = f"scan-{scan:02d}/{scenario_id}.json"
                        info, encoded = deterministic_zip_entry(name, encoded)
                        archive.writestr(info, encoded)
                        cohort_digest.update(name.encode("utf-8"))
                        cohort_digest.update(encoded)
                        key = f"scan={scan},agents={agent_count},anchors={anchor_count}"
                        cell_counts[key] = cell_counts.get(key, 0) + 1
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "groups_per_cell": groups_per_cell,
            "sample_counts": list(SAMPLE_COUNTS),
            "initializations": INITIALIZATION_COUNT,
            "training_scans": list(TRAINING_SCANS),
            "scenario_count": scenario_id,
            "cohort_sha256": cohort_digest.hexdigest(),
            "cell_counts": cell_counts,
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        info, encoded = deterministic_zip_entry("manifest.json", encoded)
        archive.writestr(info, encoded)
    return manifest


def main():
    args = arguments()
    if args.groups_per_cell < 1:
        raise ValueError("groups per cell must be positive")
    frame = load_training_data(args.input)
    manifest = generate(frame, args.output, args.groups_per_cell, args.seed)
    manifest.update(
        {
            "input": str(args.input.resolve()),
            "input_md5": EXPECTED_INPUT_MD5,
            "archive": str(args.output.resolve()),
            "archive_sha256": file_digest(args.output, "sha256"),
            "rows": len(frame),
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
