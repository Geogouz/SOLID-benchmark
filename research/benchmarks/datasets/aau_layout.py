"""AAU Meshed UWB in anchor only mode, a held-out benchmark of nine static nodes split by flights.

Source. AAU Meshed UWB Dataset (Zenodo record 10819706, CC BY 4.0). Nine static UWB nodes, identities 101 to 104 and 106
to 110, range each other and the two tags of a UAV by SDS-TWR in a fully meshed cycle of about one second, each node
publishing the exchanges it initiates on its own ranging topic. Only the real recordings recordings/*/real_ranges enter,
the twelve spiral_8m flights of 2 September 2023 and the two spiral_1to2_2m flights of 19 and 20 September 2023, the
paper's closed loop experiment with a lower spiral. The synthetic stddev* recordings, the ground truth trajectories, the
simulation configuration, the UAV tags 100 and 105 and every exchange with a tag stay unread or unused.

Truth. The published node positions of uwb_constellation.yaml, equal to the README table, in centimetres. The paper
states that the nodes' true positions were captured in the motion captured flight volume without naming the instrument.

Calibration. The publisher's constant bias per ordered pair of devices, estimated on spiral_8m seq1 with its range
evaluation tool, is subtracted from every raw range of that pair. The README lists these biases with the sign of the
paper's model, a measurement equal to the true distance plus the bias, and they are negative. The tool's statistics
file of seq1 lists the same magnitudes with the opposite sign. The build checks on seq1, the first training flight,
that subtracting the README's biases brings the median residual of the static pairs closer to zero than the raw ranges
have it, and stops otherwise. One packet scale, the RMS residual of the admissible calibrated reports of the training
flights with equal weight per ordered node pair, is applied to both halves. A calibrated reading outside the interval
from zero to the range of 200 ns of flight is a failed exchange and inadmissible, the fate of the reports the radios
publish mostly near 338 km in either sign for about one exchange in fifteen. A pair without an admissible report in any
training flight has no residual and stays out of the scale, and the calibration record names it.

Scope. Both halves score the same nine node positions and the same ordered pairs, whose biases the publisher estimated
from their true lengths on seq1. AAU therefore holds out later flights of a calibrated layout, not new positions.

Split. The fourteen flights in recorded order, by the start time of their bags. The first seven train, the last seven
evaluate. A flight whose static pair ranges moved from those of seq1 by more than 30 cm in median would indicate a
relocated node, a data defect. Such a flight is left out of its half for every method alike, and a training flight out
of the packet scale, the link fit, the deployment settings and every selection as well, since nothing may be reviewed
and changed after the freeze and a stop of stage one would block HYMN too. The calibration record, the population
records of both halves and every run record report the flight, its change and its units (excluded_flights,
training_excluded_flights), and the tables note it. A pair without an admissible report in a flight or in seq1 has no
range to compare, a data defect. The guard compares that flight on its other pairs, the half's population record and
every run record of it report the pair with its flight (unguarded_pairs), and its links stay present without samples for
every method alike, as every link without an admissible reading does. A flight without any pair to compare counts as
beyond the guard. seq1 holds at least 4,118 admissible reports by the declared counts, so the guard always keeps it and
the training half is never empty.

Units. Per flight, the reports of every ordered static pair in time order form blocks of ten, block b holding reports
10b to 10b + 9, as many blocks as the pair with the fewest reports fills. For every block, the nine nodes are ordered
by hash from the declared seed, the flight and the block, and for every anchor count k from 1 to 7 the nine cyclic
windows of k nodes in that order are the anchors of nine units, the other nodes their agents. Every node is an anchor
equally often in every block and count. An agent to anchor link carries the block of the exchanges the agent initiated
towards the anchor, its first report forming the one-reading track. The cells are the anchor counts.

The outputs under research/data/external-benchmarks/aau-layout are calibration.json from the training half and, per
half, request.json.gz, reference.json and population.json.

Either half is built only after the method freeze and the ledger allow its read (heldout.require_freeze), and an
evaluation half records its freeze. The loader, the calibration and the construction of units wait for the same freeze,
so no reading of the archive enters a computation before it. The declaration alone reads the archive before the freeze
(declared), for the counts, the failed exchanges and the sign check on seq1 it states.

Usage: aau_layout.py build --split training|evaluation
"""
import argparse
import json
import math
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import heldout

ARCHIVE = heldout.DOWNLOADS / "zenodo-10819706/AAU_Meshed_UWB_Dataset.zip"
ARCHIVE_MD5 = "a59e6d590b0b4b69385746fc464c5050"
ARCHIVE_SHA256 = "9236366321ffc51b51543552832a150162aaa271737409ad13c02e04896f8113"
PREFIX = "AAU_Meshed_UWB_Dataset/"
REAL = re.compile(re.escape(PREFIX) + r"recordings/([A-Za-z0-9_]+)/real_ranges/seq(\d+)\.bag")
CONSTELLATION = "uwb_constellation.yaml"
STATISTICS = "recordings/spiral_8m/eval/seq1/statistics.yaml"
README = "README.md"
CALIBRATION_FLIGHT = "spiral_8m/seq1"
NODES = (101, 102, 103, 104, 106, 107, 108, 109, 110)
TAGS = (100, 105)
FLIGHTS = 14
CELLS = tuple(range(1, len(NODES) - 1))
SEED = "solid.heldout.aau.v1"
RELOCATION_GUARD_CM = 30.0
RELOCATION_RULE = (f"A flight in which the median admissible calibrated range of a static pair moved from seq1 by more than "
                   f"{RELOCATION_GUARD_CM:g} cm shows a relocated node, a data defect, and a flight without any pair the guard can "
                   "compare counts as such a flight. Such a flight is left out of its half for every method alike, its units left "
                   "out, and a training flight out of the packet scale, the link fit, the deployment settings and every selection as "
                   "well, since nothing may be reviewed and changed after the freeze and a stop would block both datasets. The "
                   "calibration record, the population records of both halves and every run record report the flight, its change "
                   "and its units, and the tables of AAU note it. The guard compares ranges only and checks each half when that half "
                   "is built.")
# The rules of the AAU section that are text alone, which heldout.revise writes into the tracked section without
# reading the archive, by their path in the section.
SECTION_TEXT = {("split", "relocation_guard"): RELOCATION_RULE}
OUTPUT = heldout.OUTPUT / "aau-layout"
MESSAGE = "uwb_msgs/msg/TwoWayRangeStamped"


# ---------------------------------------------------------------------------------------------------------------------
# Members and published tables


def recordings(names):
    """The real recordings among archive members, (member, flight name) sorted by name. Every other member, the
    synthetic stddev* sequences above all, is left out."""
    found = []
    for name in names:
        match = REAL.fullmatch(name)
        if match:
            found.append((name, f"{match.group(1)}/seq{int(match.group(2))}"))
    return sorted(found)


def text(archive, member):
    return archive.read(PREFIX + member).decode("utf-8")


def mapping(body):
    """{key: value} pairs of a flat YAML flow mapping of numbers."""
    return {int(k): float(v) for k, v in re.findall(r"(\d+)\s*:\s*(-?\d+(?:\.\d+)?)", body)}


def positions(archive):
    """The published node positions in centimetres, checked against the README table and the topic of every node."""
    source = text(archive, CONSTELLATION)
    line = next(l for l in source.splitlines() if l.startswith("abs_anchor_positions:"))
    published = {int(k): tuple(100.0 * float(v) for v in values.split(",")) for k, values in re.findall(r"(\d+)\s*:\s*\[([^\]]+)\]", line)}
    topics = dict(re.findall(r"(\d+)\s*:\s*\"(/a\d\d/ranging)\"", next(l for l in source.splitlines() if l.startswith("anchor_topics:"))))
    table = {int(k): tuple(100.0 * float(v) for v in values.split(","))
             for k, values in re.findall(r"^(\d{3})\s*\|\s*\[([^\]]+)\]", text(archive, README), flags=re.M) if int(k) in NODES}
    if set(published) != set(NODES) or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in published.values()):
        raise ValueError("the constellation file lacks a static node")
    if table != published:
        raise ValueError("the constellation file and the README table disagree")
    if {int(k): v for k, v in topics.items()} != {node: f"/a{node - 100:02d}/ranging" for node in NODES}:
        raise ValueError("unexpected ranging topics of the static nodes")
    return published


def biases(archive):
    """The published constant bias of every ordered pair of static nodes in centimetres as the README lists it, the
    additive bias of a measurement, and the same pairs of the range evaluation statistics of seq1, which list the
    magnitudes with the opposite sign."""
    statistics = " ".join(text(archive, STATISTICS).split())
    table = {}
    for identity, body in re.findall(r"ID:\s*(\d+),\s*contant_bias_table:\s*\{([^}]*)\}", statistics):
        for peer, value in mapping(body).items():
            table[(int(identity), peer)] = 100.0 * value
    readme = {}
    for identity, body in re.findall(r"^ID:\s*(\d+)\s*\|\s*\{([^}]*)\}", text(archive, README), flags=re.M):
        for peer, value in mapping(body).items():
            readme[(int(identity), peer)] = 100.0 * value
    pairs = {(a, b) for a in NODES for b in NODES if a != b}
    if not pairs <= set(table) or not pairs <= set(readme):
        raise ValueError("a static pair lacks its published constant bias")
    return {pair: readme[pair] for pair in sorted(pairs)}, {pair: table[pair] for pair in sorted(pairs)}


# ---------------------------------------------------------------------------------------------------------------------
# Recordings


def read_bag(path):
    """The start time and the report streams of the static nodes of one ROS1 bag.

    streams maps an ordered pair (initiator, responder) of static nodes to its reports in time order, each (header stamp
    in ns, header sequence, raw range in cm). Only the nodes' own ranging topics are read, exchanges with a tag are
    dropped."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    topics = {f"/a{node - 100:02d}/ranging": node for node in NODES}
    streams = defaultdict(list)
    with Reader(Path(path)) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        if {c.topic for c in connections} != set(topics) or any(c.msgtype != MESSAGE for c in connections):
            raise ValueError("a static node's ranging topic is missing or of another type")
        store = get_typestore(Stores.ROS1_NOETIC)
        store.register(get_types_from_msg(connections[0].msgdef.data, MESSAGE))
        start = reader.start_time
        for connection, _, raw in reader.messages(connections=connections):
            message = store.deserialize_ros1(raw, connection.msgtype)
            if message.UWB_ID1 != topics[connection.topic]:
                raise ValueError("a node published an exchange it did not initiate")
            if message.UWB_ID2 in TAGS:
                continue
            if message.UWB_ID2 not in NODES or message.UWB_ID2 == message.UWB_ID1:
                raise ValueError("an exchange with an unknown device")
            value = float(message.range_raw)
            if not math.isfinite(value):
                raise ValueError("a nonfinite raw range")
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            streams[(message.UWB_ID1, message.UWB_ID2)].append((stamp, int(message.header.seq), 100.0 * value))
    for pair, reports in streams.items():
        if any(a[0] >= b[0] for a, b in zip(reports, reports[1:])):
            raise ValueError(f"the reports of {pair} are not strictly ordered in time")
    if set(streams) != {(a, b) for a in NODES for b in NODES if a != b}:
        raise ValueError("a recording lacks an ordered static pair")
    return start, dict(streams)


def flights(archive):
    """Every real recording in recorded order, each with its name, member, SHA-256, start time and static report streams."""
    found = []
    with tempfile.TemporaryDirectory(prefix="aau-bag-") as folder:
        for member, name in recordings(archive.namelist()):
            content = archive.read(member)
            path = Path(folder) / "flight.bag"
            path.write_bytes(content)
            start, streams = read_bag(path)
            path.unlink()
            found.append(dict(name=name, member=member, sha256=heldout.sha256_bytes(content), start_ns=start, streams=streams))
    found.sort(key=lambda flight: flight["start_ns"])
    if len(found) != FLIGHTS or len({f["start_ns"] for f in found}) != FLIGHTS:
        raise ValueError("the archive does not hold the fourteen real recordings")
    return found


def declared(path=None):
    """The published tables and the real recordings of the declared archive, for the declaration. The declaration takes
    from them the structure of the archive, the report counts of every flight, the failed exchanges of the training
    flights and the sign check on seq1, the statistics it states, and computes nothing else from a report, so it reads
    the archive before the freeze. Every other reader takes load."""
    path = Path(path or ARCHIVE)
    heldout.require_archive(path, ARCHIVE_MD5, ARCHIVE_SHA256)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        data = dict(positions=positions(archive), members={name: heldout.sha256_bytes(archive.read(PREFIX + name)) for name in (CONSTELLATION, STATISTICS, README)})
        data["biases"], data["statistics_biases"] = biases(archive)
        data["flights"] = flights(archive)
    data["excluded"] = excluded(names)
    return data


def load(path=None):
    """The published tables and the real recordings for a build or a fit, refused before the method freeze and the
    ledger allow a read of a half (heldout.require_freeze), before the archive is opened."""
    heldout.require_freeze("training")
    return declared(path)


def excluded(names):
    """The recordings that do not enter, by folder, with the reason."""
    folders = defaultdict(int)
    for name in names:
        match = re.fullmatch(re.escape(PREFIX) + r"recordings/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)/.*\.bag", name)
        if match and match.group(2) != "real_ranges":
            folders[f"{match.group(1)}/{match.group(2)}"] += 1
    return dict(sorted(folders.items()))


# ---------------------------------------------------------------------------------------------------------------------
# Split and plan


def blocks(flight):
    """The complete blocks of ten reports of a flight, limited by its ordered static pair with the fewest reports."""
    return min(len(reports) for reports in flight["streams"].values()) // heldout.BLOCK_READINGS


def split(found):
    """The flight names of each half, the first half in recorded order training."""
    names = [flight["name"] for flight in found]
    if not names or names[0] != CALIBRATION_FLIGHT:
        raise ValueError("the calibration flight must open the training half")
    half = len(names) // 2
    return dict(training=names[:half], evaluation=names[half:])


def plan(side, found):
    """Every unit of a half with its identity, flight, block, anchor count, cyclic start, anchors and agents."""
    names = split(found)[side]
    units = []
    for position, flight in enumerate(found, 1):
        if flight["name"] not in names:
            continue
        for block in range(blocks(flight)):
            order = heldout.ordered(NODES, SEED, flight["name"], block)
            for count in CELLS:
                for start in range(len(NODES)):
                    anchors = sorted((order + order)[start:start + count])
                    units.append(dict(id=10000 * position + 100 * block + 10 * count + start, flight=flight["name"], position=position,
                                      block=block, anchors=count, start=start, anchor_ids=anchors,
                                      agents=[node for node in NODES if node not in anchors]))
    return units


# ---------------------------------------------------------------------------------------------------------------------
# Calibration and units


def calibrated(values_cm, pair, table):
    return np.asarray(values_cm, dtype=float) - table[pair]


def admissible_median(reports, pair, table):
    """The median admissible calibrated range of one ordered pair's reports, None when none is admissible."""
    values = calibrated([r[2] for r in reports], pair, table)
    values = values[heldout.admissible(values)]
    return float(np.median(values)) if len(values) else None


def medians(flight, table):
    """The median admissible calibrated range of every ordered static pair of a flight."""
    return {pair: admissible_median(reports, pair, table) for pair, reports in flight["streams"].items()}


def relocation_guard(found, names, table):
    """The largest change of a static pair's median admissible calibrated range from seq1 in each named flight, over the
    pairs that hold an admissible report in both, None for a flight without such a pair. A pair without an admissible
    report has no range to compare, a data defect the guard leaves out and unguarded_pairs reports (the declaration's
    rules.outcomes). It compares ranges only, never a range with a published distance."""
    reference = medians(next(f for f in found if f["name"] == CALIBRATION_FLIGHT), table)
    result = {}
    for flight in found:
        if flight["name"] not in names:
            continue
        changes = [abs(median - reference[pair]) for pair, median in medians(flight, table).items()
                   if median is not None and reference[pair] is not None]
        result[flight["name"]] = round(max(changes), 3) if changes else None
    return result


def unguarded_pairs(found, names, table):
    """The ordered static pairs of each named flight that the guard cannot compare, those without an admissible report in
    the flight or in seq1, as a-b, for every flight that holds one. Their links stay present without samples, as for
    every link without an admissible reading, for every method alike."""
    reference = medians(next(f for f in found if f["name"] == CALIBRATION_FLIGHT), table)
    result = {}
    for flight in found:
        if flight["name"] in names:
            pairs = sorted(pair for pair, median in medians(flight, table).items() if median is None or reference[pair] is None)
            if pairs:
                result[flight["name"]] = [f"{a}-{b}" for a, b in pairs]
    return result


def relocated(changes):
    """The flights whose static ranges moved beyond the guard, a sign of a relocated node, and those without any pair
    the guard can compare."""
    return sorted(name for name, change in changes.items() if change is None or change > RELOCATION_GUARD_CM)


def planned_units(flight):
    """The units the plan draws from a flight, nine for every anchor count in each of its blocks (plan)."""
    return blocks(flight) * len(CELLS) * len(NODES)


def excluded_flights(found, changes):
    """The flights beyond the relocation guard among those of changes, each with its change and the count of its planned
    units. A relocated node moves the truth of its flight, a data defect, so such a flight is left out of its half for
    every method alike, a training flight out of the calibration, the fits and the selections as well, and the records of
    both halves report it (the declaration's rules.outcomes). A flight without any pair the guard can compare, change
    None, counts as beyond it. No such flight stops a build, since nothing may be reviewed and changed after the freeze."""
    by_name = {flight["name"]: flight for flight in found}
    return {name: dict(change_cm=changes[name], units=planned_units(by_name[name])) for name in relocated(changes)}


def failed_exchanges(data, names):
    """The static reports of the named flights and how many of them fall outside the admissible interval once calibrated."""
    values = np.concatenate([calibrated([r[2] for r in reports], pair, data["biases"])
                             for flight in data["flights"] if flight["name"] in names for pair, reports in sorted(flight["streams"].items())])
    return dict(reports=int(len(values)), inadmissible=int((~heldout.admissible(values)).sum()), below_zero=int((values < 0).sum()))


def sign_check(data):
    """The median residual of every report of the static pairs of seq1, raw, calibrated by the README's biases and
    calibrated with the sign of the statistics file. Refused unless the README's biases bring it closer to zero."""
    table, nodes = data["biases"], data["positions"]
    truth = {(a, b): math.dist(nodes[a], nodes[b]) for a, b in table}
    seq1 = next(f for f in data["flights"] if f["name"] == CALIBRATION_FLIGHT)
    raw = np.concatenate([[r[2] - truth[pair] for r in reports] for pair, reports in sorted(seq1["streams"].items())])
    corrections = np.concatenate([[table[pair]] * len(reports) for pair, reports in sorted(seq1["streams"].items())])
    sign = dict(raw_median_cm=round(float(np.median(raw)), 3), readme_sign_median_cm=round(float(np.median(raw - corrections)), 3),
                statistics_sign_median_cm=round(float(np.median(raw + corrections)), 3))
    if not abs(sign["readme_sign_median_cm"]) < abs(sign["raw_median_cm"]):
        raise ValueError("subtracting the README's biases does not reduce the median residual of seq1, review the sign")
    return sign


def calibrate(data, names):
    """The sign check of the published biases on seq1 and the packet scale of the named training flights that the
    relocation guard keeps, refused before the method freeze and the ledger allow a read of the training half. A
    training flight beyond the guard stays out of the scale, and the record names it with its change and its units
    (excluded_flights), which the units and the link fit leave out as well. seq1, the guard's reference, is always kept
    while it holds an admissible report. It counts the reports of evaluation flights among those it read, which the
    builders require to be none."""
    heldout.require_freeze("training")
    table, nodes = data["biases"], data["positions"]
    truth = {(a, b): math.dist(nodes[a], nodes[b]) for a, b in table}
    guard = relocation_guard(data["flights"], names, table)
    left_out = excluded_flights(data["flights"], guard)
    kept = [name for name in names if name not in left_out]
    if not kept:
        raise ValueError("the relocation guard keeps no training flight, since seq1 holds no admissible report")
    evaluation = split(data["flights"])["evaluation"]
    sign = sign_check(data)
    errors, reports_used, inadmissible, unscaled = [], 0, 0, []
    for pair in sorted(table):
        values = np.concatenate([calibrated([r[2] for r in f["streams"][pair]], pair, table) for f in data["flights"] if f["name"] in kept])
        accepted = heldout.admissible(values)
        inadmissible += int((~accepted).sum())
        reports_used += int(accepted.sum())
        if not accepted.any():
            # a pair without an admissible training report has no residual and stays out of the scale
            unscaled.append(f"{pair[0]}-{pair[1]}")
            continue
        errors.append(float(np.mean((values[accepted] - truth[pair]) ** 2)))
    scale = float(np.sqrt(np.mean(errors))) if errors else 0.0
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("no positive training range uncertainty")
    return dict(dataset="aau", rule=("The publisher's constant bias of every ordered static pair as the README lists it, estimated on "
                                     "spiral_8m seq1, is subtracted from its raw ranges. One packet scale, the RMS residual of the "
                                     "admissible calibrated reports of the training flights against the published distance, equal "
                                     "weight per ordered pair."),
                sign_check_on_seq1=sign, biases_cm={f"{a}-{b}": value for (a, b), value in sorted(table.items())}, scale_cm=scale,
                training_flights=kept, excluded_flights=left_out, training_pairs=len(errors), training_reports=reports_used,
                inadmissible_training_reports=inadmissible, pairs_without_admissible_training_report=unscaled, relocation_guard_cm=guard,
                unguarded_pairs=unguarded_pairs(data["flights"], names, table),
                evaluation_rows_used=sum(len(reports) for flight in data["flights"] if flight["name"] in names and flight["name"] in evaluation
                                         for reports in flight["streams"].values()))


def area_rule():
    solid = str(heldout.ROOT / "research/solid")
    if solid not in sys.path:
        sys.path.insert(0, solid)
    import deployment
    return lambda prepared: deployment.heldout_unit_area(prepared, "aau")


def construct(data, specs, scale_cm):
    """The adapter units of both tracks and the reference metadata of a half's plan, refused before the method freeze and
    the ledger allow a read of a half, and for any unit of a flight of the declared evaluation half before the ledger
    opens AAU (heldout.require_freeze of the evaluation half), before any report is taken."""
    heldout.require_freeze("training")
    if heldout.evaluation_units("aau", [dict(flight=spec["flight"]) for spec in specs]):
        heldout.require_freeze("evaluation", dataset="aau")
    by_name = {flight["name"]: flight for flight in data["flights"]}
    rule = area_rule()
    units, metadata = [], []
    for spec in specs:
        streams = by_name[spec["flight"]]["streams"]
        first = heldout.BLOCK_READINGS * spec["block"]
        links = {}
        for agent in spec["agents"]:
            for anchor in spec["anchor_ids"]:
                reports = streams[(agent, anchor)][first:first + heldout.BLOCK_READINGS]
                if len(reports) != heldout.BLOCK_READINGS:
                    raise ValueError("an incomplete block")
                links[(agent, anchor)] = calibrated([r[2] for r in reports], (agent, anchor), data["biases"]).tolist()
        anchors = {anchor: data["positions"][anchor] for anchor in spec["anchor_ids"]}
        units.extend(heldout.units_of(links, spec["agents"], anchors, spec["id"], SEED, scale_cm, rule))
        metadata.append(dict(id=spec["id"], acquisition=heldout.cell(spec["anchors"]), anchors=spec["anchors"], flight=spec["flight"],
                             block=spec["block"], start=spec["start"], agents=spec["agents"], anchor_ids=spec["anchor_ids"]))
    return units, metadata


def reference(data, metadata):
    """The scorer's reference of a half. Every unit scores against the true positions of its own agents, so its cap is
    the diameter of its agents as in the released scorer. positions_cm holds the nine nodes in three dimensions."""
    nodes = data["positions"]
    units = [dict(entry, reference=str(entry["id"])) for entry in metadata]
    return dict(units=units, coordinates_cm={str(entry["id"]): {str(a): list(nodes[a][:2]) for a in entry["agents"]} for entry in units},
                positions_cm={str(n): list(nodes[n]) for n in NODES})


# ---------------------------------------------------------------------------------------------------------------------
# Declaration and build


def declaration(data=None):
    data = data or declared()
    found = data["flights"]
    halves = split(found)
    plans = {side: plan(side, found) for side in heldout.SPLITS}
    signs = {pair: (data["biases"][pair], data["statistics_biases"][pair]) for pair in data["biases"]}
    return dict(
        archive=dict(path=heldout.relative(ARCHIVE), md5=ARCHIVE_MD5, sha256=ARCHIVE_SHA256, source="Zenodo record 10819706, AAU Meshed UWB Dataset",
                     licence=("CC BY 4.0 on the Zenodo record. The archive's LICENSE and README state a BSD-2 licence with a "
                              "condition against sale and a citation requirement for academic use, written for the publisher's "
                              "software.")),
        scope=("Both halves score the same nine node positions and the same ordered pairs, and the publisher estimated the pair "
               "biases from the true lengths of those pairs on seq1. SOLID's link fit and the height prior read the training "
               "flights of these same links and nodes. AAU therefore holds out later flights of a calibrated layout, not new "
               "positions, unlike HYMN, whose evaluation positions are new."),
        extraction=dict(rule=("The real recordings recordings/*/real_ranges/seq*.bag are read one at a time through a temporary file, "
                              "and the constellation, the seq1 statistics and the README in memory. No other member is read."),
                        prefix=PREFIX, members_sha256=data["members"], recordings={f["name"]: f["sha256"] for f in found},
                        excluded=dict(recordings=data["excluded"], reason=("The stddev* folders hold synthetic ranges, true ranges with "
                                                                            "added noise. The UAV tags 100 and 105 and every exchange "
                                                                            "with a tag are dropped."))),
        spiral_1to2_2m=("Two real recordings of 19 and 20 September 2023, 92 s and 157 s, the closed loop experiment of the paper "
                        "with the spiral lowered to 2.2 m and alternating between 1 m and 2.2 m. Their static nodes carry the same "
                        "identities and topics, and the publisher generated their synthetic versions from the same constellation "
                        "file. They are real flights and enter."),
        nodes=dict(identities=list(NODES), positions="abs_anchor_positions of uwb_constellation.yaml, equal to the README table, metres converted to centimetres",
                   heights_cm=dict(minimum=round(min(p[2] for p in data["positions"].values()), 1),
                                   maximum=round(max(p[2] for p in data["positions"].values()), 1))),
        reference=("The paper (arXiv 2408.14081, Sec. IV) states that the true positions of the nine static nodes were captured in the "
                   "flight volume covered by the motion capture system, without naming the instrument used for the nodes. The nodes "
                   "stand at heights from 0.3 m to 3.3 m, so only anchor only mode is used and scoring stays horizontal."),
        streams=("A static node publishes the exchanges it initiates on its ranging topic, UWB_ID1 the node and UWB_ID2 the peer, "
                 "range_raw in metres. The reports of an ordered pair are ordered by header stamp. range_corr, R and LOS are unused."),
        calibration=dict(rule=("The publisher's constant bias of each ordered pair of static nodes, the gamma of the README's pair-wise "
                               "bias table estimated on spiral_8m seq1 with the publisher's range evaluation tool, is subtracted from "
                               "every raw range of that pair, for every method. One packet scale, the RMS residual of the admissible "
                               "calibrated reports of the training flights against the published distance with equal weight per "
                               "ordered pair, is applied to both halves."),
                         sign=("The README lists the biases with the sign of the paper's model, a measurement equal to the true distance "
                               "plus the bias (arXiv 2408.14081, eq. 12), and they are negative. The tool's statistics file "
                               "recordings/spiral_8m/eval/seq1/statistics.yaml lists the same magnitudes with the opposite sign. The "
                               "build checks on seq1, a training flight, that subtracting the README's biases brings the median "
                               "residual of the static pairs closer to zero than the raw ranges have it, and stops otherwise."),
                         revision=("The first version of this declaration took the sign of the statistics file. Its declared check on "
                                   "seq1 stopped the first build before any fit or solve, because the raw static ranges of seq1 fall short "
                                   "of the published distances by the published magnitudes. The medians below come from seq1 alone."),
                         sign_check_on_seq1=sign_check(data),
                         statistics_magnitudes_equal=all(abs(abs(a) - abs(b)) < 1e-9 for a, b in signs.values()),
                         statistics_signs_opposite=all(a * b < 0 or a == b == 0 for a, b in signs.values()),
                         inadmissible=("A calibrated reading outside the interval from zero to the range of 200 ns of flight is a "
                                       "failed exchange and stays out of the samples. The radios publish such failures mostly near "
                                       "338 km in either sign."),
                         failed_exchanges_in_training=failed_exchanges(data, halves["training"])),
        extent="The published 5 m by 8 m area of the static nodes (README and paper Sec. IV-A), its diagonal around every known anchor.",
        split=dict(rule=("The fourteen real flights in recorded order, by the start time of their bags. The first seven train and the "
                         "last seven evaluate. seq1, the flight of the published calibration, opens the training half."),
                   flights=[dict(name=f["name"], start_ns=f["start_ns"], blocks=blocks(f)) for f in found],
                   **{f"{side}_flights": halves[side] for side in heldout.SPLITS},
                   relocation_guard=RELOCATION_RULE),
        units=dict(rule=("Per flight, the reports of every ordered static pair in time order form blocks of ten, block b holding reports "
                         "10b to 10b + 9, as many blocks as the pair with the fewest reports fills. For every block the nine nodes are "
                         "ordered by hash from the seed, the flight and the block, and for every anchor count k from 1 to 7 the nine "
                         "cyclic windows of k nodes in that order are the anchors of nine units, the other nodes their agents. An agent "
                         "to anchor link carries the block of the exchanges the agent initiated towards the anchor, its first report "
                         "forming the one-reading track, as a tag's own ranging would carry it. The exchanges an anchor initiated "
                         "stay unused."),
                   tracks=list(heldout.TRACKS), seed=SEED, cells=[heldout.cell(k) for k in CELLS], units_per_block_and_cell=len(NODES),
                   identity="10000 flight position in recorded order + 100 block + 10 anchor count + cyclic start",
                   agent_identities="the published node identities"),
        plan={side: dict(identities=len(plans[side]), identities_per_cell={heldout.cell(k): sum(u["anchors"] == k for u in plans[side]) for k in CELLS},
                         sha256=heldout.fingerprint(plans[side])) for side in heldout.SPLITS})


def calibration_path():
    return OUTPUT / "calibration.json"


def outputs(side):
    folder = OUTPUT / side
    return dict(request=folder / "request.json.gz", reference=folder / "reference.json", population=folder / "population.json")


keep_or_write = heldout.keep_or_write


def build(side):
    # Either half waits for the committed method freeze and the ledger's opening of its step, before anything is read.
    freeze = heldout.require_freeze(side, dataset="aau")
    declared = heldout.section("aau")
    data = load()
    if heldout.canonical(declaration(data)) != declared:
        raise ValueError("the AAU archive or rules differ from the declaration")
    halves = split(data["flights"])
    declaration_sha256 = heldout.sha256(heldout.RECORD)
    if side == "training":
        keep_or_write(calibration_path(), dict(calibrate(data, halves["training"]), archive_sha256=ARCHIVE_SHA256,
                                               declaration_sha256=declaration_sha256))
    calibration = heldout.training_calibration(calibration_path(), ARCHIVE_SHA256)
    guard = relocation_guard(data["flights"], halves[side], data["biases"])
    unguarded = unguarded_pairs(data["flights"], halves[side], data["biases"])
    specs = plan(side, data["flights"])
    if heldout.fingerprint(specs) != declared["plan"][side]["sha256"]:
        raise ValueError("the AAU plan differs from the declaration")
    # a flight beyond the relocation guard is left out of its half for every method alike and reported, and the flights
    # the calibration left out of the training half are reported with every half
    left_out = excluded_flights(data["flights"], guard)
    units, metadata = construct(data, [spec for spec in specs if spec["flight"] not in left_out], calibration["scale_cm"])
    counts = heldout.population(units, metadata)
    if counts["identities"] + sum(entry["units"] for entry in left_out.values()) != declared["plan"][side]["identities"]:
        raise ValueError("the built AAU half differs from its declared plan")
    files = outputs(side)
    keep_or_write(files["request"], dict(units=units))
    keep_or_write(files["reference"], reference(data, metadata))
    keep_or_write(files["population"], dict(dataset="aau", split=side, counts=counts, scale_cm=calibration["scale_cm"], relocation_guard_cm=guard,
                                            excluded_flights=left_out, training_excluded_flights=calibration.get("excluded_flights", {}),
                                            unguarded_pairs=unguarded, calibration_sha256=heldout.sha256(calibration_path()),
                                            declaration_sha256=declaration_sha256, **(freeze if side == "evaluation" else {}),
                                            files={name: heldout.sha256(path) for name, path in files.items() if name != "population"}))
    print(json.dumps(dict(split=side, **counts, excluded_flights=left_out, unguarded_pairs=unguarded)), flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--split", required=True, choices=heldout.SPLITS)
    args = parser.parse_args()
    build(args.split)


if __name__ == "__main__":
    main()
