"""HYMN in anchor only mode, a held-out benchmark with new evaluation positions.

Source. HYMN dataset release 1.3 (Zenodo record 19566178, data CC BY 4.0, code MIT). The builder reads four members of
the archive in memory. The processed UWB file holds the two-way ranges of the Zigpos tag to the ten UWB anchors, one
row per epoch of about 1.23 s, with the publisher's point association by the logged time windows and the tag's
reference position. The reference files hold the anchor and point coordinates in the local frame centred at the
tachymeter station and the time windows. Every other technology and every raw log stays unread by the builder. The
raw ZigPos logs were inspected once for their structure, to settle what a blank reading means (the declaration's
disclosures).

Truth. The tag's surveyed position per point (X, Y and Z_LOCAL_UWB of the processed file, equal to the UWB2 columns of
the point reference) and the surveyed anchor coordinates, in centimetres.

Calibration. The publisher documents no UWB range calibration, so the ranges enter as published, converted from metres.
One packet scale for the deployment is the RMS residual of the admissible published readings of the training points
against the slant distance, equal weight per point to anchor link. A missing reading (NaN, which also stands for a
published nonpositive range) is no reading, a failed exchange of an anchor the tag polled, since every ranging round of
the tag's raw ZigPos log lists all ten anchors. A published reading beyond the range of 200 ns of flight is a failed
exchange and inadmissible.

Split. The points in the publisher's scheme order, grid points by row A then column B and the transition points T01 to
T06 last, alternate between training and evaluation starting with training. Neighbouring points of every row fall on
opposite sides, so both halves span the length and width of the hall, the lower rows A01 and A07 and the transition
strip, and every evaluation position is new. The rule reads only the point labels.

Units. A unit takes eight points of one half as agents, k of the ten anchors and one window of ten consecutive epochs
shared by its agents. The twelve complete windows every point has are declared, window w holding the epochs 10w to
10w + 9 of a point in time order. Each agent to anchor link carries its readings in the window in the ten-reading
track and the reading of the window's first epoch 10w alone in the one-reading track. A link whose first epoch holds no
reading has no observation in the one-reading track, even when a later epoch answered, and a link without a reading in
the window has none in either track. The cells are the anchor counts 1 to 10 with sixty units each per half. The
agents, the anchors and the window of a unit are drawn by hash order from the declared seed and the unit's half, anchor
count and replicate.

The outputs under research/data/external-benchmarks/hymn-layout are calibration.json from the training half and, per
half, request.json.gz, reference.json and population.json.

Either half is built only after the method freeze and the ledger allow its read (heldout.require_freeze), and an
evaluation half records its freeze. The loader, the calibration and the construction of units wait for the same freeze,
so no reading of the archive enters a computation before it. The declaration alone reads the archive before the freeze
(declared), for the counts and the failed exchanges it states.

Usage: hymn_layout.py build --split training|evaluation
"""
import argparse
import csv
import datetime
import io
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import heldout

ARCHIVE = heldout.DOWNLOADS / "zenodo-19566178/HYMN-dataset-1.3.zip"
ARCHIVE_MD5 = "4874f6c931561eaaa17bb0972a052e42"
ARCHIVE_SHA256 = "7a728275e2822411e5dbb3eb1ba72fad31e043498af5218a6545e031d2de257d"
PREFIX = "TUD-ITVS-HYMN-dataset-8ae0f89/"
MEMBERS = dict(uwb="data/processed/csv/uwb.csv", anchors="data/reference/csv/anchor_coordinates.csv",
               points="data/reference/csv/point_coordinates.csv", times="data/reference/csv/time_reference.csv")
OUTPUT = heldout.OUTPUT / "hymn-layout"
SEED = "solid.heldout.hymn.v1"
ANCHORS = tuple(f"UWB_{index:02d}" for index in range(1, 11))
ANCHOR_IDS = {name: 100 + index for index, name in enumerate(ANCHORS, 1)}
POINTS = 48
AGENTS_PER_UNIT = 8
UNITS_PER_CELL = 60
WINDOWS = 12
CELLS = tuple(range(1, len(ANCHORS) + 1))
GRID = re.compile(r"A(\d{2})B(\d)")
TRANSITION = re.compile(r"T(\d{2})")


# ---------------------------------------------------------------------------------------------------------------------
# Reading


def parse_list(text):
    """A list the publisher wrote as Python text, its items as strings."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("unexpected list field")
    return [item.strip().strip("'\"") for item in text[1:-1].split(",")] if text[1:-1].strip() else []


def utc_ms(text):
    return int(datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def rows(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(PREFIX + MEMBERS[name]).decode("utf-8"))))


def centimetres(row, columns):
    values = [float(row[c]) for c in columns]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite reference coordinate")
    return tuple(100.0 * v for v in values)


def read(archive):
    """Anchors, tag positions and epochs of an open archive, with the publisher's structure checked.

    anchors maps an anchor identity to (x, y, z) in cm, points maps a point label to the tag's (x, y, z) in cm, epochs
    maps a label to its epochs in time order, each (timestamp in ms, {anchor identity: range in cm or None})."""
    anchors = {}
    for row in rows(archive, "anchors"):
        if row["point_id"] in ANCHOR_IDS:
            anchors[ANCHOR_IDS[row["point_id"]]] = centimetres(row, ("X_LOCAL", "Y_LOCAL", "Z_LOCAL"))
    if set(anchors) != set(ANCHOR_IDS.values()):
        raise ValueError("the anchor reference lacks a UWB anchor")
    reference = {row["point_id"]: centimetres(row, ("X_LOCAL_UWB2", "Y_LOCAL_UWB2", "Z_LOCAL_UWB2")) for row in rows(archive, "points")}
    windows = {row["point"]: (utc_ms(row["start_time_UTC"]), utc_ms(row["end_time_UTC"])) for row in rows(archive, "times")}
    if len(reference) != POINTS or set(reference) != set(windows):
        raise ValueError("the point and time references disagree")
    points, epochs = {}, defaultdict(list)
    for row in rows(archive, "uwb"):
        label = row["point_id"]
        names = parse_list(row["anchor_ids"])
        values = parse_list(row["ranges"])
        if len(names) != len(values) or set(names) != set(ANCHORS) or len(set(names)) != len(names):
            raise ValueError("an epoch does not list every UWB anchor once")
        tag = centimetres(row, ("X_LOCAL_UWB", "Y_LOCAL_UWB", "Z_LOCAL_UWB"))
        if points.setdefault(label, tag) != tag:
            raise ValueError("a point carries two tag positions")
        stamp = int(row["ts"])
        start, stop = windows[label]
        if not start <= stamp <= stop:
            raise ValueError("an epoch lies outside its point's published time window")
        ranges = {}
        for name, value in zip(names, values):
            reading = float(value)
            if math.isnan(reading):
                ranges[ANCHOR_IDS[name]] = None
            elif math.isfinite(reading) and reading > 0:
                ranges[ANCHOR_IDS[name]] = 100.0 * reading
            else:
                raise ValueError("a published reading is neither missing nor a positive range")
        epochs[label].append((stamp, ranges))
    if set(points) != set(reference):
        raise ValueError("the UWB file and the point reference hold different points")
    for label, tag in points.items():
        if max(abs(a - b) for a, b in zip(tag, reference[label])) > 1e-6:
            raise ValueError(f"the tag position of {label} differs from its UWB2 reference")
    for label, items in epochs.items():
        items.sort(key=lambda item: item[0])
        if any(a[0] >= b[0] for a, b in zip(items, items[1:])):
            raise ValueError(f"the epochs of {label} are not strictly ordered in time")
    return dict(anchors=anchors, points=points, epochs=dict(epochs))


def declared(path=None):
    """The data of the declared archive and the SHA-256 of every member read, for the declaration. The declaration takes
    from it the structure of the archive, the reading counts of every point and the failed exchanges of the training
    points, the statistics it states, and computes nothing else from a reading, so it reads the archive before the
    freeze. Every other reader takes load."""
    path = Path(path or ARCHIVE)
    heldout.require_archive(path, ARCHIVE_MD5, ARCHIVE_SHA256)
    with zipfile.ZipFile(path) as archive:
        data = read(archive)
        data["members"] = {name: heldout.sha256_bytes(archive.read(PREFIX + member)) for name, member in MEMBERS.items()}
    data["archive_sha256"] = ARCHIVE_SHA256
    return data


def load(path=None):
    """The data of the declared archive for a build or a fit, refused before the method freeze and the ledger allow a
    read of a half (heldout.require_freeze), before the archive is opened."""
    heldout.require_freeze("training")
    return declared(path)


# ---------------------------------------------------------------------------------------------------------------------
# Split and plan


def scheme_key(label):
    """The publisher's scheme order, grid points by row A and column B and then the transition points by number."""
    grid, transition = GRID.fullmatch(label), TRANSITION.fullmatch(label)
    if grid:
        return (0, int(grid.group(1)), int(grid.group(2)))
    if transition:
        return (1, int(transition.group(1)), 0)
    raise ValueError(f"{label} is not a point of the published scheme")


def split(labels):
    """Point identities 1 to 48 in scheme order, with alternate points on the training and the evaluation side."""
    order = sorted(labels, key=scheme_key)
    if len(order) != POINTS:
        raise ValueError("the published scheme holds 48 points")
    identities = {label: index + 1 for index, label in enumerate(order)}
    halves = {side: [identities[label] for index, label in enumerate(order) if index % 2 == position]
              for position, side in enumerate(heldout.SPLITS)}
    return identities, halves


def unit_id(side, anchors, replicate):
    return 10000 * (1 + heldout.SPLITS.index(side)) + 100 * anchors + replicate


def plan(side, points):
    """Every unit of a half with its identity, anchor count, replicate, agents, anchors and window."""
    if len(points) < AGENTS_PER_UNIT:
        raise ValueError("a half holds fewer points than a unit's agents")
    units = []
    for count in CELLS:
        for replicate in range(UNITS_PER_CELL):
            draw = (side, count, replicate)
            agents = sorted(heldout.ordered(points, SEED, *draw, "agents")[:AGENTS_PER_UNIT])
            anchors = sorted(heldout.ordered(sorted(ANCHOR_IDS.values()), SEED, *draw, "anchors")[:count])
            units.append(dict(id=unit_id(side, count, replicate), anchors=count, replicate=replicate, agents=agents, anchor_ids=anchors,
                              window=heldout.hash_order(SEED, *draw, "window") % WINDOWS))
    return units


def window_readings(epochs, anchor, window):
    """The published readings of one link in one window, in time order, the ten-reading block of the link."""
    selected = epochs[heldout.BLOCK_READINGS * window:heldout.BLOCK_READINGS * (window + 1)]
    if len(selected) != heldout.BLOCK_READINGS:
        raise ValueError("an incomplete epoch window")
    return [ranges[anchor] for _, ranges in selected if ranges[anchor] is not None]


def first_reading(epochs, anchor, window):
    """The published reading of one link in the first epoch of a window, epoch 10w, the one-reading block of the link.
    None when that epoch holds no reading, a failed exchange of an anchor the tag polled, since every ranging round of
    the tag's raw log lists all ten anchors (the declaration's disclosures), so a later answer of the window does not
    stand in for it."""
    selected = epochs[heldout.BLOCK_READINGS * window:heldout.BLOCK_READINGS * (window + 1)]
    if len(selected) != heldout.BLOCK_READINGS:
        raise ValueError("an incomplete epoch window")
    return selected[0][1][anchor]


# ---------------------------------------------------------------------------------------------------------------------
# Calibration and units


def slant_cm(a, b):
    return math.dist(a, b)


def link_readings(data, label, anchor):
    """Every published reading of one point to one anchor, in time order."""
    return np.array([r[anchor] for _, r in data["epochs"][label] if r[anchor] is not None], dtype=float)


def failed_exchanges(data, labels):
    """The published readings of the given points and how many of them fall outside the admissible interval."""
    values = np.concatenate([link_readings(data, label, anchor) for label in sorted(labels) for anchor in sorted(data["anchors"])])
    return dict(readings=int(len(values)), inadmissible=int((~heldout.admissible(values)).sum()))


def calibrate(data, labels):
    """The packet scale of the deployment from the admissible readings of the training points, equal weight per link,
    refused before the method freeze and the ledger allow a read of the training half. It counts the readings of
    evaluation points among those it read, which the builders require to be none."""
    heldout.require_freeze("training")
    identities, halves = split(data["points"])
    evaluation = {label for label, identity in identities.items() if identity in halves["evaluation"]}
    errors, readings = [], 0
    for label in sorted(labels):
        for anchor, position in sorted(data["anchors"].items()):
            values = link_readings(data, label, anchor)
            values = values[heldout.admissible(values)]
            if not len(values):
                continue
            residual = values - slant_cm(data["points"][label], position)
            errors.append(float(np.mean(residual ** 2)))
            readings += len(values)
    scale = float(np.sqrt(np.mean(errors))) if errors else 0.0
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("no positive training range uncertainty")
    return dict(dataset="hymn", rule=("Ranges as published, the publisher documents no UWB range calibration. One packet scale, "
                                      "the RMS residual of every admissible published reading of the training points against the "
                                      "slant distance, equal weight per point to anchor link."),
                correction=dict(gain=1.0, offset_cm=0.0), scale_cm=scale, training_points=len(labels), training_links=len(errors),
                training_readings=readings, failed_exchanges=failed_exchanges(data, labels),
                evaluation_rows_used=sum(len(link_readings(data, label, anchor)) for label in labels if label in evaluation for anchor in data["anchors"]))


def area_rule():
    solid = str(heldout.ROOT / "research/solid")
    if solid not in sys.path:
        sys.path.insert(0, solid)
    import deployment
    return lambda prepared: deployment.heldout_unit_area(prepared, "hymn")


def construct(data, identities, specs, scale_cm):
    """The adapter units of both tracks and the reference metadata of a half's plan, refused before the method freeze and
    the ledger allow a read of a half, and for any unit whose agents hold a point of the declared evaluation half
    before the ledger opens HYMN (heldout.require_freeze of the evaluation half), before any reading is taken."""
    heldout.require_freeze("training")
    labels = {identity: label for label, identity in identities.items()}
    if heldout.evaluation_units("hymn", [dict(points=[labels[agent] for agent in spec["agents"]]) for spec in specs]):
        heldout.require_freeze("evaluation", dataset="hymn")
    rule = area_rule()
    units, metadata = [], []
    for spec in specs:
        links, first = {}, {}
        for agent in spec["agents"]:
            epochs = data["epochs"][labels[agent]]
            for anchor in spec["anchor_ids"]:
                values = window_readings(epochs, anchor, spec["window"])
                if values:
                    links[(agent, anchor)] = values
                reading = first_reading(epochs, anchor, spec["window"])
                if reading is not None:
                    first[(agent, anchor)] = [reading]
        anchors = {anchor: data["anchors"][anchor] for anchor in spec["anchor_ids"]}
        units.extend(heldout.units_of(links, spec["agents"], anchors, spec["id"], SEED, scale_cm, rule, first))
        metadata.append(dict(id=spec["id"], acquisition=heldout.cell(spec["anchors"]), anchors=spec["anchors"],
                             replicate=spec["replicate"], window=spec["window"], agents=spec["agents"],
                             points=[labels[a] for a in spec["agents"]], anchor_ids=spec["anchor_ids"]))
    return units, metadata


def reference(data, identities, side_points, metadata):
    """The scorer's reference of a half. Every unit scores against the true positions of its own agents, so its cap is
    the diameter of its agents as in the released scorer. positions_cm holds every point of the half in three dimensions."""
    labels = {identity: label for label, identity in identities.items()}
    units = [dict(entry, reference=str(entry["id"])) for entry in metadata]
    return dict(units=units, labels={str(p): labels[p] for p in side_points},
                coordinates_cm={str(entry["id"]): {str(a): list(data["points"][labels[a]][:2]) for a in entry["agents"]} for entry in units},
                positions_cm={str(p): list(data["points"][labels[p]]) for p in side_points})


# ---------------------------------------------------------------------------------------------------------------------
# Declaration and build


def declaration(data=None):
    data = data or declared()
    identities, halves = split(data["points"])
    shortest = min(len(items) for items in data["epochs"].values())
    if shortest < heldout.BLOCK_READINGS * WINDOWS:
        raise ValueError("a point holds fewer than the declared complete windows")
    plans = {side: plan(side, halves[side]) for side in heldout.SPLITS}
    labels = {identity: label for label, identity in identities.items()}
    return dict(
        archive=dict(path=heldout.relative(ARCHIVE), md5=ARCHIVE_MD5, sha256=ARCHIVE_SHA256, source="Zenodo record 19566178, HYMN dataset release 1.3",
                     licence="Data CC BY 4.0, code MIT"),
        extraction=dict(rule=("The four members below are read from the archive in memory. No other member is read and nothing is "
                              "extracted to disk."), prefix=PREFIX, members=MEMBERS, members_sha256=data["members"]),
        anchors=dict(count=len(ANCHORS), identities={name: ANCHOR_IDS[name] for name in ANCHORS},
                     coordinates="X, Y and Z_LOCAL of the UWB rows of the anchor reference, metres converted to centimetres"),
        points=dict(count=POINTS, epochs_per_point=dict(minimum=shortest, maximum=max(len(v) for v in data["epochs"].values())),
                    epoch_rule="The rows of the processed UWB file in time order per point, each listing all ten anchors, NaN for no reading"),
        blank=("A blank reading, NaN, is a failed exchange of an anchor the tag polled. Every ranging round of the raw ZigPos logs "
               "data/raw/uwb/log_zigpos_twr_*.sqlite lists all ten anchors once with a status, so every anchor is polled in every "
               "epoch and a blank is never an anchor left out of a round. The disclosures state how the logs were inspected."),
        reference=("Surveyed in the publisher's local tangent plane frame centred at the tachymeter station (reference README). "
                   "The tag position per point is X, Y and Z_LOCAL_UWB of the processed UWB file, equal to the UWB2 columns of "
                   "the point reference. Horizontal coordinates score the units, heights enter the slant distances of the "
                   "training fits and the deployment's height prior."),
        calibration=dict(rule=("None documented by the publisher. The ranges enter as published, converted from metres. One packet "
                               "scale, the RMS residual of every admissible published reading of the training points against the "
                               "slant distance with equal weight per point to anchor link, is computed from the training half and "
                               "applied to both halves."), missing="NaN, including a published nonpositive range, is no reading",
                         inadmissible=("A published reading beyond the range of 200 ns of flight is a failed exchange and stays out of "
                                       "the samples. The interval is the engine's and not the hall's, so a gross error between the "
                                       "largest surveyed distance and the end of the interval stays an admissible reading."),
                         failed_exchanges_in_training=failed_exchanges(data, [labels[p] for p in halves["training"]])),
        extent=("None. The archive and the publisher's pages give no hall extent, so the operating area of a unit is its padded "
                "anchor box by the deployment rules."),
        split=dict(rule=("The 48 points in the publisher's scheme order, grid points by row A then column B and the transition "
                         "points T01 to T06 last, alternate between training and evaluation starting with training. Neighbouring "
                         "points of every row fall on opposite sides, so both halves span the hall's length and width, the lower "
                         "rows A01 and A07 and the transition strip, and every evaluation position is new. The rule reads only "
                         "the point labels."),
                   identities={label: identities[label] for label in sorted(identities, key=scheme_key)},
                   **{f"{side}_points": [labels[p] for p in halves[side]] for side in heldout.SPLITS}),
        units=dict(rule=("A unit takes eight points of one half as agents, k of the ten anchors and one window of ten consecutive "
                         "epochs shared by its agents, window w holding the epochs 10w to 10w + 9 of a point in time order. Each "
                         "agent to anchor link carries its published readings in the window in the ten-reading track and the "
                         "reading of the window's first epoch 10w alone in the one-reading track. A link whose first epoch holds "
                         "no reading, a failed exchange (blank), has no observation in the one-reading track even when a later "
                         "epoch of the window answered, and a link without a reading in the window has no observation in either "
                         "track. Agents, anchors and window are drawn by hash order from the seed and the unit's half, anchor "
                         "count and replicate."),
                   agents_per_unit=AGENTS_PER_UNIT, windows=WINDOWS, epochs_per_window=heldout.BLOCK_READINGS, tracks=list(heldout.TRACKS),
                   seed=SEED, cells=[heldout.cell(k) for k in CELLS], units_per_cell=UNITS_PER_CELL,
                   identity="10000 (1 for training, 2 for evaluation) + 100 anchor count + replicate",
                   agent_identities="the point identities of the split", anchor_identities="UWB_01 to UWB_10 as 101 to 110"),
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
    freeze = heldout.require_freeze(side, dataset="hymn")
    declared = heldout.section("hymn")
    data = load()
    if heldout.canonical(declaration(data)) != declared:
        raise ValueError("the HYMN archive or rules differ from the declaration")
    identities, halves = split(data["points"])
    declaration_sha256 = heldout.sha256(heldout.RECORD)
    if side == "training":
        calibration = dict(calibrate(data, [label for label, identity in identities.items() if identity in halves["training"]]),
                           archive_sha256=ARCHIVE_SHA256, declaration_sha256=declaration_sha256)
        keep_or_write(calibration_path(), calibration)
    calibration = heldout.training_calibration(calibration_path(), ARCHIVE_SHA256)
    specs = plan(side, halves[side])
    if heldout.fingerprint(specs) != declared["plan"][side]["sha256"]:
        raise ValueError("the HYMN plan differs from the declaration")
    units, metadata = construct(data, identities, specs, calibration["scale_cm"])
    counts = heldout.population(units, metadata)
    if counts["identities"] != declared["plan"][side]["identities"]:
        raise ValueError("the built HYMN half differs from its declared plan")
    files = outputs(side)
    keep_or_write(files["request"], dict(units=units))
    keep_or_write(files["reference"], reference(data, identities, halves[side], metadata))
    keep_or_write(files["population"], dict(dataset="hymn", split=side, counts=counts, scale_cm=calibration["scale_cm"],
                                            calibration_sha256=heldout.sha256(calibration_path()),
                                            declaration_sha256=heldout.sha256(heldout.RECORD),
                                            **(freeze if side == "evaluation" else {}),
                                            files={name: heldout.sha256(path) for name, path in files.items() if name != "population"}))
    print(json.dumps(dict(split=side, **counts)), flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--split", required=True, choices=heldout.SPLITS)
    args = parser.parse_args()
    build(args.split)


if __name__ == "__main__":
    main()
