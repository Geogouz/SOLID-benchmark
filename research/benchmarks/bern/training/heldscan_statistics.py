"""Composition of TWR-CLOUD-Bern scenario corpora, the numbers that set the held-scan corpora beside the evaluation corpus.

For every anchor count the statistics give the scenarios, the agents per scenario, the share of agents inside the
planar convex hull of the scenario's anchors, and the ten packet track of the anchor links and the agent links. A link
is empty (no packet), dropped (packets exist but the ten packet track is empty, the old generator's treatment of links
with one to nine packets), partial (one to nine packets in the ten packet track) or full (ten packets). Corpora built
by the release's rules are counted from their groups and the table's packet counts, any other archive from its JSON
members. Both counts agree on the same scenarios, which the tests check.
"""
import json
import zipfile
from collections import defaultdict

import numpy as np

import release_rule as rule

CATEGORIES = ("empty", "dropped", "partial", "full")


def convex_hull(points):
    """Andrew's monotone chain. Counterclockwise vertices without collinear points."""
    points = sorted(set(points))
    if len(points) <= 2:
        return points

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def encloses(vertices, point):
    """Inside or on the boundary of a counterclockwise convex polygon."""
    if len(vertices) < 3:
        return False
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) < 0:
            return False
    return True


def category(packets):
    """Category index of a link with this many packets under the release's rule."""
    return 0 if packets == 0 else (2 if packets < rule.TEN_PACKETS else 3)


class Tally:
    def __init__(self):
        self.scenarios = 0
        self.agents = 0
        self.enclosed = 0
        self.agent_counts = defaultdict(int)
        self.anchor_links = np.zeros(4, dtype=np.int64)
        self.agent_links = np.zeros(4, dtype=np.int64)

    def add(self, other):
        self.scenarios += other.scenarios
        self.agents += other.agents
        self.enclosed += other.enclosed
        for count, value in other.agent_counts.items():
            self.agent_counts[count] += value
        self.anchor_links += other.anchor_links
        self.agent_links += other.agent_links

    def record(self, anchored):
        def links(values):
            total = int(values.sum())
            with_packets = int(values[1:].sum())
            entry = {name: int(value) for name, value in zip(CATEGORIES, values)}
            entry.update(total=total, one_to_nine_share_of_links_with_packets=(values[1] + values[2]) / with_packets if with_packets else None,
                         partial_share_of_links_with_packets=values[2] / with_packets if with_packets else None)
            return entry

        return dict(scenarios=self.scenarios, mean_agents=self.agents / self.scenarios if self.scenarios else None,
                    agent_counts={str(k): v for k, v in sorted(self.agent_counts.items())},
                    enclosed_agent_share=(self.enclosed / self.agents if anchored and self.agents else None),
                    anchor_links=links(self.anchor_links), agent_links=links(self.agent_links))


def summarize(tallies):
    """Per anchor count records and the pooled anchored levels."""
    levels = {str(count): tally.record(count > 0) for count, tally in sorted(tallies.items())}
    pooled = Tally()
    for count, tally in tallies.items():
        if count > 0:
            pooled.add(tally)
    levels["anchored"] = pooled.record(True)
    return levels


def design_statistics(design, links, numbers):
    """Statistics of release rule scenarios `numbers` of one scan from the groups and the packet counts."""
    layout = design.layout
    index = {position: i for i, position in enumerate(sorted(layout.devices))}
    counts = np.zeros((len(index), len(index)), dtype=np.int64)
    for (responder, poller), values in links.items():
        counts[index[responder], index[poller]] = len(values)
    categories = np.vectorize(category)(counts)
    wanted = defaultdict(list)
    for number in numbers:
        agent_count, anchor_count, rank, _, _ = design.combination(number)
        wanted[agent_count, anchor_count].append(rank)
    tallies = defaultdict(Tally)
    for (agent_count, anchor_count), ranks in wanted.items():
        groups = np.array([[index[a] for a in design.agents[agent_count][rank]] for rank in ranks], dtype=np.int64)
        anchors = np.array([index[a] for a in design.anchors[anchor_count]], dtype=np.int64)
        tally = tallies[anchor_count]
        tally.scenarios += len(ranks)
        tally.agents += groups.size
        tally.agent_counts[agent_count] += len(ranks)
        if len(anchors):
            vertices = convex_hull([layout.coordinates[a][:2] for a in design.anchors[anchor_count]])
            inside = np.array([encloses(vertices, layout.coordinates[p][:2]) for p in sorted(layout.devices)])
            tally.enclosed += int(inside[groups].sum())
            # anchor link agent -> anchor reads the rows where the agent responded to the anchor
            tally.anchor_links += np.bincount(categories[groups[:, :, None], anchors[None, None, :]].ravel(), minlength=4)
        # agent link a -> b reads the rows where a polled b, so the responder is b
        pair = categories[groups[:, None, :], groups[:, :, None]]
        off = ~np.eye(agent_count, dtype=bool)
        tally.agent_links += np.bincount(pair[:, off].ravel(), minlength=4)
    return tallies


def _json_category(value):
    ten = (value or {}).get("10")
    one = (value or {}).get("1")
    if ten is None:
        return 0 if one is None else 1
    return 2 if len(ten) < rule.TEN_PACKETS else 3


def archive_statistics(path, accept=None):
    """Statistics of the JSON members of an archive, the first iteration's links."""
    tallies = defaultdict(Tally)
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json") or "manifest" in name or (accept is not None and not accept(name)):
                continue
            data = json.loads(archive.read(name))
            agents, anchors = data["Agents"], data["Anchors"]
            tally = tallies[len(anchors)]
            tally.scenarios += 1
            tally.agents += len(agents)
            tally.agent_counts[len(agents)] += 1
            if anchors:
                truth = data["True Positions of Anchors"]
                vertices = convex_hull([(truth[str(a)]["x"], truth[str(a)]["y"]) for a in anchors])
                agents_truth = data["True Positions of Agents"]
                tally.enclosed += sum(encloses(vertices, (agents_truth[str(a)]["x"], agents_truth[str(a)]["y"])) for a in agents)
            iteration = data["Iterations"][0]
            for value in iteration["Ranging samples to Anchors"].values():
                tally.anchor_links[_json_category(value)] += 1
            for value in iteration["Ranging samples between Agents"].values():
                tally.agent_links[_json_category(value)] += 1
    return tallies


def superset_enclosure(design, counts=(4, 8)):
    """Share of the 20 agent positions inside the planar hull of each anchor group, the auditor's measure."""
    coordinates = design.layout.coordinates
    agents = sorted(set(design.layout.devices) - set(design.anchors[rule.SUPERSET_SIZE]))
    result = {}
    for count in counts:
        vertices = convex_hull([coordinates[a][:2] for a in design.anchors[count]])
        result[str(count)] = sum(encloses(vertices, coordinates[a][:2]) for a in agents) / len(agents)
    return result


def nested_rule_enclosure(layout, counts=(4, 8)):
    """The same measure for the old generator's nested farthest point groups on the same positions."""
    import generate_training_scenarios as old

    positions = {p: tuple(value * 100.0 for value in xyz) for p, xyz in layout.coordinates.items()}
    groups = old.nested_anchor_groups(positions)
    agents = sorted(set(positions) - set(groups[rule.SUPERSET_SIZE]))
    result = {}
    for count in counts:
        vertices = convex_hull([positions[a][:2] for a in groups[count]])
        result[str(count)] = sum(encloses(vertices, positions[a][:2]) for a in agents) / len(agents)
    return result, {str(count): list(group) for count, group in groups.items()}


def position_changes(corrected, published):
    """Devices whose position differs between the published and the corrected table, per scan."""
    changes = {}
    for scan in sorted(corrected):
        new, old = corrected[scan], published[scan]
        moved = []
        for device, (after, before) in enumerate(zip(new.devices, old.devices), start=1):
            if after != before:
                a, b = new.coordinates[after], old.coordinates[before]
                moved.append(dict(device=device, published_position=before, corrected_position=after,
                                  planar_shift_cm=100 * float(np.hypot(a[0] - b[0], a[1] - b[1]))))
        shared = [p for p in new.devices if p in old.coordinates]
        coordinates_agree = all(new.coordinates[p] == old.coordinates[p] for p in shared)
        changes[str(scan)] = dict(moved=moved, coordinates_agree_on_shared_positions=coordinates_agree)
    return changes


def summary(plans, data, published, cooperative_groups, anchor_only_groups, anchor_only_counts, old_cooperative=None, old_anchor_only=None):
    """The statistics record of the evaluation corpus, both held-scan designs and, when given, the old corpora."""
    corpora = {}
    evaluation = plans[rule.EVALUATION_SCAN]
    corpora["evaluation release, scan 12, all scenarios"] = summarize(
        design_statistics(evaluation, data[rule.EVALUATION_SCAN][1], evaluation.numbers()))
    for label, groups, counts in (("cooperative v2", cooperative_groups, rule.ANCHOR_COUNTS),
                                  ("anchor only v2", anchor_only_groups, anchor_only_counts)):
        pooled = defaultdict(Tally)
        for scan in rule.TRAINING_SCANS:
            for count, tally in design_statistics(plans[scan], data[scan][1], plans[scan].numbers(groups, counts)).items():
                pooled[count].add(tally)
        corpora[f"{label}, scans 1 to 11, {groups} groups per cell"] = summarize(pooled)
    if old_cooperative is not None:
        corpora["cooperative v1 (old archive)"] = summarize(archive_statistics(old_cooperative))
    if old_anchor_only is not None:
        pooled = defaultdict(Tally)
        for scan in rule.TRAINING_SCANS:
            for count, tally in archive_statistics(old_anchor_only / f"scan-{scan:02d}/scenarios.zip").items():
                pooled[count].add(tally)
        corpora["anchor only v1 (old campaign)"] = summarize(pooled)
    designs = {}
    for scan, plan in sorted(plans.items()):
        nested, nested_groups = nested_rule_enclosure(plan.layout)
        designs[str(scan)] = dict(anchor_groups={str(k): list(v) for k, v in plan.anchors.items()},
                                  superset_min_distance_m=plan.superset_min_distance(),
                                  enclosed_superset_agents=superset_enclosure(plan),
                                  ties=rule.tie_report(plan),
                                  old_nested_rule=dict(anchor_groups=nested_groups, enclosed_agents=nested))
    corrected = {scan: data[scan][0] for scan in data}
    return dict(schema="solid.training.heldscan-statistics.v2", corpora=corpora, designs=designs,
                positions=position_changes(corrected, published))


def table_text(record):
    lines = []
    for label, levels in record["corpora"].items():
        lines.append(label)
        lines.append("  anchors  scenarios  mean agents  enclosed  anchor links 1-9 of linked (partial/dropped)  agent links 1-9 of linked (partial/dropped)")
        for level, entry in levels.items():
            enclosed = "" if entry["enclosed_agent_share"] is None else f"{100 * entry['enclosed_agent_share']:.1f}%"
            parts = []
            for key in ("anchor_links", "agent_links"):
                links = entry[key]
                share = links["one_to_nine_share_of_links_with_packets"]
                parts.append("" if share is None else f"{100 * share:.2f}% ({links['partial']}/{links['dropped']})")
            lines.append(f"  {level:>8}  {entry['scenarios']:>9}  {entry['mean_agents']:>11.2f}  {enclosed:>8}  {parts[0]:>44}  {parts[1]:>43}")
    lines.append("enclosed superset agents, release rule (old nested rule)")
    for scan, entry in record["designs"].items():
        new, old = entry["enclosed_superset_agents"], entry["old_nested_rule"]["enclosed_agents"]
        lines.append(f"  scan {scan:>2}  4 anchors {100 * new['4']:5.1f}% ({100 * old['4']:5.1f}%)  8 anchors {100 * new['8']:5.1f}% ({100 * old['8']:5.1f}%)")
    for scan, entry in record["positions"].items():
        for moved in entry["moved"]:
            lines.append(f"  scan {scan} device {moved['device']}: position {moved['published_position']} -> {moved['corrected_position']}, "
                         f"{moved['planar_shift_cm']:.0f} cm")
    return "\n".join(lines)
