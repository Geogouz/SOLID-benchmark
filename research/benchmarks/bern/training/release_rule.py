"""The scenario generation rules of the TWR-CLOUD-Bern evaluation release, applicable to every scan.

The release generator (reproduce_twr_cloud_bern_release.py in the correction package under dataset-corrections of the
data folder) writes the evaluation scenarios of scan 12 in five steps. This module applies the
same five steps to the positions and measurements of any scan of the corrected responder table.

1. The anchor superset is the set of 20 of the scan's 40 positions with the largest minimum pairwise 3D distance. Ties
   go to the lexicographically larger sorted list of all pairwise distances. The release searched a pruned space in
   which four near-duplicate scan 12 positions were removed and ten dispersed scan 12 positions were forced in. Those
   identifiers exist only in scan 12. The exact search below needs no pruning, and on scan 12 its unique optimum is
   the release's superset, which also satisfies both pruning lists. The pruning therefore never changed the result.
2. The anchor groups of 16, 12, 8 and 4 anchors are the best subsets of the superset under the same criterion, each
   chosen on its own and not nested in the others. The anchor free level has no anchors.
3. The agents are the 20 positions outside the superset. For each agent count of 4, 7, 10, 13, 16 and 19 all agent
   combinations are sorted by their mean pairwise distance and thinned uniformly by rank to at most 5000 groups.
4. Scenarios are numbered with the agent count first, then the anchor count, then the agent group. Scenario c draws
   its first iteration's packets after seeding NumPy's legacy generator with c, its second iteration's packets after
   seeding with (c - 1) * 50 + 1 and its second iteration's starting positions after seeding with (c - 1) * 50 + 2.
   The first iteration starts every agent at the origin.
5. Every directed link draws one packet and then up to ten packets without replacement from its measurements in
   table order. Anchor links read the rows in which the agent responded to the anchor, agent links the rows in which
   the first agent polled the second. A link with one to nine packets keeps all of them in the ten packet track.

The selector and sampling helpers below copy the release's code, so the tie handling and the floating point
arithmetic are the release's own. Pandas' Series.sample, which the release calls, draws its rows with
RandomState.choice without replacement, and link_samples calls the same function directly.
"""
import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np

EVALUATION_SCAN = 12
TRAINING_SCANS = tuple(range(1, 12))
POSITIONS_PER_SCAN = 40
SUPERSET_SIZE = 20
ANCHOR_COUNTS = (0, 4, 8, 12, 16, 20)
AGENT_COUNTS = (4, 7, 10, 13, 16, 19)
AGENT_GROUP_LIMIT = 5000
SEED_STRIDE = 50
ITERATIONS = 2
INITIALIZATION_SPANS = (3000, 1500, 300)
SAMPLE_COUNTS = (1, 10)
TEN_PACKETS = 10
# The release's pruning inputs for scan 12. The tests check that the exact search agrees with them.
RELEASE_EXCLUDED = frozenset({10, 27, 148, 154})
RELEASE_FORCED = (9, 16, 28, 60, 90, 109, 113, 139, 155, 169)
CORRECTED_TABLE_MD5 = "d5d71fe55c758c2a87d7efd812357c5b"
PUBLISHED_TABLE_MD5 = "ddca1cc9de31e1e525c6597519f040a2"
EVALUATION_ARCHIVE_MD5 = "757319458e2c159caddbb27bbc2d49c0"
EVALUATION_SCENARIOS = 148260


@dataclass(frozen=True)
class Layout:
    """The deployment of one scan. `devices` lists the position of device 1 to 40 like the release's
    scenario_N_positions lists, `coordinates` maps every position to its surveyed (x, y, z) in metres."""

    scan: int
    devices: tuple
    coordinates: dict


def file_digest(path, algorithm="md5"):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table(path, expected_md5=CORRECTED_TABLE_MD5):
    """The corrected responder table in its released row order."""
    import pandas as pd

    if expected_md5 is not None and file_digest(path, "md5") != expected_md5:
        raise ValueError(f"{path} is not the corrected responder table (md5 {expected_md5})")
    return pd.read_pickle(path)


def scan_layout(frame, scan):
    """Device positions and coordinates of one scan, read from both endpoints of every row."""
    rows = frame[frame["Scan_ID"] == scan]
    devices, coordinates = {}, {}
    for role in ("Responder", "Poller"):
        columns = [f"{role}_ID", f"{role}_Pos_ID", f"{role}_X", f"{role}_Y", f"{role}_Z"]
        for device, position, x, y, z in rows[columns].drop_duplicates().itertuples(index=False):
            device, position, point = int(device), int(position), (float(x), float(y), float(z))
            if devices.setdefault(device, position) != position:
                raise ValueError(f"scan {scan} places device {device} at two positions")
            if coordinates.setdefault(position, point) != point:
                raise ValueError(f"scan {scan} gives position {position} two coordinates")
    if sorted(devices) != list(range(1, POSITIONS_PER_SCAN + 1)) or len(coordinates) != POSITIONS_PER_SCAN:
        raise ValueError(f"scan {scan} does not hold 40 devices at 40 positions")
    return Layout(scan, tuple(devices[device] for device in range(1, POSITIONS_PER_SCAN + 1)), coordinates)


def scan_links(frame, scan):
    """Time of flight values of every directed link of one scan in table order, keyed (responder, poller)."""
    rows = frame[frame["Scan_ID"] == scan]
    responders = rows["Responder_Pos_ID"].to_numpy(dtype=np.int64)
    pollers = rows["Poller_Pos_ID"].to_numpy(dtype=np.int64)
    tof = rows["ToF"].to_numpy(dtype=np.float64)
    keys = responders * 1000 + pollers
    order = np.argsort(keys, kind="stable")
    keys, tof = keys[order], tof[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    return {(int(keys[start] // 1000), int(keys[start] % 1000)): tof[start:end].copy() for start, end in zip(starts, ends)}


# ---- the release's selectors -------------------------------------------------------------------------------------


def pairwise_distances(position_ids, coordinates):
    """The release's _pairwise_distances with the scan's coordinates."""
    distances = {}
    for node1 in position_ids:
        for node2 in position_ids:
            if node1 == node2:
                continue
            x1, y1, z1 = coordinates[node1]
            x2, y2, z2 = coordinates[node2]
            distances[node1, node2] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distances


def _mean_pairwise_distance(group, distances):
    total = 0
    count = 0
    for pair in itertools.combinations(group, 2):
        total += distances[pair]
        count += 1
    return total / count if count else 0


def _agent_combos_sorted_by_mean_distance(combos, distances):
    means = {tuple(sorted(combo)): _mean_pairwise_distance(combo, distances) for combo in combos}
    return sorted(means, key=lambda group: means[group])


def select_k_uniform_elements(values, k):
    """The release's thinning by rank."""
    if len(values) <= k:
        return values
    step = len(values) / k
    return [values[round(i * step)] for i in range(k)]


def _sorted_pairwise_distances_in_group(group, distances):
    sorted_distances = sorted(distances[pair] for pair in itertools.combinations(group, 2))
    return sorted_distances[0], sorted_distances


def _new_anchor_group_is_better(best_distances, new_distances):
    for best_distance, new_distance in zip(best_distances, new_distances):
        if best_distance < new_distance:
            return True
        if best_distance > new_distance:
            return False
    return False


def best_max_min_group(group_combos, distances):
    """The release's _best_max_min_group, the group with the largest minimum pair distance."""
    best_group = None
    best_min_distance = 0
    best_sorted_distances = []
    for group in group_combos:
        min_distance, sorted_distances = _sorted_pairwise_distances_in_group(group, distances)
        if best_min_distance < min_distance:
            best_group = group
            best_min_distance = min_distance
            best_sorted_distances = sorted_distances
        elif best_min_distance == min_distance and _new_anchor_group_is_better(best_sorted_distances, sorted_distances):
            best_group = group
            best_sorted_distances = sorted_distances
    return set(best_group)


# ---- the exact superset search -----------------------------------------------------------------------------------


def _compatibility(ids, distances, threshold):
    """Bit masks of the positions at least `threshold` metres from each position."""
    masks = [0] * len(ids)
    for i, left in enumerate(ids):
        for j, right in enumerate(ids):
            if i != j and distances[left, right] >= threshold:
                masks[i] |= 1 << j
    return masks


def _coloring_bound(candidates, masks):
    """Greedy colouring of the candidates. A set of mutually compatible positions holds at most one per colour."""
    colors = 0
    uncolored = candidates
    while uncolored:
        colors += 1
        available = uncolored
        while available:
            low = available & -available
            uncolored &= ~low
            available &= ~low & ~masks[low.bit_length() - 1]
    return colors


def _compatible_sets(masks, size, limit):
    """Every set of `size` mutually compatible positions as a bit mask, at most `limit` of them."""
    found = []

    def extend(chosen, count, candidates):
        if count == size:
            found.append(chosen)
            return len(found) < limit
        if count + _coloring_bound(candidates, masks) < size:
            return True
        while candidates:
            if count + bin(candidates).count("1") < size:
                return True
            low = candidates & -candidates
            candidates ^= low
            if not extend(chosen | low, count + 1, candidates & masks[low.bit_length() - 1]):
                return False
        return True

    extend(0, 0, (1 << len(masks)) - 1)
    return found


def max_min_superset(position_ids, distances, size=SUPERSET_SIZE):
    """The exact solution of the release's superset criterion over all positions.

    The largest achievable minimum distance is found by bisection over the pairwise distances, every set reaching it
    is enumerated, and the lexicographically largest sorted distance list wins. The winner must be unique, which holds
    on every Bern scan, so the result does not depend on any enumeration order."""
    ids = sorted(position_ids)
    thresholds = sorted({distances[pair] for pair in itertools.combinations(ids, 2)})
    low, high = 0, len(thresholds) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _compatible_sets(_compatibility(ids, distances, thresholds[middle]), size, 1):
            low = middle
        else:
            high = middle - 1
    sets = _compatible_sets(_compatibility(ids, distances, thresholds[low]), size, 10 ** 7)
    groups = [tuple(ids[j] for j in range(len(ids)) if mask >> j & 1) for mask in sets]
    profiles = {group: _sorted_pairwise_distances_in_group(group, distances)[1] for group in groups}
    best = max(groups, key=lambda group: profiles[group])
    if sum(profiles[group] == profiles[best] for group in groups) != 1:
        raise ValueError("the superset criterion has several optima")
    return best


def anchor_groups(layout):
    """Anchor groups per anchor count in the release's order 0, 20, 16, 12, 8, 4."""
    positions = set(layout.devices)
    distances = pairwise_distances(sorted(positions), layout.coordinates)
    superset = set(max_min_superset(positions, distances))
    groups = {0: ()}
    for count in range(SUPERSET_SIZE, 0, -4):
        best = superset if count == SUPERSET_SIZE else best_max_min_group(itertools.combinations(superset, count), distances)
        groups[count] = tuple(sorted(best))
    return groups


def agent_groups(layout, anchors):
    """The release's build_agent_groups on the scan's positions."""
    agents_superset = set(layout.devices) - set(anchors[SUPERSET_SIZE])
    distances = pairwise_distances(agents_superset, layout.coordinates)
    selected = {}
    for count in AGENT_COUNTS:
        combos = itertools.combinations(agents_superset, count)
        selected[count] = select_k_uniform_elements(_agent_combos_sorted_by_mean_distance(combos, distances), AGENT_GROUP_LIMIT)
    return selected


@dataclass(frozen=True)
class Design:
    """The release's groups of one scan and the numbering of its full scenario corpus."""

    layout: Layout
    anchors: dict
    agents: dict

    def cells(self):
        """(agent count, anchor count, first scenario number, groups) in the release's order."""
        first = 1
        for agent_count in sorted(self.agents):
            for anchor_count in sorted(self.anchors):
                yield agent_count, anchor_count, first, len(self.agents[agent_count])
                first += len(self.agents[agent_count])

    def scenario_count(self):
        return sum(cell[3] for cell in self.cells())

    def combination(self, number):
        """Agent count, anchor count, agent group rank, agents and anchors of scenario `number`."""
        for agent_count, anchor_count, first, groups in self.cells():
            if first <= number < first + groups:
                rank = number - first
                return agent_count, anchor_count, rank, self.agents[agent_count][rank], self.anchors[anchor_count]
        raise ValueError(f"scenario {number} is outside the corpus")

    def kept_ranks(self, agent_count, groups_per_cell=None):
        """Ranks of the agent groups a corpus keeps for one agent count. The release's groups of that count, already
        sorted by mean pairwise distance, fall into groups_per_cell strata of equal rank width, and the middle group
        of each stratum stays, so the kept groups spread over the whole dispersion range without a shift. All groups
        stay when groups_per_cell is None or not smaller than their number."""
        count = len(self.agents[agent_count])
        if groups_per_cell is None or count <= groups_per_cell:
            return list(range(count))
        return [(2 * stratum + 1) * count // (2 * groups_per_cell) for stratum in range(groups_per_cell)]

    def numbers(self, groups_per_cell=None, anchor_counts=ANCHOR_COUNTS):
        """Scenario numbers of a corpus that keeps the kept_ranks groups of every agent count with every listed anchor
        count. It is a subset of the corpus the release's generator writes for this scan."""
        chosen = []
        for agent_count, anchor_count, first, _ in self.cells():
            if anchor_count in anchor_counts:
                chosen.extend(first + rank for rank in self.kept_ranks(agent_count, groups_per_cell))
        return chosen

    def superset_min_distance(self):
        distances = pairwise_distances(sorted(self.layout.devices), self.layout.coordinates)
        return min(distances[pair] for pair in itertools.combinations(self.anchors[SUPERSET_SIZE], 2))


def design(layout):
    anchors = anchor_groups(layout)
    return Design(layout, anchors, agent_groups(layout, anchors))


def tie_report(design):
    """Exact ties the release's selectors would break by set iteration order, an implementation detail of Python.

    For every anchor count, the number of subsets of the superset with the chosen group's sorted distance list, and
    for every agent count, the number of neighbouring pairs with equal means in the full sorted list of combinations.
    One subset per anchor count and no equal neighbours mean the groups depend on no iteration order."""
    layout = design.layout
    distances = pairwise_distances(sorted(layout.devices), layout.coordinates)
    superset = design.anchors[SUPERSET_SIZE]
    anchor_ties = {}
    for count in (16, 12, 8, 4):
        chosen = _sorted_pairwise_distances_in_group(design.anchors[count], distances)[1]
        anchor_ties[str(count)] = sum(_sorted_pairwise_distances_in_group(group, distances)[1] == chosen
                                      for group in itertools.combinations(superset, count))
    # the release's own construction, so the means carry the release's rounding
    agents_superset = set(layout.devices) - set(superset)
    agent_ties = {}
    for count in AGENT_COUNTS:
        means = sorted(_mean_pairwise_distance(group, distances) for group in itertools.combinations(agents_superset, count))
        agent_ties[str(count)] = sum(left == right for left, right in zip(means, means[1:]))
    return dict(anchor_subsets_with_the_chosen_profile=anchor_ties, equal_neighbouring_agent_means=agent_ties)


# ---- scenarios -----------------------------------------------------------------------------------------------------


def link_samples(values, generator):
    """The release's get_ranging_between_nodes on one link's values in table order."""
    count = 0 if values is None else len(values)
    biggest_sample_size = min(TEN_PACKETS, count)
    if biggest_sample_size == 0:
        return {"1": None, "10": None}
    one = values[generator.choice(count, size=1, replace=False)[0]]
    ten = values[generator.choice(count, size=biggest_sample_size, replace=False)]
    return {"1": float(one), "10": [float(value) for value in ten]}


def _rangings(agents, anchors, links, generator):
    """Anchor links first and agent links second, the release's drawing order."""
    to_anchors = {}
    for agent in agents:
        for anchor in anchors:
            to_anchors[f"{agent}-{anchor}"] = link_samples(links.get((agent, anchor)), generator)
    between_agents = {}
    for agent in agents:
        for neighbor in agents:
            if agent != neighbor:
                between_agents[f"{agent}-{neighbor}"] = link_samples(links.get((neighbor, agent)), generator)
    return to_anchors, between_agents


def _true_positions(nodes, coordinates):
    return {str(node): {"x": coordinates[node][0] * 100, "y": coordinates[node][1] * 100, "z": coordinates[node][2] * 100}
            for node in nodes}


def scenario(number, agents, anchors, coordinates, links):
    """The release's scenario `number` of a scan with these groups, as the dictionary its JSON file holds."""
    generator = np.random.RandomState(number)
    iterations = []
    for iteration in range(1, ITERATIONS + 1):
        to_anchors, between_agents = _rangings(agents, anchors, links, generator)
        generator = np.random.RandomState((number - 1) * SEED_STRIDE + iteration)
        if iteration == 1:
            starts = {str(node): {"x": 0, "y": 0, "z": 0} for node in agents}
        else:
            starts = {str(node): {"x": generator.uniform(0, INITIALIZATION_SPANS[0]),
                                  "y": generator.uniform(0, INITIALIZATION_SPANS[1]),
                                  "z": generator.uniform(0, INITIALIZATION_SPANS[2])} for node in agents}
        iterations.append({"ID": iteration, "Initial positions of Agents": starts,
                           "Ranging samples between Agents": between_agents, "Ranging samples to Anchors": to_anchors})
    return {"Agents": list(agents), "Anchors": list(anchors),
            "True Positions of Agents": _true_positions(agents, coordinates),
            "True Positions of Anchors": _true_positions(anchors, coordinates), "Iterations": iterations}


def release_bytes(data):
    """The bytes of the release's JSON file for a scenario dictionary."""
    return json.dumps(data, indent=4).encode("utf-8")
