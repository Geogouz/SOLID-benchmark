"""Noise-aware decisions of the design rounds: exact differences, a paired cluster bootstrap and fixed multipliers.

Comparisons. A round positions a reference and a candidate on identical units. A comparison k is one deployment
(Bern, TIERS and Bregar by default, a round declaring its own), one mode, one sample track and one percentile, S75 or
S90:

    Delta_k = S(candidate) - S(reference)

with S the balanced percentile of the released scorer's rule in exact rational arithmetic
(common.balanced_percentile_exact); the released scorer's floating point value is recorded beside it. Lower errors
are better, so Delta_k > 0 means the candidate is worse. A comparison may cover a declared scope of the units of its
deployment and mode, for example the cells with three or more anchors. Its pair key then names the scope as a third
element and its record carries it.

Uncertainty. SE_k is the standard deviation of Delta_k over B paired cluster bootstrap draws with a recorded seed.
Both candidates, every mode, every scope, every sample track and both percentiles of a deployment share each draw.
Clusters, field names referring to the unit rows:

    Bern     the eleven scans, with replacement, a scan drawn twice contributing its cells twice
    TIERS    runs of three consecutive blocks within each acquisition
    Bregar   the 24 acquisitions, an environment and a channel, together without strata, an acquisition drawn
             twice contributing its cell twice
    IDLab    the radio configurations, unstratified, a configuration drawn twice contributing its cells twice, never
             a panel or a unit within a panel alone

Scenarios within Bern's cells are not resampled. With the scans resampled, that would count the within-scan variance
twice and overstate SE_k. Single-unit cells are not resampled within themselves either, which would give no variance
at all. The blocks of a TIERS recording are consecutive windows of one static layout, so neighbouring blocks may share
slow changes of the radio channel. Runs of three consecutive blocks are the clusters, about the cube root of the 22 to
25 blocks of a recording, the order of the block length of the block bootstrap for the variance of a statistic of a
dependent series (Hall, Horowitz and Jing 1995). The Bregar interval is conditional on its four environments, whose
six channels share their tag positions and anchors. Drawing the units across the environments adds the spread between
the environments to the draws, which makes the interval conservative for that target. On the audit's null it covered
0.98 to 0.99 at nominal 0.95. Drawing the six channels within their environment undercovers when the environments
differ more than their channels, 0.79 when environment effects dominate, since six channels per environment are too
few, and no design covers a difference across new environments with four of them. The panels of an IDLab configuration
are not independent, since idlab_population.panel takes the records of a short link again in every panel of its
configuration, so the configurations are the clusters, and a cell of IDLab, a configuration and an anchor count, lies
in one of them. The clusters do not depend on the folds of the round (folds.py), which only choose the fit that
positions a unit.

Every deployment has few clusters per stratum, as few as three runs in the shortest AAU flight, seven in a TIERS
recording and eleven scans on Bern, and a draw of n clusters from n shrinks the variance by (n - 1) / n. Every stratum
of n > 1 clusters therefore draws n - 1 of them and every drawn copy counts n / (n - 1) times, the rescaling bootstrap
of Rao and Wu (1988) whose variance is unbiased for a mean. Every stratum thus keeps the weight of its n clusters in
every draw. The weight cancels within a cell that lies in one stratum, and it keeps the share of every stratum in a
cell that spans strata, as an AAU cell spans the flights. Every draw's balanced CDF is summed in floating point and
settled in exact rational arithmetic wherever it comes near a level, so every draw gives the percentile of the exact
CDF, like the point estimate.

The final tables (final_tables.py) draw the designs above on the halves of TIERS, Bregar and IDLab, and three more:

    bern-evaluation  the agent groups of the Bern evaluation corpus within their agent count, a group bringing its
                     scenarios of every anchor count with both iterations and both sample tracks
    hymn             the windows of ten epochs, unstratified
    aau              runs of two consecutive blocks of ten reports within each flight

The Bern evaluation corpus comes from one scan, so its scans cannot be drawn. The release repeats every agent group of
an agent count under every anchor count, the same agents in the same positions drawing their packets from one pool of
the scan's readings, so an agent group with its scenarios of every anchor count is the cluster. The groups are drawn
within their agent count, whose number of groups the release fixes, and the interval is conditional on the scan, its
positions and its pool of readings. A HYMN unit takes eight points of its half and one window of ten epochs, the same
window for every agent. Two units share a reading only when they share its window, so the windows are the clusters. A
point cannot be one, since every unit spans eight of them, and the interval is conditional on the half's 24 points
and the ten anchors. The twelve windows are too few to join into runs, which would leave six clusters, so the
interval takes neighbouring windows of a point as independent given the points. An AAU unit lies in one block of ten
reports of one flight. Runs of two consecutive blocks, about the cube root of the 7 to 13 blocks of a flight, are
drawn within their flight as TIERS draws its runs within recordings. Every flight keeps the weight of its runs in
every draw, and the nine static nodes that every flight shares are conditioned on.

Margins and verdicts, m_k(z) = max(2 cm, z SE_k), a comparison with SE_k = 0 keeping the 2 cm margin:

- Removal: a component stays only if its removal gives Delta_k > m_k(3) for some k; otherwise it is removable.
  When two or more components are removable, their combined removal is confirmed by the same test, a conditional
  second stage, and they leave together only when it passes; otherwise they stay and the record says so.
- Adoption: a candidate enters only if Delta_k < -m_k(3) for at least two comparisons and Delta_k <= m_k(2) for
  every comparison.
- Replacement: a candidate that replaces a component by a simpler or more unified one enters when Delta_k <= m_k(3)
  for every comparison. A replacement simplifies the method like a removal, so it is judged by the removal's
  threshold, and differences within the margin favour it. When two or more replacements enter
  one by one, they are confirmed together by the same test, a conditional second stage, and they enter together only
  when it passes. Otherwise none of them enters and the record says so.
- Convergence: a numerical setting is converged when the candidate that doubles it gives |Delta_k| <= m_k(3) for
  every comparison. It is the removal's threshold applied to a difference of either sign, since a numerical setting
  that is not converged may move a result either way.

Every comparison records Delta_k exact and released, SE_k, m_k(2) and m_k(3) and its contribution to the verdict;
Bern comparisons also carry the count of held scans where the candidate is worse, better or equal, a diagnostic that
does not vote. A round may compute comparisons that are recorded but do not vote, a deployment that is measured
without a say, a sample track or a scope that a candidate's rule leaves out. Their records carry votes false, their
contributions are computed like the others, and the verdict counts only the comparisons that vote. PARAMETERS holds
the declared values, and a round records the values it used.

Rule checks: label_swap builds an exact null from a real pair by exchanging the two positions of every unit with
probability one half, and null_rates measures how often the rules keep, adopt, let a replacement enter or find a
setting converged, without effect or with a planted one.
"""
import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

import common

PARAMETERS = dict(replicates=2000, seed=20260925, floor_cm=2.0, removal_z=3.0, gain_z=3.0, harm_z=2.0, replace_z=3.0,
                  minimum_gains=2, deployments=["bern", "tiers", "bregar"])
LEVELS = (("S75", "0.75"), ("S90", "0.9"))
DIRECTIONS = ("add", "remove", "replace", "converge")
BLOCK = 50  # draws per random stream, so draw b is the same whatever the total count
# The band around a level, relative to the draw's total weight, within which a floating point CDF is settled in exact
# arithmetic. The rounding of a sum of a million weights stays below a tenth of the band.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Design:
    """How one dataset's units are resampled. The names refer to fields of the unit rows.

    top: the fields naming the clusters, drawn with replacement within the strata named by top_strata (one stratum
    when empty). run_length: above one, the values of top within a stratum, in their order, are joined into runs of
    run_length consecutive values, and the runs are the clusters. A stratum of n values holds n // run_length runs,
    at least one, of as equal lengths as the values allow. cells_in_top: every cell lies inside one cluster, so a drawn
    copy of the cluster brings its own copies of the cells. Otherwise cells are shared by clusters and keep their
    identity. rescaled: a stratum of n > 1 clusters gives n - 1 draws, and every drawn copy counts n / (n - 1) times,
    the rescaling bootstrap of Rao and Wu (1988) for strata of few clusters. The weight matters where a cell spans
    strata or strata of other sizes hold copies of owned cells, and it cancels within a cell of one stratum.
    """
    top: tuple
    top_strata: tuple = ()
    cells_in_top: bool = False
    rescaled: bool = False
    run_length: int = 1

    def describe(self):
        # an option at its default is described as before the option existed
        defaults = dict(rescaled=False, run_length=1)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()
                if key not in defaults or value != defaults[key]}


DESIGNS = {
    "bern": Design(top=("scan",), cells_in_top=True, rescaled=True),
    # the blocks of a recording are consecutive windows of one static layout, so runs of three are the clusters
    "tiers": Design(top=("acquisition", "block"), top_strata=("acquisition",), rescaled=True, run_length=3),
    # the 24 environment and channel units together, conservative for the interval conditional on the four
    # environments, see the module docstring
    "bregar": Design(top=("acquisition",), cells_in_top=True, rescaled=True),
    # the panels of a configuration reuse the records of its short links, so the configurations are the clusters, and a
    # cell of IDLab, a configuration and anchor count, lies in one of them
    "idlab": Design(top=("configuration",), cells_in_top=True, rescaled=True),
    # the designs of the final tables that no round draws, see the module docstring
    "bern-evaluation": Design(top=("agents", "group"), top_strata=("agents",), rescaled=True),
    "hymn": Design(top=("window",), rescaled=True),
    "aau": Design(top=("flight", "block"), top_strata=("flight",), rescaled=True, run_length=2),
}


def plain(value):
    """Python scalars and tuples in place of numpy scalars and lists, so keys compare, hash and order the same."""
    if isinstance(value, (list, tuple)):
        return tuple(plain(v) for v in value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def ordered(values):
    """A fixed order of hashable keys of mixed types."""
    return sorted(values, key=repr)


def levels_of(levels):
    return [(name, common.exact_level(level)) for name, level in levels]


def released_scorer(scorer=None):
    return scorer if scorer is not None else common.load_scorer()


# ------------------------------------------------------------------------------------------- unit rows
def bern_rows(scores):
    """Unit rows of one Bern mode from the rounds' score dict {(scan, scenario, iteration, samples): (error, anchors, agents)}."""
    return [dict(key=(scan, scenario, iteration, samples), track=samples, cell=(scan, anchors, agents), error=error,
                 scan=scan, scenario=scenario, anchors=anchors, agents=agents)
            for (scan, scenario, iteration, samples), (error, anchors, agents) in scores.items()]


def read_bern_scores(path, scan, mode):
    """The released scorer's scenario scores of one held scan and mode, q = 2, in the rounds' score dict layout."""
    scores = {}
    with gzip.open(path, "rt") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != mode.upper() or int(row["q"]) != 2:
                continue
            key = (scan, int(row["scenario_id"]), int(row["iteration"]), int(row["samples"]))
            error = float(row["mean_pair_error"])
            if key in scores or not math.isfinite(error):
                raise ValueError(f"invalid or duplicate score in {path}")
            scores[key] = (error, int(row["anchors"]), int(row["agents"]))
    if not scores:
        raise ValueError(f"no {mode} scores in {path}")
    return scores


EXTERNAL_FIELDS = {
    "tiers": lambda row: dict(acquisition=row["acquisition"], block=row["block"]),
    "bregar": lambda row: dict(acquisition=row["acquisition"], environment=row["environment"]),
    "idlab": lambda row: dict(panel=row["panel"], configuration=row["configuration"]),
}


def external_rows(dataset, rows, method=None, pair_set=None):
    """Unit rows of an external training half from runner or baseline score rows, optionally of one method.

    Units pair on (key, id, samples) and fall in the cell row["cell"]. With pair_set the error is the unit's mean pair
    error over that pair set of research/benchmarks/datasets/scoring.py, which the runners keep per unit under
    row["pair_sets"]. A unit without any pair of the set has no error and is left out, as the released scorer leaves out
    a scenario without eligible pairs. The pair set depends on the inputs alone, so every candidate leaves out the same
    units. The anchor count of a unit is kept when the row has one.
    """
    result = []
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        if pair_set is None:
            error = row["error_cm"] if "error_cm" in row else row["mean_pair_error_cm"]
        else:
            error = row["pair_sets"][pair_set]["mean_pair_error_cm"]
            if error is None:
                continue
        cell = row["cell"] if "cell" in row else row["acquisition"]
        entry = dict(key=(row.get("key"), row["id"], row["samples"]), track=row["samples"], cell=cell, error=error,
                     **EXTERNAL_FIELDS[dataset](row))
        if "anchors" in row:
            entry["anchors"] = row["anchors"]
        result.append(entry)
    return result


# ------------------------------------------------------------------------------------------- paired units
class Paired:
    """The units of one deployment and mode, or of a declared scope of them, positioned by the reference and by the
    candidate."""

    def __init__(self, dataset, mode, reference, candidate, design, scope=None):
        self.dataset, self.mode, self.design, self.scope = dataset, mode, design, scope
        sides = []
        for rows in (reference, candidate):
            table = {}
            for row in rows:
                row = {name: plain(value) for name, value in row.items()}
                if row["key"] in table:
                    raise ValueError(f"duplicate unit {row['key']} in {dataset} {mode}")
                table[row["key"]] = row
            sides.append(table)
        if sides[0].keys() != sides[1].keys():
            raise ValueError(f"the reference and the candidate positioned different units in {dataset} {mode}")
        if not sides[0]:
            raise ValueError(f"no units in {dataset} {mode}")
        self.keys = ordered(sides[0])
        fields = ("track", "cell") + tuple(dict.fromkeys(design.top + design.top_strata))
        for key in self.keys:
            if any(sides[0][key][f] != sides[1][key][f] for f in fields):
                raise ValueError(f"unit {key} changed its cell or cluster between candidates in {dataset} {mode}")
        self.rows = [sides[0][key] for key in self.keys]
        self.reference = np.array([sides[0][key]["error"] for key in self.keys], dtype=float)
        self.candidate = np.array([sides[1][key]["error"] for key in self.keys], dtype=float)
        if not (np.isfinite(self.reference).all() and np.isfinite(self.candidate).all()):
            raise ValueError(f"non finite errors in {dataset} {mode}")
        self.track = [row["track"] for row in self.rows]
        self.cell = [row["cell"] for row in self.rows]
        self.top = [tuple(row[f] for f in design.top) for row in self.rows]
        self.top_stratum = [tuple(row[f] for f in design.top_strata) for row in self.rows]

    def cells_of(self, side, track, keep=lambda row: True):
        """The errors of one position and track grouped by cell, optionally of a subset of units."""
        cells = {}
        for row, value, t in zip(self.rows, getattr(self, side), self.track):
            if t == track and keep(row):
                cells.setdefault(row["cell"], []).append(float(value))
        return cells


class TrackPlan:
    """One sample track of a paired set: its units, cells and the sort orders of both positions."""

    def __init__(self, paired, track, top_index):
        self.track = track
        self.units = np.array([i for i, t in enumerate(paired.track) if t == track])
        cells = ordered({paired.cell[i] for i in self.units})
        code = {cell: c for c, cell in enumerate(cells)}
        self.cells = cells
        self.cell_code = np.array([code[paired.cell[i]] for i in self.units])
        self.sizes = np.bincount(self.cell_code, minlength=len(cells)).astype(np.int64)
        self.owned = paired.design.cells_in_top
        if self.owned:
            tops = {}
            for i, c in zip(self.units, self.cell_code):
                tops.setdefault(c, set()).add(top_index[paired.top[i]])
            if any(len(t) != 1 for t in tops.values()):
                raise ValueError(f"a cell of {paired.dataset} {paired.mode} spans several clusters")
            self.cells_per_top = np.zeros(max(top_index.values()) + 1, dtype=np.int64)
            for t in tops.values():
                self.cells_per_top[next(iter(t))] += 1
        self.values = {}
        for side in ("reference", "candidate"):
            errors = getattr(paired, side)[self.units]
            order = np.argsort(errors, kind="stable")
            self.values[side] = (order, errors[order])

    def weights(self, multiplicity, counts):
        """The weights of every draw as integers (m, d, K*), unit i weighing m_i / d_i of the balanced CDF, whose total
        is K*. The weight of a cluster in a draw is its copies times the rescaling weight of its stratum (_draws).

        With owned cells every drawn copy of a cluster brings its cells whole, m_i being the weight of unit i's
        cluster, d_i the size of its cell and K* the weight of the cell copies. With shared cells m_i is the weight of
        unit i, d_i = n*_c the drawn weight of its cell c and K* the cells with any.
        """
        m = multiplicity[:, self.units]
        if self.owned:
            return m, np.broadcast_to(self.sizes[self.cell_code], m.shape), counts @ self.cells_per_top
        drawn = np.zeros((m.shape[0], len(self.cells)), dtype=np.int64)
        for c in range(len(self.cells)):
            drawn[:, c] = m[:, self.cell_code == c].sum(axis=1)
        return m, drawn[:, self.cell_code], (drawn > 0).sum(axis=1)

    def percentiles(self, side, weights, levels):
        """The balanced percentiles of one position per draw: the first sorted unit where the CDF reaches the level.

        The CDF is summed in floating point. A draw where it comes within TOLERANCE of a level at some unit is settled
        in exact rational arithmetic from there, so every result is the one of the exact CDF, whatever the least common
        multiple of the draw's cell sizes."""
        m, d, total = weights
        if (total <= 0).any():
            raise ValueError("a bootstrap draw holds no unit of a comparison, so its percentile is undefined")
        order, values = self.values[side]
        m, d = m[:, order], d[:, order]
        with np.errstate(divide="ignore", invalid="ignore"):
            cumulative = np.cumsum(np.where(m > 0, m / np.maximum(d, 1), 0.0), axis=1)
        band = TOLERANCE * np.maximum(total, 1)[:, None]
        result = np.empty((m.shape[0], len(levels)))
        for k, (_, level) in enumerate(levels):
            target = float(level) * total
            index = np.count_nonzero(cumulative < target[:, None], axis=1)
            near = np.abs(cumulative - target[:, None]) <= band
            for b in np.flatnonzero(near.any(axis=1) | (index >= len(values))):
                start = int(np.argmax(near[b])) if near[b].any() else 0
                index[b] = exact_crossing(m[b], d[b], start, level * int(total[b]))
            result[:, k] = values[index]
        return result


def exact_crossing(m, d, start, target):
    """The first sorted position where the exact CDF of the weights m_i / d_i reaches target, a Fraction, searched from
    start. The floating point CDF lies below target by more than its rounding before start, so the exact CDF does
    too, and the weights before start are summed once, exactly, per denominator."""
    head = m[:start] > 0
    cumulative = Fraction(0)
    if head.any():
        denominators, inverse = np.unique(d[:start][head], return_inverse=True)
        numerators = np.zeros(len(denominators), dtype=object)
        np.add.at(numerators, inverse, m[:start][head].astype(object))
        cumulative = sum((Fraction(int(n), int(q)) for n, q in zip(numerators, denominators)), Fraction(0))
    for j in range(start, len(m)):
        if m[j] > 0:
            cumulative += Fraction(int(m[j]), int(d[j]))
            if cumulative >= target:
                return j
    raise ValueError("the exact balanced CDF of a draw never reaches its level")


def consecutive(key):
    """The order of the values of top along a recording, numbers by value and names by their text."""
    return tuple((0, value, "") if isinstance(value, (int, float)) and not isinstance(value, bool) else (1, 0, str(value)) for value in key)


def _dataset_plan(paired_sets, design=None):
    """The clusters and their strata over every mode of one dataset in a fixed order, the cluster of every value of
    top, {top: cluster}, and the clusters of every stratum, {stratum: [cluster]}. A cluster is a value of top, or with a
    run length above one a run of consecutive values of its stratum (Design)."""
    stratum_of = {}
    for p in paired_sets:
        for top, stratum in zip(p.top, p.top_stratum):
            if stratum_of.setdefault(top, stratum) != stratum:
                raise ValueError(f"cluster {top} of {p.dataset} lies in two strata")
    length = 1 if design is None else design.run_length
    if length < 1:
        raise ValueError(f"a run holds at least one value of top, not {length}")
    top_keys = ordered(stratum_of)
    if length == 1:
        top_index = {key: g for g, key in enumerate(top_keys)}
        strata = {}
        for key in top_keys:
            strata.setdefault(stratum_of[key], []).append(top_index[key])
        return top_index, strata
    members = {}
    for key in top_keys:
        members.setdefault(stratum_of[key], []).append(key)
    top_index, strata = {}, {}
    for stratum in ordered(members):
        keys = sorted(members[stratum], key=consecutive)
        runs = max(1, len(keys) // length)
        first = sum(len(clusters) for clusters in strata.values())
        strata[stratum] = list(range(first, first + runs))
        for place, key in enumerate(keys):
            top_index[key] = first + place * runs // len(keys)
    return top_index, strata


def rescaling(strata, rescaled):
    """The weight of one drawn copy of every cluster as an integer, n / (n - 1) for a stratum of n > 1 clusters of a
    rescaled design and one otherwise, times the least common multiple of the n - 1 and over the greatest common divisor
    of the results, so that strata of one size weigh one."""
    clusters = sum(len(members) for members in strata.values())
    factor = np.ones(clusters, dtype=np.int64)
    if not rescaled:
        return factor
    multiple = math.lcm(*(len(members) - 1 for members in strata.values() if len(members) > 1))
    for members in strata.values():
        n = len(members)
        factor[members] = multiple * n // (n - 1) if n > 1 else multiple
    return factor // (math.gcd(*factor.tolist()) or 1)


def _draws(dataset, plan, replicates, seed, design=None):
    """Per block of BLOCK draws, the weight of every cluster, its number of copies, n of the n clusters of a stratum,
    or n - 1 when the design is rescaled and n > 1, each copy then weighing n / (n - 1) in the integers of rescaling.
    The design is the dataset's declared one by default."""
    _, strata = plan
    design = design or DESIGNS.get(dataset)
    rescaled = bool(design is not None and design.rescaled)
    factor = rescaling(strata, rescaled)
    for block in range(math.ceil(replicates / BLOCK)):
        rng = np.random.default_rng([seed, block, *dataset.encode()])
        counts = np.zeros((BLOCK, len(factor)), dtype=np.int64)
        for stratum in ordered(strata):
            members = strata[stratum]
            size = len(members) - 1 if rescaled and len(members) > 1 else len(members)
            counts[:, members] = rng.multinomial(size, [1.0 / len(members)] * len(members), size=BLOCK)
        yield (counts * factor)[:min(BLOCK, replicates - block * BLOCK)]


def scan_signs(paired, track, levels):
    """Bern diagnostic per level: the held scans where the candidate's balanced percentile is above, below or equal to the reference's."""
    by_scan = {}
    for row, reference, candidate, t in zip(paired.rows, paired.reference, paired.candidate, paired.track):
        if t == track:
            cells = by_scan.setdefault(row["scan"], ({}, {}))
            cells[0].setdefault(row["cell"], []).append(float(reference))
            cells[1].setdefault(row["cell"], []).append(float(candidate))
    result = {}
    for name, level in levels:
        signs = dict(worse=0, better=0, equal=0)
        for scan in ordered(by_scan):
            reference, candidate = by_scan[scan]
            delta = common.balanced_percentile_exact(candidate, level) - common.balanced_percentile_exact(reference, level)
            signs["worse" if delta > 0 else "better" if delta < 0 else "equal"] += 1
        result[name] = signs
    return result


def pair_key(key):
    """The dataset, mode and scope of a pair key, (dataset, mode) or (dataset, mode, scope). The scope is None for all
    the units of the mode."""
    if len(key) not in (2, 3):
        raise ValueError(f"a pair key is (dataset, mode) or (dataset, mode, scope), not {key}")
    return key[0], key[1], key[2] if len(key) == 3 else None


def compare(pairs, designs=None, replicates=None, seed=None, levels=LEVELS, scorer=None):
    """Exact differences, their released values and bootstrap draws for every comparison of one candidate.

    pairs maps (dataset, mode) or (dataset, mode, scope) to (reference rows, candidate rows). Returns one entry per
    dataset, mode, scope, track and level with the exact and released balanced percentiles of both positions, Delta_k
    and its B draws. The scoped pairs of a dataset share its draws.
    """
    designs = designs or DESIGNS
    replicates = PARAMETERS["replicates"] if replicates is None else replicates
    seed = PARAMETERS["seed"] if seed is None else seed
    exact_levels = levels_of(levels)
    released = released_scorer(scorer)
    if replicates < 2:
        raise ValueError("the bootstrap needs at least two draws")
    by_dataset = {}
    for key, (reference, candidate) in sorted(pairs.items(), key=lambda item: tuple(str(part) for part in item[0])):
        dataset, mode, scope = pair_key(key)
        by_dataset.setdefault(dataset, []).append(Paired(dataset, mode, reference, candidate, designs[dataset], scope))
    comparisons = []
    for dataset, paired_sets in by_dataset.items():
        plan = _dataset_plan(paired_sets, designs[dataset])
        top_index = plan[0]
        tracks = [(p, TrackPlan(p, track, top_index)) for p in paired_sets for track in sorted(set(p.track))]
        index = {id(p): np.array([top_index[k] for k in p.top]) for p in paired_sets}
        draws = {id(plan_track): [] for _, plan_track in tracks}
        for counts in _draws(dataset, plan, replicates, seed, designs[dataset]):
            for p, plan_track in tracks:
                weights = plan_track.weights(counts[:, index[id(p)]], counts)
                draws[id(plan_track)].append(plan_track.percentiles("candidate", weights, exact_levels)
                                             - plan_track.percentiles("reference", weights, exact_levels))
        for p, plan_track in tracks:
            drawn = np.concatenate(draws[id(plan_track)])
            cells = {side: p.cells_of(side, plan_track.track) for side in ("reference", "candidate")}
            signs = scan_signs(p, plan_track.track, exact_levels) if dataset == "bern" else None
            for k, (name, level) in enumerate(exact_levels):
                exact = {side: float(common.balanced_percentile_exact(cells[side], level)) for side in cells}
                floating = {side: float(released.balanced_percentile(cells[side], float(level))) for side in cells}
                entry = dict(dataset=dataset, mode=p.mode, track=plan_track.track, level=name, units=int(len(plan_track.units)),
                             cells=len(plan_track.cells), reference_cm=exact["reference"], candidate_cm=exact["candidate"],
                             difference_cm=exact["candidate"] - exact["reference"], released_reference_cm=floating["reference"],
                             released_candidate_cm=floating["candidate"], released_difference_cm=floating["candidate"] - floating["reference"],
                             replicates=drawn[:, k])
                if p.scope is not None:
                    entry["scope"] = p.scope
                if signs is not None:
                    entry["scan_signs"] = signs[name]
                comparisons.append(entry)
    return comparisons


def margin(se, z, floor_cm):
    return max(floor_cm, z * se)


def votes_of(comparison):
    """Whether a comparison votes, true unless the round marked it as recorded only."""
    return comparison.get("votes", True)


def judge(comparisons, direction, parameters=None):
    """SE and margins per comparison, each comparison's contribution and the verdict of the direction. The verdict
    counts the comparisons that vote, and the others keep their contribution in the record."""
    parameters = dict(PARAMETERS, **(parameters or {}))
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if not any(votes_of(comparison) for comparison in comparisons):
        raise ValueError("a decision needs at least one comparison that votes")
    records = contributions(comparisons, direction, parameters)
    return records, verdict_of(records, direction, parameters)


def contributions(comparisons, direction, parameters):
    """The records of the comparisons with their SE, margins and contribution to a verdict of the direction. A
    contribution depends on its own comparison alone."""
    floor, records = parameters["floor_cm"], []
    for comparison in comparisons:
        se = float(np.std(comparison["replicates"], ddof=1))
        delta = comparison["difference_cm"]
        record = {k: v for k, v in comparison.items() if k != "replicates"}
        record.update(se_cm=se, margins_cm={f"{z:g}": margin(se, z, floor) for z in sorted({parameters["harm_z"], parameters["gain_z"], parameters["removal_z"],
                                                                                           parameters["replace_z"]})})
        if direction == "remove":
            record["contribution"] = "keeps" if delta > margin(se, parameters["removal_z"], floor) else "none"
        elif direction == "replace":
            record["contribution"] = "harm" if delta > margin(se, parameters["replace_z"], floor) else "none"
        elif direction == "converge":
            record["contribution"] = "differs" if abs(delta) > margin(se, parameters["removal_z"], floor) else "none"
        else:
            gain = delta < -margin(se, parameters["gain_z"], floor)
            harm = delta > margin(se, parameters["harm_z"], floor)
            record["contribution"] = "harm" if harm else "gain" if gain else "neutral"
        records.append(record)
    return records


def verdict_of(records, direction, parameters, votes=votes_of):
    """The verdict of the direction from the contributions of the records for which votes is true, the comparisons
    that vote by default."""
    labels = lambda value: [label(r) for r in records if r["contribution"] == value and votes(r)]  # noqa: E731
    if direction == "remove":
        keeps = labels("keeps")
        result = dict(direction=direction, removable=not keeps, keeping=keeps)
    elif direction == "replace":
        harms = labels("harm")
        result = dict(direction=direction, enters=not harms, harms=harms)
    elif direction == "converge":
        differing = labels("differs")
        result = dict(direction=direction, converged=not differing, differing=differing)
    else:
        gains, harms = labels("gain"), labels("harm")
        result = dict(direction=direction, adopted=len(gains) >= parameters["minimum_gains"] and not harms, gains=gains, harms=harms)
    return result


def verdict_without(record, dataset):
    """A diagnostic of a decision record that no verdict reads: the verdict recomputed from the record's contributions
    with the comparisons of one dataset voting no more, the number of its comparisons that voted, and whether that
    verdict differs from the record's. With no other comparison that votes there is no verdict, and it counts as
    differing."""
    direction = record["verdict"]["direction"]
    left_out = sum(1 for comparison in record["comparisons"] if comparison["dataset"] == dataset and votes_of(comparison))
    kept = lambda comparison: votes_of(comparison) and comparison["dataset"] != dataset  # noqa: E731
    if not any(kept(comparison) for comparison in record["comparisons"]):
        return dict(dataset=dataset, votes_left_out=left_out, verdict=None, differs=True)
    result = verdict_of(record["comparisons"], direction, dict(PARAMETERS, **record["parameters"]), votes=kept)
    key = VERDICT_KEYS[direction]
    return dict(dataset=dataset, votes_left_out=left_out, verdict=result, differs=result[key] != record["verdict"][key])


def label(record):
    scope = f"/{record['scope']}" if record.get("scope") is not None else ""
    return f"{record['dataset']}/{record['mode']}/{record['track']}/{record['level']}{scope}"


VERDICT_KEYS = dict(add="adopted", remove="removable", replace="enters", converge="converged")


def fixed_verdict(comparisons, direction, margin_cm=2.0):
    """The verdict of the earlier fixed 2 cm rule on the same differences, for disclosure beside the deciding one. It
    reads the comparisons that vote."""
    voting_comparisons = [c for c in comparisons if votes_of(c)]
    worse = [label(c) for c in voting_comparisons if c["difference_cm"] > margin_cm]
    better = [label(c) for c in voting_comparisons if c["difference_cm"] < -margin_cm]
    result = dict(direction=direction, margin_cm=margin_cm, worse_beyond_margin=worse, better_beyond_margin=better)
    if direction == "add":
        result[VERDICT_KEYS[direction]] = not worse and bool(better)
    elif direction == "converge":
        result[VERDICT_KEYS[direction]] = not worse and not better
    else:
        result[VERDICT_KEYS[direction]] = not worse
    return result


def voting(pairs, parameters):
    """The pairs of the deployments that vote, and the names of the ones left out."""
    kept = {key: value for key, value in pairs.items() if key[0] in parameters["deployments"]}
    return kept, sorted({key[0] for key in pairs} - {key[0] for key in kept})


def decide(pairs, direction, parameters=None, designs=None, levels=LEVELS, scorer=None, votes=None):
    """The full decision of one candidate as a JSON record: comparisons, SE, margins, contributions and verdict.

    Without votes, the pairs of the deployments outside parameters["deployments"] are left out before any comparison,
    as the earlier rounds declared. With votes, a function of a comparison entry, every pair is compared and recorded,
    and a comparison votes only when votes(entry) is true."""
    parameters = dict(PARAMETERS, **(parameters or {}))
    designs = designs or DESIGNS
    if votes is None:
        kept, excluded = voting(pairs, parameters)
    else:
        kept, excluded = pairs, []
    comparisons = compare(kept, designs, parameters["replicates"], parameters["seed"], levels, scorer)
    if votes is not None:
        for comparison in comparisons:
            comparison["votes"] = bool(votes(comparison))
    records, result = judge(comparisons, direction, parameters)
    used = sorted({key[0] for key in kept})
    record = dict(parameters=parameters, levels=[[name, str(level)] for name, level in levels], excluded_deployments=excluded,
                  designs={dataset: designs[dataset].describe() for dataset in used}, numpy=np.__version__, comparisons=records,
                  verdict=result, fixed_margin_verdict=fixed_verdict(comparisons, direction, parameters["floor_cm"]))
    if votes is not None:
        record["recorded_only_deployments"] = sorted({c["dataset"] for c in comparisons} - {c["dataset"] for c in comparisons if c["votes"]})
    return record


def final_stage(stage_one, final=None):
    """The conditional second stage over the removal records of every component and, when run, of the combined candidate."""
    removable = sorted(name for name, record in stage_one.items() if record["verdict"]["removable"])
    if len(removable) < 2:
        return dict(required=False, removable=removable, removed=removable, kept=[])
    if final is None:
        return dict(required=True, removable=removable, removed=None, kept=None, pending="the combined candidate has not run")
    if final["verdict"]["direction"] != "remove":
        raise ValueError("the combined candidate is judged by the removal verdict")
    passed = final["verdict"]["removable"]
    return dict(required=True, removable=removable, final_passed=passed, removed=removable if passed else [], kept=[] if passed else removable)


def replacement_stage(stage_one, joint=None):
    """The conditional second stage over the replacement records of every replacement and, when run, of the joint one.

    A replacement that entered alone enters the method when it is the only one. Two or more that entered alone are
    confirmed together by the replacement verdict of the joint candidate. They enter together only when it passes,
    otherwise none of them enters. A replacement without a record, one not run, does not enter."""
    entered = sorted(name for name, record in stage_one.items() if record is not None and record["verdict"]["enters"])
    if len(entered) < 2:
        return dict(required=False, entered=entered, adopted=entered, left_out=[])
    if joint is None:
        return dict(required=True, entered=entered, adopted=None, left_out=None, pending="the joint candidate has not run")
    if joint["verdict"]["direction"] != "replace":
        raise ValueError("the joint candidate is judged by the replacement verdict")
    passed = joint["verdict"]["enters"]
    return dict(required=True, entered=entered, joint_passed=passed, adopted=entered if passed else [], left_out=[] if passed else entered)


# ------------------------------------------------------------------------------------------- rule checks
def label_swap(reference, candidate, rng):
    """An exact null from a real pair: every unit's two positions trade places with probability one half.

    A unit is its key without the sample track, so both tracks of a unit trade together; the clusters stay intact.
    """
    units = ordered({plain(row["key"])[:-1] for row in reference})
    swap = dict(zip(units, rng.random(len(units)) < 0.5))
    by_key = {plain(row["key"]): row for row in candidate}
    first, second = [], []
    for row in reference:
        key = plain(row["key"])
        other, exchanged = by_key[key], swap[key[:-1]]
        first.append(dict(row, error=other["error"] if exchanged else row["error"]))
        second.append(dict(row, error=row["error"] if exchanged else other["error"]))
    return first, second


def null_rates(simulate, simulations, parameters=None, designs=None, levels=LEVELS, seed=0, scorer=None):
    """How often the rules keep (removal), adopt (adding), let a replacement enter or find a setting converged, over
    data sets drawn by simulate(rng).

    Under a null the keep and adoption rates are the false keep and false adoption rates, the entry rate is how often
    a harmless replacement enters, and the convergence rate is how often a setting whose doubling changes nothing is
    found converged. With a planted gain the adoption rate is the power, with a planted harm the entry rate is how often
    a harmful replacement enters, and with a planted effect of either sign the convergence rate is how often a setting
    that is not converged passes. The rule of PARAMETERS and the earlier fixed 2 cm rule judge the same differences.
    """
    parameters = dict(PARAMETERS, **(parameters or {}))
    rng = np.random.default_rng(seed)
    counts = dict(keep=0, adopt=0, enter=0, converge=0, fixed_keep=0, fixed_adopt=0, fixed_enter=0, fixed_converge=0)
    for simulation in range(simulations):
        kept, _ = voting(simulate(rng), parameters)
        comparisons = compare(kept, designs, parameters["replicates"], parameters["seed"] + simulation, levels, scorer)
        counts["keep"] += not judge(comparisons, "remove", parameters)[1]["removable"]
        counts["adopt"] += judge(comparisons, "add", parameters)[1]["adopted"]
        counts["enter"] += judge(comparisons, "replace", parameters)[1]["enters"]
        counts["converge"] += judge(comparisons, "converge", parameters)[1]["converged"]
        counts["fixed_keep"] += not fixed_verdict(comparisons, "remove", parameters["floor_cm"])["removable"]
        counts["fixed_adopt"] += fixed_verdict(comparisons, "add", parameters["floor_cm"])["adopted"]
        counts["fixed_enter"] += fixed_verdict(comparisons, "replace", parameters["floor_cm"])["enters"]
        counts["fixed_converge"] += fixed_verdict(comparisons, "converge", parameters["floor_cm"])["converged"]
    return dict(simulations=simulations, **{name: value / simulations for name, value in counts.items()})


# ------------------------------------------------------------------------------------------- round outputs
def round_pairs(work, reference, candidate, datasets, modes, scans):
    """Pairs of two candidates of a union round from its per unit score files, with the hashes of the files read.

    Layout of the rounds: Bern cooperative work/bern/cooperative/scan-XX/<mode>/<name>/scenario_scores.csv.gz, Bern
    anchor only work/bern/anc/scan-XX/<name>/scenario_scores.csv.gz, external work/<dataset>/<name>/<mode>/scores.json.
    modes maps a dataset to the modes to read.
    """
    work = Path(work)
    pairs, files = {}, {}

    def bern(name, mode):
        scores = {}
        for scan in scans:
            path = (work / "bern/anc" / f"scan-{scan:02d}" / name if mode == "anc"
                    else work / "bern/cooperative" / f"scan-{scan:02d}" / mode / name) / "scenario_scores.csv.gz"
            files[str(path.relative_to(work))] = common.sha256(path)
            scores.update(read_bern_scores(path, scan, mode))
        return bern_rows(scores)

    def external(dataset, name, mode):
        path = work / dataset / name / mode / "scores.json"
        files[str(path.relative_to(work))] = common.sha256(path)
        return external_rows(dataset, json.loads(path.read_text()))

    for dataset in datasets:
        for mode in modes[dataset]:
            load = (lambda name: bern(name, mode)) if dataset == "bern" else (lambda name: external(dataset, name, mode))
            pairs[(dataset, mode)] = (load(reference), load(candidate))
    return pairs, files
