"""The inputs every method receives on the same-layout halves of TIERS, IDLab and Bregar beyond the campaigns' own.

The campaigns build the observations and the anchors of every unit, and they stay as built. Two inputs follow SOLID's
deployment rules instead (research/solid/deployment.py unit_area), so that every method receives what SOLID receives.
Every anchor aided and anchor only unit takes as its operating area the area those rules give it, the box of its known
anchors padded by its anchor ranges and bounded by the published site extent where the dataset publishes one. ADE-BP,
the one baseline that reads an area, receives that area wherever it runs, on TIERS, IDLab and Bregar. Every method starts from
the same coordinates, drawn uniformly inside the area both tracks of the unit share, the intersection of the area of
its one-reading track and that of its ten-reading track, by the rule of the held-out builders
(heldout.initial_positions), from a generator seeded by the dataset's declared seed, the mode and the unit's identity,
one draw for both tracks. The one-reading track's area can be the smaller one, so a start drawn in the ten-reading
track's area alone could lie outside the area its one-reading solve is given. An anchor free unit has no known anchor
and so no area, and it keeps the starts its campaign draws, which every method shares already. TWR-CLOUD-Bern keeps its
released inputs, and SOLID and ADE-BP share its one area there.

The baseline runners and SOLID's runners of the three datasets pass every request through levelled once, right after
the campaign builds it, and every run record keeps record(dataset), the rule and the seed.
"""
import copy
import sys
from collections import defaultdict
from pathlib import Path

import heldout

HERE = Path(__file__).resolve().parent
SOLID = HERE.parents[1] / "solid"
# the declared seeds of the starting coordinates, one per dataset
SEEDS = dict(tiers="solid.layout.tiers.starts.v1", idlab="solid.layout.idlab.starts.v1", bregar="solid.layout.bregar.starts.v1")
# the datasets whose units carry the operating area, the ones where ADE-BP runs
AREAS = ("tiers", "idlab", "bregar")
RULE = ("Every anchor aided and anchor only unit takes as its operating area the area of SOLID's deployment rules "
        "(research/solid/deployment.py unit_area), the box of its known anchors padded by its anchor ranges and bounded by the "
        "published site extent where the dataset publishes one, and ADE-BP receives that area wherever it runs. Every method "
        "starts from the same coordinates, drawn uniformly inside the area both tracks of the unit share, the intersection of the "
        "areas of its one-reading and its ten-reading track, from a generator seeded by the dataset's declared seed, the mode and "
        "the unit's identity (heldout.initial_positions), one draw for both tracks. Anchor free units have no area and keep the "
        "starts of their campaign, which every method shares. Observations and anchors stay as the campaign builds them.")


def area_rule(dataset):
    """SOLID's operating area of a prepared unit of the dataset, by the deployment rules."""
    if str(SOLID) not in sys.path:
        sys.path.insert(0, str(SOLID))
    import deployment
    return lambda problem: deployment.unit_area(problem, dataset)


def record(dataset):
    """What a run record keeps of the rule, its text and the dataset's seed."""
    return dict(rule=RULE, seed=SEEDS[dataset], areas=dataset in AREAS)


def shared_area(areas):
    """The area every one of the areas holds, their intersection, refused when they do not overlap. It is the one rule of
    heldout.shared_area, which the held-out builders draw their starts in too."""
    return heldout.shared_area(areas)


def levelled(request, dataset, mode):
    """A request with the starting coordinates of every anchor aided and anchor only unit drawn inside the area both
    tracks share, the intersection of SOLID's operating areas of its tracks, the same for both tracks, and in TIERS,
    IDLab and Bregar with every such unit's area set to SOLID's area of the unit's own track (RULE). Anchor free requests stay as
    they are. The request given is left unchanged."""
    if mode == "AF":
        return request
    rule = area_rule(dataset)
    result = dict(request, units=copy.deepcopy(request["units"]))
    tracks = defaultdict(dict)
    for unit in result["units"]:
        tracks[unit["id"]][unit["samples"]] = unit
    for identity, units in tracks.items():
        agents = units[max(units)]["problem"]["agent_ids"]
        if any(unit["problem"]["agent_ids"] != agents for unit in units.values()):
            raise ValueError(f"the tracks of unit {identity} hold other agents")
        areas = {samples: rule(unit["problem"]) for samples, unit in sorted(units.items())}
        start = heldout.initial_positions(shared_area(list(areas.values())), len(agents), SEEDS[dataset], f"{mode}/{identity}")
        for samples, unit in units.items():
            unit["problem"]["initial_positions"] = [dict(x=float(x), y=float(y)) for x, y in start]
            if dataset in AREAS:
                unit["area"] = areas[samples]
    return result
