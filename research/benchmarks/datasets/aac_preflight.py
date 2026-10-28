"""Check derived TIERS AAC inputs without fitting or running positioning methods."""

from collections import Counter
import copy
import html
from itertools import combinations
import math
from pathlib import Path

import calibration
import pilot

OUTPUT = pilot.audit.DATA / "aac-preflight.json"
PAPER = "https://arxiv.org/html/2305.08532v1#S4.SS1"
ARENA_DIAGONAL_CM = math.hypot(800, 900)
METHODS = ("solid", "jin-2021", "ade-bp-2015", "fnl", "strong")


def population(windows):
    rows = [dict(anchors=k, agents=8-k, subsets=math.comb(8, k),
                 units_per_sample_track=windows * math.comb(8, k)) for k in range(1, 7)]
    return dict(by_anchor_count=rows, windows=windows,
                units_per_sample_track=sum(r["units_per_sample_track"] for r in rows),
                sample_tracks=[1, 10], methods=list(METHODS))


def operating_area(anchors):
    """Intersect anchor-centred bounds from the published arena diameter, not test poses."""
    result = {}
    for axis in ("x", "y"):
        values = [a["position"][axis] for a in anchors]
        low, high = max(values) - ARENA_DIAGONAL_CM, min(values) + ARENA_DIAGONAL_CM
        if low >= high:
            raise ValueError("anchor span exceeds the published arena bound")
        result[f"minimum_{axis}_cm"], result[f"maximum_{axis}_cm"] = low, high
    return result


def construct(unit, anchor_ids, anchor_coordinates):
    """Only designated anchor coordinates are accepted. Ranges remain uncorrected."""
    original = unit["problem"]
    nodes = original["agent_ids"]
    selected = set(anchor_ids)
    if (nodes != list(range(1, 9)) or original["mode"] != "AnchorFree" or original["anchors"]
            or unit["samples"] not in (1, 10) or not 1 <= len(selected) <= 6
            or len(selected) != len(anchor_ids) or not selected <= set(nodes)):
        raise ValueError("expected eight-node TIERS AF input and one to six distinct anchors")
    if set(anchor_coordinates) != selected:
        raise ValueError("provide only designated anchor coordinates")
    if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in anchor_coordinates.values()):
        raise ValueError("invalid anchor coordinates")
    if len(original["initial_positions"]) != len(nodes):
        raise ValueError("initial agent identities differ")
    raw = calibration.raw_request(dict(units=[unit], baselines=[]))["units"][0]
    observations, sources, seen = [], [], set()
    for observation in raw["problem"]["observations"]:
        a, b = observation["from"], observation["to"]
        pair = (a, b)
        if (set(observation) != {"from", "to", "to_is_anchor", "ranges_cm"}
                or observation["to_is_anchor"] or a >= b or pair in seen
                or len(observation["ranges_cm"]) != unit["samples"]
                or any(not math.isfinite(v) or v <= 0 for v in observation["ranges_cm"])):
            raise ValueError("invalid forward range stream")
        seen.add(pair)
        if a in selected and b in selected:
            continue
        if a in selected:
            observation["from"], observation["to"] = b, a
        observation["to_is_anchor"] = observation["to"] in selected
        observations.append(observation)
        sources.append(f"{a}-{b}")
    if seen != set(combinations(nodes, 2)):
        raise ValueError("expected all 28 original links")
    anchors = [dict(id=n, position=dict(x=anchor_coordinates[n][0], y=anchor_coordinates[n][1]),
                    height_cm=anchor_coordinates[n][2]) for n in sorted(selected)]
    problem = dict(agent_ids=[n for n in nodes if n not in selected], anchors=anchors,
                   observations=observations, mode="AnchorAided",
                   initial_positions=[copy.deepcopy(p) for n, p in zip(nodes, original["initial_positions"])
                                      if n not in selected])
    return problem, sources


def scoring_reference(problem, coordinates):
    # Anchors must not enlarge the unknown-agent diameter or the scored pair set.
    return {str(n): coordinates[str(n)] for n in problem["agent_ids"]}


def prepare():
    if OUTPUT.exists():
        raise ValueError("preflight already retained, do not overwrite it")
    original_manifest = pilot.check_inputs()
    pilot.check_inputs(calibration.OUTPUT)
    audit_path = pilot.audit.DATA / "input-audit.json"
    # The input audit of the research record is pinned in the retained Bern evidence, a reader's own audit is not.
    retained = {} if pilot.READER else pilot.evidence.read(pilot.evidence.retained_data.EVIDENCE)["report_sources"]
    if not pilot.READER and pilot.identity(audit_path) != retained[str(audit_path.relative_to(pilot.ROOT))]:
        raise ValueError("current input audit changed")
    checked = pilot.evidence.read(audit_path)
    for recording in checked["tiers"]:
        if recording["sha256"] != original_manifest["sources"][str(pilot.audit.DATA / recording["file"])]:
            raise ValueError("audited TIERS recording differs from retained pilot")
    request = pilot.evidence.read(pilot.OUTPUT / "request.json")
    reference = pilot.evidence.read(pilot.OUTPUT / "reference.json")
    metadata = {row["id"]: row for row in reference["units"]}
    if len(request["units"]) != 60 or len(metadata) != 30:
        raise ValueError("expected the retained thirty-window TIERS population")
    counts = Counter(row["acquisition"] for row in metadata.values())
    if counts != dict(army=10, corner=10, diagonal=10):
        raise ValueError("unexpected acquisition population")
    poses = {Path(row["file"]).name.split("_")[0]: row["pose_median_native"] for row in checked["tiers"]}
    demos = []
    for unit in request["units"]:
        meta = metadata[unit["id"]]
        if meta["window"] != 0:
            continue
        for k in (1, 4, 6):
            anchor_ids = list(range(1, k+1))
            known = {n: [100*v for v in poses[meta["acquisition"]][str(n)]] for n in anchor_ids}
            problem, links = construct(unit, anchor_ids, known)
            events = {pair: copy.deepcopy(meta["messages"][pair]) for pair in links}
            for observation, pair in zip(problem["observations"], links):
                reports = events[pair]["reports"][:unit["samples"]]
                if observation["ranges_cm"] != [100*r[1] for r in reports]:
                    raise ValueError("range values no longer match original message identities")
                events[pair] = dict(indices=events[pair]["indices"][:unit["samples"]], reports=reports)
            demos.append(dict(source_unit=unit["id"], acquisition=meta["acquisition"], samples=unit["samples"],
                anchor_count=k, problem=problem, source_links=links, original_messages=events,
                proposed_operating_area=operating_area(problem["anchors"])))
    if len(demos) != 18:
        raise ValueError("incomplete demonstration population")
    paths = [audit_path, pilot.OUTPUT / "inputs.json", pilot.OUTPUT / "request.json",
             pilot.OUTPUT / "reference.json", calibration.OUTPUT / "calibration.json",
             calibration.OUTPUT / "deployment-profile.json", Path(__file__),
             Path(__file__).with_name("README.md"),
             pilot.ROOT / "runtime/engine/src/cooperative/aac.rs",
             pilot.ROOT / "runtime/engine/src/calibration.rs",
             pilot.ROOT / "research/methods/src/bayesian.rs"]
    result = dict(schema="solid.external.aac-preflight.v1", status="review-required",
        positioning_runs=0, models_fitted=0, evaluation_outcomes_used=0,
        proposed_population=population(len(metadata)), published_arena_cm=[800, 900], arena_source=PAPER,
        checks=dict(demonstration_units=len(demos), sample_identity="passed", anchor_roles="passed",
                    unknown_truth_exported=False, executable_predictor_request=False),
        inputs={pilot.audit.name(p): pilot.identity(p) for p in pilot.present(paths)}, demonstrations=demos)
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT])
    print("Retained 18 input-only AAC demonstrations. No fitting or positioning executed.")


def section(result):
    count = result["proposed_population"]["units_per_sample_track"]
    demos = result["checks"]["demonstration_units"]
    url = html.escape(OUTPUT.as_uri(), quote=True)
    return f'''<details id="external-aac-preflight" class="report-document">
<summary>External AAC proposal and input checks · review required</summary>
<p><strong>No external AAC positioning results yet.</strong> {demos} input-only demonstrations retain
the original one-sample and ten-sample messages. Known anchor coordinates are separated from unknown agents.
No model was fitted and no positioning method was executed.</p>
<p>The proposal uses every anchor subset of size one through six in all thirty retained TIERS windows.
This gives {count:,} units per sample track and method, drawn from only three physical configurations.
Review the <a href="#external-protocol-proposed-tiers-aac-comparison">protocol and remaining checks</a>
before execution.</p>
<p>AAC needs its own Case 1 error-profile fit. The AF constant-scale profile is not interchangeable.
ADE-BP needs a Case 1 likelihood and an explicit area prior. The proposed area uses the published arena
size and designated anchors only, not unknown-agent truth.</p>
<p><a href="{url}">Exact demonstration inputs and source hashes</a></p></details>'''


if __name__ == "__main__":
    prepare()
