"""Select shared IDLab record and anchor assignments without inspecting accuracy."""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import idlab_admission as admission
import idlab_split
import pilot


OUTPUT = admission.OUTPUT.with_name("idlab-population.json")
INPUTS = OUTPUT.with_name("idlab-population-inputs.json.gz")
SEED = "solid.idlab.population.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def construct(split):
    nodes = sorted(split["nodes"])
    if len(nodes) < 3 or len(set(nodes)) != len(nodes):
        raise ValueError("pairwise anchored scoring needs at least three distinct nodes")
    pools, seen = defaultdict(lambda: defaultdict(list)), set()
    for group in split["groups"]:
        node, peer = group["node"], group["peer"]
        if node not in nodes or peer not in nodes or node == peer:
            raise ValueError("invalid physical link")
        settings = group["settings"]
        if settings["txpower_tag"] != settings["txpower_anchor"]:
            raise ValueError("opposite power roles cannot be pooled without an explicit rule")
        key = tuple(sorted(settings.items()))
        pool = pools[key][tuple(sorted((node, peer)))]
        part, training = group["evaluation"], group["calibration"]
        start, stop = part["row_start"], part["row_stop"]
        invalid = set(part["invalid_publisher_rows"])
        if (start < 0 or stop < start or training["row_stop"] != start
                or stop - start != part["recorded_ranges"]
                or len(invalid) != len(part["invalid_publisher_rows"])
                or any(row < start or row >= stop for row in invalid)
                or stop - start - len(invalid) != part["valid_ranges"]):
            raise ValueError("invalid frozen evaluation interval")
        for row in range(start, stop):
            identity = (node, row)
            if identity in seen:
                raise ValueError("source record appears in multiple link settings")
            seen.add(identity)
            if row not in invalid:
                pool.append(list(identity))
    configurations = []
    for key, links in sorted(pools.items()):
        settings = dict(key)
        selected = [dict(nodes=list(pair), records=sorted(rows,
                    key=lambda row: digest([SEED, "records", settings, pair, row])))
                    for pair, rows in sorted(links.items())]
        configurations.append(dict(settings=settings, links=selected,
            panels=max(1, max((len(link["records"]) + 9) // 10 for link in selected))))
    valid = sum(len(link["records"]) for c in configurations for link in c["links"])
    if (not configurations or len(seen) != split["totals"]["evaluation"]["recorded_ranges"]
            or valid != split["totals"]["evaluation"]["valid_ranges"]):
        raise ValueError("population does not cover the frozen evaluation partition")
    return dict(nodes=nodes, configurations=configurations)


def panel(population, configuration, index):
    if not 0 <= index < configuration["panels"]:
        raise ValueError("panel index is outside the selected population")
    settings, links = configuration["settings"], []
    for link in configuration["links"]:
        rows = link["records"]
        if rows:
            block = index % ((len(rows) + 9) // 10)
            # Reuse real records across panels, never duplicate a record within a link input.
            start = min(10 * block, max(0, len(rows) - 10))
            ten = rows[start:start + 10]
            one = [min(ten, key=lambda row: digest([SEED, "one", settings, index, link["nodes"], row]))]
        else:
            one, ten = [], []
        links.append(dict(nodes=link["nodes"], one=one, up_to_ten=ten))
    order = sorted(population["nodes"], key=lambda node: digest([SEED, "anchors", settings, index, node]))
    return dict(nodes=population["nodes"], settings=settings, index=index, links=links, anchor_order=order)


def anchor_subsets(order):
    """Keep at least two unknown agents and balance every node's anchor frequency."""
    return [sorted((order + order)[start:start + count])
            for count in range(1, len(order) - 1) for start in range(len(order))]


def summarize(population):
    uses, sample_counts, configurations = Counter(), Counter(), []
    for configuration in population["configurations"]:
        hashes = []
        for index in range(configuration["panels"]):
            selected = panel(population, configuration, index)
            hashes.append(digest(selected))
            for link in selected["links"]:
                sample_counts[len(link["up_to_ten"])] += 1
                uses.update(map(tuple, link["up_to_ten"]))
        configurations.append(dict(settings=configuration["settings"], panels=configuration["panels"],
            physical_links=len(configuration["links"]), panel_sha256=hashes))
    expected = {tuple(row) for c in population["configurations"] for link in c["links"] for row in link["records"]}
    if set(uses) != expected:
        raise ValueError("selected panels omit valid held records")
    panels = sum(c["panels"] for c in configurations)
    n = len(population["nodes"])
    roles = len(anchor_subsets(population["nodes"]))
    per_sample = dict(AF=panels, AAC=panels * roles, ANC=panels * roles)
    return dict(nodes=n, radio_settings=len(configurations), panels=panels,
        physical_link_settings=sum(len(c["links"]) for c in population["configurations"]),
        valid_evaluation_records=len(expected), omitted_valid_records=0,
        record_uses_before_role_assignment=sum(uses.values()),
        reused_record_uses_before_role_assignment=sum(uses.values()) - len(expected),
        full_network_link_inputs_by_actual_sample_count=dict(sorted(sample_counts.items())),
        anchor_counts=list(range(1, n - 1)), assignments_per_anchor_count=n,
        assignments_per_anchored_panel=roles, inputs_per_sample_track=per_sample,
        total_mode_inputs=2 * sum(per_sample.values()), configurations=configurations)


def prepare():
    if OUTPUT.exists() or INPUTS.exists():
        raise ValueError("IDLab population already retained")
    if pilot.audit.uncommitted():
        raise ValueError("commit the population source and protocol before preparation")
    checked = pilot.evidence.read(admission.OUTPUT)
    split = pilot.evidence.read(idlab_split.OUTPUT)
    pilot.evidence.official_scorer.require_hash(admission.OUTPUT, split["admission_sha256"], "IDLab population source")
    for evidence in (checked, split):
        for name, expected in evidence["sources"].items():
            pilot.evidence.official_scorer.require_hash(pilot.audit.located(name), expected, "IDLab population source")
    population = construct(split)
    counts = summarize(population)
    with gzip.GzipFile(filename=str(INPUTS), mode="wb", mtime=0) as stream:
        stream.write(json.dumps(population, separators=(",", ":"), allow_nan=False).encode())
    result = dict(schema="solid.external.idlab-population.v1", status="population-selected-execution-not-frozen",
        source_revision=pilot.audit.revision(),
        split_sha256=pilot.identity(idlab_split.OUTPUT), seed=SEED,
        sources={pilot.audit.name(p): pilot.identity(p) for p in (Path(__file__).resolve(), INPUTS)},
        sample_tracks=["one", "up_to_ten"],
        sample_rule="Pool opposite directions at equal radio power. Hash-order source identities and take ten-record blocks. Use the last ten for a short remainder, or all records when fewer than ten exist. Cycle short links across panels. The one-record track selects a member of its paired block by identity hash.",
        reuse_rule="A source record occurs at most once per link input but can recur across panels and anchor assignments. These are repeated measurements of one fixed layout, not independent acquisitions.",
        anchor_rule="Hash-order all node IDs independently for each setting and panel. For each anchor count from one to n minus two, take all n cyclic windows. Never use coordinates, range values, connectivity or accuracy to select roles.",
        weights="Within each mode and sample track, weight radio settings equally. Within a setting, weight panels equally. Within an anchored panel, weight anchor counts and their assignments equally. Score every unknown-agent pair, including cross-component pairs.",
        missingness_rule="Short or exhausted record pools never remove an observed link. Invalid inputs, role exclusions and unverified attempts are not negative distance evidence. Availability fitting and its opportunity definition remain pending.",
        fairness="All applicable methods receive the same selected records and roles. Retain published training, aggregation, initialization and supported spatial processing. Unsupported modes remain N/A. Do not impose a solve wall-time cutoff or select settings from evaluation accuracy.",
        limitations="Anchor assignments sample possible subsets. Fourteen or fifteen anchors leave fewer than two unknown agents and no unknown-agent pairwise score. Radio settings on one layout do not establish unseen-site generalization.",
        pending=["Resolve acquisition opportunities and fit required availability parameters using training only.",
                 "Complete native method preparation and bounded sparse-network adapter checks.",
                 "Freeze inference inputs, profiles, numerical settings, scorer and measured execution budget."],
        models_fitted=0, positioning_solves=0, evaluation_rows_used_for_training=0, **counts)
    pilot.evidence.write(OUTPUT, result)
    pilot.register_files([OUTPUT, INPUTS])
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "configurations")}, indent=2))


if __name__ == "__main__":
    prepare()
