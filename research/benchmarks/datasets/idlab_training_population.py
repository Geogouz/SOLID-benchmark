"""The IDLab training population: the frozen campaign's construction applied to the reserved calibration half.

The split reserves the first half of every link and radio setting's rows for calibration and the rest for
evaluation. The frozen campaign builds its population from the evaluation rows; this script builds the same kind of
population, same configurations, panels, anchor subsets, sample tracks and cells, from the calibration rows, so that
design decisions can be measured on IDLab without touching an evaluation row. Profiles, radio settings, the fitted
correction and the coordinates are those of the frozen campaign. Output: external-benchmarks/idlab-training-campaign
with base.json.gz, catalog.json.gz and freeze.json.

Usage: idlab_training_population.py
"""
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
import idlab_admission as admission  # noqa: E402
import idlab_campaign as campaign  # noqa: E402
import idlab_population as population  # noqa: E402

CAMPAIGN = admission.OUTPUT.parent / "idlab-campaign"
OUTPUT = admission.OUTPUT.parent / "idlab-training-campaign"


def construct_training(split):
    """population.construct on the calibration intervals, with the same validity rules."""
    nodes = sorted(split["nodes"])
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
        part, evaluation = group["calibration"], group["evaluation"]
        start, stop = part["row_start"], part["row_stop"]
        invalid = set(part["invalid_publisher_rows"])
        if (start < 0 or stop < start or evaluation["row_start"] != stop
                or stop - start != part["recorded_ranges"]
                or len(invalid) != len(part["invalid_publisher_rows"])
                or any(row < start or row >= stop for row in invalid)
                or stop - start - len(invalid) != part["valid_ranges"]):
            raise ValueError("invalid frozen calibration interval")
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
        selected = [dict(nodes=list(pair), records=sorted(rows, key=lambda row: population.digest([population.SEED, "records", settings, pair, row])))
                    for pair, rows in sorted(links.items())]
        configurations.append(dict(settings=settings, links=selected,
                                   panels=max(1, max((len(link["records"]) + 9) // 10 for link in selected))))
    valid = sum(len(link["records"]) for c in configurations for link in c["links"])
    totals = split["totals"]["calibration"]
    if not configurations or len(seen) != totals["recorded_ranges"] or valid != totals["valid_ranges"]:
        raise ValueError("population does not cover the frozen calibration partition")
    return dict(nodes=nodes, configurations=configurations)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    split = json.loads(Path(admission.OUTPUT.with_name("idlab-split.json")).read_text())
    selected = construct_training(split)
    frozen = json.loads(gzip.decompress((CAMPAIGN / "base.json.gz").read_bytes()))
    panels = [dict(population.panel(selected, configuration, i), configuration=c)
              for c, configuration in enumerate(selected["configurations"]) for i in range(configuration["panels"])]
    base = dict(panels=panels, settings=frozen["settings"], fitted=frozen["fitted"], availability=frozen["availability"],
                coordinates_cm=frozen["coordinates_cm"], values=campaign.load_values(selected, admission.FOLDER))
    batches = list(campaign.catalog(base))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("base.json.gz", base), ("catalog.json.gz", batches)):
        (OUTPUT / name).write_bytes(gzip.compress(json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    counts = population.summarize(selected)
    freeze = dict(schema="solid.external.idlab-training-campaign.v1", partition="calibration rows of the frozen split",
                  split_sha256=sha256(admission.OUTPUT.with_name("idlab-split.json")), campaign_base_sha256=sha256(CAMPAIGN / "base.json.gz"),
                  configurations=counts["radio_settings"], panels=counts["panels"], batches=len(batches),
                  inputs=sum(b["units"] for b in batches), source_records=len(base["values"]),
                  files={name: sha256(OUTPUT / name) for name in ("base.json.gz", "catalog.json.gz")},
                  source_sha256=sha256(Path(__file__)), evaluation_rows_used=0)
    (OUTPUT / "freeze.json").write_text(json.dumps(freeze, indent=1) + "\n")
    print(json.dumps({k: freeze[k] for k in ("configurations", "panels", "batches", "inputs", "source_records")}))


if __name__ == "__main__":
    main()
