"""The counts and curve facts that the paper's text states, computed from the final tables and their curves.

The counts follow the definitions of the paper. A case is a dataset, a mode it supports and a reading count, 24 in
all, over the headline group (every cell in AF and AAC, the scenarios with three or more anchors in ANC) and the joined
pairs. SOLID is lowest when its S75 lies below that of every baseline of the case, and it is significantly better or
worse than the best baseline, the baseline with the lowest S75, when the paired bootstrap interval of that difference
excludes zero. The comparisons on TWR-CLOUD and on HYMN and AAU count every difference of SOLID with a baseline at
S50, S75 and S90 in the headline groups, and a win is a difference whose interval lies below zero. The cases with one
or two anchors are the ANC groups anchors-1 and anchors-2 with both reading counts.

The curve facts read the balanced CDFs that experiments/figures/export_cdfs.py exported for the figures. Each is the
level, or the gap in cm, at which one curve passes another, so the text that reads Fig. 3 and the CDF figures of the
companion document says what the curves show. The unit facts give the anchors of the agent of Fig. 2, which Section IV-B
names as its case of failed exchanges.

Usage: paper_counts.py [--tables research/artifacts/solid/tables] [--curves research/artifacts/solid/figures/data]
                       [--out FILE]
       prints the facts and, with --out, writes them as JSON
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("bern", "tiers", "bregar", "idlab", "hymn", "aau")
HEADLINE = {"AF": "every cell", "AAC": "every cell", "ANC": "headline, 3 or more anchors"}
FIGURE_AGENT = 122  # the agent that Fig. 2 draws, plot_posterior.py --agent 122


def rows(tables, dataset, name):
    with open(tables / dataset / "evaluation" / name, newline="") as stream:
        return list(csv.DictReader(stream))


def counts(tables):
    cases, wins, few = [], {"bern": [0, 0], "heldout": [0, 0]}, [0, 0]
    misses = []
    for dataset in DATASETS:
        percentiles, differences = rows(tables, dataset, "percentiles.csv"), rows(tables, dataset, "differences.csv")
        for mode in ("AF", "AAC", "ANC"):
            groups = [HEADLINE[mode]] + (["anchors-1", "anchors-2"] if mode == "ANC" else [])
            for group in groups:
                for track in ("1", "10"):
                    s75 = {r["method"]: float(r["S75_cm"]) for r in percentiles
                           if r["mode"] == mode and r["group"] == group and r["track"] == track and r["pair_set"] == "joined"
                           and r["S75_cm"] not in ("", "none")}
                    if "solid" not in s75 or len(s75) < 2:
                        continue
                    best_value, best = min((v, m) for m, v in s75.items() if m != "solid")
                    if group != HEADLINE[mode]:
                        few[0] += 1
                        few[1] += s75["solid"] < best_value
                        continue
                    pairs = {r["difference"].split(" minus ")[1]: r for r in differences
                             if r["mode"] == mode and r["group"] == group and r["track"] == track and r["pair_set"] == "joined"}
                    low, high = float(pairs[best]["S75_low_cm"]), float(pairs[best]["S75_high_cm"])
                    cases.append(dict(dataset=dataset, mode=mode, track=int(track), solid_s75_cm=s75["solid"], best=best,
                                      best_s75_cm=best_value, lowest=s75["solid"] < best_value, better=high < 0, worse=low > 0))
                    key = "bern" if dataset == "bern" else "heldout" if dataset in ("hymn", "aau") else None
                    if key is None:
                        continue
                    for baseline, r in pairs.items():
                        for level in ("S50", "S75", "S90"):
                            if r[f"{level}_low_cm"] in ("", "none"):
                                continue
                            win = float(r[f"{level}_high_cm"]) < 0
                            wins[key][0] += 1
                            wins[key][1] += win
                            if not win and key == "heldout":
                                misses.append(dict(dataset=dataset, track=int(track), baseline=baseline, level=level,
                                                   difference_cm=float(r[f"{level}_cm"]),
                                                   significant=float(r[f"{level}_low_cm"]) > 0))
    return dict(cases=len(cases), lowest=sum(c["lowest"] for c in cases), better=sum(c["better"] for c in cases),
                worse=sum(c["worse"] for c in cases), bern_comparisons=wins["bern"][0], bern_wins=wins["bern"][1],
                heldout_comparisons=wins["heldout"][0], heldout_wins=wins["heldout"][1], heldout_misses=misses,
                one_or_two_anchor_cases=few[0], one_or_two_anchor_lowest=few[1], per_case=cases)


def curves(folder, dataset, mode, track):
    record = json.loads((folder / f"cdfs-{dataset}.json").read_text())
    levels = [round(v, 3) for v in record["levels"]]
    return levels, {c["method"]: c["quantiles_cm"] for c in record["curves"]
                    if c["mode"] == mode and c["track"] == track and c["group"] == HEADLINE[mode] and c["pair_set"] == "joined"}


def spans(levels, flags, shortest=0.01):
    """The stretches of levels where a flag holds, those shorter than shortest left out."""
    out, start = [], None
    for k, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = k
        if not flag and start is not None:
            if levels[k - 1] - levels[start] >= shortest:
                out.append([levels[start], levels[k - 1]])
            start = None
    return out


def curve_facts(folder):
    facts = {}
    for dataset, mode in (("bern", "AF"), ("bern", "AAC"), ("bern", "ANC"), ("hymn", "ANC"), ("aau", "ANC"),
                          ("bregar", "ANC"), ("idlab", "AF"), ("idlab", "AAC"), ("idlab", "ANC"), ("tiers", "AF"),
                          ("tiers", "AAC"), ("tiers", "ANC")):
        for track in (1, 10):
            levels, c = curves(folder, dataset, mode, track)
            solid = c["solid"]
            entry = {}
            for method, q in c.items():
                if method == "solid":
                    continue
                entry[method] = dict(left_of_solid=spans(levels, [q[k] < solid[k] for k in range(len(levels))]))
                below_median = [solid[k] - q[k] for k in range(len(levels)) if levels[k] <= 0.5]
                entry[method]["largest_lead_below_median_cm"] = round(max(below_median), 2)
            for level in (0.5, 0.75, 0.9, 0.95, 0.99):
                k = levels.index(level)
                entry[f"level {level}"] = {m: round(q[k], 1) for m, q in c.items()}
            facts[f"{dataset} {mode} {track}"] = entry
    return facts


def posterior_unit(folder):
    """The anchors of the agent that Fig. 2 shows, by horizontal true distance, with whether it heard each, failed with it,
    or made no attempt because no agent of the scenario heard that anchor (Section IV-B)."""
    record = json.loads((folder / "posterior-illustration.json").read_text())
    anchors = {a[0]: a[1:3] for a in record["anchors"]}
    key = lambda p: (round(p[0], 2), round(p[1], 2))
    ids = {key(v): k for k, v in anchors.items()}
    heard_by_any = {ids[key(p)] for agent in record["agents"] for p in agent["present"]}
    agent = next(a for a in record["agents"] if a["agent_id"] == FIGURE_AGENT)
    heard, failed = {ids[key(p)] for p in agent["present"]}, {ids[key(p)] for p in agent["missing"]}
    rows = []
    for anchor, (x, y) in anchors.items():
        state = "heard" if anchor in heard else "failed" if anchor in failed else "no attempt"
        rows.append(dict(anchor=anchor, distance_m=round(((agent["truth"][0] - x) ** 2 + (agent["truth"][1] - y) ** 2) ** 0.5 / 100, 2),
                         outcome=state, heard_by_some_agent=anchor in heard_by_any))
    return dict(agent=agent["agent_id"], scenario=record["unit"]["scenario"], kept_cells=agent["kept_cells"],
                grid_cells=agent["grid_cells"], dropped_share=agent["dropped_share"],
                anchors=sorted(rows, key=lambda r: r["distance_m"]))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, default=ROOT / "research/artifacts/solid/tables")
    parser.add_argument("--curves", type=Path, default=ROOT / "research/artifacts/solid/figures/data")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = dict(counts=counts(args.tables), curves=curve_facts(args.curves), posterior_unit=posterior_unit(args.curves))
    c = result["counts"]
    print(f"{c['cases']} cases: SOLID lowest in {c['lowest']}, significantly better than the best baseline in {c['better']}, "
          f"significantly worse in {c['worse']}")
    print(f"TWR-CLOUD: {c['bern_wins']} of {c['bern_comparisons']} comparisons won significantly")
    print(f"HYMN and AAU: {c['heldout_wins']} of {c['heldout_comparisons']} comparisons won significantly")
    for m in c["heldout_misses"]:
        print(f"   not won: {m['dataset']} {m['track']} reading(s) {m['level']} against {m['baseline']}, {m['difference_cm']:+.1f} cm"
              f"{', significant' if m['significant'] else ''}")
    print(f"one or two anchors: SOLID lowest in {c['one_or_two_anchor_lowest']} of {c['one_or_two_anchor_cases']} cases")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
