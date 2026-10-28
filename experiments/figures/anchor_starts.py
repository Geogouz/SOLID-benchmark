"""How often FNL and STRONG return the positions they start from, from the scored runs behind the final tables.

A solve stays at its start when its estimate equals, agent for agent, the positions it starts from, which the scorer
flags in the row of every scenario of FNL and STRONG. For TIERS, IDLab, Bregar, HYMN and AAU this script reads the
baseline scores that the final tables of the evaluation part name as inputs, checks every file against the hash
that the tables record, and counts the stays of FNL and STRONG per mode, and in ANC also per anchor count. The scores of
TWR-CLOUD carry no flag, so its counts come from the benchmark report of each of its evaluation runs, which counts them
(stayed_at_start), after a check that the report is the one the tables read. Beside each count with one or two anchors
the record gives the S75 of every method from the final tables and the methods that are lowest at the printed precision
of 0.1 cm, so it shows who leads where SOLID does not. Nothing is solved.

Usage: anchor_starts.py [--runs ROOT] [--tables DIR] [--out FILE] [--table FILE]
       ROOT is the runs root the tables read, out/ of the repository unless given, in place of out/ of a path the tables
       record, DIR the folder of the final tables, research/artifacts/solid/tables unless given, and --table writes the
       tabular of the companion document's table of these counts
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from anchor_sides import array

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "out"
TABLES = ROOT / "research/artifacts/solid/tables"
DATASETS = (("bern", "TWR-CLOUD"), ("tiers", "TIERS"), ("idlab", "IDLab"), ("bregar", "Bregar"), ("hymn", "HYMN"),
            ("aau", "AAU"))
MODES = ("AF", "AAC", "ANC")
STARTED = ("fnl", "strong")
FEW = (1, 2)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def local(path, runs):
    """A path that the tables record: below the runs root in place of out/, relative to the repository, or as it is
    when absolute (final_tables.py build --runs)."""
    if Path(path).is_absolute():
        return Path(path)
    if path.startswith("out/"):
        return runs / path[len("out/"):]
    return ROOT / path


def rows(path):
    """The score rows of a file, those of a JSON array one at a time (anchor_sides.array), since a score file can hold
    gigabytes."""
    if path.suffix == ".gz":
        return (json.loads(line) for line in gzip.open(path, "rt"))
    return array(path)


def scores(dataset, mode, tables):
    """The input of the tables that holds the baseline scores of one mode, None for a mode without them. Bregar's lie
    where bregar_layout.py evaluate writes them, in the data folder."""
    if dataset == "bregar":
        ends = ("/bregar-layout/results/evaluation/scores.jsonl.gz",) if mode == "ANC" else ()
    else:
        ends = (f"/baselines/{dataset}/evaluation/{mode.lower()}/scores.json",)
    recorded = [e for e in tables["inputs"] if e["path"].endswith(ends)] if ends else []
    assert len(recorded) <= 1, (dataset, mode)
    return recorded[0] if recorded else None


def count(n):
    """A count as the paper prints it, with thin spaces between groups of three digits from five digits on."""
    return f"{n:,}".replace(",", r"\,") if n >= 10000 else str(n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()

    sources, stays, few = {}, {}, []
    for dataset, _ in DATASETS:
        if dataset == "bern":
            continue
        tables = json.load(open(args.tables / f"{dataset}/evaluation/tables.json"))
        stays[dataset] = {}
        for mode in MODES:
            entry = scores(dataset, mode, tables)
            if entry is None:
                continue
            path = local(entry["path"], args.runs)
            assert sha256(path) == entry["sha256"], f"{path} is not the file the tables read"
            sources[f"{dataset}/{mode}"] = entry
            solves, stayed = Counter(), Counter()
            for r in rows(path):
                if r["method"] in STARTED:
                    group = (r["method"], r["anchors"] if mode == "ANC" and r["anchors"] in FEW else None)
                    solves[group] += 1
                    stayed[group] += bool(r["stayed_at_start"])
            stays[dataset][mode] = {m: dict(solves=sum(v for k, v in solves.items() if k[0] == m),
                                            stayed=sum(v for k, v in stayed.items() if k[0] == m))
                                    for m in STARTED if any(k[0] == m for k in solves)}
            if mode != "ANC" or f"anchors-{FEW[0]}" not in tables["modes"]["ANC"]["groups"]:
                continue
            for anchors in FEW:
                stays[dataset][f"ANC, {anchors} anchor{'s' if anchors > 1 else ''}"] = {
                    m: dict(solves=solves[(m, anchors)], stayed=stayed[(m, anchors)]) for m in STARTED}
                group = tables["modes"]["ANC"]["groups"][f"anchors-{anchors}"]["tracks"]
                assert sum(t["units"] for t in group.values()) == solves[("fnl", anchors)] == solves[("strong", anchors)]
                for readings, track in sorted(group.items(), key=lambda kv: int(kv[0])):
                    s75 = {m: round(v["S75"]["value_cm"], 1) for m, v in track["pair_sets"]["joined"]["methods"].items()}
                    few.append(dict(dataset=dataset, anchors=anchors, readings=int(readings), scenarios=track["units"],
                                    S75_cm=s75, lowest=sorted(m for m in s75 if s75[m] == min(s75.values()))))

    tables = json.load(open(args.tables / "bern/evaluation/tables.json"))
    recorded = {e["path"]: e["sha256"] for e in tables["inputs"]}
    stays["bern"] = {}
    for mode in MODES:
        for method in STARTED:
            run = tables["runs"].get(mode, {}).get(method)
            if run is None:
                continue
            path = local(run["record"], args.runs)
            assert sha256(path) == recorded[run["record"]], f"{path} is not the report the tables read"
            sources[f"bern/{mode}/{method}"] = dict(path=run["record"], sha256=recorded[run["record"]])
            for m, c in (json.load(open(path)).get("stayed_at_start") or {}).items():
                stays["bern"].setdefault(mode, {})[m] = dict(solves=c["solves"], stayed=c["stayed"])

    for dataset, entries in stays.items():
        for mode, entry in entries.items():
            assert len({e["solves"] for e in entry.values()}) == 1, (dataset, mode, entry)
    elsewhere = [g for g in few if "solid" not in g["lowest"]]
    result = dict(
        sources=sources,
        stays=stays,
        one_and_two_anchors=few,
        summary=dict(
            combinations=len(few),
            solid_lowest=len(few) - len(elsewhere),
            fnl_or_strong_lowest=sum(set(g["lowest"]) <= set(STARTED) for g in elsewhere),
            one_anchor_every_scenario_stayed=sorted(d for d, e in stays.items() if "ANC, 1 anchor" in e and
                                                    all(c["stayed"] == c["solves"] for c in e["ANC, 1 anchor"].values())),
            S75_range_cm=[min(v for g in few for v in g["S75_cm"].values()),
                          max(v for g in few for v in g["S75_cm"].values())],
        ),
    )
    print(json.dumps(result["summary"], indent=1))
    for g in few:
        print(g["dataset"], g["anchors"], g["readings"], g["scenarios"], g["lowest"], g["S75_cm"])
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", args.out)
    if args.table:
        lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\hline", r"Dataset & Mode & All solves & FNL & STRONG \\", r"\hline"]
        for dataset, name in DATASETS:
            modes = [m for m in (*MODES, "ANC, 1 anchor", "ANC, 2 anchors") if m in stays[dataset]]
            for i, mode in enumerate(modes):
                entry = stays[dataset][mode]
                cells = [count(entry[m]["stayed"]) if m in entry else "--" for m in STARTED]
                label = mode.replace("ANC, ", r"\quad ")
                lines.append(" & ".join([name if i == 0 else "", label, count(next(iter(entry.values()))["solves"]),
                                         *cells]) + r" \\")
        lines += [r"\hline", r"\multicolumn{5}{@{}l@{}}{\rule{0pt}{2.4ex}A dash marks a mode in which STRONG does not run.}\\",
                  r"\end{tabular}"]
        args.table.write_text("\n".join(lines) + "\n")
        print("wrote", args.table)


if __name__ == "__main__":
    main()
