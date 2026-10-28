"""Table III of the paper, the fitted link parameters of every site, from the method's profiles and deployment records.

Each site's anchor only profile holds the four link parameters its fit gave, kappa, the excess probability, the
relative excess mean and the relative offset spread, with Bregar per environment. The deployment records hold the
median reported anchor scale of each site's training part, from which the grid step follows. The parameters print with
four significant digits, a parameter at a bound of its fit as that bound, and the scale with one decimal.

Usage: fit_table.py [--out out/tables/fit-table.tex] [--check FILE]
       writes the tabular of the table, which the draft inputs, and with --check compares its rows with those that FILE
       holds, a draft's copy of the table or a section that types it
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "research/artifacts/solid/profiles"
DEPLOYMENT = ROOT / "research/artifacts/solid/fits/deployment.json"
HELDOUT_DEPLOYMENT = ROOT / "research/artifacts/solid/fits/heldout-deployment.json"
BOUNDS = ((1 - 1e-6, r"$1-10^{-6}$"), (1e-6, r"$10^{-6}$"), (1e-3, r"$10^{-3}$"))


def value(v):
    for bound, text in BOUNDS:
        if abs(v - bound) <= 1e-12 * max(1.0, bound):
            return text
    return f"{v:#.4g}".rstrip(".")


def sites():
    deployment = json.loads(DEPLOYMENT.read_text())["datasets"]
    heldout = json.loads(HELDOUT_DEPLOYMENT.read_text())["datasets"]
    for name, key, scales in (("TWR-CLOUD", "bern", deployment), ("TIERS", "tiers", deployment), ("IDLab", "idlab", deployment)):
        yield name, json.loads((PROFILES / f"{key}-anchor-only.json").read_text())["anchor_only"], scales[key]["anchor_scale_cm"]["median"]
    bregar = json.loads((PROFILES / "bregar-anchor-only.json").read_text())["environments"]
    for e in range(4):
        yield (f"Bregar {e}", bregar[f"environment{e}"]["anchor_only"],
               deployment["bregar"]["environments"][f"environment{e}"]["anchor_scale_cm"]["median"])
    for name, key in (("HYMN", "hymn"), ("AAU", "aau")):
        yield name, json.loads((PROFILES / f"{key}-anchor-only.json").read_text())["anchor_only"], heldout[key]["anchor_scale_cm"]["median"]


def rows():
    out = []
    for name, p, scale in sites():
        cells = [value(p[k]) for k in ("kappa", "excess_probability", "excess_mean_rel", "persistent_sd_rel")]
        out.append(" & ".join([name, *cells, f"{scale:.1f}"]) + r" \\")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "out/tables/fit-table.tex")
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    table = rows()
    tabular = [r"\begin{tabular}{@{}lccccc@{}}", r"\hline", r"Site & $\kappa$ & $\pib$ & $\murel$ & $\taurel$ & $\tilde s$ (cm) \\",
               r"\hline", *table, r"\hline", r"\end{tabular}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(tabular) + "\n")
    print("\n".join(table))
    print("wrote", args.out)
    if args.check:
        held = [line.strip() for line in args.check.read_text().splitlines()
                if re.match(r"^(TWR-CLOUD|TIERS|IDLab|Bregar \d|HYMN|AAU) &", line.strip())]
        same = held == table
        print("the draft's rows", "equal these rows" if same else "DIFFER from these rows")
        if not same:
            for a, b in zip(held, table):
                if a != b:
                    print("   draft  ", a)
                    print("   records", b)


if __name__ == "__main__":
    main()
