"""The paper's final tables from the runs of the declared analysis, and the declaration itself.

The analysis is declared before every run the tables read. declare writes
research/artifacts/solid/splits/final-analysis.json from the constants of this module and the designs of decision.py,
and the file is committed and pinned by the method freeze (research/benchmarks/datasets/heldout.py freeze_pins) before
the runs a table reads. It fixes for every dataset its halves and the runs a table reads, the modes and the groups of
each mode, the sample tracks, the percentiles, the pair sets, the methods, and the paired cluster bootstrap of the
dataset with its draws and seed. It declares no extra SOLID row, which the builder would compare with SOLID. build
reads every value of the analysis from the committed declaration, which must be the one this module declares, and
every output records the declaration's SHA-256.

A table per dataset, half, mode, group, sample track and pair set holds every method's balanced S50, S75 and S90 with
their intervals, SOLID minus every baseline with paired intervals, the share of pairs and of units the pair set scores,
and per method the failed and aborted units and the median solve time per unit with the threads and processes of its
run. The anchor only mode reports the headline over the units with three or more anchors and then the one anchor and
the two anchor rows, and the anchor aided mode its headline over every cell and then the same one anchor and two anchor
rows. The joined pairs are the headline, the ranged pairs a supplementary figure, and TWR-CLOUD-Bern
keeps its released scorer's figure beside them. The intervals come from the paired cluster bootstrap of decision.py,
and every method, mode, group, track and pair set of a dataset shares each draw. A SOLID run of a held-out dataset
names the declared outcomes of its training half that the held-out declaration has the tables mark, a link fit that
did not converge among them (heldout_runner.marks), and tables.json keeps them with the run while tables.tex marks
SOLID with a dagger and a note for each. A budget check of the confirmatory round without a converged step marks SOLID
the same way in the anchor only tables that each comparison still moving at its last doubling spans, as the budget
entry of the method record names them (budget_marks). The data defects of a held-out half, the AAU flights its builds
left out of the half or of the training half beyond the relocation guard and the pairs the guard could not compare,
hold for every method alike, and tables.json and tables.tex note them without a mark (heldout_notes).

build refuses a dataset
- while the declaration is missing, differs from its committed version or differs from the declaration of this module,
- whose runs name more than one source identity or machine, or a SOLID binary that differs from its sources,
- with a SOLID run of another method record than the committed research/artifacts/solid/selections/method.json, and on
  an evaluation half with a run of SOLID's row that did not solve the committed tracked profile of its dataset and mode,
- with a baseline run whose selection record fails its dataset's selection check, or an ADE-BP run of another fit of
  its ranging model than the committed one,
- whose methods hold different units, and
- on an evaluation half, unless the verified method freeze pins this declaration and every run was made by the frozen
  executables, with the SHA-256 the freeze pins for its executable, solid-units, heldout-adapter, the Bern CLI or the
  adapter of TIERS, Bregar or IDLab, and its source identity for every executable, and unless every run names a commit
  that holds this declaration and the method freeze as they are,
- on an evaluation half, while a table of an evaluation half differs from its committed version, or when one, as it is
  or in any committed version, was written under another declaration, and
- on the evaluation half of HYMN or AAU Meshed UWB, unless the committed ledger closes the dataset under the method
  freeze and pins every file the table reads with the SHA-256 it records.

Every table of an evaluation half is committed as soon as it is written. The builder refuses the next table of an
evaluation half, of any dataset, until the last one is committed as it was written, so the history holds every table and
the declaration it was written under. A table dropped before its commit leaves no trace, but the declaration it was
written under cannot change without the code of this module, which the method freeze fixes with the round's sources.

The outputs go to research/artifacts/solid/tables/<dataset>/<split>. tables.json holds every number, the runs and the
SHA-256 of every file read, and percentiles.csv, differences.csv and tables.tex hold the same numbers. They are written
in a fixed order without a time stamp, so the same runs give the same bytes. The evaluation halves are the reported
ones. The training halves pass through the same builder as a dry run. A run that does not exist yet is listed there as
missing, and its mode is left out when it is SOLID's or a baseline's.

A reader's reproduction (SOLID_READER=1) builds the tables from the reader's own runs under the analysis this module
declares (declaration), and every guard that exists only to prove the research process is a no-op there: the committed
declaration, the one source identity and host per dataset, the method record and its budget marks, the selection
records, ADE-BP's committed fits, the method freeze and the run hashes it pins, the committed profiles and tables and the
ledger. Every check of the runs against each other and against their built halves stays. The anchor only rosters of
Bregar, HYMN and AAU Meshed UWB hold ADE-BP there, as the reported tables do (HEIGHT_SPREAD). Every run is read below
one runs root, out/ of the repository unless --runs names another, in the declared layout with out/ standing for the
runs root, and a built half or a run that the declaration places in research/data, Bregar's baselines from
bregar_layout.py evaluate among them, in the data folder SOLID_DATA when it is set. Every baseline of a mode is read
from the mode's one baseline run, ADE-BP's rows beside the other baselines', except a baseline whose run lies alone in
the subfolder of that run named after it, which is read from there (alone). --tables names another folder for the
outputs.

Usage: final_tables.py declare [--replace]
       final_tables.py check
       final_tables.py build --dataset bern|tiers|bregar|idlab|hymn|aau [--split evaluation|training] [--runs DIR] [--tables DIR]
       --runs and --tables serve a reader's reproduction (SOLID_READER=1)
"""
import argparse
import bisect
import csv
import functools
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import decision  # noqa: E402
import scoring  # noqa: E402

# the Bern runner, whose command a Bern baseline run must hold (bern_selection imports it), and the folder of its
# evaluation runs that the tables read (research/benchmarks/bern/run_bern.py RUNS)
if str(common.ROOT / "research/benchmarks/bern") not in sys.path:
    sys.path.append(str(common.ROOT / "research/benchmarks/bern"))
RUNS = "out/bern-baselines-v3"
# a reader's reproduction, which reads the reader's own runs and has no research record (the module's docstring)
READER = os.environ.get("SOLID_READER") == "1"
# the data folder, research/data of the repository unless SOLID_DATA names another
DATA = os.environ.get("SOLID_DATA")

SCHEMA = "solid.analysis.final-tables.v1"
TABLES_SCHEMA = "solid.tables.final.v1"
# paths relative to the repository root
DECLARATION = "research/artifacts/solid/splits/final-analysis.json"
TABLES = "research/artifacts/solid/tables"
METHOD_RECORD = "research/artifacts/solid/selections/method.json"
# the tracked profiles SOLID's runs of an evaluation half solve, <PROFILES>/<dataset>-<kind of the mode>.json, which the
# evaluation stage composes from the committed method record and commits before the runs (held-out stage one those of
# HYMN and AAU Meshed UWB)
PROFILES = "research/artifacts/solid/profiles"
PROFILE_KINDS = dict(AF="agent-only", AAC="anchor-aided", ANC="anchor-only")
# the method freeze record, its schema and the key of its pins, and the ledger of the held-out benchmarks with the step
# of held-out stage one (research/benchmarks/datasets/heldout.py FREEZE, FREEZE_SCHEMA, FREEZE_KEY, LEDGER and STAGE_ONE)
FREEZE = "research/artifacts/solid/selections/method-freeze.json"
FREEZE_SCHEMA = "solid.selection.method-freeze.v1"
FREEZE_KEY = "heldout_benchmarks"
LEDGER = "research/artifacts/solid/splits/heldout-evaluations.json"
STAGE_ONE = "stage-one"
# the held-out datasets, whose evaluation the ledger closes
HELDOUT = ("hymn", "aau")
# the fit records of ADE-BP's ranging model on the training halves that its runs name (ade_training.py)
ADE_FITS = dict(tiers="research/artifacts/baselines/tiers-ade-fit.json", idlab="research/artifacts/baselines/idlab-ade-fit.json")
LEVELS = (("S50", "0.5"), ("S75", "0.75"), ("S90", "0.9"))
TRACKS = (1, 10)
INTERVAL = "0.95"
REPLICATES = 2000
SEED = 20260926
HEADLINE_MIN_ANCHORS = scoring.HEADLINE_MIN_ANCHORS
RELEASED = "released"
ORDER = ("solid", "jin-2021", "ade-bp-2015", "fnl", "strong")
LABELS = {"solid": "SOLID", "jin-2021": "Jin 2021", "ade-bp-2015": "ADE-BP 2015", "fnl": "FNL 2025", "strong": "STRONG 2021"}
AGENT_ONLY = ("solid", "jin-2021", "fnl")
ANCHORED = ("solid", "jin-2021", "ade-bp-2015", "fnl", "strong")
# the roster of scenes with real height spread, where ADE-BP, which has no spatial implementation, does not run. ADE-BP
# joined the tables of Bregar, HYMN and AAU Meshed UWB on 4 October 2026 in the place ORDER gives it, so a reader's
# reproduction builds them with it, as the paper reports them
HEIGHT_SPREAD = ANCHORED if READER else ("solid", "jin-2021", "fnl", "strong")
# the failure messages of a solve that crashed its process, SOLID's runner and the baseline adapters
ABORTS = (common.ABORT_PREFIX, "adapter aborted")
# SOLID rows beside the method's. The builder supports a declared one, dict(name, label, dataset, modes, runs={split: run
# directory with {mode}}) with the run directory in the layout of the dataset's SOLID runs, but the tables hold no SOLID
# variant row, so none is declared.
EXTRA_ROWS = ()
# the most draw and unit elements of one table held at once, a whole block of draws below 400,000 units
CHUNK = 20_000_000
# the code of the analysis, recorded in every output
CODE = ("research/solid/final_tables.py", "research/solid/decision.py", "research/solid/common.py", "research/benchmarks/datasets/scoring.py")

ROLE = ("The analysis of the paper's final tables. It is committed before every run the tables read, and research/solid/final_tables.py "
        "builds every table from the runs it names and from nothing else. The tables of the evaluation halves are the only source "
        "of the paper's numbers. The training halves pass through the same builder as a dry run and are never reported. The "
        "dashboard under research/dashboard is a development view.")
RULES = dict(
    declaration=("This declaration is committed before every run the tables read, and every output records its SHA-256. The "
                 "builder writes nothing while the declaration is missing, differs from its committed version or differs from the "
                 "declaration of research/solid/final_tables.py. The method freeze research/artifacts/solid/selections/method-freeze.json "
                 "pins its SHA-256 beside the code and the executables, and a table of an evaluation half is built only under the "
                 "verified method freeze that pins it. Every run such a table reads was made by the frozen executables. A run of "
                 "solid-units or heldout-adapter names the SHA-256 the freeze pins, and every run names the source identity of the "
                 "frozen executables, so no run made with earlier code enters a table. The freeze pins solid-units, heldout-adapter, "
                 "the TWR-CLOUD-Bern CLI and the adapters of TIERS, Bregar and IDLab. Every run, SOLID's and every baseline's, names "
                 "the commit it ran at, and that commit holds this declaration and the method freeze as they are, so no run made "
                 "before the freeze was committed enters a table either. A table of HYMN or AAU Meshed UWB reads only the files "
                 "that the ledger's closing entry of its dataset or of held-out stage one pins, each with the SHA-256 the entry "
                 "records. Every table of an "
                 "evaluation half is committed as soon as it is written. The builder refuses an evaluation half while a table of an "
                 "evaluation half differs from its committed version, and while a table of an evaluation half, as it is or in any "
                 "committed version, names another declaration."),
    pairing=("The rows of every method pair with SOLID's on the key, identity and sample count of a unit. Every method holds the "
             "same units, each in the same cell and cluster with the same anchor and pair counts, and every mode the declared "
             "roster. Anything else is refused."),
    score=("A unit's error is its mean capped pair error over the pairs its measured links join, the joined pair set of "
           "research/benchmarks/datasets/scoring.py and the headline of every table. The ranged pairs, those joined by links that "
           "carry a range, are a supplementary figure beside it, and TWR-CLOUD-Bern keeps its released scorer's figure, the "
           "benchmark's own, beside both. A unit without a pair of a set is left out of that set's percentiles, as the released "
           "scorer leaves out a scenario without eligible pairs. Before any scorer reads the predictions, a coordinate that is not "
           "finite or whose magnitude exceeds 1e7 cm is treated as missing, for every method alike."),
    groups=("The anchor free mode reports every cell together. The anchor aided mode reports every cell together as its headline and "
            "then every anchor count below three alone, the one anchor and the two anchor rows, each when it holds units, since a "
            "method whose anchored decoding needs three anchors can fail with fewer. The anchor only mode reports the headline over "
            "the units with three or more anchors first and then every smaller anchor count alone, the one anchor and the two anchor "
            "rows, each when it holds units."),
    percentiles=("The balanced S50, S75 and S90 of each group per sample track, one and ten, every cell weighted equally by the "
                 "released scorer's rule. The reported value is the released scorer's arithmetic with the cells in the order of "
                 "their first unit, the units taken in the order of their keys and, on TWR-CLOUD-Bern, in the order its released "
                 "scorer read them. The value of the exact balanced CDF is recorded beside it."),
    intervals=("Every percentile and every difference carries the 95 percent interval of its paired cluster bootstrap, the 2.5th and "
               "the 97.5th percentiles of its draws, and their standard deviation. decision.py draws the clusters of the dataset's "
               "design within their strata with replacement, n minus one of the n clusters of a stratum, each drawn copy counting n "
               "over n minus one times, the rescaling bootstrap of Rao and Wu (1988), and every draw's percentile is that of the "
               "exact balanced CDF. Every method, mode, group, sample "
               "track and pair set of a dataset shares each draw, so every difference is paired. A table that some draw leaves "
               "without any of its units has no interval, and it records how many draws left it empty."),
    differences=("SOLID minus each baseline per group, sample track, pair set and percentile, with its paired interval. A declared "
                 "extra SOLID row is compared with SOLID, the extra row minus SOLID."),
    shares=("Every table reports its share of pairs scored, the pairs of its set over every agent pair of its units, and its share of "
            "units scored, the units holding a pair of the set over all its units. Both follow from the inputs alone and are the "
            "same for every method, which the pairing checks."),
    failures=("Per method, group and sample track, the units whose solve reported a failure and the units among them whose solve "
              "crashed its process, an abort of SOLID's runner, of an adapter or of the Bern CLI. On TWR-CLOUD-Bern a failed unit is "
              "one whose released score penalizes a pair for a missing or invalid position."),
    runtime=("Per method, group and sample track, the median solve time per unit in milliseconds with the threads per process and "
             "the processes of its run. On TWR-CLOUD-Bern it is the median over every solve of the run, the one its report keeps."),
    identity=("One binary per dataset. Every run a table of a dataset and half reads names one source identity, the digest of the "
              "sources its executable is built from (research/cli/build.rs), and one host, and the runs of one executable that "
              "record its SHA-256 name one. A run that lacks the identity or the host is refused, and so is a SOLID run whose "
              "binary differs from the sources of its checkout. Hosts are compared without their platform, whose kernel release a "
              "reboot of the same machine changes, and the tables record every platform."),
    ade_fit=("An ADE-BP run on TIERS or IDLab names the SHA-256 of the committed fit of its ranging model on the training half, "
             f"{ADE_FITS['tiers']} or {ADE_FITS['idlab']}, and a run of another fit is refused."),
    method=("Every SOLID run, the declared extra rows among them, records the SHA-256 of the method record "
            f"{METHOD_RECORD}, and the record is committed as it is. A run of another method record or of none is refused. On "
            f"an evaluation half SOLID's row solves the tracked profile of its dataset and mode, {PROFILES}/<dataset>-<agent-only, "
            "anchor-aided or anchor-only>.json, which the evaluation stage composes from the committed method record and commits "
            "before the runs and which every runner checks against that composition. Its run names that profile and the SHA-256 "
            "of the profile it solved, which must be that of the profile as committed and as the commit of the run holds it."),
    budget=("When the budget check of the confirmatory round converged at none of its declared doublings, the method takes the "
            "largest budgets it tried and its record names every anchor only comparison that still moved at the last doubling. "
            "SOLID then carries a dagger in every anchor only table of its dataset that such a comparison spans, the headline for "
            "a comparison over three or more anchors and every anchor only table for one over all units, with a note that names "
            "the comparison's sample track, percentile, difference and margin."),
    selections=("Jin 2021, FNL 2025 and STRONG 2021, and ADE-BP 2015's rounds in the anchor aided mode, take the options their "
                "selection on the dataset's training half chose, one record per dataset and method. A baseline run of a method whose "
                "options its mode selects is refused unless the check-selection command of its runner accepts that method's record and "
                "the run names the record's SHA-256 and the declared base configurations and shared inputs the record was selected "
                "with. On TWR-CLOUD-Bern the records come from the eleven held training scans, and a run's command.json must name the "
                "record's SHA-256 and selected candidate, its method-config.json must hold the declared settings of that candidate and "
                "its command the flags research/benchmarks/bern/run_bern.py builds from them. Every other Bern baseline run holds the "
                "retained configuration of research/benchmarks/bern/baselines.json. The check reads the runs and writes nothing."),
    halves=("The evaluation halves are the reported ones, and every declared mode, method and extra row of an evaluation half must "
            "be present. A training half is a dry run of the builder. A run that does not exist yet is listed there as missing, and "
            "its mode is left out when it is SOLID's or a baseline's."))

DATASETS = {
    "bern": dict(
        label="TWR-CLOUD-Bern", layout="bern", design="bern-evaluation", modes=dict(AF=AGENT_ONLY, AAC=ANCHORED, ANC=ANCHORED),
        pair_sets=(scoring.PAIR_SET, *scoring.SUPPLEMENTARY, RELEASED),
        splits=dict(evaluation=("The released evaluation corpus of TWR-CLOUD-Bern, the 148,260 scenarios of its evaluation scan with both "
                                "iterations and both sample tracks, run by research/benchmarks/bern/run_bern.py --score.")),
        units_per_track=dict(AF=49_420, AAC=247_100, ANC=247_100),
        runs=dict(solid="out/solid/bern/{mode}", baselines=RUNS + "/{method}/{mode}"),
        reason=("The evaluation corpus comes from one scan. The release repeats every agent group of an agent count under every "
                "anchor count, the same agents in the same positions drawing their packets from one pool of the scan's readings, so an "
                "agent group is the cluster, with its scenarios of every anchor count, both iterations and both sample tracks. The "
                "groups are drawn within their agent count, whose number of groups the release fixes. The interval is conditional on "
                "the scan, its positions and its pool of readings."),
        selection=("baselines_bern_training.py check-selection, research/artifacts/baselines/bern-<method>-selection.json of Jin "
                   "2021, FNL 2025, STRONG 2021 and ADE-BP 2015's rounds on the eleven held training scans, and for ADE-BP in the anchor "
                   "only mode the retained configuration of research/benchmarks/bern/baselines.json.")),
    "tiers": dict(
        label="TIERS", layout="run", design="tiers", modes=dict(AF=AGENT_ONLY, AAC=ANCHORED, ANC=ANCHORED),
        pair_sets=(scoring.PAIR_SET, *scoring.SUPPLEMENTARY),
        splits=dict(evaluation=("The second half of every recording's blocks by rank, the evaluation half of "
                                "research/artifacts/solid/splits/layout-split.json. The block of rank k holds the reports 10k to 10k+9 of "
                                "every link, so the halves split every link's reports by their order and overlap in time across links "
                                "that report at different rates."),
                    training="The first half of every recording's blocks by rank, the training half, for the dry run."),
        runs=dict(solid="out/solid/tiers/{split}/{mode}", baselines="out/baselines/tiers/{split}/{mode}"),
        reason=("Runs of three consecutive blocks within each recording, the TIERS design of decision.py. The blocks of a recording "
                "are consecutive windows of one static layout, and a run keeps neighbouring blocks together. The interval is "
                "conditional on the three recordings."),
        selection=("layout_baselines.py tiers check-selection, research/artifacts/baselines/tiers-<method>-selection.json of Jin "
                   "2021, FNL 2025, STRONG 2021 and ADE-BP 2015's rounds on the training half.")),
    "bregar": dict(
        label="Bregar", layout="summary", design="bregar", modes=dict(ANC=HEIGHT_SPREAD),
        pair_sets=(scoring.PAIR_SET, *scoring.SUPPLEMENTARY),
        splits=dict(evaluation="The second and third blocks of ten reports of every environment and channel.",
                    training="The first block of ten reports of every environment and channel, for the dry run."),
        runs=dict(solid="out/solid/bregar/{split}/{mode}", baselines="research/data/external-benchmarks/bregar-layout/results/{split}"),
        reason=("The 24 environment and channel units together, without strata, the Bregar design of decision.py. The interval is "
                "conditional on the four environments, whose six channels share their tag positions and anchors. Drawing the units "
                "across the environments adds the spread between the environments to the draws, which makes the interval "
                "conservative for that target. On the audit's synthetic null it covered 0.98 to 0.99 at nominal 0.95. Drawing the six "
                "channels within their environment undercovers when the environments differ more than their channels, 0.79 when "
                "environment effects dominate, since six channels per environment are too few. No design gives coverage across new "
                "environments with four of them."),
        selection=("bregar_layout.py check-selection, research/data/external-benchmarks/bregar-layout/<method>-selection-v3.json of "
                   "Jin 2021, FNL 2025 and STRONG 2021 per environment on the training block.")),
    "idlab": dict(
        label="IDLab", layout="runs", design="idlab", modes=dict(AF=AGENT_ONLY, AAC=ANCHORED, ANC=ANCHORED),
        pair_sets=(scoring.PAIR_SET, *scoring.SUPPLEMENTARY),
        splits=dict(evaluation="The frozen campaign of the reserved evaluation reports, idlab-campaign.",
                    training="The campaign of the reserved calibration rows, idlab-training-campaign, for the dry run."),
        runs=dict(solid="out/solid/idlab/{split}/{mode}", baselines="out/baselines/idlab/{split}/{mode}"),
        reason=("The radio configurations, the IDLab design of decision.py. The panels of a configuration take the records of its "
                "short links again (idlab_population.panel), so they are not independent, and a cell of a configuration and an "
                "anchor count lies in one configuration."),
        selection=("idlab_layout_baselines.py check-selection, research/artifacts/baselines/idlab-<method>-selection.json of Jin "
                   "2021, FNL 2025, STRONG 2021 and ADE-BP 2015's rounds on the training half.")),
    "hymn": dict(
        label="HYMN", layout="run", design="hymn", modes=dict(ANC=HEIGHT_SPREAD), pair_sets=(scoring.PAIR_SET, *scoring.SUPPLEMENTARY),
        splits=dict(evaluation="The evaluation points of research/artifacts/solid/splits/heldout-benchmarks.json.",
                    training="The training points, for the dry run."),
        runs=dict(solid="out/solid/hymn/{split}/{mode}", baselines="out/baselines/hymn/{split}/{mode}",
                  half="research/data/external-benchmarks/hymn-layout/{split}"),
        unit_fields=("window",),
        reason=("The windows of ten epochs. Two units share a reading only when they share its window, and a point cannot be a "
                "cluster since every unit spans eight of them. The interval is conditional on the 24 evaluation points and the ten "
                "anchors. The twelve windows are too few to join into runs, so the interval takes neighbouring windows of a point as "
                "independent given the points."),
        selection=("heldout_baselines.py hymn check-selection, research/artifacts/baselines/hymn-<method>-selection.json of Jin "
                   "2021, FNL 2025 and STRONG 2021 on the training half.")),
    "aau": dict(
        label="AAU Meshed UWB", layout="run", design="aau", modes=dict(ANC=HEIGHT_SPREAD), pair_sets=(scoring.PAIR_SET, *scoring.SUPPLEMENTARY),
        splits=dict(evaluation="The last seven flights of research/artifacts/solid/splits/heldout-benchmarks.json.",
                    training="The first seven flights, for the dry run."),
        runs=dict(solid="out/solid/aau/{split}/{mode}", baselines="out/baselines/aau/{split}/{mode}",
                  half="research/data/external-benchmarks/aau-layout/{split}"),
        unit_fields=("flight", "block"),
        reason=("Runs of two consecutive blocks of ten reports within each flight, as TIERS draws runs of its blocks within "
                "recordings. A drawn run counts n over n minus one times, n the runs of its flight, so every flight keeps its weight "
                "in the anchor count cells, which span the flights. The nine static nodes that every flight shares are conditioned "
                "on."),
        selection=("heldout_baselines.py aau check-selection, research/artifacts/baselines/aau-<method>-selection.json of Jin "
                   "2021, FNL 2025 and STRONG 2021 on the training half.")),
}


class Refused(ValueError):
    """A table that the declared analysis does not allow, a guard that failed or runs that do not pair."""


# ---------------------------------------------------------------------------------------------------------------------
# The declaration


def declaration():
    """The analysis this module declares, from its constants and the designs of decision.py."""
    datasets = {}
    for name, spec in DATASETS.items():
        design = decision.DESIGNS[spec["design"]]
        entry = dict(label=spec["label"], splits={split: dict(reported=split == "evaluation", units=text) for split, text in spec["splits"].items()},
                     modes={mode: list(roster) for mode, roster in spec["modes"].items()}, pair_sets=list(spec["pair_sets"]),
                     design=dict(name=spec["design"], **design.describe(), reason=spec["reason"]), replicates=REPLICATES, seed=SEED,
                     layout=spec["layout"], runs=dict(spec["runs"]), selection=spec["selection"])
        if "unit_fields" in spec:
            entry["unit_fields"] = list(spec["unit_fields"])
        if "units_per_track" in spec:
            entry["units_per_track"] = dict(spec["units_per_track"])
        datasets[name] = entry
    return dict(schema=SCHEMA, statement="research/solid/final_tables.py", role=ROLE, rules=RULES,
                percentiles=[[name, level] for name, level in LEVELS], tracks=list(TRACKS), interval=INTERVAL,
                groups=dict(minimum_anchors=HEADLINE_MIN_ANCHORS),
                pair_sets=dict(headline=scoring.PAIR_SET, supplementary=list(scoring.SUPPLEMENTARY), released=dict(bern=RELEASED)),
                methods=dict(order=list(ORDER), labels=LABELS, solid=f"The method of the method record {METHOD_RECORD}."),
                method_record=METHOD_RECORD, extra_rows=[dict(row) for row in EXTRA_ROWS], datasets=datasets)


def encoded(value):
    return (json.dumps(value, indent=1, allow_nan=False) + "\n").encode()


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".partial")
    staged.write_bytes(data)
    staged.replace(path)


def evaluation_tables(root):
    """The tables of evaluation halves written so far."""
    return sorted((Path(root) / TABLES).glob("*/evaluation/tables.json"))


def evaluation_table(name):
    """Whether a path of the repository is the tables.json of an evaluation half."""
    parts = Path(name).parts
    return name.startswith(TABLES + "/") and len(parts) >= 3 and parts[-2:] == ("evaluation", "tables.json")


def committed_tables(root):
    """Every version of a table of an evaluation half that a commit of HEAD's history holds, as (commit, path, bytes)."""
    history = git(root, "log", "--format=%H", "HEAD", "--", TABLES)
    if history is None or history.returncode != 0:
        return []
    versions = []
    for commit in history.stdout.decode().split():
        listed = git(root, "ls-tree", "-r", "--name-only", commit, "--", TABLES)
        for name in (listed.stdout.decode().splitlines() if listed is not None and listed.returncode == 0 else []):
            if evaluation_table(name):
                versions.append((commit, name, shown(commit, Path(root) / name, root)))
    return versions


def declare(root=None, replace=False):
    """Write the declaration. An existing one that differs is replaced only on request and never once a table of an
    evaluation half exists, in the working tree or in any committed version, since the analysis is fixed before every
    run the tables read."""
    root = Path(root or common.ROOT)
    path = root / DECLARATION
    data = encoded(declaration())
    if path.exists():
        if path.read_bytes() == data:
            return False
        if not replace:
            raise Refused(f"{DECLARATION} exists with another analysis. A declared analysis is replaced only by a reviewed commit "
                          "before any evaluation run, with declare --replace")
        written = [relative(table, root) for table in evaluation_tables(root)] + [f"{name} at {commit[:12]}" for commit, name, _ in committed_tables(root)]
        if written:
            raise Refused(f"{DECLARATION} is fixed, a table of an evaluation half exists ({written[0]})")
    write_bytes(path, data)
    return True


def check(root=None, current=None):
    """The top level fields where the tracked declaration differs from the declaration of this module, or from the one
    given. No field in a reader's reproduction, which builds under the declaration of this module (reader_declaration)."""
    if READER:
        return []
    root = Path(root or common.ROOT)
    tracked = json.loads((root / DECLARATION).read_text())
    current = json.loads(encoded(declaration() if current is None else current))
    return sorted(key for key in set(tracked) | set(current) if tracked.get(key) != current.get(key))


# ---------------------------------------------------------------------------------------------------------------------
# Files and git


def relative(path, root):
    path, root = Path(path), Path(root)
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def located(root, path, runs_root=None):
    """A path of the declaration, relative to the repository root, where the build reads it: below the runs root in
    place of out/ when one is given, and in the data folder SOLID_DATA in place of research/data/ when it is set."""
    if runs_root is not None and path.startswith("out/"):
        return Path(runs_root) / path[len("out/"):]
    if DATA and path.startswith("research/data/"):
        return Path(DATA) / path[len("research/data/"):]
    return Path(root) / path


def git(root, *arguments):
    try:
        return subprocess.run(["git", *arguments], cwd=root, capture_output=True)
    except OSError:
        return None


def committed(path, root):
    """Whether a file is tracked by git and identical to its version at HEAD."""
    name = relative(path, root)
    tracked, unchanged = git(root, "ls-files", "--error-unmatch", "--", name), git(root, "diff", "--quiet", "HEAD", "--", name)
    return tracked is not None and unchanged is not None and tracked.returncode == 0 and unchanged.returncode == 0


def shown(commit, path, root):
    """The bytes of a file at a commit, None when the commit or the file at it is unknown."""
    run = git(root, "show", f"{commit}:{relative(path, root)}")
    return run.stdout if run is not None and run.returncode == 0 else None


class Reader:
    """The files a table reads, each with the SHA-256 of its bytes."""

    def __init__(self, root):
        self.root, self.files = Path(root), {}

    def file(self, path):
        path = Path(path)
        self.files[relative(path, self.root)] = common.sha256(path)
        return path

    def bytes(self, path):
        data = Path(path).read_bytes()
        self.files[relative(path, self.root)] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path):
        return json.loads(self.bytes(path))

    def rows(self, path):
        """Score rows, a JSON list or compressed JSON lines."""
        data = self.bytes(path)
        if Path(path).name.endswith(".jsonl.gz"):
            return [json.loads(line) for line in gzip.decompress(data).decode().splitlines() if line.strip()]
        return json.loads(data)


# ---------------------------------------------------------------------------------------------------------------------
# Runs


@dataclass
class Run:
    """One method's run of one mode as a table reads it: its record, the identity of its code, its host and executable,
    its threads and processes, and what the guards compare, the commit it ran at, the method record of a SOLID run and
    the selection record or the command of a baseline run."""
    method: str
    mode: str
    record: str
    executable: str
    identity: object = None
    host: object = None
    binary_sha256: object = None
    threads: object = None
    processes: object = None
    solid: bool = False
    matches_sources: bool = True
    commit: object = None
    method_record: object = None
    # the profile a SOLID run solved, its path in the repository and the SHA-256 of the file it solved
    profile: object = None
    profile_sha256: object = None
    selection: object = None
    population: object = None
    command: object = None
    runtime_ms: object = None
    aborted: frozenset = frozenset()
    # the declared outcomes a SOLID run of a held-out dataset names, which the tables mark beside SOLID's rows
    marks: tuple = ()

    def summary(self):
        entry = dict(record=self.record, executable=self.executable, binary_sha256=self.binary_sha256, host=self.host, threads_per_process=self.threads,
                     processes=self.processes)
        if self.solid:
            entry.update(method_record_sha256=self.method_record, commit=self.commit)
        elif self.selection is not None:
            entry["selection_sha256"] = self.selection
        if self.runtime_ms is not None:
            entry["median_solve_ms"] = self.runtime_ms
        if self.marks:
            entry["marks"] = list(self.marks)
        return entry


def expect(record, where, **values):
    for key, value in values.items():
        if key in record and record[key] != value:
            raise Refused(f"{where} is a run of {key} {record[key]}, not {value}")


def solid_run(reader, folder, method, dataset, split, mode):
    """A SOLID run in the layout of common.write_run: its record and the path of its scores."""
    path = folder / "run.json"
    record = reader.json(path)
    where = relative(path, reader.root)
    expect(record, where, dataset=dataset, split=split, mode=mode)
    if record.get("complete") is not True:
        raise Refused(f"{where} is not a complete run of its half")
    source, execution = record.get("source_identity") or {}, record.get("execution") or {}
    run = Run(method, mode, where, "solid-units", identity=source.get("binary"), host=execution.get("host"), binary_sha256=record.get("binary_sha256"),
              threads=execution.get("threads_per_process"), processes=execution.get("processes"), solid=True,
              matches_sources=source.get("binary_matches_sources") is True, commit=source.get("commit"),
              method_record=(record.get("method_record") or {}).get("sha256"), population=record.get("population_sha256"),
              marks=tuple(record.get("marks") or ()), profile=record.get("profile"), profile_sha256=record.get("profile_sha256"))
    return run, folder / "scores.json"


def baseline_runs(reader, folder, layout, executable, methods, split, mode):
    """The baselines' runs of one mode and the path of their scores: one record for every method (layout run, TIERS
    and the held-out datasets), one entry per method (runs, IDLab) or the summary of a half (summary, Bregar)."""
    if layout == "summary":
        path, scores = folder / "summary.json", folder / "scores.jsonl.gz"
        record = reader.json(path)
        found = sorted(entry["id"] for entry in record.get("methods") or [])
        if record.get("status") != "complete":
            raise Refused(f"{relative(path, reader.root)} is not a complete run of its half")
        entries = {method: record for method in found}
    else:
        path, scores = folder / "run.json", folder / "scores.json"
        record = reader.json(path)
        found = sorted(record.get("methods") or [])
        if layout == "runs":
            entries = record.get("runs")
            if not isinstance(entries, dict):
                raise Refused(f"{relative(path, reader.root)} keeps no entry per method, the layout of a run before one binary served every method")
            if any(entry.get("stride", 1) != 1 for entry in entries.values()):
                raise Refused(f"{relative(path, reader.root)} holds a strided run")
        else:
            entries = {method: record for method in found}
    where = relative(path, reader.root)
    expect(record, where, split=split, mode=mode)
    if found != sorted(methods) or sorted(entries) != sorted(methods):
        raise Refused(f"{where} holds the methods {found}, not the declared {sorted(methods)}")
    runs = {}
    for method in methods:
        entry = entries[method]
        expect(entry, where, split=split, mode=mode)
        runs[method] = Run(method, mode, where, executable, identity=entry.get("binary_identity"), host=entry.get("host"),
                           binary_sha256=entry.get("binary_sha256"), threads=entry.get("threads_per_process"), processes=entry.get("workers"),
                           commit=entry.get("commit"),
                           selection=((entry.get("selections") or {}).get(method) or {}).get("record_sha256"),
                           population=entry.get("population_sha256"))
    return runs, scores


def bern_run(reader, folder, method, mode, solid):
    """A run of research/benchmarks/bern/run_bern.py --score: its record and the folder of its scores."""
    command, report = reader.json(folder / "command.json"), reader.json(folder / "benchmark.json")
    where = relative(folder / "benchmark.json", reader.root)
    cli = "solid" if solid else method
    if command.get("method") != cli or command.get("mode") != mode or report.get("mode_filter") != mode:
        raise Refused(f"{where} is not a run of {cli} in {mode}")
    if command.get("limit") != 0 or report.get("limit", 0) != 0:
        raise Refused(f"{where} is not a run of the complete corpus")
    identity = report.get("implementation_revision")
    if machine(command.get("host")) != machine(report.get("host")):
        raise Refused(f"{where} and its command name two hosts")
    run = Run(method, mode, where, "solid", identity=identity, host=report.get("host"), binary_sha256=report.get("binary_sha256"),
              threads=report.get("threads_per_process"), processes=report.get("workers"), solid=solid, command=command.get("command"),
              commit=command.get("commit"),
              runtime_ms=report["solver_p50_micros"] / 1000,
              aborted=frozenset((a["scenario"], a["iteration"], a["samples"]) for a in report.get("aborted_solves") or []))
    if solid:
        run.matches_sources = command.get("sources") == identity
        run.method_record = (command.get("method_record") or {}).get("sha256")
        run.profile, run.profile_sha256 = command.get("profile"), command.get("profile_sha256")
    return run, folder


def present(layout, folder):
    """Whether the files of a run in the given layout exist."""
    names = dict(solid=("run.json", "scores.json"), run=("run.json", "scores.json"), runs=("run.json", "scores.json"),
                 summary=("summary.json", "scores.jsonl.gz"),
                 bern=("command.json", "benchmark.json", "official/scenario_scores.csv.gz", "official/track_summary.csv",
                       "joined/scenario_scores.csv.gz", "joined/summary.json"))[layout]
    return all((Path(folder) / name).is_file() for name in names)


EXECUTABLES = dict(tiers="tiers-adapter", bregar="bregar-adapter", idlab="idlab-layout-adapter", hymn="heldout-adapter", aau="heldout-adapter")


def alone(folder, layout):
    """The layout of the run of one baseline alone in a subfolder of a mode's baseline run, None without one. A reader's
    reproduction reads a baseline from there, as a run of that baseline made apart from the others lies there, the
    research record's runs of ADE-BP on Bregar, HYMN and AAU Meshed UWB among them. A reader's own runners write every
    baseline into the mode's one run."""
    if not READER:
        return None
    if present("summary", folder):
        return "summary"
    return ("runs" if layout == "runs" else "run") if present("run", folder) else None


def mode_runs(reader, root, dataset, split, spec, mode, extras, runs_root=None):
    """Every run of one mode, SOLID first, then the baselines and the extra rows, the paths of their scores and the
    runs that do not exist. On an evaluation half every run must exist. On a training half a mode without SOLID's or a
    baseline's runs gives None, and an extra row without its run is left out. The declared folders of the runs lie
    below the runs root given in place of out/ (located)."""
    template = spec["runs"]
    roster = spec["modes"][mode]
    layout = spec["layout"]
    bern = layout == "bern"
    folders = dict(solid=located(root, template["solid"].format(split=split, mode=mode.lower()), runs_root))
    if bern:
        folders.update({method: located(root, template["baselines"].format(method=method, mode=mode.lower()), runs_root) for method in roster[1:]})
    wanted = [("solid", folders["solid"], "bern" if bern else "solid")]
    if bern:
        wanted += [(method, folders[method], "bern") for method in roster[1:]]
    else:
        wanted.append(("baselines", located(root, template["baselines"].format(split=split, mode=mode.lower()), runs_root), layout))
    applying = [row for row in extras if mode in row["modes"]]
    if split == "evaluation" and any(split not in row["runs"] for row in applying):
        raise Refused(f"an extra row of {dataset} {mode} names no run of the evaluation half")
    rows = [(row["name"], located(root, row["runs"][split].format(mode=mode.lower()), runs_root), "bern" if bern else "solid")
            for row in applying if split in row["runs"]]
    absent = [relative(folder, root) for _, folder, kind in wanted + rows if not present(kind, folder)]
    if absent and split == "evaluation":
        raise Refused(f"the evaluation half of {dataset} {mode} lacks the runs {absent}")
    if any(not present(kind, folder) for _, folder, kind in wanted):
        return None, None, absent
    runs, scores = {}, {}
    for name, folder, _ in wanted + [row for row in rows if present(row[2], row[1])]:
        if bern:
            runs[name], scores[name] = bern_run(reader, folder, name, mode, name == "solid" or name not in roster)
        elif name == "baselines":
            apart = {method: alone(folder / method, layout) for method in roster[1:]}
            shared = [method for method in roster[1:] if apart[method] is None]
            parts = ([(folder, layout, shared)] if shared else []) + [(folder / method, apart[method], [method])
                                                                     for method in roster[1:] if apart[method] is not None]
            for place, kind, methods in parts:
                found, path = baseline_runs(reader, place, kind, EXECUTABLES[dataset], methods, split, mode)
                runs.update(found)
                scores.update({method: path for method in found})
        else:
            runs[name], scores[name] = solid_run(reader, folder, name, dataset, split, mode)
    return runs, scores, absent


# ---------------------------------------------------------------------------------------------------------------------
# Units


Entry = namedtuple("Entry", "key track cell anchors top stratum pairs possible errors failed aborted elapsed")


def row_entries(rows, design, pair_sets, metadata=None):
    """The entries of one method's score rows. The cell is the row's, its acquisition where it names none, and the
    clusters are the fields of the design, taken from the unit's built half where the rows lack them."""
    for row in rows:
        if metadata is not None:
            if row["id"] not in metadata:
                raise Refused(f"unit {row['id']} is not a unit of its built half")
            row = dict(row, **metadata[row["id"]])
        sets = row.get("pair_sets")
        if not isinstance(sets, dict) or any(name not in sets for name in (*pair_sets, "all")):
            raise Refused(f"the row of unit {row['key']} {row['id']} keeps no mean pair error of the pair sets {list(pair_sets)}")
        failure = row["failure"]
        yield Entry(key=(row["key"], row["id"], row["samples"]), track=row["samples"], cell=decision.plain(row["cell"] if "cell" in row else row["acquisition"]),
                    anchors=row.get("anchors"), top=tuple(decision.plain(row[f]) for f in design.top),
                    stratum=tuple(decision.plain(row[f]) for f in design.top_strata), pairs=tuple(sets[name]["eligible_pairs"] for name in pair_sets),
                    possible=sets["all"]["eligible_pairs"], errors=tuple(sets[name]["mean_pair_error_cm"] for name in pair_sets),
                    failed=failure is not None, aborted=failure is not None and failure.startswith(ABORTS), elapsed=row["elapsed_ms"])


def bern_official(path, mode):
    """The units of one mode in a released scorer's scenario scores in the order the scorer met them, each as (key,
    anchors, agents, released pairs, penalized pairs, the error as the scorer printed it)."""
    result = []
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] == mode and int(row["q"]) == 2:
                result.append(((int(row["scenario_id"]), int(row["iteration"]), int(row["samples"])), int(row["anchors"]), int(row["agents"]),
                               int(row["eligible_pairs"]), int(row["penalized_pairs"]), row["mean_pair_error"]))
    return result


@functools.lru_cache(maxsize=1)
def release_cells():
    """The cells of a TWR-CLOUD-Bern scan's full release corpus in the release's numbering, each as (first scenario
    number, agent groups, agent count, anchor count). Every anchor count of an agent count repeats the same agent groups
    in the same order, the release's groups of the scan's 20 agent positions thinned by rank to at most 5000
    (research/benchmarks/bern/training/release_rule.py, whose Design.cells numbers them)."""
    training = common.ROOT / "research/benchmarks/bern/training"
    if str(training) not in sys.path:
        sys.path.insert(0, str(training))
    import release_rule as rule
    groups = {count: [None] * min(math.comb(rule.POSITIONS_PER_SCAN - rule.SUPERSET_SIZE, count), rule.AGENT_GROUP_LIMIT)
              for count in rule.AGENT_COUNTS}
    design = rule.Design(None, {count: () for count in rule.ANCHOR_COUNTS}, groups)
    if design.scenario_count() != rule.EVALUATION_SCENARIOS:
        raise Refused(f"the release numbering holds {design.scenario_count()} scenarios, not the corpus's {rule.EVALUATION_SCENARIOS}")
    return tuple((first, count, agents, anchors) for agents, anchors, first, count in design.cells())


@functools.lru_cache(maxsize=1)
def release_firsts():
    return [cell[0] for cell in release_cells()]


def agent_group(scenario, anchors, agents):
    """The rank of a scenario's agent group among the groups of its agent count, which recur under every anchor count,
    refused unless the release's numbering places the scenario in its cell."""
    place = bisect.bisect_right(release_firsts(), scenario) - 1
    first, count, cell_agents, cell_anchors = release_cells()[place] if place >= 0 else (None, 0, None, None)
    if first is None or scenario >= first + count or (cell_agents, cell_anchors) != (agents, anchors):
        raise Refused(f"scenario {scenario} with {anchors} anchors and {agents} agents is not numbered by the release's rule")
    return scenario - first


def bern_entries(reader, folder, mode, design, pair_sets, aborted):
    """The entries of one run of run_bern.py --score in the order its released scorer met the units. The errors of every
    rule come from joined/scenario_scores.csv.gz, which must hold the released scores of official/ unit for unit. A
    unit's agent group is its rank among the groups of its agent count, which the release repeats under every anchor
    count (agent_group)."""
    official = bern_official(reader.file(folder / "official/scenario_scores.csv.gz"), mode)
    joined = scoring.read_scenario_units(reader.file(folder / "joined/scenario_scores.csv.gz"))
    where = relative(folder, reader.root)
    if len(official) != len(joined) or len({unit[0] for unit in official}) != len(official):
        raise Refused(f"the joined and the released scores of {where} hold different units")
    for key, anchors, agents, eligible, penalized, printed in official:
        unit = joined.get(key)
        if unit is None or (unit["anchors"], unit["agents"]) != (anchors, agents) or unit[RELEASED][0] != eligible:
            raise Refused(f"unit {key} of {where} differs between its joined and its released scores")
        released = unit[RELEASED][1]
        if ("nan" if released is None else f"{released:.12g}") != printed:
            raise Refused(f"the joined scores of {where} do not reproduce its released score of unit {key}")
        fields = dict(scenario=key[0], iteration=key[1], samples=key[2], anchors=anchors, agents=agents,
                      group=agent_group(key[0], anchors, agents))
        yield Entry(key=key, track=key[2], cell=(anchors, agents), anchors=anchors, top=tuple(fields[f] for f in design.top),
                    stratum=tuple(fields[f] for f in design.top_strata), pairs=tuple(unit[name][0] for name in pair_sets),
                    possible=agents * (agents - 1) // 2, errors=tuple(unit[name][1] for name in pair_sets), failed=penalized > 0,
                    aborted=key in aborted, elapsed=None)


class Units:
    """The units of one dataset, half and mode, the same for every method: their keys in the order their values enter the
    released arithmetic, sample tracks, cells, anchor counts, clusters, the pair counts of every pair set and of every
    agent pair, and every method's errors, failed and aborted units and solve times."""

    def __init__(self, dataset, mode, pair_sets, method, entries, sort=True):
        entries = list(entries)
        if sort:
            entries.sort(key=lambda entry: entry.key)
        self.dataset, self.mode, self.pair_sets = dataset, mode, tuple(pair_sets)
        self.keys = [entry.key for entry in entries]
        self.index = {key: i for i, key in enumerate(self.keys)}
        if len(self.index) != len(self.keys):
            raise Refused(f"{method} positioned a unit of {dataset} {mode} twice")
        if not self.keys:
            raise Refused(f"{method} positioned no unit of {dataset} {mode}")
        self.signature = [(e.track, e.cell, e.anchors, e.top, e.stratum, e.pairs, e.possible) for e in entries]
        self.track = np.array([e.track for e in entries], dtype=np.int64)
        self.anchors = np.array([-1 if e.anchors is None else e.anchors for e in entries], dtype=np.int64)
        self.cell = [e.cell for e in entries]
        self.top = [e.top for e in entries]
        self.top_stratum = [e.stratum for e in entries]
        self.pairs = np.array([e.pairs for e in entries], dtype=np.int64).T.reshape(len(self.pair_sets), len(entries))
        self.possible = np.array([e.possible for e in entries], dtype=np.int64)
        self.errors, self.failed, self.aborted, self.elapsed = {}, {}, {}, {}
        self.add(method, entries)

    def add(self, method, entries):
        """One method's entries, refused unless they cover the units exactly, each in its cell and cluster with its
        anchor and pair counts, with an error under a pair set exactly when the unit holds a pair of it."""
        if method in self.errors:
            raise Refused(f"{method} appears twice in {self.dataset} {self.mode}")
        n = len(self.keys)
        errors = np.full((len(self.pair_sets), n), np.nan)
        failed, aborted, elapsed = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.full(n, np.nan)
        seen = np.zeros(n, dtype=bool)
        for entry in entries:
            i = self.index.get(entry.key)
            if i is None:
                raise Refused(f"{method} positioned unit {entry.key} of {self.dataset} {self.mode}, which SOLID did not")
            if seen[i]:
                raise Refused(f"{method} positioned unit {entry.key} of {self.dataset} {self.mode} twice")
            seen[i] = True
            if (entry.track, entry.cell, entry.anchors, entry.top, entry.stratum, entry.pairs, entry.possible) != self.signature[i]:
                raise Refused(f"unit {entry.key} of {self.dataset} {self.mode} has another cell, cluster or pair count for {method} than for SOLID")
            for p, (pairs, error) in enumerate(zip(entry.pairs, entry.errors)):
                if not 0 <= pairs <= entry.possible:
                    raise Refused(f"unit {entry.key} of {self.dataset} {self.mode} counts {pairs} pairs under {self.pair_sets[p]} of its "
                                  f"{entry.possible} agent pairs")
                if (error is None) != (pairs == 0) or (error is not None and not np.isfinite(error)):
                    raise Refused(f"unit {entry.key} of {self.dataset} {self.mode} has an error of {method} under {self.pair_sets[p]} "
                                  f"that disagrees with its {pairs} pairs")
                if error is not None:
                    errors[p, i] = error
            failed[i], aborted[i] = entry.failed, entry.aborted
            if entry.elapsed is not None:
                elapsed[i] = entry.elapsed
        if not seen.all():
            raise Refused(f"{method} positioned {int(seen.sum())} of the {n} units SOLID positioned in {self.dataset} {self.mode}")
        self.errors[method], self.failed[method], self.aborted[method], self.elapsed[method] = errors, failed, aborted, elapsed


def heldout_half(reader, root, template, split, fields):
    """The fields of every unit of a built held-out half by identity, the SHA-256 of its population record and the notes
    of its data defects (heldout_notes), refused when its reference changed after its build. The half lies in the data
    folder (located)."""
    folder = located(root, template.format(split=split))
    population = reader.json(folder / "population.json")
    reference_path = folder / "reference.json"
    reference = reader.json(reference_path)
    if reader.files[relative(reference_path, root)] != (population.get("files") or {}).get("reference"):
        raise Refused(f"{relative(reference_path, root)} changed after its build")
    return ({entry["id"]: {name: entry[name] for name in fields} for entry in reference["units"]}, reader.files[relative(folder / "population.json", root)],
            heldout_notes(population, split))


def heldout_notes(population, split):
    """The data defects of a built held-out half that its tables note for every method alike, the declared outcomes of
    the held-out declaration's rules.outcomes. These are the AAU flights left out of the half, and of the training half
    with its fits and selections, beyond the relocation guard, and the pairs the guard could not compare for want of an
    admissible report. HYMN has none."""
    def flights(entries, half):
        return [f"{name} is left out of the {half} half for every method, "
                + (f"its static ranges moved {entry['change_cm']} cm from seq1" if entry.get("change_cm") is not None
                   else "no pair of it could be compared with seq1") + f", {entry['units']} units" for name, entry in sorted(entries.items())]
    notes = flights(population.get("excluded_flights") or {}, split)
    if split != "training":
        notes += flights(population.get("training_excluded_flights") or {}, "training")
    return notes + [f"{name} holds pairs without an admissible report, which the relocation guard left out, {', '.join(pairs)}"
                    for name, pairs in sorted((population.get("unguarded_pairs") or {}).items())]


def mode_units(reader, root, dataset, split, spec, mode, runs, scores, design, metadata):
    """The units of one mode with every method's values, SOLID's order the key order, or on TWR-CLOUD-Bern the order its
    released scorer met them."""
    pair_sets = spec["pair_sets"]
    if spec["layout"] == "bern":
        units = None
        for method, run in runs.items():
            entries = bern_entries(reader, scores[method], mode, design, pair_sets, run.aborted)
            if units is None:
                units = Units(dataset, mode, pair_sets, method, entries, sort=False)
            else:
                units.add(method, entries)
        declared = spec["units_per_track"][mode]
        counts = {int(track): int((units.track == track).sum()) for track in sorted(set(units.track.tolist()))}
        if any(count != declared for count in counts.values()):
            raise Refused(f"the {dataset} {mode} units per track {counts} are not the corpus's {declared}")
        return units
    by_path = defaultdict(list)
    for method, path in scores.items():
        by_path[path].append(method)
    units = None
    for path, methods in by_path.items():
        rows = reader.rows(path)
        grouped = defaultdict(list)
        # a baseline row names its method, and SOLID's rows, or an extra row's, are the only method of their file
        for row in rows:
            grouped[row["method"] if "method" in row else methods[0]].append(row)
        if sorted(grouped) != sorted(methods):
            raise Refused(f"{relative(path, root)} holds rows of {sorted(grouped)}, not of {sorted(methods)}")
        for method in methods:
            entries = row_entries(grouped[method], design, pair_sets, metadata)
            if units is None:
                if method != "solid":
                    raise Refused("SOLID's rows come first")
                units = Units(dataset, mode, pair_sets, method, entries)
            else:
                units.add(method, entries)
        del rows, grouped
    return units


# ---------------------------------------------------------------------------------------------------------------------
# Tables and draws


def report_groups(mode, anchors, minimum):
    """The groups of one mode as (label, mask) in reporting order: every cell in the anchor free mode, every cell and
    then every anchor count below `minimum` alone in the anchor aided mode, whose headline stays every cell, and in the
    anchor only mode the headline over the units with `minimum` or more anchors and then every smaller anchor count
    alone, the groups of scoring.report_groups there. The anchor aided rows of one and two anchors show a method whose
    anchored decoding fails with so few anchors. Empty groups are left out."""
    every = ("every cell", np.ones(len(anchors), dtype=bool))
    if mode == "ANC" and (anchors < 1).any():
        raise Refused("an anchor only unit without an anchor count")
    smaller = [(f"anchors-{k}", anchors == k) for k in sorted({int(a) for a in anchors if 1 <= a < minimum})]
    if mode == "ANC":
        groups = [(f"headline, {minimum} or more anchors", anchors >= minimum), *smaller]
    elif mode == "AAC":
        groups = [every, *smaller]
    else:
        groups = [every]
    return [(label, mask) for label, mask in groups if mask.any()]


class Table(decision.TrackPlan):
    """The scored units of one table for the draws of decision.py: their cells and clusters and every method's errors in
    sorted order, so that the weights of a draw serve every method of the table. TrackPlan.weights and
    TrackPlan.percentiles do the work, as they do for the two candidates of a round."""

    def __init__(self, units, indices, pair_set, methods, top_index, owned):
        self.track = None
        self.units = np.arange(len(indices))
        cells = [units.cell[i] for i in indices]
        self.cells = decision.ordered(set(cells))
        code = {cell: c for c, cell in enumerate(self.cells)}
        self.cell_code = np.array([code[cell] for cell in cells], dtype=np.int64)
        self.sizes = np.bincount(self.cell_code, minlength=len(self.cells)).astype(np.int64)
        self.tops = np.array([top_index[units.top[i]] for i in indices], dtype=np.int64)
        self.owned = owned
        if owned:
            tops = defaultdict(set)
            for c, t in zip(self.cell_code, self.tops):
                tops[c].add(int(t))
            if any(len(t) != 1 for t in tops.values()):
                raise Refused(f"a cell of {units.dataset} {units.mode} spans several clusters")
            self.cells_per_top = np.zeros(max(top_index.values()) + 1, dtype=np.int64)
            for t in tops.values():
                self.cells_per_top[next(iter(t))] += 1
        self.values = {}
        for method in methods:
            errors = units.errors[method][pair_set][indices]
            order = np.argsort(errors, kind="stable")
            self.values[method] = (order, errors[order])


def bootstrap(tables, name, design, plan, replicates, seed, levels):
    """Per table and method the draws of every level, (replicates, levels). Every table shares each draw of the
    dataset's clusters, and a draw is cut into parts of at most CHUNK unit weights. A draw that holds none of a table's
    units has no percentile there and gives NaN."""
    stores = [{method: [] for method in table.values} for table in tables]
    for counts in decision._draws(name, plan, replicates, seed, design):
        for table, store in zip(tables, stores):
            rows = max(1, CHUNK // max(1, len(table.tops)))
            for start in range(0, len(counts), rows):
                part = counts[start:start + rows]
                weights = table.weights(part[:, table.tops], part)
                held = weights[2] > 0
                if not held.all():
                    weights = tuple(np.asarray(w)[held] for w in weights)
                for method, parts in store.items():
                    values = np.full((len(part), len(levels)), np.nan)
                    if held.any():
                        values[held] = table.percentiles(method, weights, levels)
                    parts.append(values)
    return [{method: np.concatenate(parts) for method, parts in store.items()} for store in stores]


def point(cells, values, levels, percentile):
    """The balanced percentiles of one method's errors by the released scorer's arithmetic, the cells in the order of
    their first unit, and in exact arithmetic."""
    grouped = {}
    for cell, value in zip(cells, values):
        grouped.setdefault(cell, []).append(float(value))
    return [(float(percentile(grouped, float(level))), float(common.balanced_percentile_exact(grouped, level))) for _, level in levels]


def estimate(value, exact, draws, interval):
    """A value with the interval and the standard deviation of its draws. When some draws hold none of the table's
    units the value has no interval, and the count of those draws is kept."""
    empty = int(np.isnan(draws).sum())
    if empty:
        return dict(value_cm=value, exact_cm=exact, interval_cm=None, se_cm=None, empty_draws=empty)
    low, high = np.quantile(draws, [float((1 - interval) / 2), float((1 + interval) / 2)], method="inverted_cdf")
    return dict(value_cm=value, exact_cm=exact, interval_cm=[float(low), float(high)], se_cm=float(np.std(draws, ddof=1)))


def comparisons(roster, extras):
    """The differences of a mode, SOLID minus every baseline and every extra row minus SOLID, as (minuend, subtrahend)."""
    return [("solid", method) for method in roster if method != "solid"] + [(name, "solid") for name in extras]


def analyse(modes, rosters, extras, pair_sets, design_name, design, replicates, seed, levels, interval, minimum, tracks, percentile, runs=None):
    """The tables of one dataset and half from the units of its modes.

    modes maps a mode to its Units, rosters a mode to its declared methods, SOLID first, and extras a mode to the names of
    its extra SOLID rows. Every table of every mode shares the draws of the dataset's clusters. runs maps a mode and a
    method to its Run, for the threads and processes of the solve times. Returns {mode: section}."""
    levels = decision.levels_of(levels)
    plan = decision._dataset_plan(list(modes.values()), design)
    layout, tables = {}, []
    for mode, units in modes.items():
        methods = [*rosters[mode], *extras.get(mode, ())]
        groups = []
        for label, mask in report_groups(mode, units.anchors, minimum):
            entries = []
            for track in tracks:
                members = mask & (units.track == track)
                if not members.any():
                    continue
                sets = []
                for p, name in enumerate(pair_sets):
                    scored = np.flatnonzero(members & (units.pairs[p] > 0))
                    table = Table(units, scored, p, methods, plan[0], design.cells_in_top) if len(scored) else None
                    tables += [table] if table is not None else []
                    sets.append((name, scored, table))
                entries.append((track, members, sets))
            groups.append((label, entries))
        layout[mode] = (units, methods, groups)
    drawn = dict(zip(map(id, tables), bootstrap(tables, design_name, design, plan, replicates, seed, levels)))
    return {mode: section(units, methods, groups, comparisons(rosters[mode], extras.get(mode, ())), drawn, levels, Fraction(interval),
                          percentile, (runs or {}).get(mode, {}))
            for mode, (units, methods, groups) in layout.items()}


def section(units, methods, groups, pairs_of, drawn, levels, interval, percentile, runs):
    """One mode's tables: per group and sample track every method's failed and aborted units and solve times, and per pair
    set the shares it scores, every method's percentiles and every difference."""
    result = dict(methods=methods, differences=[f"{a} minus {b}" for a, b in pairs_of], groups={})
    for label, entries in groups:
        tracks = {}
        for track, members, sets in entries:
            solves = {method: solve_entry(units, method, members, runs.get(method)) for method in methods}
            tables = {name: table_entry(units, p, members, scored, table, methods, pairs_of, drawn, levels, interval, percentile)
                      for p, (name, scored, table) in enumerate(sets)}
            tracks[str(track)] = dict(units=int(members.sum()), methods=solves, pair_sets=tables)
        result["groups"][label] = dict(tracks=tracks)
    return result


def solve_entry(units, method, members, run):
    """A method's failed and aborted units and its median solve time over the units of one group and track, with the
    threads and processes of its run. A Bern run keeps only the median over all its solves."""
    if run is not None and run.runtime_ms is not None:
        runtime = dict(median_ms=run.runtime_ms, scope="every solve of the run")
    else:
        times = units.elapsed[method][members]
        times = times[np.isfinite(times)]
        runtime = dict(median_ms=float(np.median(times)) if len(times) else None)
    runtime.update(threads_per_process=run.threads if run else None, processes=run.processes if run else None)
    return dict(failed_units=int(units.failed[method][members].sum()), aborted_units=int(units.aborted[method][members].sum()), runtime=runtime)


def table_entry(units, p, members, scored, table, methods, pairs_of, drawn, levels, interval, percentile):
    """One table: the share of pairs and units its pair set scores, and every method's percentiles and every difference,
    each with its interval."""
    total = int(members.sum())
    entry = dict(units=total, scored_units=int(len(scored)), unit_share=len(scored) / total, pairs=int(units.pairs[p][members].sum()),
                 possible_pairs=int(units.possible[members].sum()), cells=len({units.cell[i] for i in scored}))
    entry["pair_share"] = entry["pairs"] / entry["possible_pairs"] if entry["possible_pairs"] else None
    names = [name for name, _ in levels]
    if table is None:
        entry.update(methods={method: dict.fromkeys(names) for method in methods},
                     differences={f"{a} minus {b}": dict.fromkeys(names) for a, b in pairs_of})
        return entry
    cells = [units.cell[i] for i in scored]
    values = {method: point(cells, units.errors[method][p][scored], levels, percentile) for method in methods}
    draws = drawn[id(table)]
    entry["methods"] = {method: {name: estimate(*values[method][k], draws[method][:, k], interval) for k, name in enumerate(names)}
                        for method in methods}
    entry["differences"] = {f"{a} minus {b}": {name: estimate(values[a][k][0] - values[b][k][0], values[a][k][1] - values[b][k][1],
                                                               draws[a][:, k] - draws[b][:, k], interval) for k, name in enumerate(names)}
                            for a, b in pairs_of}
    return entry


# ---------------------------------------------------------------------------------------------------------------------
# Guards
# Every guard of this section is a no-op in a reader's reproduction (READER), which builds under the declaration of this
# module (reader_declaration) from the reader's own runs.


def reader_declaration(analysis=None):
    """The declaration of a reader's reproduction, its content, bytes and SHA-256 as committed_declaration gives them:
    the analysis this module declares, or the one given, as declare would write it."""
    data = encoded(declaration() if analysis is None else analysis)
    return json.loads(data), data, hashlib.sha256(data).hexdigest()


def committed_declaration(root):
    """The declaration, its bytes and SHA-256, refused while it is missing, not a declaration or not committed as it is."""
    path = Path(root) / DECLARATION
    if not path.is_file():
        raise Refused(f"{DECLARATION} is missing. The analysis is declared and committed before any table, with declare")
    data = path.read_bytes()
    try:
        declared = json.loads(data)
    except ValueError as error:
        raise Refused(f"{DECLARATION} is not a JSON record") from error
    if not isinstance(declared, dict) or declared.get("schema") != SCHEMA:
        raise Refused(f"{DECLARATION} is not a declaration of schema {SCHEMA}")
    if not committed(path, root):
        raise Refused(f"{DECLARATION} differs from its committed version. The analysis is committed before any table")
    return declared, data, hashlib.sha256(data).hexdigest()


def machine(host):
    """The part of a run's host that names the machine, its host record without the platform, whose kernel release a
    reboot of the same machine changes. The platform is recorded and never compared."""
    return {key: value for key, value in host.items() if key != "platform"} if isinstance(host, dict) else host


def one_binary(runs):
    """The one source identity and machine of a dataset's runs with the platforms they record, refused unless every run
    names them and they agree, every SOLID binary matches its sources and the runs of one executable that record its
    SHA-256 name one."""
    if READER:
        return None
    unnamed = sorted({run.record for run in runs if not run.identity or not run.host})
    if unnamed:
        raise Refused(f"the runs {unnamed} name no source identity or no host, so one binary per dataset cannot be shown")
    identities = sorted({run.identity for run in runs})
    if len(identities) != 1:
        raise Refused(f"the runs of one dataset come from {len(identities)} source identities {identities}, one binary per dataset")
    hosts = sorted({json.dumps(machine(run.host), sort_keys=True) for run in runs})
    if len(hosts) != 1:
        raise Refused(f"the runs of one dataset come from {len(hosts)} hosts {hosts}, one binary per dataset")
    stale = sorted({run.record for run in runs if run.solid and not run.matches_sources})
    if stale:
        raise Refused(f"the SOLID runs {stale} ran a binary that differs from the sources of their checkout")
    binaries = defaultdict(set)
    for run in runs:
        if run.binary_sha256:
            binaries[run.executable].add(run.binary_sha256)
    split = sorted(name for name, found in binaries.items() if len(found) > 1)
    if split:
        raise Refused(f"the runs of {split} name more than one binary of that executable")
    platforms = sorted({run.host.get("platform") for run in runs if isinstance(run.host, dict) and run.host.get("platform")})
    return dict(source_identity=identities[0], host=machine(runs[0].host), platforms=platforms)


def method_guard(reader, root, runs):
    """The committed method record, refused unless every SOLID run records its SHA-256."""
    if READER:
        return None
    path = Path(root) / METHOD_RECORD
    if not path.is_file():
        raise Refused(f"{METHOD_RECORD} is missing, so no SOLID run can be shown to run the confirmed method")
    if not committed(path, root):
        raise Refused(f"{METHOD_RECORD} differs from its committed version")
    reader.file(path)
    digest = reader.files[relative(path, root)]
    other = sorted({run.record for run in runs if run.solid and run.method_record != digest})
    if other:
        raise Refused(f"the SOLID runs {other} record another method record than {METHOD_RECORD} or none")
    return dict(path=METHOD_RECORD, sha256=digest)


def budget_marks(reader, root, dataset, runs, minimum):
    """The runs of a dataset, {mode: {method: Run}}, with the marks that a budget check without a converged step leaves
    on SOLID, read from the budget entry of the committed method record (confirm_round.budget_outcome). Every comparison
    of the dataset that still moved at the last doubling marks SOLID in the anchor only tables it spans, the headline
    over the units with `minimum` or more anchors for a comparison over them and every anchor only table for one over
    all units, with a note that names its track, percentile, difference and margin. Such a mark names the groups it
    marks, and a mark without groups, a declared outcome of a held-out half, marks every table of its mode. The runs
    come back as they are when the check converged."""
    if READER:
        return runs
    record = reader.json(Path(root) / METHOD_RECORD)
    found = defaultdict(list)
    for entry in (record.get("budget") or {}).get("unconverged") or []:
        if entry.get("dataset") != dataset:
            continue
        headline = entry.get("rows") == "headline"
        units = f"the units with {minimum} or more anchors" if headline else "every unit"
        found[str(entry["mode"]).upper()].append(dict(
            groups=[f"headline, {minimum} or more anchors"] if headline else None,
            text=(f"The budget check did not converge. At its last doubling SOLID's {entry['level']} of the {entry['track']} sample track "
                  f"over {units} still moved by {entry['difference_cm']:+.1f} cm, beyond its margin of {entry['margin_cm']:.1f} cm")))
    return {mode: {name: replace(run, marks=run.marks + tuple(found[mode])) if name == "solid" and found.get(mode) else run
                   for name, run in by_method.items()} for mode, by_method in runs.items()}


def portable(argument, root):
    """A command argument with a path of the repository written relative to it."""
    prefix = str(Path(root)) + "/"
    return argument[len(prefix):] if isinstance(argument, str) and argument.startswith(prefix) else argument


def selecting(runs):
    """The methods whose options the mode of one of the runs selects, in the order of their first run."""
    import baseline_selection
    return list(dict.fromkeys(run.method for run in runs if baseline_selection.selects(run.method, run.mode)))


def bern_selection(root, runs):
    """The records of the Bern baseline runs, {method: path}, checked against run_bern.py without writing anything. A
    run of a method whose options its mode selects must have taken the record run_bern.py accepts now
    (baseline_selection.bern_record), which the check-selection of baselines_bern_training.py reads as well, and its
    command.json must name that record's SHA-256, its selected candidate and the SHA-256 of the declared base
    configurations. Its method-config.json must hold the declared settings of that candidate. Every run's command must
    be the one run_bern.py builds, from the selected candidate or from the retained configuration of baselines.json."""
    import run_bern
    paths = {}
    for run in runs:
        folder = Path(root) / Path(run.record).parent
        command = json.loads((folder / "command.json").read_text())
        scenarios, flags, chosen, settings = run_bern.planned(run.method, run.mode, folder)
        expected = ["benchmark", "--scenarios", str(scenarios), "--mode", run.mode.lower(), "--sample-counts", "1,10", "--iterations", "1,2", *flags]
        if [portable(portable(a, root), common.ROOT) for a in (command.get("command") or [])[1:]] != [portable(portable(a, root), common.ROOT)
                                                                                                    for a in expected]:
            raise Refused(f"the Bern run of {run.method} {run.mode} did not run the command run_bern.py builds from "
                          + ("its selected candidate" if chosen else "the retained configuration of baselines.json"))
        if chosen is None:
            continue
        taken, path = command.get("selection") or {}, run_bern.selection_path(run.method)
        for key in ("sha256", "selected", "base_configurations_sha256"):
            if taken.get(key) != chosen[key]:
                raise Refused(f"the Bern run of {run.method} {run.mode} names another {key} than {path.name} and the declared base "
                              f"configurations, {taken.get(key)} and not {chosen[key]}")
        written = folder / "method-config.json"
        if not written.is_file() or json.loads(written.read_text()) != json.loads(json.dumps(settings)):
            raise Refused(f"{relative(written, root)} does not hold the declared settings of the candidate {chosen['selected']} of {path.name}")
        paths[run.method] = path
    return paths


def tiers_selection(root, runs):
    """layout_baselines.py tiers check-selection of every method whose options a run's mode selects, {method: path}."""
    import layout_baselines
    paths = {}
    for method in selecting(runs):
        layout_baselines.tiers_selection(method)
        paths[method] = layout_baselines.record_path(method)
    return paths


def bregar_selection(root, runs):
    """bregar_layout.py check-selection of every method whose options a run's mode selects, with the inputs the
    selection read unchanged, {method: path}."""
    import bregar_layout
    paths = {}
    for method in selecting(runs):
        record = bregar_layout.selection(method)
        for name, expected in record["inputs_sha256"].items():
            if common.sha256(bregar_layout.OUTPUT / name) != expected:
                raise Refused(f"the Bregar input {name} changed after the selection of {method}")
        paths[method] = bregar_layout.selection_path(method)
    return paths


def idlab_selection(root, runs):
    """idlab_layout_baselines.py check-selection of every method whose options a run's mode selects, {method: path}."""
    import idlab_layout_baselines
    paths = {}
    for method in selecting(runs):
        idlab_layout_baselines.headline_selection(method)
        paths[method] = idlab_layout_baselines.selection_path(method)
    return paths


def heldout_selection(dataset, root, runs):
    """heldout_baselines.py check-selection of every method whose options a run's mode selects, {method: path}."""
    import heldout_baselines
    paths = {}
    for method in selecting(runs):
        heldout_baselines.selection(dataset, method)
        paths[method] = heldout_baselines.selection_path(dataset, method)
    return paths


# The selection check of every dataset, each given the root and the baseline runs and returning the record of every
# method whose options a run's mode selects, {method: path}: the check-selection command of the dataset's runner, and
# on TWR-CLOUD-Bern the check of every run's command, selection and method settings against run_bern.py.
SELECTION_CHECKS = dict(bern=bern_selection, tiers=tiers_selection, bregar=bregar_selection, idlab=idlab_selection,
                        hymn=lambda root, runs: heldout_selection("hymn", root, runs), aau=lambda root, runs: heldout_selection("aau", root, runs))


def run_entry(root, run):
    """What a baseline run's record keeps of the run of its method, the entry of the method in an IDLab record and the
    record itself otherwise."""
    record = json.loads((Path(root) / run.record).read_text())
    return (record.get("runs") or {}).get(run.method, record) if isinstance(record.get("runs"), dict) else record


def selection_guard(dataset, runs, check, root=None):
    """The selection records of a dataset's baselines, {method: dict(path, sha256)}, or None without a run of a method
    whose options its mode selects. Refused unless the check, SELECTION_CHECKS of the dataset, accepts the record of
    every such method and every such run names the record's SHA-256 and the declared base configurations and shared
    inputs the record was selected with. On TWR-CLOUD-Bern the check compares every run's command.json and
    method-config.json itself. A check that fails is a refusal, whether it refuses a record, calls another signature, finds
    no method freeze or ledger entry of the held-out benchmarks (a RuntimeError) or cannot run an executable. The check
    and the guard read and never write."""
    if READER:
        return None
    import baseline_selection
    root = Path(root or common.ROOT)
    baselines = [run for by_method in runs.values() for run in by_method.values() if not run.solid]
    selected = [run for run in baselines if baseline_selection.selects(run.method, run.mode)]
    try:
        paths = {method: Path(path) for method, path in check(root, baselines).items()}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        raise Refused(f"the {dataset} selection check fails: {error}") from error
    missing = sorted({run.method for run in selected} - set(paths))
    if missing:
        raise Refused(f"the {dataset} selection check returned no record of {', '.join(missing)}")
    records = {method: (common.sha256(path), json.loads(path.read_text())) for method, path in paths.items()}
    if dataset != "bern":
        for run in selected:
            digest, record = records[run.method]
            if run.selection != digest:
                raise Refused(f"the {run.method} run {run.record} names another selection record than the checked {paths[run.method].name}")
            entry = run_entry(root, run)
            for key in ("base_configurations_sha256", "shared_inputs"):
                if entry.get(key) != record.get(key):
                    raise Refused(f"the {run.method} run {run.record} ran with another {key} than {paths[run.method].name} was selected with")
    if not records:
        return None
    return {method: dict(path=str(path.relative_to(common.ROOT)) if path.is_relative_to(common.ROOT) else str(path), sha256=records[method][0])
            for method, path in sorted(paths.items())}


def recorded_freeze(root):
    """The content and the bytes of the method freeze record of a repository, refused while it is missing, not a method
    freeze record or not committed as it is."""
    path = Path(root) / FREEZE
    if not path.is_file():
        raise Refused(f"{FREEZE} is missing. The tables of an evaluation half are built only under the method freeze")
    data = path.read_bytes()
    try:
        content = json.loads(data)
    except ValueError as error:
        raise Refused(f"{FREEZE} is not a JSON record") from error
    if not isinstance(content, dict) or content.get("schema") != FREEZE_SCHEMA:
        raise Refused(f"{FREEZE} is not a method freeze record of schema {FREEZE_SCHEMA}")
    if not committed(path, root):
        raise Refused(f"{FREEZE} differs from its committed version. The method freeze is a commit")
    return content, data


def method_freeze(root):
    """The content and the bytes of the method freeze record, refused unless it holds the method, the code, the
    executables, the environment and its pins as they are (heldout.verified), which the repository of the freeze alone
    can show."""
    if READER:
        return None
    import heldout
    if Path(root).resolve() != heldout.ROOT.resolve():
        raise Refused(f"the method freeze of {heldout.ROOT} cannot verify the repository {root}")
    try:
        heldout.verified()
    except heldout.FreezeMissing as error:
        raise Refused(f"The method freeze does not hold as it is. {error}") from error
    return recorded_freeze(root)


def evaluation_guard(root, data, digest, runs, freeze):
    """Refused unless the method freeze, (content, bytes) of its verified record, pins this declaration, and unless every
    run was made by the frozen executables, a run of an executable the freeze pins with the SHA-256 it pins and every
    run with the source identity of the frozen executables, so that no run made with the code before the freeze enters
    a table. Every run, SOLID's, a baseline's and a TWR-CLOUD-Bern run alike, must name the commit it ran at, and that
    commit must hold this declaration and the method freeze as they are, so that a run made with the frozen executables
    before the freeze was committed enters no table either."""
    if READER:
        return
    content, frozen = freeze
    pinned = (content.get(FREEZE_KEY) or {}).get(DECLARATION)
    if pinned != digest:
        raise Refused(f"the method freeze {FREEZE} pins {'no declaration' if pinned is None else 'another declaration'} under {FREEZE_KEY}. The "
                      "analysis is pinned by the method freeze before every run the tables read")
    binaries = {name: entry for name, entry in (content.get("binaries") or {}).items() if isinstance(entry, dict)}
    identities = {entry.get("identity") for entry in binaries.values()}
    if len(identities) != 1 or None in identities:
        raise Refused(f"the method freeze {FREEZE} names no one source identity of its executables")
    identity = identities.pop()
    for run in runs:
        if run.identity != identity:
            raise Refused(f"{run.record} ran an executable of the sources {str(run.identity)[:12]}, not those of the method freeze "
                          f"{identity[:12]}. Every run the tables of an evaluation half read is made by the frozen executables")
        expected = (binaries.get(run.executable) or {}).get("sha256")
        if expected is not None and run.binary_sha256 != expected:
            raise Refused(f"{run.record} ran another build of {run.executable} than the one the method freeze pins")
        if not run.commit:
            raise Refused(f"{run.record} records no commit, so the declaration cannot be shown to precede it")
        if shown(run.commit, Path(root) / DECLARATION, root) != data:
            raise Refused(f"{run.record} ran at commit {run.commit}, which lacks the declaration or holds another version of it. The "
                          "analysis is committed before every run the tables read")
        if shown(run.commit, Path(root) / FREEZE, root) != frozen:
            raise Refused(f"{run.record} ran at commit {run.commit}, which lacks the method freeze or holds another version of it. "
                          "Every run the tables of an evaluation half read follows the method freeze")


def profile_guard(reader, root, dataset, runs):
    """Refused unless every run of SOLID's row on an evaluation half, runs {mode: {method: Run}}, solved the tracked
    profile of its dataset and mode, which the evaluation stage composes from the committed method record and commits
    before the runs, and which every runner checks against that composition as it reads it (profiles.resolve). The run
    names that profile and the SHA-256 of the profile it solved, which must be the SHA-256 of the profile as it is
    committed and as the commit of the run holds it. A run of any other profile file, a candidate's among them, is
    refused."""
    if READER:
        return
    for mode, by_method in runs.items():
        run = by_method.get("solid")
        if run is None:
            continue
        name = f"{PROFILES}/{dataset}-{PROFILE_KINDS[mode]}.json"
        path = Path(root) / name
        if run.profile != name:
            raise Refused(f"{run.record} solved the profile {run.profile}, not the tracked profile {name} of its dataset and mode")
        if not path.is_file() or not committed(path, root):
            raise Refused(f"{name} is missing or differs from its committed version. The evaluation stage commits the profiles before "
                          "SOLID's runs")
        digest = hashlib.sha256(reader.bytes(path)).hexdigest()
        if run.profile_sha256 != digest:
            raise Refused(f"{run.record} solved another profile than the committed {name}")
        held = shown(run.commit, path, root) if run.commit else None
        if held is None or hashlib.sha256(held).hexdigest() != digest:
            raise Refused(f"{run.record} ran at a commit that does not hold the committed {name}")


def tables_guard(root, digest):
    """Refused while a table of an evaluation half differs from its committed version, one written and not yet
    committed, edited or deleted, and when a table of an evaluation half, as it is or in any committed version, was
    written under another declaration. Every table of an evaluation half is committed as soon as it is written, before
    the builder writes the next, so the history holds every one."""
    if READER:
        return
    status = git(root, "status", "--porcelain", "--untracked-files=all", "--", TABLES)
    if status is None or status.returncode != 0:
        raise Refused("git cannot show that every table of an evaluation half is committed")
    pending = sorted({str(Path(line[3:]).parent) for line in status.stdout.decode().splitlines() if "/evaluation/" in line[3:]})
    if pending:
        raise Refused(f"{pending[0]} differs from its committed version. Every table of an evaluation half is committed as soon as it is "
                      "written, before the next")
    for path in evaluation_tables(root):
        recorded = (json.loads(path.read_text()).get("declaration") or {}).get("sha256")
        if recorded != digest:
            raise Refused(f"{relative(path, root)} was written under another declaration. The declaration is fixed once an evaluation "
                          "table exists")
    for commit, name, table in committed_tables(root):
        try:
            recorded = (json.loads(table).get("declaration") or {}).get("sha256")
        except (TypeError, ValueError):
            recorded = None
        if recorded != digest:
            raise Refused(f"{name} at commit {commit[:12]} was written under another declaration. The declaration is fixed once an "
                          "evaluation table exists")


def ledger_files(root, dataset, frozen):
    """The files that the committed ledger of the held-out benchmarks pins for a dataset, {path: SHA-256}, from its
    closing entry of the dataset under the method freeze of the record's bytes, and the records of held-out stage one
    that the closing entry of stage one pins under the same freeze, SOLID's tracked profile of the dataset among them.
    Refused while the ledger is missing or not committed, or does not close the dataset under this freeze."""
    if READER:
        return None
    path = Path(root) / LEDGER
    if not path.is_file() or not committed(path, root):
        raise Refused(f"{LEDGER} is missing or differs from its committed version. A table of {dataset} reads only what its ledger pins")
    record = json.loads(path.read_text())
    freeze_digest = hashlib.sha256(frozen).hexdigest()

    def closing(step):
        return [entry for entry in record.get("entries") or [] if entry.get("step") == step and entry.get("event") == "completed"
                and entry.get("method_freeze_sha256") == freeze_digest]
    if not closing(dataset):
        raise Refused(f"{LEDGER} does not close {dataset} under the method freeze, so its evaluation is not complete")
    stage_one = closing(STAGE_ONE)
    records = (stage_one[-1].get("records") or {}) if stage_one else {}
    return dict(records, **(closing(dataset)[-1].get("files") or {}))


def ledger_guard(reader, dataset, files):
    """Refused unless every file a table of a held-out dataset read, but the method record, is one the ledger pins with
    the SHA-256 it records."""
    if READER:
        return
    for name, digest in sorted(reader.files.items()):
        if name == METHOD_RECORD:
            continue
        if name not in files:
            raise Refused(f"a table of {dataset} reads {name}, which the ledger's closing entry of {dataset} does not pin")
        if files[name] != digest:
            raise Refused(f"{name} changed after the ledger closed {dataset}")


def ade_guard(reader, root, dataset, runs):
    """The committed fit record of ADE-BP's ranging model on a dataset's training half, refused unless every ADE-BP run
    names its SHA-256. None where ADE-BP takes no such fit or does not run."""
    if READER:
        return None
    ade = [run for by_method in runs.values() for method, run in by_method.items() if method == "ade-bp-2015" and not run.solid]
    if dataset not in ADE_FITS or not ade:
        return None
    path = Path(root) / ADE_FITS[dataset]
    if not path.is_file() or not committed(path, root):
        raise Refused(f"{ADE_FITS[dataset]} is missing or differs from its committed version, so ADE-BP's ranging model cannot be shown")
    reader.file(path)
    digest = reader.files[ADE_FITS[dataset]]
    for run in ade:
        record = reader.json(Path(root) / run.record)
        entry = record["runs"][run.method] if isinstance(record.get("runs"), dict) else record
        if entry.get("ade_fit_sha256") != digest:
            raise Refused(f"the ADE-BP run {run.record} of {run.mode} names another ranging model fit than {ADE_FITS[dataset]}")
    return dict(path=ADE_FITS[dataset], sha256=digest)


# ---------------------------------------------------------------------------------------------------------------------
# Build


def build(dataset, split="evaluation", root=None, percentile=None, checks=None, analysis=None, freeze=None, runs_root=None,
          tables_root=None):
    """The tables of one dataset and half under the committed declaration, written to
    research/artifacts/solid/tables/<dataset>/<split>. percentile is the released scorer's balanced percentile unless
    given, checks the selection checks, SELECTION_CHECKS unless given, analysis the declaration the committed one must
    hold, declaration() unless given, and freeze the verified method freeze of an evaluation half, method_freeze unless
    given. A reader's reproduction builds under analysis itself, declaration() unless given (reader_declaration), reads
    the runs below runs_root in place of out/ when it is given (located) and writes to tables_root/<dataset>/<split> when
    it is given. Returns the tables' record."""
    root = Path(root or common.ROOT)
    if READER:
        declared, data, digest = reader_declaration(analysis)
    else:
        declared, data, digest = committed_declaration(root)
    changed = check(root, analysis)
    if changed:
        raise Refused(f"{DECLARATION} differs from the declaration of research/solid/final_tables.py in {changed}. A table reads only the "
                      "analysis the frozen code declares")
    spec = declared["datasets"].get(dataset)
    if spec is None:
        raise Refused(f"the declaration names no dataset {dataset}")
    if split not in spec["splits"]:
        raise Refused(f"the declaration names no {split} half of {dataset}")
    described = {key: value for key, value in spec["design"].items() if key not in ("name", "reason")}
    if decision.DESIGNS.get(spec["design"]["name"]) is None or decision.DESIGNS[spec["design"]["name"]].describe() != described:
        raise Refused(f"the design {spec['design']['name']} of decision.py differs from the declared one")
    design = decision.Design(**{key: tuple(value) if isinstance(value, list) else value for key, value in described.items()})
    extras = [row for row in declared["extra_rows"] if row["dataset"] == dataset]
    reader = Reader(root)
    runs, scores, missing = {}, {}, {}
    for mode in spec["modes"]:
        found, paths, absent = mode_runs(reader, root, dataset, split, spec, mode, extras, runs_root)
        if absent:
            missing[mode] = absent
        if found is not None:
            runs[mode], scores[mode] = found, paths
    if not runs:
        raise Refused(f"no mode of the {split} half of {dataset} has its runs, missing {missing}")
    every = [run for by_method in runs.values() for run in by_method.values()]
    identity = one_binary(every)
    method = method_guard(reader, root, every)
    runs = budget_marks(reader, root, dataset, runs, declared["groups"]["minimum_anchors"])
    selection = selection_guard(dataset, runs, (checks or SELECTION_CHECKS)[dataset], root)
    ade = ade_guard(reader, root, dataset, runs)
    frozen = ledgered = None
    if split == "evaluation":
        frozen = (freeze or method_freeze)(root)
        evaluation_guard(root, data, digest, every, frozen)
        profile_guard(reader, root, dataset, runs)
        tables_guard(root, digest)
        if dataset in HELDOUT:
            ledgered = ledger_files(root, dataset, None if frozen is None else frozen[1])
    metadata, notes = None, []
    if "unit_fields" in spec:
        metadata, population, notes = heldout_half(reader, root, spec["runs"]["half"], split, spec["unit_fields"])
        other = sorted({run.record for run in every if run.population != population})
        if other:
            raise Refused(f"the runs {other} were not made on the built half they are scored with")
    percentile = percentile or common.load_scorer().balanced_percentile
    modes = {}
    for mode in runs:
        modes[mode] = mode_units(reader, root, dataset, split, spec, mode, runs[mode], scores[mode], design, metadata)
        if spec["layout"] == "bern":
            released_figures(reader, modes[mode], runs[mode], scores[mode], spec["pair_sets"], percentile)
    if ledgered is not None:
        ledger_guard(reader, dataset, ledgered)
    rosters = {mode: spec["modes"][mode] for mode in modes}
    extra = {mode: [row["name"] for row in extras if row["name"] in runs[mode]] for mode in modes}
    sections = analyse(modes, rosters, extra, spec["pair_sets"], spec["design"]["name"], design, spec["replicates"], spec["seed"],
                       declared["percentiles"], declared["interval"], declared["groups"]["minimum_anchors"], declared["tracks"], percentile, runs)
    labels = dict(declared["methods"]["labels"], **{row["name"]: row["label"] for row in extras})
    record = dict(schema=TABLES_SCHEMA, dataset=dataset, label=spec["label"], split=split, reported=spec["splits"][split]["reported"],
                  declaration=dict(path=DECLARATION, sha256=digest), method_record=method, identity=identity, selection=selection,
                  method_freeze=None if frozen is None else dict(path=FREEZE, sha256=hashlib.sha256(frozen[1]).hexdigest()),
                  ledger=None if ledgered is None else dict(path=LEDGER, files=len(ledgered)), ade_fit=ade,
                  design=dict(spec["design"]), bootstrap=dict(replicates=spec["replicates"], seed=spec["seed"], interval=declared["interval"],
                                                               numpy=np.__version__),
                  code={name: common.sha256(common.ROOT / name) for name in CODE}, percentiles=declared["percentiles"],
                  pair_sets=list(spec["pair_sets"]), labels={name: labels[name] for name in sorted(labels)},
                  missing=missing, runs={mode: {method: run.summary() for method, run in by_method.items()} for mode, by_method in runs.items()},
                  modes=sections, inputs=[dict(path=path, sha256=reader.files[path]) for path in sorted(reader.files)], **(dict(notes=notes) if notes else {}))
    write_outputs(Path(tables_root or root / TABLES) / dataset / split, record)
    return record


def released_figures(reader, units, runs, scores, pair_sets, percentile):
    """Refused unless the units of every Bern run give its released scorer's track summary and the joined and ranged
    figures its joined/summary.json holds, the same arithmetic over every unit of a track in the scorer's order."""
    for method, run in runs.items():
        folder = scores[method]
        with open(reader.file(folder / "official/track_summary.csv"), newline="") as stream:
            official = {int(row["samples"]): [float(row[f"balanced_s{q}"]) for q in (50, 75, 90)] for row in csv.DictReader(stream)
                        if row["mode"] == units.mode and int(row["q"]) == 2}
        joined = reader.json(folder / "joined/summary.json")["tracks"]
        for track in sorted(set(units.track.tolist())):
            for p, name in enumerate(pair_sets):
                scored = np.flatnonzero((units.track == track) & (units.pairs[p] > 0))
                grouped = {}
                for i in scored:
                    grouped.setdefault(units.cell[i], []).append(float(units.errors[method][p][i]))
                values = [float(percentile(grouped, q)) for q in (0.5, 0.75, 0.9)] if grouped else None
                if name in joined.get(str(track), {}) and values != joined[str(track)][name]["balanced"]:
                    raise Refused(f"the {name} figure of {relative(folder, reader.root)} differs from its joined summary")
                if name == RELEASED and values != official.get(track):
                    raise Refused(f"the units of {relative(folder, reader.root)} do not give its released scorer's track summary")


# ---------------------------------------------------------------------------------------------------------------------
# Outputs


def csv_text(header, rows):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def number(value):
    return "" if value is None else repr(value)


def entries(record):
    """Every table of a record as (mode, group, track, pair set, track entry, table entry)."""
    for mode, section in record["modes"].items():
        for group, entry in section["groups"].items():
            for track, track_entry in entry["tracks"].items():
                for pair_set, table in track_entry["pair_sets"].items():
                    yield mode, group, track, pair_set, track_entry, table


# the columns of one percentile or difference in the CSV files
FIELDS = ("cm", "exact_cm", "low_cm", "high_cm", "se_cm", "empty_draws")


def fields(value):
    """The columns of one estimate, empty where it has none."""
    if value is None:
        return [""] * len(FIELDS)
    low, high = value["interval_cm"] or (None, None)
    return [number(value["value_cm"]), number(value["exact_cm"]), number(low), number(high), number(value["se_cm"]),
            number(value.get("empty_draws", 0))]


def percentiles_csv(record):
    names = [name for name, _ in record["percentiles"]]
    header = ["dataset", "split", "mode", "group", "track", "pair_set", "method", "units", "scored_units", "unit_share", "pairs", "possible_pairs",
              "pair_share", "cells", *(f"{name}_{field}" for name in names for field in FIELDS), "failed_units", "aborted_units", "median_ms",
              "threads_per_process", "processes"]
    rows = []
    for mode, group, track, pair_set, track_entry, table in entries(record):
        for method, values in table["methods"].items():
            solve = track_entry["methods"][method]
            rows.append([record["dataset"], record["split"], mode, group, track, pair_set, method, table["units"], table["scored_units"],
                         number(table["unit_share"]), table["pairs"], table["possible_pairs"], number(table["pair_share"]), table["cells"],
                         *(column for name in names for column in fields(values[name])), solve["failed_units"], solve["aborted_units"],
                         number(solve["runtime"]["median_ms"]), number(solve["runtime"]["threads_per_process"]), number(solve["runtime"]["processes"])])
    return csv_text(header, rows)


def differences_csv(record):
    names = [name for name, _ in record["percentiles"]]
    header = ["dataset", "split", "mode", "group", "track", "pair_set", "difference", *(f"{name}_{field}" for name in names for field in FIELDS)]
    rows = [[record["dataset"], record["split"], mode, group, track, pair_set, difference, *(column for name in names for column in fields(values[name]))]
            for mode, group, track, pair_set, _, table in entries(record) for difference, values in table["differences"].items()]
    return csv_text(header, rows)


def latex_value(value, signed=False):
    if value is None:
        return "--"
    form = "{:+.1f}" if signed else "{:.1f}"
    if value["interval_cm"] is None:
        return f"{form.format(value['value_cm'])} [--]"
    low, high = value["interval_cm"]
    return f"{form.format(value['value_cm'])} [{form.format(low)}, {form.format(high)}]"


def share(value):
    return "--" if value is None else f"{100 * value:.1f}\\,\\%"


LATEX_SPECIALS = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
                  "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}


def latex_text(text):
    """Plain text for a LaTeX table, its special characters escaped."""
    return "".join(LATEX_SPECIALS.get(character, character) for character in str(text))


def latex(record):
    """One tabular of the percentiles and one of the differences per mode, group and pair set, every track in each. The
    declared outcomes a SOLID run names (Run.marks) mark SOLID with a dagger in every tabular of its mode, and each is a
    note under the tabular. A mark that names its groups, one a budget check without a converged step leaves
    (budget_marks), marks the tabulars of those groups alone. Every data defect of a held-out half is a note as well,
    which holds for every method alike (heldout_notes)."""
    names = [name for name, _ in record["percentiles"]]
    lines = [f"% {record['label']}, {record['split']} half. Written by research/solid/final_tables.py under the declared analysis "
             f"{record['declaration']['path']}, SHA-256 {record['declaration']['sha256']}.",
             "% Balanced percentiles in cm, each with the 95 percent interval of its paired cluster bootstrap in brackets.", ""]
    common_notes = [f"\\multicolumn{{{len(names) + 2}}}{{l}}{{\\footnotesize {latex_text(note)}.}} \\\\" for note in record.get("notes") or []]
    for mode, section in record["modes"].items():
        marks = ((record.get("runs") or {}).get(mode, {}).get("solid") or {}).get("marks") or []
        for group, entry in section["groups"].items():
            shown = [mark if isinstance(mark, str) else mark["text"] for mark in marks
                     if isinstance(mark, str) or mark.get("groups") is None or group in mark["groups"]]
            label = dict(record["labels"], solid=record["labels"]["solid"] + "$^\\dagger$") if shown else record["labels"]
            notes = [f"\\multicolumn{{{len(names) + 2}}}{{l}}{{\\footnotesize $^\\dagger$ {latex_text(mark)}.}} \\\\" for mark in shown] + common_notes
            for pair_set in record["pair_sets"]:
                columns = "ll" + "r" * len(names)
                lines.append(f"% {record['label']}, {mode}, {group}, {pair_set} pairs")
                lines.append(f"\\begin{{tabular}}{{{columns}}}\\toprule")
                lines.append("Track & Method & " + " & ".join(names) + " \\\\\\midrule")
                for k, (track, track_entry) in enumerate(entry["tracks"].items()):
                    table = track_entry["pair_sets"][pair_set]
                    if k:
                        lines.append("\\midrule")
                    for j, method in enumerate(section["methods"]):
                        cells = [latex_value(table["methods"][method][name]) for name in names]
                        lines.append(f"{track if j == 0 else ''} & {label[method]} & " + " & ".join(cells) + " \\\\")
                    lines.append(f"\\multicolumn{{{len(columns)}}}{{l}}{{\\footnotesize {share(table['pair_share'])} of pairs and "
                                 f"{share(table['unit_share'])} of units scored}} \\\\")
                lines.extend(notes)
                lines.append("\\bottomrule\\end{tabular}")
                lines.append("")
                if not section["differences"]:
                    continue
                lines.append(f"% {record['label']}, {mode}, {group}, {pair_set} pairs, paired differences in cm")
                lines.append(f"\\begin{{tabular}}{{{columns}}}\\toprule")
                lines.append("Track & Difference & " + " & ".join(names) + " \\\\\\midrule")
                for k, (track, track_entry) in enumerate(entry["tracks"].items()):
                    table = track_entry["pair_sets"][pair_set]
                    if k:
                        lines.append("\\midrule")
                    for j, difference in enumerate(section["differences"]):
                        minuend, _, subtrahend = difference.partition(" minus ")
                        cells = [latex_value(table["differences"][difference][name], signed=True) for name in names]
                        lines.append(f"{track if j == 0 else ''} & {label[minuend]} $-$ {label[subtrahend]} & " + " & ".join(cells) + " \\\\")
                lines.extend(notes)
                lines.append("\\bottomrule\\end{tabular}")
                lines.append("")
    return "\n".join(lines)


def write_outputs(folder, record):
    write_bytes(folder / "tables.json", encoded(record))
    write_bytes(folder / "percentiles.csv", percentiles_csv(record).encode())
    write_bytes(folder / "differences.csv", differences_csv(record).encode())
    write_bytes(folder / "tables.tex", latex(record).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("declare", "check", "build"))
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--split", default="evaluation", choices=("evaluation", "training"))
    parser.add_argument("--replace", action="store_true", help="replace a declaration that differs, only before any evaluation table")
    parser.add_argument("--runs", type=Path, help="the runs root in place of out/ of the repository, in a reader's reproduction")
    parser.add_argument("--tables", type=Path, help=f"the folder of the tables in place of {TABLES}, in a reader's reproduction")
    args = parser.parse_args()
    if (args.runs or args.tables) and not READER:
        parser.error("--runs and --tables serve a reader's reproduction, SOLID_READER=1")
    if args.command == "declare":
        written = declare(replace=args.replace)
        print(f"{DECLARATION} {'written' if written else 'unchanged'}, SHA-256 {common.sha256(common.ROOT / DECLARATION)}", flush=True)
    elif args.command == "check":
        if READER:
            print("a reader's reproduction builds under the declaration of final_tables.py itself", flush=True)
            return
        changed = check()
        if changed:
            raise SystemExit(f"{DECLARATION} differs from the declaration of final_tables.py in {changed}")
        print(f"{DECLARATION} holds the declaration of final_tables.py", flush=True)
    else:
        if not args.dataset:
            parser.error("build needs --dataset")
        record = build(args.dataset, args.split, runs_root=args.runs, tables_root=args.tables)
        for mode, group, track, pair_set, _, table in entries(record):
            solid = table["methods"]["solid"]["S75"]
            print(f"{args.dataset} {args.split} {mode} {group} track {track} {pair_set}: SOLID S75 "
                  f"{'none' if solid is None else latex_value(solid)}, {table['scored_units']} of {table['units']} units scored", flush=True)
        print(f"written {args.tables or TABLES}/{args.dataset}/{args.split}", flush=True)


if __name__ == "__main__":
    main()
