"""SOLID anchor only on one half of HYMN, a held-out benchmark, eight surveyed points and k of ten anchors per unit.

Reads the half that research/benchmarks/datasets/hymn_layout.py built under external-benchmarks/hymn-layout of the data
folder.
Either half runs only after the method freeze, research/artifacts/solid/selections/method-freeze.json, with the records
of held-out stage one committed and while the ledger research/artifacts/solid/splits/heldout-evaluations.json opens
HYMN. Every unit takes the tracked HYMN profile of stage one, which must equal its composition from the method record,
research/artifacts/solid/fits/hymn_layout_fit.json and the held-out deployment record by profiles.py, and the operating
area of the deployment rules. The anchor count cells are 1 to 10, and the headline is balanced over those with three
or more anchors. The run goes to out/solid/hymn/<split>/anc with the solved profile beside the predictions. A reader's
run (SOLID_READER=1) has neither the freeze nor the ledger and solves the tracked profile as written.

Usage: run_hymn.py --split training|evaluation [--workers 8]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_runner  # noqa: E402


def run(split, workers):
    return heldout_runner.run("hymn", split, workers)


if __name__ == "__main__":
    heldout_runner.main("hymn", __doc__)
