"""SOLID anchor only on one half of AAU Meshed UWB, a held-out benchmark of nine static nodes split by flights.

Reads the half that research/benchmarks/datasets/aau_layout.py built under external-benchmarks/aau-layout of the data
folder.
Either half runs only after the method freeze, research/artifacts/solid/selections/method-freeze.json, with the records
of held-out stage one committed and while the ledger research/artifacts/solid/splits/heldout-evaluations.json opens
AAU. Every unit takes the tracked AAU profile of stage one, which must equal its composition from the method record,
research/artifacts/solid/fits/aau_layout_fit.json and the held-out deployment record by profiles.py, and the operating
area of the deployment rules within the published 5 m by 8 m area. The anchor count cells are 1 to 7, and the headline
is balanced over those with three or more anchors. The run goes to out/solid/aau/<split>/anc with the solved profile
beside the predictions, and it names the evaluation flights the relocation guard excluded. A reader's run
(SOLID_READER=1) has neither the freeze nor the ledger and solves the tracked profile as written.

Usage: run_aau.py --split training|evaluation [--workers 8]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_runner  # noqa: E402


def run(split, workers):
    return heldout_runner.run("aau", split, workers)


if __name__ == "__main__":
    heldout_runner.main("aau", __doc__)
