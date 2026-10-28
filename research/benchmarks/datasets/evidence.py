"""Shared locations and loaders of the benchmark data collection."""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "bern"))
retained_data = importlib.import_module("retained_data")
official_scorer = importlib.import_module("official_scorer")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def records(directory):
    for path in sorted(directory.glob("part-*.jsonl")):
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)


def retained_input(digest):
    """The retained input file with this SHA256, located through the data collection's inventory, or for a reader
    (SOLID_READER=1) as the one file of the folder that the first 16 hexadecimal digits of the SHA256 name."""
    if os.environ.get("SOLID_READER") == "1":
        folder = retained_data.DATA / "inputs" / digest[:16]
        found = [path for path in folder.iterdir() if path.is_file() and path.name != ".DS_Store"] if folder.is_dir() else []
        if len(found) != 1:
            raise FileNotFoundError(f"{folder} must hold the one input file of SHA256 {digest}")
        official_scorer.require_hash(found[0], digest, "input")
        return found[0]
    files = read(retained_data.DATA / "inventory.json")["files"]
    original = next(key for key, row in files.items() if row["sha256"] == digest)
    return retained_data.resolve(original)


def load_scorer():
    """The released TWR-CLOUD-Bern scorer as a module."""
    source = retained_input(official_scorer.SCORER_SHA256)
    spec = importlib.util.spec_from_file_location("released_scorer", source)
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorer
    spec.loader.exec_module(scorer)
    return scorer
