#!/usr/bin/env python3
"""Remove archive metadata that the released scorer mistakes for scenarios."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent


def load_scorer_module():
    path = DIRECTORY / "official_scorer.py"
    specification = importlib.util.spec_from_file_location("official_scorer", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


official_scorer = load_scorer_module()


def prepare(source: Path, output: Path, force: bool = False) -> dict:
    official_scorer.require_hash(
        source, official_scorer.SCENARIO_SHA256, "scenario archive"
    )
    if output.exists() and not force:
        identity = official_scorer.verify_scorer_scenarios(source, output)
        return {"status": "reused", **identity}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="solid-bern-scenarios-", dir=output.parent) as temp:
        candidate = Path(temp) / output.name
        shutil.copyfile(source, candidate)
        subprocess.run(
            ["zip", "-q", "-d", str(candidate), "__MACOSX/*"],
            check=True,
        )
        identity = official_scorer.verify_scorer_scenarios(source, candidate)
        candidate.replace(output)
    return {"status": "created", **identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.force), indent=2))


if __name__ == "__main__":
    main()
