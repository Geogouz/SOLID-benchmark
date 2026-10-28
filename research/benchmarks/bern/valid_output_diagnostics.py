#!/usr/bin/env python3
"""Measure valid estimator output without changing the released score."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


SCENARIO_COLUMNS = [
    "scenario_id",
    "iteration",
    "samples",
    "mode",
    "anchors",
    "agents",
    "valid_agents",
    "failed_agents",
    "missing_agent_ids",
    "nonfinite_agent_ids",
    "eligible_pairs",
    "valid_eligible_pairs",
    "conditional_mean_pair_error",
]
OUTPUT_FILES = (
    "valid_output_scenario_scores.csv.gz",
    "valid_output_summary.json",
)


def load_released_scorer(path: Path):
    name = "_solid_released_twr_cloud_bern_scorer"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load released scorer {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _iterations(data: dict) -> list[dict]:
    value = data.get("Iterations", [])
    return list(value.values()) if isinstance(value, dict) else list(value)


def _track_summary(released, totals: dict, cell_values: dict) -> dict:
    conditional_units = sum(len(cell) for cell in cell_values.values())
    return {
        "scenario_units": totals["scenario_units"],
        "expected_agent_outputs": totals["expected_agent_outputs"],
        "valid_agent_outputs": totals["valid_agent_outputs"],
        "failed_agent_outputs": totals["failed_agent_outputs"],
        "missing_agent_outputs": totals["missing_agent_outputs"],
        "nonfinite_agent_outputs": totals["nonfinite_agent_outputs"],
        "agent_success_rate": _rate(
            totals["valid_agent_outputs"], totals["expected_agent_outputs"]
        ),
        "agent_failure_rate": _rate(
            totals["failed_agent_outputs"], totals["expected_agent_outputs"]
        ),
        "scenario_units_with_any_agent_failure": totals[
            "scenario_units_with_any_agent_failure"
        ],
        "scenario_units_with_partial_agent_failure": totals[
            "scenario_units_with_partial_agent_failure"
        ],
        "scenario_units_with_all_agents_failed": totals[
            "scenario_units_with_all_agents_failed"
        ],
        "scenario_unit_any_failure_rate": _rate(
            totals["scenario_units_with_any_agent_failure"],
            totals["scenario_units"],
        ),
        "eligible_pairs": totals["eligible_pairs"],
        "valid_eligible_pairs": totals["valid_eligible_pairs"],
        "valid_pair_coverage": _rate(
            totals["valid_eligible_pairs"], totals["eligible_pairs"]
        ),
        "zero_eligible_pair_units": totals["zero_eligible_pair_units"],
        "units_with_eligible_pairs_but_no_valid_pair": totals[
            "units_with_eligible_pairs_but_no_valid_pair"
        ],
        "conditional_scored_scenario_units": conditional_units,
        "conditional_scenario_unit_coverage": _rate(
            conditional_units, totals["scenario_units"]
        ),
        "reporting_cells": len(totals["reporting_cells"]),
        "conditional_scored_cells": sum(bool(cell) for cell in cell_values.values()),
        "conditional_balanced_s50_cm": _finite_or_none(
            released.balanced_percentile(cell_values, 0.50)
        ),
        "conditional_balanced_s75_cm": _finite_or_none(
            released.balanced_percentile(cell_values, 0.75)
        ),
        "conditional_balanced_s90_cm": _finite_or_none(
            released.balanced_percentile(cell_values, 0.90)
        ),
    }


def calculate(
    released,
    scenarios: Path,
    predictions_csv: Path,
    mode: str,
    scenario_writer=None,
) -> tuple[list[dict], dict]:
    mode = released.canonical_mode(mode)
    predictions = released.load_predictions(predictions_csv)
    missing_prediction = object()
    scenario_rows = []
    totals = {sample: defaultdict(int) for sample in (1, 10)}
    cell_values = {sample: defaultdict(list) for sample in (1, 10)}
    for sample_totals in totals.values():
        sample_totals["reporting_cells"] = set()

    for scenario_id, data in released.iter_json_scenarios(scenarios):
        agents = sorted(int(node) for node in data.get("Agents", []))
        anchors = sorted(int(node) for node in data.get("Anchors", []))
        if mode == "AF" and anchors:
            continue
        if mode in {"AAC", "ANC"} and not anchors:
            continue
        agent_set = set(agents)
        true_positions = data.get("True Positions of Agents", {})
        true_points = {
            int(node): released.true_point(position, 2)
            for node, position in true_positions.items()
            if int(node) in agent_set
        }
        penalty = released.scenario_diameter(true_points)
        for iteration_index, iteration in enumerate(_iterations(data), start=1):
            iteration_id = int(iteration.get("ID", iteration_index))
            for samples in (1, 10):
                sample_totals = totals[samples]
                estimated_points = {}
                missing_agent_ids = []
                nonfinite_agent_ids = []
                for node in agents:
                    key = (scenario_id, iteration_id, samples, mode, 2, node)
                    point = predictions.pop(key, missing_prediction)
                    if point is missing_prediction:
                        missing_agent_ids.append(node)
                    elif not released.is_finite_point(point, 2):
                        nonfinite_agent_ids.append(node)
                    else:
                        estimated_points[node] = point
                valid_agents = len(estimated_points)
                failed_agents = len(agents) - valid_agents
                eligible_pairs = released.eligible_pairs_for_mode(
                    agents, mode, iteration, samples
                )
                valid_pairs = [
                    (left, right)
                    for left, right in eligible_pairs
                    if left in estimated_points and right in estimated_points
                ]
                errors = [
                    min(
                        abs(
                            released.dist(
                                estimated_points[left], estimated_points[right]
                            )
                            - released.dist(true_points[left], true_points[right])
                        ),
                        penalty,
                    )
                    for left, right in valid_pairs
                ]
                conditional_error = sum(errors) / len(errors) if errors else math.nan
                anchors_in_cell = 0 if mode == "AF" else len(anchors)
                cell = (mode, 2, samples, anchors_in_cell, len(agents))
                sample_totals["reporting_cells"].add(cell)
                if math.isfinite(conditional_error):
                    cell_values[samples][cell].append(conditional_error)

                sample_totals["scenario_units"] += 1
                sample_totals["expected_agent_outputs"] += len(agents)
                sample_totals["valid_agent_outputs"] += valid_agents
                sample_totals["failed_agent_outputs"] += failed_agents
                sample_totals["missing_agent_outputs"] += len(missing_agent_ids)
                sample_totals["nonfinite_agent_outputs"] += len(nonfinite_agent_ids)
                sample_totals["scenario_units_with_any_agent_failure"] += int(
                    failed_agents > 0
                )
                sample_totals["scenario_units_with_partial_agent_failure"] += int(
                    0 < failed_agents < len(agents)
                )
                sample_totals["scenario_units_with_all_agents_failed"] += int(
                    bool(agents) and failed_agents == len(agents)
                )
                sample_totals["eligible_pairs"] += len(eligible_pairs)
                sample_totals["valid_eligible_pairs"] += len(valid_pairs)
                sample_totals["zero_eligible_pair_units"] += int(not eligible_pairs)
                sample_totals[
                    "units_with_eligible_pairs_but_no_valid_pair"
                ] += int(bool(eligible_pairs) and not valid_pairs)
                row = {
                    "scenario_id": scenario_id,
                    "iteration": iteration_id,
                    "samples": samples,
                    "mode": mode,
                    "anchors": anchors_in_cell,
                    "agents": len(agents),
                    "valid_agents": valid_agents,
                    "failed_agents": failed_agents,
                    "missing_agent_ids": "|".join(map(str, missing_agent_ids)),
                    "nonfinite_agent_ids": "|".join(map(str, nonfinite_agent_ids)),
                    "eligible_pairs": len(eligible_pairs),
                    "valid_eligible_pairs": len(valid_pairs),
                    "conditional_mean_pair_error": (
                        f"{conditional_error:.12g}"
                        if math.isfinite(conditional_error)
                        else ""
                    ),
                }
                if scenario_writer is None:
                    scenario_rows.append(row)
                else:
                    scenario_writer.writerow(row)

    extra_prediction_rows = len(predictions)
    if extra_prediction_rows:
        raise ValueError(
            f"prediction archive contains {extra_prediction_rows} rows outside the official population"
        )
    summary = {
        "schema_version": "solid.twr-cloud-bern.valid-output-diagnostic.v1",
        "status": "custom companion diagnostic",
        "mode": mode,
        "dimension": 2,
        "agent_success_definition": (
            "the expected agent has a finite submitted x and y coordinate"
        ),
        "conditional_accuracy_definition": (
            "released q=2 eligible-pair error with the released scenario-diameter cap, "
            "after omitting pairs whose endpoint lacks a valid output"
        ),
        "ranking_use": "diagnostic only",
        "tracks": {
            str(samples): _track_summary(
                released, totals[samples], cell_values[samples]
            )
            for samples in (1, 10)
        },
    }
    return scenario_rows, summary


def generate(
    *,
    scorer: Path,
    scenarios: Path,
    predictions_csv: Path,
    mode: str,
    output_directory: Path,
) -> dict:
    released = load_released_scorer(scorer)
    output_directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        output_directory / "valid_output_scenario_scores.csv.gz", "wt", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SCENARIO_COLUMNS)
        writer.writeheader()
        _, summary = calculate(
            released,
            scenarios,
            predictions_csv,
            mode,
            scenario_writer=writer,
        )
    (output_directory / "valid_output_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary
