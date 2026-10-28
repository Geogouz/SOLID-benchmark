#!/usr/bin/env python3
"""Fit Bayesian baseline profiles from the official Bern training scans."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


SCHEMA_VERSION = "solid.bayesian-baseline-profile.v1"
EXPECTED_INPUT_MD5 = "ddca1cc9de31e1e525c6597519f040a2"
TRAINING_SCANS = tuple(range(1, 12))
RANGE_SLOPE_CM_PER_NS = 28.50909085
RANGE_OFFSET_CM = 19.20725779


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_data(path):
    if file_digest(path, "md5") != EXPECTED_INPUT_MD5:
        raise ValueError("the responder table checksum does not match the official release")
    frame = pd.read_pickle(path)
    frame = frame[frame["Scan_ID"].isin(TRAINING_SCANS)].copy()
    if set(frame["Class"].unique()) != {"Train"}:
        raise ValueError("the selected rows are not exclusively marked as training data")
    required = [
        "ToF",
        "Poller_Pos_ID",
        "Responder_Pos_ID",
        "Poller_X",
        "Poller_Y",
        "Poller_Z",
        "Responder_X",
        "Responder_Y",
        "Responder_Z",
    ]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("the selected training values contain non-finite numbers")
    return frame


def calibrated_ranges(frame):
    ranges = RANGE_SLOPE_CM_PER_NS * frame["ToF"].to_numpy(dtype=float)
    return np.maximum(ranges + RANGE_OFFSET_CM, 0.0)


def planar_distances(frame):
    horizontal = (frame["Poller_X"] - frame["Responder_X"]).to_numpy(dtype=float)
    vertical = (frame["Poller_Y"] - frame["Responder_Y"]).to_numpy(dtype=float)
    return np.hypot(horizontal, vertical) * 100.0


def area_bounds(frame):
    x = np.concatenate(
        [frame["Poller_X"].to_numpy(dtype=float), frame["Responder_X"].to_numpy(dtype=float)]
    ) * 100.0
    y = np.concatenate(
        [frame["Poller_Y"].to_numpy(dtype=float), frame["Responder_Y"].to_numpy(dtype=float)]
    ) * 100.0
    x_margin = 0.05 * np.ptp(x)
    y_margin = 0.05 * np.ptp(y)
    return {
        "minimum_x_cm": float(np.min(x) - x_margin),
        "maximum_x_cm": float(np.max(x) + x_margin),
        "minimum_y_cm": float(np.min(y) - y_margin),
        "maximum_y_cm": float(np.max(y) + y_margin),
    }


def ade_negative_log_likelihood(parameters, measured, residual):
    positive = parameters[0] * measured + parameters[1]
    negative = parameters[2] * measured + parameters[3]
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        return math.inf
    tails = np.where(residual >= 0.0, residual / positive, -residual / negative)
    return float(np.sum(np.log(positive + negative) + tails))


def fit_ade_model(measured, truth):
    residual = measured - truth
    fit = minimize(
        ade_negative_log_likelihood,
        np.array([0.02, 20.0, 0.01, 40.0]),
        args=(measured, residual),
        method="Nelder-Mead",
        bounds=((0.0, None), (1e-6, None), (0.0, None), (1e-6, None)),
        options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-4},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"ADE maximum-likelihood fit failed with {fit.message}")
    return {
        "positive_slope": float(fit.x[0]),
        "positive_intercept_cm": float(fit.x[1]),
        "negative_slope": float(fit.x[2]),
        "negative_intercept_cm": float(fit.x[3]),
    }, float(fit.fun)


def fit_constant_ade_model(measured, truth):
    residual = measured - truth

    def objective(intercepts):
        return ade_negative_log_likelihood(
            np.array([0.0, intercepts[0], 0.0, intercepts[1]]),
            measured,
            residual,
        )

    fit = minimize(
        objective,
        np.array([30.0, 50.0]),
        method="Nelder-Mead",
        bounds=((1e-6, None), (1e-6, None)),
        options={"maxiter": 1000, "xatol": 1e-10, "fatol": 1e-4},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"constant ADE maximum-likelihood fit failed with {fit.message}")
    return {
        "positive_slope": 0.0,
        "positive_intercept_cm": float(fit.x[0]),
        "negative_slope": 0.0,
        "negative_intercept_cm": float(fit.x[1]),
    }, float(fit.fun)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_profiles(input_path):
    frame = load_training_data(input_path)
    area = area_bounds(frame)
    measured = calibrated_ranges(frame)
    truth = planar_distances(frame)
    ranging_model, ade_nll = fit_ade_model(measured, truth)
    constant_ranging_model, constant_ade_nll = fit_constant_ade_model(measured, truth)

    configurations = {
        "ade-bp-2015.json": {
            "method": "ade-bp-2015",
            "area": area,
            "particles": 500,
            "iterations": 4,
            "importance_iterations": 256,
            "ranging_model": ranging_model,
        },
    }
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "training",
        "input_md5": EXPECTED_INPUT_MD5,
        "input_sha256": file_digest(input_path, "sha256"),
        "training_scans": list(TRAINING_SCANS),
        "training_rows": int(len(frame)),
        "evaluation_rows_used": 0,
        "calibration": {
            "role": "shared frozen Bern measurement calibration",
            "fitted_by_this_stage": False,
            "slope_cm_per_ns": RANGE_SLOPE_CM_PER_NS,
            "offset_cm": RANGE_OFFSET_CM,
        },
        "area": area,
        "ade_fit": {
            "likelihood_fit_winner": "distance-dependent",
            "distance_definition": "planar Euclidean centimetres",
            "negative_log_likelihood": ade_nll,
            "parameters": ranging_model,
            "candidates": {
                "constant": {
                    "negative_log_likelihood": constant_ade_nll,
                    "parameters": constant_ranging_model,
                },
                "distance-dependent": {
                    "negative_log_likelihood": ade_nll,
                    "parameters": ranging_model,
                },
            },
        },
        "method_controls": {
            "ade_bp_particles": 500,
            "ade_bp_iterations": 4,
            "ade_bp_importance_iterations_cap": 256,
            "ade_bp_mean_convergence_fraction": 0.001,
            "ade_bp_covariance_convergence_fraction": 0.0001,
            "ade_bp_required_stable_refinements": 2,
            "ade_bp_ranging_model": "distance-dependent Model 2",
            "repeated_measurement_adapter": {
                "ade-bp-2015": "per-link median",
            },
        },
    }
    return configurations, evidence


def main():
    args = arguments()
    configurations, evidence = build_profiles(args.input)
    for filename, configuration in configurations.items():
        write_json(args.output_directory / "configs" / filename, configuration)
    write_json(args.output_directory / "profile-v1.json", evidence)
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
