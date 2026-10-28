#!/usr/bin/env python3.12
"""Extract canonical Bern calibration rows without localization access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_SHA256 = "65c077bb1aa9b8793d9b206a406ecfd574047ad345fbafdbf93cfcdfad26fa21"
INPUT_MD5 = "d5d71fe55c758c2a87d7efd812357c5b"
ROW_HEADER = b"SOLID.BERN.CALIBRATION.ROWS.V1\n"
TRAINING_SCANS = tuple(range(1, 12))
SCAN_COUNTS = {
    1: 174_363,
    2: 181_526,
    3: 176_367,
    4: 179_537,
    5: 175_715,
    6: 174_897,
    7: 174_111,
    8: 180_338,
    9: 174_896,
    10: 182_267,
    11: 173_869,
}
REQUIRED_INTEGER_COLUMNS = (
    "Scan_ID",
    "Timestamp",
    "Listening_Period",
    "Responder_ID",
    "Responder_Pos_ID",
    "Poller_ID",
    "Poller_Pos_ID",
)
REQUIRED_COLUMNS = (
    "Class",
    *REQUIRED_INTEGER_COLUMNS,
    "ToF",
    "Node_Distance_3D",
)
ROW_DTYPE = np.dtype(
    [
        ("source_ordinal", "<u8"),
        ("scan_id", "<i8"),
        ("timestamp", "<i8"),
        ("listening_period", "<i8"),
        ("poller_id", "<i8"),
        ("poller_pos_id", "<i8"),
        ("responder_id", "<i8"),
        ("responder_pos_id", "<i8"),
        ("tof_bits", "<u8"),
        ("distance_bits", "<u8"),
    ],
    align=False,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_rows(path: Path) -> pd.DataFrame:
    if file_digest(path, "sha256") != INPUT_SHA256 or file_digest(path, "md5") != INPUT_MD5:
        raise ValueError("the responder table identity is not frozen")
    frame = pd.read_pickle(path)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns {sorted(missing)}")
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("the responder table index is not a RangeIndex")
    if frame.index.start != 0 or frame.index.step != 1 or frame.index.stop != len(frame):
        raise ValueError("the source row ordinals are not contiguous")

    selected = frame.loc[
        frame["Class"].eq("Train") & frame["Scan_ID"].isin(TRAINING_SCANS),
        list(REQUIRED_COLUMNS),
    ].copy()
    validate_selected_rows(selected)
    return selected


def validate_selected_rows(frame: pd.DataFrame) -> None:
    if len(frame) != sum(SCAN_COUNTS.values()):
        raise ValueError("the selected calibration row count is not frozen")
    if not frame["Class"].eq("Train").all():
        raise ValueError("a selected calibration row is not training data")
    for column in REQUIRED_INTEGER_COLUMNS:
        if frame[column].dtype != np.dtype("int64"):
            raise ValueError(f"{column} is not signed 64 bit integer data")
    for column in ("ToF", "Node_Distance_3D"):
        if frame[column].dtype != np.dtype("float64"):
            raise ValueError(f"{column} is not binary64 data")
        values = frame[column].to_numpy(copy=False)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{column} contains an invalid selected value")
    actual_counts = frame.groupby("Scan_ID", sort=True).size().to_dict()
    if actual_counts != SCAN_COUNTS:
        raise ValueError("the per-scan calibration row counts are not frozen")


def canonical_records(frame: pd.DataFrame) -> np.ndarray:
    source_ordinals = frame.index.to_numpy(dtype=np.uint64, copy=True)
    columns = {
        column: frame[column].to_numpy(dtype=np.int64, copy=True)
        for column in REQUIRED_INTEGER_COLUMNS
    }
    tof = canonical_floats(frame["ToF"].to_numpy(dtype=np.float64, copy=True))
    distance = canonical_floats(
        frame["Node_Distance_3D"].to_numpy(dtype=np.float64, copy=True)
    )
    tof_bits = tof.view(np.uint64)
    distance_bits = distance.view(np.uint64)
    order = np.lexsort(
        (
            source_ordinals,
            distance_bits,
            tof_bits,
            columns["Responder_Pos_ID"],
            columns["Responder_ID"],
            columns["Poller_Pos_ID"],
            columns["Poller_ID"],
            columns["Listening_Period"],
            columns["Timestamp"],
            columns["Scan_ID"],
        )
    )
    records = np.empty(len(frame), dtype=ROW_DTYPE)
    records["source_ordinal"] = source_ordinals[order]
    records["scan_id"] = columns["Scan_ID"][order]
    records["timestamp"] = columns["Timestamp"][order]
    records["listening_period"] = columns["Listening_Period"][order]
    records["poller_id"] = columns["Poller_ID"][order]
    records["poller_pos_id"] = columns["Poller_Pos_ID"][order]
    records["responder_id"] = columns["Responder_ID"][order]
    records["responder_pos_id"] = columns["Responder_Pos_ID"][order]
    records["tof_bits"] = tof_bits[order]
    records["distance_bits"] = distance_bits[order]
    return records


def canonical_floats(values: np.ndarray) -> np.ndarray:
    values[values == 0.0] = 0.0
    return values


def write_row_manifests(records: np.ndarray, output: Path) -> list[dict[str, object]]:
    if len(ROW_HEADER) != 31 or ROW_DTYPE.itemsize != 80:
        raise RuntimeError("the calibration row layout is not frozen")
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / f"outer-holdout-scan-{scan:02d}.rows.bin" for scan in TRAINING_SCANS
    ] + [output / "full-training.rows.bin"]
    if any(path.exists() for path in targets):
        raise FileExistsError("a calibration row manifest already exists")

    with tempfile.TemporaryDirectory(prefix="bern-calibration-rows-", dir=output) as temporary:
        temporary_path = Path(temporary)
        scan_chunks = {}
        for scan in TRAINING_SCANS:
            chunk = temporary_path / f"scan-{scan:02d}.bin"
            records[records["scan_id"] == scan].tofile(chunk)
            expected_size = SCAN_COUNTS[scan] * ROW_DTYPE.itemsize
            if chunk.stat().st_size != expected_size:
                raise ValueError("a per-scan row chunk has an invalid byte length")
            scan_chunks[scan] = chunk

        entries = []
        for held_scan in (*TRAINING_SCANS, None):
            fit_id = (
                "full-training"
                if held_scan is None
                else f"outer-holdout-scan-{held_scan:02d}"
            )
            fit_scans = [scan for scan in TRAINING_SCANS if scan != held_scan]
            row_count = sum(SCAN_COUNTS[scan] for scan in fit_scans)
            target = output / f"{fit_id}.rows.bin"
            pending = temporary_path / f"{fit_id}.pending"
            with pending.open("xb") as destination:
                destination.write(ROW_HEADER)
                destination.write(struct.pack("<Q", row_count))
                for scan in fit_scans:
                    with scan_chunks[scan].open("rb") as source:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            expected_size = len(ROW_HEADER) + 8 + row_count * ROW_DTYPE.itemsize
            if pending.stat().st_size != expected_size:
                raise ValueError("a calibration row manifest has an invalid byte length")
            os.link(pending, target)
            pending.unlink()
            entries.append(
                {
                    "fit_id": fit_id,
                    "fit_scan_ids": fit_scans,
                    "held_scan_id": held_scan,
                    "row_count": row_count,
                    "row_manifest_path": target.name,
                    "row_manifest_sha256": file_digest(target, "sha256"),
                }
            )
    directory = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return entries


def extract(input_path: Path, output: Path) -> list[dict[str, object]]:
    frame = load_selected_rows(input_path)
    records = canonical_records(frame)
    return write_row_manifests(records, output)


def main() -> None:
    args = arguments()
    entries = extract(args.input, args.output)
    print(json.dumps({"input_sha256": INPUT_SHA256, "fits": entries}, separators=(",", ":")))


if __name__ == "__main__":
    main()
