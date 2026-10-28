#!/usr/bin/env python3
"""Fetch the public data the pipeline reads into the data folder and verify it against settings/data/pins.json.

The data folder is research/data under the repository unless the variable SOLID_DATA names another one in the same
layout. A file arrives as <file>.partial, which becomes <file> once its size, SHA-256 and published MD5 match its pin,
and a file that the builders accept only with a source record then gets <file>.source.json. A file already present
with its pinned SHA-256 is kept. An API record (api_checks in the pins) changes between fetches, so its fields are
checked instead of its SHA-256. The script uses the standard library only; if Python cannot verify the servers'
certificates, let SSL_CERT_FILE name a CA bundle.

Usage:
    python tools/fetch_data.py                        # fetch every pinned file that is missing
    python tools/fetch_data.py --dataset tiers idlab  # only these datasets
    python tools/fetch_data.py --check [--strict]     # verify what is present and write nothing
    python tools/fetch_data.py --list
"""

import argparse
import hashlib
import http.client
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "settings/data/pins.json"
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
AGENT = "SOLID data fetch (tools/fetch_data.py)"
BLOCK = 8 << 20


def digests(path):
    """The size, SHA-256 and MD5 of a file, read once and read only."""
    sha256, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return size, sha256.hexdigest(), md5.hexdigest()


def field(record, dotted):
    """The value of a dotted field of a JSON record; '#' is the length of a list."""
    for part in dotted.split("."):
        record = len(record) if part == "#" else record[int(part) if isinstance(record, list) else part]
    return record


def problem(pin, path):
    """Why the file at path is not the pinned file, or None. API records are checked by their fields."""
    size, sha256, md5 = digests(path)
    if sha256 == pin["sha256"] and size == pin["bytes"] and pin.get("md5", md5) == md5:
        return None
    if "api_checks" not in pin:
        return f"{size} bytes, SHA-256 {sha256[:16]}, expected {pin['bytes']} bytes, {pin['sha256'][:16]}"
    try:
        record = json.loads(path.read_bytes())
        wrong = [name for name, value in pin["api_checks"].items() if field(record, name) != value]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return f"unreadable API record ({error})"
    return f"API record fields differ: {', '.join(wrong)}" if wrong else None


def sidecar(pin, path):
    """The source record that the builders expect beside the file, with the keys of the pin in their order."""
    if not pin.get("sidecar"):
        return None
    size, sha256, md5 = digests(path) if "api_checks" in pin else (pin["bytes"], pin["sha256"], pin.get("md5"))
    values = dict(url=pin["url"], bytes=size, md5=md5, sha256=sha256)
    return {key: values[key] for key in pin["sidecar"]}


def sidecar_path(path):
    return path.with_name(path.name + ".source.json")


def stored(path):
    """The source record beside a file, or None when there is none or it is unreadable."""
    try:
        return json.loads(sidecar_path(path).read_text())
    except (OSError, ValueError):
        return None


def write_sidecar(pin, path):
    """Write the source record of a verified file unless the one beside it is already that record."""
    record, target = sidecar(pin, path), sidecar_path(path)
    if record is None or stored(path) == record:
        return
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(target)


def download(pin, partial, resume):
    """Download the pinned URL into partial. A retry resumes the partial file of a pin with a known size; a new run
    starts it again, since a partial file left by an earlier run may hold anything."""
    start = partial.stat().st_size if resume and partial.exists() and "api_checks" not in pin else 0
    if start >= pin["bytes"]:
        start = 0
    headers = {"User-Agent": AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    with urllib.request.urlopen(urllib.request.Request(pin["url"], headers=headers), timeout=60) as response:
        resumed = start and response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {start}-")
        with partial.open("ab" if resumed else "wb") as stream:
            while block := response.read(BLOCK):
                stream.write(block)


def fetch(pin):
    path = DATA / pin["path"]
    if path.exists():
        reason = problem(pin, path)
        if reason:
            print(f"refused   {pin['path']}: present but {reason}; move it away and run again")
            return False
        write_sidecar(pin, path)
        print(f"present   {pin['path']}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    print(f"fetching  {pin['path']} ({pin['bytes']:,} bytes) from {urlsplit(pin['url']).netloc}", flush=True)
    for attempt in range(3):
        try:
            download(pin, partial, resume=attempt > 0)
            break
        except (OSError, http.client.HTTPException) as error:
            print(f"          attempt {attempt + 1} failed: {error}", flush=True)
    else:
        return False
    reason = problem(pin, partial)
    if reason:
        print(f"MISMATCH  {pin['path']}: download has {reason}; kept as {partial.name}")
        return False
    partial.replace(path)
    write_sidecar(pin, path)
    print(f"verified  {pin['path']}")
    return True


def check(pin):
    """The state of a pinned file: verified, missing or a mismatch, without writing anything."""
    path = DATA / pin["path"]
    if not path.exists():
        return "missing", ""
    reason = problem(pin, path)
    if reason:
        return "mismatch", reason
    expected = sidecar(pin, path)
    if expected is not None:
        if not sidecar_path(path).exists():
            return "mismatch", "no source record (fetch_data.py writes it without downloading)"
        if stored(path) != expected:
            return "mismatch", "source record differs from the pin"
    if "api_checks" in pin and digests(path)[1] != pin["sha256"]:
        return "verified", "API record fields checked, its bytes differ from the copy the paper used"
    return "verified", ""


def arguments(pins):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", choices=sorted(pins["datasets"]), help="restrict to these datasets")
    parser.add_argument("--check", action="store_true", help="verify the files present, write nothing")
    parser.add_argument("--strict", action="store_true", help="with --check, a missing file is an error too")
    parser.add_argument("--list", action="store_true", help="list the pinned files")
    return parser.parse_args()


def main():
    pins = json.loads(PINS.read_text())
    args = arguments(pins)
    selected = [pin for pin in pins["files"] if not args.dataset or pin["dataset"] in args.dataset]
    if args.list:
        for pin in selected:
            print(f"{pin['dataset']:<18} {pin['bytes']:>15,}  {pin['sha256'][:16]}  "
                  f"{urlsplit(pin['url']).netloc:<27} {pin['path']}")
        for entry in pins["not_fetched"]:
            if not args.dataset or entry["dataset"] in args.dataset:
                print(f"{entry['dataset']:<18} not fetched: {entry['what']}")
        return 0
    print(f"data folder {DATA}")
    if args.check:
        counts = dict(verified=0, missing=0, mismatch=0)
        for pin in selected:
            state, reason = check(pin)
            counts[state] += 1
            print(f"{state:<9} {pin['path']}" + (f": {reason}" if reason else ""))
            if (DATA / (pin["path"] + ".partial")).exists():
                print(f"          an unfinished {Path(pin['path']).name}.partial lies beside it")
        print(f"{counts['verified']} verified, {counts['missing']} missing, {counts['mismatch']} mismatched")
        return 1 if counts["mismatch"] or (args.strict and counts["missing"]) else 0
    results = [fetch(pin) for pin in selected]
    print(f"{sum(results)} of {len(results)} files in place")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
