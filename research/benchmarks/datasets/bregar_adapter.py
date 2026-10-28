"""Prepare common Bregar inputs and check adapters on training examples only."""

from collections import defaultdict
import json
import os
import subprocess
import zipfile

import numpy as np

import bregar
import pilot

OUTPUT = bregar.audit.DATA / "bregar-adapter"
BINARY = pilot.ROOT / "research/target/release/bregar-adapter"


def calibrated(values, offset):
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or not np.isfinite(offset):
        raise ValueError("invalid raw readings or offset")
    result = raw - offset
    if not np.isfinite(result).all():
        raise ValueError("nonfinite calibrated readings")
    return result


def fit_profiles(cells, fitted):
    """Equal weight per eligible training link within each radio channel."""
    offsets = {(r["anchor"], r["channel"]): r["offset_m"] for r in fitted["corrections"]}
    if set(offsets) != {(f"A{i}", ch) for i in range(1, 9) for ch in bregar.CHANNELS}:
        raise ValueError("incomplete fitted offsets")
    environments = bregar.training_environments(fitted)
    channels, failures = defaultdict(list), []
    for row in sorted(cells, key=lambda r: r["member"]):
        if row["environment"] not in environments or (row["environment"], row["anchor"]) == ("environment2", "A6"):
            continue
        corrected = calibrated(row["ranges_m"], offsets[row["anchor"], row["channel"]])
        accepted = corrected >= 0
        if not np.isfinite(row["distance_m"]) or row["distance_m"] < 0:
            raise ValueError("invalid training reference")
        residual = 100 * (corrected[accepted] - row["distance_m"])
        channels[row["channel"]].append(dict(reports=len(corrected), accepted=int(accepted.sum()),
            zero=int(np.sum(corrected == 0)), mse=float(np.mean(residual ** 2)) if len(residual) else None))
        if not accepted.all():
            failures.append(dict(member=row["member"], source_lines=(np.flatnonzero(~accepted) + 3).tolist(),
                raw_m=np.asarray(row["ranges_m"])[~accepted].tolist(), corrected_m=corrected[~accepted].tolist()))
    profiles = []
    for ch in bregar.CHANNELS:
        rows = channels[ch]
        errors = [r["mse"] for r in rows if r["mse"] is not None]
        scale = float(np.sqrt(np.mean(errors))) if errors else 0.
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("no positive training range uncertainty")
        profiles.append(dict(channel=ch, scale_cm=scale, training_links=len(rows),
            accepted_links=len(errors), training_reports=sum(r["reports"] for r in rows),
            failed_reports=sum(r["reports"] - r["accepted"] for r in rows), zero_reports=sum(r["zero"] for r in rows)))
    nested = {"outer_held_environment": fitted["outer_held_environment"]} if "outer_held_environment" in fitted else {}
    return dict(**nested, held_environment=fitted["held_environment"], training_environments=sorted(environments),
        profiles=profiles, failed_training_readings=failures, held_environment_rows_used_for_fitting=0)


def construct(anchors_cm, streams, offsets, scale_cm, sid, block=0):
    """Streams contain numeric IDs and raw ranges, never unknown coordinates or LOS labels."""
    anchors_cm = np.asarray(anchors_cm, dtype=float)
    if block not in (0, 1, 2):
        raise ValueError("unexpected fixed raw block")
    start = 10 * block
    if anchors_cm.shape != (8, 3) or not np.isfinite(anchors_cm).all() or not np.isfinite(scale_cm) or scale_cm <= 0:
        raise ValueError("invalid anchors or training scale")
    agents = sorted(streams)
    anchor_ids = sorted({anchor for rows in streams.values() for anchor in rows})
    if not agents or len(anchor_ids) not in (7, 8) or any(a not in range(1, 9) for a in anchor_ids):
        raise ValueError("unexpected derived ANC population")
    if any(set(rows) != set(anchor_ids) for rows in streams.values()):
        raise ValueError("missing declared raw link")
    anchors = [dict(id=a + 100, position=dict(x=float(anchors_cm[a - 1, 0]), y=float(anchors_cm[a - 1, 1])),
                    height_cm=float(anchors_cm[a - 1, 2])) for a in anchor_ids]
    center = np.mean(anchors_cm[np.array(anchor_ids) - 1, :2], axis=0)
    # Both sample counts have identical starts derived from known anchors only.
    radius = max(float(np.max(np.linalg.norm(anchors_cm[np.array(anchor_ids) - 1, :2] - center, axis=1))), 1.)
    initial = np.random.default_rng(20260920 + sid).uniform(-radius, radius, (len(agents), 2)) + center
    units, records = [], []
    for samples in (1, 10):
        observations, links = [], []
        for agent in agents:
            for anchor, raw in sorted(streams[agent].items()):
                if len(raw) and len(raw) < start + 10:
                    raise ValueError("incomplete fixed raw block")
                selected = np.asarray(raw[start:start + samples], dtype=float)
                corrected = calibrated(selected, offsets[anchor])
                accepted = corrected >= 0
                links.append(dict(agent=agent, anchor=anchor, source_lines=list(range(3 + start, 3 + start + len(selected))),
                    raw_m=selected.tolist(), corrected_m=corrected.tolist(),
                    failed_lines=(np.flatnonzero(~accepted) + 3 + start).tolist()))
                if accepted.any():
                    observations.append(dict(**{"from": agent, "to": anchor + 100}, to_is_anchor=True,
                        tof_ns=[], calibrated_samples=[dict(distance_cm=float(d * 100), scale_cm=scale_cm)
                                                      for d in corrected[accepted]]))
        units.append(dict(id=sid, samples=samples, scale_cm=scale_cm, problem=dict(mode="AnchorOnly",
            agent_ids=agents, anchors=anchors, initial_positions=[dict(x=float(x), y=float(y)) for x, y in initial],
            observations=observations)))
        records.append(dict(id=sid, samples=samples, links=links))
    return units, records


def check_outputs(request, rows, methods=bregar.LABELS):
    """The solve counts of every method, refused unless every unit holds one output of every method of the roster, by
    default the frozen comparison's, and every output keeps the unit's agents."""
    units = {(u["id"], u["samples"]): u for u in request["units"]}
    expected = {(sid, n, method) for sid, n in units for method in methods}
    keys = [(r["id"], r["samples"], r["method"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete adapter method population")
    counts = defaultdict(lambda: dict(solves=0, failed_solves=0, agent_attempts=0, failed_agents=0))
    for row in rows:
        agents = units[row["id"], row["samples"]]["problem"]["agent_ids"]
        if row["agent_ids"] != agents:
            raise ValueError("adapter dropped a requested agent")
        estimate = row["estimate"]
        valid = 0
        if estimate is not None:
            if estimate["agent_ids"] != agents or len(estimate["positions"]) != len(agents):
                raise ValueError("solver changed the agent population")
            valid = sum(p is not None and all(p.get(k) is not None and np.isfinite(p[k]) for k in ("x", "y"))
                        for p in estimate["positions"])
        count = counts[row["method"]]
        count["solves"] += 1
        count["failed_solves"] += int(valid != len(agents))
        count["agent_attempts"] += len(agents)
        count["failed_agents"] += len(agents) - valid
    return dict(counts)


def run():
    if OUTPUT.exists():
        raise ValueError("Bregar adapter check already prepared")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=pilot.ROOT, text=True).strip():
        raise ValueError("commit the adapter and protocol before fitting")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pilot.ROOT, text=True).strip()
    fitted, checked = (pilot.evidence.read(p) for p in (bregar.CALIBRATION, bregar.RAW_AUDIT))
    sources = {str(p): pilot.identity(p) for p in (bregar.CALIBRATION, bregar.RAW_AUDIT, bregar.SPLIT)}
    for path, digest in ((bregar.ARCHIVE, fitted["archive_sha256"]), (bregar.RAW_AUDIT, fitted["raw_audit_sha256"]),
                         (bregar.SPLIT, fitted["split_sha256"])):
        pilot.evidence.official_scorer.require_hash(path, digest, "Bregar adapter source")
    baselines, settings_sources = bregar.baseline_settings()
    sources.update(settings_sources)
    requests, metadata, profiles = [], [], []
    with zipfile.ZipFile(bregar.ARCHIVE) as archive:
        cells = bregar.calibration_cells(archive, checked)
        for index, fit in enumerate(fitted["folds"]):
            profile = fit_profiles(cells, fit)
            profiles.append(profile)
            environment = f"environment{(index + 1) % 4}"
            if fit["held_environment"] != f"environment{index}" or environment not in fit["training_environments"]:
                raise ValueError("unexpected adapter training split")
            selected = [r for r in cells if r["environment"] == environment and r["channel"] == 1]
            positions = sorted({r["position"] for r in selected})
            identities = {p: i + 1 for i, p in enumerate(positions)}
            streams, members = defaultdict(dict), []
            for row in selected:
                agent, anchor = identities[row["position"]], int(row["anchor"][1:])
                streams[agent][anchor] = row["ranges_m"]
                members.append(dict(agent=agent, anchor=anchor, member=row["member"]))
            offsets = {int(r["anchor"][1:]): r["offset_m"] for r in fit["corrections"] if r["channel"] == 1}
            anchors = bregar.coordinates(archive.read(f"data_set/{environment}/anchors.csv"), True) * 100
            units, records = construct(anchors, streams, offsets, profile["profiles"][0]["scale_cm"], index + 1)
            for unit in units:
                unit["strong_radius_cm"] = unit["scale_cm"]
            requests.extend(units)
            metadata.append(dict(id=index + 1, held_environment=fit["held_environment"],
                input_environment=environment, channel=1, block=0, members=members, selections=records))
            print(f"Fitted {fit['held_environment']} uncertainty from training environments only", flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLID_")}
    subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", "research/Cargo.toml",
                    "-p", "solid-cli", "--bin", "bregar-adapter"], cwd=pilot.ROOT, env=env, check=True)
    if subprocess.check_output([str(BINARY), "--identity"], text=True).strip() != pilot.isolation.source_identity():
        raise ValueError("Bregar binary identity mismatch")
    request = dict(baselines=baselines, units=requests)
    pilot.evidence.write(OUTPUT / "request.json", request)
    preparation = dict(source_revision=revision, binary_sha256=pilot.identity(BINARY), sources=sources,
        source_hashes={str(p): pilot.identity(p) for p in (pilot.HERE / "bregar_adapter.py", pilot.HERE / "bregar_predict.rs",
            pilot.HERE / "README.md", pilot.ROOT / "research/cli/Cargo.toml")},
        request_sha256=pilot.identity(OUTPUT / "request.json"), profiles=profiles, inputs=metadata,
        scope="Training-only adapter check. No held-environment inference or accuracy ranking.",
        failure_rule="Corrected range below zero is failed. Zero is valid. No clamping or refill.",
        uncertainty="One RMS scale per channel, equal weight per accepted training link, all LOS and NLOS included.",
        method_settings="Existing spatial solvers. STRONG uses one fitted RMS radius for this check, not a tuned optimum.",
        missingness_model_used=False, evaluation_solves=0, final_execution_frozen=False)
    pilot.evidence.write(OUTPUT / "preparation.json", preparation)
    subprocess.run([str(BINARY), str(OUTPUT / "request.json"), str(OUTPUT / "predictions.jsonl")],
                   cwd=pilot.ROOT, env=env, check=True)
    records = [json.loads(line) for line in (OUTPUT / "predictions.jsonl").read_text().splitlines()]
    methods = check_outputs(request, records)
    result = dict(source_revision=revision, methods=methods, adapter_units=len(requests),
        training_solves=len(records), evaluation_solves=0, accuracy_results_computed=0,
        files={name: pilot.identity(OUTPUT / name) for name in ("preparation.json", "request.json", "predictions.jsonl")})
    pilot.evidence.write(OUTPUT / "check.json", result)
    pilot.register_files([OUTPUT / name for name in (*result["files"], "check.json")])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
