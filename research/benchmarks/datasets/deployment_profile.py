"""One validated measurement profile for the retained metric-input AF method."""

import copy
import json
import math


SCHEMA = "solid.deployment-profile.v1"
METHOD = "solid-af-asymmetric-availability-v1"
INPUTS = dict(measurement="uncorrected-range", units="cm", mode="AF", samples=[1, 10], aggregation="median",
              ranging_opportunities="all pairs within observed components")


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"expected fields {', '.join(sorted(expected))}")


def number(value, minimum=None, strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("profile values must be finite numbers")
    if minimum is not None and (value < minimum or (strict and value == minimum)):
        raise ValueError("profile value is outside its allowed range")


def validate(profile):
    fields(profile, ("schema", "name", "method", "inputs", "calibration", "uncertainty", "range_error",
                     "availability", "operating_area", "provenance"))
    if profile["schema"] != SCHEMA or profile["method"] != METHOD:
        raise ValueError("unsupported profile schema or positioning method")
    if not isinstance(profile["name"], str) or not profile["name"].strip():
        raise ValueError("profile name is required")
    if (profile["inputs"] != INPUTS or any(type(n) is not int for n in profile["inputs"]["samples"])
            or profile["operating_area"] is not None):
        raise ValueError("this profile supports metric AF with one or ten samples and no area prior")
    fields(profile["calibration"], ("gain", "offset_cm"))
    number(profile["calibration"]["gain"], 0, strict=True)
    number(profile["calibration"]["offset_cm"])
    fields(profile["uncertainty"], ("constant_scale_cm",))
    number(profile["uncertainty"]["constant_scale_cm"], 0, strict=True)
    for section, keys in (("range_error", ("positive_scale", "negative_scale")), ("availability", ("intercept", "slope"))):
        fields(profile[section], ("1", "10"))
        for values in profile[section].values():
            fields(values, keys)
            for key, value in values.items():
                number(value, None if key == "intercept" else 0, strict=section == "range_error")
    fields(profile["provenance"], ("range_fit", "availability_fit", "compatibility"))
    for key in ("range_fit", "availability_fit"):
        record = profile["provenance"][key]
        if not isinstance(record, dict) or not record.get("source") or record.get("evaluation_rows_used") != 0:
            raise ValueError("profile needs training-only provenance for both fitted parts")
    if not profile["provenance"]["compatibility"]:
        raise ValueError("profile compatibility must be declared")
    return profile


def load(path):
    with path.open() as stream:
        return validate(json.load(stream))


def prepare_request(profile, raw):
    """Raw samples and prepared samples have different fields to prevent double correction."""
    validate(profile)
    fields(raw, ("baselines", "units"))
    request = copy.deepcopy(raw)
    free = {n: dict(profile["range_error"][n], **profile["availability"][n]) for n in ("1", "10")}
    request["profile"] = dict(free=free, anchored={})
    for unit in request["units"]:
        fields(unit, ("id", "samples", "problem"))
        if type(unit["samples"]) is not int or unit["samples"] not in INPUTS["samples"]:
            raise ValueError("unsupported profile sample count")
        problem = unit["problem"]
        fields(problem, ("mode", "agent_ids", "anchors", "initial_positions", "observations"))
        if problem["mode"] != "AnchorFree" or problem["anchors"]:
            raise ValueError("AF profile cannot be applied to an anchored mode")
        for link in problem["observations"]:
            fields(link, ("from", "to", "to_is_anchor", "ranges_cm"))
            if link["to_is_anchor"] or len(link["ranges_cm"]) != unit["samples"]:
                raise ValueError("measurement does not match the profile contract")
            samples = []
            for value in link.pop("ranges_cm"):
                number(value, 0)
                corrected = profile["calibration"]["gain"] * value + profile["calibration"]["offset_cm"]
                number(corrected)
                samples.append(dict(distance_cm=max(0., corrected), scale_cm=profile["uncertainty"]["constant_scale_cm"]))
            link.update(tof_ns=[], calibrated_samples=samples)
    return request
