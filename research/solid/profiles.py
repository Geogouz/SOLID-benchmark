"""Compose the SOLID profiles of every dataset from the fit records and the deployment rules, and write them to
research/artifacts/solid/profiles.

A profile holds a link model and deployment settings. The link model is the fit record's. The external link models are
the fits on the same-layout training halves written by fit_layout.py, the IDLab one the fit on its reserved training
rows, and the first stage scales of the cooperative profiles are the frozen campaign fits. The deployment settings, the
height prior of the anchor links, the grid step and the budgets, follow the rules of deployment.py from the
training-half record research/artifacts/solid/fits/deployment.json, the same rules for every dataset. The budgets of
the anchor only files are the base budgets of deployment.py scaled by the converged factor of the budget check record
research/artifacts/solid/selections/budget-check.json, by one while no record exists. The operating area of an
external unit is computed by deployment.py when the unit is solved, so these profiles hold none. The Bern profiles
are composed by compose_unified_profiles.py from the same rules.

One file per dataset and mode, the same method everywhere. The anchor only files hold the link model and the
deployment settings, the cooperative files hold the first stage scales per sample count and the block likelihood of
the link model with the median first stage and 64 starts. Every file writes every key the engine requires. The link
model carries the fit's degrees of freedom, offset shape and false early detection rate, each at the method's value
when the fit does not carry it, with coupled packets. The anchor only blocks take the pairwise risk decision, and
every anchor only block and cooperative profile holds the method's presence law, the law of the presence decision
research/artifacts/solid/selections/bern-presence-law-decision.json with the quantile rule of the reach and enabled. The
composition functions take a fit record so that design rounds build their candidates from the same code.

The method the confirmatory round confirms (research/solid/confirm_round.py) is written to its method record
research/artifacts/solid/selections/method.json, whose schema confirm_round.method_record documents. Once it exists,
every composition is the method's: the fit files of its link model variant, the link model of the reduced or joint
refit of the link components it removed, its anchor only budgets, its switches and, when it derives its first stage,
the multipliers the round derived from its link model. Every file the record names must hold the hash it records, and
every composition must equal the round's own composition of the method's spec, confirm_round.compose. Once a record of
the round exists and before the method record does, the compositions of the evaluation refuse the fits of the
pre-round method.

The runners compose the profile of their dataset and mode in memory from the same records. A tracked profile file
passed to a runner must hold that composition, so a stale tracked file is never solved.

A reader's reproduction (SOLID_READER=1) has none of the records of the round. Its method record is the method of
research/solid/method_constants.py (reader_method), its presence law the law written there, and its compositions apply
the method's one switch, the maximum likelihood reach (switched), without the round's own composition to compare with.
A tracked profile file passed to a runner is solved as written, the settings the paper's runs solved.

Usage: profiles.py [--check]
"""
import argparse
import copy
import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import method_constants  # noqa: E402

LINK_KEYS = ("kappa", "excess_probability", "excess_mean_rel", "persistent_sd_rel")
# fields of the link likelihood that a fit record may carry beyond the four parameters
CARRIED_LINK_KEYS = ("offset_shape", "early_detection_per_cm")
# the fields of the link model beyond the four parameters that every block of a profile names
CARRIED_FIELDS = ("nu",) + CARRIED_LINK_KEYS
# the degrees of freedom of the method, for a fit record that does not state the value it was fitted with
METHOD_NU = method_constants.NU
# the value of every field of the link model beyond the four parameters for a fit that does not carry it, the method's,
# since the engine requires every field in every block
LINK_DEFAULTS = dict(nu=METHOD_NU, offset_shape=method_constants.OFFSET_SHAPE, early_detection_per_cm=0.0)
STARTS = method_constants.STARTS
# the reference excess probability of the frozen nine candidate Bern anchor only round
BERN_ROUND_EXCESS = 0.10
PRESENCE_DECISION = common.SELECTIONS / "bern-presence-law-decision.json"
# the keys of the presence law the engine requires beyond those the presence decision chose, at the method's values
PRESENCE_DEFAULTS = dict(reach="quantile", enabled=True)
# the record of the method the confirmatory round confirmed, written by confirm_round.method_record, and the records of
# the round's two runs, either of which means the round has run
METHOD_RECORD = common.SELECTIONS / "method.json"
METHOD_SCHEMA = "solid.selection.method.v1"
ROUND_RECORDS = (common.SELECTIONS / "confirmatory-run-1.json", common.SELECTIONS / "confirmatory-run-2.json")
MODES = {"agent-only": "af", "anchor-aided": "aac", "anchor-only": "anc"}


def reader_method():
    """The method record of a reader's reproduction, the method of method_constants.py in the keys of the round's
    method record that the compositions read, without the hashes of the files the round pinned."""
    return dict(schema=METHOD_SCHEMA, statement="research/solid/method_constants.py", spec=copy.deepcopy(method_constants.SPEC),
                fits=dict(method_constants.FITS), reduced=None, budget=dict(factor=method_constants.BUDGET),
                first_stage=dict(derived=False, multipliers=None, links=None), complete=True)


def method_record():
    """The record of the method the confirmatory round confirmed, None while the round has not run. Refused when a
    record of the round exists without it, so that no evaluation composes the pre-round method, and when a file it
    names no longer holds the hash it records. A reader's reproduction takes the method of method_constants.py."""
    if common.READER:
        return reader_method()
    if not METHOD_RECORD.exists():
        if any(path.exists() for path in ROUND_RECORDS):
            raise ValueError(f"The confirmatory round has run, so the evaluation composes the method of {METHOD_RECORD}, which is missing. "
                             "Decide Run II with confirm_round.py decide-2")
        return None
    record = common.read_json(METHOD_RECORD)
    if record.get("schema") != METHOD_SCHEMA or not record.get("complete"):
        raise ValueError(f"{METHOD_RECORD} is not a complete method record of schema {METHOD_SCHEMA}")
    changed = [path for path, digest in record["files"].items() if not (common.ROOT / path).exists() or common.sha256(common.ROOT / path) != digest]
    if changed:
        raise ValueError(f"{len(changed)} files of the method record changed since the round wrote it, for example {', '.join(changed[:4])}")
    return record


def method_provenance():
    """What a run record keeps of the method it solves: the method record and its hash, none before the round. A
    reader's run names the constants of its method instead, with no record."""
    if common.READER:
        return dict(record=None, sha256=None, constants="research/solid/method_constants.py")
    if not METHOD_RECORD.exists():
        return None
    return dict(record=str(METHOD_RECORD.relative_to(common.ROOT)) if METHOD_RECORD.is_relative_to(common.ROOT) else str(METHOD_RECORD),
                sha256=common.sha256(METHOD_RECORD))


def pre_round_fit(name):
    """A tracked fit of the method as it stood before the confirmatory round, refused once a record of the round
    exists, since the evaluation then composes the method the round confirmed."""
    if any(path.exists() for path in ROUND_RECORDS):
        raise ValueError(f"The confirmatory round has run, so no evaluation composes the pre-round fit {name}. Compose from {METHOD_RECORD}")
    return common.read_json(common.FITS / name)


def campaign_first_stage():
    """Whether the cooperative profiles of the method take the frozen campaign fits as their first stage, as they do
    unless the method record derives the first stage from the link model."""
    method = method_record()
    return method is None or not method["first_stage"]["derived"]


def presence_law():
    """The method's presence law written out in full: the law of the presence decision, with the quantile rule of the
    reach and the law enabled, every key the engine requires. A reader's reproduction takes the law of the decision
    from method_constants.py."""
    law = dict(method_constants.PRESENCE_LAW) if common.READER else dict(common.read_json(PRESENCE_DECISION)["law"])
    law.update({key: value for key, value in PRESENCE_DEFAULTS.items() if key not in law})
    return law


def switched(profile, mode, changes):
    """The switches of the method applied to a profile in place, as confirm_round.apply_changes applies them, for a
    reader's reproduction, which composes without the round. The method has one switch, the reach replacement
    ml_reach, which sets the reach of every presence law of the profile, those of the anchor only blocks or of every
    sample count, to the maximum likelihood rule. Any other switch is refused."""
    others = sorted(set(changes) - {"ml_reach"})
    if others:
        raise ValueError(f"a reader's composition applies the reach replacement of the method alone, not {others}")
    if "ml_reach" in changes:
        if mode == "anc":
            holders = [entry["anchor_only"] for entry in profile["environments"].values()] if "environments" in profile else [profile["anchor_only"]]
        else:
            holders = [profile[n] for n in ("1", "10")]
        for holder in holders:
            holder["presence"] = dict(holder.get("presence") or {}, reach=method_constants.REACH)
    return profile


def carried_link_fields(fit):
    """The fields of the link model beyond the four parameters, the degrees of freedom, the offset shape and the false
    early detection rate, each the fit's or, for a fit that does not carry it, the method's, since the engine requires
    every field (select_bern_cooperative.carried_link_fields)."""
    return {key: fit.get(key, LINK_DEFAULTS[key]) for key in CARRIED_FIELDS}


def link_parameters(fit, record=None):
    """The link model of a fit: the four parameters, the degrees of freedom, the offset shape and the false early
    detection rate. A field the fit lacks is taken from its record, and one that neither carries takes the method's
    value."""
    source = dict(record or {}, **fit)
    link = {key: fit[key] for key in LINK_KEYS}
    link.update({key: source.get(key, default) for key, default in LINK_DEFAULTS.items()})
    link["nu"] = float(link["nu"])
    return link


def finite_numbers(values):
    """Whether every value is a finite number. A fit record keeps a value that is not finite as null
    (fit_links._clean), and the profile composed from it keeps the null."""
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values)


def finite_link(block):
    """Whether a fit or the anchor only block of a profile holds a finite number in every parameter of the link model it
    takes from a fit. The held-out runner solves no unit with a link model that does not, and every such solve is
    scored as SOLID's failure (the held-out declaration's rules.outcomes)."""
    return finite_numbers([block.get(key) for key in LINK_KEYS] + [block.get("early_detection_per_cm", LINK_DEFAULTS["early_detection_per_cm"])])


def anchor_only(fit, record=None, **settings):
    """An anchor only block made of the link model of a fit with coupled packets, the pairwise risk decision and the
    method's presence law, and the deployment settings."""
    block = dict(link_parameters(fit, record), independent_packets=False, decision="risk", presence=presence_law())
    block.update(settings)
    return block


# The engine's own law for raw timings, the production calibration of TWR-CLOUD-Bern, CalibrationModel's default in
# runtime/engine/src/calibration.rs. The cooperative profiles of TIERS and IDLab, whose inputs are calibrated samples
# that never read it, name it as well, so no profile leaves the engine to fill it in.
ENGINE_CALIBRATION = dict(method_constants.CALIBRATION)
# the coefficients of that law, which every cooperative profile names
CALIBRATION_KEYS = tuple(ENGINE_CALIBRATION)
# the keys of a cooperative profile that the engine takes as optional, filling in the median residual density and the
# TWR-CLOUD-Bern law when absent, and that every cooperative profile of the method names
COOPERATIVE_KEYS = ("block_range", "calibration")


def complete_cooperative(profile, where):
    """A cooperative profile of one dataset and mode, {track: profile}, refused unless every track names the block
    likelihood and the calibration (COOPERATIVE_KEYS), the calibration with every coefficient of the engine's law, so
    that no composition leaves the engine to fill either in from its defaults. Returns the profile."""
    for track, entry in profile.items():
        missing = [key for key in COOPERATIVE_KEYS if not isinstance(entry.get(key), dict)]
        if missing:
            raise ValueError(f"{where} names no {' and no '.join(missing)} on track {track}, which the engine would fill in from its defaults")
        if set(entry["calibration"]) != set(ENGINE_CALIBRATION):
            raise ValueError(f"{where} names a calibration on track {track} with the coefficients {sorted(entry['calibration'])}, not those of the "
                             f"engine's law {sorted(ENGINE_CALIBRATION)}")
    return profile


def cooperative(range_error, block=None, starts=STARTS):
    """A cooperative profile: the first stage scales, the method's presence law and, when a link model is given, the
    block likelihood of the method with the median first stage and the multistart of the adopted design."""
    profile = dict(range_error=dict(positive_scale=range_error["positive_scale"], negative_scale=range_error["negative_scale"]),
                   presence=presence_law())
    if block is not None:
        profile.update(block_range=dict(block), starts=starts)
    return profile


def block_range(fit, record=None, *, height_nodes, height_weights, **settings):
    """The block likelihood settings of a cooperative profile from a link fit: the link model with coupled packets, the
    height prior of the anchor links and the median first stage."""
    block = dict(link_parameters(fit, record), height_nodes=height_nodes, height_weights=height_weights, refine_from_median=True,
                 independent_packets=False)
    block.update(settings)
    return block


def heights(settings):
    return dict(height_nodes=settings["height_nodes"], height_weights=settings["height_weights"])


def bern_anchor_only(fit, change, held):
    """An anchor only block of the frozen nine candidate Bern round: the link model of a fold fit with the round's
    reference excess probability and the Bern deployment settings of the fold of `held`, the held scan the profile is
    solved on, with one candidate change applied. The fold has no default, so no held scan takes the settings of every
    scan, its own heights among them."""
    settings = dict(deployment.settings("bern", held=held), excess_probability=BERN_ROUND_EXCESS)
    if change == "fitted":
        settings.pop("excess_probability")
    else:
        settings.update(change)
    return anchor_only(fit, **settings)


def bern():
    """The unified Bern profiles: one link model for the three modes, composed by compose_unified_profiles.py from the
    method record once the round has confirmed the method, and before it from the tracked mixed fit, the production
    cooperative fit, the frozen selection records and the deployment rules."""
    import compose_unified_profiles
    profiles, _ = compose_unified_profiles.unified_profiles()
    yield from profiles.items()


def tiers(record=None, budget=None):
    """The three TIERS profiles from a layout fit record, the tracked pre-round one by default, with the anchor only
    budgets of the factor `budget`, by default the converged factor of the budget check record."""
    if record is None:
        record = pre_round_fit("tiers_layout_fit.json")
    base = json.load(gzip.open(common.EXTERNAL / "tiers-static-campaign/base.json.gz"))
    fit = record["four"]
    placement = deployment.settings("tiers")
    yield "tiers-anchor-only.json", dict(anchor_only=anchor_only(fit, record, **deployment.budgeted(placement, budget)))
    block = block_range(fit, record, **heights(placement))
    free = base["settings"]["calibrated-AF"]["profile"]["free"]
    yield "tiers-agent-only.json", complete_cooperative({samples: dict(cooperative(free[samples], block), calibration=dict(ENGINE_CALIBRATION))
                                                         for samples in ("1", "10")}, "tiers-agent-only.json")
    anchored = base["settings"]["calibrated-AAC"]["profile"]["anchored"]
    yield "tiers-anchor-aided.json", complete_cooperative({samples: dict(cooperative(anchored[samples]["2"], block), calibration=dict(ENGINE_CALIBRATION))
                                                           for samples in ("1", "10")}, "tiers-anchor-aided.json")


# the base of the IDLab training campaign, whose settings hold the frozen first stage the evaluation campaign holds too
IDLAB_TRAINING_BASE = "idlab-training-campaign/base.json.gz"


def idlab(record=None, budget=None):
    """The three IDLab profiles from the link fit on the reserved training rows, with the anchor only budgets of the
    factor `budget`, by default the converged factor of the budget check record. The first stage multipliers come from
    the settings of the training campaign's base, which idlab_training_population.py copies from the evaluation
    campaign's, so no evaluation measurement is read."""
    if record is None:
        record = pre_round_fit("idlab_link_fit.json")["b10"]
    base = json.load(gzip.open(common.EXTERNAL / IDLAB_TRAINING_BASE))
    fit = record["four"]
    placement = deployment.settings("idlab")
    yield "idlab-anchor-only.json", dict(anchor_only=anchor_only(fit, record, **deployment.budgeted(placement, budget)))
    block = block_range(fit, record, **heights(placement))
    af, aac = base["settings"]["AF"]["profile"], base["settings"]["AAC"]["profile"]
    yield "idlab-agent-only.json", complete_cooperative({samples: dict(cooperative(af["range_error"], block), calibration=dict(ENGINE_CALIBRATION))
                                                         for samples in ("1", "10")}, "idlab-agent-only.json")
    yield "idlab-anchor-aided.json", complete_cooperative({samples: dict(cooperative(aac["range_error"], block), calibration=dict(ENGINE_CALIBRATION))
                                                           for samples in ("1", "10")}, "idlab-anchor-aided.json")


def bregar(records=None, budget=None):
    """The Bregar anchor only profile, one link model and one set of deployment settings per environment, with the
    budgets of the factor `budget`, by default the converged factor of the budget check record."""
    if records is None:
        records = pre_round_fit("bregar_layout_fits.json")["environments"]
    environments = {}
    for environment, record in records.items():
        placement = deployment.budgeted(deployment.settings("bregar", environment), budget)
        environments[environment] = dict(anchor_only=anchor_only(record["four"], record, **placement))
    yield "bregar-anchor-only.json", dict(environments=environments)


COMPOSITIONS = dict(tiers=tiers, idlab=idlab, bregar=bregar)


def method_fit(dataset, method):
    """The fit record of an external deployment in the method: the record of the fit file of the method's link model,
    with the link model of the reduced or joint refit of the link components it removed."""
    data = common.read_json(common.FITS / method["fits"][dataset])
    record = data["b10"] if dataset == "idlab" else data["environments"] if dataset == "bregar" else data
    reduced = method["reduced"]
    if not reduced:
        return record
    stored = common.read_json(common.FITS / reduced["files"][dataset])
    if dataset == "bregar":
        return {environment: dict(entry, four=stored["environments"][environment][reduced["key"]]["plain"]) for environment, entry in record.items()}
    if dataset == "idlab":
        return dict(record, four=stored["b10"][reduced["key"]]["plain"])
    return dict(record, four=stored[reduced["key"]]["plain"])


def method_profiles(dataset, method):
    """The profiles of an external deployment in the method, from its record: the composition of its fit with the
    method's budgets, its switches and, when it derives its first stage, the multipliers the round derived. Each must
    equal the round's own composition of the method's spec, confirm_round.compose, and is refused otherwise. A
    reader's reproduction applies the method's switch alone (switched), since its method derives no first stage and it
    has no round to compare with."""
    spec, first_stage = method["spec"], method["first_stage"]
    if common.READER:
        for name, profile in COMPOSITIONS[dataset](method_fit(dataset, method), budget=method["budget"]["factor"]):
            mode = MODES[name.split("-", 1)[1].removesuffix(".json")]
            profile = switched(profile, mode, spec["changes"])
            yield name, profile if mode == "anc" else complete_cooperative(profile, name)
        return
    import confirm_round
    recorded = lambda name, fold, suffix, profile: first_stage["multipliers"][name]  # noqa: E731
    for name, profile in COMPOSITIONS[dataset](method_fit(dataset, method), budget=method["budget"]["factor"]):
        mode = MODES[name.split("-", 1)[1].removesuffix(".json")]
        profile = confirm_round.apply_changes(profile, mode, spec["changes"])
        if first_stage["derived"] and mode in confirm_round.REPLACEMENTS["derived_first_stage"]:
            for samples in ("1", "10"):
                profile[samples]["range_error"] = dict(first_stage["multipliers"][dataset][samples])
        differing = confirm_round.differences(confirm_round.compose(dataset, mode, spec, None, recorded), profile)
        if differing:
            raise ValueError(f"The composition of {name} from the method record differs from the round's composition of its spec in "
                             f"{len(differing)} fields, for example {['.'.join(path) for path in differing[:4]]}")
        yield name, profile if mode == "anc" else complete_cooperative(profile, name)


def evaluation(dataset):
    """The profiles of an external deployment the evaluation solves: the method's from its record once the round has
    confirmed it, the pre-round method's before the round."""
    method = method_record()
    if method is None:
        yield from COMPOSITIONS[dataset]()
    else:
        yield from method_profiles(dataset, method)


SOURCES = dict(bern=bern, tiers=lambda: evaluation("tiers"), idlab=lambda: evaluation("idlab"), bregar=lambda: evaluation("bregar"))


def compose(name):
    """The composition of one tracked profile file from the fit records and the deployment record."""
    dataset = name.split("-", 1)[0]
    if dataset not in SOURCES:
        raise ValueError(f"no composition for the profile {name}")
    for candidate, profile in SOURCES[dataset]():
        if candidate == name:
            return profile
    raise ValueError(f"no composition for the profile {name}")


def resolve(name, path=None):
    """The profile a runner solves for the tracked profile `name`, and its source. Without a path it is the composition
    of the fit records. A path inside research/artifacts/solid/profiles must be the tracked file `name` and hold its
    composition, so a stale tracked file is refused, and a reader's reproduction solves the tracked file as written.
    Any other path, a round's candidate for example, is solved as written."""
    if path is None:
        return compose(name), f"{name} composed from the fit records and the deployment record"
    path = Path(path)
    if common.tracked_profile(path):
        if path.name != name:
            raise ValueError(f"{path} is not the tracked profile {name} of this dataset and mode")
        if common.READER:
            return common.read_json(path), str(path.resolve().relative_to(common.ROOT))
        return common.verify_profile(path, compose(name)), str(path.resolve().relative_to(common.ROOT))
    return common.read_json(path), str(path.resolve().relative_to(common.ROOT)) if path.resolve().is_relative_to(common.ROOT) else str(path)


def provenance(directory, profile, source):
    """Write the solved profile beside a run's predictions and return the run record's profile fields, the deployment
    record and the method record among them."""
    path = Path(directory) / "profile.json"
    common.write_json(path, profile)
    return dict(profile=source, profile_file=path.name, profile_sha256=common.sha256(path), profile_fingerprint=common.fingerprint(profile),
                deployment_record_sha256=common.sha256(deployment.RECORD), method_record=method_provenance())


# ---------------------------------------------------------------------------------------------------------------------
# Held-out benchmarks
#
# HYMN and AAU Meshed UWB, declared in research/artifacts/solid/splits/heldout-benchmarks.json, take the anchor only
# profile of the method the confirmatory round confirmed, composed from its method record like every evaluation. The
# link model is the fit on their training half with the method's variant and removals (research/solid/fit_heldout.py
# links), the deployment settings those of the held-out deployment record, the budgets the method's factor of them and
# the switches the method's own. They stay out of SOURCES, so the tracked profiles above, their check and their tests
# never read a held-out dataset.

def heldout_method():
    """The method record the held-out profiles compose, refused while the round has not confirmed a method."""
    method = method_record()
    if method is None:
        raise ValueError(f"The held-out benchmarks take the method the confirmatory round confirmed, and {METHOD_RECORD} is missing. "
                         "Decide Run II with confirm_round.py decide-2")
    return method


def heldout_fit(dataset, method=None):
    """The tracked link fit of a held-out dataset, refused unless it was fitted on that dataset's training half under the
    current declaration for the method of the method record, its spec and the record's SHA-256. A fit that did not
    converge, or holds a value that is not finite, is the method's link model as it stands, the declared outcome of the
    held-out declaration's rules.outcomes, and the tables mark it (heldout_runner.marks). A reader's reproduction,
    which has no method record, checks the method's spec alone."""
    import heldout
    method = method or heldout_method()
    name = f"{dataset}_layout_fit.json"
    record = common.read_json(common.FITS / name)
    identity = record.get("identity", {})
    if (identity.get("dataset"), identity.get("split"), identity.get("declaration_sha256")) != (dataset, "training", common.sha256(heldout.RECORD)):
        raise ValueError(f"{name} was not fitted on the {dataset} training half under this declaration")
    fitted = record.get("method") or {}
    if fitted.get("spec") != method["spec"] or (not common.READER and fitted.get("sha256") != common.sha256(METHOD_RECORD)):
        raise ValueError(f"{name} was fitted for another method than the one of {METHOD_RECORD.name}, spec {fitted.get('spec')} instead of "
                         f"{method['spec']}. Fit it again with research/solid/fit_heldout.py links")
    return record


def heldout_profile(dataset, record=None, method=None):
    """The anchor only profile of a held-out dataset in the method, the link model of its layout fit record with the
    held-out deployment settings at the method's budget factor and the method's switches of the anchor only mode
    (confirm_round.apply_changes, and switched in a reader's reproduction)."""
    method = method or heldout_method()
    record = record or heldout_fit(dataset, method)
    settings = deployment.budgeted(deployment.heldout_settings(dataset), method["budget"]["factor"])
    profile = dict(anchor_only=anchor_only(record["four"], record, **settings))
    if common.READER:
        return switched(profile, "anc", method["spec"]["changes"])
    import confirm_round
    return confirm_round.apply_changes(profile, "anc", method["spec"]["changes"])


def hymn(record=None, method=None):
    """The HYMN anchor only profile of the method from its layout fit record and the held-out deployment record."""
    yield "hymn-anchor-only.json", heldout_profile("hymn", record, method)


def aau(record=None, method=None):
    """The AAU anchor only profile of the method from its layout fit record and the held-out deployment record."""
    yield "aau-anchor-only.json", heldout_profile("aau", record, method)


HELDOUT_SOURCES = dict(hymn=hymn, aau=aau)


def heldout_compose(name):
    """The composition of one held-out profile file from its fit record and the held-out deployment record."""
    dataset = name.split("-", 1)[0]
    if dataset not in HELDOUT_SOURCES:
        raise ValueError(f"no held-out composition for the profile {name}")
    for candidate, profile in HELDOUT_SOURCES[dataset]():
        if candidate == name:
            return profile
    raise ValueError(f"no held-out composition for the profile {name}")


def heldout_resolve(name, path=None):
    """The held-out profile a runner solves, with its source, under the rules of resolve, so a reader's reproduction
    solves the tracked file as written."""
    if path is None:
        return heldout_compose(name), f"{name} composed from the held-out fit and deployment records"
    path = Path(path)
    if common.tracked_profile(path):
        if path.name != name:
            raise ValueError(f"{path} is not the tracked profile {name} of this dataset")
        if common.READER:
            return common.read_json(path), str(path.resolve().relative_to(common.ROOT))
        return common.verify_profile(path, heldout_compose(name)), str(path.resolve().relative_to(common.ROOT))
    return common.read_json(path), str(path.resolve().relative_to(common.ROOT)) if path.resolve().is_relative_to(common.ROOT) else str(path)


def heldout_provenance(directory, profile, source):
    """Write the solved profile beside a held-out run's predictions and return the run record's profile fields, the
    held-out deployment record and the method record among them."""
    path = Path(directory) / "profile.json"
    common.write_json(path, profile)
    return dict(profile=source, profile_file=path.name, profile_sha256=common.sha256(path), profile_fingerprint=common.fingerprint(profile),
                heldout_deployment_record_sha256=common.sha256(deployment.HELDOUT_RECORD), method_record=method_provenance())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify that every tracked profile holds its composition, write nothing")
    args = parser.parse_args()
    stale = []
    for dataset, source in SOURCES.items():
        composed = source()
        if args.check:
            # a dataset whose profiles cannot be composed, its campaign or record missing or refused, is reported stale too
            try:
                composed = list(composed)
            except (OSError, ValueError) as error:
                stale.append(f"The {dataset} profiles could not be composed: {error}")
                continue
        for name, profile in composed:
            path = common.PROFILES / name
            if args.check:
                try:
                    common.verify_profile(path, profile)
                    print(f"current   {path.relative_to(common.ROOT)}")
                except (OSError, ValueError) as error:
                    stale.append(str(error))
                continue
            previous = common.read_json(path) if path.exists() else None
            common.write_json(path, profile)
            print(f"{'unchanged' if previous == profile else 'written  '} {path.relative_to(common.ROOT)}")
    if stale:
        raise SystemExit("\n".join(stale))


if __name__ == "__main__":
    main()
