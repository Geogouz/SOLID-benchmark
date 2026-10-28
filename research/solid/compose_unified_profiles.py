"""Compose the single Bern profile of the unified SOLID from tracked fits, frozen selection records and the deployment
rules.

One link model for the three modes: the four link parameters of the mixed block fit on all eleven training scans (both
link types, one and ten packet blocks) at the degrees of freedom of the degrees-of-freedom decision, with the fit's
offset shape and false early detection rate, the method's values when it has none, the fitted excess probability as
selected jointly, the two stage start and 64 starts in the cooperative solver as adopted by the held-scan rounds, the
two multipliers of the production cooperative fit for the first stage as kept by the first-stage round, the calibration
of the mixed fit, the engine's law of raw timings, and the method's presence law of the presence decision in every mode,
written out with every key the engine requires, like the coupled packets and the pairwise risk decision of the anchor
only block. Every cooperative profile names its block likelihood and its calibration and is refused otherwise
(profiles.complete_cooperative). The deployment settings follow the rules of deployment.py on the eleven training
scans, the same rules as every other dataset. They are the operating area (the bounding box of the training scans' node
positions, one area for the whole deployment as the harness applies it), the height prior of the unknown nodes' heights
and the grid step and budgets, the budgets scaled by the converged factor of the budget check record
research/artifacts/solid/selections/budget-check.json. Nothing in the profiles depends on the mode beyond the inputs each
mode has.

Once the confirmatory round has confirmed the method, the profiles are the method's, composed from its record
research/artifacts/solid/selections/method.json (research/solid/confirm_round.py method_record). The link model is the
method's on all eleven training scans, the all-scan fit of its variant or the all-scan record of the reduced or joint
refit of the link components it removed, with the calibration of that fit, the engine's own law of raw timings. The
deployment settings follow the same rules with the method's budgets, and the method's switches and, when it derives
its first stage, the multipliers the round derived from this link model complete it. Every profile must agree field by
field with the round's own composition of the method's spec on every held scan, the values that belong to a fold aside,
and is refused otherwise. The provenance record states the settings the profiles run, read from the profiles after the
spec's switches, its budgets and the derived first stage (effective_settings), so a presence law the spec switches off
is recorded off and a reach the spec replaces is recorded replaced. The settings of the deployment rules before the
switches keep a field of their own. Once a record of the round exists and before the method record does, the
composition refuses the pre-round fits.

A reader's reproduction (SOLID_READER=1) composes the method of research/solid/method_constants.py
(profiles.method_record) from the shipped all-scan fit, the production cooperative fit and the deployment record, with
the method's switch applied by profiles.switched and no round to compare with, and writes the provenance record under
out/solid, since it holds no design records.

Writes research/artifacts/solid/profiles/bern-{anchor-only,agent-only,anchor-aided}.json and the provenance record
research/artifacts/solid/selections/bern-unified-profile.json. Refuses to run unless every record is complete.

Usage: compose_unified_profiles.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import profiles as compositions  # noqa: E402

STARTS = compositions.STARTS
RECORDS = ("bern-anchor-only", "bern-cooperative-block", "bern-cooperative-block-af-round2", "bern-cooperative-block-aac-starts64",
           "bern-ablation-cooperative", "bern-ablation-warm-start-af", "bern-ablation-anchor-only", "bern-ablation-final",
           "bern-ablation-presence-aac", "bern-ablation-presence-aac-timing", "bern-presence-law", "bern-presence-law-decision",
           "bern-first-stage", "bern-nu", "bern-nu-decision", "bern-ceiling")


TRACKS = ("1", "10")
# the profile file of every mode, as the provenance record names the modes
MODE_FILES = {"anchor-only": "bern-anchor-only.json", "agent-only": "bern-agent-only.json", "anchor-aided": "bern-anchor-aided.json"}


def effective_settings(profiles):
    """The settings the three composed Bern profiles run, read from the profiles themselves after every switch of the
    spec, its budgets and the derived first stage, as (shared, modes). shared holds the one link model, the one presence
    law and the one calibration that serve every mode and track, each refused when a profile departs from it. modes
    holds per mode every other setting of its link block, with the area of the anchor only mode and, per track of a
    cooperative mode, the first stage multipliers and the starts."""
    # the link fields of the round's profiles, confirm_round.LINK_FIELDS
    link_keys = compositions.LINK_KEYS + compositions.CARRIED_FIELDS
    found = dict(link_model=[], presence=[], calibration=[])
    modes = {}
    for kind, name in MODE_FILES.items():
        profile = profiles[name]
        if kind == "anchor-only":
            block = profile["anchor_only"]
            found["link_model"].append({key: block[key] for key in link_keys})
            found["presence"].append(block["presence"])
            modes[kind] = dict(settings={key: value for key, value in block.items() if key not in link_keys and key != "presence"},
                               area=profile["area"])
            continue
        modes[kind] = {}
        for n, entry in profile.items():
            block = entry["block_range"]
            found["link_model"].append({key: block[key] for key in link_keys})
            found["presence"].append(entry["presence"])
            found["calibration"].append(entry["calibration"])
            modes[kind][n] = dict(first_stage=dict(entry["range_error"]), starts=entry["starts"],
                                  settings={key: value for key, value in block.items() if key not in link_keys})
    shared = {}
    for key, values in found.items():
        if any(value != values[0] for value in values):
            raise ValueError(f"The Bern profiles run more than one {key.replace('_', ' ')}, which the provenance record states as one")
        shared[key] = copy.deepcopy(values[0])
    return shared, modes


def method_profiles(method):
    """The three Bern profiles of the method from its record and their provenance record, as (profiles, record). The
    record states the settings the profiles run after the spec's switches (effective_settings), and keeps the settings of
    the deployment rules before the switches under deployment.settings_before_switches. A reader's reproduction
    applies the method's switch alone (profiles.switched), and its method derives no first stage."""
    if not common.READER:
        import confirm_round
    spec, first_stage = method["spec"], method["first_stage"]
    fitted = common.read_json(common.FITS / method["fits"]["bern"])["0"]
    fit = fitted["four"]
    if method["reduced"]:
        fit = common.read_json(common.FITS / method["reduced"]["files"]["bern"])["folds"]["0"][method["reduced"]["key"]]["plain"]
    link = dict({key: fit[key] for key in compositions.LINK_KEYS}, **compositions.carried_link_fields(fit))
    calibration = {key: fitted["identity"]["calibration"][key] for key in compositions.CALIBRATION_KEYS}
    placement, area, presence = deployment.settings("bern"), deployment.bern_area(), compositions.presence_law()
    frozen = common.read_json(common.FITS / "bern_cooperative_fit.json")["range_error"]
    anchor_only = dict(area=area, anchor_only=dict(link, **deployment.budgeted(placement, method["budget"]["factor"]), independent_packets=False,
                                                   decision="risk", presence=presence))
    block = dict(link, height_nodes=placement["height_nodes"], height_weights=placement["height_weights"], refine_from_median=True,
                 independent_packets=False)
    cooperative_profiles = {n: dict(range_error=dict(frozen[n]), calibration=dict(calibration), block_range=dict(block), starts=STARTS,
                                    presence=dict(presence)) for n in TRACKS}
    profiles = {}
    for name, mode, profile in (("bern-anchor-only.json", "anc", anchor_only), ("bern-agent-only.json", "af", cooperative_profiles),
                                ("bern-anchor-aided.json", "aac", cooperative_profiles)):
        if common.READER:
            profile = compositions.switched(copy.deepcopy(profile), mode, spec["changes"])
        else:
            profile = confirm_round.apply_changes(copy.deepcopy(profile), mode, spec["changes"])
        if first_stage["derived"] and mode in confirm_round.REPLACEMENTS["derived_first_stage"]:
            # the derived multipliers belong to the link model of these profiles and to no other
            if any({key: profile[n]["block_range"][key] for key in first_stage["links"]["bern"]} != first_stage["links"]["bern"] for n in TRACKS):
                raise ValueError(f"The first stage of {name} in the method record was derived from another link model")
            for n in TRACKS:
                profile[n]["range_error"] = dict(first_stage["multipliers"]["bern"][n])
        # the round's own composition of the spec on every held scan, which a reader's reproduction does not have
        problems = [] if common.READER else confirm_round.evaluation_problems(profile, mode, spec)
        if problems:
            raise ValueError(f"The composition of {name} from the method record departs from the round's composition of its spec in "
                             f"{len(problems)} fields, for example {problems[:3]}")
        profiles[name] = profile if mode == "anc" else compositions.complete_cooperative(profile, name)
    # the record states what the profiles run, after the switches, the budgets and the derived first stage
    shared, modes = effective_settings(profiles)
    record = dict(schema="solid.profile.bern-unified.v7", method_record=compositions.method_provenance(), spec=spec, link_model=shared["link_model"],
                  presence=shared["presence"], calibration=shared["calibration"], modes=modes,
                  link_model_source=dict(path=f"research/artifacts/solid/fits/{method['fits']['bern']}", fit_identity=fitted["identity"],
                                         reduced=method["reduced"]),
                  first_stage=dict(derived=first_stage["derived"],
                                   source="the multipliers the round derived from the link model" if first_stage["derived"]
                                   else "the two multipliers of the production cooperative fit research/artifacts/solid/fits/bern_cooperative_fit.json"),
                  deployment=dict(path="research/artifacts/solid/fits/deployment.json", sha256=common.sha256(deployment.RECORD),
                                  settings_before_switches=deployment.budgeted(placement, method["budget"]["factor"]), area=area, budget=method["budget"]),
                  profiles={name: common.fingerprint(profile) for name, profile in profiles.items()})
    return profiles, record


def unified_profiles():
    """The three Bern profiles of the unified system and their provenance record, as (profiles, record): the method's
    from its record once the round has confirmed it, and before it the pre-round method's."""
    method = compositions.method_record()
    if method is not None:
        return method_profiles(method)
    records = {name: common.read_json(common.SELECTIONS / f"{name}.json") for name in RECORDS}
    for name, record in records.items():
        if not record.get("complete"):
            raise ValueError(f"Selection record {name} is not complete")
    if records["bern-cooperative-block"]["selected"] != "excess_fitted":
        raise ValueError("The joint selection did not choose the fitted excess probability")
    if not records["bern-presence-law-decision"].get("selected"):
        raise ValueError("The presence decision selected no law")
    # the decided law is written out in full, and the ceiling round kept the success cap out of the method
    presence = compositions.presence_law()
    if records["bern-ceiling"].get("adopted"):
        raise ValueError("The ceiling round adopted the success cap, which the engine no longer has")
    nu = float(records["bern-nu-decision"]["nu"])
    fit_name = "bern_mixed_fit_nu3.json" if nu == 3.0 else "bern_mixed_fit.json"
    mixed = compositions.pre_round_fit(fit_name)["0"]
    if not mixed["four"]["converged"] or mixed["identity"]["held_scan"] is not None or mixed["identity"]["evaluation_rows_used"] != 0:
        raise ValueError("The mixed fit must be the converged all-scan fit without evaluation rows")
    link = compositions.link_parameters(mixed["four"], mixed)
    link.pop("nu")
    # the calibration of the mixed fit, the engine's law of raw timings, which every cooperative profile names in full
    fitted = mixed["identity"].get("calibration") or {}
    calibration = {key: fitted[key] for key in compositions.CALIBRATION_KEYS if key in fitted}
    cooperative = common.read_json(common.FITS / "bern_cooperative_fit.json")
    placement, area = deployment.settings("bern"), deployment.bern_area()
    # the anchor only budgets are the base budgets scaled by the converged factor of the budget check record
    budget = deployment.converged_budget()
    anchor_only = dict(area=area, anchor_only=dict(link, **deployment.budgeted(placement, budget), nu=nu, independent_packets=False, decision="risk",
                                                   presence=presence))
    block = dict(link, height_nodes=placement["height_nodes"], height_weights=placement["height_weights"], refine_from_median=True, nu=nu,
                 independent_packets=False)
    cooperative_profiles = {n: dict(range_error=cooperative["range_error"][n], calibration=dict(calibration), block_range=dict(block), starts=STARTS,
                                    presence=dict(presence)) for n in ("1", "10")}
    profiles = {"bern-anchor-only.json": anchor_only}
    for name in ("bern-agent-only.json", "bern-anchor-aided.json"):
        profiles[name] = compositions.complete_cooperative(copy.deepcopy(cooperative_profiles), name)
    record = dict(schema="solid.profile.bern-unified.v5", link_model=link, nu=nu, presence=presence, calibration=calibration,
                  link_model_source=dict(path=f"research/artifacts/solid/fits/{fit_name}", sha256=common.sha256(common.FITS / fit_name),
                                         fit_identity=mixed["identity"]),
                  cooperative_fit=dict(path="research/artifacts/solid/fits/bern_cooperative_fit.json", sha256=common.sha256(common.FITS / "bern_cooperative_fit.json"),
                                       use="the two multipliers of the first stage, kept by the first-stage round"),
                  selections={name: dict(sha256=common.sha256(common.SELECTIONS / f"{name}.json"), selected=record.get("selected"), adopted=record.get("adopted"))
                              for name, record in records.items()},
                  deployment=dict(path="research/artifacts/solid/fits/deployment.json", sha256=common.sha256(deployment.RECORD),
                                  settings=deployment.budgeted(placement, budget), area=area, budget_factor=budget,
                                  budget=deployment.budget_provenance()),
                  settings=dict(starts=STARTS, refine_from_median=True, excess_probability="fitted, joint selection over the three modes",
                                presence="the method's law in every mode, reach estimated from the scenario's successful links, unit weight",
                                anchor_only_deployment=("area, height prior, grid step and budgets by the deployment rules of research/solid/deployment.py "
                                                        "on the eleven training scans, the same rules as every other dataset")),
                  removed=dict(dataset_priors="hull prior, formation prior, shared excess grid and per-anchor offsets, used only on external datasets, removed from the engine",
                               shared_bias="no effect in the round four ablation, removed from the engine",
                               gross_excess="disabled in every profile, removed from the engine",
                               warm_start="graduated warm starts on the same likelihood trail the median first stage by 7 to 11 cm S75, removed",
                               fitted_presence_laws="the availability law, its weight and the anchor detection law left the engine after the presence decision",
                               success_ceiling="the cap of the success probability at the scenario's success ratio, not adopted by the ceiling round, removed from the engine",
                               device_bias="the per device range bias, which helped on no dataset, removed from the engine"),
                  disclosure=("The agent only second round's written rule ordered selection before the gate and picked the 32 start variant, which "
                              "fails the ten sample S90 gate by 1.8 cm; the 64 start variant passes the gate in both tracks and is adopted under the "
                              "repository's established gate-first ordering. The presence decision revised the presence round's margin after the "
                              "round, as its record states. Both records are kept in every case."),
                  profiles={name: common.fingerprint(profile) for name, profile in profiles.items()})
    return profiles, record


def main():
    profiles, record = unified_profiles()
    for name, profile in profiles.items():
        common.write_json(common.PROFILES / name, profile)
        print("written", name)
    common.write_json((common.OUT if common.READER else common.SELECTIONS) / "bern-unified-profile.json", record)
    print("record written, link model", {k: round(v, 4) if isinstance(v, float) else v for k, v in record["link_model"].items()}, "presence", record["presence"])


if __name__ == "__main__":
    main()
