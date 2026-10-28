"""SOLID's training-half records of the held-out benchmarks HYMN and AAU Meshed UWB, all from their training halves.

links       fits the link model of the method on the blocks of each dataset's training half (fit_layout.hymn_groups and
            aau_groups) and writes fits/hymn_layout_fit.json and fits/aau_layout_fit.json. The method record of the
            confirmatory round, research/artifacts/solid/selections/method.json, sets the whole model. The fit takes the
            options of the fits of the method's link model variant (confirm_round.FIT_OPTIONS of its suffix) with
            fit_layout.fit_groups on at most 2000 blocks per track. When the method removed link components, the model
            without them is refitted on the same blocks and weights by the refit of the round's reduced fits
            (fit_reduced.run_fit) and checked as the round checks it, and that refit is the record's link model with the
            full fit beside it. Every fit must come from the likelihood of the round's freeze, which the method freeze
            records. The record keeps the method's spec and the SHA-256 of the method record, and profiles.heldout_fit
            refuses a record of another method.
deployment  applies the deployment rules to the units of both training halves and writes fits/heldout-deployment.json.
profiles    writes the composed profiles profiles/hymn-anchor-only.json and profiles/aau-anchor-only.json.

The freeze pins the sources, so no stop after it could ever be fixed, and every check of these commands that could
stop has the outcome the held-out declaration states (rules.outcomes). A fit of another likelihood than the round's and
a refit that is not the full model minus the removed components are build defects, which no reading can cause, and stop
the command. The freeze check compares the fitter's likelihood with the round's before the ledger opens any step, and
the round's own refits of the method's components pass the settings check under the same sources first. A fit that
does not converge is the method's link model as it stands, never fitted again with other settings, and the tables
mark it. A full fit with a value that is not finite is not refitted, so the method's link model is not finite either,
and every solve with it is scored as SOLID's failure. A failed nested check of the refit is recorded with the refit and
marked. A true agent position of a training half outside its unit's operating area is counted in the deployment record,
counts against SOLID, which alone receives that area, and is marked.

Every command with --check rebuilds its records and compares them with the tracked files instead of writing them. A
link fit is reproduced when every parameter of its fits agrees within a relative 1e-6 and its blocks, identity and
method are equal. Every command runs only after the method freeze and the ledger allow a read of a training half
(heldout.require_freeze), so no link parameter or deployment setting of these datasets exists before it.

A reader's reproduction (SOLID_READER=1) has no freeze, ledger or method record. It fits the method of
research/solid/method_constants.py, whose link model takes the fit options of its suffix, none, and removes no link
component, so neither the round nor its reduced refits enter. It has no likelihood of the round's freeze to compare a
fit with, and its --check compares the spec and the fit options of the method a record names, not the record's hash.

Usage: fit_heldout.py links [--datasets hymn,aau] [--workers 15] [--check]
       fit_heldout.py deployment [--check]
       fit_heldout.py profiles [--check]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import fit_layout  # noqa: E402
import fit_links  # noqa: E402
import heldout  # noqa: E402
import method_constants  # noqa: E402
import profiles  # noqa: E402

TOLERANCE = 1e-6
# at most this many blocks per track, the cap of every layout fit of the round, fit_reduced.CAP, and the default of
# fit_layout.py --blocks
CAP = 2000


def fit_path(dataset):
    return common.FITS / f"{dataset}_layout_fit.json"


def model_options(options):
    """The link model settings of the fit options of a variant, the options of fit_layout.py that
    confirm_round.FIT_OPTIONS names. An option the layout fitter does not take is refused."""
    parser = argparse.ArgumentParser(prog="the fit options of the method", add_help=False)
    parser.add_argument("--offset-shape", default="gaussian", choices=("gaussian", "student_t"))
    parser.add_argument("--early-detection", action="store_true")
    parser.add_argument("--nu", type=float, default=fit_links.NU)
    parsed, unknown = parser.parse_known_args(list(options))
    if unknown:
        raise ValueError(f"the fit options {options} hold {unknown}, which the layout fitter does not take")
    return dict(offset_shape=parsed.offset_shape, fit_early_detection=parsed.early_detection, nu=parsed.nu)


def method_model(method):
    """The fit options of the method's link model variant, the options of the fits of its suffix in the round, those of
    method_constants.py in a reader's reproduction."""
    suffix = method["spec"]["suffix"]
    if common.READER:
        if suffix != method_constants.SUFFIX:
            raise ValueError(f"the method's suffix {suffix!r} is not the suffix {method_constants.SUFFIX!r} of the method's constants")
        return list(method_constants.FIT_OPTIONS)
    import confirm_round
    if suffix not in confirm_round.FIT_OPTIONS:
        raise ValueError(f"the method's suffix {suffix!r} names no fit of the round")
    return list(confirm_round.FIT_OPTIONS[suffix])


def round_likelihood():
    """The SHA-256 of the likelihood of the round's freeze, which every fit must come from, none in a reader's
    reproduction, which has no freeze."""
    if common.READER:
        return None
    content, _ = heldout.verified()
    return content["round_freeze"]["likelihood"]["sha256"]


def finite(record):
    """Whether a fit record holds a finite number in every parameter of the link model and in its negative log
    likelihood. A record keeps a value that is not finite as null (fit_links._clean)."""
    return profiles.finite_numbers([record.get(key, fit_links.OPTIONAL_DEFAULTS.get(key)) for key in fit_links.PARAMETERS] + [record.get("nll")])


def refit(blocks, weights, full, components, label):
    """The link model without the removed components on the blocks and weights of the full fit, by the refit of the
    round's reduced fits, with the settings of the full fit and its optimum as one more start, and its checks. The model
    must be the full model minus the components, a build defect that stops the fit otherwise. The check that a model
    nested in the full one does not beat it is recorded with the refit, and a failure stops nothing, since the refit is
    the method's link model (the declaration's rules.outcomes)."""
    import fit_reduced
    key = fit_reduced.combination_key(components)
    _, _, _, result = fit_reduced.run_fit("heldout", None, key, blocks, weights, full)
    fit_reduced.check_settings(full, result, key, label)
    both = finite(full) and finite(result)
    if all(part in fit_links.NESTED for part in components):
        if not both:
            result["nested_check"] = dict(passed=False, failure="a fit holds a value that is not finite")
        else:
            try:
                result["nested_check"] = fit_reduced.nested_check(blocks, weights, full, result, key, label)
            except ValueError as error:
                result["nested_check"] = dict(passed=False, failure=str(error))
    result["nll_difference_to_full"] = result["nll"] - full["nll"] if both else None
    return result


def fit(dataset, workers, method=None, likelihood=None):
    """The link fit record of a held-out dataset for the method of the method record, refused when a fit comes from
    another likelihood than the round's, a build defect. A fit that did not converge is the method's link model as it
    stands, and a full fit with a value that is not finite is not refitted, so the method's model has no value and every
    solve with it fails (the declaration's rules.outcomes). A reader's reproduction has no likelihood of the round to
    compare with."""
    method = method or profiles.heldout_method()
    if likelihood is None:
        likelihood = round_likelihood()
    options = method_model(method)
    groups, identity = fit_layout.HELDOUT_GROUPS[dataset]()
    seed = fit_layout.HELDOUT_SEEDS[dataset]
    result = fit_layout.fit_groups(groups, CAP, seed, model_options(options), workers)
    fits = dict(full=result["four"])
    components = sorted(method["spec"]["reduced"])
    if components:
        import fit_reduced
        if finite(result["four"]):
            # the blocks and weights of the full fit, drawn again and verified against it as the round's refits draw them
            blocks, weights, _ = fit_reduced.layout_set(groups, result, seed, dataset)
            fits["reduced"] = refit(blocks, weights, result["four"], components, dataset)
        else:
            # a full fit with a value that is not finite has no optimum to refit from, so the method's link model has no
            # value and every solve with it fails
            fits["reduced"] = dict(result["four"], **{key: None for key in (*fit_links.PARAMETERS, "nll")}, converged=False,
                                   refit="not run, the full fit holds a value that is not finite")
        result.update(full=result["four"], four=fits["reduced"], reduced=dict(components=components, key=fit_reduced.combination_key(components)))
    for name, record in fits.items():
        if likelihood is not None and (record.get("likelihood") or {}).get("sha256") != likelihood:
            raise ValueError(f"the {name} link fit of {dataset} comes from likelihood {str((record.get('likelihood') or {}).get('sha256'))[:12]}, "
                             f"not the round's {str(likelihood)[:12]}")
    result["identity"] = identity
    result["method"] = dict(profiles.method_provenance() or {}, spec=method["spec"], fit_options=options)
    return result


def parameters(record):
    return {key: fit_links.parameter(record, key) for key in fit_links.PARAMETERS}


def relative_difference(a, b):
    """The relative difference of two parameters, none between two values that are not finite, kept as null, and an
    infinite one between such a value and a number."""
    if a is None or b is None:
        return 0.0 if a is None and b is None else math.inf
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def method_fields(record):
    """What --check compares of the method a fit record names: all of it, and in a reader's reproduction, which has no
    method record, its spec and fit options."""
    method = record.get("method")
    return {key: (method or {}).get(key) for key in ("spec", "fit_options")} if common.READER else method


def fit_differences(tracked, fresh):
    """The largest relative difference of a parameter of the fits of two records and the names of other fields that
    differ."""
    worst = 0.0
    for key in ("four", "full"):
        if (key in tracked) != (key in fresh):
            return math.inf, [key]
        if key not in tracked:
            continue
        a, b = parameters(tracked[key]), parameters(fresh[key])
        worst = max(worst, max(relative_difference(a[k], b[k]) for k in a))
    others = [key for key in ("blocks", "blocks_cap", "blocks_per_track", "blocks_per_length", "identity", "method", "reduced")
              if (method_fields(tracked) != method_fields(fresh) if key == "method" else tracked.get(key) != fresh.get(key))]
    return worst, others


def summary(record):
    """The parameters, the negative log likelihood and the convergence of a fit, a value that is not finite as None."""

    def rounded(value, digits):
        return round(value, digits) if profiles.finite_numbers([value]) else None
    return {key: rounded(record.get(key, fit_links.OPTIONAL_DEFAULTS.get(key)), 4) for key in fit_links.PARAMETERS} | \
        dict(nll=rounded(record.get("nll"), 3), converged=record.get("converged"))


def links(args):
    method = profiles.heldout_method()
    likelihood = round_likelihood()
    failed = []
    for dataset in args.datasets.split(","):
        record = fit(dataset, args.workers, method, likelihood)
        path = fit_path(dataset)
        print(dataset, summary(record["four"]), record["blocks_per_track"], flush=True)
        for name in ("full", "four"):
            if name in record and not (record[name].get("converged") and finite(record[name])):
                print(f"{dataset}: the {name} link fit did not converge or holds a value that is not finite. It is kept as it stands, and the "
                      "tables mark it (the declaration's rules.outcomes)", flush=True)
        if args.check:
            worst, others = fit_differences(common.read_json(path), record)
            status = "reproduced" if worst <= TOLERANCE and not others else "differs"
            print(f"{dataset}: {status}, largest relative parameter difference {worst:.3g}" + (f", other differences {others}" if others else ""), flush=True)
            if status != "reproduced":
                failed.append(dataset)
                common.write_json(common.ROOT / "out/solid/heldout-check" / path.name, record)
            continue
        if path.exists():
            raise SystemExit(f"{path.relative_to(common.ROOT)} exists, run with --check")
        common.write_json(path, record)
        print(f"written {path.relative_to(common.ROOT)}", flush=True)
    if failed:
        raise SystemExit(f"the held-out link fits of {', '.join(failed)} differ from the tracked records, the refits are under out/solid/heldout-check")


def report(record):
    outside = 0
    for name, item in record["datasets"].items():
        c, s = item["containment"], item["settings"]
        outside += c["linked_outside"] + c["unlinked_outside"]
        print(f"{name}: {c['units']} units, {c['agents']} agents, outside the area {c['linked_outside']} linked and {c['unlinked_outside']} "
              f"unlinked, {c['units_bound_by_extent']} bound by the extent, heights {s['height_nodes']}, grid step {s['grid_step']} cm", flush=True)
    return outside


def deployment_record(args):
    previous = common.read_json(deployment.HELDOUT_RECORD) if deployment.HELDOUT_RECORD.exists() else None
    record = deployment.heldout_build(deployment.HELDOUT_DATASETS)
    outside = report(record)
    if args.check:
        if previous != record:
            raise SystemExit("the tracked held-out deployment record differs from the rules applied to the training halves")
        print("held-out deployment record verified", flush=True)
    else:
        common.write_json(deployment.HELDOUT_RECORD, record)
        print(f"written {deployment.HELDOUT_RECORD.relative_to(common.ROOT)}", flush=True)
    if outside:
        # the area is SOLID's alone, so an agent outside it counts against SOLID, and the tables mark the count
        print(f"{outside} true agent positions of the training halves lie outside their unit's operating area. The record counts them, and "
              "the areas stay as the rules make them (the declaration's rules.outcomes)", flush=True)


def profile_files(args):
    stale = []
    for source in profiles.HELDOUT_SOURCES.values():
        for name, profile in source():
            path = common.PROFILES / name
            if args.check:
                try:
                    common.verify_profile(path, profile)
                    print(f"current   {path.relative_to(common.ROOT)}", flush=True)
                except (OSError, ValueError) as error:
                    stale.append(str(error))
                continue
            previous = common.read_json(path) if path.exists() else None
            common.write_json(path, profile)
            print(f"{'unchanged' if previous == profile else 'written  '} {path.relative_to(common.ROOT)}", flush=True)
    if stale:
        raise SystemExit("\n".join(stale))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("links", "deployment", "profiles"))
    parser.add_argument("--datasets", default="hymn,aau")
    parser.add_argument("--workers", type=int, default=15, help="processes for the starts of every fit")
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the tracked records instead of writing them")
    args = parser.parse_args()
    if any(d not in deployment.HELDOUT_DATASETS for d in args.datasets.split(",")):
        parser.error("the held-out datasets are hymn and aau")
    heldout.require_freeze("training")
    dict(links=links, deployment=deployment_record, profiles=profile_files)[args.command](args)


if __name__ == "__main__":
    main()
