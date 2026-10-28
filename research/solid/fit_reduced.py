"""Refits of the link model without one component each, the reduced models of the ablation matrix at their own best fit.

Switching a component off while the other parameters keep the values fitted with it measures the removal plus a
misfit of what remains. The ablation compares nested models, so every reduced model is refitted by the same optimizer
(research/solid/fit_links.py) on exactly the blocks and weights of the full fit, with the full fit's offset shape,
early detection setting and degrees of freedom. The reduced models are the model without the excess slab (the packet
scale and the offset refitted), without the persistent offset, with a Gaussian core (nu 1e6), and with independent
packets, every packet its own block with its block's weight. Every refit also starts from the full fit's optimum, its
removed components at their fixed values.

For Bern the records are the eleven held-scan folds and the all-scan fit, with blocks drawn exactly as
fit_bern_mixed_blocks.py draws them with the parameters in the record's identity. For TIERS and Bregar they are
the training halves, with blocks drawn exactly as fit_layout.py draws them. For IDLab it is the reserved training
partition, with blocks built exactly as fit_idlab.py builds them. With --fold they are the fits of one fold of the
confirmatory round (research/solid/folds.py), fitted by fit_layout.py and fit_idlab.py --fold, whose blocks are drawn
from the clusters of that fold alone, and every record must name that fold of the declared assignment.

Every reconstruction is verified twice. The blocks per track must match the record, and the weighted negative log
likelihood of the reconstructed blocks at the record's parameters must equal the record's value, which also requires
the record to come from the likelihood in use. A record of another likelihood is refused.

Every refit is a refit of the full model minus its components. It frees the full fit's parameters except those the
components fix, with the same offset shape, the same further likelihood keys and, apart from the Gaussian core, the
same degrees of freedom, so a full model with an early detection rate or a Student t offset keeps them in every
reduced model. A refit that differs in any of these is refused.

The reduced models without the excess and without the offset lie inside the full model's parameter space, so the full
fit can never be worse, and NLL(full) <= NLL(reduced) + 1e-6 is asserted for both. The full model reaches a removed
component only up to its bound (an excess probability of 1e-6, an offset of 1e-4 reference scales), so when the
literal check fails the reduced optimum is placed on that bound and evaluated under the full model. The fit passes
only if the full fit is not worse than that point of its own space and the remaining gap, the price of the bound, stays
within BOUNDARY_LIMIT, 0.01 nats; the gap is recorded. The Gaussian core and the independent packets are not nested in
the full model, and the record keeps their difference to the full fit without a check.

Writes research/artifacts/solid/fits/reduced/<fit file name>, holding {variant: {"plain": fit}} per fit record, under
the record's key. A dataset's file is rewritten as each of its refits completes, with the refits still running under
"pending". A refit, a reconstruction or a check that fails is recorded as {"failed": message} in place of its fit, the
other refits go on, and the script exits with an error naming the failures once every file is written.

With --combined a,b the script fits instead the one model without every listed component together, the joint refit
that the combined removal of a design round needs, and writes it to fits/reduced/combined/<fit file name> under the
key a+b, the names sorted. Two or more of the four components are required. A combination of nested components is
checked like a single one, with the reduced optimum placed on the bound of every removed component.

Usage: fit_reduced.py [--datasets bern,tiers,bregar,idlab] [--processes 64] [--archive PATH] [--suffix S]
                      [--bern-folds bern_mixed_fold_fits_nu3.json] [--bern-all bern_mixed_fit_nu3.json]
                      [--combined no_excess,no_offset] [--fold 0|1]
       --suffix reads and writes the layout and IDLab fit files of fit_layout.py and fit_idlab.py --suffix S, and
       --fold K those of one fold of the round, which take tiers, bregar and idlab alone
"""
import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_layout  # noqa: E402
import fit_links  # noqa: E402
import folds  # noqa: E402
import layout_split  # noqa: E402

OUT_DIR = common.FITS / "reduced"
COMBINED = "combined"
LAYOUT_FILES = dict(tiers="tiers_layout_fit.json", bregar="bregar_layout_fits.json")
IDLAB_FILE = "idlab_link_fit.json"
BERN_FILE = "bern_mixed_fold_fits_nu3.json"
BERN_ALL = "bern_mixed_fit_nu3.json"
CAP = 2000
NESTED_TOLERANCE = 1e-6
BOUNDARY_LIMIT = 0.01
RECONSTRUCTION_TOLERANCE = 1e-9


def verify(blocks, weights, four, label):
    """The record's negative log likelihood recomputed on the reconstructed blocks at the record's parameters."""
    current = fit_links.likelihood_identity()
    recorded = four.get("likelihood") or {}
    if recorded.get("sha256") != current["sha256"]:
        raise ValueError(f"{label}: the fit record comes from likelihood {recorded.get('name')} {str(recorded.get('sha256'))[:12]}, "
                         f"not {current['name']} {current['sha256'][:12]}; refit the full model first")
    nll = fit_links.model_nll(blocks, weights, four)
    if abs(nll - four["nll"]) > RECONSTRUCTION_TOLERANCE * max(1.0, abs(four["nll"])):
        raise ValueError(f"{label}: the reconstructed blocks give NLL {nll:.9f} at the record's parameters, the record holds {four['nll']:.9f}")
    return nll


def combination_key(variants):
    """The key of a reduced model in its file, the removed components sorted and joined by a plus sign."""
    return "+".join(sorted(variants))


def components(key):
    return tuple(key.split("+"))


def nested_check(blocks, weights, full, reduced, variant, label):
    """The full fit against a reduced fit nested in it, raising when the full optimizer missed a better point. variant
    names one component or, joined by a plus sign, several nested ones removed together."""
    gap = full["nll"] - reduced["nll"]
    entry = dict(full_nll=full["nll"], reduced_nll=reduced["nll"], gap_nats=gap, tolerance_nats=NESTED_TOLERANCE)
    if gap <= NESTED_TOLERANCE:
        return dict(entry, passed="literal")
    params = dict(reduced)
    for component in components(variant):
        params.update(fit_links.embedded_params(component, params))
    boundary = fit_links.model_nll(blocks, weights, dict(full, **{key: params[key] for key in fit_links.PARAMETERS}))
    entry.update(boundary_nll=boundary, boundary_gap_nats=boundary - reduced["nll"])
    if full["nll"] > boundary + NESTED_TOLERANCE:
        raise ValueError(f"{label} {variant}: the full fit (NLL {full['nll']:.6f}) is worse than the reduced optimum on the bound of its "
                         f"own space (NLL {boundary:.6f}); refit the full model")
    if boundary - reduced["nll"] > BOUNDARY_LIMIT:
        raise ValueError(f"{label} {variant}: the bound of the removed component costs {boundary - reduced['nll']:.4f} nats, the "
                         "likelihood is not continuous there")
    return dict(entry, passed="up to the bound of the removed component")


def check_settings(full, reduced, variant, label):
    """Refuse a reduced fit that is not the full model minus its components. Such a fit frees the full fit's parameters
    except those the components fix, with the full fit's offset shape and further likelihood keys and, apart from the
    Gaussian core, its degrees of freedom."""
    fixed = set()
    for component in components(variant):
        fixed.update(fit_links.variant_settings(component, {}).get("fixed") or {})
    expected = dict(free=sorted(name for name in full["free"] if name not in fixed), offset_shape=full.get("offset_shape"),
                    extra=full.get("extra") or {}, nu=fit_links.GAUSSIAN_NU if "gaussian_core" in components(variant) else full.get("nu"))
    found = dict(free=sorted(reduced["free"]), offset_shape=reduced.get("offset_shape"), extra=reduced.get("extra") or {}, nu=reduced.get("nu"))
    differing = {key: (found[key], expected[key]) for key in expected if found[key] != expected[key]}
    if differing:
        raise ValueError(f"{label} {variant}: the refit is not the full model minus its components, "
                         + ", ".join(f"{key} {value[0]} instead of {value[1]}" for key, value in differing.items()))


def layout_set(groups, record, seed, label, verify_nll=True, strata=False, verify_counts=True):
    """The blocks and weights of one layout fit record, verified against it, with the track of every block when
    strata is set. verify_counts false leaves out the check of the blocks per track, for the stand-ins of a dry run,
    which copy the record of another part."""
    plain = {track: [b for _, _, _, b in tagged] for track, tagged in groups.items()}
    chosen, weights, counts, tracks = fit_layout.stratified_sample(plain, record.get("blocks_cap", CAP), np.random.default_rng(seed))
    if verify_counts and counts != record["blocks_per_track"]:
        raise ValueError(f"{label}: the reconstructed tracks {counts} differ from the fit record {record['blocks_per_track']}")
    if verify_nll:
        verify(chosen, weights, record["four"], label)
    return (chosen, weights, record["four"], tracks) if strata else (chosen, weights, record["four"])


def layout_file(dataset, suffix="", fold=None):
    """The layout or IDLab fit file of a model variant, of one fold of the round when a fold is given."""
    return (IDLAB_FILE if dataset == "idlab" else LAYOUT_FILES[dataset]).replace(".json", f"{suffix}{folds.name(fold)}.json")


def part_name(fold):
    """The part of a training half a fit reads, the whole half or one fold of the round."""
    return "the whole training half" if fold is None else f"fold {fold}"


def fold_label(fold):
    """The part of a label that names a fold of the round, none for the whole training half."""
    return "" if fold is None else f" fold {fold}"


def fold_of(dataset, record, fold, label):
    """The fold a fit record was fitted on, refused unless it is the fold its file names."""
    found = folds.fitted(dataset, record)
    if found != fold:
        raise ValueError(f"{label}: the fit record was fitted on {part_name(found)}, not on {part_name(fold)}")
    return fold


def layout_jobs(datasets, suffix="", fold=None):
    """(dataset, record key, blocks, weights, full fit) for every layout record, those of one fold when given."""
    jobs = []
    if "tiers" in datasets:
        record = common.read_json(common.FITS / layout_file("tiers", suffix, fold))
        groups, _ = fit_layout.tiers_groups(fold_of("tiers", record, fold, "tiers"))
        jobs.append(("tiers", None, *layout_set(groups, record, fit_layout.SEEDS["tiers"], f"tiers{fold_label(fold)}")))
    if "bregar" in datasets:
        records = common.read_json(common.FITS / layout_file("bregar", suffix, fold))["environments"]
        groups, _ = fit_layout.bregar_groups(fold)
        for environment, parts in sorted(groups.items()):
            seed = fit_layout.SEEDS["bregar"] + int(environment[-1])
            fold_of("bregar", records[environment], fold, f"bregar {environment}")
            jobs.append(("bregar", environment, *layout_set(parts, records[environment], seed, f"bregar {environment}{fold_label(fold)}")))
    return jobs


def idlab_inputs():
    """The reserved IDLab training cells, the surveyed coordinates in millimetres, and the correction and packet scale
    of the fit, read exactly as fit_idlab.py reads them."""
    import idlab_admission
    import idlab_split
    import idlab_training
    import pilot
    training = common.read_json(common.EXTERNAL / "idlab-training.json")
    correction, scale = training["correction"], training["profiles"]["scale_cm"]
    split = pilot.evidence.read(idlab_split.OUTPUT)
    surveyed = np.loadtxt(idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")
    coordinates = {int(row[0]): row[1:] for row in surveyed}
    cells = idlab_training.load_training(split, idlab_admission.FOLDER, coordinates)
    return cells, coordinates, correction, scale


def idlab_blocks(links=False, fold=None):
    """The blocks, weights and tracks of the IDLab fit, built from the reserved training rows exactly as fit_idlab.py
    builds them, those of one fold of the round when a fold is given, with the link of every block and the surveyed
    coordinates when links is set."""
    import fit_idlab
    cells, coordinates, correction, scale = idlab_inputs()
    blocks, strata, within, tagged = fit_idlab.training_blocks(cells, correction, scale)
    if fold is not None:
        blocks, strata, within, tagged = fit_idlab.fold_blocks(blocks, strata, within, tagged, fold)
    weights = fit_links.stratified_weights(strata, within)
    return (blocks, weights, strata, tagged, coordinates) if links else (blocks, weights, strata)


def idlab_set(record, label="idlab", verify_nll=True):
    """The blocks and weights of the IDLab fit record, of the fold it names, verified against it."""
    blocks, weights, strata = idlab_blocks(fold=folds.fitted("idlab", record))
    counts = {str(track): strata.count(track) for track in sorted(set(strata))}
    if counts != record.get("blocks_per_track", counts) or len(blocks) != record["blocks"]:
        raise ValueError(f"{label}: the reconstructed tracks {counts} differ from the fit record")
    if verify_nll:
        verify(blocks, weights, record["four"], label)
    return blocks, weights, strata


def idlab_jobs(datasets, suffix="", fold=None):
    """(dataset, record key, blocks, weights, full fit) for the IDLab record, that of one fold when given."""
    if "idlab" not in datasets:
        return []
    record = common.read_json(common.FITS / layout_file("idlab", suffix, fold))["b10"]
    fold_of("idlab", record, fold, "idlab")
    blocks, weights, _ = idlab_set(record, f"idlab{fold_label(fold)}")
    return [("idlab", None, blocks, weights, record["four"])]


def collect_bern(held, identity, archive):
    import fit_bern_mixed_blocks as bern
    chosen, strata = bern.collect(held, identity["per_scan"], identity["blocks_per_kind"], identity["seed"], archive)
    return held, chosen, fit_links.stratified_weights(strata)


def combined_settings(variants, settings):
    """The fit settings of the model without every component of `variants`, from the settings of the full model."""
    model = dict(settings)
    for variant in variants:
        model = fit_links.variant_settings(variant, model)
    return model


def fit_combined(blocks, weights, variants, full=None, **settings):
    """The link model without every component of `variants` together, refitted by the same optimizer on the same
    blocks and weights, every packet its own block when independent packets are among them. full, the full fit's
    record, adds its optimum as one more start."""
    model = combined_settings(variants, settings)
    if "independent_packets" in variants:
        blocks, weights = fit_links.single_packets(blocks, weights)
    return fit_links.fit_link_model(blocks, weights, starts=fit_links.full_start(full, model), **model)


def run_fit(dataset, key, variant, blocks, weights, full):
    started = time.time()
    parts, settings = components(variant), fit_links.model_settings(full)
    if len(parts) == 1:
        result = fit_links.fit_variant(blocks, weights, variant, full=full, **settings)
    else:
        result = fit_combined(blocks, weights, parts, full=full, **settings)
    return dataset, key, variant, dict(result, seconds=time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="bern,tiers,bregar")
    parser.add_argument("--processes", type=int, default=64)
    parser.add_argument("--archive", type=Path, default=common.OUT / "unify/bern-training-scenarios-v2.zip")
    parser.add_argument("--bern-folds", default=BERN_FILE, help="tracked eleven fold fit file under research/artifacts/solid/fits")
    parser.add_argument("--suffix", default="", help="suffix of the layout and IDLab fit files, as given to fit_layout.py and fit_idlab.py")
    parser.add_argument("--bern-all", default=BERN_ALL, help="tracked all-scan fit file under research/artifacts/solid/fits")
    parser.add_argument("--combined", default=None, help="comma separated components refitted together instead of one by one")
    parser.add_argument("--fold", type=int, default=None, choices=folds.FOLDS, help="refit the fits of one fold of the confirmatory round")
    args = parser.parse_args()
    datasets = args.datasets.split(",")
    if args.fold is not None and "bern" in datasets:
        parser.error("Bern holds its own held-scan folds, so --fold takes tiers, bregar and idlab alone")
    if args.combined:
        combination = sorted(set(args.combined.split(",")))
        if len(combination) < 2 or set(combination) - set(fit_links.VARIANTS):
            parser.error(f"--combined needs two or more of {', '.join(fit_links.VARIANTS)}")
        variants, out_dir = [combination_key(combination)], OUT_DIR / COMBINED
    else:
        variants, out_dir = list(fit_links.VARIANTS), OUT_DIR
    results = defaultdict(lambda: defaultdict(dict))
    identity = dict(variants=variants if args.combined else list(fit_links.VARIANTS), nested=list(fit_links.NESTED), gaussian_nu=fit_links.GAUSSIAN_NU,
                    nested_tolerance_nats=NESTED_TOLERANCE, likelihood=fit_links.likelihood_identity(),
                    split_record_sha256=common.sha256(layout_split.RECORD), evaluation_rows_used=0,
                    source={p.name: common.sha256(p) for p in (Path(__file__), Path(fit_links.__file__), Path(fit_layout.__file__))})
    if "idlab" in datasets:
        # the IDLab blocks come from fit_idlab.training_blocks
        identity["source"]["fit_idlab.py"] = common.sha256(Path(__file__).resolve().parent / "fit_idlab.py")
    if args.combined:
        identity["combined"] = combination
    if args.fold is not None:
        # the blocks of a fold's fit are those of its clusters, research/solid/folds.py
        identity.update(fold=args.fold, source=dict(identity["source"], **{"folds.py": common.sha256(Path(folds.__file__))}))
    out_dir.mkdir(parents=True, exist_ok=True)
    sets, pending, failures = {}, defaultdict(set), []
    started = time.time()

    def write(dataset):
        """The dataset's file with every refit completed so far and the refits still running."""
        entries = {key: dict(done) for key, done in results[dataset].items()}
        waiting = sorted(f"{key or ''} {variant}".strip() for key, variant in pending[dataset])
        name = args.bern_folds if dataset == "bern" else layout_file(dataset, args.suffix, args.fold)
        if dataset == "bern":
            record = dict(identity, folds=entries, sources={file: common.sha256(common.FITS / file) for file in (args.bern_folds, args.bern_all)})
        elif dataset == "bregar":
            record = dict(identity, environments=entries, source_sha256=common.sha256(common.FITS / name))
        elif dataset == "idlab":
            record = dict(identity, b10=entries.get(None, {}), source_sha256=common.sha256(common.FITS / name))
        else:
            record = dict(identity, **entries.get(None, {}), source_sha256=common.sha256(common.FITS / name))
        common.write_json(out_dir / name, dict(record, pending=waiting))

    def fail(dataset, key, variant, error):
        results[dataset][key][variant] = dict(failed=str(error))
        failures.append(f"{dataset} {key or ''} {variant}: {error}")
        print(f"failed: {dataset} {key or ''} {variant}: {error}", flush=True)

    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {}
        for dataset, key, blocks, weights, full in layout_jobs(datasets, args.suffix, args.fold) + idlab_jobs(datasets, args.suffix, args.fold):
            sets[(dataset, key)] = (blocks, weights, full)
            for variant in variants:
                futures[pool.submit(run_fit, dataset, key, variant, blocks, weights, full)] = (dataset, key, variant)
                pending[dataset].add((key, variant))
        if "bern" in datasets:
            held_fits = common.read_json(common.FITS / args.bern_folds)
            whole = common.read_json(common.FITS / args.bern_all)
            records = {held: whole["0"] if held == 0 else held_fits[str(held)] for held in range(12)}
            collections = {pool.submit(collect_bern, held, records[held]["identity"], args.archive): held for held in range(12)}
            for held in range(12):
                pending["bern"].update((str(held), variant) for variant in variants)
            for future in as_completed(collections):
                held = collections[future]
                try:
                    _, chosen, weights = future.result()
                    record = records[held]
                    if len(chosen) != record["blocks"]:
                        raise ValueError(f"Bern fold {held}: the reconstructed blocks differ from the fit record")
                    verify(chosen, weights, record["four"], f"bern fold {held}")
                except Exception as error:   # noqa: BLE001 - a fold that cannot be rebuilt is recorded, the others go on
                    for variant in variants:
                        pending["bern"].discard((str(held), variant))
                        fail("bern", str(held), variant, error)
                    write("bern")
                    continue
                sets[("bern", str(held))] = (chosen, weights, record["four"])
                for variant in variants:
                    futures[pool.submit(run_fit, "bern", str(held), variant, chosen, weights, record["four"])] = ("bern", str(held), variant)
                print(f"bern fold {held} collected and verified after {time.time() - started:.0f}s", flush=True)
        for done, future in enumerate(as_completed(futures), 1):
            dataset, key, variant = futures[future]
            pending[dataset].discard((key, variant))
            label = f"{dataset} {key or ''}"
            try:
                _, _, _, fit = future.result()
                blocks, weights, full = sets[(dataset, key)]
                check_settings(full, fit, variant, label)
                if all(part in fit_links.NESTED for part in components(variant)):
                    fit["nested_check"] = nested_check(blocks, weights, full, fit, variant, label)
                fit["nll_difference_to_full"] = fit["nll"] - full["nll"]
                results[dataset][key][variant] = dict(plain=fit)
                print(f"{done} fitted: {label} {variant} " + str({k: round(fit[k], 4) for k in fit_links.LINK_KEYS})
                      + f" NLL {fit['nll_difference_to_full']:+.3f} to full, {fit['seconds']:.0f}s", flush=True)
            except Exception as error:   # noqa: BLE001 - a failed refit or check is recorded, the others go on
                fail(dataset, key, variant, error)
            write(dataset)
    for dataset in datasets:
        write(dataset)
        print(f"written {out_dir / (args.bern_folds if dataset == 'bern' else layout_file(dataset, args.suffix, args.fold))}", flush=True)
    print(f"reduced fits done in {time.time() - started:.0f}s", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} refits failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
