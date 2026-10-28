"""The two folds of the confirmatory round on the training halves of TIERS, Bregar and IDLab.

The round judges Bern out of fold, every held scan positioned with the link model fitted on the other scans. It judges
TIERS, Bregar and IDLab the same way. Each training half is split into two folds, every link model of the round is
fitted on each fold alone, and the units of a fold are positioned with the fit of the other fold, so no unit is
positioned by a link model, a refit or a first stage fitted on its own measurements.

This module declares what each split takes, with the seed SEED: the TIERS training blocks, alternating within each
recording, the Bregar channels within each environment by bandwidth, and the IDLab radio configurations by channel and
transmit power. Within every stratum the blocks, channels or configurations are sorted by the settings the folds
balance and then by their hash, the SHA-256 of the seed and their names, and alternate along that whole order between
fold 0 and fold 1, starting with fold 0, so a group that shares its balanced settings is split as evenly as its size
allows. The code calls each block, channel or configuration a cluster of the split. The splits are not the bootstrap
designs: the bootstrap of the decision rule draws the units of both folds together under each dataset's own design in
research/solid/decision.py.

- TIERS. The folds split the training blocks within every recording, an acquisition as the layout split record names
  it. The blocks of each recording alternate by the hash of the seed, the acquisition and the block, so every recording
  has half its blocks in each fold.
- Bregar. The folds split the channels within every environment, each channel of an environment one acquisition. They
  are sorted by the environment, the bandwidth of the channel, 900 MHz for channels 4 and 7 and 500 MHz for channels 1,
  2, 3 and 5, and the hash of the seed, the environment and the acquisition. Every environment holds six channels, so
  each fold receives one 900 MHz and two 500 MHz channels of every environment.
- IDLab. The folds split the radio configurations, sorted by the channel, the transmit power and the hash of the seed
  and the configuration, so every channel and every transmit power of a channel is split as evenly as its count allows.
  The panels of a configuration reuse its rows, since idlab_population.panel takes a block index modulo the block count
  of every link, so no split of the panels would keep the rows of a fit apart from the rows it positions. The link fit
  of fit_idlab.py gives every link setting the same weight, and every link setting of the reserved rows belongs to one
  configuration, so a fit on the configurations of one fold positions those of the other out of sample.

The deployment settings, the area, the height nodes and the grid step, stay those of the whole training half in both
folds. Both folds share one physical layout, so these settings are properties of the layout rather than fitted link
noise, the calibration on the same layout of these training halves. The input calibrations every method receives stay
those of the whole training half too, the affine range correction and the packet scale of IDLab from every reserved row
and the offsets and packet scales of every Bregar channel from its training block, and they are fitted on rows that
include those the folds position. The calibration of Bregar cannot be cross-fitted inside its training block, block 0
of every environment, since every channel is calibrated from its own rows there, so the round records every verdict
also without the votes of Bregar.

A fit of one fold records its fold, the seed, the identity of the assignment, the blocks, channels or configurations it
was fitted on and those whose units it positions, and every reader refuses a fold of another assignment. The fits of the whole
training halves record no fold and stay the method's fits for the evaluation halves.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SEED = 20260927
FOLDS = (0, 1)
DATASETS = ("tiers", "bregar", "idlab")
# what the folds of every dataset split, the settings it is sorted by, and the names each hash takes after the seed
CLUSTERS = dict(tiers="the training blocks within every acquisition, hashed with the acquisition and the block",
                bregar=("the acquisitions, an environment and a channel, sorted by the environment, the bandwidth of the channel and the hash "
                        "of the environment and the acquisition"),
                idlab="the radio configurations, sorted by the channel, the transmit power and the hash of the configuration")
# the bandwidth in MHz of every UWB channel of Bregar, the wide channels 4 and 7 and the others
BREGAR_BANDWIDTH_MHZ = {1: 500, 2: 500, 3: 500, 4: 900, 5: 500, 7: 900}
# the settings of the clusters of Bregar and IDLab whose count per fold the declaration records, in the order they are sorted by
BALANCE = dict(bregar=("environment", "bandwidth_mhz"), idlab=("channel", "txpower"))
BREGAR_REFERENCE = "bregar-layout/training/reference.json"
IDLAB_BASE = "idlab-training-campaign/base.json.gz"


def check(fold):
    """A fold of the round, 0 or 1, refused otherwise."""
    if isinstance(fold, bool) or fold not in FOLDS:
        raise ValueError(f"a fold of the round is one of {FOLDS}, not {fold!r}")
    return fold


def other(fold):
    """The fold whose fit positions the units of a fold, the other one."""
    return FOLDS[1 - check(fold)]


def name(fold):
    """The part of a fit file's name that names the fold it was fitted on, none for the whole training half."""
    return "" if fold is None else f"_fold{check(fold)}"


def order(*names):
    """The hash of a cluster, the last key of its place in the order of its stratum: the SHA-256 of the seed and the
    cluster's names."""
    return hashlib.sha256(json.dumps([SEED, *names]).encode()).hexdigest()


# ------------------------------------------------------------------------------------------- the clusters
def source(dataset):
    """The file that names the clusters of a dataset: the layout split record for TIERS, the reference of the Bregar
    training half and the base of the IDLab training campaign."""
    if dataset == "tiers":
        import layout_split
        return Path(layout_split.RECORD)
    if dataset == "bregar":
        return common.EXTERNAL / BREGAR_REFERENCE
    if dataset == "idlab":
        return common.EXTERNAL / IDLAB_BASE
    raise ValueError(f"the round has no folds of {dataset}")


_CACHE = {}


def cached(what, dataset, build):
    """A value built from the source file of a dataset, built again when the file changes."""
    path = source(dataset)
    status = path.stat()
    key = (what, dataset, str(path), status.st_mtime_ns, status.st_size)
    if key not in _CACHE:
        _CACHE[key] = build(path)
    return _CACHE[key]


def tiers_strata(path):
    """The training blocks of every TIERS acquisition, each with its key, its hash of the acquisition and the block."""
    tiers = common.read_json(path)["tiers"]
    blocks = {acquisition: sorted(entry["training_blocks"]) for acquisition, entry in sorted(tiers["acquisitions"].items())}
    if sorted(block for values in blocks.values() for block in values) != sorted(tiers["training_blocks"]):
        raise ValueError("the TIERS training blocks of the split record differ from those of its acquisitions")
    return {acquisition: [((acquisition, block), (order(acquisition, block),)) for block in values] for acquisition, values in blocks.items()}


def bregar_table(path):
    """The environment and the bandwidth of the channel of every Bregar training acquisition, {acquisition: {environment,
    bandwidth_mhz}}."""
    import layout_split
    units = layout_split.bregar_units("training")
    table, seen = {}, set()
    for unit in common.read_json(path)["units"]:
        if unit["id"] in units:
            seen.add(unit["id"])
            entry = dict(environment=unit["reference"], bandwidth_mhz=BREGAR_BANDWIDTH_MHZ[unit["channel"]])
            if table.setdefault(unit["acquisition"], entry) != entry:
                raise ValueError(f"the Bregar acquisition {unit['acquisition']} holds two environments or channel bandwidths")
    if seen != units or len(table) != len(units):
        raise ValueError("the Bregar training reference does not hold one acquisition per training unit")
    return dict(sorted(table.items()))


def bregar_strata(path):
    """The Bregar training acquisitions as one stratum, each with its key, the environment, the bandwidth of its channel
    and its hash of the environment and the acquisition."""
    return {"acquisitions": [(acquisition, (entry["environment"], entry["bandwidth_mhz"], order(entry["environment"], acquisition)))
                             for acquisition, entry in bregar_table(path).items()]}


def idlab_table(path):
    """The radio settings of every configuration of the IDLab training campaign, {configuration: settings}."""
    settings = {}
    for panel in json.load(gzip.open(path))["panels"]:
        if settings.setdefault(panel["configuration"], panel["settings"]) != panel["settings"]:
            raise ValueError(f"configuration {panel['configuration']} of the IDLab training campaign holds two radio settings")
    if len({json.dumps(value, sort_keys=True) for value in settings.values()}) != len(settings):
        raise ValueError("two configurations of the IDLab training campaign hold the same radio settings")
    return dict(sorted(settings.items()))


def idlab_settings():
    """The radio settings of every configuration of the IDLab training campaign, {configuration: settings}."""
    return cached("settings", "idlab", idlab_table)


def idlab_attributes(configuration, settings):
    """The balanced settings of an IDLab configuration, its channel and its transmit power, the same for the anchors and
    the tag."""
    if settings["txpower_anchor"] != settings["txpower_tag"]:
        raise ValueError(f"configuration {configuration} of the IDLab training campaign transmits with two powers")
    return dict(channel=settings["channel"], txpower=settings["txpower_tag"])


def idlab_strata():
    """The IDLab configurations as one stratum, each with its key, the channel, the transmit power and its hash of the
    configuration."""
    members = []
    for configuration, settings in idlab_settings().items():
        entry = idlab_attributes(configuration, settings)
        members.append((configuration, (entry["channel"], entry["txpower"], order(configuration))))
    return {"configurations": members}


STRATA = dict(tiers=lambda: cached("strata", "tiers", tiers_strata), bregar=lambda: cached("strata", "bregar", bregar_strata), idlab=idlab_strata)


def strata(dataset):
    """The clusters of a dataset per stratum, each with the key of its place in the order of the stratum, {stratum:
    [(cluster, key)]}, the key a tuple of the balanced settings and the hash. A TIERS cluster is (acquisition, block),
    a Bregar cluster its acquisition and an IDLab cluster its configuration. The acquisitions of Bregar and the
    configurations of IDLab each form one stratum."""
    if dataset not in STRATA:
        raise ValueError(f"the round has no folds of {dataset}")
    return STRATA[dataset]()


def assign(table):
    """{cluster: fold}: the clusters of every stratum sorted by their keys, alternating along that whole order from
    fold 0."""
    result = {}
    for members in table.values():
        for place, (cluster, _) in enumerate(sorted(members, key=lambda member: member[1])):
            result[cluster] = FOLDS[place % len(FOLDS)]
    return result


def attributes(dataset):
    """The balanced settings of every cluster of Bregar and IDLab, {cluster: {setting: value}}."""
    if dataset == "bregar":
        return cached("attributes", "bregar", bregar_table)
    if dataset == "idlab":
        return {configuration: idlab_attributes(configuration, settings) for configuration, settings in idlab_settings().items()}
    raise ValueError(f"the folds of {dataset} balance no settings")


def balance(dataset):
    """The count of clusters per fold of every value of each balanced setting of Bregar or IDLab and of every pair of
    them, {setting: {value: [fold 0, fold 1]}}, the pairs keyed by both values joined with a slash."""
    table, (first, second) = assignment(dataset), BALANCE[dataset]
    result = {first: {}, second: {}, f"{first}/{second}": {}}
    for cluster, entry in attributes(dataset).items():
        for setting, value in ((first, entry[first]), (second, entry[second]), (f"{first}/{second}", f"{entry[first]}/{entry[second]}")):
            result[setting].setdefault(str(value), [0] * len(FOLDS))[table[cluster]] += 1
    return {setting: dict(sorted(values.items())) for setting, values in result.items()}


def assignment(dataset):
    """The fold of every cluster of a dataset, {cluster: fold}."""
    return cached("assignment", dataset, lambda path: assign(strata(dataset)))


def clusters(dataset, fold):
    """The clusters of one fold, sorted."""
    check(fold)
    return sorted(cluster for cluster, value in assignment(dataset).items() if value == fold)


def listed(values):
    """Clusters as JSON holds them, a TIERS cluster as the list [acquisition, block]."""
    return [list(value) if isinstance(value, tuple) else value for value in values]


def tiers_blocks(fold):
    """The TIERS training blocks of one fold. The blocks are numbered across the acquisitions, so a block names its
    cluster."""
    return {block for _, block in clusters("tiers", fold)}


def bregar_acquisitions(fold):
    """The Bregar training acquisitions of one fold."""
    return set(clusters("bregar", fold))


def idlab_configurations(fold):
    """The IDLab radio configurations of one fold."""
    return set(clusters("idlab", fold))


def idlab_configuration(settings):
    """The configuration of the IDLab training campaign that holds these radio settings."""
    table = cached("configuration", "idlab", lambda path: {json.dumps(value, sort_keys=True): key for key, value in idlab_settings().items()})
    key = json.dumps(settings, sort_keys=True)
    if key not in table:
        raise ValueError(f"no configuration of the IDLab training campaign holds the radio settings {settings}")
    return table[key]


# ------------------------------------------------------------------------------------------- the records
def declaration(dataset):
    """The assignment of a dataset as the freeze of the round records it: the seed, the clusters it splits, the clusters
    of every fold and their count per stratum, for Bregar and IDLab their count per fold of every balanced setting, and
    for IDLab the radio settings of every configuration."""
    table = assignment(dataset)
    counts = {stratum: [sum(1 for cluster, _ in members if table[cluster] == fold) for fold in FOLDS] for stratum, members in strata(dataset).items()}
    result = dict(dataset=dataset, seed=SEED, clusters=CLUSTERS[dataset], folds={str(fold): listed(clusters(dataset, fold)) for fold in FOLDS},
                  counts=counts)
    if dataset in BALANCE:
        result["balance"] = balance(dataset)
    if dataset == "idlab":
        result["settings"] = {str(configuration): settings for configuration, settings in idlab_settings().items()}
    return result


def identity(dataset):
    """The SHA-256 of a dataset's declared assignment, which every fit of one of its folds records."""
    return cached("identity", dataset, lambda path: common.fingerprint(declaration(dataset)))


def record(dataset, fold):
    """The fold entry of a fit of one fold: the fold, the seed, the identity of the assignment, the clusters it was
    fitted on and the clusters whose units it positions, those of the other fold."""
    return dict(fold=check(fold), seed=SEED, assignment_sha256=identity(dataset), fitted=listed(clusters(dataset, fold)),
                positions=listed(clusters(dataset, other(fold))))


def fitted(dataset, fit):
    """The fold a fit record was fitted on, None for the whole training half. A fold entry of another assignment is
    refused, since its blocks can no longer be drawn again."""
    entry = fit.get("fold")
    if entry is None:
        return None
    if entry != record(dataset, entry.get("fold")):
        raise ValueError(f"the {dataset} fit of fold {entry.get('fold')} comes from another fold assignment than research/solid/folds.py "
                         "declares. Fit it again")
    return entry["fold"]


# ------------------------------------------------------------------------------------------- the units
def cluster_of(dataset, row):
    """The cluster of a unit row of a runner or of the decision rule."""
    if dataset == "tiers":
        return (row["acquisition"], row["block"])
    if dataset == "bregar":
        return row["acquisition"]
    if dataset == "idlab":
        return row["configuration"]
    raise ValueError(f"the round has no folds of {dataset}")


def check_rows(dataset, fold, rows):
    """Refuse the unit rows of a run of one fold that hold a unit of another fold or of no fold."""
    table = assignment(dataset)
    wrong = [cluster for cluster in (cluster_of(dataset, row) for row in rows) if table.get(cluster) != fold]
    if wrong:
        raise ValueError(f"the {dataset} run of fold {fold} holds {len(wrong)} units of other clusters, for example {sorted(set(map(str, wrong)))[:4]}")
    return rows
