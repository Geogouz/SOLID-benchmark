"""Maximum likelihood fit of the link model on training residual blocks.

A block is (y, s, d_true) with packet distances y, packet scales s and the true distance. The model is the block
likelihood of the method, `block_loglik(blocks, params)` of research/solid/link_likelihood.py, which returns the log
likelihood of every block at its true distance. The fitted values transfer directly into a profile.

The fit maximizes the weighted log likelihood over unconstrained coordinates of the free parameters.

| Parameter | Coordinate | Range |
| --- | --- | --- |
| kappa, the packet scale factor | log | 1e-3 to 1e3 |
| excess_probability | logit | 1e-6 to 1 - 1e-6 |
| excess_mean_rel | log | 1e-3 to 1e3 |
| persistent_sd_rel | log | 1e-4 to 1e3 |
| early_detection_per_cm, optional | log | 1e-8 to 1e-2 per cm, and zero |

The offset shape (Gaussian or Student t) and the degrees of freedom of the packet core are fixed within a fit, and a
reduced model fixes the parameters of the removed component (an excess probability of zero, an offset of zero). The
rate of false early detections is fitted only when asked and stays at zero otherwise. A reduced model keeps whatever
its full model fitted. The log coordinate of the rate cannot reach zero, so a fit whose rate ends on the lower edge
descends once more from its best point with the rate fixed at zero and keeps that point when it is likelier: the
estimate is the maximum over the closed interval from zero to 1e-2, and a rate of zero has no standard error.

The optimizer runs a declared grid of starts, every excess probability of PI_STARTS with every scale combination of
SCALE_COMBOS around moment estimates of the blocks. From each start a Nelder-Mead search with an explicit initial
simplex (a step of SIMPLEX_STEP in every coordinate) is followed by an L-BFGS-B polish with central difference
gradients. The best end point is then restarted, a fresh simplex and a fresh polish, until a restart gains less than
RESTART_GAIN nats, at most MAX_RESTARTS times. The fit is converged only when its final restart gained less than that.
The record keeps the negative log likelihood, the winning start, the end of every start, the number of distinct local
optima among them (end points more than OPTIMUM_TOLERANCE nats apart), and standard errors from the numerical Hessian
at the optimum together with the sandwich form that accounts for unequal block weights.

The earlier fitter ran Nelder-Mead with scipy's default simplex, which moves a coordinate that starts at zero by only
0.00025. Starting at a logit of zero, the excess probability never left 0.5.

The likelihood identity of every record is the SHA-256 of link_likelihood.py taken once when this module imports it, so
a record names the code that computed it even when the file changes while a fit runs. parity_identity() returns the
SHA-256 the engine's parity fixture was generated under, which must equal it for the parity tests to vouch for the fits.

fit4 keeps the flat record of plain numbers that older scripts print and store, computed by the same optimizer.
"""
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

NU = 3.0
LOG_T_NORM = math.lgamma((NU + 1) / 2) - math.lgamma(NU / 2) - 0.5 * math.log(NU * math.pi)
BGRID = np.concatenate([[0.0], np.geomspace(0.02, 12.0, 47)])
BW = np.diff(np.concatenate([BGRID, [BGRID[-1] + (BGRID[-1] - BGRID[-2])]]))
UGRID, UW = np.polynomial.hermite_e.hermegauss(20)   # probabilists' Hermite: weight exp(-u^2/2)
LOG_UW = np.log(UW / UW.sum())
GAUSSIAN_NU = 1.0e6


def _logt(e, s):
    return LOG_T_NORM - np.log(s) - 0.5 * (NU + 1) * np.log1p(e * e / (NU * s * s))


def set_nu(nu):
    """Set the degrees of freedom of the packet core of the legacy block_nll4 for this process."""
    global NU, LOG_T_NORM
    NU = float(nu)
    LOG_T_NORM = math.lgamma((NU + 1) / 2) - math.lgamma(NU / 2) - 0.5 * math.log(NU * math.pi)


def block_nll4(blocks, kappa, pi, mu_rel, u_rel, weights=None):
    """The legacy fitter's block model, a zero mean Gaussian offset on twenty Hermite nodes and the excess on the 48 node
    grid, the conventional median of the packet scales as the reference scale."""
    total = 0.0
    for idx, (y, s, d) in enumerate(blocks):
        ss = s * kappa
        sref = float(np.median(ss))
        mu = mu_rel * sref
        tau = u_rel * sref
        e = y - d                                   # (n,)
        u = tau * UGRID                              # (U,)
        eu = e[None, :] - u[:, None]                 # (U, n)
        spike = _logt(eu, ss[None, :]).sum(axis=1)   # (U,)
        b = mu * BGRID
        lpb = -b / mu - math.log(mu) + np.log(mu * BW)
        slab = np.logaddexp.reduce(_logt(eu[:, None, :] - b[None, :, None], ss[None, None, :]).sum(axis=2) + lpb[None, :], axis=1)  # (U,)
        if pi <= 0:
            ll_u = spike
        elif pi >= 1:
            ll_u = slab
        else:
            ll_u = np.logaddexp(math.log(1 - pi) + spike, math.log(pi) + slab)
        ll = np.logaddexp.reduce(ll_u + LOG_UW)
        total -= ll if weights is None else weights[idx] * ll
    return total


# ---------------------------------------------------------------------------------------------------------------


def _likelihood():
    """The likelihood of the method, link_likelihood.block_loglik, the same model the engine tabulates, with its source
    and the SHA-256 of the source as imported."""
    import link_likelihood
    source = Path(link_likelihood.__file__)
    return link_likelihood.block_loglik, "link_likelihood.block_loglik", source, hashlib.sha256(source.read_bytes()).hexdigest()


block_loglik, LIKELIHOOD_NAME, LIKELIHOOD_SOURCE, LIKELIHOOD_SHA256 = _likelihood()
PARITY_FIXTURE = HERE.parents[1] / "runtime/engine/tests/fixtures/link_likelihood_parity.json"


def likelihood_identity():
    """The name and source hash of the likelihood the fits use, stored with every fit record, the hash of the source
    this module imported."""
    return dict(name=LIKELIHOOD_NAME, source=LIKELIHOOD_SOURCE.name, sha256=LIKELIHOOD_SHA256)


def parity_identity():
    """The SHA-256 of the link_likelihood.py the engine's parity fixture was generated under."""
    return json.loads(PARITY_FIXTURE.read_text())["likelihood_sha256"]


# ---------------------------------------------------------------------------------------------------------------
# The optimizer
PARAMETERS = ("kappa", "excess_probability", "excess_mean_rel", "persistent_sd_rel", "early_detection_per_cm")
LINK_KEYS = PARAMETERS[:4]
RATE = "early_detection_per_cm"
BOUNDS = dict(kappa=(1e-3, 1e3), excess_probability=(1e-6, 1 - 1e-6), excess_mean_rel=(1e-3, 1e3), persistent_sd_rel=(1e-4, 1e3),
              early_detection_per_cm=(1e-8, 1e-2))
# the rates of false early detection every start of the grid is tried with, one negligible and one near Bern's
EARLY_DETECTION_STARTS = (1e-6, 1e-4)
# the value an optional parameter takes in a model that does not fit it, and so in records written before it existed
OPTIONAL_DEFAULTS = dict(early_detection_per_cm=0.0)


def parameter(record, name):
    """A parameter of a fit record, an optional one at its default when the record predates it."""
    return record[name] if name in record or name not in OPTIONAL_DEFAULTS else OPTIONAL_DEFAULTS[name]


PI_STARTS = (0.05, 0.25, 0.5, 0.8, 0.95)
SCALE_COMBOS = ((1.0, 1.0, 1.0), (1 / 3, 3.0, 3.0), (3.0, 1 / 3, 1 / 3))   # multipliers of the kappa, excess mean and offset anchors
FIXED_EXCESS_COMBOS = tuple((k, u) for k in (1 / 3, 1.0, 3.0) for u in (1 / 3, 1.0, 3.0))   # kappa and offset multipliers without the excess
SIMPLEX_STEP = 0.5
NM_OPTIONS = dict(xatol=1e-2, fatol=1e-2, maxfev=3000)     # the polish that follows refines the simplex's end
POLISH_OPTIONS = dict(maxiter=500, ftol=1e-12, gtol=1e-4, maxls=20)
GRADIENT_STEP = 1e-4
HESSIAN_STEP = 1e-3
RESTART_GAIN = 0.01
MAX_RESTARTS = 5
OPTIMUM_TOLERANCE = 0.1
BOUND_MARGIN = 2 * HESSIAN_STEP                       # a coordinate this close to its bound counts as at the bound
WALL = 1e6
INVALID = 1e300


def _encode(name, value):
    if name == "excess_probability":
        return float(logit(value))
    return math.log(value)


def _decode(name, z):
    if name == "excess_probability":
        return float(expit(z))
    return math.exp(z)


def _box(name):
    low, high = BOUNDS[name]
    return _encode(name, low), _encode(name, high)


def _robust_sd(values):
    values = np.asarray(values, dtype=float)
    return float(1.4826 * np.median(np.abs(values - np.median(values)))) if len(values) > 1 else 0.0


def moment_anchors(blocks, nu=NU):
    """Rough moment estimates around which the start grid is laid out.

    The packet scale factor comes from the spread of the packets inside blocks of three or more packets over their
    median scale, the offset and the excess mean from the spread of the block median residuals over the effective
    scale."""
    if not blocks:
        raise ValueError("no blocks to fit")
    t_mad = float(student_t.ppf(0.75, min(nu, 1e6)))
    ratios, medians, scales = [], [], []
    for y, s, d in blocks:
        y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
        medians.append(float(np.median(y)) - float(d))
        scales.append(float(np.median(s)))
        if len(y) >= 3:
            ratios.append(float(np.median(np.abs(y - np.median(y)))) / t_mad / float(np.median(s)))
    positive = [r for r in ratios if r > 0]
    kappa = float(np.clip(np.median(positive), 0.01, 100.0)) if positive else 0.3
    spread = _robust_sd(medians)
    offset = float(np.clip(spread / (kappa * max(float(np.median(scales)), 1e-9)), 0.3, 30.0))
    return dict(kappa=kappa, excess_mean_rel=offset, persistent_sd_rel=offset)


def start_grid(free, anchors):
    """The declared starts in natural units, every excess probability of PI_STARTS with every scale combination of
    SCALE_COMBOS, or the nine kappa and offset combinations of FIXED_EXCESS_COMBOS when the excess is fixed. A free rate
    of false early detections crosses every start with EARLY_DETECTION_STARTS. Starts that coincide on the free
    parameters are kept once."""
    grid = []
    if "excess_probability" in free:
        for pi in PI_STARTS:
            for k, m, u in SCALE_COMBOS:
                grid.append(dict(kappa=anchors["kappa"] * k, excess_probability=pi, excess_mean_rel=anchors["excess_mean_rel"] * m,
                                 persistent_sd_rel=anchors["persistent_sd_rel"] * u))
    else:
        for k, u in FIXED_EXCESS_COMBOS:
            grid.append(dict(kappa=anchors["kappa"] * k, excess_probability=0.5, excess_mean_rel=anchors["excess_mean_rel"],
                             persistent_sd_rel=anchors["persistent_sd_rel"] * u))
    rates = EARLY_DETECTION_STARTS if "early_detection_per_cm" in free else EARLY_DETECTION_STARTS[:1]
    grid = [dict(start, early_detection_per_cm=rate) for start in grid for rate in rates]
    unique, seen = [], set()
    for start in grid:
        key = tuple(round(start[name], 12) for name in free)
        if key not in seen:
            seen.add(key)
            unique.append({name: start[name] for name in free})
    return unique


class LinkObjective:
    """The weighted negative log likelihood of the blocks over the free parameters in unconstrained coordinates."""

    def __init__(self, blocks, weights, free, fixed, likelihood):
        self.blocks = blocks
        self.weights = np.ones(len(blocks)) if weights is None else np.asarray(weights, dtype=float)
        if self.weights.shape != (len(blocks),):
            raise ValueError("one weight per block")
        self.free = tuple(free)
        self.fixed = dict(fixed)
        self.likelihood = likelihood
        boxes = [_box(name) for name in self.free]
        self.lower = np.array([b[0] for b in boxes])
        self.upper = np.array([b[1] for b in boxes])
        self.evaluations = 0

    def params(self, z):
        params = dict(self.fixed)
        for name, value in zip(self.free, np.clip(np.asarray(z, dtype=float), self.lower, self.upper)):
            params[name] = _decode(name, value)
        return params

    def encode(self, natural):
        z = np.array([_encode(name, float(np.clip(natural[name], *BOUNDS[name]))) for name in self.free])
        return np.clip(z, self.lower, self.upper)

    def block_values(self, z):
        self.evaluations += 1
        return np.asarray(self.likelihood(self.blocks, self.params(z)), dtype=float)

    def value(self, z):
        total = -float(np.dot(self.weights, self.block_values(z)))
        return total if math.isfinite(total) else INVALID

    def penalized(self, z):
        """The objective of the simplex search, continued outside the box by a quadratic wall."""
        z = np.asarray(z, dtype=float)
        inside = np.clip(z, self.lower, self.upper)
        return self.value(inside) + WALL * float(np.sum((z - inside) ** 2))

    def value_and_gradient(self, z):
        z = np.clip(np.asarray(z, dtype=float), self.lower, self.upper)
        f = self.value(z)
        gradient = np.empty(len(z))
        for i in range(len(z)):
            up, down = z.copy(), z.copy()
            up[i] = min(z[i] + GRADIENT_STEP, self.upper[i])
            down[i] = max(z[i] - GRADIENT_STEP, self.lower[i])
            gradient[i] = (self.value(up) - self.value(down)) / (up[i] - down[i])
        return f, gradient


def _simplex(z, lower, upper):
    rows = [z]
    for i in range(len(z)):
        vertex = z.copy()
        vertex[i] += SIMPLEX_STEP if z[i] + SIMPLEX_STEP <= upper[i] else -SIMPLEX_STEP
        rows.append(vertex)
    return np.array(rows)


def _descend(objective, z0):
    """Nelder-Mead from z0 with the explicit simplex, then the L-BFGS-B polish inside the box."""
    z0 = np.clip(np.asarray(z0, dtype=float), objective.lower, objective.upper)
    simplex = minimize(objective.penalized, z0, method="Nelder-Mead",
                       options=dict(initial_simplex=_simplex(z0, objective.lower, objective.upper), **NM_OPTIONS))
    start = np.clip(simplex.x, objective.lower, objective.upper)
    start_value = objective.value(start)
    polish = minimize(objective.value_and_gradient, start, jac=True, method="L-BFGS-B",
                      bounds=list(zip(objective.lower, objective.upper)), options=POLISH_OPTIONS)
    better = math.isfinite(polish.fun) and polish.fun <= start_value
    return dict(z=np.clip(polish.x, objective.lower, objective.upper) if better else start, nll=float(polish.fun) if better else start_value,
                iterations=int(simplex.nit) + int(polish.nit), polish_success=bool(polish.success), polish_message=str(polish.message))


def _descend_start(task):
    """One start of the grid in a worker process, with its own count of likelihood evaluations."""
    blocks, weights, free, fixed, likelihood, z0 = task
    objective = LinkObjective(blocks, weights, free, fixed, likelihood)
    run = _descend(objective, z0)
    run["evaluations"] = objective.evaluations
    return run


def distinct_optima(values, tolerance=OPTIMUM_TOLERANCE):
    """End values grouped greedily from the lowest, a group holding the values within `tolerance` nats of its lowest."""
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1] > tolerance:
            groups.append(value)
    return groups


def _curvature(objective, z):
    """The numerical Hessian of the negative log likelihood and the per block score outer products over the free
    coordinates that are not at a bound."""
    interior = [i for i in range(len(z)) if objective.lower[i] + BOUND_MARGIN < z[i] < objective.upper[i] - BOUND_MARGIN]
    k = len(interior)
    if k == 0:
        return interior, None, None
    f0 = objective.value(z)
    h = HESSIAN_STEP

    def shifted(steps):
        point = z.copy()
        for i, step in steps:
            point[i] += step
        return objective.value(point)

    hessian = np.empty((k, k))
    for a, i in enumerate(interior):
        hessian[a, a] = (shifted([(i, h)]) - 2 * f0 + shifted([(i, -h)])) / (h * h)
        for b in range(a):
            j = interior[b]
            value = (shifted([(i, h), (j, h)]) - shifted([(i, h), (j, -h)]) - shifted([(i, -h), (j, h)]) + shifted([(i, -h), (j, -h)])) / (4 * h * h)
            hessian[a, b] = hessian[b, a] = value
    scores = np.empty((len(objective.blocks), k))
    for a, i in enumerate(interior):
        up, down = z.copy(), z.copy()
        up[i] += GRADIENT_STEP
        down[i] -= GRADIENT_STEP
        scores[:, a] = (objective.block_values(up) - objective.block_values(down)) / (2 * GRADIENT_STEP)
    weighted = scores * objective.weights[:, None]
    return interior, hessian, weighted.T @ weighted


def _natural_se(name, value, se):
    if name == "excess_probability":
        return value * (1 - value) * se
    return value * se


def _clean(value):
    """Plain Python numbers for the JSON records."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def validate_blocks(blocks):
    """Refuse a block without packets, with scales that do not match its packets, or with values that are not finite
    or scales that are not positive, before any of them turns the likelihood into NaN."""
    if not blocks:
        raise ValueError("no blocks to fit")
    for index, (y, s, d) in enumerate(blocks):
        y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
        if y.ndim != 1 or len(y) == 0 or s.shape != y.shape:
            raise ValueError(f"block {index} has no packets or scales that do not match its packets")
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s)) and np.all(s > 0) and math.isfinite(float(d))):
            raise ValueError(f"block {index} holds a value that is not finite or a scale that is not positive")


def fit_link_model(blocks, weights=None, *, offset_shape="gaussian", fit_early_detection=False, nu=None, fixed=None, extra=None,
                   likelihood=None, grid=None, starts=None, workers=1):
    """Maximum likelihood fit of the link model on residual blocks. Returns the fit record.

    blocks are (y, s, d_true) triples and weights one weight per block (all one by default). offset_shape is "gaussian"
    or "student_t" and nu the degrees of freedom of the packet core. With fit_early_detection the rate of false early
    detections per cm is fitted over the closed interval from zero to its upper bound, otherwise it stays at zero. fixed
    holds parameters held at given values (the reduced models), extra further keys passed to the likelihood unchanged.
    likelihood defaults to block_loglik and grid to start_grid, and starts, in natural units, run after the grid. With
    workers above one the starts run in that many processes, which changes nothing but the wall time."""
    validate_blocks(blocks)
    nu = NU if nu is None else float(nu)
    likelihood = likelihood or block_loglik
    fixed = dict(fixed or {})
    if not fit_early_detection:
        fixed.setdefault(RATE, 0.0)
    unknown = set(fixed) - set(PARAMETERS)
    if unknown:
        raise ValueError(f"unknown fixed parameters {sorted(unknown)}")
    free = [name for name in PARAMETERS if name not in fixed]
    anchors = moment_anchors(blocks, nu)
    base = dict(extra or {}, offset_shape=offset_shape, nu=nu)
    objective = LinkObjective(blocks, weights, free, dict(base, **fixed), likelihood)
    starts = [dict(s) for s in (grid if grid is not None else start_grid(free, anchors))] + [dict(s) for s in starts or ()]
    tasks = [(blocks, objective.weights, free, objective.fixed, likelihood, objective.encode(start)) for start in starts]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            runs = list(pool.map(_descend_start, tasks))
    else:
        runs = [_descend_start(task) for task in tasks]
    objective.evaluations += sum(run["evaluations"] for run in runs)
    winner = int(np.argmin([run["nll"] for run in runs]))
    best, gains, converged, iterations = runs[winner], [], False, sum(run["iterations"] for run in runs)
    for _ in range(MAX_RESTARTS):
        again = _descend(objective, best["z"])
        iterations += again["iterations"]
        gain = best["nll"] - again["nll"]
        gains.append(gain)
        if again["nll"] < best["nll"]:
            best = again
        if gain < RESTART_GAIN:
            converged = True
            break
    # a rate of false early detections on its lower edge descends once more at zero, the closed end of its interval,
    # and keeps that point when it is likelier, with the curvature of the other parameters
    measured, z, at_zero, others = objective, best["z"], None, None
    if RATE in free and best["z"][free.index(RATE)] <= objective.lower[free.index(RATE)] + BOUND_MARGIN:
        others = LinkObjective(blocks, objective.weights, [name for name in free if name != RATE], dict(objective.fixed, **{RATE: 0.0}), likelihood)
        zero = _descend(others, np.delete(best["z"], free.index(RATE)))
        iterations += zero["iterations"]
        at_zero = dict(nll=zero["nll"], edge_nll=best["nll"], kept=zero["nll"] < best["nll"])
        if at_zero["kept"]:
            measured, z, best = others, zero["z"], zero
    params = measured.params(z)
    interior, hessian, meat = _curvature(measured, z)
    se, sandwich, positive = {name: None for name in free}, {name: None for name in free}, None
    if hessian is not None:
        try:
            np.linalg.cholesky(hessian)
            positive = True
        except np.linalg.LinAlgError:
            positive = False
        if positive:
            covariance = np.linalg.inv(hessian)
            robust = covariance @ meat @ covariance
            for a, i in enumerate(interior):
                name = measured.free[i]
                se[name] = _natural_se(name, params[name], math.sqrt(covariance[a, a]))
                sandwich[name] = _natural_se(name, params[name], math.sqrt(max(robust[a, a], 0.0)))
    _, gradient = measured.value_and_gradient(z)
    at_bounds = [name for i, name in enumerate(measured.free) if i not in interior] + ([RATE] if measured is others else [])
    evaluations = objective.evaluations + (others.evaluations if others else 0)
    record = dict(
        {name: params[name] for name in PARAMETERS}, offset_shape=offset_shape, nu=nu, extra=dict(extra or {}),
        nll=best["nll"], converged=converged, iterations=iterations, evaluations=evaluations,
        standard_errors=se, standard_errors_sandwich=sandwich, free=free, fixed=sorted(fixed),
        winning_start=winner, distinct_optima=len(distinct_optima([run["nll"] for run in runs])),
        blocks=len(blocks), weight_sum=float(objective.weights.sum()),
        optimizer=dict(
            method="declared start grid, Nelder-Mead with an explicit simplex then an L-BFGS-B polish from every start, "
                   "restarts from the best point until a restart gains under the restart gain, and a rate of false early "
                   "detections on its lower edge descended once more at zero",
            simplex_step=SIMPLEX_STEP, restart_gain_nats=RESTART_GAIN, max_restarts=MAX_RESTARTS, optimum_tolerance_nats=OPTIMUM_TOLERANCE,
            anchors=anchors, restart_gains=gains, at_bounds=at_bounds, hessian_positive_definite=positive, rate_at_zero=at_zero,
            gradient_max_abs=float(np.max(np.abs([g for i, g in enumerate(gradient) if i in interior]))) if interior else 0.0,
            polish=dict(success=best["polish_success"], message=best["polish_message"]),
            optima_nll=distinct_optima([run["nll"] for run in runs]),
            starts=[dict(start=start, nll=run["nll"], end={name: _decode(name, v) for name, v in zip(free, run["z"])})
                    for start, run in zip(starts, runs)]),
        likelihood=likelihood_identity() if likelihood is block_loglik else dict(name=getattr(likelihood, "__name__", "custom")))
    return _clean(record)


def model_nll(blocks, weights, record, likelihood=None):
    """The weighted negative log likelihood of the blocks at the parameters of a fit record."""
    params = dict(record.get("extra") or {})
    params.update({key: record[key] for key in PARAMETERS if key in record})
    params.update(offset_shape=record.get("offset_shape", "gaussian"), nu=record.get("nu", NU))
    weights = np.ones(len(blocks)) if weights is None else np.asarray(weights, dtype=float)
    return -float(np.dot(weights, (likelihood or block_loglik)(blocks, params)))


def model_settings(record):
    """The settings that produced a fit record (offset shape, fitted early detection rate, nu, extra likelihood keys), for
    refits of reduced models on the same blocks."""
    free = record.get("free") or []
    return dict(offset_shape=record.get("offset_shape", "gaussian"), fit_early_detection=RATE in free, nu=record.get("nu", NU),
                extra=record.get("extra") or None)


def fit4(blocks, weights=None):
    """The four parameter fit of the method (Gaussian offset, the packet core of NU) by fit_link_model, in the legacy flat
    record of plain numbers."""
    record = fit_link_model(blocks, weights, nu=NU)
    return {key: record[key] for key in ("kappa", "excess_probability", "excess_mean_rel", "persistent_sd_rel", "nll", "converged", "iterations")}


# ---------------------------------------------------------------------------------------------------------------
# Strata and weights
def stratified_weights(strata, within=None):
    """Block weights with the same total weight in every stratum, summing to the block count. A stratum is a sample
    track (the unit's sample count), never a raw packet count. `within` gives relative weights inside a stratum, equal by
    default."""
    strata = list(strata)
    count = len(strata)
    within = np.ones(count) if within is None else np.asarray(within, dtype=float)
    labels = sorted(set(strata), key=str)
    weights = np.empty(count)
    for label in labels:
        members = np.array([i for i, s in enumerate(strata) if s == label])
        weights[members] = within[members] / within[members].sum() * count / len(labels)
    return weights


def stratum_totals(strata, weights):
    totals = defaultdict(float)
    for stratum, weight in zip(strata, weights):
        totals[str(stratum)] += float(weight)
    return dict(sorted(totals.items()))


# ---------------------------------------------------------------------------------------------------------------
# Reduced models of the ablation
VARIANTS = ("no_excess", "no_offset", "gaussian_core", "independent_packets")
# the reduced models inside the parameter space of the full model, whose fits can never beat the full fit
NESTED = ("no_excess", "no_offset")


def variant_settings(variant, settings):
    """The fit settings of a reduced model from the settings of the full model (offset shape, early detection, nu,
    fixed)."""
    settings = dict(settings)
    fixed = dict(settings.pop("fixed", None) or {})
    if variant == "no_excess":
        fixed.update(excess_probability=0.0, excess_mean_rel=1.0)
    elif variant == "no_offset":
        fixed.update(persistent_sd_rel=0.0)
    elif variant == "gaussian_core":
        settings["nu"] = GAUSSIAN_NU
    elif variant != "independent_packets":
        raise ValueError(f"unknown reduced model {variant}")
    return dict(settings, fixed=fixed)


def single_packets(blocks, weights):
    """Every packet its own block with the weight of its block, the data of the independent packets model."""
    singles, single_weights = [], []
    for index, (y, s, d) in enumerate(blocks):
        for k in range(len(y)):
            singles.append((np.asarray(y[k:k + 1], dtype=float), np.asarray(s[k:k + 1], dtype=float), d))
            single_weights.append(1.0 if weights is None else float(weights[index]))
    return singles, np.asarray(single_weights)


def full_start(full, model):
    """The full fit's optimum as a start of a reduced model's refit, the full fit's parameters on the parameters the
    reduced model frees, its removed components at their fixed values."""
    if not full:
        return None
    fixed = set(model.get("fixed") or {}) | (set() if model.get("fit_early_detection") else {RATE})
    return [{name: parameter(full, name) for name in PARAMETERS if name not in fixed}]


def fit_variant(blocks, weights, variant, full=None, **settings):
    """The link model without one component, refitted by the same optimizer on the same blocks and weights, the
    reduced model at its own best fit. settings are those of the full fit (offset_shape, fit_early_detection, nu,
    extra), and full, the full fit's record, adds its optimum as one more start."""
    model = variant_settings(variant, settings)
    if variant == "independent_packets":
        blocks, weights = single_packets(blocks, weights)
    return fit_link_model(blocks, weights, starts=full_start(full, model), **model)


def embedded_params(variant, reduced):
    """The reduced optimum moved into the full parameter space, at the bound that stands for the removed component."""
    params = {key: parameter(reduced, key) for key in PARAMETERS}
    if variant == "no_excess":
        params["excess_probability"] = BOUNDS["excess_probability"][0]
    elif variant == "no_offset":
        params["persistent_sd_rel"] = BOUNDS["persistent_sd_rel"][0]
    else:
        raise ValueError(f"{variant} is not nested in the full model")
    return params
