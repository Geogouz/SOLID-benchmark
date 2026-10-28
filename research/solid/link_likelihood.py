"""The exact link likelihood of SOLID, the Python counterpart of the engine's runtime/engine/src/link.rs.

Model. A link carries packets y_k (cm) with reported scales s_k. The core scale of packet k is sigma_k = kappa * s_k,
floored at 1e-3 cm, and its core density is Student t with nu degrees of freedom. The reference scale s_ref is the
conventional median of the sigma_k, the mean of the two middle values for an even count. The packets share one total
shift w = u + b: u the persistent offset, zero mean, Gaussian with standard deviation tau = persistent_sd_rel * s_ref
(offset_shape "gaussian") or Student t with OFFSET_NU = 3 degrees of freedom and scale tau ("student_t"), whatever the
degrees of freedom of the packet core; b >= 0 the excess, zero with probability 1 - pi and otherwise exponential with
mean mu = excess_mean_rel * s_ref (floored at 1e-3 cm), pi = excess_probability. With g(v) = prod_k t_nu(y_k - v; sigma_k),

    L(d) = integral g(v) f_w(v - d) dv,

f_w the density of w. The ablation independent_packets makes L the product of single packet likelihoods, each with its
own s_ref = sigma_k. With the false early detection rate lambda = early_detection_per_cm > 0 each packet's density is
p(y | s) = exp(-lambda max(s, 0)) t_nu(y - s; sigma) + lambda exp(-lambda y) 1{0 <= y < s} at the detection distance
s = d + w, which integrates to one over y, and g(s) = prod_k p(y_k | s); far beyond every reading L levels off at
prod_k lambda exp(-lambda y_k). A zero rate gives the model without the term, value for value.

Numerics. The integral runs in the variable t of v = c + a sinh(t) around every centre c, Gauss-Legendre on panels graded
towards the centre: every reading, where that packet's core carries the integrand while the others lie in their tails
or read early (a separated cluster, an outlier, a false early detection) and where its early reading switches on; the
mode of g; d, where the shift density has its offset peak and its excess jump; between the mode and d the peak of the
product of two Gaussians of the packet and offset widths, where the integrand concentrates far from the packets when
both are nearly Gaussian; the peak of g times the offset density, where it concentrates far beyond the packets when
their tails fall slower than the offset's; and with early detections zero distance, below which no packet has to
survive them, so that at a high rate over a long link the integrand concentrates there. Consecutive centres split the
line at their midpoint, so every sharp feature has panels of its own, every jump falls on a panel edge, and the heavy
tails need few nodes. No centre's scale exceeds another's plus their distance, so the panels grade from a narrow feature
to a wide one next to it. A narrow offset within a few of its widths of a reading otherwise lost part of its mass to the
reading's wide panels, and at 1e-2 early detections per cm over 57 m four packets lost 10 nats below zero distance to
the wide panels of the nearest centre. The slab density of a Student t offset, t convolved with the exponential, is
tabulated once per excess mean in units of tau by the exact recursion of the exponential convolution. Everything runs in
the log domain.

block_loglik(blocks, params) returns the log likelihood of every block at its true distance, table(y, s, params, d_grid)
the log likelihood of one block at every distance of d_grid.
"""
import math

import numpy as np
from scipy import special

# the degrees of freedom of a Student t offset, whatever those of the packet core
OFFSET_NU = 3.0
# the floor of the excess mean in centimetres
MU_FLOOR_CM = 1e-3
# Newton steps of the searches for panel centres (_search), those every block takes and the most in all, and the move
# of an iterate, relative to its step limit, below which a search has converged
SEARCH_STEPS = (25, 2000)
SEARCH_TOLERANCE = 1e-9
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_PANELS = 6
_FAR_CM = 1e7
_CHUNK = 256
_SLAB_CACHE = {}


def _settings(params):
    shape = params.get("offset_shape", "gaussian")
    if shape not in ("gaussian", "student_t"):
        raise ValueError(f"unknown offset shape {shape}")
    if float(params.get("location_cm", 0.0) or 0.0) != 0.0:
        raise ValueError("the link model has no location term")
    return dict(
        kappa=float(params["kappa"]),
        pi=float(min(max(params["excess_probability"], 0.0), 1.0)),
        m=float(params["excess_mean_rel"]),
        t=float(params["persistent_sd_rel"]),
        shape=shape,
        nu=float(params.get("nu", 3.0)),
        independent=bool(params.get("independent_packets", False)),
        lam=max(float(params.get("early_detection_per_cm", 0.0) or 0.0), 0.0),
    )


def _log_t_norm(nu):
    return math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)


def _log_t(e, s, nu, norm):
    return norm - np.log(s) - 0.5 * (nu + 1) * np.log1p(e * e / (nu * s * s))


def _log(x):
    with np.errstate(divide="ignore"):
        return np.log(x)


def _log_packets(y, sig, v, nu, norm, lam):
    """log of the packet product at the detection distance v, summed over the last axis of the broadcast of y, sig and
    v: every packet's core, surviving a false early detection with probability exp(-lam max(v, 0)), mixed with the
    early reading lam exp(-lam y) for 0 <= y < v."""
    core = _log_t(y - v, sig, nu, norm)
    if lam <= 0:
        return core.sum(axis=-1)
    core = core - lam * np.maximum(v, 0.0)
    early = np.where((y >= 0) & (y < v), math.log(lam) - lam * y, -np.inf)
    return np.logaddexp(core, early).sum(axis=-1)


class _StudentSlab:
    """log of (t_nu * Exp) in units of the offset scale: q(z) = integral_0^inf exp(-b / r) / r t_nu(z - b) db, r = mu / tau,
    on a grid of step 0.02 within 5 offset scales of the peak, growing geometrically by 0.4 % beyond out to 2000 + 50 r
    on either side, by the exact recursion of the exponential convolution with Gauss-Legendre steps in the variable of
    z = sinh(xi), and its asymptotic form beyond the grid. A near zero offset makes r large (1129 for the UAV
    profile), and the left tail, the offset's tail averaged over an excess as wide as its distance, needs the grid out
    to many excess means before the asymptotic form holds. An excess narrower than a step of the grid, as the floor of
    the excess mean makes it under a wide offset, decays within the step, and there the step runs in the variable
    u = (hi - s) / r on panels growing from the step's upper end."""

    def __init__(self, nu, r):
        self.nu, self.r, self.norm = nu, r, _log_t_norm(nu)
        z1 = 2000.0 + 50.0 * r
        z0 = -z1
        outer = 5.0 * 1.004 ** np.arange(int(math.log(z1 / 5.0) / math.log(1.004)) + 2)
        z = np.unique(np.concatenate([-outer[outer <= -z0][::-1], [z0], np.linspace(-5.0, 5.0, 501), outer]))
        z = z[(z >= z0) & (z <= outer[outer >= z1][0])]
        lo, hi = z[:-1], z[1:]
        xa, xb = np.arcsinh(lo), np.arcsinh(hi)
        half, mid = 0.5 * (xb - xa), 0.5 * (xa + xb)
        xi = mid[:, None] + half[:, None] * _GL_X[None, :]
        s = np.sinh(xi)
        # in the log domain, so that a nearly Gaussian offset (large nu) keeps its far tails finite
        log_terms = -(hi[:, None] - s) / r - math.log(r) + _log_t(s, 1.0, nu, self.norm) + np.log(np.cosh(xi) * half[:, None] * _GL_W[None, :])
        log_seg = special.logsumexp(log_terms, axis=1)
        # steps over which the excess decays: integral_0^U exp(-u) t_nu(hi - r u) du, U = (hi - lo) / r, on panels ending
        # at 1, 3, 7, 15 and 31 or at U
        sharp = (hi - lo) > r
        if sharp.any():
            span = (hi[sharp] - lo[sharp]) / r
            edges = np.minimum(np.array([0.0, 1.0, 3.0, 7.0, 15.0, 31.0])[None, :], span[:, None])
            a, b = edges[:, :-1], edges[:, 1:]
            u = 0.5 * (a + b)[..., None] + 0.5 * (b - a)[..., None] * _GL_X
            weight = 0.5 * (b - a)[..., None] * _GL_W
            log_panels = -u + _log_t(hi[sharp][:, None, None] - r * u, 1.0, nu, self.norm) + _log(weight)
            log_seg[sharp] = special.logsumexp(log_panels.reshape(len(span), -1), axis=1)
        # the first value by panels of one excess mean over sixty means, where the t density is smooth
        b = (np.arange(60)[:, None] + 0.5 + 0.5 * _GL_X[None, :]) * r
        wb = 0.5 * r * np.broadcast_to(_GL_W[None, :], b.shape)
        log_q = np.empty(len(z))
        log_q[0] = float(special.logsumexp(-b / r - math.log(r) + _log_t(z[0] - b, 1.0, nu, self.norm) + np.log(wb)))
        log_decay = -(hi - lo) / r
        for k in range(len(log_seg)):
            log_q[k + 1] = np.logaddexp(log_decay[k] + log_q[k], log_seg[k])
        self.z, self.log_q = z, log_q

    def __call__(self, z):
        z = np.asarray(z, float)
        value = np.interp(z, self.z, self.log_q)
        # beyond the grid: on the left the offset's tail times the moment generating function of the excess at its log
        # slope, on the right the offset moved by the excess mean plus the exponential tail
        slope = (self.nu + 1) * np.abs(z) / (self.nu + z * z)
        left = _log_t(z, 1.0, self.nu, self.norm) - np.log1p(slope * self.r)
        far = _log_t(z - self.r, 1.0, self.nu, self.norm)
        with np.errstate(over="ignore"):
            exponential = np.where(z > 0, -z / self.r - math.log(self.r), -np.inf)
        outside = np.where(z < 0, left, np.logaddexp(far, exponential))
        return np.where((z < self.z[0]) | (z > self.z[-1]), outside, value)


def _student_slab(nu, r):
    key = (round(nu, 12), round(r, 12))
    if key not in _SLAB_CACHE:
        if len(_SLAB_CACHE) > 8:
            _SLAB_CACHE.clear()
        _SLAB_CACHE[key] = _StudentSlab(nu, r)
    return _SLAB_CACHE[key]


def _slab_values(z, r):
    """log slab density of a Student t offset in units of its scale at z, shape (B, N), for the excess mean r of every
    block in the same units, shape (B,), one tabulation per distinct r."""
    out = np.empty_like(z)
    for value in np.unique(r):
        rows = r == value
        out[rows] = _student_slab(OFFSET_NU, value)(z[rows])
    return out


def _log_shift(w, tau, mu, st, r):
    """log density of u + b at w. tau and mu broadcast against w, and r holds the excess mean of every block in units of
    its offset scale. Without an offset only the excess part is returned, the spike is added by the caller."""
    pi = st["pi"]
    log_spike = math.log1p(-pi) if pi < 1 else -np.inf
    log_slab = math.log(pi) if pi > 0 else -np.inf
    if st["t"] <= 0:
        return np.where(w >= 0, log_slab - np.log(mu) - np.maximum(w, 0) / mu, -np.inf)
    if st["shape"] == "gaussian":
        spike = log_spike - 0.5 * (w / tau) ** 2 - np.log(tau) - 0.5 * math.log(2 * math.pi)
        emg = log_slab - np.log(mu) + tau * tau / (2 * mu * mu) - w / mu + special.log_ndtr(w / tau - tau / mu)
        return np.logaddexp(spike, emg)
    spike = log_spike + _log_t(w, tau, OFFSET_NU, _log_t_norm(OFFSET_NU))
    return np.logaddexp(spike, log_slab + _slab_values(w / tau, r) - np.log(tau))


def _search(v, bound, slopes, lo=-np.inf, hi=np.inf):
    """Newton steps on a log density of every block from v, slopes(v, rows) returning its first two derivatives at v
    for the blocks rows. A step is limited to the block's limit and taken at the limit along the gradient where the log
    is convex, and the iterate stays within lo and hi. Every block takes the first SEARCH_STEPS[0] steps with the limit
    bound. A block then steps on while its next step moves it by more than SEARCH_TOLERANCE of bound beyond the
    rounding of v, up to SEARCH_STEPS[1] steps in all. Its limit doubles at every step that reaches the limit towards
    no point met beyond, and a step that would leave the bracket of the points met on either side of the peak goes to
    the middle of the bracket instead. A far peak, as a nearly Gaussian core with one packet metres from the rest puts
    the mode, is then reached in as many steps as it takes to double the limit to it, and a step at the limit over a
    peak narrower than the limit, back and forth, ends in the bracket. The first steps alone once stopped short of a
    far mode, and the likelihood of such a block at its true distance came out up to 1,285 nats too low."""
    def newton(v, rows, limit):
        d1, d2 = slopes(v, rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            return d1, np.where(d2 < 0, -d1 / d2, np.sign(d1) * limit)

    lo, hi = np.broadcast_to(lo, v.shape), np.broadcast_to(hi, v.shape)
    for _ in range(SEARCH_STEPS[0]):
        v = np.clip(v + np.clip(newton(v, slice(None), bound)[1], -bound, bound), lo, hi)
    rows = np.arange(len(v))
    limit, left, right = bound.copy(), np.full(len(v), -np.inf), np.full(len(v), np.inf)
    for _ in range(SEARCH_STEPS[0], SEARCH_STEPS[1]):
        at = v[rows]
        d1, raw = newton(at, rows, limit[rows])
        left[rows] = np.where(d1 > 0, at, left[rows])
        right[rows] = np.where(d1 < 0, at, right[rows])
        step = np.clip(raw, -limit[rows], limit[rows])
        after = at + step
        inside = (after >= left[rows]) & (after <= right[rows])
        with np.errstate(invalid="ignore"):
            after = np.clip(np.where(inside, after, 0.5 * (left[rows] + right[rows])), lo[rows], hi[rows])
        moving = np.abs(after - at) > SEARCH_TOLERANCE * bound[rows] + 4.0 * np.spacing(np.abs(at))
        if not moving.any():
            break
        ahead = np.where(step > 0, right[rows] == np.inf, left[rows] == -np.inf)
        limit[rows] = np.where((np.abs(raw) >= limit[rows]) & ahead, 2.0 * limit[rows], limit[rows])
        rows = rows[moving]
        v[rows] = after[moving]
    return v


def _mode(y, sig, nu, start):
    """A mode of the packet product by Newton steps on its log from `start` (_search), limited to two core scales."""
    a = nu * sig * sig

    def slopes(v, rows):
        e = y[rows] - v[:, None]
        q = a[rows] + e * e
        return ((nu + 1) * e / q).sum(axis=1), ((nu + 1) * (e * e - a[rows]) / (q * q)).sum(axis=1)

    return _search(start.copy(), 2.0 * sig.max(axis=1), slopes)


def _offset_peak(y, sig, nu, d, c1, tau, shape):
    """The peak of the packet product times the offset density at v - d, and the width of that product there. Newton
    steps on its log (_search) run from d with the early readings left out, limited to two offset scales and kept
    between the mode c1 and d. Where the log is convex the width is the offset scale. Without an offset the peak is d,
    of infinite width, so the scale of the centre at d holds. Where the packet tails fall slower than the offset's, as
    those of a core of some tens of degrees of freedom under a Gaussian offset, the integrand far beyond the packets
    peaks there, away from both the mode and d."""
    a = nu * sig * sig
    scale = np.where(tau > 0, tau, 1.0)

    def slopes(v, rows):
        e = y[rows] - v[:, None]
        q = a[rows] + e * e
        w = v - d[rows]
        if shape == "gaussian":
            o1, o2 = -w / scale[rows] ** 2, -1.0 / scale[rows] ** 2
        else:
            b = OFFSET_NU * scale[rows] * scale[rows]
            o1, o2 = -(OFFSET_NU + 1) * w / (b + w * w), -(OFFSET_NU + 1) * (b - w * w) / (b + w * w) ** 2
        return ((nu + 1) * e / q).sum(axis=1) + o1, ((nu + 1) * (e * e - a[rows]) / (q * q)).sum(axis=1) + o2

    v = _search(d.copy(), 2.0 * scale, slopes, np.minimum(c1, d), np.maximum(c1, d))
    d2 = slopes(v, slice(None))[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        width = np.where(d2 < 0, 1.0 / np.sqrt(-d2), scale)
    return np.where(tau > 0, v, d), np.where(tau > 0, width, np.inf)


def _centres(y, sig, nu, norm, lam, d, s_g, a_d, s_u, tau, shape):
    """Centres and scales of the panels, sorted: every reading, where that packet's core carries the integrand while the
    others lie in their tails or read early, and where its early reading switches on; the mode of the packet product,
    by Newton steps from the reading where the product is largest; d; between the mode and d the peak of the product
    of two Gaussians of the packet width s_g and the offset width s_u, where the integrand concentrates when both are
    nearly Gaussian; the peak of the packet product times the offset density (_offset_peak); and with early
    detections zero distance, with the length over which the packets' survival falls by a factor e. No scale exceeds
    another centre's scale plus the distance between the two."""
    at_readings = _log_packets(y[:, None, :], sig[:, None, :], y[:, :, None], nu, norm, lam)
    start = np.take_along_axis(y, np.argmax(at_readings, axis=1)[:, None], axis=1)[:, 0]
    c1 = _mode(y, sig, nu, start)
    width = np.where(s_u > 0, s_u, s_g)
    between = c1 + (d - c1) * s_g ** 2 / (s_g ** 2 + width ** 2)
    joint = 1.0 / np.sqrt(1.0 / s_g ** 2 + 1.0 / width ** 2)
    peak, peak_width = _offset_peak(y, sig, nu, d, c1, tau, shape)
    centres = np.concatenate([y, np.stack([c1, d, between, peak], axis=1)], axis=1)
    scales = np.concatenate([0.5 * sig, np.stack([0.5 * s_g, a_d, 0.5 * joint, 0.5 * peak_width], axis=1)], axis=1)
    if lam > 0:
        # zero distance, where the survival of the early detections starts. At a high rate over a long link the tails
        # of the cores and of a Student t offset below zero outweigh the readings.
        zero = np.zeros((len(y), 1))
        centres = np.concatenate([centres, zero], axis=1)
        scales = np.concatenate([scales, zero + 1.0 / (lam * y.shape[1])], axis=1)
    scales = np.min(scales[:, None, :] + np.abs(centres[:, :, None] - centres[:, None, :]), axis=2)
    idx = np.argsort(centres, axis=1, kind="stable")
    return np.take_along_axis(centres, idx, axis=1), np.take_along_axis(scales, idx, axis=1)


def _nodes(centres, scales):
    """Nodes v and log weights of the panels around every centre, the line split at midpoints between centres."""
    B, K = centres.shape
    mids = 0.5 * (centres[:, 1:] + centres[:, :-1])
    lower = np.concatenate([np.full((B, 1), -np.inf), mids], axis=1)
    upper = np.concatenate([mids, np.full((B, 1), np.inf)], axis=1)
    fractions = (np.arange(_PANELS + 1) / _PANELS) ** 2
    vs, lws = [], []
    for side, bound in ((-1.0, lower), (1.0, upper)):
        reach = np.where(np.isfinite(bound), np.abs(bound - centres), _FAR_CM)
        length = np.arcsinh(reach / scales)
        edges = length[..., None] * fractions
        a, b = edges[..., :-1], edges[..., 1:]
        half, mid = 0.5 * (b - a), 0.5 * (a + b)
        t = mid[..., None] + half[..., None] * _GL_X
        weight = half[..., None] * _GL_W
        v = centres[..., None, None] + side * scales[..., None, None] * np.sinh(t)
        lw = _log(weight) + np.log(scales[..., None, None]) + np.log(np.cosh(t))
        vs.append(v.reshape(B, -1))
        lws.append(lw.reshape(B, -1))
    return np.concatenate(vs, axis=1), np.concatenate(lws, axis=1)


def _group_loglik(y, s, d, st):
    """log L at d for blocks of equal packet count: y, s of shape (B, n), d of shape (B,)."""
    nu = st["nu"]
    norm = _log_t_norm(nu)
    sig = np.maximum(s * st["kappa"], 1e-3)
    s_ref = np.median(sig, axis=1)
    tau = st["t"] * s_ref
    mu = np.maximum(st["m"] * s_ref, MU_FLOOR_CM)
    s_g = 1.0 / np.sqrt(((nu + 1) / (nu * sig * sig)).sum(axis=1))
    s_u = tau if st["shape"] == "gaussian" else tau * math.sqrt(OFFSET_NU / (OFFSET_NU + 1))
    a_d = 0.5 * np.where(tau > 0, np.minimum(s_u, s_g), s_g)
    # the excess mean of a Student t offset in units of its scale, of the floored excess mean where the floor binds
    r = np.where(st["m"] * s_ref >= MU_FLOOR_CM, st["m"] / st["t"], mu / tau) if st["shape"] == "student_t" and st["t"] > 0 else None
    centres, scales = _centres(y, sig, nu, norm, st["lam"], d, s_g, a_d, s_u * np.ones_like(s_g), tau, st["shape"])
    v, lw = _nodes(centres, scales)
    log_g = _log_packets(y[:, None, :], sig[:, None, :], v[:, :, None], nu, norm, st["lam"])
    log_f = _log_shift(v - d[:, None], tau[:, None], mu[:, None], st, r)
    total = special.logsumexp(log_g + log_f + lw, axis=1)
    if st["t"] <= 0 and st["pi"] < 1:
        spike = math.log1p(-st["pi"]) + _log_packets(y, sig, d[:, None], nu, norm, st["lam"])
        total = np.logaddexp(total, spike)
    return total


def block_loglik(blocks, params):
    """Log likelihood of every block at its true distance. blocks: a list of (y, s, d_true) with packet distances y and
    reported scales s in cm; params: kappa, excess_probability, excess_mean_rel, persistent_sd_rel, offset_shape, nu
    (the degrees of freedom of the packet core, a Student t offset keeping OFFSET_NU) and optionally
    early_detection_per_cm and independent_packets."""
    st = _settings(params)
    if st["independent"]:
        singles, owner = [], []
        for index, (y, s, d) in enumerate(blocks):
            y, s = np.atleast_1d(np.asarray(y, float)), np.atleast_1d(np.asarray(s, float))
            for k in range(len(y)):
                singles.append((y[k:k + 1], s[k:k + 1], d))
                owner.append(index)
        st = dict(st, independent=False)
        values = _loglik(singles, st)
        return np.bincount(np.asarray(owner, int), weights=values, minlength=len(blocks))
    return _loglik(blocks, st)


def _loglik(blocks, st):
    out = np.empty(len(blocks))
    groups = {}
    for index, (y, s, d) in enumerate(blocks):
        groups.setdefault(len(np.atleast_1d(y)), []).append(index)
    for n, members in groups.items():
        y = np.array([np.atleast_1d(np.asarray(blocks[i][0], float)) for i in members]).reshape(len(members), n)
        s = np.array([np.atleast_1d(np.asarray(blocks[i][1], float)) for i in members]).reshape(len(members), n)
        d = np.array([float(blocks[i][2]) for i in members])
        for start in range(0, len(members), _CHUNK):
            part = slice(start, start + _CHUNK)
            out[np.asarray(members[part])] = _group_loglik(y[part], s[part], d[part], st)
    return out


def table(y, s, params, d_grid):
    """Log likelihood of one block at every distance of d_grid."""
    y = np.atleast_1d(np.asarray(y, float))
    s = np.atleast_1d(np.asarray(s, float))
    return block_loglik([(y, s, float(d)) for d in np.asarray(d_grid, float).ravel()], params)
