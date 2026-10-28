//! The link likelihood of SOLID: one exact density of a link's packet block as a function of the predicted
//! distance, tabulated on a lattice and continued analytically beyond it. Shared by the anchor only estimator
//! and the block range factor of the cooperative solver.
//!
//! # Model
//!
//! A link carries packets `y_k` (cm) with reported scales `s_k`. The core scale of packet `k` is
//! `sigma_k = kappa * s_k` (floored at 1e-3 cm) and its core density is Student t with `nu` degrees of freedom,
//! `t_nu(e; sigma_k)`, 3 degrees of freedom in the method. The reference scale `s_ref` is the conventional median
//! of the `sigma_k`, the mean of the two middle values for an even count. All packets of the link share one
//! total shift `w = u + b`:
//!
//! - `u` is the persistent offset, zero mean, Gaussian with standard deviation `tau = persistent_sd_rel * s_ref`
//!   (`offset_shape` "gaussian") or Student t with OFFSET_NU = 3 degrees of freedom and scale `tau`
//!   ("student_t"), whatever the degrees of freedom of the packet core;
//! - `b >= 0` is the persistent excess, zero with probability `1 - pi` and otherwise exponential with mean
//!   `mu = excess_mean_rel * s_ref`, where `pi = excess_probability`.
//!
//! With the packet product `g(v) = prod_k t_nu(y_k - v; sigma_k)` the likelihood of the predicted distance `d` is
//!
//! ```text
//! L(d) = integral f_w(w) g(d + w) dw = integral g(v) f_w(v - d) dv,
//! ```
//!
//! `f_w` the density of `w`. The excess distribution has mass exactly one. The ablation `independent_packets`
//! replaces `L` by the product over packets of single packet likelihoods, each with its own `s_ref = sigma_k`
//! and so its own offset and excess.
//!
//! False early detections (`early_detection_per_cm = lambda > 0`): the receiver of each packet may trigger on
//! noise before the first path, at the rate `lambda` per centimetre of the detection distance `s = d + w`, and then
//! reads the trigger point. Each packet's density becomes
//!
//! ```text
//! p(y | s) = exp(-lambda max(s, 0)) t_nu(y - s; sigma) + lambda exp(-lambda y) 1{0 <= y < s},
//! ```
//!
//! which integrates to one over `y`, and `g(s) = prod_k p(y_k | s)`. Far beyond every reading each packet is an
//! early detection and `L` levels off at the floor `F = prod_k lambda exp(-lambda y_k)`. With `lambda = 0` every
//! value is the one of the model without the term, bit for bit.
//!
//! # Numerics
//!
//! The table holds `log L` in uniform runs whose points are multiples of their step, a power of two fraction of
//! 2 cm. It covers at least `[y_min - 40 mu - 12 tau - 12 sigma_max, y_max + 12 tau + 12 sigma_max]` (Gaussian
//! offsets) or `[.. - 40 mu - r_u tau - r_c sigma_max, .. + r_u tau + r_c sigma_max]` (Student t offsets), with `r`
//! the distance in scales at which a Student t density falls exp(-72) below its peak, as a Gaussian does at 12, kept
//! between 12 and 40 for the offset and 12 and 24 for the packets: 40 and 24 for heavy tails, 12 in the Gaussian
//! limit. With early detections the table reaches down to zero distance, and one that starts there ends there. The
//! runs depend on the packets and the model alone, so changing a range moves no grid point.
//!
//! Every value is the integral over `v` on a finer lattice `v = d + j * delta`, `delta` the step divided by a
//! power of two: the trapezoid rule of `g(v) f_w(v - d)`, which converges exponentially
//! fast for these analytic integrands once `delta` resolves the product (`delta <= 1.2` times its narrowest
//! width). The offset kernel keeps its sampled tails and matches its mass and variance on its three central
//! points, which keeps an offset narrower than the lattice exact to second order. When it is cheaper, Gaussian
//! offsets convolve `g` with the offset kernel on the whole lattice and apply the excess by the recursion of the
//! exponential convolution over log-linear segments corrected by the curvature of log G (third order). Student t
//! offsets take the same recursion after a convolution with the heavy tailed kernel that sums the core of `g`
//! exactly and its tails by a radix-2 fast Fourier transform, whose rounding stays near 1e-16 of the tails'
//! scale. Every sum runs over positive terms in scaled linear arithmetic, with a log domain sum where the sum
//! falls below 1e-280, whose largest terms may have lost bits as subnormal numbers. Early readings make `g` jump at every reading; each jump's first order trapezoid error,
//! `(theta - 1/2) delta D f(xi - x)` for a jump of height `D` at the fraction `theta` of its lattice cell, is removed
//! wherever the lattice resolves the kernel. No packet has to survive the early detections below zero distance, and
//! there the tails of Student t packets and of a Student t offset fall only polynomially, so at a high rate over a
//! long link they outweigh the readings. The lattice then reaches as far below zero as they count.
//!
//! The first step keeps the linear interpolation error of the likelihood's peak below 0.01 nats. Wherever the
//! second differences of the table still indicate a larger error, which happens at the crossover between the
//! offset regime and the packet tail regime far from the packets, a window with half the step replaces the
//! run, refined again as needed.
//!
//! Beyond the table the likelihood continues with its asymptotic form, the log sum of a core term (the packet
//! product concentrated at its mean `c` with mass `Z_g` and variance `V`, smoothed by the shift: `Z_g f_w(c - d)`
//! with `V` added to a Gaussian offset's variance, or averaged over `V` to second order for a Student t offset,
//! whose excess slab is the offset moved by the excess mean plus the exponential tail times the offset's bulk
//! tilt `exp(Var(u) / (2 mu^2))`) and a tail term (the packet product at `x = d` averaged over the shift through
//! its local log slope and curvature). Each term is a separate mode of the integrand only while the
//! other factor leaves it near its expansion point: with `rho` the fraction of the way to the other term's point
//! that the shift moves a term's mode, the tail term takes the weight `m(rho_T)` and the core term
//! `1 - (1 - m(rho_C)) m(rho_T)`, `m` falling from one at 0.3 to zero at 0.5, so a single Gaussian mode, where
//! `rho_C + rho_T = 1`, counts once. Where the packet product stays log concave, as nearly Gaussian packets (large
//! `nu`) keep it, that single mode lies at a distance from `c` and from `x` that grows with `|x - c|`, over which
//! the Gaussian core and the extrapolated tail miss the product's heavier tails. There both terms hand over, as
//! their modes move from two to four core standard deviations, to mode terms: the integrand expanded to second
//! order about its mode, which Newton steps reach, three and more while the last step still expects to gain over
//! 1e-4 nats, the tail's with the offset alone and the excess by its moment generating function. A mode term
//! counts only where the steps reach a mode, the core's only where the packet product and the integrand stay log
//! concave at it, and the tail's, for nearly Gaussian packets, only where the packet product stays log concave at
//! it. Under a Gaussian offset such packets far from the table leave the integrand one mode where the packet product
//! turns from Gaussian to the polynomial tails of its Student t densities, which neither the terms nor the mode
//! terms describe, and there the continuation fades onto the plain core term as that term falls from 1e5 to 2e5
//! nats below its peak, so it keeps decaying where the likelihood carries no weight in any posterior. With early
//! detections `g = g_rest + F H(v - y_max)`: a floor term `F S_w(y_max - x)`, `S_w` the survival function of the
//! shift, integrates the second part exactly, and the core and tail terms describe `g_rest`, so the continuation
//! reaches the floor exactly with a derivative that vanishes smoothly; the floor dominates wherever the mode terms
//! would matter, and the terms keep their own forms. Each
//! end of the table extends until every factor's asymptotic form meets the table within 0.05 nats, and on the
//! far side of a Gaussian offset, while the tail term describes a mode of its own, until it leads the core term
//! by 10 nats, past the crossover where neither term alone describes the likelihood. For a Student t offset each end
//! at a positive distance also extends until the curvature of the log shift density times the core variance is at
//! most 1/8, where the core term's second order average holds even for a product with modes far apart.
//! A correction that decays away from each end makes the continuation meet the table with the same value and
//! slope, so `eval` is continuous and `eval_smooth` is continuously differentiable everywhere, with a derivative
//! that never vanishes outside the table, except as it approaches the floor of early detections. The anchor only
//! estimator evaluates every table at every cell of its grid, most of them beyond the table, and there takes the
//! continuation tabulated over the grid's distances (`FarTable`), which meets it within 1e-10 of the value at the
//! middle of every interval between its nodes and, as its test checks, within 1e-8 of the value in between.

use crate::{AnchorOnlyProfile, BlockRangeProfile, OffsetShape};

/// The coarsest step of the table in centimetres; every step is this value divided by a power of two.
const TABLE_STEP_CM: f64 = 2.0;
/// The finest step of the table and of its integration lattice.
const MIN_STEP_CM: f64 = TABLE_STEP_CM / 4096.0;
/// Floor of the packet core scale in centimetres.
const SIGMA_FLOOR_CM: f64 = 1e-3;
/// Floor of the excess mean in centimetres.
const MU_FLOOR_CM: f64 = 1e-3;
/// Degrees of freedom of a Student t offset, whatever those of the packet core.
pub(crate) const OFFSET_NU: f64 = 3.0;
/// Lattice points per block of the log domain sums, a power of two.
const LOG_DOMAIN_BLOCK: usize = 32;
/// Blocks, or groups, of a lattice beyond which the log domain sums also check groups of that many blocks, or groups.
const LOG_DOMAIN_GROUPS: usize = 2048;
/// Largest linear interpolation error of the table in nats, estimated from its second differences.
const INTERPOLATION_TOLERANCE: f64 = 0.01;
/// Nats below a value of the table from which the tails of the integrand below the lattice leave it out, 6e-6 of it.
const BELOW_TOLERANCE: f64 = 12.0;
/// Most lattice points the tails below a table add to its lattice, 2^15. The core of the packet product is summed
/// exactly against the whole kernel, and below zero distance it spans the tails, so a run then costs their number
/// times the lattice's.
const BELOW_POINTS: f64 = 32_768.0;
/// Smallest scaled sum the table trusts in linear arithmetic. Products of normal numbers below 2.2e-308 lose bits
/// as subnormals, so a smaller sum, whose largest terms may be subnormal, is summed in the log domain instead.
const LINEAR_FLOOR: f64 = 1e-280;
/// Relative widening of every certified bound, 2^-40 of the size of the terms it covers. The rounding it covers, a few
/// operations of 2^-53 each and the last bits of the library's exponential and logarithm, is thousands of times
/// smaller.
pub(crate) const BOUND_MARGIN: f64 = 1.0 / 1_099_511_627_776.0;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Normalization of the Student t density with `nu` degrees of freedom.
pub(crate) fn log_t_norm_for(nu: f64) -> f64 {
    libm::lgamma((nu + 1.0) / 2.0) - libm::lgamma(nu / 2.0) - 0.5 * (nu * std::f64::consts::PI).ln()
}

/// Log density of the Student t error `e` with scale `s`, `norm` from `log_t_norm_for(nu)`.
#[inline]
pub(crate) fn log_t(e: f64, s: f64, nu: f64, norm: f64) -> f64 {
    norm - s.ln() - 0.5 * (nu + 1.0) * (e * e / (nu * s * s)).ln_1p()
}

/// Natural log of the standard normal distribution function, accurate over the whole line.
fn log_ndtr(z: f64) -> f64 {
    if z > 0.0 {
        (-0.5 * libm::erfc(z / SQRT_2)).ln_1p()
    } else if z > -37.0 {
        (0.5 * libm::erfc(-z / SQRT_2)).ln()
    } else {
        let r = 1.0 / (z * z);
        -0.5 * z * z - (-z).ln() - LN_SQRT_2PI + (1.0 - r * (1.0 - r * (3.0 - r * (15.0 - 105.0 * r)))).ln()
    }
}

/// Continued fraction of the incomplete beta function by the modified Lentz method.
fn beta_fraction(x: f64, a: f64, b: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..5000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < tiny { 1.0 / tiny } else { 1.0 / d };
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < tiny { 1.0 / tiny } else { 1.0 / d };
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        let step = d * c;
        h *= step;
        if (step - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Natural log of the survival function of the standard Student t distribution with `nu` degrees of freedom, by the
/// regularized incomplete beta function, accurate far into both tails.
fn log_student_survival(z: f64, nu: f64) -> f64 {
    let (a, b) = (0.5 * nu, 0.5);
    let x = nu / (nu + z * z);
    // log of half the regularized incomplete beta I_x(a, b), the mass beyond |z|
    let log_front = libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b) + a * x.ln() + b * (-x).ln_1p();
    let log_tail = if x < (a + 1.0) / (a + b + 2.0) {
        log_front + beta_fraction(x, a, b).ln() - a.ln()
    } else {
        (-(log_front + beta_fraction(1.0 - x, b, a).ln() - b.ln()).exp()).ln_1p()
    } - std::f64::consts::LN_2;
    if z >= 0.0 {
        log_tail
    } else {
        (-log_tail.exp()).ln_1p()
    }
}

/// log(exp(a) + exp(b)) with -inf handled.
#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// The conventional median: the middle value, or the mean of the two middle values for an even count.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// The reach, in scales, of a Student t with `nu` degrees of freedom: the distance where it decays like a Gaussian at
/// twelve standard deviations, `sqrt(nu expm1(144 / (nu + 1)))`, between twelve scales and `most`. Heavy tails reach
/// `most`, a nearly Gaussian t twelve scales.
fn student_reach(nu: f64, most: f64) -> f64 {
    (nu * (144.0 / (nu + 1.0)).exp_m1()).sqrt().clamp(12.0, most)
}

/// The largest step `TABLE_STEP_CM / 2^k` not above `limit`, never below `floor`.
fn power_of_two_step(limit: f64, top: f64, floor: f64) -> f64 {
    let mut step = top;
    while step > limit && step / 2.0 >= floor {
        step /= 2.0;
    }
    step
}

/// Nodes and weights of the Gauss-Legendre rule of `n` points on [-1, 1].
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut rule = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            derivative = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / derivative;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        rule.push((x, 2.0 / ((1.0 - x * x) * derivative * derivative)));
    }
    rule
}

/// The per link error model of the likelihood, shared by the anchor only estimator and the block range factor
/// of the cooperative solver.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockModel {
    pub kappa: f64,
    pub excess_probability: f64,
    pub excess_mean_rel: f64,
    pub persistent_sd_rel: f64,
    /// Degrees of freedom of the Student t packet error, 3 in the method. A Student t offset keeps OFFSET_NU.
    pub nu: f64,
    /// Distribution of the persistent offset.
    pub offset_shape: OffsetShape,
    /// Rate per centimetre of a false early detection of each packet, zero when off.
    pub early_detection: f64,
    /// Ablation: every packet draws its own persistent offset and excess, the block likelihood becomes the
    /// product of independent packet likelihoods.
    pub independent: bool,
}

impl From<&AnchorOnlyProfile> for BlockModel {
    fn from(profile: &AnchorOnlyProfile) -> Self {
        Self {
            kappa: profile.kappa,
            excess_probability: profile.excess_probability,
            excess_mean_rel: profile.excess_mean_rel,
            persistent_sd_rel: profile.persistent_sd_rel,
            nu: profile.nu,
            offset_shape: profile.offset_shape,
            early_detection: profile.early_detection_per_cm,
            independent: profile.independent_packets,
        }
    }
}

impl From<&BlockRangeProfile> for BlockModel {
    fn from(profile: &BlockRangeProfile) -> Self {
        Self {
            kappa: profile.kappa,
            excess_probability: profile.excess_probability,
            excess_mean_rel: profile.excess_mean_rel,
            persistent_sd_rel: profile.persistent_sd_rel,
            nu: profile.nu,
            offset_shape: profile.offset_shape,
            early_detection: profile.early_detection_per_cm,
            independent: profile.independent_packets,
        }
    }
}

/// Log density of `u + b` at `w` for a Gaussian offset of standard deviation `tau` and an exponential excess of
/// mean `mu`: the exponentially modified Gaussian, the exponential itself when `tau` is zero.
fn log_emg(w: f64, tau: f64, mu: f64) -> f64 {
    if tau > 0.0 {
        -mu.ln() + tau * tau / (2.0 * mu * mu) - w / mu + log_ndtr(w / tau - tau / mu)
    } else if w >= 0.0 {
        -mu.ln() - w / mu
    } else {
        f64::NEG_INFINITY
    }
}

/// `integral_0^1 exp(z s) ds = expm1(z) / z`.
fn exponential_moment(z: f64) -> f64 {
    if z.abs() < 1e-5 {
        1.0 + z / 2.0 + z * z / 6.0
    } else {
        z.exp_m1() / z
    }
}

/// `integral_0^1 s (s - 1) exp(z s) ds = (exp(z) (2 - z) - (2 + z)) / z^3`.
fn bent_moment(z: f64) -> f64 {
    if z.abs() < 0.05 {
        -1.0 / 6.0 - z / 12.0 - z * z / 40.0 - z * z * z / 180.0 - z * z * z * z / 1008.0
    } else {
        (z.exp() * (2.0 - z) - (2.0 + z)) / (z * z * z)
    }
}

/// In place radix-2 fast Fourier transform of `re + i im`, whose length is a power of two, without the `1 / n`
/// of the inverse. The twiddle factors are computed directly, so the rounding error stays near `1e-16 log2 n`.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let twiddles: Vec<(f64, f64)> = (0..n / 2)
        .map(|k| {
            let angle = sign * 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (angle.cos(), angle.sin())
        })
        .collect();
    let mut len = 2;
    while len <= n {
        let (half, stride) = (len / 2, n / len);
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let (wr, wi) = twiddles[k * stride];
                let (a, b) = (start + k, start + k + half);
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// The linear convolution of two real sequences by one complex transform of both packed together and one inverse.
/// Each enters scaled to a largest magnitude of one, since the rounding of the larger one spreads into the transform
/// of the smaller one, and the tails of a packet product at 1e-6 of its peak against kernel weights at 1e-15 left the
/// sums 68 nats below the peak of a table 0.6 nats high.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len() + b.len() - 1;
    let n = len.next_power_of_two();
    let largest = |x: &[f64]| x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let (scale_a, scale_b) = (largest(a), largest(b));
    if scale_a == 0.0 || scale_b == 0.0 {
        return vec![0.0; len];
    }
    let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
    re[..a.len()].iter_mut().zip(a).for_each(|(r, v)| *r = v / scale_a);
    im[..b.len()].iter_mut().zip(b).for_each(|(i, v)| *i = v / scale_b);
    fft(&mut re, &mut im, false);
    let (mut pr, mut pi) = (vec![0.0; n], vec![0.0; n]);
    for k in 0..n {
        // with Z = A + i B: A_k = (Z_k + conj Z_(n-k)) / 2 and B_k = (Z_k - conj Z_(n-k)) / 2i
        let m = (n - k) % n;
        let (zr, zi, cr, ci) = (re[k], im[k], re[m], -im[m]);
        let (ar, ai) = (0.5 * (zr + cr), 0.5 * (zi + ci));
        let (br, bi) = (0.5 * (zi - ci), -0.5 * (zr - cr));
        pr[k] = ar * br - ai * bi;
        pi[k] = ar * bi + ai * br;
    }
    fft(&mut pr, &mut pi, true);
    pr.truncate(len);
    let scale = scale_a * scale_b / n as f64;
    pr.iter_mut().for_each(|v| *v *= scale);
    pr
}

/// Dot product with four accumulators, so the compiler can vectorize it.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0_f64; 4];
    let mut left = a.chunks_exact(4);
    let mut right = b.chunks_exact(4);
    for (x, y) in (&mut left).zip(&mut right) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in left.remainder().iter().zip(right.remainder()) {
        total += x * y;
    }
    total
}

/// One factor of the link likelihood: a set of packets with its own reference scale, offset and excess. The
/// block likelihood has one factor, the independent packet ablation one per packet.
#[derive(Clone, Debug)]
struct Factor {
    y: Vec<f64>,
    sigma: Vec<f64>,
    nu: f64,
    norm: f64,
    /// Normalization of a Student t offset, `log_t_norm_for(OFFSET_NU)`.
    offset_norm: f64,
    tau: f64,
    mu: f64,
    pi: f64,
    shape: OffsetShape,
    /// Rate per centimetre of a false early detection, zero when off.
    lambda: f64,
}

/// One factor's log likelihood on the output grid and the statistics of its packet product used by the
/// continuation: log mass, mean and variance of the core, where the product is within 27.6 nats of its peak.
struct FactorTable {
    values: Vec<f64>,
    log_z: f64,
    center: f64,
    core_var: f64,
}

impl Factor {
    fn new(distances: &[f64], scales: &[f64], model: &BlockModel, norm: f64) -> Self {
        let sigma: Vec<f64> = scales.iter().map(|s| (s * model.kappa).max(SIGMA_FLOOR_CM)).collect();
        let s_ref = median(&sigma);
        let tau = (model.persistent_sd_rel * s_ref).max(0.0);
        Self {
            y: distances.to_vec(),
            sigma,
            nu: model.nu,
            norm,
            offset_norm: log_t_norm_for(OFFSET_NU),
            tau,
            mu: (model.excess_mean_rel * s_ref).max(MU_FLOOR_CM),
            pi: model.excess_probability.clamp(0.0, 1.0),
            // without an offset both shapes are the same model
            shape: if tau > 0.0 { model.offset_shape } else { OffsetShape::Gaussian },
            lambda: model.early_detection.max(0.0),
        }
    }

    fn y_min(&self) -> f64 {
        self.y.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn y_max(&self) -> f64 {
        self.y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    fn sigma_max(&self) -> f64 {
        self.sigma.iter().copied().fold(0.0, f64::max)
    }

    fn log_spike(&self) -> f64 {
        if self.pi < 1.0 {
            (-self.pi).ln_1p()
        } else {
            f64::NEG_INFINITY
        }
    }

    fn log_slab(&self) -> f64 {
        if self.pi > 0.0 {
            self.pi.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Lower bound of the width of the packet product anywhere, the inverse square root of the largest
    /// curvature of its log.
    fn packet_width(&self) -> f64 {
        let curvature: f64 = self.sigma.iter().map(|s| (self.nu + 1.0) / (self.nu * s * s)).sum();
        1.0 / curvature.sqrt()
    }

    /// Width of the offset density at its centre.
    fn offset_width(&self) -> f64 {
        match self.shape {
            OffsetShape::Gaussian => self.tau,
            OffsetShape::StudentT => self.tau * (OFFSET_NU / (OFFSET_NU + 1.0)).sqrt(),
        }
    }

    /// The predicted distances this factor needs tabulated.
    fn range(&self) -> (f64, f64) {
        let (reach, packets) = match self.shape {
            OffsetShape::Gaussian => (12.0 * self.tau, 12.0 * self.sigma_max()),
            OffsetShape::StudentT => (student_reach(OFFSET_NU, 40.0) * self.tau, student_reach(self.nu, 24.0) * self.sigma_max()),
        };
        (self.y_min() - 40.0 * self.mu - reach - packets, self.y_max() + reach + packets)
    }

    /// The first output step: 0.283 widths keeps the linear interpolation error of a Gaussian bump of that width
    /// below 0.01 nats. The offset width counts alone when the offset regime dominates the far right.
    fn initial_step(&self) -> f64 {
        let s_g = self.packet_width();
        let s_u = self.offset_width();
        let s_l = (s_u * s_u + s_g * s_g).sqrt();
        let width = if s_u > 0.25 * s_g { s_u } else { s_l };
        power_of_two_step(0.283 * width, TABLE_STEP_CM, MIN_STEP_CM)
    }

    /// The integration lattice step for the output step `h`: 1.2 widths of the narrowest product of the packets
    /// and the offset, never finer than an eighth of the packet width. A narrow offset leaves the jump of the
    /// excess density at zero on the lattice, which then resolves the excess mean.
    fn lattice_step(&self, h: f64) -> f64 {
        let s_g = self.packet_width();
        let s_u = self.offset_width();
        let s_p = if s_u > 0.0 { 1.0 / (1.0 / (s_g * s_g) + 1.0 / (s_u * s_u)).sqrt() } else { s_g };
        let mut limit = (1.2 * s_p).max(s_g / 8.0);
        if s_u < limit {
            limit = limit.min((0.25 * self.mu).max(s_u));
        }
        power_of_two_step(limit.min(h), h, MIN_STEP_CM)
    }

    /// Log of the packet product at the detection distance `v`. With early detections every packet's density is the
    /// mixture of its core, which survives a false trigger with probability `exp(-lambda max(v, 0))`, and the early
    /// reading `lambda exp(-lambda y)` for `0 <= y < v`.
    fn log_g(&self, v: f64) -> f64 {
        if self.lambda == 0.0 {
            return self.y.iter().zip(&self.sigma).map(|(y, s)| log_t(y - v, *s, self.nu, self.norm)).sum();
        }
        self.log_g_at(v, false)
    }

    /// Log of the packet product at `v` with early detections; `above` switches on the early readings of packets at
    /// `v` itself, the limit from above of a jump.
    fn log_g_at(&self, v: f64, above: bool) -> f64 {
        let survival = -self.lambda * v.max(0.0);
        let log_rate = self.lambda.ln();
        self.y
            .iter()
            .zip(&self.sigma)
            .map(|(y, s)| {
                let own = survival + log_t(y - v, *s, self.nu, self.norm);
                if *y >= 0.0 && (*y < v || (above && *y == v)) {
                    log_add_exp(own, log_rate - self.lambda * y)
                } else {
                    own
                }
            })
            .sum()
    }

    /// The jumps of the packet product where early readings switch on, one per distinct packet reading `y >= 0`:
    /// the position and the log product just below and just above it. None without early detections.
    fn jumps(&self) -> Vec<(f64, f64, f64)> {
        if self.lambda == 0.0 {
            return Vec::new();
        }
        let mut at: Vec<f64> = self.y.iter().copied().filter(|y| *y >= 0.0).collect();
        at.sort_by(f64::total_cmp);
        at.dedup();
        at.into_iter().map(|xi| (xi, self.log_g_at(xi, false), self.log_g_at(xi, true))).collect()
    }

    /// Log of the packet product at `v` with its first three derivatives in `v`, away from its jumps.
    fn log_g_derivatives(&self, v: f64) -> [f64; 4] {
        let nu = self.nu;
        let mut out = [0.0; 4];
        let survival_slope = if v > 0.0 { self.lambda } else { 0.0 };
        for (y, s) in self.y.iter().zip(&self.sigma) {
            let e = y - v;
            let a = nu * s * s;
            let q = a + e * e;
            let t0 = log_t(e, *s, nu, self.norm);
            let t1 = (nu + 1.0) * e / q;
            let t2 = (nu + 1.0) * (e * e - a) / (q * q);
            let t3 = (nu + 1.0) * 2.0 * e * (e * e - 3.0 * a) / (q * q * q);
            if self.lambda == 0.0 {
                out[0] += t0;
                out[1] += t1;
                out[2] += t2;
                out[3] += t3;
                continue;
            }
            // the core c = -lambda max(v, 0) + log t mixed with the constant early reading: with the core's weight
            // w = exp(c) / (exp(c) + early), log p' = w c', log p'' = w (1 - w) c'^2 + w c'' and
            // log p''' = w (1 - w) (1 - 2 w) c'^3 + 3 w (1 - w) c' c'' + w c'''
            let own = -self.lambda * v.max(0.0) + t0;
            let c1 = t1 - survival_slope;
            if *y >= 0.0 && *y < v {
                let total = log_add_exp(own, self.lambda.ln() - self.lambda * y);
                let w = (own - total).exp();
                out[0] += total;
                out[1] += w * c1;
                out[2] += w * (1.0 - w) * c1 * c1 + w * t2;
                out[3] += w * (1.0 - w) * (1.0 - 2.0 * w) * c1 * c1 * c1 + 3.0 * w * (1.0 - w) * c1 * t2 + w * t3;
            } else {
                out[0] += own;
                out[1] += c1;
                out[2] += t2;
                out[3] += t3;
            }
        }
        out
    }

    /// The floor of the packet product where every packet is an early detection, `F = prod lambda exp(-lambda y)`,
    /// as `(log F, y_max)`: the product equals `F` plus a part that vanishes far beyond the last reading. None
    /// without early detections or with a negative reading, whose core alone remains far away.
    fn floor(&self) -> Option<(f64, f64)> {
        if self.lambda == 0.0 || self.y.iter().any(|y| *y < 0.0) {
            return None;
        }
        let log_floor = self.y.iter().map(|y| self.lambda.ln() - self.lambda * y).sum();
        Some((log_floor, self.y_max()))
    }

    /// Beyond the last reading, log of the packet product without its floor, `g - F = F (prod (1 + a_k) - 1)` with
    /// `a_k` the ratio of a packet's surviving core to its early reading, with its first three derivatives.
    fn log_rest_derivatives(&self, v: f64, log_floor: f64) -> [f64; 4] {
        let nu = self.nu;
        // z_k = log core - log early and its derivatives
        let packets: Vec<[f64; 4]> = self
            .y
            .iter()
            .zip(&self.sigma)
            .map(|(y, sd)| {
                let e = y - v;
                let a = nu * sd * sd;
                let q = a + e * e;
                [
                    -self.lambda * v.max(0.0) + log_t(e, *sd, nu, self.norm) - (self.lambda.ln() - self.lambda * y),
                    (nu + 1.0) * e / q - if v > 0.0 { self.lambda } else { 0.0 },
                    (nu + 1.0) * (e * e - a) / (q * q),
                    (nu + 1.0) * 2.0 * e * (e * e - 3.0 * a) / (q * q * q),
                ]
            })
            .collect();
        let top = packets.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        if top < -20.0 {
            // far beyond the readings every a_k is below 2e-9 and log(prod (1 + a_k) - 1) = log sum a_k to 1e-9:
            // the log sum exp of z_k, its derivatives by the weights exp(z_k - top)
            let weights: Vec<f64> = packets.iter().map(|p| (p[0] - top).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mean = |f: &dyn Fn(&[f64; 4]) -> f64| weights.iter().zip(&packets).map(|(w, p)| w * f(p)).sum::<f64>() / total;
            let m1 = mean(&|p| p[1]);
            let m2 = mean(&|p| p[1] * p[1] + p[2]);
            let m3 = mean(&|p| p[1] * p[1] * p[1] + 3.0 * p[1] * p[2] + p[3]);
            return [log_floor + top + total.ln(), m1, m2 - m1 * m1, m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1];
        }
        // S = sum log(1 + a_k) = sum softplus(z_k) and its derivatives
        let mut s = [0.0; 4];
        for &[z, z1, z2, z3] in &packets {
            let w = 1.0 / (1.0 + (-z).exp());
            s[0] += if z > 0.0 { z + (-z).exp().ln_1p() } else { z.exp().ln_1p() };
            s[1] += w * z1;
            s[2] += w * (1.0 - w) * z1 * z1 + w * z2;
            s[3] += w * (1.0 - w) * (1.0 - 2.0 * w) * z1 * z1 * z1 + 3.0 * w * (1.0 - w) * z1 * z2 + w * z3;
        }
        // h = log(exp(S) - 1) with q = 1 / (1 - exp(-S)): h' = S' q, h'' = S'' q - S'^2 q (q - 1),
        // h''' = S''' q - 3 S' S'' q (q - 1) + S'^3 q (q - 1) (2 q - 1)
        let h = if s[0] > 30.0 { s[0] + (-(-s[0]).exp()).ln_1p() } else { s[0].exp_m1().ln() };
        let q = 1.0 / -(-s[0]).exp_m1();
        [
            log_floor + h,
            s[1] * q,
            s[2] * q - s[1] * s[1] * q * (q - 1.0),
            s[3] * q - 3.0 * s[1] * s[2] * q * (q - 1.0) + s[1] * s[1] * s[1] * q * (q - 1.0) * (2.0 * q - 1.0),
        ]
    }

    /// Log density of the shift `u + b` at `z`.
    fn log_shift_density(&self, z: f64) -> f64 {
        let slab = match self.shape {
            OffsetShape::Gaussian => log_emg(z, self.tau, self.mu),
            OffsetShape::StudentT => {
                // the density of u + b far from the offset peak: the offset shifted by the excess mean plus the
                // exponential, as in the core term
                log_add_exp(log_t(z - self.mu, self.tau, OFFSET_NU, self.offset_norm), if z > 0.0 { -self.mu.ln() - z / self.mu } else { f64::NEG_INFINITY })
            }
        };
        let spike = if self.tau > 0.0 { self.log_offset(z) } else { f64::NEG_INFINITY };
        log_add_exp(self.log_spike() + spike, self.log_slab() + slab)
    }

    /// Log survival function of the shift, `P(u + b > z)`, through the identity
    /// `S_(u+b)(z) = S_u(z) + pi mu f_(u+b)(z)` of the exponential excess.
    fn log_shift_survival(&self, z: f64) -> f64 {
        let offset = if self.tau > 0.0 {
            match self.shape {
                OffsetShape::Gaussian => log_ndtr(-z / self.tau),
                OffsetShape::StudentT => log_student_survival(z / self.tau, OFFSET_NU),
            }
        } else if z < 0.0 {
            0.0
        } else {
            f64::NEG_INFINITY
        };
        let slab = match self.shape {
            OffsetShape::Gaussian => log_emg(z, self.tau, self.mu),
            OffsetShape::StudentT => log_add_exp(log_t(z - self.mu, self.tau, OFFSET_NU, self.offset_norm), if z > 0.0 { -self.mu.ln() - z / self.mu } else { f64::NEG_INFINITY }),
        };
        log_add_exp(offset, self.log_slab() + self.mu.ln() + slab)
    }

    /// Log density of the offset `u` at `w`.
    fn log_offset(&self, w: f64) -> f64 {
        match self.shape {
            OffsetShape::Gaussian => -0.5 * (w / self.tau) * (w / self.tau) - self.tau.ln() - LN_SQRT_2PI,
            OffsetShape::StudentT => log_t(w, self.tau, OFFSET_NU, self.offset_norm),
        }
    }

    /// Mass and second moment of the offset density over `[a, b]`, `a < 0 < b`.
    fn offset_moments(&self, a: f64, b: f64, rule: &[(f64, f64)]) -> (f64, f64) {
        match self.shape {
            OffsetShape::Gaussian => {
                let (za, zb) = (a / self.tau, b / self.tau);
                let mass = 1.0 - 0.5 * libm::erfc(zb / SQRT_2) - 0.5 * libm::erfc(-za / SQRT_2);
                let density = |z: f64| (-0.5 * z * z - LN_SQRT_2PI).exp();
                let second = self.tau * self.tau * (mass - zb * density(zb) + za * density(za));
                (mass, second)
            }
            OffsetShape::StudentT => {
                let mut mass = 0.0;
                let mut second = 0.0;
                for (lo, hi) in [(a, 0.0), (0.0, b)] {
                    let (m, s2) = self.sinh_moments(lo, hi, rule);
                    mass += m;
                    second += s2;
                }
                (mass, second)
            }
        }
    }

    /// Mass and second moment of the Student t offset over `[lo, hi]` on one side of zero, by Gauss-Legendre in
    /// the variable `xi` of `w = tau sinh(xi)`, eight panels.
    fn sinh_moments(&self, lo: f64, hi: f64, rule: &[(f64, f64)]) -> (f64, f64) {
        let (xa, xb) = ((lo / self.tau).asinh(), (hi / self.tau).asinh());
        let panels = 8;
        let width = (xb - xa) / panels as f64;
        let mut mass = 0.0;
        let mut second = 0.0;
        for p in 0..panels {
            let mid = xa + (p as f64 + 0.5) * width;
            for (node, weight) in rule {
                let xi = mid + 0.5 * width * node;
                let w = self.tau * xi.sinh();
                let jacobian = self.tau * xi.cosh() * 0.5 * width * weight;
                let density = self.log_offset(w).exp();
                mass += density * jacobian;
                second += w * w * density * jacobian;
            }
        }
        (mass, second)
    }

    /// The offset kernel on the lattice offsets `m_lo..=m_hi` of step `delta`: point samples `delta f_u(m delta)`
    /// for `|m| >= 2` and three central weights that match the kernel's mass and second moment over the lattice
    /// range. A broad offset keeps its samples, a narrow one keeps its variance.
    fn offset_kernel(&self, delta: f64, m_lo: isize, m_hi: isize, rule: &[(f64, f64)]) -> Vec<f64> {
        let mut kernel = vec![0.0; (m_hi - m_lo + 1) as usize];
        let index = |m: isize| (m - m_lo) as usize;
        if self.tau <= 0.0 {
            kernel[index(0)] = 1.0;
            return kernel;
        }
        let mut mass_tails = 0.0;
        let mut second_tails = 0.0;
        for m in m_lo..=m_hi {
            if m.abs() >= 2 {
                let w = m as f64 * delta;
                let p = delta * self.log_offset(w).exp();
                kernel[index(m)] = p;
                mass_tails += p;
                second_tails += p * w * w;
            }
        }
        let (mass, second) = self.offset_moments((m_lo as f64 - 0.5) * delta, (m_hi as f64 + 0.5) * delta, rule);
        let side = ((second - second_tails) / (2.0 * delta * delta)).max(0.0);
        kernel[index(1)] = side;
        kernel[index(-1)] = side;
        kernel[index(0)] = (mass - mass_tails - 2.0 * side).max(0.0);
        kernel
    }

    /// Student t offsets: `delta (t * Exp)(m delta)` on the lattice offsets `m_lo..=m_hi`, the forward recursion of
    /// the exponential convolution with Gauss-Legendre steps in the variable of `w = tau sinh(xi)`, which
    /// resolves the peak of a narrow offset.
    fn student_slab(&self, delta: f64, m_lo: isize, m_hi: isize, rule: &[(f64, f64)]) -> Vec<f64> {
        let (tau, mu) = (self.tau, self.mu);
        let density = |w: f64| self.log_offset(w).exp();
        // integral over s in [a, b] of exp(-(b - s) / mu) / mu t(s)
        let segment = |a: f64, b: f64| -> f64 {
            let mut total = 0.0;
            let pieces: [(f64, f64); 2] = if a < 0.0 && b > 0.0 { [(a, 0.0), (0.0, b)] } else { [(a, b), (0.0, 0.0)] };
            for (lo, hi) in pieces {
                if hi <= lo {
                    continue;
                }
                let (xa, xb) = ((lo / tau).asinh(), (hi / tau).asinh());
                let half = 0.5 * (xb - xa);
                let mid = 0.5 * (xa + xb);
                for (node, weight) in rule {
                    let xi = mid + half * node;
                    let s = tau * xi.sinh();
                    total += (-(b - s) / mu).exp() / mu * density(s) * tau * xi.cosh() * half * weight;
                }
            }
            total
        };
        let w0 = m_lo as f64 * delta;
        // the first value by panels of one excess mean over sixty means: the t density is smooth there
        let mut q = 0.0;
        for p in 0..60 {
            let (lo, hi) = (p as f64 * mu, (p + 1) as f64 * mu);
            for (node, weight) in rule {
                let b = 0.5 * (lo + hi) + 0.5 * (hi - lo) * node;
                q += (-b / mu).exp() / mu * density(w0 - b) * 0.5 * (hi - lo) * weight;
            }
        }
        let decay = (-delta / mu).exp();
        let mut slab = Vec::with_capacity((m_hi - m_lo + 1) as usize);
        slab.push(delta * q);
        for m in m_lo..m_hi {
            let (a, b) = (m as f64 * delta, (m + 1) as f64 * delta);
            q = decay * q + segment(a, b);
            slab.push(delta * q);
        }
        slab
    }

    /// Gaussian offsets: `delta EMG(m delta)` on the lattice offsets `m_lo..=m_hi`, the trapezoid weights of the
    /// exponential with half weight at its jump when there is no offset.
    fn gaussian_slab(&self, delta: f64, m_lo: isize, m_hi: isize) -> Vec<f64> {
        (m_lo..=m_hi)
            .map(|m| {
                let w = m as f64 * delta;
                if self.tau > 0.0 {
                    delta * log_emg(w, self.tau, self.mu).exp()
                } else if m == 0 {
                    0.5 * delta / self.mu
                } else if m > 0 {
                    delta * (-w / self.mu).exp() / self.mu
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The reach of the offset kernel on the lattice.
    fn kernel_reach(&self) -> f64 {
        match self.shape {
            OffsetShape::Gaussian => 12.0 * self.tau,
            OffsetShape::StudentT => student_reach(OFFSET_NU, 40.0) * self.tau,
        }
    }

    /// Log likelihood on the output grid `d0 + i h`, `i < count`, with the statistics of the packet product. The
    /// lattice starts the offset's reach below `d0`, or lower where the tails below count (`lower_end`).
    fn tabulate(&self, d0: f64, h: f64, count: usize, rule: &[(f64, f64)]) -> FactorTable {
        let delta = self.lattice_step(h);
        let lo = (d0 - self.kernel_reach()).min(self.y_min() - 12.0 * self.sigma_max()) - 2.0 * delta;
        let table = self.tabulate_from(lo, d0, h, count, rule);
        match self.lower_end(lo, lo - BELOW_POINTS * delta, d0, h, &table.values) {
            Some(end) => self.tabulate_from(end, d0, h, count, rule),
            None => table,
        }
    }

    /// With early detections and a Student t offset, a lower end of the lattice below `lo`, and no lower than `floor`,
    /// that also holds the tails of the integrand where they count against the `values` of the run from `d0` in steps
    /// of `h`, None where none count. No packet has to survive the early detections below zero distance, and there the
    /// tails of Student t packets and of the offset fall only polynomially, so at a high rate over a long link they
    /// outweigh the readings. At 1e-2 per cm a packet 49 m out put most of its likelihood below zero distance, and a
    /// lattice from 40 offset scales below the table missed it by 0.18 nats, and by 7.5 nats 17 m beyond the table.
    /// The integrand falls there like the power `-k` of the distance `D` to the nearest reading or to `d`,
    /// `k = n (nu + 1) + OFFSET_NU + 1`, so its tail below `v` is about its value at `v` times `D / (k - 1)`. The end
    /// steps down, by a quarter more each time, until no tail comes within BELOW_TOLERANCE of a value. Every other
    /// table keeps its lattice.
    fn lower_end(&self, lo: f64, floor: f64, d0: f64, h: f64, values: &[f64]) -> Option<f64> {
        if self.lambda == 0.0 || self.shape != OffsetShape::StudentT {
            return None;
        }
        let (count, y_min) = (values.len(), self.y_min());
        let power = self.y.len() as f64 * (self.nu + 1.0) + OFFSET_NU + 1.0;
        // the tails change slowly with d, so a long run is checked at every stride-th value and its last
        let stride = (count / 64).max(1);
        let counts = |v: f64| {
            (0..count).step_by(stride).chain(std::iter::once(count - 1)).any(|i| {
                let d = d0 + i as f64 * h;
                let length = (y_min.min(d) - v).max(self.tau) / (power - 1.0);
                self.log_g(v) + self.log_shift_density(v - d) + length.ln() > values[i] - BELOW_TOLERANCE
            })
        };
        // between zero and a lattice that starts above it the survival of the packets grows towards zero
        let probes = if lo > 0.0 { vec![lo, 0.5 * lo, 0.0] } else { vec![lo] };
        if !probes.into_iter().any(counts) {
            return None;
        }
        let mut end = lo.min(0.0);
        let mut step = 0.25 * (y_min - end).max(self.tau);
        while end > floor && counts(end) {
            end -= step;
            step *= 1.25;
        }
        Some(end.max(floor))
    }

    /// Log likelihood on the output grid `d0 + i h`, `i < count`, with the statistics of the packet product, on a
    /// lattice from `lo`.
    fn tabulate_from(&self, lo: f64, d0: f64, h: f64, count: usize, rule: &[(f64, f64)]) -> FactorTable {
        let delta = self.lattice_step(h);
        let r = (h / delta).round() as usize;
        let x0 = d0;
        let x_last = x0 + (count - 1) as f64 * h;
        let reach = self.kernel_reach();
        let hi = (x_last + reach + 12.0 * self.mu).max(self.y_max() + 12.0 * self.sigma_max()) + 2.0 * delta;
        let jl = ((x0 - lo) / delta).ceil() as usize;
        let nf = jl + ((hi - x0) / delta).ceil() as usize + 1;
        let lattice = |j: usize| x0 + (j as f64 - jl as f64) * delta;
        let log_g: Vec<f64> = (0..nf).map(|j| self.log_g(lattice(j))).collect();
        let peak = log_g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let g: Vec<f64> = log_g.iter().map(|v| (v - peak).exp()).collect();
        // statistics of the packet product: mass on the whole lattice, mean and variance on its core; with early
        // detections those of g - F H(v - y_max), the product without the floor of early readings beyond the last
        // packet, which the floor term of the continuation integrates exactly
        let core_peak = peak;
        let core_g_owned;
        let core_g: &[f64] = match self.floor() {
            Some((log_floor, y_max)) => {
                let floor = (log_floor - peak).exp();
                core_g_owned = g
                    .iter()
                    .enumerate()
                    .map(|(j, gj)| if lattice(j) > y_max { (gj - floor).max(0.0) } else { *gj })
                    .collect::<Vec<f64>>();
                &core_g_owned
            }
            None => &g,
        };
        let total: f64 = core_g.iter().sum();
        let (mut core_mass, mut core_first) = (0.0, 0.0);
        for (j, gj) in core_g.iter().enumerate() {
            if *gj >= 1e-12 {
                core_mass += gj;
                core_first += gj * lattice(j);
            }
        }
        let center = core_first / core_mass;
        let core_var = core_g
            .iter()
            .enumerate()
            .filter(|(_, gj)| **gj >= 1e-12)
            .map(|(j, gj)| gj * (lattice(j) - center) * (lattice(j) - center))
            .sum::<f64>()
            / core_mass;
        let log_z = core_peak + (delta * total).ln();
        // with a floor the lattice holds g_rest = g - F H(v - y_max), which vanishes far away, and the step F H is
        // convolved in closed form: its jump at y_max leaves the lattice
        let step = self.floor().map(|(log_floor, y_max)| (y_max, (log_floor - peak).exp()));
        // the jumps of the lattice product in lattice units: position, fraction of its cell and height in the scale
        // of g
        let jumps: Vec<(f64, f64, f64)> = self
            .jumps()
            .into_iter()
            .filter_map(|(xi, below, above)| {
                let t = (xi - x0) / delta + jl as f64;
                let mut height = (above - peak).exp() - (below - peak).exp();
                if let Some((y_max, floor)) = step {
                    if xi == y_max {
                        height -= floor;
                    }
                }
                (t >= 0.0 && t <= (nf - 1) as f64).then(|| (t, t - t.floor(), height))
            })
            .collect();
        let lattice_g: &[f64] = if step.is_some() { core_g } else { &g };
        // the step in lattice units, the index of y_max and F in the scale of g
        let step_index = step.map(|(y_max, floor)| ((y_max - x0) / delta + jl as f64, floor));
        let values = match self.shape {
            OffsetShape::Gaussian => {
                // the offset kernel is non zero within 38.6 standard deviations
                let half = ((38.6 * self.tau / delta).ceil() as isize + 2).max(2);
                if ((2 * half + 12) as usize) < count {
                    let mut conv = self.short_convolution(lattice_g, delta, half, rule, &jumps);
                    self.add_step(&mut conv, delta, step_index);
                    let values = self.recursion_values(&conv, &log_g, peak, delta, jl, r, count);
                    if values.iter().all(|v| v.is_finite()) {
                        values
                    } else {
                        self.direct_values(lattice_g, &log_g, peak, delta, jl, r, count, rule, &jumps, step_index)
                    }
                } else {
                    self.direct_values(lattice_g, &log_g, peak, delta, jl, r, count, rule, &jumps, step_index)
                }
            }
            OffsetShape::StudentT => {
                // The smallest value on the lattice is about the core's mass times the kernel's weight at the
                // lattice's extent. The transform's rounding, about 2e-13 of the largest tail of g (below 1e-6 of
                // its peak once the core is summed exactly) times the largest kernel weight it meets, serves while it
                // stays below 1e-3 of that value. Otherwise the kernel's centre meets the tails in exact sums as
                // well, out to where the transform serves the rest; a kernel that falls too fast for that, as a
                // nearly Gaussian one (large nu) does, is summed exactly out to where it underflows, since the
                // rounding would swamp the far values and every refinement after them.
                let reach = nf - 1;
                let spike = self.offset_kernel(delta, -(reach as isize), reach as isize, rule);
                let smallest = core_mass * spike[0].min(spike[2 * reach]);
                let serves = |weight: f64| 2e-16 * weight < smallest;
                let beyond = |band: usize| spike[reach + band + 1].max(spike[reach - band - 1]);
                let centre = if serves(spike.iter().copied().fold(0.0, f64::max)) {
                    Some(None)
                } else {
                    (1..=reach / 8).find(|&band| serves(beyond(band))).map(Some)
                };
                let mut conv = match centre {
                    Some(centre) => self.fft_convolution(lattice_g, delta, &spike, centre, &jumps),
                    None => {
                        let mut half = 2_isize;
                        while (half as usize) < nf && self.log_offset(half as f64 * delta) - self.log_offset(0.0) > -745.0 {
                            half = (half * 2).min(nf as isize);
                        }
                        self.short_convolution(lattice_g, delta, half, rule, &jumps)
                    }
                };
                self.add_step(&mut conv, delta, step_index);
                let values = self.recursion_values(&conv, &log_g, peak, delta, jl, r, count);
                if values.iter().all(|v| v.is_finite()) {
                    values
                } else {
                    self.direct_values(lattice_g, &log_g, peak, delta, jl, r, count, rule, &jumps, step_index)
                }
            }
        };
        FactorTable { values, log_z, center, core_var }
    }

    /// Adds to the offset convolution on the lattice the floor step `F H(v - y_max)` convolved with the offset in
    /// closed form, `F S_u(y_max - v)`; `step` holds the index of `y_max` in lattice units and `F`.
    fn add_step(&self, conv: &mut [f64], delta: f64, step: Option<(f64, f64)>) {
        let Some((at, floor)) = step else {
            return;
        };
        for (j, value) in conv.iter_mut().enumerate() {
            let z = (at - j as f64) * delta;
            let survival = if self.tau > 0.0 {
                match self.shape {
                    OffsetShape::Gaussian => log_ndtr(-z / self.tau),
                    OffsetShape::StudentT => log_student_survival(z / self.tau, OFFSET_NU),
                }
            } else if z < 0.0 {
                0.0
            } else {
                f64::NEG_INFINITY
            };
            *value += floor * survival.exp();
        }
    }

    /// The first order trapezoid error, at the lattice index `at`, of the jumps of the packet product against the
    /// kernel `density`: a jump of height `D` at the fraction `theta` of its cell adds
    /// `(theta - 1/2) delta D density(xi - x)` to the lattice sum, the lattice holding the value from below at a
    /// jump on a point.
    fn jump_error(jumps: &[(f64, f64, f64)], at: f64, delta: f64, density: impl Fn(f64) -> f64) -> f64 {
        jumps.iter().map(|(t, theta, height)| (theta - 0.5) * delta * height * density((t - at) * delta)).sum()
    }

    /// The offset convolution of `g` on the whole lattice with a kernel non zero on `-half..=half`.
    fn short_convolution(&self, g: &[f64], delta: f64, half: isize, rule: &[(f64, f64)], jumps: &[(f64, f64, f64)]) -> Vec<f64> {
        let nf = g.len();
        let spike = self.offset_kernel(delta, -half, half, rule);
        let mut conv: Vec<f64> = (0..nf)
            .map(|j| {
                let lo = (j as isize - half).max(0) as usize;
                let hi = ((j as isize + half) as usize).min(nf - 1);
                // g[j + m] spike[m + half] for m in lo - j..=hi - j
                let k0 = (lo as isize - j as isize + half) as usize;
                dot(&g[lo..=hi], &spike[k0..k0 + (hi - lo + 1)])
            })
            .collect();
        self.correct_jumps(&mut conv, delta, jumps);
        conv
    }

    /// Removes from the offset convolution on the lattice the first order trapezoid error of the jumps of the
    /// packet product, where the lattice resolves the offset density.
    fn correct_jumps(&self, conv: &mut [f64], delta: f64, jumps: &[(f64, f64, f64)]) {
        if jumps.is_empty() || self.offset_width() < delta {
            return;
        }
        for (j, value) in conv.iter_mut().enumerate() {
            let corrected = *value - Self::jump_error(jumps, j as f64, delta, |w| self.log_offset(w).exp());
            if corrected > 0.0 {
                *value = corrected;
            }
        }
    }

    /// The offset convolution of `g` on the whole lattice with the heavy tailed Student t kernel `spike`, its weights
    /// at the offsets `-(nf - 1)..=nf - 1`: exact sums for the core of `g`, where it is within 1e-6 of its peak, and
    /// the fast Fourier transform for its tails. The transform's rounding is about 2e-13 of the largest tail term
    /// times the largest kernel weight it meets; with `centre`, the kernel's weights within that many offsets of
    /// zero meet the tails in exact sums as well, and the rounding follows the largest weight beyond them.
    fn fft_convolution(&self, g: &[f64], delta: f64, spike: &[f64], centre: Option<usize>, jumps: &[(f64, f64, f64)]) -> Vec<f64> {
        let nf = g.len();
        let reach = nf - 1;
        // conv[j] = sum over m of g[j + m] spike[m + reach], the convolution of g with the reversed kernel:
        // reversed[i] = spike at the offset reach - i
        let reversed: Vec<f64> = spike.iter().rev().copied().collect();
        let tails: Vec<f64> = g.iter().map(|v| if *v >= 1e-6 { 0.0 } else { *v }).collect();
        let full = match centre {
            None => convolve(&tails, &reversed),
            Some(band) => {
                let outer: Vec<f64> = reversed.iter().enumerate().map(|(i, w)| if i.abs_diff(reach) <= band { 0.0 } else { *w }).collect();
                convolve(&tails, &outer)
            }
        };
        let mut conv: Vec<f64> = (0..nf).map(|j| full[j + reach].max(0.0)).collect();
        if let Some(band) = centre {
            // the tails against the kernel's centre: g[j + m] spike[m + reach] for |m| <= band
            for (j, value) in conv.iter_mut().enumerate() {
                let (lo, hi) = (j.saturating_sub(band), (j + band).min(reach));
                *value += dot(&tails[lo..=hi], &spike[lo + reach - j..=hi + reach - j]);
            }
        }
        for (k, gk) in g.iter().enumerate() {
            if *gk >= 1e-6 {
                // g[k] meets the offset k - j at conv[j], reversed[reach - k + j]
                let row = &reversed[reach - k..reach - k + nf];
                for (value, weight) in conv.iter_mut().zip(row) {
                    *value += gk * weight;
                }
            }
        }
        self.correct_jumps(&mut conv, delta, jumps);
        conv
    }

    /// The table by the trapezoid rule of `g(v) f_w(v - d)` on the lattice, one sum per output point.
    #[allow(clippy::too_many_arguments)]
    fn direct_values(
        &self,
        g: &[f64],
        log_g: &[f64],
        peak: f64,
        delta: f64,
        jl: usize,
        r: usize,
        count: usize,
        rule: &[(f64, f64)],
        jumps: &[(f64, f64, f64)],
        step: Option<(f64, f64)>,
    ) -> Vec<f64> {
        let nf = g.len();
        let m_lo = -((jl + r * (count - 1)) as isize);
        let m_hi = (nf - 1 - jl) as isize;
        let spike = self.offset_kernel(delta, m_lo, m_hi, rule);
        let mut slab = match self.shape {
            OffsetShape::Gaussian => self.gaussian_slab(delta, m_lo, m_hi),
            OffsetShape::StudentT => self.student_slab(delta, m_lo, m_hi, rule),
        };
        let slab_mass: f64 = slab.iter().sum();
        if slab_mass > 0.0 {
            slab.iter_mut().for_each(|p| *p /= slab_mass);
        }
        let kernel: Vec<f64> = spike.iter().zip(&slab).map(|(a, b)| (1.0 - self.pi) * a + self.pi * b).collect();
        let first = kernel.iter().position(|k| *k > 0.0).unwrap_or(0);
        let last = kernel.iter().rposition(|k| *k > 0.0).unwrap_or(0);
        let g_first = g.iter().position(|v| *v > 0.0).unwrap_or(0);
        let g_last = g.iter().rposition(|v| *v > 0.0).unwrap_or(0);
        let mut values: Vec<f64> = (0..count)
            .map(|i| {
                // lattice index j pairs with kernel index j + offset
                let offset = (-m_lo) as usize - (jl + r * i);
                let start = first.saturating_sub(offset).max(g_first);
                let end = (last + 1).saturating_sub(offset).min(g_last + 1);
                let mut sum = if end > start { dot(&g[start..end], &kernel[start + offset..end + offset]) } else { 0.0 };
                if !jumps.is_empty() {
                    // the jumps of the product against the offset density where the lattice resolves it and against
                    // the Gaussian slab density; a Student t slab is broad on the lattice
                    let resolved = self.offset_width() >= delta;
                    let corrected = sum
                        - Self::jump_error(jumps, (jl + r * i) as f64, delta, |w| {
                            let spike = if resolved { (1.0 - self.pi) * self.log_offset(w).exp() } else { 0.0 };
                            let slab = match self.shape {
                                OffsetShape::Gaussian => self.pi * log_emg(w, self.tau, self.mu).exp(),
                                OffsetShape::StudentT => 0.0,
                            };
                            spike + slab
                        });
                    if corrected > 0.0 {
                        sum = corrected;
                    }
                }
                if let Some((at, floor)) = step {
                    // the floor step against the whole shift, F S_w(y_max - x)
                    sum += floor * self.log_shift_survival((at - (jl + r * i) as f64) * delta).exp();
                }
                if sum > LINEAR_FLOOR && sum.is_finite() {
                    peak + sum.ln()
                } else {
                    f64::NAN
                }
            })
            .collect();
        self.log_domain_values(&mut values, log_g, delta, jl, r);
        values
    }

    /// Fills every value marked NaN, one whose terms underflowed, with the same sum in the log domain from the point
    /// sampled kernel, the log sum at its lattice index `j` of `log_g[k] + log_kernel_weight((k - j) delta)`, in the
    /// lattice's order. The weights depend on the offset `k - j` alone and are computed once. The lattice falls into
    /// blocks of LOG_DOMAIN_BLOCK points, and a block whose largest `log_g` plus the largest weight over its offsets
    /// lies 50 nats below a term already met is left out, every term in it weighing less than e^-50 of that one. The
    /// first terms met are the one at the point itself and the one where the point before had its largest term, which
    /// moves little from one point to the next. Two nearly Gaussian packets far apart put most of their table in the
    /// log domain, and with every sum over the whole lattice 200 m apart took seconds to tabulate and 900 m minutes.
    ///
    /// A lattice of more than LOG_DOMAIN_GROUPS blocks also falls into groups of LOG_DOMAIN_BLOCK blocks, and those
    /// into groups of LOG_DOMAIN_BLOCK groups, each left out as a block is. Every block in a group left out would be
    /// left out on its own, the terms met only growing, so the sums are the same, and a point takes a few thousand
    /// checks where it took one per block. Tables that reach down to zero distance under packets of 0.3 cm hundreds
    /// of metres away hold millions of such points on lattices of millions of blocks.
    fn log_domain_values(&self, values: &mut [f64], log_g: &[f64], delta: f64, jl: usize, r: usize) {
        if !values.iter().any(|v| v.is_nan()) {
            return;
        }
        let nf = log_g.len();
        // the sizes of the blocks and of the groups of every level, in lattice points
        let mut sizes = vec![LOG_DOMAIN_BLOCK];
        while sizes.len() < 3 && nf > LOG_DOMAIN_GROUPS * sizes[sizes.len() - 1] {
            sizes.push(sizes[sizes.len() - 1] * LOG_DOMAIN_BLOCK);
        }
        // the weight of the offset m at index m + nf - 1, and past the last offset none, so that every block or group
        // of offsets that starts at an offset has its full length
        let mut kernel: Vec<f64> =
            (0..2 * nf - 1).map(|index| self.log_kernel_weight((index as f64 - (nf - 1) as f64) * delta, delta)).collect();
        kernel.resize(2 * nf - 1 + sizes[sizes.len() - 1], f64::NEG_INFINITY);
        // the largest weight of the offsets of every level's length from every index, by doubling the length of the
        // runs, and the largest log_g of every block and group
        let mut heaviest: Vec<Vec<f64>> = Vec::with_capacity(sizes.len());
        let (mut runs, mut run) = (kernel.clone(), 1);
        for (level, size) in sizes.iter().enumerate() {
            while run < *size {
                for index in 0..runs.len() - run {
                    runs[index] = runs[index].max(runs[index + run]);
                }
                run *= 2;
            }
            heaviest.push(if level + 1 < sizes.len() { runs.clone() } else { std::mem::take(&mut runs) });
        }
        let tallest: Vec<Vec<f64>> = sizes
            .iter()
            .map(|size| log_g.chunks(*size).map(|block| block.iter().copied().fold(f64::NEG_INFINITY, f64::max)).collect())
            .collect();
        let mut guess = (0..nf).fold(0, |best, k| if log_g[k] > log_g[best] { k } else { best });
        for (i, value) in values.iter_mut().enumerate().filter(|(_, v)| v.is_nan()) {
            let j = jl + r * i;
            let term = |k: usize| log_g[k] + kernel[k + nf - 1 - j];
            let mut met = term(guess).max(term(j));
            let mut best = f64::NEG_INFINITY;
            let mut acc = 0.0;
            // the groups still to visit, the nearest last, as (level, index)
            let top = sizes.len() - 1;
            let mut pending: Vec<(usize, usize)> = (0..tallest[top].len()).rev().map(|index| (top, index)).collect();
            while let Some((level, index)) = pending.pop() {
                let first = index * sizes[level];
                if tallest[level][index] + heaviest[level][first + nf - 1 - j] < met - 50.0 {
                    continue;
                }
                if level > 0 {
                    let inner = (index * LOG_DOMAIN_BLOCK..((index + 1) * LOG_DOMAIN_BLOCK).min(tallest[level - 1].len())).rev();
                    pending.extend(inner.map(|below| (level - 1, below)));
                    continue;
                }
                for k in first..(first + LOG_DOMAIN_BLOCK).min(nf) {
                    let term = term(k);
                    if term > best {
                        acc = acc * (best - term).exp() + 1.0;
                        best = term;
                        guess = k;
                    } else if term > f64::NEG_INFINITY {
                        acc += (term - best).exp();
                    }
                }
                met = met.max(best);
            }
            *value = best + acc.ln();
        }
    }

    /// Log of the kernel weight at the lattice offset `w` for the log domain fallback: the point sampled closed
    /// form with an offset, and without one the spike on the lattice point and the trapezoid weights of the
    /// exponential excess, half at its jump.
    fn log_kernel_weight(&self, w: f64, delta: f64) -> f64 {
        match self.shape {
            OffsetShape::Gaussian if self.tau > 0.0 => {
                delta.ln()
                    + log_add_exp(self.log_spike() + self.log_offset(w), self.log_slab() + log_emg(w, self.tau, self.mu))
            }
            OffsetShape::StudentT if self.tau > 0.0 => delta.ln() + self.log_shift_density(w),
            _ if w == 0.0 => ((1.0 - self.pi) + self.pi * delta / (2.0 * self.mu)).ln(),
            _ if w > 0.0 => self.log_slab() + (delta / self.mu).ln() - w / self.mu,
            _ => f64::NEG_INFINITY,
        }
    }

    /// The excess part `integral exp(-b / mu) / mu G(v + b) db` on the whole lattice from the offset convolution
    /// `conv`: the backward recursion of the exponential convolution, every segment integrated exactly against
    /// log G interpolated log-linearly and corrected by the curvature of log G, third order in the step.
    fn excess_of(&self, conv: &[f64], delta: f64) -> Vec<f64> {
        let nf = conv.len();
        let mu = self.mu;
        let decay = (-delta / mu).exp();
        let linear_weight = mu / delta * (1.0 - decay) - decay;
        // curvature of log G at every lattice point from its second differences, unknown where G underflows
        let logs: Vec<f64> = conv.iter().map(|v| if *v > 0.0 { v.ln() } else { f64::NAN }).collect();
        let curvature: Vec<f64> = (0..nf)
            .map(|j| {
                if j == 0 || j + 1 == nf {
                    return f64::NAN;
                }
                (logs[j + 1] - 2.0 * logs[j] + logs[j - 1]) / (delta * delta)
            })
            .collect();
        let mut excess = vec![0.0; nf];
        excess[nf - 1] = conv[nf - 1];
        for j in (0..nf - 1).rev() {
            let (a, b) = (conv[j], conv[j + 1]);
            let local = if a > 0.0 && b > 0.0 {
                // log G = chord + curvature b (b - delta) / 2 on the segment, integrated exactly to first order
                let z = (b / a).ln() - delta / mu;
                let (c0, c1) = (curvature[j], curvature[j + 1]);
                let bend = match (c0.is_finite(), c1.is_finite()) {
                    (true, true) => 0.5 * (c0 + c1),
                    (true, false) => c0,
                    (false, true) => c1,
                    (false, false) => 0.0,
                };
                let plain = a * delta / mu * exponential_moment(z);
                let corrected = plain + a * delta / mu * 0.5 * bend * delta * delta * bent_moment(z);
                if corrected > 0.0 {
                    corrected
                } else {
                    plain
                }
            } else {
                a * (1.0 - decay) + (b - a) * linear_weight
            };
            excess[j] = decay * excess[j + 1] + local;
        }
        excess
    }

    /// The table from the offset convolution `conv` on the whole lattice: `(1 - pi) G + pi E` at every output point,
    /// E by the recursion of the exponential convolution. A value whose terms all underflowed is summed in the log
    /// domain with the closed form kernel of `log_kernel_weight`. Where that still leaves a value that is not finite,
    /// the caller sums the table directly.
    #[allow(clippy::too_many_arguments)]
    fn recursion_values(
        &self,
        conv: &[f64],
        log_g: &[f64],
        peak: f64,
        delta: f64,
        jl: usize,
        r: usize,
        count: usize,
    ) -> Vec<f64> {
        let excess = self.excess_of(conv, delta);
        let mut values: Vec<f64> = (0..count)
            .map(|i| {
                let j = jl + r * i;
                let total = (1.0 - self.pi) * conv[j] + self.pi * excess[j];
                if total > LINEAR_FLOOR && total.is_finite() {
                    peak + total.ln()
                } else {
                    f64::NAN
                }
            })
            .collect();
        self.log_domain_values(&mut values, log_g, delta, jl, r);
        values
    }
}

/// The asymptotic form of one factor beyond the table and the corrections that join it to the table ends.
#[derive(Clone, Debug)]
struct Tail {
    factor: Factor,
    log_z: f64,
    center: f64,
    core_var: f64,
    left: Junction,
    right: Junction,
}

/// A correction `(offset + (slope + offset / length) t) exp(-t / length)` of the asymptotic form at the distance
/// `t` beyond a table end: it equals `offset` with derivative `slope` in `t` at the end and decays away from it.
/// With `above_floor` it corrects the log of the part above the floor of early detections instead of the whole.
#[derive(Clone, Copy, Debug, Default)]
struct Junction {
    end: f64,
    offset: f64,
    slope: f64,
    length: f64,
    above_floor: bool,
}

impl Junction {
    /// Value and derivative in `t` at the distance `t >= 0` beyond the end.
    fn at(&self, t: f64) -> (f64, f64) {
        let a = self.slope + self.offset / self.length;
        let decay = (-t / self.length).exp();
        ((self.offset + a * t) * decay, (a - (self.offset + a * t) / self.length) * decay)
    }
}

/// `-0.5 ln(1 - z)` for `z <= 1/2`, continued linearly beyond with a continuous derivative, and its derivative.
fn curvature_correction(z: f64) -> (f64, f64) {
    if z <= 0.5 {
        (-0.5 * (-z).ln_1p(), 0.5 / (1.0 - z))
    } else {
        (0.5 * std::f64::consts::LN_2 + (z - 0.5), 1.0)
    }
}

/// Log of the average of `exp(l1 w + l2 w^2 / 2)` over a centred Gaussian `w` of variance `v`,
/// `l1^2 v / (2 (1 - l2 v)) - log(1 - l2 v) / 2`, and its derivative when `l1`, `l2` have the derivatives `l2`, `l3`.
/// It is the Gaussian integral of a log-quadratic function; where `l2 v` exceeds one half, `1 / (1 - z)` and the
/// logarithm continue linearly with continuous derivatives.
fn gaussian_average(l1: f64, l2: f64, l3: f64, v: f64) -> (f64, f64) {
    let bend = l2 * v;
    let (h, dh) = bend_factor(bend);
    let (q, dq) = curvature_correction(bend);
    (0.5 * l1 * l1 * v * h + q, l1 * l2 * v * h + 0.5 * l1 * l1 * v * dh * l3 * v + dq * l3 * v)
}

/// `1 / (1 - z)` for `z <= 1/2`, continued linearly with a continuous derivative, and its derivative.
fn bend_factor(z: f64) -> (f64, f64) {
    if z <= 0.5 {
        (1.0 / (1.0 - z), 1.0 / ((1.0 - z) * (1.0 - z)))
    } else {
        (2.0 + 4.0 * (z - 0.5), 4.0)
    }
}

/// Zero up to `a`, one from `b`, a cubic step with a continuous derivative between; the value and its derivative.
fn rising(t: f64, a: f64, b: f64) -> (f64, f64) {
    if t <= a {
        (0.0, 0.0)
    } else if t >= b {
        (1.0, 0.0)
    } else {
        let s = (t - a) / (b - a);
        (s * s * (3.0 - 2.0 * s), 6.0 * s * (1.0 - s) / (b - a))
    }
}

/// One up to `a`, zero from `b`: `1 - rising`, with its derivative.
fn falling(t: f64, a: f64, b: f64) -> (f64, f64) {
    let (r, dr) = rising(t, a, b);
    (1.0 - r, -dr)
}

/// `falling(t, a, b)` of a quantity `t` with the derivative `t_slope`, and its derivative, zero wherever `falling`
/// is flat.
fn falling_along(t: f64, t_slope: f64, a: f64, b: f64) -> (f64, f64) {
    let (value, slope) = falling(t, a, b);
    (value, if slope == 0.0 { 0.0 } else { slope * t_slope })
}

/// A weight and its derivative times a factor and its derivative.
fn times((weight, weight_slope): (f64, f64), (factor, factor_slope): (f64, f64)) -> (f64, f64) {
    (weight * factor, weight_slope * factor + weight * factor_slope)
}

/// Final Newton decrement `p1^2 / (-p2)` in nats up to which a mode term counts in full. From ten times as much it
/// counts for nothing.
const TRUSTED_DECREMENT: f64 = 1e-4;
/// Most Newton steps of a mode term.
const MODE_STEPS: usize = 20;

/// A mode term and the state of its integrand at the mode the Newton steps reached.
struct ModeTerm {
    /// The term and its derivative in `x`.
    value: f64,
    slope: f64,
    /// Log derivatives of the packet product at the mode, and the derivative of the mode in `x`.
    g: [f64; 4],
    e_slope: f64,
    /// Curvature of the log integrand at the mode, with its derivative in `x`.
    curvature: (f64, f64),
    /// One while the steps have reached the mode, zero where they have not, with its derivative in `x`.
    trust: (f64, f64),
}

/// Weight of an asymptotic term whose mode has moved the fraction `rho` of the way towards the other term's
/// expansion point: one up to 0.3, zero from 0.5, and its derivative in `rho`.
fn mode_weight(rho: f64) -> (f64, f64) {
    falling(rho, 0.3, 0.5)
}

/// Relative difference up to which the derivative of a mode carried along its Newton steps stands for the derivative
/// of the mode itself.
const MODE_SLOPE_TOLERANCE: f64 = 1e-6;
/// Nats below the peak of the plain core term from which the continuation of nearly Gaussian packets fades onto it.
/// Before the fade the continuation of the scanned blocks first rose 1.04e6 nats below their peaks, and no posterior
/// sees anything that far down.
const FAR_DEPTH: f64 = 1e5;

/// The moment generating function of the exponential excess `1 / (1 - z)` for `z <= 0`, continued by its second
/// order Taylor polynomial for `z > 0`, and its derivative.
fn excess_factor(z: f64) -> (f64, f64) {
    if z <= 0.0 {
        let r = 1.0 / (1.0 - z);
        (r, r * r)
    } else {
        (1.0 + z + z * z, 1.0 + 2.0 * z)
    }
}

/// The second derivative of `excess_factor`.
fn excess_curvature(z: f64) -> f64 {
    if z <= 0.0 {
        let r = 1.0 / (1.0 - z);
        2.0 * r * r * r
    } else {
        2.0
    }
}

impl Tail {
    /// The variance of the offset used by the tail term.
    fn offset_variance(&self) -> f64 {
        let f = &self.factor;
        match f.shape {
            OffsetShape::Gaussian => f.tau * f.tau,
            OffsetShape::StudentT => f.tau * f.tau * OFFSET_NU / (OFFSET_NU - 2.0),
        }
    }

    /// Log density of the shift at `w` when the packet product is replaced by its core, with its derivative in
    /// `w`: the shift density with the core variance added for Gaussian offsets, and for Student t offsets the
    /// shift density averaged over the core to second order. Also the displacement of the core's mode towards `x`
    /// with its derivative in `w`, which tells whether the core term and the tail term describe different modes.
    fn core_density(&self, w: f64) -> (f64, f64, f64, f64) {
        let f = &self.factor;
        let (log_spike, log_slab) = (f.log_spike(), f.log_slab());
        match f.shape {
            OffsetShape::Gaussian => {
                let te = (f.tau * f.tau + self.core_var).sqrt().max(1e-9);
                let log_phi = -0.5 * (w / te) * (w / te) - te.ln() - LN_SQRT_2PI;
                let ls = log_spike + log_phi;
                let le = log_slab + log_emg(w, te, f.mu);
                let lf = log_add_exp(ls, le);
                let mut slope = 0.0;
                let mut curvature = 0.0;
                let te2 = te * te;
                if ls > f64::NEG_INFINITY {
                    let share = (ls - lf).exp();
                    slope += share * (-w / te2);
                    curvature += share * (w * w / (te2 * te2) - 1.0 / te2);
                }
                if log_slab > f64::NEG_INFINITY {
                    // pi EMG' = pi (phi - EMG) / mu and pi EMG'' = pi (phi' - EMG') / mu
                    let (phi, emg) = ((log_slab + log_phi - lf).exp(), (le - lf).exp());
                    slope += (phi - emg) / f.mu;
                    curvature += (phi * (-w / te2) - (phi - emg) / f.mu) / f.mu;
                }
                // the core, a Gaussian of variance V, moves by its tilted mean V (log f)' under the shift density
                let v = self.core_var;
                (lf, slope, v * slope, v * (curvature - slope * slope))
            }
            OffsetShape::StudentT => {
                let [lf, l1, l2, l3] = self.shift_derivatives(w);
                // the core of the packet product, of variance V, averaged over the shift density expanded to second
                // order: exact for a Gaussian offset and vanishing in polynomial tails
                let v = self.core_var;
                let (average, average_slope) = gaussian_average(l1, l2, l3, v);
                // the mode of the core under the shift density moves by V l1 / (1 - l2 V)
                let (h, dh) = bend_factor(l2 * v);
                (lf + average, l1 + average_slope, v * l1 * h, v * (l2 * h + l1 * dh * l3 * v))
            }
        }
    }

    /// Whether the core term of a Student t offset holds at `d`, where the curvature of the log shift density at the
    /// core's distance times the core variance stays within CORE_BEND, so the density is nearly log quadratic over the
    /// core. A packet product with modes metres apart, or one that early detections tilt towards zero distance, has a
    /// core as wide as its distance from the table's end, and there the average continued past a curvature times
    /// variance of one half once rose to a false peak 9 m beyond the table, 5.5 nats above its end. Gaussian offsets
    /// convolve the core exactly, and a distance of zero or less is never solved for.
    fn core_holds(&self, d: f64) -> bool {
        self.factor.shape == OffsetShape::Gaussian || d <= 0.0 || self.shift_derivatives(self.center - d)[2] * self.core_var <= CORE_BEND
    }

    /// Log density of the Student t offset at `z` and its first three derivatives.
    fn student_derivatives(&self, z: f64) -> [f64; 4] {
        let f = &self.factor;
        let a = OFFSET_NU * f.tau * f.tau;
        let q = a + z * z;
        [
            log_t(z, f.tau, OFFSET_NU, f.offset_norm),
            -(OFFSET_NU + 1.0) * z / q,
            (OFFSET_NU + 1.0) * (z * z - a) / (q * q),
            (OFFSET_NU + 1.0) * 2.0 * z * (3.0 * a - z * z) / (q * q * q),
        ]
    }

    /// Log density of the offset `u` alone at `z` and its first three derivatives.
    fn offset_derivatives(&self, z: f64) -> [f64; 4] {
        let f = &self.factor;
        match f.shape {
            OffsetShape::Gaussian => {
                let tau = f.tau.max(1e-9);
                [-0.5 * (z / tau) * (z / tau) - tau.ln() - LN_SQRT_2PI, -z / (tau * tau), -1.0 / (tau * tau), 0.0]
            }
            OffsetShape::StudentT => self.student_derivatives(z),
        }
    }

    /// Log density of the shift `u + b` at `z` and its first three derivatives. For a Gaussian offset the exact
    /// mixture of the offset and the exponentially modified Gaussian. For a Student t offset a mixture of the
    /// offset, the offset moved by the excess mean and the exponential: the slab is the offset near z with a small
    /// excess, `t(z - mu)`, plus the excess near z with the offset in its bulk, `exp(-z / mu) E[exp(u / mu)] / mu`,
    /// where the bulk tilt `exp(Var(u) / (2 mu^2))` is exact for a Gaussian offset and within 0.04 nats of the
    /// tilt over the bulk of a Student t with nu >= 5.
    fn shift_derivatives(&self, z: f64) -> [f64; 4] {
        let f = &self.factor;
        let (log_spike, log_slab) = (f.log_spike(), f.log_slab());
        match f.shape {
            OffsetShape::Gaussian => {
                let tau = f.tau.max(1e-9);
                let t2 = tau * tau;
                let log_phi = -0.5 * (z / tau) * (z / tau) - tau.ln() - LN_SQRT_2PI;
                let ls = log_spike + log_phi;
                let le = log_slab + log_emg(z, tau, f.mu);
                let lf = log_add_exp(ls, le);
                // shares in f of the spike, of the slab's Gaussian and of the slab, and the derivatives of the
                // Gaussian relative to itself
                let spike = if ls > f64::NEG_INFINITY { (ls - lf).exp() } else { 0.0 };
                let (phi, emg) =
                    if log_slab > f64::NEG_INFINITY { ((log_slab + log_phi - lf).exp(), (le - lf).exp()) } else { (0.0, 0.0) };
                let p1 = -z / t2;
                let p2 = z * z / (t2 * t2) - 1.0 / t2;
                let p3 = -z * z * z / (t2 * t2 * t2) + 3.0 * z / (t2 * t2);
                // pi EMG^(k) = pi (phi^(k-1) - EMG^(k-1)) / mu, relative to f
                let e1 = (phi - emg) / f.mu;
                let e2 = (phi * p1 - e1) / f.mu;
                let e3 = (phi * p2 - e2) / f.mu;
                let (f1, f2, f3) = (spike * p1 + e1, spike * p2 + e2, spike * p3 + e3);
                [lf, f1, f2 - f1 * f1, f3 - 3.0 * f1 * f2 + 2.0 * f1 * f1 * f1]
            }
            OffsetShape::StudentT => {
                let mut parts = vec![(log_spike, self.student_derivatives(z)), (log_slab, self.student_derivatives(z - f.mu))];
                if z > 0.0 {
                    let tilt = 0.5 * self.offset_variance() / (f.mu * f.mu);
                    parts.push((log_slab, [-f.mu.ln() - z / f.mu + tilt, -1.0 / f.mu, 0.0, 0.0]));
                }
                let lf = parts.iter().fold(f64::NEG_INFINITY, |acc, (weight, c)| log_add_exp(acc, weight + c[0]));
                let (mut m1, mut m2, mut m3) = (0.0, 0.0, 0.0);
                for (weight, c) in &parts {
                    let share = (weight + c[0] - lf).exp();
                    if share > 0.0 {
                        m1 += share * c[1];
                        m2 += share * (c[2] + c[1] * c[1]);
                        m3 += share * (c[3] + 3.0 * c[1] * c[2] + c[1] * c[1] * c[1]);
                    }
                }
                [lf, m1, m2 - m1 * m1, m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1]
            }
        }
    }

    /// The mode term at `x`: the log of the integral of `g(v) h(v - x)` by the integrand's second order expansion
    /// about its mode, which Newton steps reach from `e`, with `h` the density whose log derivatives `density`
    /// returns. `e` is a function of `x` with the derivative `e_slope`, and `g` and `s` hold the log derivatives of
    /// the packet product and of `h` at `e`. None where the log integrand is not concave along the way.
    ///
    /// The steps are three, and more while the final Newton decrement `p1^2 / (-p2)`, the nats the expansion still
    /// expects to gain towards its maximum, exceeds TRUSTED_DECREMENT. The term's trust falls from one to zero as
    /// the decrement grows from there to ten times as much. Where the shift is about to wipe out a mode of the
    /// integrand, as the pull of a Gaussian offset does to the core of Student t packets, whose log slope is
    /// bounded, the steps land between the modes on an almost flat curvature, and the expansion would add
    /// `p1^2 / (-2 p2)` nats of a mode that is not there.
    fn mode_term(
        &self,
        x: f64,
        (mut e, mut e_slope): (f64, f64),
        (mut g, mut s): ([f64; 4], [f64; 4]),
        density: &dyn Fn(f64) -> [f64; 4],
    ) -> Option<ModeTerm> {
        let f = &self.factor;
        // first and second derivatives of the log integrand in v at e, with their derivatives in x along e(x)
        let slopes = |g: &[f64; 4], s: &[f64; 4], e_slope: f64| {
            (
                g[1] + s[1],
                g[2] + s[2],
                g[2] * e_slope + s[2] * (e_slope - 1.0),
                g[3] * e_slope + s[3] * (e_slope - 1.0),
            )
        };
        // three steps, and more while the decrement stays above the trusted one
        let mut steps = 0;
        let (p1, p2) = loop {
            let (p1, p2, p1_slope, p2_slope) = slopes(&g, &s, e_slope);
            if p2.is_nan() || p2 >= 0.0 {
                return None;
            }
            if steps >= 3 && (p1 * p1 / (-p2) < TRUSTED_DECREMENT || steps == MODE_STEPS) {
                break (p1, p2);
            }
            e -= p1 / p2;
            e_slope -= (p1_slope * p2 - p1 * p2_slope) / (p2 * p2);
            g = f.log_g_derivatives(e);
            s = density(e - x);
            steps += 1;
        };
        // The mode moves with x as p1(e, x) = 0 makes it, by s2 / p2. The derivative carried along the steps holds that
        // of the last iterate, which a step taken far from the mode leaves wrong even once the steps have converged,
        // -1.3 where the mode moves by 0.30 in a nearly Gaussian core 67 m past its table, and there it gives way to
        // the mode's own. It once gave the continuation slopes of the wrong sign hundreds of metres away.
        let mode_slope = s[2] / p2;
        if !((e_slope - mode_slope).abs() <= MODE_SLOPE_TOLERANCE * (1.0 + mode_slope.abs())) {
            e_slope = mode_slope;
        }
        let (_, _, p1_slope, p2_slope) = slopes(&g, &s, e_slope);
        let value = g[0] + s[0] + 0.5 * p1 * p1 / (-p2) + 0.5 * (std::f64::consts::TAU / (-p2)).ln();
        let slope = g[1] * e_slope + s[1] * (e_slope - 1.0) + p1 * p1_slope / (-p2) + 0.5 * p1 * p1 * p2_slope / (p2 * p2)
            - 0.5 * p2_slope / p2;
        let decrement_slope = 2.0 * p1 * p1_slope / (-p2) + p1 * p1 * p2_slope / (p2 * p2);
        let trust = falling_along(p1 * p1 / (-p2), decrement_slope, TRUSTED_DECREMENT, 10.0 * TRUSTED_DECREMENT);
        (value.is_finite() && slope.is_finite() && trust.1.is_finite()).then_some(ModeTerm {
            value,
            slope,
            g,
            e_slope,
            curvature: (p2, p2_slope),
            trust,
        })
    }

    /// The mode term that takes over from the core term at `x` and its weight, each with its derivative in `x`.
    /// The core term replaces the packet product by a Gaussian. A product that stays log concave where the shift
    /// density moves the core's mode, as nearly Gaussian packets do, keeps the integrand `g(v) f(v - x)` one mode
    /// there, at a distance from `c` that grows with `|x - c|`, and the Gaussian misses the product's heavier tails
    /// over that distance; the mode term integrates the product itself. The weight rises from zero to one as the
    /// core's mode, `c + shift`, moves from two to four core standard deviations, and falls to zero where the
    /// product or the integrand is not log concave at it, so near the core and for heavy tailed packets the core
    /// term stands alone. Both are log concave at the first estimate of the mode and at the mode the steps reach, since
    /// Student t packets pulled by a far offset leave two modes, one at the core and one near `x`, and the steps can
    /// jump to the second, the tail term's, where the product's tails are log convex.
    fn core_hand_over(&self, x: f64, shift: f64, shift_slope: f64) -> Option<[f64; 4]> {
        let f = &self.factor;
        let v = self.core_var;
        if !(v > 0.0 && v.is_finite()) {
            return None;
        }
        let sd = v.sqrt();
        let (rise, rise_slope) = rising(shift.abs() / sd, 2.0, 4.0);
        if rise == 0.0 {
            return None;
        }
        let e = self.center + shift;
        let e_slope = -shift_slope;
        let (g, s) = (f.log_g_derivatives(e), self.shift_derivatives(e - x));
        let (concave_g, concave_g_slope) = falling(g[2] * v, -0.5, -0.25);
        let (concave, concave_slope) = falling((g[2] + s[2]) * v, -0.5, -0.25);
        let weight = rise * concave_g * concave;
        if weight.is_nan() || weight <= 0.0 {
            return None;
        }
        let weight_slope = rise_slope * shift.signum() * e_slope / sd * concave_g * concave
            + rise * concave_g_slope * g[3] * e_slope * v * concave
            + rise * concave_g * concave_slope * (g[3] * e_slope + s[3] * (e_slope - 1.0)) * v;
        let mode = self.mode_term(x, (e, e_slope), (g, s), &|z| self.shift_derivatives(z))?;
        let at_mode = [
            mode.trust,
            falling_along(mode.g[2] * v, mode.g[3] * mode.e_slope * v, -0.5, -0.25),
            falling_along(mode.curvature.0 * v, mode.curvature.1 * v, -0.5, -0.25),
        ];
        let (weight, weight_slope) = at_mode.into_iter().fold((weight, weight_slope), times);
        (weight > 0.0 && weight_slope.is_finite()).then_some([mode.value, mode.slope, weight, weight_slope])
    }

    /// The mode term that takes over from the tail term at `x` and its weight, each with its derivative in `x`;
    /// `l` holds the first three log derivatives of the packet product at `x`. The tail term extrapolates the
    /// product's second order expansion at `x` to where the offset moves the mode. Where the product is log
    /// concave at `x` that distance grows with `|x - c|` and the extrapolation misses the curvature of the product
    /// over it; the mode term integrates the product against the offset alone and takes the excess, as the tail
    /// term does, by its moment generating function at the product's log slope at the mode, which is the slope of
    /// the offset term in `x`. The weight rises from zero to one as the offset moves the mode from two to four
    /// core standard deviations from `x`, and falls to zero as the curvature of the log integrand at the mode's
    /// first estimate falls from one half to one quarter of the offset's, before the mode term loses its mode. A
    /// Student t offset meets its second order expansion only near the Gaussian limit, the kurtosis biasing the mode
    /// term by about 0.75 / (nu + 1) nats, so for it the weight also rises from 30 to 100 degrees of freedom.
    fn tail_hand_over(&self, x: f64, l: [f64; 3], offset_var: f64) -> Option<[f64; 4]> {
        let f = &self.factor;
        let v = self.core_var;
        let gaussian_like = match f.shape {
            OffsetShape::Gaussian => 1.0,
            OffsetShape::StudentT => rising(OFFSET_NU, 30.0, 100.0).0,
        };
        if !(v > 0.0 && v.is_finite()) || gaussian_like == 0.0 {
            return None;
        }
        let sd = v.sqrt();
        let (h, dh) = bend_factor(l[1] * offset_var);
        let shift = offset_var * l[0] * h;
        let shift_slope = offset_var * (l[1] * h + l[0] * dh * l[2] * offset_var);
        let (rise, rise_slope) = rising(shift.abs() / sd, 2.0, 4.0);
        if rise == 0.0 {
            return None;
        }
        // the log integrand's curvature at the mode's first estimate, relative to the offset's
        let e = x + shift;
        let e_slope = 1.0 + shift_slope;
        let (g, s) = (f.log_g_derivatives(e), self.offset_derivatives(e - x));
        let (concave, concave_slope) = rising(-(g[2] + s[2]) * offset_var, 0.25, 0.5);
        let weight = gaussian_like * rise * concave;
        if weight.is_nan() || weight <= 0.0 {
            return None;
        }
        let weight_slope = gaussian_like
            * (rise_slope * shift.signum() * shift_slope / sd * concave
                - rise * concave_slope * (g[3] * e_slope + s[3] * (e_slope - 1.0)) * offset_var);
        let mode = self.mode_term(x, (e, e_slope), (g, s), &|z| self.offset_derivatives(z))?;
        let (mut weight, mut weight_slope) = times((weight, weight_slope), mode.trust);
        // Nearly Gaussian packets far from the table leave the integrand one mode, between the core and x. Where that
        // mode lies where the product turns log convex, a thousand core scales from the packets, both terms' plain
        // forms miss it by thousands of nats, and the tail term's steps reach it in some places and fail in others.
        // There the tail term's mode term fades as the core term's does, so the continuation falls onto the plain
        // terms and stays monotone. It once rose 11,460 nats at one distance 67 m past a table and 77,000 nats at
        // another 139 m past one.
        let gaussian = rising(f.nu, 30.0, 100.0).0;
        if gaussian > 0.0 {
            let (product, product_slope) = falling_along(mode.g[2] * v, mode.g[3] * mode.e_slope * v, -0.5, -0.25);
            (weight, weight_slope) = times((weight, weight_slope), (1.0 - gaussian + gaussian * product, gaussian * product_slope));
        }
        let (g1, g1_slope) = (mode.g[1], mode.g[2] * mode.e_slope);
        let (r, dr) = excess_factor(g1 * f.mu);
        let mix = 1.0 - f.pi + f.pi * r;
        let (value, slope) = (mode.value + mix.ln(), mode.slope + f.pi * dr * f.mu * g1_slope / mix);
        (weight > 0.0 && value.is_finite() && slope.is_finite() && weight_slope.is_finite()).then_some([value, slope, weight, weight_slope])
    }

    /// The tail term near zero detection distance with early detections, below every reading. The survival of every
    /// packet, `exp(-lambda max(v, 0))`, bends log g at `v = 0`, where its slope drops by `n lambda` for `n`
    /// packets, so the expansion of log g at `x` holds on the side of `x` alone. Each side takes its own expansion,
    /// averaged over the shift as the tail term does, and weighted by the share of the shift, tilted by that
    /// expansion, which reaches the side. For the spike that share is the one of the Gaussian of mean
    /// `V l1 / (1 - l2 V)` and variance `V / (1 - l2 V)`, exact for a Gaussian offset and a quadratic log g on each
    /// side. For the slab it is the one of the same Gaussian moved by the tilted excess mean, with that mean added to
    /// its standard deviation in quadrature, so an excess far wider than the offset spreads the bend as the excess
    /// does. The term is continuous with a continuous derivative across the bend. `l` holds log g and its first three
    /// derivatives at `x`. None where the far side weighs less than exp(-40) of the near one, where the term is the
    /// near side's.
    fn bent_tail(&self, x: f64, [lg, l1, l2, l3]: [f64; 4], offset_var: f64) -> Option<(f64, f64)> {
        let f = &self.factor;
        if f.lambda == 0.0 || x >= f.y_min() {
            return None;
        }
        let bend = f.y.len() as f64 * f.lambda;
        // log g and its slope below the bend and above it, where the survival takes n lambda x off
        let (below, above) = if x > 0.0 { ((lg + bend * x, l1 + bend), (lg, l1)) } else { ((lg, l1), (lg - bend * x, l1 - bend)) };
        let (h, dh) = bend_factor(l2 * offset_var);
        let sd = (offset_var * h).sqrt();
        let sd_slope = 0.5 * sd * dh * l3 * offset_var / h;
        // the log weight of the spike and of the slab on one side with its derivative in x, the side's expansion averaged
        // over the shift plus the log share of the tilted shift beyond the bend on that side, `sign` -1 below and 1 above
        let parts = |(g0, g1): (f64, f64), sign: f64| {
            let (average, average_slope) = gaussian_average(g1, l2, l3, offset_var);
            let z = g1 * f.mu;
            let (r, dr) = excess_factor(z);
            let mean = offset_var * g1 * h;
            let mean_slope = offset_var * (l2 * h + g1 * dh * l3 * offset_var);
            // the tilted excess mean, d log r / d l1, and its derivative
            let excess = f.mu * dr / r;
            let excess_slope = f.mu * f.mu * l2 * (excess_curvature(z) * r - dr * dr) / (r * r);
            let share = |centre: f64, centre_slope: f64, spread: f64, spread_slope: f64| {
                if spread == 0.0 {
                    // without an offset the spike is a point, all on the side of the bend where it falls and half on
                    // the bend itself, where both sides' expansions meet
                    let side = sign * (x + centre);
                    let log_share = if side > 0.0 { 0.0 } else if side < 0.0 { f64::NEG_INFINITY } else { -std::f64::consts::LN_2 };
                    return (log_share, 0.0);
                }
                let u = sign * (x + centre) / spread;
                let u_slope = sign * (1.0 + centre_slope) / spread - u * spread_slope / spread;
                let log_share = log_ndtr(u);
                (log_share, (-0.5 * u * u - LN_SQRT_2PI - log_share).exp() * u_slope)
            };
            let wide = sd.hypot(excess);
            let (spike_share, spike_share_slope) = share(mean, mean_slope, sd, sd_slope);
            let (slab_share, slab_share_slope) =
                share(mean + excess, mean_slope + excess_slope, wide, (sd * sd_slope + excess * excess_slope) / wide);
            let base = g0 + average;
            let base_slope = g1 + average_slope;
            [
                (f.log_spike() + base + spike_share, base_slope + spike_share_slope),
                (f.log_slab() + r.ln() + base + slab_share, base_slope + dr * f.mu * l2 / r + slab_share_slope),
            ]
        };
        let terms = [parts(below, -1.0), parts(above, 1.0)];
        let weight = |side: &[(f64, f64); 2]| log_add_exp(side[0].0, side[1].0);
        let (low, high) = (weight(&terms[0]), weight(&terms[1]));
        let near = if x > 0.0 { high } else { low };
        if !(low.is_finite() && high.is_finite()) || low.min(high) < near - 40.0 {
            return None;
        }
        let value = log_add_exp(low, high);
        let slope = terms.iter().flatten().filter(|t| t.0 > f64::NEG_INFINITY).map(|(v, s)| (v - value).exp() * s).sum();
        Some((value, slope))
    }

    /// The core term, the tail term and the floor term of the asymptotic form at `d`, each with its derivative.
    /// With early detections the packet product is `g = g_rest + F H(v - y_max)`: the floor term
    /// `F S_w(y_max - x)` integrates the second part exactly, the core and tail terms describe the first, whose
    /// tail beyond the last reading is `g - F`. Without early detections the floor term is `-inf`.
    ///
    /// The core term expands the integrand about the core of the packet product at `c`, the tail term about the
    /// packets at `x`; where the product stays log concave they hand over to mode terms, as `core_hand_over` and
    /// `tail_hand_over` describe. Each is a separate mode only while the other factor of the integrand leaves it
    /// near its expansion point. A nearly
    /// Gaussian packet core under a wider offset pulls the tail term's mode almost all the way to `c`, where it
    /// repeats the core term, and a wide core under a narrow offset pulls the core term's mode to `x`, where it
    /// repeats the tail term. With `rho` the fraction of the way a mode moves, the tail term takes the weight
    /// `m(rho_T)` and the core term `1 - (1 - m(rho_C)) m(rho_T)`, with `m` of `mode_weight`: both weights are one
    /// while the modes stay apart, and the terms and their derivatives are then unchanged, and for a single
    /// Gaussian mode, where `rho_C + rho_T = 1`, the weights add up to one.
    fn terms(&self, d: f64) -> [(f64, f64); 3] {
        let f = &self.factor;
        let x = d;
        let w = self.center - x;
        let (core_log, core_slope_w, core_shift, core_shift_slope) = self.core_density(w);
        let mut core = self.log_z + core_log;
        let mut core_slope = -core_slope_w;
        let [lg, l1, l2, l3] = match f.floor() {
            Some((log_floor, y_max)) if x > y_max => f.log_rest_derivatives(x, log_floor),
            _ => f.log_g_derivatives(x),
        };
        let offset_var = self.offset_variance();
        let (average, average_slope) = gaussian_average(l1, l2, l3, offset_var);
        let z = l1 * f.mu;
        let (r, dr) = excess_factor(z);
        let mix = 1.0 - f.pi + f.pi * r;
        let mut tail = lg + average + mix.ln();
        let mut tail_slope = l1 + average_slope + f.pi * dr * f.mu * l2 / mix;
        if let Some(bent) = self.bent_tail(x, [lg, l1, l2, l3], offset_var) {
            (tail, tail_slope) = bent;
        }
        let floor = match f.floor() {
            Some((log_floor, y_max)) => {
                // d/dd log S_w(y_max - x) = f_w(y_max - x) / S_w(y_max - x)
                let survival = f.log_shift_survival(y_max - x);
                (log_floor + survival, (f.log_shift_density(y_max - x) - survival).exp())
            }
            None => (f64::NEG_INFINITY, 0.0),
        };
        // the tail term's mode moves from x by the tilted mean of the offset, V l1 / (1 - l2 V), and of the
        // excess, d log(mix) / d l1; the core term's mode moves from c as core_density reports
        let (h, dh) = bend_factor(l2 * offset_var);
        let tail_shift = offset_var * l1 * h + f.pi * f.mu * dr / mix;
        let tail_shift_slope = offset_var * (l2 * h + l1 * dh * l3 * offset_var)
            + f.pi * f.mu * (excess_curvature(z) * mix - f.pi * dr * dr) / (mix * mix) * f.mu * l2;
        // without early detections, whose floor dominates far away, the terms hand over to mode terms where the
        // packet product stays log concave
        let hand = |term: &mut f64, term_slope: &mut f64, [mode, mode_slope, weight, weight_slope]: [f64; 4]| {
            let gap = mode - *term;
            *term_slope += weight_slope * gap + weight * (mode_slope - *term_slope);
            *term += weight * gap;
        };
        if f.lambda == 0.0 {
            if core.is_finite() {
                if let Some(handed) = self.core_hand_over(x, core_shift, core_shift_slope) {
                    hand(&mut core, &mut core_slope, handed);
                }
            }
            if tail.is_finite() {
                if let Some(handed) = self.tail_hand_over(x, [l1, l2, l3], offset_var) {
                    hand(&mut tail, &mut tail_slope, handed);
                }
            }
        }
        let (tail_weight, tail_weight_slope) = if w == 0.0 || tail == f64::NEG_INFINITY {
            (0.0, 0.0)
        } else {
            let rho = tail_shift / w;
            let rho_slope = (tail_shift_slope + rho) / w;
            if rho.is_finite() && rho_slope.is_finite() {
                let (m, dm) = mode_weight(rho);
                (m, if dm == 0.0 { 0.0 } else { dm * rho_slope })
            } else {
                (0.0, 0.0)
            }
        };
        let (core_own, core_own_slope) = if w == 0.0 {
            (1.0, 0.0)
        } else {
            let rho = -core_shift / w;
            let rho_slope = (core_shift_slope + rho) / w;
            if rho.is_finite() && rho_slope.is_finite() {
                let (m, dm) = mode_weight(rho);
                (m, if dm == 0.0 { 0.0 } else { dm * rho_slope })
            } else {
                (1.0, 0.0)
            }
        };
        let core_weight = 1.0 - (1.0 - core_own) * tail_weight;
        let core_weight_slope = core_own_slope * tail_weight - (1.0 - core_own) * tail_weight_slope;
        let weigh = |(value, slope): (f64, f64), weight: f64, weight_slope: f64| {
            if weight == 1.0 && weight_slope == 0.0 {
                (value, slope)
            } else if weight <= 0.0 || value == f64::NEG_INFINITY {
                (f64::NEG_INFINITY, 0.0)
            } else {
                (value + weight.ln(), slope + weight_slope / weight)
            }
        };
        [
            weigh((core, core_slope), core_weight, core_weight_slope),
            weigh((tail, tail_slope), tail_weight, tail_weight_slope),
            floor,
        ]
    }

    /// The part of the asymptotic form above the floor of early detections, the log sum of the core and tail terms,
    /// and the floor term, each with its derivative.
    fn split(&self, d: f64) -> ((f64, f64), (f64, f64)) {
        let [(core, core_slope), (tail, tail_slope), floor] = self.terms(d);
        let top = core.max(tail);
        if top == f64::NEG_INFINITY {
            return ((top, 0.0), floor);
        }
        let (a, b) = ((core - top).exp(), (tail - top).exp());
        ((top + (a + b).ln(), (a * core_slope + b * tail_slope) / (a + b)), floor)
    }

    /// The junction at the upper end of the table, from the table's `value` and `slope` there, over `length`, with
    /// the slope the table keeps at that end. It corrects the whole asymptotic form, except for a factor with a floor
    /// of early detections, whose likelihood falls onto the floor, a bend the junction must not outlast. Where the
    /// table holds more than a thousandth above the floor term, the junction corrects the log of the part above it and
    /// fades with that part, since a correction of the whole would dip below the floor and rise back. Where the table
    /// has reached its floor, its one sided end slope, taken across the bend, may come out above zero, so the table
    /// keeps it at zero at most, and the correction decays within half the length over which the asymptotic form's
    /// slope changes by its own size, so it never outlasts the fall it corrects.
    fn right_junction(&self, end: f64, (value, slope): (f64, f64), length: f64) -> (Junction, f64) {
        let whole = |slope: f64, length: f64| {
            let (asymptotic, asymptotic_slope) = self.asymptotic(end);
            Junction { end, offset: value - asymptotic, slope: slope - asymptotic_slope, length, above_floor: false }
        };
        if self.factor.floor().is_none() {
            return (whole(slope, length), slope);
        }
        let ((rest, rest_slope), (floor, floor_slope)) = self.split(end);
        let share = -(floor - value).exp_m1();
        if share >= 1e-3 && rest > f64::NEG_INFINITY {
            let above_slope = (slope - (1.0 - share) * floor_slope) / share;
            let junction = Junction { end, offset: value + share.ln() - rest, slope: above_slope - rest_slope, length, above_floor: true };
            return (junction, slope);
        }
        let delta = 0.01;
        let (a1, a2) = (self.asymptotic(end).1, self.asymptotic(end + delta).1);
        let bend = (a2 - a1).abs() / delta;
        let length = if bend > 0.0 { length.min(0.5 * a1.abs() / bend).max(MIN_STEP_CM) } else { length };
        let slope = slope.min(0.0);
        (whole(slope, length), slope)
    }

    /// The asymptotic log likelihood of the factor at `d` and its derivative: the log sum of its terms, for nearly
    /// Gaussian packets far below the peak faded onto the plain core term (`far_fade`).
    fn asymptotic(&self, d: f64) -> (f64, f64) {
        let terms = self.terms(d);
        if terms[2].0 == f64::NEG_INFINITY {
            let ((core, core_slope), (tail, tail_slope)) = (terms[0], terms[1]);
            let top = core.max(tail);
            if top == f64::NEG_INFINITY {
                return (top, 0.0);
            }
            let (a, b) = ((core - top).exp(), (tail - top).exp());
            return self.far_fade(d, (top + (a + b).ln(), (a * core_slope + b * tail_slope) / (a + b)));
        }
        let top = terms.iter().map(|t| t.0).fold(f64::NEG_INFINITY, f64::max);
        if top == f64::NEG_INFINITY {
            return (top, 0.0);
        }
        let weights: Vec<f64> = terms.iter().map(|t| (t.0 - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        let slope = weights.iter().zip(&terms).map(|(w, t)| if *w > 0.0 { w * t.1 } else { 0.0 }).sum::<f64>() / total;
        (top + total.ln(), slope)
    }

    /// The asymptotic form `(value, slope)` of nearly Gaussian packets under a Gaussian offset without early
    /// detections, faded log linearly onto the plain core term `P`, the packet product's core smoothed by the shift, as
    /// `P` falls from FAR_DEPTH to twice as far below its own peak. The offset then pulls the integrand's mode to where
    /// the packet product turns from Gaussian to the polynomial tails of its Student t densities, a thousand core
    /// scales from the packets, and the core and the tail terms each miss the likelihood by thousands of nats, while
    /// their weights and mode terms moved the sum up and down by as much: 11,460 nats at one distance 67 m past a
    /// table, 77,000 nats where a weight left zero 139 m past another, and slopes of the wrong sign hundreds of metres
    /// out. `P` decays with its own shift density and lies below the full form there, so the fade lowers the value and
    /// the continuation decays monotonically. So far below its peak the likelihood carries no weight in any posterior.
    /// Without an offset the tail term holds the packet product itself, and a Student t offset, whose log slope falls
    /// away from its centre, leaves the integrand's mode at the core, where the core term holds, so neither fades.
    fn far_fade(&self, d: f64, (value, slope): (f64, f64)) -> (f64, f64) {
        let gaussian = rising(self.factor.nu, 30.0, 100.0).0;
        if gaussian == 0.0 || self.factor.shape != OffsetShape::Gaussian || self.factor.tau == 0.0 || !value.is_finite() {
            return (value, slope);
        }
        let (plain_peak, _, _, _) = self.core_density(0.0);
        let (plain, plain_slope_w, _, _) = self.core_density(self.center - d);
        let depth = plain_peak - plain;
        let (kept, kept_slope) = falling(depth, FAR_DEPTH, 2.0 * FAR_DEPTH);
        let kept = 1.0 - gaussian * (1.0 - kept);
        if kept == 1.0 || !plain.is_finite() {
            return (value, slope);
        }
        // d/dd of the depth is the slope of P in w, w = c - d
        let kept_slope = gaussian * kept_slope * plain_slope_w;
        let (plain, plain_slope) = (self.log_z + plain, -plain_slope_w);
        (kept * value + (1.0 - kept) * plain, kept * slope + (1.0 - kept) * plain_slope + kept_slope * (value - plain))
    }

    /// Value and derivative of the continuation beyond the table.
    fn continuation(&self, d: f64) -> (f64, f64) {
        if d >= self.right.end && self.right.above_floor {
            let ((rest, rest_slope), (floor, floor_slope)) = self.split(d);
            let (j, dj) = self.right.at(d - self.right.end);
            let value = log_add_exp(rest + j, floor);
            let share = (rest + j - value).exp();
            return (value, share * (rest_slope + dj) + (1.0 - share) * floor_slope);
        }
        let (value, slope) = self.asymptotic(d);
        if d >= self.right.end {
            let (j, dj) = self.right.at(d - self.right.end);
            (value + j, slope + dj)
        } else {
            let (j, dj) = self.left.at(self.left.end - d);
            (value + j, slope - dj)
        }
    }
}

/// A uniform run of the table, `values[i]` at `start + i * step`, with the slopes at its two ends; a neighbouring
/// run shares the end point and its slope.
#[derive(Clone, Debug)]
struct Piece {
    start: f64,
    step: f64,
    /// The inverse of the step. The step is a power of two, so multiplying by the inverse equals dividing by the step
    /// exactly, and saves a division at every evaluation.
    inverse: f64,
    values: Vec<f64>,
    start_slope: f64,
    end_slope: f64,
}

impl Piece {
    fn end(&self) -> f64 {
        self.start + (self.values.len() - 1) as f64 * self.step
    }

    #[inline]
    fn linear(&self, d: f64) -> f64 {
        let n = self.values.len();
        let t = ((d - self.start) * self.inverse).max(0.0);
        let i = (t.floor() as usize).min(n - 2);
        let f = t - i as f64;
        self.values[i] * (1.0 - f) + self.values[i + 1] * f
    }

    /// Cubic Hermite interpolation with central difference slopes inside and the stored slopes at the ends.
    fn smooth(&self, d: f64) -> (f64, f64) {
        let v = &self.values;
        let n = v.len();
        let h = self.step;
        let t = ((d - self.start) / h).max(0.0);
        let i = (t.floor() as usize).min(n - 2);
        let f = t - i as f64;
        let slope_at = |k: usize| {
            if k == 0 {
                self.start_slope * h
            } else if k == n - 1 {
                self.end_slope * h
            } else {
                0.5 * (v[k + 1] - v[k - 1])
            }
        };
        let (m0, m1) = (slope_at(i), slope_at(i + 1));
        let f2 = f * f;
        let f3 = f2 * f;
        let value = (2.0 * f3 - 3.0 * f2 + 1.0) * v[i] + (f3 - 2.0 * f2 + f) * m0 + (-2.0 * f3 + 3.0 * f2) * v[i + 1] + (f3 - f2) * m1;
        let slope = (6.0 * f2 - 6.0 * f) * v[i] + (3.0 * f2 - 4.0 * f + 1.0) * m0 + (-6.0 * f2 + 6.0 * f) * v[i + 1] + (3.0 * f2 - 2.0 * f) * m1;
        (value, slope / h)
    }
}

/// A run under construction with the values of every factor.
#[derive(Clone, Debug)]
struct Draft {
    start: f64,
    step: f64,
    parts: Vec<Vec<f64>>,
}

impl Draft {
    fn len(&self) -> usize {
        self.parts[0].len()
    }

    fn end(&self) -> f64 {
        self.start + (self.len() - 1) as f64 * self.step
    }

    fn total(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.parts.iter().map(|p| p[i]).sum()).collect()
    }

    fn slice(&self, a: usize, b: usize) -> Draft {
        Draft {
            start: self.start + a as f64 * self.step,
            step: self.step,
            parts: self.parts.iter().map(|p| p[a..=b].to_vec()).collect(),
        }
    }
}

/// Second order one sided slope at the start or the end of a run.
fn end_slope_of(values: &[f64], step: f64, at_end: bool) -> f64 {
    let n = values.len();
    if at_end {
        (3.0 * values[n - 1] - 4.0 * values[n - 2] + values[n - 3]) / (2.0 * step)
    } else {
        (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * step)
    }
}

/// Second order slope at the point shared by two runs, from its neighbours at `left` before and `right` after.
fn shared_slope(before: f64, value: f64, after: f64, left: f64, right: f64) -> f64 {
    (left * left * after - right * right * before + (right * right - left * left) * value) / (left * right * (left + right))
}

/// Log likelihood of one link's packet block as a function of the predicted distance: uniform runs of the table,
/// finer where the linear interpolation needs it, and the continuation of every factor beyond them.
#[derive(Clone, Debug)]
pub(crate) struct LinkTable {
    pieces: Vec<Piece>,
    tails: Vec<Tail>,
}

/// Largest mismatch in nats between a factor's table and its asymptotic form at the ends of the table.
const JUNCTION_TOLERANCE: f64 = 0.05;
/// Largest curvature of the log shift density times the core variance at an end of the table of a Student t offset,
/// where the core term's second order average of that density over the core holds.
const CORE_BEND: f64 = 0.125;
/// Most extensions of one end of the table.
const MAX_EXTENSIONS: usize = 12;

impl LinkTable {
    /// The table of a packet block given as calibrated distances and scales in centimetres. `norm` is
    /// `log_t_norm_for(model.nu)`.
    pub(crate) fn from_packets(distances: &[f64], scales: &[f64], model: &BlockModel, norm: f64) -> Self {
        debug_assert!((norm - log_t_norm_for(model.nu)).abs() < 1e-12);
        let factors: Vec<Factor> = if model.independent && distances.len() > 1 {
            distances.iter().zip(scales).map(|(y, s)| Factor::new(&[*y], &[*s], model, norm)).collect()
        } else {
            vec![Factor::new(distances, scales, model, norm)]
        };
        let (lo, hi) = factors
            .iter()
            .map(Factor::range)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), (c, d)| (a.min(c), b.max(d)));
        // With early detections the table reaches down to zero distance. There the survival of every packet bends
        // the packet product, and a high rate makes the product broad enough for its core statistics to mislead the
        // continuation below the readings, which once invented values tens of thousands of nats above the table's
        // peak, and without an offset left every value there not finite.
        let to_zero = factors.iter().any(|f| f.lambda > 0.0) && lo > 0.0;
        let lo = if to_zero { 0.0 } else { lo };
        let rule = gauss_legendre(16);
        let step = factors.iter().map(Factor::initial_step).fold(TABLE_STEP_CM, f64::min);
        let d0 = (lo / step).floor() * step;
        let count = (((hi - d0) / step).ceil() as usize + 1).max(3);
        let base: Vec<FactorTable> = factors.iter().map(|f| f.tabulate(d0, step, count, &rule)).collect();
        let tails: Vec<Tail> = factors
            .iter()
            .zip(&base)
            .map(|(factor, table)| Tail {
                factor: factor.clone(),
                log_z: table.log_z,
                center: table.center,
                core_var: table.core_var,
                left: Junction::default(),
                right: Junction::default(),
            })
            .collect();
        let draft = Draft { start: d0, step, parts: base.into_iter().map(|t| t.values).collect() };
        let mut drafts = refine(&factors, draft, &rule);
        let span = (hi - lo).max(1.0);
        let top = drafts.iter().flat_map(Draft::total).fold(f64::NEG_INFINITY, f64::max);
        // extend each end until every factor's asymptotic form meets the table and, beyond the packets with a
        // Gaussian offset, until the packet tail term leads the core term by ten nats: the continuation then
        // starts past the crossover between the offset regime and the packet tail regime, where neither term
        // alone describes the likelihood. A Student t offset also extends each end at a positive distance until the
        // core term's average of the shift density over the core holds (Tail::core_holds). A table that reaches down
        // to zero distance for its early detections ends there, since no solver asks below it, where every extension
        // would add half the distance to the packets.
        for at_end in [true, false] {
            if !at_end && to_zero {
                continue;
            }
            for _ in 0..MAX_EXTENSIONS {
                let edge = if at_end { drafts.last().unwrap() } else { &drafts[0] };
                let (d, index) = if at_end { (edge.end(), edge.len() - 1) } else { (edge.start, 0) };
                let values: Vec<f64> = edge.parts.iter().map(|p| p[index]).collect();
                let met = tails.iter().zip(&values).all(|(tail, v)| {
                    // the crossover is ahead while the tail term describes a mode of its own and the core term is
                    // within ten nats of it and of the floor of early detections, for a Student t offset only while
                    // the core term decays faster
                    let crossing_ahead = at_end && {
                        let [(core, core_slope), (tail_term, tail_slope), (floor, _)] = tail.terms(d);
                        let steeper = tail.factor.shape == OffsetShape::Gaussian || core_slope < tail_slope;
                        tail_term > f64::NEG_INFINITY && steeper && core > tail_term - 10.0 && core > floor - 10.0
                    };
                    !crossing_ahead && tail.core_holds(d) && (v - tail.asymptotic(d).0).abs() <= JUNCTION_TOLERANCE
                });
                let reach = tails.iter().map(|t| (d - t.center).abs()).fold(0.0, f64::max);
                // the table grows at most eightfold, and not where it falls 700 nats below its peak, beyond the
                // range of its scaled sums
                if met || reach > 8.0 * span || values.iter().sum::<f64>() < top - 700.0 {
                    break;
                }
                let extension = ((0.5 * reach).max(16.0 * step) / step).ceil() as usize;
                let start = if at_end { d } else { d - extension as f64 * step };
                let mut parts: Vec<Vec<f64>> = factors.iter().map(|f| f.tabulate(start, step, extension + 1, &rule).values).collect();
                for (part, value) in parts.iter_mut().zip(&values) {
                    // the shared point keeps its value, so the table stays continuous
                    if at_end {
                        part[0] = *value;
                    } else {
                        part[extension] = *value;
                    }
                }
                let added = refine(&factors, Draft { start, step, parts }, &rule);
                if at_end {
                    drafts.extend(added);
                } else {
                    drafts.splice(0..0, added);
                }
            }
        }
        Self::assemble(drafts, tails)
    }

    /// The runs with their shared slopes and the continuation of every factor joined to the ends.
    fn assemble(drafts: Vec<Draft>, mut tails: Vec<Tail>) -> Self {
        let totals: Vec<Vec<f64>> = drafts.iter().map(Draft::total).collect();
        let count = drafts.len();
        let mut pieces = Vec::with_capacity(count);
        for k in 0..count {
            let (draft, values) = (&drafts[k], &totals[k]);
            let n = values.len();
            let start_slope = if k == 0 {
                end_slope_of(values, draft.step, false)
            } else {
                let before = &totals[k - 1];
                shared_slope(before[before.len() - 2], values[0], values[1], drafts[k - 1].step, draft.step)
            };
            let end_slope = if k + 1 == count {
                end_slope_of(values, draft.step, true)
            } else {
                shared_slope(values[n - 2], values[n - 1], totals[k + 1][1], draft.step, drafts[k + 1].step)
            };
            pieces.push(Piece { start: draft.start, step: draft.step, inverse: 1.0 / draft.step, values: values.clone(), start_slope, end_slope });
        }
        let (first, last) = (&drafts[0], &drafts[count - 1]);
        let (mut end_slopes, mut kept) = (Vec::with_capacity(tails.len()), true);
        for (f, tail) in tails.iter_mut().enumerate() {
            let span = |end: f64, tail: &Tail, step: f64| (4.0 * step).max(0.25 * (end - tail.center).abs());
            let (left_value, left_slope) = tail.asymptotic(first.start);
            let left = Junction {
                end: first.start,
                offset: first.parts[f][0] - left_value,
                slope: left_slope - end_slope_of(&first.parts[f], first.step, false),
                length: span(first.start, tail, first.step),
                above_floor: false,
            };
            let end = (last.parts[f][last.len() - 1], end_slope_of(&last.parts[f], last.step, true));
            let (right, end_slope) = tail.right_junction(last.end(), end, span(last.end(), tail, last.step));
            kept &= end_slope == end.1;
            end_slopes.push(end_slope);
            tail.left = left;
            tail.right = right;
        }
        // an end slope kept at zero by a factor at its floor is the table's end slope too
        if !kept {
            pieces[count - 1].end_slope = end_slopes.iter().sum();
        }
        Self { pieces, tails }
    }

    /// Value and derivative of the continuation beyond the table, the sum over its factors.
    fn outside(&self, d: f64) -> (f64, f64) {
        self.tails.iter().map(|t| t.continuation(d)).fold((0.0, 0.0), |(v, s), (a, b)| (v + a, s + b))
    }

    /// The run holding `d`, None beyond the table.
    #[inline]
    fn piece(&self, d: f64) -> Option<&Piece> {
        let first = &self.pieces[0];
        if self.pieces.len() == 1 {
            return (d >= first.start && d <= first.end()).then_some(first);
        }
        if d.is_nan() || d < first.start {
            return None;
        }
        // the first run that ends at or beyond d, the runs following each other
        self.pieces.get(self.pieces.partition_point(|p| p.end() < d))
    }

    /// The log likelihood by linear interpolation of the table, the continuation beyond it.
    #[inline]
    pub(crate) fn eval(&self, d: f64) -> f64 {
        match self.piece(d) {
            Some(piece) => piece.linear(d),
            None => self.outside(d).0,
        }
    }

    /// Value and derivative with respect to the distance by cubic Hermite interpolation of the table with central
    /// difference slopes inside every run and second order slopes at its ends, the continuation beyond it. The
    /// cooperative optimizer needs a continuous gradient, whereas the anchor only estimator keeps the linear
    /// interpolation of `eval`.
    pub(crate) fn eval_smooth(&self, d: f64) -> (f64, f64) {
        match self.piece(d) {
            Some(piece) => piece.smooth(d),
            None => self.outside(d),
        }
    }

    /// The tabulated range.
    pub(crate) fn range(&self) -> (f64, f64) {
        (self.pieces[0].start, self.pieces.last().unwrap().end())
    }

    /// The table with its continuation tabulated between `nearest` and `farthest`, the distances it will be
    /// evaluated at.
    pub(crate) fn far(&self, nearest: f64, farthest: f64) -> FarTable<'_> {
        let (lo, hi) = self.range();
        let continuation = |d: f64| self.outside(d);
        let below = FarNodes::new(if nearest < lo { hermite_nodes(nearest, lo, continuation) } else { Vec::new() });
        let above = FarNodes::new(if farthest > hi { hermite_nodes(hi, farthest, continuation) } else { Vec::new() });
        let ends: Vec<f64> = self.pieces.iter().map(Piece::end).collect();
        let buckets = ((hi - lo) / TABLE_STEP_CM).floor() as usize + 1;
        // the first run ending no nearer than a bucket before the bucket's start
        let runs = (0..buckets).map(|b| ends.partition_point(|end| *end < lo + (b as f64 - 1.0) * TABLE_STEP_CM)).collect();
        FarTable { table: self, lo, hi, below, above, ends, runs }
    }
}

/// Relative accuracy of the tabulated continuation at the middle of every interval between its nodes, far below the
/// continuation's own error against the likelihood and the error of the table's linear interpolation. Elsewhere
/// the cubic may miss by more, and its test allows a hundred times as much, 1e-8 of the value.
const FAR_TOLERANCE: f64 = 1e-10;
/// Spacing in centimetres of the first nodes of the tabulated continuation, halved where needed.
const FAR_STEP_CM: f64 = 8.0;
/// Smallest spacing of those nodes in centimetres.
const FAR_MIN_STEP_CM: f64 = 1e-3;
/// Width in centimetres of the buckets that index the nodes.
const FAR_BUCKET_CM: f64 = 1.0;

/// A table whose continuation is tabulated beyond both ends, for a caller that evaluates the table at very many
/// distances, as the anchor-only grid does at every cell and height node. The continuation costs microseconds a
/// distance, its mode terms Newton steps each, where the table itself is a lookup. The nodes hold the
/// continuation's value and slope, cubic Hermite interpolation joins them, and an interval is halved until the
/// interpolation meets the continuation at its midpoint within FAR_TOLERANCE of the value, the midpoint then a
/// node as well. Inside the table the values are the table's, and beyond the nodes the continuation's.
pub(crate) struct FarTable<'a> {
    table: &'a LinkTable,
    lo: f64,
    hi: f64,
    /// Nodes from the nearest distance to the table's start.
    below: FarNodes,
    /// Nodes from the table's end to the farthest distance.
    above: FarNodes,
    /// The end of every run of the table.
    ends: Vec<f64>,
    /// For every bucket of TABLE_STEP_CM from the table's start the run where the search for a distance of that
    /// bucket starts.
    runs: Vec<usize>,
}

impl FarTable<'_> {
    /// The log likelihood at `d`, the table's inside it and the tabulated continuation beyond it.
    #[inline]
    pub(crate) fn eval(&self, d: f64) -> f64 {
        let nodes = if d > self.hi {
            &self.above
        } else if d < self.lo {
            &self.below
        } else {
            // the first run ending at or beyond d, as the table's own search finds it
            let mut k = self.runs[((d - self.lo) / TABLE_STEP_CM) as usize];
            while self.ends[k] < d {
                k += 1;
            }
            return self.table.pieces[k].linear(d);
        };
        match nodes.at(d) {
            Some(value) if value.is_finite() => value,
            _ => self.table.eval(d),
        }
    }

    /// The run and the interval of the run that `eval` interpolates at `d`, a distance of the table, found by the
    /// same search and the same arithmetic.
    #[inline]
    fn locate(&self, d: f64) -> (usize, usize) {
        let mut k = self.runs[((d - self.lo) / TABLE_STEP_CM) as usize];
        while self.ends[k] < d {
            k += 1;
        }
        let piece = &self.table.pieces[k];
        let t = ((d - piece.start) * piece.inverse).max(0.0);
        (k, (t.floor() as usize).min(piece.values.len() - 2))
    }

    /// The structures that bound `eval` over intervals of distance, see `FarBounds`.
    pub(crate) fn bounds(&self) -> FarBounds {
        let knots = Extremes::new(self.table.pieces.iter().flat_map(|piece| piece.values.iter().map(|v| (*v, *v))));
        let mut offsets = Vec::with_capacity(self.table.pieces.len());
        let mut count = 0;
        for piece in &self.table.pieces {
            offsets.push(count);
            count += piece.values.len();
        }
        let first_below = self.below.at.first().map_or(f64::NAN, |d| self.table.eval(*d));
        FarBounds { knots, offsets, below: self.below.extremes(), above: self.above.extremes(), first_below }
    }
}

/// Certified bounds of `FarTable::eval` over any interval of distance [a, b], the function it evaluates rather than
/// the continuation it tabulates.
///
/// Inside the table `eval` interpolates linearly between the knots of its runs, so its extremes over [a, b] lie at the
/// knots within or at the two ends: the ends are evaluated as `eval` evaluates them, and the knots in between come
/// from a segment tree over every knot of every run. Beyond the table every cubic of the tabulated continuation
/// takes its extremes on the coordinates of an interval at their ends or at the real roots of its derivative: those
/// of every whole interval are computed once into a segment tree, and the two partial intervals at the ends of [a, b]
/// are solved as a query asks. The table's own value at the first node below it, where `eval` asks the continuation
/// itself, enters as a point. Wherever `eval` could ask the continuation over a stretch, before the first node, beyond
/// the last or on a cubic that is not finite, the bounds are minus and plus infinity, which the caller refines to
/// single distances. Every bound is widened by BOUND_MARGIN of the size of the values that make it.
pub(crate) struct FarBounds {
    /// Every knot of every run, the runs one after the other.
    knots: Extremes,
    /// The index of the first knot of every run in `knots`.
    offsets: Vec<usize>,
    /// The extremes of every whole interval between the nodes below the table and above it.
    below: Extremes,
    above: Extremes,
    /// `eval` at the first node below the table, the continuation's value there, not a number without nodes.
    first_below: f64,
}

impl FarBounds {
    /// Lower and upper bounds of `far.eval(d)` over every distance d in [a, b], `far` the table these bounds were
    /// built from.
    pub(crate) fn over(&self, far: &FarTable, a: f64, b: f64) -> (f64, f64) {
        if a.is_nan() || b.is_nan() || a > b {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let mut bounds = (f64::INFINITY, f64::NEG_INFINITY);
        let mut merge = |(low, high): (f64, f64)| bounds = (bounds.0.min(low), bounds.1.max(high));
        if a < far.lo {
            merge(far.below.bounds(&self.below, a, b.min(far.lo), Some(self.first_below)));
        }
        if a <= far.hi && b >= far.lo {
            merge(self.inside(far, a.max(far.lo), b.min(far.hi)));
        }
        if b > far.hi {
            merge(far.above.bounds(&self.above, a.max(far.hi), b, None));
        }
        bounds
    }

    /// The bounds over [a, b] within the table: the values at both ends and the knots after the interval of `a` up to
    /// the start of the interval of `b`.
    fn inside(&self, far: &FarTable, a: f64, b: f64) -> (f64, f64) {
        let (at_a, at_b) = (far.eval(a), far.eval(b));
        if at_a.is_nan() || at_b.is_nan() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let (mut low, mut high) = (at_a.min(at_b), at_a.max(at_b));
        let ((ka, ia), (kb, ib)) = (far.locate(a), far.locate(b));
        let (first, last) = (self.offsets[ka] + ia + 1, self.offsets[kb] + ib);
        if first <= last {
            let (knot_low, knot_high) = self.knots.over(first, last);
            low = low.min(knot_low);
            high = high.max(knot_high);
        }
        widened(low, high, low.abs().max(high.abs()))
    }
}

/// `low` and `high` widened by BOUND_MARGIN of `size` and one.
#[inline]
pub(crate) fn widened(low: f64, high: f64, size: f64) -> (f64, f64) {
    let margin = BOUND_MARGIN * (1.0 + size);
    (low - margin, high + margin)
}

/// The smallest lower and the largest upper value over any run of consecutive entries, from a segment tree of both.
/// A value that is not a number counts as minus infinity below and plus infinity above.
struct Extremes {
    count: usize,
    low: Vec<f64>,
    high: Vec<f64>,
}

impl Extremes {
    fn new(entries: impl Iterator<Item = (f64, f64)>) -> Self {
        let entries: Vec<(f64, f64)> = entries.collect();
        let count = entries.len();
        let (mut low, mut high) = (vec![f64::INFINITY; 2 * count], vec![f64::NEG_INFINITY; 2 * count]);
        for (i, (l, h)) in entries.into_iter().enumerate() {
            low[count + i] = if l.is_nan() { f64::NEG_INFINITY } else { l };
            high[count + i] = if h.is_nan() { f64::INFINITY } else { h };
        }
        for i in (1..count).rev() {
            low[i] = low[2 * i].min(low[2 * i + 1]);
            high[i] = high[2 * i].max(high[2 * i + 1]);
        }
        Self { count, low, high }
    }

    /// The extremes over the entries `first..=last`.
    fn over(&self, first: usize, last: usize) -> (f64, f64) {
        let (mut left, mut right) = (first + self.count, last + 1 + self.count);
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        while left < right {
            if left & 1 == 1 {
                low = low.min(self.low[left]);
                high = high.max(self.high[left]);
                left += 1;
            }
            if right & 1 == 1 {
                right -= 1;
                low = low.min(self.low[right]);
                high = high.max(self.high[right]);
            }
            left >>= 1;
            right >>= 1;
        }
        (low, high)
    }
}

/// The distances of the nodes in increasing order, the cubic of every interval between two nodes, and for every
/// bucket of FAR_BUCKET_CM from the first node the first node that lies no nearer than a bucket before it, where the
/// search for a distance of that bucket starts.
struct FarNodes {
    at: Vec<f64>,
    cubics: Vec<Cubic>,
    first: Vec<usize>,
}

impl FarNodes {
    fn new(nodes: Vec<[f64; 3]>) -> Self {
        let at: Vec<f64> = nodes.iter().map(|node| node[0]).collect();
        let cubics = nodes.windows(2).map(|pair| Cubic::new(pair[0], pair[1])).collect();
        let first = match (at.first(), at.last()) {
            (Some(start), Some(end)) => {
                let buckets = ((end - start) / FAR_BUCKET_CM).floor() as usize + 1;
                (0..buckets).map(|b| at.partition_point(|d| *d < start + (b as f64 - 1.0) * FAR_BUCKET_CM)).collect()
            }
            _ => Vec::new(),
        };
        Self { at, cubics, first }
    }

    /// The tabulated value at `d`, on the interval up to the first node no nearer than `d`, as a bisection over the
    /// nodes finds it; None outside the nodes.
    #[inline]
    fn at(&self, d: f64) -> Option<f64> {
        let start = *self.at.first()?;
        let bucket = ((d - start) / FAR_BUCKET_CM).floor();
        if bucket.is_nan() || bucket < 0.0 {
            return None;
        }
        let mut k = *self.first.get(bucket as usize)?;
        while k < self.at.len() && self.at[k] < d {
            k += 1;
        }
        (k > 0 && k < self.at.len()).then(|| self.cubics[k - 1].at(d))
    }

    /// The bounds of every cubic over the coordinates its interval gives it, nothing for an interval of no width,
    /// which `at` never takes.
    fn extremes(&self) -> Extremes {
        Extremes::new(self.cubics.iter().enumerate().map(|(k, cubic)| {
            if self.at[k + 1] > self.at[k] {
                cubic.extremes(0.0, cubic.coordinate(self.at[k + 1]))
            } else {
                (f64::INFINITY, f64::NEG_INFINITY)
            }
        }))
    }

    /// Bounds of the values `FarTable::eval` takes through these nodes over the distances [a, b]: the cubic of the
    /// interval up to the first node no nearer than a distance, and `first`, where given, at the first node itself.
    /// Minus and plus infinity where the continuation could answer, before the first node or beyond the last, and on
    /// a cubic that is not finite. `extremes` holds the bounds of every whole interval.
    fn bounds(&self, extremes: &Extremes, a: f64, b: f64, first: Option<f64>) -> (f64, f64) {
        let unbounded = (f64::NEG_INFINITY, f64::INFINITY);
        let (Some(&start), Some(&end)) = (self.at.first(), self.at.last()) else {
            return unbounded;
        };
        if !(a >= start && b <= end && a <= b) {
            return unbounded;
        }
        let mut bounds = (f64::INFINITY, f64::NEG_INFINITY);
        if a == start {
            match first {
                Some(value) if value.is_nan() => return unbounded,
                Some(value) => bounds = (value, value),
                None => {}
            }
            if b == start {
                return bounds;
            }
        }
        let mut merge = |(low, high): (f64, f64)| bounds = (bounds.0.min(low), bounds.1.max(high));
        // the intervals of the first distance above the first node and of b, as `at` finds them
        let (k_a, k_b) = (self.at.partition_point(|d| *d < a).max(1) - 1, self.at.partition_point(|d| *d < b) - 1);
        let (first_cubic, last_cubic) = (&self.cubics[k_a], &self.cubics[k_b]);
        if k_a == k_b {
            merge(first_cubic.extremes(first_cubic.coordinate(a), last_cubic.coordinate(b)));
        } else {
            merge(first_cubic.extremes(first_cubic.coordinate(a), first_cubic.coordinate(self.at[k_a + 1])));
            if k_b > k_a + 1 {
                merge(extremes.over(k_a + 1, k_b - 1));
            }
            merge(last_cubic.extremes(0.0, last_cubic.coordinate(b)));
        }
        bounds
    }
}

/// The cubic Hermite interpolant between two nodes `[d, value, slope]`, in the coordinate `t` from zero at the first
/// node to one at the second: `((c3 t + c2) t + c1) t + c0`.
#[derive(Clone, Copy, Debug)]
struct Cubic {
    start: f64,
    inverse: f64,
    c: [f64; 4],
}

impl Cubic {
    fn new(left: [f64; 3], right: [f64; 3]) -> Self {
        let h = right[0] - left[0];
        let (v0, v1, m0, m1) = (left[1], right[1], h * left[2], h * right[2]);
        Self { start: left[0], inverse: 1.0 / h, c: [v0, m0, 3.0 * (v1 - v0) - 2.0 * m0 - m1, 2.0 * (v0 - v1) + m0 + m1] }
    }

    #[inline]
    fn at(&self, d: f64) -> f64 {
        let t = (d - self.start) * self.inverse;
        ((self.c[3] * t + self.c[2]) * t + self.c[1]) * t + self.c[0]
    }

    /// The coordinate of `d` as `at` computes it.
    #[inline]
    fn coordinate(&self, d: f64) -> f64 {
        (d - self.start) * self.inverse
    }

    /// The value at the coordinate `t` as `at` computes it.
    #[inline]
    fn value(&self, t: f64) -> f64 {
        ((self.c[3] * t + self.c[2]) * t + self.c[1]) * t + self.c[0]
    }

    /// Bounds of `at` over the coordinates [t0, t1]: the extremes of the cubic at both ends and at the real roots of
    /// its derivative between them, with the vertex of the derivative where two roots nearly meet, widened by
    /// BOUND_MARGIN of the size of the coefficients, which covers the rounding of the evaluation, of the coordinate
    /// and of the roots. Minus and plus infinity for a cubic that is not finite or whose size nears the largest
    /// number, where `FarTable::eval` asks the continuation instead.
    fn extremes(&self, t0: f64, t1: f64) -> (f64, f64) {
        let size: f64 = self.c.iter().map(|c| c.abs()).sum();
        if !(size <= 1e300 && t0 <= t1) {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let (v0, v1) = (self.value(t0), self.value(t1));
        let (mut low, mut high) = (v0.min(v1), v0.max(v1));
        // the roots of 3 c3 t^2 + 2 c2 t + c1, the stable way, and the vertex
        let (a, b, c) = (3.0 * self.c[3], 2.0 * self.c[2], self.c[1]);
        let mut candidates = [f64::NAN; 3];
        if a == 0.0 {
            if b != 0.0 {
                candidates[0] = -c / b;
            }
        } else {
            candidates[2] = -b / (2.0 * a);
            let discriminant = b * b - 4.0 * a * c;
            if discriminant >= 0.0 {
                let q = -0.5 * (b + discriminant.sqrt().copysign(b));
                candidates[0] = q / a;
                if q != 0.0 {
                    candidates[1] = c / q;
                }
            }
        }
        for t in candidates {
            if t > t0 && t < t1 {
                let value = self.value(t);
                low = low.min(value);
                high = high.max(value);
            }
        }
        widened(low, high, size)
    }
}

/// Nodes `[d, value, slope]` of `f`, which returns a value and its slope, from `a` to `b`: FAR_STEP_CM apart at
/// first, every interval halved until its interpolation meets `f` at its midpoint within FAR_TOLERANCE, or down to
/// FAR_MIN_STEP_CM, and the midpoint of every interval a node.
fn hermite_nodes(a: f64, b: f64, f: impl Fn(f64) -> (f64, f64)) -> Vec<[f64; 3]> {
    let node = |d: f64| {
        let (value, slope) = f(d);
        [d, value, slope]
    };
    let count = ((b - a) / FAR_STEP_CM).ceil().max(1.0) as usize;
    let mut nodes = vec![node(a)];
    for k in 1..=count {
        let end = if k == count { b } else { a + (b - a) * k as f64 / count as f64 };
        // the right ends of the intervals still to check, the nearest last
        let mut pending = vec![node(end)];
        while let Some(&right) = pending.last() {
            let left = nodes[nodes.len() - 1];
            let middle = node(0.5 * (left[0] + right[0]));
            let miss = (Cubic::new(left, right).at(middle[0]) - middle[1]).abs();
            if miss <= FAR_TOLERANCE * (1.0 + middle[1].abs()) || !miss.is_finite() || right[0] - left[0] <= FAR_MIN_STEP_CM {
                nodes.push(middle);
                nodes.push(right);
                pending.pop();
            } else {
                pending.push(middle);
            }
        }
    }
    nodes
}

/// Values of every factor on the run of `draft`, with windows of half the step, refined again as needed, wherever
/// the linear interpolation error estimated from the second differences exceeds the tolerance. Windows stretch to
/// an end of the run within three points of it, so that every run keeps three points.
fn refine(factors: &[Factor], draft: Draft, rule: &[(f64, f64)]) -> Vec<Draft> {
    let count = draft.len();
    let total = draft.total();
    let flagged: Vec<usize> = (1..count - 1)
        .filter(|&i| {
            let e = (total[i - 1] - 2.0 * total[i] + total[i + 1]).abs() / 8.0;
            e.is_nan() || e > INTERPOLATION_TOLERANCE
        })
        .collect();
    // no feature of the likelihood is narrower than the packet product, so windows stop at 1/64 of its width
    let floor = factors.iter().map(Factor::packet_width).fold(f64::INFINITY, f64::min) / 64.0;
    if flagged.is_empty() || draft.step / 2.0 < MIN_STEP_CM.max(floor) {
        return vec![draft];
    }
    let mut windows: Vec<(usize, usize)> = Vec::new();
    for &i in &flagged {
        let (a, b) = (i.saturating_sub(2), (i + 2).min(count - 1));
        match windows.last_mut() {
            Some(last) if a <= last.1 + 3 => last.1 = last.1.max(b),
            _ => windows.push((a, b)),
        }
    }
    if windows[0].0 < 3 {
        windows[0].0 = 0;
    }
    let last = windows.len() - 1;
    if windows[last].1 + 3 > count - 1 {
        windows[last].1 = count - 1;
    }
    let mut drafts = Vec::new();
    let mut cursor = 0;
    for (a, b) in windows {
        if a > cursor {
            drafts.push(draft.slice(cursor, a));
        }
        let start = draft.start + a as f64 * draft.step;
        let step = draft.step / 2.0;
        let sub_count = 2 * (b - a) + 1;
        let mut parts: Vec<Vec<f64>> = factors.iter().map(|f| f.tabulate(start, step, sub_count, rule).values).collect();
        for (part, coarse) in parts.iter_mut().zip(&draft.parts) {
            // the shared end points keep their values, so the table stays continuous
            part[0] = coarse[a];
            part[sub_count - 1] = coarse[b];
        }
        drafts.extend(refine(factors, Draft { start, step, parts }, rule));
        cursor = b;
    }
    if cursor < count - 1 {
        drafts.push(draft.slice(cursor, count - 1));
    }
    drafts
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn model(kappa: f64, pi: f64, m: f64, t: f64) -> BlockModel {
        BlockModel {
            kappa,
            excess_probability: pi,
            excess_mean_rel: m,
            persistent_sd_rel: t,
            nu: 3.0,
            offset_shape: OffsetShape::Gaussian,
            early_detection: 0.0,
            independent: false,
        }
    }

    fn table(y: &[f64], s: &[f64], model: &BlockModel) -> LinkTable {
        LinkTable::from_packets(y, s, model, log_t_norm_for(model.nu))
    }

    fn points(t: &LinkTable) -> usize {
        t.pieces.iter().map(|p| p.values.len()).sum()
    }

    fn finest(t: &LinkTable) -> f64 {
        t.pieces.iter().map(|p| p.step).fold(TABLE_STEP_CM, f64::min)
    }

    fn model_from_json(p: &serde_json::Value) -> BlockModel {
        BlockModel {
            kappa: p["kappa"].as_f64().unwrap(),
            excess_probability: p["excess_probability"].as_f64().unwrap(),
            excess_mean_rel: p["excess_mean_rel"].as_f64().unwrap(),
            persistent_sd_rel: p["persistent_sd_rel"].as_f64().unwrap(),
            nu: p["nu"].as_f64().unwrap_or(3.0),
            offset_shape: if p["offset_shape"].as_str() == Some("student_t") { OffsetShape::StudentT } else { OffsetShape::Gaussian },
            early_detection: p["early_detection_per_cm"].as_f64().unwrap_or(0.0),
            independent: p["independent_packets"].as_bool().unwrap_or(false),
        }
    }

    fn floats(value: &serde_json::Value) -> Vec<f64> {
        value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    }

    /// Writes the engine's values for the cases of `LINK_CASES` to `LINK_OUT`, for the Python checks:
    /// `LINK_CASES=cases.json LINK_OUT=out.json cargo test --release dump_cases -- --ignored`.
    #[test]
    #[ignore]
    fn dump_cases() {
        let (Ok(cases), Ok(out)) = (std::env::var("LINK_CASES"), std::env::var("LINK_OUT")) else {
            return;
        };
        let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cases).unwrap()).unwrap();
        let mut results = Vec::new();
        for case in cases.as_array().unwrap() {
            let model = model_from_json(&case["params"]);
            let started = std::time::Instant::now();
            let t = table(&floats(&case["y"]), &floats(&case["s"]), &model);
            let micros = started.elapsed().as_micros() as f64;
            let d = floats(&case["d"]);
            let smooth: Vec<(f64, f64)> = d.iter().map(|x| t.eval_smooth(*x)).collect();
            let (lo, hi) = t.range();
            results.push(serde_json::json!({
                "lo": lo, "hi": hi, "step": finest(&t), "count": points(&t), "pieces": t.pieces.len(), "micros": micros,
                "linear": d.iter().map(|x| t.eval(*x)).collect::<Vec<f64>>(),
                "smooth": smooth.iter().map(|v| v.0).collect::<Vec<f64>>(),
                "slope": smooth.iter().map(|v| v.1).collect::<Vec<f64>>(),
            }));
        }
        std::fs::write(out, serde_json::to_string(&results).unwrap()).unwrap();
    }

    /// Blocks and models spanning the fitted parameter sets: Bern raw timings with distance dependent scales,
    /// TIERS, a near zero offset under a wide packet scale, a wide offset, IDLab, a Bregar environment, no offset at
    /// all, pure excess, both offset shapes and an offset far narrower than the packets; nearly Gaussian packets
    /// (nu = 1e6, the fitter's Gaussian core), under a wide offset, under an offset narrower than the packets and
    /// without any offset, with and without the excess, and 30 degrees of freedom.
    fn variety() -> Vec<(Vec<f64>, Vec<f64>, BlockModel)> {
        let calibration = crate::CalibrationModel::default();
        let bern = |y: &[f64]| y.iter().map(|v| calibration.scale_cm(*v)).collect::<Vec<f64>>();
        let mut student = model(0.28, 0.31, 8.0, 2.5);
        student.offset_shape = OffsetShape::StudentT;
        let mut independent = model(0.29, 0.37, 9.0, 3.0);
        independent.independent = true;
        let early = |mut m: BlockModel, rate: f64| {
            m.early_detection = rate;
            m
        };
        let ten = [812.0, 815.5, 809.0, 811.2, 845.0, 813.3, 810.1, 808.7, 816.4, 812.9];
        let mut early_student = early(model(0.28, 0.31, 8.0, 2.5), 1e-3);
        early_student.offset_shape = OffsetShape::StudentT;
        let early_independent = early(independent, 1.1e-4);
        let with_nu = |mut m: BlockModel, nu: f64| {
            m.nu = nu;
            m
        };
        let mut gaussian_core_student = with_nu(model(0.28, 0.31, 8.0, 2.5), 1e6);
        gaussian_core_student.offset_shape = OffsetShape::StudentT;
        vec![
            (ten.to_vec(), bern(&ten), with_nu(model(0.28, 0.31, 8.0, 2.5), 1e6)),
            (vec![812.0], bern(&[812.0]), with_nu(model(0.28, 0.31, 8.0, 2.5), 1e6)),
            (ten[..4].to_vec(), bern(&ten[..4]), gaussian_core_student),
            (vec![812.0], bern(&[812.0]), with_nu(model(0.9, 0.2, 3.0, 0.3), 1e6)),
            (ten.to_vec(), bern(&ten), with_nu(model(0.9, 0.2, 3.0, 0.3), 1e6)),
            (ten.to_vec(), bern(&ten), with_nu(model(0.28, 0.0, 8.0, 0.0), 1e6)),
            (ten.to_vec(), bern(&ten), with_nu(model(0.28, 0.31, 8.0, 0.0), 1e6)),
            (ten[..4].to_vec(), bern(&ten[..4]), with_nu(model(0.28, 0.31, 8.0, 2.5), 30.0)),
            (ten.to_vec(), bern(&ten), early(model(0.28, 0.31, 8.0, 2.5), 1.1e-4)),
            (ten[..4].to_vec(), bern(&ten[..4]), early(model(0.28, 0.31, 8.0, 2.5), 1e-3)),
            (vec![812.0], bern(&[812.0]), early(model(0.28, 0.31, 8.0, 2.5), 1e-3)),
            (ten[..4].to_vec(), bern(&ten[..4]), early_student),
            (ten[..4].to_vec(), bern(&ten[..4]), early_independent),
            (vec![1203.0, 1190.0, 1211.0, 1199.5], vec![102.69; 4], early(model(0.0823, 0.95, 2.956, 0.0026), 1e-3)),
            (ten.to_vec(), bern(&ten), model(0.28, 0.31, 8.0, 2.5)),
            (vec![812.0], bern(&[812.0]), model(0.28, 0.31, 8.0, 2.5)),
            (ten[..4].to_vec(), bern(&ten[..4]), student),
            (ten[..4].to_vec(), bern(&ten[..4]), model(0.27, 0.25, 7.0, 1e-4)),
            (ten[..4].to_vec(), bern(&ten[..4]), independent),
            (ten.iter().map(|v| v - 300.0).collect(), vec![11.17; 10], model(0.1863, 0.95, 3.83, 3.69)),
            (vec![1203.0, 1190.0, 1211.0, 1199.5], vec![102.69; 4], model(0.0823, 0.95, 2.956, 0.0026)),
            (vec![1203.0], vec![102.69], model(0.0401, 0.95, 8.11, 21.6)),
            (ten.to_vec(), vec![80.26; 10], model(0.0297, 0.95, 11.88, 15.66)),
            (ten[..4].to_vec(), vec![71.0, 55.0, 90.0, 63.0], model(0.0306, 0.95, 15.68, 11.65)),
            (ten[..4].to_vec(), bern(&ten[..4]), model(0.28, 0.4, 6.0, 0.0)),
            (vec![812.0], bern(&[812.0]), model(0.28, 1.0, 8.0, 2.5)),
        ]
    }

    /// The integral of the likelihood over the predicted distance: the trapezoid rule on every run of the table
    /// and Simpson steps growing by 5 % on the continuation out to 100 km.
    fn mass(t: &LinkTable) -> f64 {
        let mut total = 0.0;
        for piece in &t.pieces {
            let n = piece.values.len();
            total += piece.values.iter().enumerate().map(|(i, v)| if i == 0 || i == n - 1 { 0.5 } else { 1.0 } * v.exp() * piece.step).sum::<f64>();
        }
        let (lo, hi) = t.range();
        for (start, direction) in [(lo, -1.0), (hi, 1.0)] {
            let f = |x: f64| t.eval(start + direction * x).exp();
            let (mut a, mut width) = (0.0, finest(t));
            while a < 1e7 {
                let b = a + width;
                total += (b - a) / 6.0 * (f(a) + 4.0 * f(0.5 * (a + b)) + f(b));
                a = b;
                width *= 1.05;
            }
        }
        total
    }

    /// The cases of the parity fixture written by research/solid/link_likelihood_reference.py.
    fn fixture_cases() -> Vec<serde_json::Value> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/link_likelihood_parity.json");
        let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        fixture["cases"].as_array().unwrap().clone()
    }

    /// The parity fixture written by research/solid/link_likelihood_reference.py: blocks drawn for every tracked fit
    /// and the corrected Bern values at 1, 4 and 10 packets, both offset shapes, the independent packet ablation and
    /// early detections; nearly Gaussian packets (nu = 1e6); blocks of 1 to 4 and 10 packets drawn with false early
    /// readings far below the rest; the regimes of an offset near zero or absent, no excess or an excess probability
    /// near zero or one, the bounds of the early detection rate and wide packets; nearly Gaussian packets with one
    /// packet 20 or 30 m from the rest, no offset with early detections, a Student t offset at high rates below links of
    /// 30 to 60 m and packets of 0.3 cm; and the models of `variety` at 1, 4 and 10 packets. Every block holds its
    /// readings and points out to ten metres past both ends of its accuracy range, where the continuation takes over,
    /// and with early detections points from zero up to the range, which the table covers. Log L comes from the
    /// independent quadrature reference and from the Python module research/solid/link_likelihood.py. Inside the
    /// accuracy range the engine meets both within 0.02 nats; beyond it within 0.1 nats, five times tighter than the
    /// 0.5 nats asked of the continuation. Either bound grows by 2e-6 of the size of log L. A nearly Gaussian block
    /// without an offset and one packet 20 m from the other has log L below -2e5, falls ten nats and more over a
    /// lattice step, and the sums of its table miss by up to a millionth of that size, 0.25 nats.
    #[test]
    fn matches_the_reference_and_the_python_module() {
        let cases = fixture_cases();
        assert!(cases.len() >= 60);
        let mut worst = [0.0_f64; 4];
        let mut misses = Vec::new();
        for case in &cases {
            let t = table(&floats(&case["y"]), &floats(&case["s"]), &model_from_json(&case["params"]));
            let inside = case["inside"].as_u64().unwrap() as usize;
            let (reference, python) = (floats(&case["reference"]), floats(&case["python"]));
            let peak = reference.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for (k, d) in floats(&case["d"]).into_iter().enumerate() {
                let values = [t.eval(d), t.eval_smooth(d).0];
                let error = values.iter().map(|v| (v - reference[k]).abs()).fold(0.0, f64::max);
                let parity = values.iter().map(|v| (v - python[k]).abs()).fold(0.0, f64::max);
                let (bound, slot) = if k < inside { (0.02, 0) } else { (0.1, 2) };
                let bound = bound + 2e-6 * reference[k].abs();
                if !(error <= bound && parity <= bound) {
                    misses.push(format!("{} n={} d={d}: engine {values:?} reference {} python {}", case["name"], case["y"].as_array().unwrap().len(), reference[k], python[k]));
                }
                if peak - reference[k] <= 1000.0 {
                    worst[slot] = worst[slot].max(error);
                    worst[slot + 1] = worst[slot + 1].max(parity);
                }
            }
        }
        assert!(misses.is_empty(), "{} points miss:\n{}", misses.len(), misses.join("\n"));
        println!(
            "largest differences within 1000 nats of the peak, inside: reference {:.4} python {:.4}; beyond: reference {:.4} python {:.4}",
            worst[0], worst[1], worst[2], worst[3]
        );
    }

    #[test]
    fn the_student_survival_matches_scipy() {
        // scipy.stats.t.logsf(z, nu)
        let expected = [
            (3.0, 0.0, -std::f64::consts::LN_2),
            (3.0, 1.0, -1.6321892244941227),
            (3.0, 5.0, -4.867026104920422),
            (3.0, 50.0, -11.639784763243977),
            (3.0, -3.0, -0.02925832348536299),
            (3.0, 10000.0, -27.533297712883947),
            (1e6, 1.7, -3.110792608609465),
            (1e6, 10.0, -53.22873555112494),
            (1e6, -2.0, -0.023013047448695315),
            (0.7, 20.0, -3.2659102957075086),
        ];
        for (nu, z, want) in expected {
            let got = log_student_survival(z, nu);
            assert!((got - want).abs() < 1e-9 * (1.0 + want.abs()), "nu {nu} z {z}: {got} against {want}");
        }
    }

    #[test]
    fn the_fast_convolution_matches_the_direct_sum() {
        // a peaked sequence against a heavy tailed kernel, as in the Student t offset convolution
        for (na, nb) in [(1, 1), (7, 5), (300, 599), (1000, 1999)] {
            let a: Vec<f64> = (0..na).map(|i| (-((i as f64 - 0.4 * na as f64) / 3.0).powi(2)).exp() + 1e-6 / (1.0 + i as f64)).collect();
            let b: Vec<f64> = (0..nb).map(|i| 1.0 / (1.0 + ((i as f64 - nb as f64 / 2.0) / 4.0).powi(2)).powi(2)).collect();
            let fast = convolve(&a, &b);
            assert_eq!(fast.len(), na + nb - 1);
            for (n, value) in fast.iter().enumerate() {
                let direct: f64 = (0..na).filter(|&k| n >= k && n - k < nb).map(|k| a[k] * b[n - k]).sum();
                assert!((value - direct).abs() <= 1e-13 * (1.0 + direct.abs()) + 1e-6 * direct.abs(), "{n}: {value} {direct}");
            }
        }
    }

    #[test]
    #[ignore]
    fn time_fixture_cases() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/link_likelihood_parity.json");
        let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let started = std::time::Instant::now();
            let t = table(&floats(&case["y"]), &floats(&case["s"]), &model_from_json(&case["params"]));
            let (lo, hi) = t.range();
            eprintln!(
                "{} n={} {} rate={} : {:.1} ms, pieces {}, points {}, range {lo:.0} {hi:.0}",
                case["name"],
                case["n"],
                case["params"]["offset_shape"],
                case["params"]["early_detection_per_cm"],
                1e3 * started.elapsed().as_secs_f64(),
                t.pieces.len(),
                points(&t)
            );
        }
    }

    #[test]
    fn builds_finite_tables_on_the_lattice() {
        for (y, s, m) in variety() {
            let t = table(&y, &s, &m);
            for (k, piece) in t.pieces.iter().enumerate() {
                assert!(piece.values.iter().all(|v| v.is_finite()));
                assert!(piece.values.len() >= 2);
                assert_eq!((piece.start / piece.step).fract(), 0.0, "run start {} off the lattice of step {}", piece.start, piece.step);
                assert_eq!((TABLE_STEP_CM / piece.step).log2().fract(), 0.0);
                if k > 0 {
                    assert_eq!(piece.start, t.pieces[k - 1].end());
                }
            }
        }
    }

    #[test]
    fn the_likelihood_of_one_packet_has_unit_mass() {
        // a single packet's likelihood is its density over the measured distance, so it integrates to one over
        // the predicted distance exactly when the excess, the offset and the core are normalized; the former
        // excess quadrature summed to 1.073
        let calibration = crate::CalibrationModel::default();
        for shape in [OffsetShape::Gaussian, OffsetShape::StudentT] {
            for (pi, t) in [(1.0, 2.5), (0.31, 2.5), (1.0, 0.0), (0.5, 0.01), (0.95, 21.6)] {
                let mut m = model(0.28, pi, 8.0, t);
                m.offset_shape = shape;
                let y = 640.0;
                let total = mass(&table(&[y], &[calibration.scale_cm(y)], &m));
                assert!((total - 1.0).abs() < 2e-3, "{shape:?} pi {pi} t {t}: mass {total}");
            }
        }
    }

    /// The log likelihood where every packet is an early detection: sum of `ln(lambda) - lambda y` over the packets.
    fn early_floor(y: &[f64], m: &BlockModel) -> f64 {
        y.iter().map(|v| m.early_detection.ln() - m.early_detection * v).sum()
    }

    #[test]
    fn has_no_plateau_and_decays_beyond_the_table() {
        // without early detections the likelihood decays strictly on both sides; with them it levels off beyond
        // the packets at the floor where every packet is an early detection, a property of the model
        for (y, s, m) in variety() {
            let t = table(&y, &s, &m);
            for piece in &t.pieces {
                assert!(piece.values.windows(3).all(|w| !(w[0] == w[1] && w[1] == w[2])), "flat run in the table");
            }
            let (lo, hi) = t.range();
            let mut previous = (t.eval(hi), t.eval(lo));
            let mut offset = 0.5;
            while offset < 1e7 {
                let (right, left) = (t.eval(hi + offset), t.eval(lo - offset));
                let (_, right_slope) = t.eval_smooth(hi + offset);
                let (_, left_slope) = t.eval_smooth(lo - offset);
                assert!(right.is_finite() && left.is_finite() && right_slope.is_finite());
                assert!(left < previous.1 && left_slope > 0.0, "left side not strictly decaying {offset} cm beyond the table");
                if m.early_detection > 0.0 {
                    assert!(right <= previous.0 + 1e-12 && right_slope <= 0.0, "right side rises {offset} cm beyond the table");
                    assert!(right >= early_floor(&y, &m) - 1e-9, "below the floor {offset} cm beyond the table");
                } else {
                    assert!(right < previous.0 && right_slope < 0.0, "right side not strictly decaying {offset} cm beyond the table");
                }
                previous = (right, left);
                offset *= 1.7;
            }
        }
    }

    /// The blocks of the continuation scans: every model of `variety` at 1, 4 and 10 packets, the ten readings of
    /// `variety` moved to the model's first reading with the model's own scales, and every block of the parity
    /// fixture, which holds every tracked fit at 1, 4 and 10 packets.
    pub(crate) fn scanned_blocks() -> Vec<(Vec<f64>, Vec<f64>, BlockModel)> {
        let calibration = crate::CalibrationModel::default();
        let ten = [812.0, 815.5, 809.0, 811.2, 845.0, 813.3, 810.1, 808.7, 816.4, 812.9];
        let mut blocks = Vec::new();
        for (y, s, m) in variety() {
            let bern = y.iter().zip(&s).all(|(v, scale)| *scale == calibration.scale_cm(*v));
            for n in [1, 4, 10] {
                let readings: Vec<f64> = ten[..n].iter().map(|v| v - ten[0] + y[0]).collect();
                let scales = (0..n).map(|k| if bern { calibration.scale_cm(readings[k]) } else { s[k % s.len()] }).collect();
                blocks.push((readings, scales, m));
            }
        }
        for case in fixture_cases() {
            blocks.push((floats(&case["y"]), floats(&case["s"]), model_from_json(&case["params"])));
        }
        blocks
    }

    /// Beyond the table the likelihood decays. Every 0.01 cm out to ten metres past each end, and then in steps growing
    /// by one percent out to 100 km, the continuation never rises away from the table and never exceeds the table's
    /// peak. With early detections the upper side levels off at the floor of early readings, whose term climbs to the
    /// floor as the left tail of the shift clears the last reading, and the continuation may climb as far as that term
    /// still climbs from the table's end, a trace for a Student t offset and none for a Gaussian one. Rounding may
    /// leave 1e-12 of the value. The mode terms once took three Newton steps whether or not they reached
    /// a mode, and past the table's upper end a ten packet Bern block rose to 145,000 nats above its peak. The junction
    /// of a factor falling onto its floor dipped below the floor and rose back, and the tail term below every reading
    /// jumped where the survival of the packets bends at zero distance. A table with early detections reaches down to
    /// zero distance, below which no solver looks, and there a high rate over a long link under a Student t offset
    /// lets the likelihood still rise towards the detection distances below zero that no packet has to survive.
    #[test]
    fn the_continuation_decays_away_from_the_table() {
        let blocks = scanned_blocks();
        let scan = |(y, s, m): &(Vec<f64>, Vec<f64>, BlockModel)| {
            let t = table(y, s, m);
            let (lo, hi) = t.range();
            let peak = t.pieces.iter().flat_map(|p| p.values.iter().copied()).fold(f64::NEG_INFINITY, f64::max);
            let climb: f64 =
                t.tails.iter().filter_map(|tail| tail.factor.floor().map(|(_, y_max)| -tail.factor.log_shift_survival(y_max - hi))).sum();
            let ends: &[(f64, f64, f64)] = if m.early_detection > 0.0 { &[(hi, 1.0, climb)] } else { &[(hi, 1.0, climb), (lo, -1.0, 0.0)] };
            for &(end, direction, allowed) in ends {
                let mut lowest = t.eval(end);
                let (mut k, mut offset) = (1, 0.01);
                while offset < 1e7 {
                    let value = t.eval(end + direction * offset);
                    assert!(value.is_finite() && value <= peak, "{value} above the peak {peak} {offset} cm past {end} (n={}, {m:?})", y.len());
                    assert!(
                        value <= lowest + allowed + 1e-12 * (1.0 + value.abs()),
                        "{value} rises from {lowest} {offset} cm past {end} (n={}, {m:?})",
                        y.len()
                    );
                    lowest = lowest.min(value);
                    if k < 100_000 {
                        k += 1;
                        offset = 0.01 * k as f64;
                    } else {
                        offset *= 1.01;
                    }
                }
            }
        };
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::thread::scope(|scope| {
            for chunk in blocks.chunks(blocks.len().div_ceil(workers)) {
                scope.spawn(move || chunk.iter().for_each(scan));
            }
        });
    }

    /// Nearly Gaussian packets (nu = 1e6) under the tracked fits of Bern, TIERS, IDLab and Bregar, both offset shapes,
    /// with the fitted excess, without it and without the offset, one packet at 4 and at 15 m and four at 8 m. Every
    /// 5 cm out to 800 m past both ends of the table, above zero distance, the continuation decays away from the table
    /// with a slope that points towards it. The tail term's mode search once reached the core's mode 67 m past a table
    /// and rose 11,460 nats above the continuation on either side, and the derivative of the mode carried along its
    /// steps gave slopes of the wrong sign hundreds of metres away.
    #[test]
    fn the_far_field_of_nearly_gaussian_packets_decays() {
        let calibration = crate::CalibrationModel::default();
        let fits: [(f64, f64, f64, f64, Option<f64>); 4] = [
            (0.28503058644924556, 0.5004221957061215, 5.330540859822179, 2.0240787649328875, None),
            (0.1862761985520489, 0.95, 3.8316288315373983, 3.6944265511667607, Some(11.168277015691466)),
            (0.02972068930104647, 0.95, 11.878700307738802, 15.65898799688768, Some(80.26037949746505)),
            (0.037575433856812744, 0.95, 13.297703384723643, 5.929785862239445, Some(71.0)),
        ];
        let mut blocks = Vec::new();
        for (kappa, pi, mean, tau, scale) in fits {
            for shape in [OffsetShape::Gaussian, OffsetShape::StudentT] {
                for (pi, mean, tau) in [(pi, mean, tau), (0.0, 1.0, tau), (pi, mean, 0.0)] {
                    let mut m = model(kappa, pi, mean, tau);
                    m.nu = 1e6;
                    m.offset_shape = shape;
                    for y in [vec![400.0], vec![1500.0], vec![800.0, 801.5, 799.0, 800.7]] {
                        let s = y.iter().map(|v| scale.unwrap_or_else(|| calibration.scale_cm(*v))).collect();
                        blocks.push((y, s, m));
                    }
                }
            }
        }
        let scan = |(y, s, m): &(Vec<f64>, Vec<f64>, BlockModel)| {
            let t = table(y, s, m);
            let (lo, hi) = t.range();
            for (end, direction) in [(hi, 1.0), (lo, -1.0)] {
                let mut lowest = t.eval(end);
                let mut offset = 5.0;
                while offset <= 80_000.0 && end + direction * offset >= 0.0 {
                    let (value, slope) = t.eval_smooth(end + direction * offset);
                    assert!(value.is_finite() && value <= lowest + 1e-12 * (1.0 + value.abs()), "{value} rises from {lowest} {offset} cm past {end} ({y:?}, {m:?})");
                    assert!(direction * slope < 0.0, "slope {slope} {offset} cm past {end} ({y:?}, {m:?})");
                    lowest = lowest.min(value);
                    offset += 5.0;
                }
            }
        };
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::thread::scope(|scope| {
            for chunk in blocks.chunks(blocks.len().div_ceil(workers)) {
                scope.spawn(move || chunk.iter().for_each(scan));
            }
        });
    }

    /// The anchor-only grid evaluates every table at the slant distance of every cell and height node, most of them
    /// beyond the table, through the continuation tabulated between the grid's nearest and farthest distances. Every
    /// 0.37 cm out to 20 m past both ends of every scanned block the tabulation meets the continuation within a
    /// hundred times its tolerance of the value, which holds the error below 1e-7 nats where the likelihood is within
    /// a thousand nats of zero.
    #[test]
    fn the_tabulated_continuation_meets_the_continuation() {
        let blocks = scanned_blocks();
        let check = |(y, s, m): &(Vec<f64>, Vec<f64>, BlockModel)| {
            let t = table(y, s, m);
            let (lo, hi) = t.range();
            let (nearest, farthest) = ((lo - 2000.0).max(0.0), hi + 2000.0);
            let far = t.far(nearest, farthest);
            let mut worst = 0.0_f64;
            let mut d = nearest;
            while d <= farthest {
                let (tabulated, exact) = (far.eval(d), t.eval(d));
                worst = worst.max((tabulated - exact).abs() / (1.0 + exact.abs()));
                d += 0.37;
            }
            worst
        };
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let worst = std::thread::scope(|scope| {
            let handles: Vec<_> = blocks
                .chunks(blocks.len().div_ceil(workers))
                .map(|chunk| scope.spawn(move || chunk.iter().map(check).fold(0.0_f64, f64::max)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).fold(0.0_f64, f64::max)
        });
        eprintln!("largest relative miss of the tabulated continuation {worst:.3e}");
        assert!(worst <= 100.0 * FAR_TOLERANCE, "{worst}");
    }

    /// The next larger and smaller floating point numbers of a positive finite distance.
    fn up(d: f64) -> f64 {
        if d == 0.0 { f64::from_bits(1) } else { f64::from_bits(d.to_bits() + 1) }
    }

    fn down(d: f64) -> f64 {
        if d == 0.0 { 0.0 } else { f64::from_bits(d.to_bits() - 1) }
    }

    /// The certified bounds of every scanned block's tabulated table hold `eval` at every distance of an interval:
    /// random intervals from a thousandth of a centimetre to the whole span, each checked at 65 evenly spaced
    /// distances, at every knot and node inside it and at the floating point neighbours of those, and intervals at the
    /// ends of the table and of the nodes. The spans reach from the grid's nearest distance below the table to its
    /// farthest beyond it, lie inside the table, or start at zero distance. Every bound is finite, since no cubic of
    /// the tabulation fails, and an interval of one distance is bounded within 1e-9 of the value.
    #[test]
    fn far_bounds_hold_every_value_of_their_interval() {
        let blocks = scanned_blocks();
        let check = |(y, s, m): &(Vec<f64>, Vec<f64>, BlockModel)| {
            let t = table(y, s, m);
            let (lo, hi) = t.range();
            let mut state = y.iter().fold(0x9e37_79b9_7f4a_7c15_u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3));
            let mut uniform = move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1_u64 << 53) as f64
            };
            for (nearest, farthest) in [((lo - 2000.0).max(0.0), hi + 2000.0), (lo + 0.3 * (hi - lo), hi - 0.2 * (hi - lo)), (0.0, hi + 300.0)] {
                let far = t.far(nearest, farthest);
                let bounds = far.bounds();
                let knots: Vec<f64> = t
                    .pieces
                    .iter()
                    .flat_map(|p| (0..p.values.len()).map(move |i| p.start + i as f64 * p.step))
                    .chain(far.below.at.iter().copied())
                    .chain(far.above.at.iter().copied())
                    .collect();
                let check_interval = |a: f64, b: f64| {
                    let (low, high) = bounds.over(&far, a, b);
                    assert!(low.is_finite() && high.is_finite(), "[{a}, {b}] of [{nearest}, {farthest}], table [{lo}, {hi}]: {low} {high}");
                    let mut points: Vec<f64> = (0..=64).map(|k| a + (b - a) * k as f64 / 64.0).collect();
                    let from = knots.partition_point(|d| *d < a);
                    for &knot in knots[from..].iter().take_while(|d| **d <= b) {
                        points.extend([knot, up(knot), down(knot)]);
                    }
                    for d in points.into_iter().filter(|d| *d >= a && *d <= b) {
                        let value = far.eval(d);
                        assert!(low <= value && value <= high, "{value} at {d} outside [{low}, {high}] over [{a}, {b}], table [{lo}, {hi}], nodes [{nearest}, {farthest}]");
                    }
                    (low, high)
                };
                let span = farthest - nearest;
                for trial in 0..400 {
                    let width = match trial % 4 {
                        0 => 1e-3,
                        1 => 0.5 * uniform(),
                        2 => 20.0 * uniform(),
                        _ => span * uniform(),
                    }
                    .min(span);
                    let a = nearest + (span - width) * uniform();
                    check_interval(a, (a + width).min(farthest));
                }
                for (a, b) in [(nearest, nearest), (nearest, nearest + 1.0), (lo - 1.0, lo + 1.0), (hi - 1.0, hi + 1.0), (farthest - 1.0, farthest), (farthest, farthest), (nearest, farthest)] {
                    let (a, b) = (a.clamp(nearest, farthest), b.clamp(nearest, farthest));
                    if a <= b {
                        check_interval(a, b);
                    }
                }
                for d in knots.iter().copied().filter(|d| *d >= nearest && *d <= farthest).step_by(7) {
                    let (low, high) = check_interval(d, d);
                    let value = far.eval(d);
                    assert!(high - low <= 1e-9 * (1.0 + value.abs()), "{low} {high} at {d} for {value}");
                }
            }
        };
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        std::thread::scope(|scope| {
            for chunk in blocks.chunks(blocks.len().div_ceil(workers)) {
                scope.spawn(move || chunk.iter().for_each(check));
            }
        });
    }

    /// Distances outside the tabulated nodes, where `eval` asks the continuation itself, and a cubic that is not finite
    /// leave the bounds open, and an empty or inverted interval does too.
    #[test]
    fn far_bounds_are_open_where_the_continuation_answers() {
        let calibration = crate::CalibrationModel::default();
        let t = table(&[812.0], &[calibration.scale_cm(812.0)], &model(0.28, 0.31, 8.0, 2.5));
        let (lo, hi) = t.range();
        let far = t.far(lo - 500.0, hi + 500.0);
        let bounds = far.bounds();
        let open = (f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(bounds.over(&far, lo - 501.0, lo), open);
        assert_eq!(bounds.over(&far, hi, hi + 501.0), open);
        assert_eq!(bounds.over(&far, hi, f64::NAN), open);
        assert_eq!(bounds.over(&far, hi, lo), open);
        // at the first node below the table itself `eval` asks the continuation, a point value
        let at = far.below.at[0];
        let (low, high) = bounds.over(&far, at, at);
        assert!(low <= far.eval(at) && far.eval(at) <= high && high - low < 1e-9 * (1.0 + far.eval(at).abs()));
        // without nodes on a side every distance there is the continuation's
        let inside = t.far(lo, hi);
        assert_eq!(inside.bounds().over(&inside, lo - 1.0, lo + 1.0), open);
        assert_eq!(inside.bounds().over(&inside, hi - 1.0, hi + 1.0), open);
        let broken = Cubic { start: 0.0, inverse: 1.0, c: [0.0, f64::NAN, 0.0, 0.0] };
        assert_eq!(broken.extremes(0.0, 1.0), open);
        assert_eq!(Cubic { start: 0.0, inverse: 1.0, c: [1e308, 1e308, 0.0, 0.0] }.extremes(0.0, 1.0), open);
    }

    /// The extremes of a cubic over a part of its interval come from its ends and the roots of its derivative, a
    /// double root included, and hold every value in between.
    #[test]
    fn cubic_extremes_hold_the_cubic() {
        let cubics = [
            [0.0, 1.0, -3.0, 2.0],
            [5.0, -2.0, 6.0, -4.0],
            [1.0, 0.0, 0.0, 0.0],
            [-3.0, 0.3, -0.9, 0.0],
            [2.0, 3.0, -3.0, 1.0],
            [-700.0, 12.0, -30.0, 18.0],
            [1e-3, -1e-9, 1e-9, -1e-9],
        ];
        for c in cubics {
            let cubic = Cubic { start: 10.0, inverse: 0.125, c };
            for (t0, t1) in [(0.0, 1.0), (0.0, 0.25), (0.3, 0.9), (0.5, 0.5), (0.999, 1.0)] {
                let (low, high) = cubic.extremes(t0, t1);
                for k in 0..=10_000 {
                    let t = t0 + (t1 - t0) * k as f64 / 10_000.0;
                    let value = cubic.value(t);
                    assert!(low <= value && value <= high, "{c:?} over [{t0}, {t1}]: {value} at {t} outside [{low}, {high}]");
                }
                let sampled = (0..=10_000).map(|k| cubic.value(t0 + (t1 - t0) * k as f64 / 10_000.0)).fold(f64::NEG_INFINITY, f64::max);
                assert!(high - sampled < 1e-6, "{c:?} over [{t0}, {t1}]: {high} against {sampled}");
            }
        }
    }

    #[test]
    fn early_detections_level_off_at_their_floor() {
        // far beyond every reading each packet is explained as an early detection: the continuation reaches the
        // model's floor exactly, with a derivative that vanishes smoothly
        let calibration = crate::CalibrationModel::default();
        for rate in [1.1e-4, 1e-3] {
            for (y, independent) in [(vec![640.0], false), (vec![640.0, 643.5, 637.0, 652.0], false), (vec![640.0, 643.5, 637.0, 652.0], true)] {
                let s: Vec<f64> = y.iter().map(|v| calibration.scale_cm(*v)).collect();
                let mut m = model(0.28, 0.31, 8.0, 2.5);
                m.early_detection = rate;
                m.independent = independent;
                let t = table(&y, &s, &m);
                let floor = early_floor(&y, &m);
                for d in [2e4, 1e5, 1e7] {
                    let (value, slope) = t.eval_smooth(d);
                    assert!((value - floor).abs() < 1e-6, "rate {rate} at {d}: {value} against the floor {floor}");
                    assert!(slope.is_finite() && slope <= 0.0 && slope > -1e-6);
                    assert_eq!(t.eval(d), value);
                }
            }
        }
    }

    #[test]
    fn derivatives_match_finite_differences_everywhere() {
        for (y, s, m) in scanned_blocks() {
            let t = table(&y, &s, &m);
            let (lo, hi) = t.range();
            let step = finest(&t);
            let mut points: Vec<f64> = (0..400).map(|k| lo + (hi - lo) * (k as f64 + 0.37) / 400.0).collect();
            for anchor in t.pieces.iter().map(|p| p.start).chain([hi]) {
                for offset in [-3.0 * step, -0.5 * step, -1e-3, 1e-3, 0.5 * step, 3.0 * step] {
                    points.push(anchor + offset);
                }
            }
            // beyond the table, through the weights of the asymptotic terms and their hand over to mode terms, and
            // across the bend of the packets' survival at zero distance
            let mut offset = 1.0;
            while offset < 1e5 {
                points.push(lo - offset);
                points.push(hi + offset);
                offset *= 1.3;
            }
            if lo > 0.0 {
                points.extend([-3.0, -1.0, -0.3, -0.1, -0.03, -0.01, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]);
            }
            for d in points {
                let (value, slope) = t.eval_smooth(d);
                let e = 1e-4 * step.min(1.0);
                let numeric = (t.eval_smooth(d + e).0 - t.eval_smooth(d - e).0) / (2.0 * e);
                assert!(value.is_finite());
                // central differences across a knot of the C1 interpolant differ by e times the jump of the second
                // derivative
                assert!((numeric - slope).abs() < 2e-4 * (1.0 + slope.abs()), "d {d}: slope {slope} numeric {numeric}");
            }
        }
    }

    #[test]
    fn the_table_is_continuous_at_every_junction() {
        for (y, s, m) in variety() {
            let t = table(&y, &s, &m);
            let (lo, hi) = t.range();
            for anchor in t.pieces.iter().map(|p| p.start).chain([hi]) {
                let (a, b) = (anchor - 1e-9, anchor + 1e-9);
                assert!((t.eval(a) - t.eval(b)).abs() < 1e-6, "value jump at {anchor} (range {lo} {hi})");
                let (u, v) = (t.eval_smooth(a), t.eval_smooth(b));
                assert!((u.0 - v.0).abs() < 1e-6 && (u.1 - v.1).abs() < 1e-5 * (1.0 + u.1.abs()), "{u:?} {v:?} at {anchor}");
            }
        }
    }

    #[test]
    fn the_log_domain_sums_leave_out_only_what_cannot_count() {
        // two nearly Gaussian packets far apart put most of their table in the log domain, once summed over the whole
        // lattice for every value: 50 m apart took a second to tabulate, 200 m nine and 900 m minutes
        let calibration = crate::CalibrationModel::default();
        let mut m = model(0.28, 0.31, 8.0, 2.5);
        m.nu = 1e6;
        for apart in [5000.0, 20000.0] {
            let y = [812.0, 812.0 + apart];
            let s: Vec<f64> = y.iter().map(|v| calibration.scale_cm(*v)).collect();
            let started = std::time::Instant::now();
            let t = table(&y, &s, &m);
            assert!(started.elapsed().as_secs_f64() < 5.0, "{apart} cm apart took {:?}", started.elapsed());
            assert!(t.pieces.iter().all(|p| p.values.iter().all(|v| v.is_finite())));
        }
        // every value of a lattice in the log domain, against the sum over the whole lattice in its order, and on a
        // lattice of 192,000 points, which the groups of blocks cover, every 97th value
        for (apart, delta, every) in [(5000.0, 2.0, 1), (20000.0, 2.0, 1), (90000.0, 0.5, 97)] {
            let y = [812.0, 812.0 + apart];
            let s: Vec<f64> = y.iter().map(|v| calibration.scale_cm(*v)).collect();
            let f = Factor::new(&y, &s, &m, log_t_norm_for(m.nu));
            let (lo, r) = (y[0] - 3000.0, 16);
            let nf = ((y[1] + 3000.0 - lo) / delta) as usize;
            let log_g: Vec<f64> = (0..nf).map(|k| f.log_g(lo + k as f64 * delta)).collect();
            let mut values = vec![f64::NAN; (nf - 1) / r + 1];
            f.log_domain_values(&mut values, &log_g, delta, 0, r);
            let weights: Vec<f64> = (0..2 * nf - 1).map(|index| f.log_kernel_weight((index as f64 - (nf - 1) as f64) * delta, delta)).collect();
            for (i, value) in values.iter().enumerate().step_by(every) {
                let (mut best, mut acc) = (f64::NEG_INFINITY, 0.0);
                for (k, lg) in log_g.iter().enumerate() {
                    let term = lg + weights[k + nf - 1 - r * i];
                    if term > best {
                        acc = acc * (best - term).exp() + 1.0;
                        best = term;
                    } else if term > f64::NEG_INFINITY {
                        acc += (term - best).exp();
                    }
                }
                let whole = best + acc.ln();
                assert!((value - whole).abs() <= 1e-13 * whole.abs(), "{apart} cm apart at {i}: {value} {whole}");
            }
        }
    }

    #[test]
    fn grid_points_depend_on_the_lattice_only() {
        // two blocks whose ranges differ share the points of their common lattice: a grid point is a multiple of
        // the step wherever the range starts
        let m = model(0.28, 0.31, 8.0, 2.5);
        let a = table(&[812.0, 815.0], &[25.0, 25.0], &m);
        let b = table(&[812.0, 1015.0], &[25.0, 25.0], &m);
        let (pa, pb) = (&a.pieces[0], &b.pieces[0]);
        assert_eq!(pa.step, pb.step);
        assert_eq!(((pa.start - pb.start) / pa.step).fract(), 0.0);
    }

    /// Construction time of the tables of a Bern unit, a hundred links of ten packets or of one, printed by
    /// `cargo test --release link::tests::construction_time -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn construction_time() {
        let calibration = crate::CalibrationModel::default();
        let mut state = 1_u64;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut unit = |links: usize, packets: usize, lo: f64, hi: f64| -> Vec<(Vec<f64>, Vec<f64>)> {
            (0..links)
                .map(|_| {
                    let d = lo + (hi - lo) * uniform();
                    let y: Vec<f64> = (0..packets)
                        .map(|_| d + 8.0 * (uniform() - 0.5) + if uniform() < 0.2 { 60.0 * uniform() } else { 0.0 })
                        .collect();
                    let s = y.iter().map(|v| calibration.scale_cm(*v)).collect();
                    (y, s)
                })
                .collect()
        };
        let bern10 = unit(100, 10, 150.0, 1500.0);
        let bern1 = unit(100, 1, 150.0, 1500.0);
        let cases = [
            ("bern nu3 fit, 100 links x 10", &bern10, model(0.285, 0.5004, 5.33, 2.024)),
            ("bern corrected, 100 links x 10", &bern10, model(0.28, 0.31, 8.0, 2.5)),
            ("bern corrected, 100 links x 1", &bern1, model(0.28, 0.31, 8.0, 2.5)),
        ];
        for (label, blocks, m) in cases {
            for (shape, rate) in [(OffsetShape::Gaussian, 0.0), (OffsetShape::Gaussian, 1.1e-4), (OffsetShape::StudentT, 0.0), (OffsetShape::StudentT, 1.1e-4)] {
                let mut m = m;
                m.offset_shape = shape;
                m.early_detection = rate;
                let started = std::time::Instant::now();
                let mut total = 0;
                let reps = 3;
                for _ in 0..reps {
                    for (y, s) in blocks.iter() {
                        total += points(&table(y, s, &m));
                    }
                }
                let ms = 1e3 * started.elapsed().as_secs_f64() / reps as f64;
                println!("NEW {label} {shape:?} early {rate}: {ms:.1} ms per unit, {} points per link", total / (reps * blocks.len()));
            }
        }
    }
}
