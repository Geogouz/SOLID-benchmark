//! The adaptive exact posterior of the anchor only estimator.
//!
//! One evaluator holds the per cell arithmetic of an agent's log posterior, and every path that needs the value of a
//! cell takes it from there, so a kept cell carries bitwise the value the uniform grid gives it. A quadtree over the
//! cell indices refines the posterior level by level, from the whole grid down to single cells. Every node holds
//! certified bounds of the log posterior of each of its cells: per height node, the bounds of every link's table over
//! the slant distances its cells can take and those of the monotone presence terms, summed in the order of a cell and
//! combined over the height nodes by the log sum of a cell. A node whose upper bound lies more than
//! delta = ln(N / EPSILON) below the best lower bound or exact value seen, N the grid's cells, drops with its bound
//! mass. A node whose bounds lie within FLAT_GAP of each other stays whole as a coarse leaf, and every other node
//! splits. The mass left out is certified at most EPSILON of the posterior's, the kept cells hold every cell within
//! delta of the best and so the smallest leading set with 1 - EPSILON of the mass, and no cap limits the kept cells.
//! The cost follows how ambiguous an agent's position is rather than the size of the area.

use std::collections::BTreeMap;

use crate::link::{FarBounds, FarTable, LinkTable, BOUND_MARGIN};
use crate::presence::{PresenceShape, ResolvedPresence};
use crate::{EngineError, PlanarArea, Point2};

/// The largest share of an agent's posterior mass its kept cells may leave out, certified. The kept cells hold every
/// cell whose log posterior lies within ln(N / EPSILON) of the best, N the grid's cells, the mass of all other cells
/// together being at most EPSILON of the best cell's, and with them the smallest leading set with 1 - EPSILON of the
/// mass.
pub(crate) const EPSILON: f64 = 1e-4;
/// Nats between the upper and the lower bound of a node within which it stays whole, a coarse leaf. A draw enters a
/// coarse leaf with the weight of its upper bound and accepts a cell of it with the probability of the cell's value
/// below that bound, so at least e^-FLAT_GAP of the draws that reach it.
pub(crate) const FLAT_GAP: f64 = 1.0;
/// The most nodes one posterior holds at once, its kept leaves and the nodes of the next level, 2^23 of 32 bytes. Where
/// splitting would hold more, the nodes of the largest bound mass split and the others drop with their certified mass,
/// and the agent counts as guarded. Its peak memory is twice this many nodes, the level being refined and the next.
pub(crate) const GUARD_NODES: usize = 1 << 23;
/// The fewest nodes a guard may hold, the four children of one node. With fewer, a posterior that has kept no leaf yet
/// could split no node at all and would drop every one.
pub(crate) const MIN_GUARD_NODES: usize = 4;
const _: () = assert!(GUARD_NODES >= MIN_GUARD_NODES);
/// The most cell centres of a grid axis, 2^22, the bound `Grid::new` checks before it allocates anything.
pub(crate) const AXIS_GUARD: f64 = 4_194_304.0;
/// Reaches from which a failed attempt's Gaussian presence term is its clamp's constant, ln(1 - 1e-12) exactly. The
/// probability exp(-d^2 / (2 r^2)) falls below the clamp 1e-12 from sqrt(2 ln 1e12) = 7.434 reaches on, and 7.45 keeps a
/// margin of 0.2 percent, far above the rounding of a distance. The cutoff of the Gaussian law is d_cut = 7.45 r: a
/// failed attempt at d_cut or beyond from a cell adds that constant in its place in the cell's order, and one at d_cut or
/// beyond from every cell of a block adds the constant bounds its term takes there, without its distance's square root
/// and exponentials, so every value and bound is bitwise that of every term.
pub(crate) const GAUSSIAN_CUTOFF_REACHES: f64 = 7.45;
/// Widths of the logistic transition beyond the reach from which a failed attempt's logistic term, -softplus(z) with
/// z = (r - d) / (w r), is smaller than LOGISTIC_LEFT_OUT: z <= -27.64 gives e^z < 1e-12, and 27.7 keeps a margin. The
/// cutoff of the logistic law is d_cut = r (1 + 27.7 w), and the posterior leaves out the terms of the failed attempts at
/// d_cut or beyond, each negative and smaller than LOGISTIC_LEFT_OUT, so a cell's value exceeds that of every term by
/// less than their count times it, and the lower bound of a block drops by that count times it.
pub(crate) const LOGISTIC_CUTOFF_WIDTHS: f64 = 27.7;
/// The largest size of a failed attempt's logistic term the posterior leaves out, in nats.
pub(crate) const LOGISTIC_LEFT_OUT: f64 = 1e-12;

#[inline]
pub(crate) fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Presence log likelihood of the whole opportunity pattern of one agent at one point. `dists` holds the slant
/// distance to every ranging opportunity, `present` whether it produced ranges.
pub(crate) fn detection_loglik(dists: &[f64], present: &[bool], law: &ResolvedPresence) -> f64 {
    dists.iter().zip(present).map(|(d, p)| law.log_probability(*d, *p).0).sum()
}

/// A log sum of values that arrive one at a time, in the order they arrive.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LogSum {
    top: f64,
    scaled: f64,
}

impl Default for LogSum {
    fn default() -> Self {
        Self { top: f64::NEG_INFINITY, scaled: 0.0 }
    }
}

impl LogSum {
    /// Adds e^value, nothing for a value that is not a number.
    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        if value > self.top {
            self.scaled = self.scaled * (self.top - value).exp() + 1.0;
            self.top = value;
        } else if value > f64::NEG_INFINITY {
            self.scaled += (value - self.top).exp();
        }
    }

    pub(crate) fn value(&self) -> f64 {
        if self.top == f64::NEG_INFINITY || self.top == f64::INFINITY {
            self.top
        } else {
            self.top + self.scaled.ln()
        }
    }
}

/// The random stream of one agent.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut v = self.0;
        v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        v ^ (v >> 31)
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform draw from 0 to `count` - 1, exactly uniform: a draw from the last partial multiple of `count` below
    /// 2^64 is drawn again.
    pub(crate) fn below(&mut self, count: u64) -> u64 {
        let threshold = count.wrapping_neg() % count;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % count;
            }
        }
    }
}

/// The cell centres of a grid, `xs` along the first axis and `ys` along the second. Cell (i, j) lies at
/// (xs[i], ys[j]), and its index is j * xs.len() + i.
pub(crate) struct Grid {
    pub(crate) xs: Vec<f64>,
    pub(crate) ys: Vec<f64>,
    pub(crate) step: f64,
}

impl Grid {
    /// The centres over `area` at `step`, from half a step inside the lower bound by repeated additions of the step
    /// while below the upper bound, and the middle of an axis where no centre lies below it. An area beyond the
    /// supported coordinate magnitude or a step that is not a positive number is refused, and so is an axis whose
    /// count of centres, at most the axis over the step rounded up plus one, may exceed AXIS_GUARD, before anything is
    /// allocated. A step below half the spacing of the numbers at the area's coordinates leaves a centre where it was
    /// when added, so the axis would never reach its upper bound: the first addition that does not move a centre
    /// refuses the step.
    pub(crate) fn new(area: &PlanarArea, step: f64) -> Result<Self, EngineError> {
        let bounds = [area.minimum_x_cm, area.maximum_x_cm, area.minimum_y_cm, area.maximum_y_cm];
        if !bounds.iter().all(|v| crate::model::sane_coordinate(*v)) {
            return Err(EngineError::InvalidProfile("the operating area lies beyond the supported coordinate magnitude"));
        }
        if !(step.is_finite() && step > 0.0) {
            return Err(EngineError::InvalidProfile("the grid step must be a positive number"));
        }
        let h = step;
        // the centres of an axis before anything is allocated, one per step at most
        let per_axis = |low: f64, high: f64| ((high - low) / h).ceil().max(0.0) + 1.0;
        if !(per_axis(area.minimum_x_cm, area.maximum_x_cm) <= AXIS_GUARD && per_axis(area.minimum_y_cm, area.maximum_y_cm) <= AXIS_GUARD) {
            return Err(EngineError::InvalidProfile("the grid step is too fine for the operating area"));
        }
        let axis = |low: f64, high: f64| {
            let mut centres = Vec::new();
            let mut v = low + h / 2.0;
            while v < high {
                centres.push(v);
                let next = v + h;
                if next == v {
                    return Err(EngineError::InvalidProfile("the grid step does not move the cell centres of the operating area"));
                }
                v = next;
            }
            if centres.is_empty() {
                centres.push((low + high) / 2.0);
            }
            Ok(centres)
        };
        let xs = axis(area.minimum_x_cm, area.maximum_x_cm)?;
        let ys = axis(area.minimum_y_cm, area.maximum_y_cm)?;
        Ok(Self { xs, ys, step: h })
    }

    pub(crate) fn count(&self) -> usize {
        self.xs.len() * self.ys.len()
    }

    /// The nearest and the farthest slant distance from `anchor` to a cell centre at a height node, a centimetre
    /// wider on either side.
    pub(crate) fn slant_span(&self, anchor: &[f64; 3], heights: &[f64]) -> (f64, f64) {
        let (x_low, x_high) = (self.xs[0], self.xs[self.xs.len() - 1]);
        let (y_low, y_high) = (self.ys[0], self.ys[self.ys.len() - 1]);
        let near_x = anchor[0].clamp(x_low, x_high) - anchor[0];
        let near_y = anchor[1].clamp(y_low, y_high) - anchor[1];
        let far_x = (x_low - anchor[0]).abs().max((x_high - anchor[0]).abs());
        let far_y = (y_low - anchor[1]).abs().max((y_high - anchor[1]).abs());
        let dz: Vec<f64> = heights.iter().map(|z| (z - anchor[2]).abs()).collect();
        let near_z = dz.iter().copied().fold(f64::INFINITY, f64::min);
        let far_z = dz.iter().copied().fold(0.0, f64::max);
        let nearest = (near_x * near_x + near_y * near_y + near_z * near_z).sqrt();
        let farthest = (far_x * far_x + far_y * far_y + far_z * far_z).sqrt();
        ((nearest - 1.0).max(0.0), farthest + 1.0)
    }
}

/// Runs `f` on `len` zeros, on the stack when they are few.
#[inline]
fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    if len <= 32 {
        let mut scratch = [0.0; 32];
        f(&mut scratch[..len])
    } else {
        f(&mut vec![0.0; len])
    }
}

/// The smallest and the largest |x - a| over the centres x from `low` to `high` of an axis, as `x - a` computes them.
/// Rounding is monotone and symmetric, so the centres at the two ends bound every centre between, and none lies
/// nearer than zero where `a` lies among them.
#[inline]
fn axis_range(low: f64, high: f64, a: f64) -> (f64, f64) {
    let (at_low, at_high) = ((low - a).abs(), (high - a).abs());
    let near = if a < low {
        at_low
    } else if a > high {
        at_high
    } else {
        0.0
    };
    (near, at_low.max(at_high))
}

/// One range link of an agent: its anchor, its table tabulated over the grid's slant distances to the anchor with the
/// bounds of that tabulation, and the squared height difference to the anchor at every height node.
struct Link<'a> {
    anchor: [f64; 3],
    far: FarTable<'a>,
    bounds: FarBounds,
    dz2: Vec<f64>,
}

/// The failed attempts of an agent some of which lie at the law's cutoff or beyond from some cell of the grid, in
/// buckets of the cutoff's side over their horizontal places, so that a cell or a block finds those within reach without
/// visiting the others (GAUSSIAN_CUTOFF_REACHES, LOGISTIC_CUTOFF_WIDTHS).
struct FailuresWithinReach {
    /// The index of the first failed attempt among the evaluator's anchors, the present ones coming first.
    first: usize,
    /// The number of failed attempts.
    count: usize,
    /// The cutoff d_cut in slant distance, and the radius of a search, a centimetre and a billionth beyond it, so that
    /// every anchor within the cutoff of a place lies in the buckets the search visits.
    cut: f64,
    radius: f64,
    /// What a failed attempt beyond the cutoff adds.
    beyond: Beyond,
    /// The failed attempts of every bucket, (floor((x - origin_x) / cut), floor((y - origin_y) / cut)), as ascending
    /// indices among the evaluator's anchors.
    origin: [f64; 2],
    buckets: BTreeMap<(i64, i64), Vec<usize>>,
    /// The smallest and the largest bucket along each axis.
    span: [(i64, i64); 2],
}

/// What a failed attempt beyond the law's cutoff adds to a cell's sum and to a block's bounds.
#[derive(Clone, Copy)]
enum Beyond {
    /// Under the Gaussian law its term, the clamp's constant, in its place in the cell's order, and the bounds of that
    /// term, `ResolvedPresence::bounds` at any distances beyond the cutoff, in its place in the block's order, so that
    /// every sum is bitwise that of every term.
    Constant { term: f64, bounds: (f64, f64) },
    /// Under the logistic law nothing, its term negative and smaller than LOGISTIC_LEFT_OUT, which the lower bound of a
    /// block takes.
    LeftOut,
}

impl FailuresWithinReach {
    /// The failed attempts `anchors[first..]` under `law`, None where every one lies within the cutoff of every cell of
    /// `grid` at every height node, a centimetre wider, so that every term is evaluated as ever.
    fn new(grid: &Grid, anchors: &[[f64; 3]], first: usize, heights: &[f64], law: &ResolvedPresence) -> Option<Self> {
        let (cut, beyond) = match law.shape {
            PresenceShape::Gaussian => {
                let cut = GAUSSIAN_CUTOFF_REACHES * law.reach;
                (cut, Beyond::Constant { term: (-1e-12_f64).ln_1p(), bounds: law.bounds(cut, cut, false) })
            }
            PresenceShape::Logistic => (law.reach * (1.0 + LOGISTIC_CUTOFF_WIDTHS * law.width), Beyond::LeftOut),
        };
        let failed = &anchors[first..];
        if !(cut.is_finite() && cut > 0.0) || !failed.iter().any(|a| grid.slant_span(a, heights).1 >= cut) {
            return None;
        }
        let origin = [failed.iter().map(|a| a[0]).fold(f64::INFINITY, f64::min), failed.iter().map(|a| a[1]).fold(f64::INFINITY, f64::min)];
        let mut buckets: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
        let mut span = [(i64::MAX, i64::MIN); 2];
        for (k, a) in failed.iter().enumerate() {
            let key = (Self::bucket(a[0], origin[0], cut), Self::bucket(a[1], origin[1], cut));
            span[0] = (span[0].0.min(key.0), span[0].1.max(key.0));
            span[1] = (span[1].0.min(key.1), span[1].1.max(key.1));
            buckets.entry(key).or_default().push(first + k);
        }
        Some(Self { first, count: failed.len(), cut, radius: cut * (1.0 + 1e-9) + 1.0, beyond, origin, buckets, span })
    }

    #[inline]
    fn bucket(v: f64, origin: f64, side: f64) -> i64 {
        ((v - origin) / side).floor() as i64
    }

    /// Into `out`, ascending, the failed attempts of every bucket that meets [x0 - radius, x1 + radius] x
    /// [y0 - radius, y1 + radius], which holds every failed attempt within the cutoff of a place of the box.
    fn around(&self, x0: f64, x1: f64, y0: f64, y1: f64, out: &mut Vec<usize>) {
        out.clear();
        let range = |low: f64, high: f64, axis: usize| {
            let (first, last) = self.span[axis];
            (Self::bucket(low - self.radius, self.origin[axis], self.cut).max(first), Self::bucket(high + self.radius, self.origin[axis], self.cut).min(last))
        };
        let ((bx0, bx1), (by0, by1)) = (range(x0, x1, 0), range(y0, y1, 1));
        if bx0 > bx1 || by0 > by1 {
            return;
        }
        let visited = (bx1 - bx0 + 1) as u128 * (by1 - by0 + 1) as u128;
        if visited <= self.buckets.len() as u128 {
            for bx in bx0..=bx1 {
                for by in by0..=by1 {
                    if let Some(list) = self.buckets.get(&(bx, by)) {
                        out.extend_from_slice(list);
                    }
                }
            }
        } else {
            for ((bx, by), list) in &self.buckets {
                if (bx0..=bx1).contains(bx) && (by0..=by1).contains(by) {
                    out.extend_from_slice(list);
                }
            }
        }
        out.sort_unstable();
    }
}

/// The log posterior of one agent at the cells of a grid, the arithmetic every value of a cell comes from.
///
/// Per height node an accumulator starts at 0.0 and adds each link's table at the slant distance of the cell and the
/// node, the links in their canonical order. With a presence law the node's sum then adds the sequential sum of the
/// presence terms of the present anchors and then the missing ones. The value is the log sum over the height nodes of
/// the node sums plus their log weights, folded from minus infinity. The bounds of a block of cells follow the same
/// order with bounds in place of values.
///
/// Presence terms within reach. Where some failed attempt lies at the law's cutoff or beyond from some cell of the grid,
/// a cell evaluates the terms of the successful attempts and of the failed ones within the cutoff only, found through
/// buckets over the failed attempts (FailuresWithinReach). Under the Gaussian law a failed attempt at the cutoff or
/// beyond adds the clamp's constant, which its term equals bitwise, in its place in the same order, so every value and
/// every bound is bitwise that of every term, without the square roots and exponentials of the far failed attempts.
/// Under the logistic law those terms, smaller than LOGISTIC_LEFT_OUT, are left out, and a cell with every failed
/// attempt within the cutoff takes bitwise the sum of every term. A grid within the cutoff of every failed attempt keeps
/// the arithmetic without the cutoff. Successful attempts are always evaluated.
pub(crate) struct CellEvaluator<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
    links: Vec<Link<'a>>,
    heights: Vec<f64>,
    weights: Vec<f64>,
    /// The anchors of every ranging opportunity, the present ones first and then the missing ones, and whether each
    /// produced ranges.
    anchors: Vec<[f64; 3]>,
    outcomes: Vec<bool>,
    law: Option<ResolvedPresence>,
    /// The failed attempts some of which lie beyond the law's cutoff from some cell, None where none does.
    failures: Option<FailuresWithinReach>,
}

impl<'a> CellEvaluator<'a> {
    /// The evaluator of an agent on `grid`: its range links `links` in canonical order, each an anchor and its table,
    /// which the evaluator tabulates between the grid's nearest and farthest slant distance to the anchor; the anchors
    /// of its successful attempts `present` and of its failed ones `missing`, each in canonical order; the height
    /// nodes `heights` with their normalized log weights `weights`; and the presence law `law`, None for no presence
    /// term.
    pub(crate) fn new(
        grid: &'a Grid,
        links: &[([f64; 3], &'a LinkTable)],
        present: &[[f64; 3]],
        missing: &[[f64; 3]],
        heights: &[f64],
        weights: &[f64],
        law: Option<ResolvedPresence>,
    ) -> Self {
        assert_eq!(heights.len(), weights.len(), "one log weight per height node");
        let links = links
            .iter()
            .map(|(anchor, table)| {
                let (nearest, farthest) = grid.slant_span(anchor, heights);
                let far = table.far(nearest, farthest);
                let bounds = far.bounds();
                let dz2 = heights.iter().map(|z| (z - anchor[2]) * (z - anchor[2])).collect();
                Link { anchor: *anchor, far, bounds, dz2 }
            })
            .collect();
        let anchors: Vec<[f64; 3]> = present.iter().chain(missing).copied().collect();
        let failures = law.as_ref().and_then(|law| FailuresWithinReach::new(grid, &anchors, present.len(), heights, law));
        Self {
            xs: &grid.xs,
            ys: &grid.ys,
            links,
            heights: heights.to_vec(),
            weights: weights.to_vec(),
            anchors,
            outcomes: present.iter().map(|_| true).chain(missing.iter().map(|_| false)).collect(),
            law,
            failures,
        }
    }

    /// The cells along each axis.
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.xs.len(), self.ys.len())
    }

    /// Whether some failed attempt lies at the law's cutoff or beyond from some cell of the grid, so that some cell's
    /// sum or some block's bounds may leave its term to the count of the failed attempts beyond reach.
    #[cfg(test)]
    pub(crate) fn beyond_reach(&self) -> bool {
        self.failures.is_some()
    }

    /// The summed range terms of the links at cell (i, j) and every height node, before any presence term and weight.
    pub(crate) fn range_terms(&self, i: usize, j: usize, terms: &mut [f64]) {
        let (x, y) = (self.xs[i], self.ys[j]);
        terms.fill(0.0);
        for link in &self.links {
            let dy = y - link.anchor[1];
            let dx = x - link.anchor[0];
            let horizontal = dx * dx + dy * dy;
            for (term, dz2) in terms.iter_mut().zip(&link.dz2) {
                *term += link.far.eval((horizontal + dz2).sqrt());
            }
        }
    }

    /// The slant distances from (x, y) at height `z` to the anchor of every opportunity.
    #[inline]
    fn opportunity_distances(&self, x: f64, y: f64, z: f64, dists: &mut [f64]) {
        for (d, a) in dists.iter_mut().zip(&self.anchors) {
            let dx = x - a[0];
            let dy = y - a[1];
            let dz = z - a[2];
            *d = (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }

    /// The log posterior of cell (i, j).
    pub(crate) fn value(&self, i: usize, j: usize) -> f64 {
        with_scratch(self.heights.len(), |terms| {
            self.range_terms(i, j, terms);
            if let Some(law) = &self.law {
                let (x, y) = (self.xs[i], self.ys[j]);
                match &self.failures {
                    None => with_scratch(self.anchors.len(), |dists| {
                        for (term, z) in terms.iter_mut().zip(&self.heights) {
                            self.opportunity_distances(x, y, *z, dists);
                            *term += detection_loglik(dists, &self.outcomes, law);
                        }
                    }),
                    Some(failures) => self.add_presence_within_reach(failures, law, x, y, terms),
                }
            }
            terms.iter().zip(&self.weights).map(|(t, w)| t + w).fold(f64::NEG_INFINITY, log_add_exp)
        })
    }

    /// Adds to every height node's sum the presence terms at (x, y) in the order of `detection_loglik`, summed as it sums:
    /// the successful attempts, and the failed ones within the law's cutoff by their terms. A failed attempt at the cutoff
    /// or beyond takes, under the Gaussian law, the clamp's constant in its place, which its term equals bitwise, and under
    /// the logistic law it is left out.
    fn add_presence_within_reach(&self, failures: &FailuresWithinReach, law: &ResolvedPresence, x: f64, y: f64, terms: &mut [f64]) {
        let mut near = Vec::new();
        failures.around(x, x, y, y, &mut near);
        for (term, z) in terms.iter_mut().zip(&self.heights) {
            let distance = |a: &[f64; 3]| {
                let dx = x - a[0];
                let dy = y - a[1];
                let dz = z - a[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            };
            let successes = self.anchors[..failures.first].iter().map(|a| law.log_probability(distance(a), true).0);
            let sum: f64 = match failures.beyond {
                Beyond::Constant { term: constant, .. } => {
                    let mut candidates = near.iter().copied().peekable();
                    let failed = (failures.first..self.anchors.len()).map(|k| {
                        if candidates.next_if_eq(&k).is_none() {
                            return constant;
                        }
                        let d = distance(&self.anchors[k]);
                        if d < failures.cut {
                            law.log_probability(d, false).0
                        } else {
                            constant
                        }
                    });
                    successes.chain(failed).sum()
                }
                Beyond::LeftOut => {
                    let failed = near.iter().filter_map(|&k| {
                        let d = distance(&self.anchors[k]);
                        (d < failures.cut).then(|| law.log_probability(d, false).0)
                    });
                    successes.chain(failed).sum()
                }
            };
            *term += sum;
        }
    }

    /// Certified lower and upper bounds of the log posterior of every cell (i, j) with i from `i0` to `i1` and j from
    /// `j0` to `j1`. The slant distances of those cells to an anchor at a height node lie between the distances from
    /// the nearest and the farthest ends of the block's centres, computed by the cells' own expression, which is
    /// monotone in each coordinate difference. Each link's table and each presence term is bounded over that interval,
    /// the bounds are summed per height node in the order of a cell, whose sums they bound since rounding is monotone,
    /// and combined over the nodes by the log sum of a cell. The log sums are widened by BOUND_MARGIN of their size per
    /// height node, since their rounding need not be monotone.
    ///
    /// Where failed attempts may lie beyond the law's cutoff, one at the cutoff or beyond from every cell of the block
    /// takes, under the Gaussian law, the constant bounds of its term there in its place, so that the bounds are bitwise
    /// those of every term, and under the logistic law nothing in the upper bound and its largest size left out below the
    /// lower one, which then holds the sum of every term too. Under the logistic law, one within the cutoff of some cells
    /// only takes the bounds of its term with an upper bound of at least zero, which a cell that leaves it out reaches.
    /// Every other failed attempt takes the bounds of its term as ever.
    pub(crate) fn bounds(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> (f64, f64) {
        let (x0, x1, y0, y1) = (self.xs[i0], self.xs[i1], self.ys[j0], self.ys[j1]);
        let nodes = self.heights.len();
        with_scratch(2 * nodes, |sums| {
            let (lower, upper) = sums.split_at_mut(nodes);
            for link in &self.links {
                let (near_x, far_x) = axis_range(x0, x1, link.anchor[0]);
                let (near_y, far_y) = axis_range(y0, y1, link.anchor[1]);
                let (near, far) = (near_x * near_x + near_y * near_y, far_x * far_x + far_y * far_y);
                for ((low, high), dz2) in lower.iter_mut().zip(upper.iter_mut()).zip(&link.dz2) {
                    let (l, h) = link.bounds.over(&link.far, (near + dz2).sqrt(), (far + dz2).sqrt());
                    *low += l;
                    *high += h;
                }
            }
            if let Some(law) = &self.law {
                // the slant distances from the block's cells at height z to an anchor lie between these two
                let span = |a: &[f64; 3], z: f64| {
                    let (near_x, far_x) = axis_range(x0, x1, a[0]);
                    let (near_y, far_y) = axis_range(y0, y1, a[1]);
                    let dz = z - a[2];
                    ((near_x * near_x + near_y * near_y + dz * dz).sqrt(), (far_x * far_x + far_y * far_y + dz * dz).sqrt())
                };
                let mut near_failures = Vec::new();
                if let Some(failures) = &self.failures {
                    failures.around(x0, x1, y0, y1, &mut near_failures);
                }
                for ((low, high), z) in lower.iter_mut().zip(upper.iter_mut()).zip(&self.heights) {
                    let (mut present_low, mut present_high) = (0.0, 0.0);
                    match &self.failures {
                        None => {
                            for (a, outcome) in self.anchors.iter().zip(&self.outcomes) {
                                let (near, far) = span(a, *z);
                                let (l, h) = law.bounds(near, far, *outcome);
                                present_low += l;
                                present_high += h;
                            }
                        }
                        Some(failures) => {
                            for a in &self.anchors[..failures.first] {
                                let (near, far) = span(a, *z);
                                let (l, h) = law.bounds(near, far, true);
                                present_low += l;
                                present_high += h;
                            }
                            match failures.beyond {
                                Beyond::Constant { bounds: constant, .. } => {
                                    let mut candidates = near_failures.iter().copied().peekable();
                                    for k in failures.first..self.anchors.len() {
                                        let (l, h) = if candidates.next_if_eq(&k).is_none() {
                                            constant
                                        } else {
                                            let (near, far) = span(&self.anchors[k], *z);
                                            if near >= failures.cut {
                                                constant
                                            } else {
                                                law.bounds(near, far, false)
                                            }
                                        };
                                        present_low += l;
                                        present_high += h;
                                    }
                                }
                                Beyond::LeftOut => {
                                    let mut beyond = failures.count;
                                    for &k in &near_failures {
                                        let (near, far) = span(&self.anchors[k], *z);
                                        if near >= failures.cut {
                                            continue;
                                        }
                                        beyond -= 1;
                                        let (l, h) = law.bounds(near, far, false);
                                        present_low += l;
                                        // a cell beyond the cutoff leaves the term out, zero, above every term of a failure
                                        present_high += if far >= failures.cut { h.max(0.0) } else { h };
                                    }
                                    present_low -= beyond as f64 * LOGISTIC_LEFT_OUT;
                                }
                            }
                        }
                    }
                    *low += present_low;
                    *high += present_high;
                }
            }
            let fold = |sums: &[f64]| sums.iter().zip(&self.weights).map(|(t, w)| t + w).fold(f64::NEG_INFINITY, log_add_exp);
            let mut low = fold(lower);
            let mut high = if upper.contains(&f64::INFINITY) { f64::INFINITY } else { fold(upper) };
            if low.is_nan() {
                low = f64::NEG_INFINITY;
            }
            if high.is_nan() {
                high = f64::INFINITY;
            }
            let scale = nodes as f64 * BOUND_MARGIN;
            (low - scale * (1.0 + low.abs()), high + scale * (1.0 + high.abs()))
        })
    }

    /// The log likelihood of the attempt outcomes under each of `laws` at cell (i, j) and every height node, added
    /// to `outcomes` together with the range terms and log weights alone in `total`, the terms the maximum likelihood
    /// reach integrates.
    fn add_outcomes(&self, i: usize, j: usize, laws: &[ResolvedPresence], total: &mut LogSum, outcomes: &mut [LogSum]) {
        let (x, y) = (self.xs[i], self.ys[j]);
        with_scratch(self.heights.len(), |terms| {
            self.range_terms(i, j, terms);
            with_scratch(self.anchors.len(), |dists| {
                for ((term, weight), z) in terms.iter().zip(&self.weights).zip(&self.heights) {
                    let range = term + weight;
                    total.add(range);
                    self.opportunity_distances(x, y, *z, dists);
                    for (sum, law) in outcomes.iter_mut().zip(laws) {
                        sum.add(range + detection_loglik(dists, &self.outcomes, law));
                    }
                }
            });
        });
    }
}

/// A node of the quadtree over the cell indices. At `level` k it covers the cells [i, i + 2^k) x [j, j + 2^k), clipped
/// to the grid, and holds bounds of their log posterior. A node of level zero is one cell, and its bounds are its
/// value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Node {
    pub(crate) level: u8,
    pub(crate) i: u32,
    pub(crate) j: u32,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

impl Node {
    fn root(nx: usize, ny: usize) -> Self {
        let mut level = 0;
        while (1usize << level) < nx.max(ny) {
            level += 1;
        }
        Self { level, i: 0, j: 0, lower: f64::NEG_INFINITY, upper: f64::INFINITY }
    }

    /// The last cell of the node along each axis.
    #[inline]
    pub(crate) fn last(&self, nx: usize, ny: usize) -> (usize, usize) {
        let side = 1usize << self.level;
        ((self.i as usize + side).min(nx) - 1, (self.j as usize + side).min(ny) - 1)
    }

    /// The cells of the node along each axis.
    #[inline]
    pub(crate) fn extent(&self, nx: usize, ny: usize) -> (usize, usize) {
        let (i1, j1) = self.last(nx, ny);
        (i1 + 1 - self.i as usize, j1 + 1 - self.j as usize)
    }

    #[inline]
    pub(crate) fn cells(&self, nx: usize, ny: usize) -> usize {
        let (w, h) = self.extent(nx, ny);
        w * h
    }

    /// The natural log of the node's mass bound, its cells times the exponential of its upper bound.
    fn mass(&self, nx: usize, ny: usize) -> f64 {
        (self.cells(nx, ny) as f64).ln() + self.upper
    }

    /// The middle cell of the node, the lower middle along an axis of even extent.
    fn centre(&self, nx: usize, ny: usize) -> (usize, usize) {
        let (i1, j1) = self.last(nx, ny);
        let (i, j) = (self.i as usize, self.j as usize);
        (i + (i1 - i) / 2, j + (j1 - j) / 2)
    }

    /// The children of a node above level zero within the grid, in canonical order: first along the first axis, then
    /// along the second.
    fn children(&self, nx: usize, ny: usize, out: &mut Vec<Node>) {
        let level = self.level - 1;
        let half = 1u32 << level;
        let child = |i: u32, j: u32| Node { level, i, j, lower: f64::NEG_INFINITY, upper: f64::INFINITY };
        let right = ((self.i + half) as usize) < nx;
        let up = ((self.j + half) as usize) < ny;
        out.push(child(self.i, self.j));
        if right {
            out.push(child(self.i + half, self.j));
        }
        if up {
            out.push(child(self.i, self.j + half));
        }
        if right && up {
            out.push(child(self.i + half, self.j + half));
        }
    }

    fn child_count(&self, nx: usize, ny: usize) -> usize {
        let half = 1usize << (self.level - 1);
        let right = self.i as usize + half < nx;
        let up = self.j as usize + half < ny;
        1 + usize::from(right) + usize::from(up) + usize::from(right && up)
    }
}

/// Bounds every node of one level of the quadtree, exactly at level zero, and returns the largest lower bound or
/// value among them, minus infinity without one. `nodes` and `cells` count the nodes bounded and the cells evaluated.
fn bound_level(evaluator: &CellEvaluator, frontier: &mut [Node], nodes: &mut usize, cells: &mut usize) -> f64 {
    let (nx, ny) = evaluator.shape();
    let mut best = f64::NEG_INFINITY;
    for node in frontier.iter_mut() {
        if node.level == 0 {
            let value = evaluator.value(node.i as usize, node.j as usize);
            (node.lower, node.upper) = (value, value);
            *cells += 1;
        } else {
            let (i1, j1) = node.last(nx, ny);
            (node.lower, node.upper) = evaluator.bounds(node.i as usize, i1, node.j as usize, j1);
            *nodes += 1;
        }
        if node.lower > best {
            best = node.lower;
        }
    }
    best
}

/// The exact value at the middle cell of the first node of the largest upper bound, minus infinity without such a
/// node, which raises the best value seen early and with it the drop threshold.
fn probe(evaluator: &CellEvaluator, frontier: &[Node], cells: &mut usize) -> f64 {
    let (nx, ny) = evaluator.shape();
    let mut chosen: Option<&Node> = None;
    for node in frontier {
        if node.upper > chosen.map_or(f64::NEG_INFINITY, |c| c.upper) {
            chosen = Some(node);
        }
    }
    match chosen {
        Some(node) if node.level > 0 => {
            let (i, j) = node.centre(nx, ny);
            *cells += 1;
            evaluator.value(i, j)
        }
        _ => f64::NEG_INFINITY,
    }
}

/// The kept part of an agent's posterior with its certificate.
pub(crate) struct Support {
    /// The kept leaves in canonical order: single cells, whose bounds are their value, and coarse leaves, whose every
    /// cell is kept.
    pub(crate) leaves: Vec<Node>,
    /// The cells of the grid along each axis.
    shape: (usize, usize),
    /// The natural log of the certified upper bound of the mass left out, the sum of every dropped node's cells times
    /// the exponential of its upper bound.
    dropped: f64,
    /// The natural log of a lower bound of the mass kept.
    kept: f64,
    /// Whether the node guard cut the refinement.
    pub(crate) guarded: bool,
    /// The nodes bounded above level zero.
    pub(crate) nodes: usize,
    /// The cells evaluated exactly.
    pub(crate) cells: usize,
}

impl Support {
    /// The support of an agent without any evidence about its position, one leaf over the whole grid with equal
    /// bounds, from which every cell is drawn with equal probability.
    pub(crate) fn uniform(grid: &Grid) -> Self {
        let (nx, ny) = (grid.xs.len(), grid.ys.len());
        let leaf = Node { lower: 0.0, upper: 0.0, ..Node::root(nx, ny) };
        Self { leaves: vec![leaf], shape: (nx, ny), dropped: f64::NEG_INFINITY, kept: (grid.count() as f64).ln(), guarded: false, nodes: 0, cells: 0 }
    }

    /// The natural log of the certified upper bound of the mass left out.
    pub(crate) fn dropped_log(&self) -> f64 {
        self.dropped
    }

    /// A support that keeps nothing, that of an agent whose value is finite on no cell, or with `guarded` that of one
    /// whose every node the guard cut, with the mass `dropped` left out.
    #[cfg(test)]
    pub(crate) fn empty(grid: &Grid, guarded: bool, dropped: f64) -> Self {
        let shape = (grid.xs.len(), grid.ys.len());
        Self { leaves: Vec::new(), shape, dropped, kept: f64::NEG_INFINITY, guarded, nodes: 0, cells: 0 }
    }

    /// The certified share of the posterior mass left out, at most EPSILON unless the node guard cut the refinement.
    /// Zero where nothing was left out, one where nothing was kept.
    pub(crate) fn dropped_share(&self) -> f64 {
        if self.dropped == f64::NEG_INFINITY {
            0.0
        } else {
            1.0 / (1.0 + (self.kept - self.dropped).exp())
        }
    }

    /// Every kept cell (i, j) in canonical order: the leaves in order, the cells of a coarse leaf along the first
    /// axis and then the second.
    pub(crate) fn for_each_cell(&self, mut f: impl FnMut(usize, usize)) {
        for leaf in &self.leaves {
            let (i1, j1) = leaf.last(self.shape.0, self.shape.1);
            for j in leaf.j as usize..=j1 {
                for i in leaf.i as usize..=i1 {
                    f(i, j);
                }
            }
        }
    }
}

/// The adaptive posterior of the evaluator's agent, refined level by level from the root of the quadtree.
///
/// At every level, in canonical order, every node is bounded, exactly at level zero, and the best value seen rises to
/// the largest lower bound and to the exact value at the middle cell of the node of the largest upper bound. A node
/// drops, its cells times the exponential of its upper bound joining the dropped mass, where its upper bound is minus
/// infinity or more than delta = ln(N / EPSILON) below the best value seen, or at level zero where its value is not a
/// number. Otherwise a node of level zero is a kept cell, a node whose bounds lie within FLAT_GAP a coarse leaf, and
/// any other node splits. Where the kept leaves and the children would exceed `guard` nodes, the nodes of the largest
/// bound mass that fit split, ties in canonical order, the others drop with their bound mass, the best value seen
/// restarts from the lower bounds of what is kept, and the support counts as guarded. A last pass drops the kept
/// leaves below the final threshold. Every decision depends on bounds and exact values alone, so the support is
/// deterministic. A guard below MIN_GUARD_NODES is refused, since it could leave a posterior without any node to split.
///
/// The certificate: the dropped mass is at most the sum of the dropped bound masses, each below e^(best - delta) per
/// cell, so at most EPSILON e^best, while the kept mass is at least e^best and at least the kept cells' values and the
/// coarse leaves' lower bounds. An agent whose best value is not finite keeps nothing.
pub(crate) fn adaptive(evaluator: &CellEvaluator, guard: usize) -> Support {
    adaptive_with(evaluator, guard, FLAT_GAP)
}

/// `adaptive` with the coarse leaves' gap `flat_gap` in place of FLAT_GAP, which the tests widen to force coarse
/// leaves.
pub(crate) fn adaptive_with(evaluator: &CellEvaluator, guard: usize, flat_gap: f64) -> Support {
    assert!(guard >= MIN_GUARD_NODES, "a node guard of {guard} cannot hold the four children of a node");
    let (nx, ny) = evaluator.shape();
    let delta = ((nx * ny) as f64 / EPSILON).ln();
    let (mut nodes, mut cells) = (0, 0);
    let mut frontier = vec![Node::root(nx, ny)];
    let mut leaves: Vec<Node> = Vec::new();
    let mut dropped = LogSum::default();
    let mut best = f64::NEG_INFINITY;
    let mut guarded = false;
    loop {
        let level = frontier[0].level;
        best = best.max(bound_level(evaluator, &mut frontier, &mut nodes, &mut cells));
        if level > 0 {
            let probed = probe(evaluator, &frontier, &mut cells);
            if probed > best {
                best = probed;
            }
        }
        let threshold = best - delta;
        let mut pending: Vec<Node> = Vec::new();
        for node in frontier.drain(..) {
            if node.upper == f64::NEG_INFINITY || (level == 0 && node.upper.is_nan()) || node.upper < threshold {
                dropped.add(node.mass(nx, ny));
            } else if level == 0 || node.upper - node.lower <= flat_gap {
                leaves.push(node);
            } else {
                pending.push(node);
            }
        }
        if pending.is_empty() {
            break;
        }
        let children: usize = pending.iter().map(|node| node.child_count(nx, ny)).sum();
        let mut split = vec![true; pending.len()];
        if leaves.len() + children > guard {
            guarded = true;
            let mut order: Vec<usize> = (0..pending.len()).collect();
            order.sort_by(|a, b| pending[*b].mass(nx, ny).total_cmp(&pending[*a].mass(nx, ny)).then(a.cmp(b)));
            let mut held = leaves.len();
            let mut fits = true;
            for &k in &order {
                let count = pending[k].child_count(nx, ny);
                fits &= held + count <= guard;
                if fits {
                    held += count;
                } else {
                    split[k] = false;
                    dropped.add(pending[k].mass(nx, ny));
                }
            }
            // the best value seen may lie in a dropped node, so it restarts from the lower bounds of what is kept, which
            // hold a cell at least that good, and the kept cells cannot all fall below the threshold later
            best = leaves.iter().chain(pending.iter().zip(&split).filter(|(_, s)| **s).map(|(node, _)| node)).map(|node| node.lower).fold(f64::NEG_INFINITY, f64::max);
        }
        for (node, split) in pending.iter().zip(&split) {
            if *split {
                node.children(nx, ny, &mut frontier);
            }
        }
        if frontier.is_empty() {
            break;
        }
    }
    if !best.is_finite() {
        for leaf in leaves.drain(..) {
            dropped.add(leaf.mass(nx, ny));
        }
    }
    let threshold = best - delta;
    let mut kept = LogSum::default();
    leaves.retain(|leaf| {
        if leaf.upper < threshold {
            dropped.add(leaf.mass(nx, ny));
            false
        } else {
            kept.add(if leaf.level == 0 { leaf.upper } else { (leaf.cells(nx, ny) as f64).ln() + leaf.lower });
            true
        }
    });
    let mut kept = kept.value();
    if !guarded && best.is_finite() {
        kept = kept.max(best);
    }
    Support { leaves, shape: (nx, ny), dropped: dropped.value(), kept, guarded, nodes, cells }
}

/// Draws cells from a support by posterior weight.
pub(crate) struct Sampler<'s> {
    support: &'s Support,
    /// The running sums of the leaves' weights relative to the largest: e^(v - top) for a cell of value v and
    /// n e^(u - top) for a coarse leaf of n cells and upper bound u.
    sums: Vec<f64>,
    /// The last leaf of a positive weight.
    last: usize,
}

impl<'s> Sampler<'s> {
    /// None for a support without a kept cell.
    pub(crate) fn new(support: &'s Support) -> Option<Self> {
        let (nx, ny) = support.shape;
        let top = support.leaves.iter().map(|leaf| leaf.upper).fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            return None;
        }
        let mut total = 0.0;
        let mut last = None;
        let sums = support
            .leaves
            .iter()
            .enumerate()
            .map(|(k, leaf)| {
                let weight = if leaf.level == 0 { (leaf.upper - top).exp() } else { leaf.cells(nx, ny) as f64 * (leaf.upper - top).exp() };
                if weight > 0.0 {
                    last = Some(k);
                }
                total += weight;
                total
            })
            .collect();
        Some(Self { support, sums, last: last? })
    }

    /// One cell by posterior weight: a leaf by its weight, a coarse leaf's cell uniformly, accepted with the probability
    /// e^(v - u) of its exact value v below the leaf's upper bound u and otherwise drawn again from the start, so that
    /// every kept cell is drawn with probability proportional to e^v. A coarse leaf with equal bounds accepts without
    /// evaluating. `cells` counts the cells evaluated.
    ///
    /// The law holds only where every value lies at or below its leaf's upper bound, which the certified bounds
    /// guarantee. A value above it would be accepted as if it lay at the bound and so drawn too rarely, and one that
    /// is not a number never, so either ends the draw with None instead, and debug builds assert it.
    pub(crate) fn draw(&self, evaluator: Option<&CellEvaluator>, stream: &mut Rng, cells: &mut usize) -> Option<(usize, usize)> {
        let (nx, ny) = self.support.shape;
        let total = self.sums[self.sums.len() - 1];
        loop {
            let u = stream.uniform() * total;
            let k = self.sums.partition_point(|sum| *sum <= u).min(self.last);
            let leaf = &self.support.leaves[k];
            if leaf.level == 0 {
                return Some((leaf.i as usize, leaf.j as usize));
            }
            let (width, height) = leaf.extent(nx, ny);
            let index = stream.below((width * height) as u64) as usize;
            let (i, j) = (leaf.i as usize + index % width, leaf.j as usize + index / width);
            if leaf.lower == leaf.upper {
                return Some((i, j));
            }
            let evaluator = evaluator.expect("a coarse leaf of unequal bounds needs the evaluator");
            *cells += 1;
            let value = evaluator.value(i, j);
            debug_assert!(value <= leaf.upper, "the value {value} of cell ({i}, {j}) lies above the upper bound of its coarse leaf {leaf:?}");
            if !(value <= leaf.upper) {
                return None;
            }
            if stream.uniform() < (value - leaf.upper).exp() {
                return Some((i, j));
            }
        }
    }
}

/// The log likelihood of an agent's attempt outcomes under each of `laws`, its position and height integrated over the
/// kept cells of its range only posterior `support`, the support of `evaluator` without presence: the log sum over
/// every kept cell and height node of the range terms, the node's log weight and the presence terms, less that of the
/// range terms and log weights alone, the kept range only mass, which comes first. The cells of coarse leaves are
/// enumerated one by one, and nothing is held but the running sums. Minus infinity under every law for a support
/// without a kept cell. `cells` counts the cells evaluated.
pub(crate) fn outcome_log_likelihoods(evaluator: &CellEvaluator, support: &Support, laws: &[ResolvedPresence], cells: &mut usize) -> (f64, Vec<f64>) {
    let mut total = LogSum::default();
    let mut outcomes = vec![LogSum::default(); laws.len()];
    support.for_each_cell(|i, j| {
        *cells += 1;
        evaluator.add_outcomes(i, j, laws, &mut total, &mut outcomes);
    });
    let total = total.value();
    if total == f64::NEG_INFINITY {
        return (total, vec![f64::NEG_INFINITY; laws.len()]);
    }
    (total, outcomes.iter().map(|sum| sum.value() - total).collect())
}

/// The natural log of the mass a posterior keeps, the log sum of the exact value of every kept cell of `support`, the
/// support of `evaluator`, in canonical order: a single cell leaf's value, and the cells of a coarse leaf evaluated
/// one by one. Minus infinity for a support without a kept cell. `cells` counts the cells evaluated.
pub(crate) fn kept_mass(evaluator: &CellEvaluator, support: &Support, cells: &mut usize) -> f64 {
    let (nx, ny) = support.shape;
    let mut mass = LogSum::default();
    for leaf in &support.leaves {
        if leaf.level == 0 {
            mass.add(leaf.upper);
            continue;
        }
        let (i1, j1) = leaf.last(nx, ny);
        for j in leaf.j as usize..=j1 {
            for i in leaf.i as usize..=i1 {
                *cells += 1;
                mass.add(evaluator.value(i, j));
            }
        }
    }
    mass.value()
}

/// The finest cell of largest log posterior found by `posterior_peak`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Peak {
    /// The centre of the cell in centimetres.
    pub(crate) x: f64,
    pub(crate) y: f64,
    /// Its log posterior, bitwise the per cell arithmetic's.
    pub(crate) value: f64,
    /// Whether the node guard cut the search, the peak then being the best cell among those it kept.
    pub(crate) guarded: bool,
    /// The nodes bounded above level zero and the cells evaluated.
    pub(crate) nodes: usize,
    pub(crate) cells: usize,
}

/// The centre of the cell of largest log posterior of one agent over a box, by branch and bound on the quadtree of the
/// box's grid, the start of the anchor-aided multistart.
///
/// Signature and inputs:
/// - `links`: the agent's range links in the caller's canonical order, each the anchor's position [x, y, height] in
///   centimetres and the link's table. The function tabulates each table between the grid's nearest and farthest
///   slant distance to its anchor, as the anchor only estimator tabulates it.
/// - `present` and `missing`: the anchors [x, y, height] of the agent's successful and failed attempts, each in the
///   caller's canonical order.
/// - `heights`: the height nodes with their normalized log weights, (z, ln w), as `aac::Graph::heights` holds them.
/// - `law`: the presence law, None for no presence term.
/// - `area` and `step`: the box and the side of a cell. The cell centres are those of the anchor only grid over that
///   area at that step: from half a step inside the lower bound by repeated additions of the step while below the
///   upper bound, and the middle of an axis where none lies below it.
///
/// The value of every cell is bitwise the anchor only estimator's per cell arithmetic (`CellEvaluator`): per height
/// node the links in the given order from 0.0, then the presence terms of the present and then the missing anchors,
/// then the log sum over the nodes of each node's sum plus its log weight. The search bounds the nodes of every level,
/// drops a node whose upper bound lies below the best lower bound or value seen, evaluates every cell that remains at
/// level zero exactly and returns the centre of the cell of the largest value. Ties go to the lowest index j * nx + i,
/// the first cell in rows from the lowest y and then the lowest x. A value that is not a number or minus infinity never
/// wins, and there is no peak when no cell's value exceeds minus infinity. There is none either for a box the grid
/// refuses: an area beyond the supported coordinate magnitude, a step that is not a positive number, or more than
/// AXIS_GUARD centres along an axis, or for a step that does not move the centres. The result is the exact maximum
/// unless a level of the search needs more than GUARD_NODES nodes, where it keeps the nodes of the largest upper bounds
/// (`peak_of` reports it).
pub(crate) fn posterior_peak(
    links: &[([f64; 3], &LinkTable)],
    present: &[[f64; 3]],
    missing: &[[f64; 3]],
    heights: &[(f64, f64)],
    law: Option<&ResolvedPresence>,
    area: &PlanarArea,
    step: f64,
) -> Option<Point2> {
    let grid = Grid::new(area, step).ok()?;
    let (zs, weights): (Vec<f64>, Vec<f64>) = heights.iter().copied().unzip();
    let evaluator = CellEvaluator::new(&grid, links, present, missing, &zs, &weights, law.copied());
    peak_of(&evaluator, GUARD_NODES).map(|peak| Point2::new(peak.x, peak.y))
}

/// The branch and bound of `posterior_peak` on an evaluator, holding at most `guard` nodes of a level. A guard below
/// MIN_GUARD_NODES is refused, since it could leave the search without any node to split.
pub(crate) fn peak_of(evaluator: &CellEvaluator, guard: usize) -> Option<Peak> {
    assert!(guard >= MIN_GUARD_NODES, "a node guard of {guard} cannot hold the four children of a node");
    let (nx, ny) = evaluator.shape();
    let (mut nodes, mut cells) = (0, 0);
    let mut frontier = vec![Node::root(nx, ny)];
    let mut best = f64::NEG_INFINITY;
    let mut guarded = false;
    loop {
        let level = frontier[0].level;
        best = best.max(bound_level(evaluator, &mut frontier, &mut nodes, &mut cells));
        if level == 0 {
            let mut chosen: Option<&Node> = None;
            for node in &frontier {
                let better = match chosen {
                    None => node.upper > f64::NEG_INFINITY,
                    Some(c) => node.upper > c.upper || (node.upper == c.upper && (node.j, node.i) < (c.j, c.i)),
                };
                if better {
                    chosen = Some(node);
                }
            }
            return chosen.map(|node| Peak {
                x: evaluator.xs[node.i as usize],
                y: evaluator.ys[node.j as usize],
                value: node.upper,
                guarded,
                nodes,
                cells,
            });
        }
        let probed = probe(evaluator, &frontier, &mut cells);
        if probed > best {
            best = probed;
        }
        let mut survivors: Vec<Node> = frontier.drain(..).filter(|node| !(node.upper == f64::NEG_INFINITY || node.upper < best)).collect();
        let children: usize = survivors.iter().map(|node| node.child_count(nx, ny)).sum();
        if children > guard {
            guarded = true;
            let mut order: Vec<usize> = (0..survivors.len()).collect();
            order.sort_by(|a, b| survivors[*b].upper.total_cmp(&survivors[*a].upper).then(a.cmp(b)));
            let mut keep = vec![false; survivors.len()];
            let mut held = 0;
            for &k in &order {
                let count = survivors[k].child_count(nx, ny);
                if held + count > guard {
                    break;
                }
                held += count;
                keep[k] = true;
            }
            let mut index = 0;
            survivors.retain(|_| {
                index += 1;
                keep[index - 1]
            });
            // the best value seen may lie in a dropped node, so it restarts from the kept nodes' lower bounds
            best = survivors.iter().map(|node| node.lower).fold(f64::NEG_INFINITY, f64::max);
        }
        for node in &survivors {
            node.children(nx, ny, &mut frontier);
        }
        if frontier.is_empty() {
            return None;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::link::{log_t_norm_for, BlockModel};
    use crate::presence::PresenceShape;
    use crate::OffsetShape;

    /// The value of every cell of the evaluator's grid in index order, the exhaustive reference: no pruning, no cap,
    /// the shared evaluator.
    pub(crate) fn exhaustive(evaluator: &CellEvaluator) -> Vec<f64> {
        let (nx, ny) = evaluator.shape();
        (0..nx * ny).map(|c| evaluator.value(c % nx, c / nx)).collect()
    }

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

    /// A table of ten packets around `d` with the Bern scale law.
    pub(crate) fn table_at(d: f64, m: &BlockModel) -> LinkTable {
        let calibration = crate::CalibrationModel::default();
        let y: Vec<f64> = (0..10).map(|k| d + 3.0 * ((k * 7 % 10) as f64 / 10.0 - 0.45)).collect();
        let s: Vec<f64> = y.iter().map(|v| calibration.scale_cm(*v)).collect();
        LinkTable::from_packets(&y, &s, m, log_t_norm_for(m.nu))
    }

    fn area(x0: f64, x1: f64, y0: f64, y1: f64) -> PlanarArea {
        PlanarArea { minimum_x_cm: x0, maximum_x_cm: x1, minimum_y_cm: y0, maximum_y_cm: y1 }
    }

    #[test]
    fn the_grid_keeps_its_centres_and_refuses_axes_beyond_the_guard() {
        let grid = Grid::new(&area(0.0, 100.0, -5.0, 5.0), 10.0).unwrap();
        assert_eq!(grid.xs, (0..10).map(|k| 5.0 + 10.0 * k as f64).collect::<Vec<_>>());
        assert_eq!(grid.ys, vec![0.0]);
        // an axis shorter than half a step takes its middle
        let narrow = Grid::new(&area(0.0, 3.0, 0.0, 100.0), 10.0).unwrap();
        assert_eq!(narrow.xs, vec![1.5]);
        // 2^22 steps along an axis may hold 2^22 + 1 centres by the bound, which the guard refuses
        let width = AXIS_GUARD - 1.0;
        assert!(Grid::new(&area(0.0, width, 0.0, 10.0), 1.0).is_ok());
        assert!(matches!(Grid::new(&area(0.0, width + 1.0, 0.0, 10.0), 1.0), Err(EngineError::InvalidProfile(_))));
        assert!(matches!(Grid::new(&area(0.0, 10.0, 0.0, 1.0e7), 1.0), Err(EngineError::InvalidProfile(_))));
        assert!(matches!(Grid::new(&area(0.0, 10.0, 0.0, 10.0), f64::NAN), Err(EngineError::InvalidProfile(_))));
        assert!(matches!(Grid::new(&area(0.0, 2.0e7, 0.0, 10.0), 100.0), Err(EngineError::InvalidProfile(_))));
    }

    /// A step below half the spacing of the numbers at an axis's coordinates, 1.9e-9 cm near 1e7 cm, leaves a centre where
    /// it was when added, so the axis once grew without end. The axis guard lets such a step through on a narrow area,
    /// and the first addition that does not move a centre refuses it. A step just large enough still builds the axis.
    #[test]
    fn a_step_that_does_not_move_the_centres_is_refused() {
        let (step, edge) = (5.0e-10, 1.0e7 - 1.0e-3);
        for narrow in [area(edge, 1.0e7, 0.0, 1.0e-3), area(0.0, 1.0e-3, edge, 1.0e7)] {
            // both axes pass the axis guard, so the stall alone refuses the step
            let per_axis = |low: f64, high: f64| ((high - low) / step).ceil() + 1.0;
            assert!(per_axis(narrow.minimum_x_cm, narrow.maximum_x_cm) <= AXIS_GUARD && per_axis(narrow.minimum_y_cm, narrow.maximum_y_cm) <= AXIS_GUARD);
            assert!((edge + step / 2.0) + step == edge + step / 2.0);
            assert!(matches!(Grid::new(&narrow, step), Err(EngineError::InvalidProfile(message)) if message.contains("does not move")), "{narrow:?}");
        }
        let grid = Grid::new(&area(edge, 1.0e7, 0.0, 1.0e-3), 5.0e-6).unwrap();
        assert!(grid.xs.windows(2).all(|pair| pair[1] > pair[0]) && grid.xs.len() > 100, "{}", grid.xs.len());
    }

    /// The value of cell (i, j) with every presence term, the arithmetic without the cutoff of the failed attempts.
    fn value_with_every_term(evaluator: &CellEvaluator, i: usize, j: usize) -> f64 {
        let mut terms = vec![0.0; evaluator.heights.len()];
        evaluator.range_terms(i, j, &mut terms);
        if let Some(law) = &evaluator.law {
            let (x, y) = (evaluator.xs[i], evaluator.ys[j]);
            let mut dists = vec![0.0; evaluator.anchors.len()];
            for (term, z) in terms.iter_mut().zip(&evaluator.heights) {
                evaluator.opportunity_distances(x, y, *z, &mut dists);
                *term += detection_loglik(&dists, &evaluator.outcomes, law);
            }
        }
        terms.iter().zip(&evaluator.weights).map(|(t, w)| t + w).fold(f64::NEG_INFINITY, log_add_exp)
    }

    /// From 7.45 reaches on, a failed attempt's Gaussian term is the clamp's constant ln(1 - 1e-12) bitwise, with no
    /// slope, whatever the reach, and a little inside 7.434 reaches it is not.
    #[test]
    fn a_failures_gaussian_term_beyond_the_cutoff_is_the_clamps_constant() {
        let constant = (-1e-12_f64).ln_1p();
        for reach in [1.0, 37.5, 400.0, 1500.0, 2.0e5] {
            let law = ResolvedPresence { shape: PresenceShape::Gaussian, reach, width: 0.15 };
            let cut = GAUSSIAN_CUTOFF_REACHES * reach;
            let mut distances: Vec<f64> = (0..4000).map(|k| cut * (1.0 + k as f64 * 2.5e-4)).collect();
            distances.extend([cut, f64::from_bits(cut.to_bits() + 1), 100.0 * cut, 1.0e7]);
            for d in distances {
                let (term, slope) = law.log_probability(d, false);
                assert_eq!((term.to_bits(), slope), (constant.to_bits(), 0.0), "reach {reach} at {d}");
            }
            assert_ne!(law.log_probability(7.40 * reach, false).0, constant);
        }
    }

    /// From r (1 + 27.7 w) on, a failed attempt's logistic term lies between -LOGISTIC_LEFT_OUT and zero, whatever the
    /// reach and the width.
    #[test]
    fn a_failures_logistic_term_beyond_the_cutoff_is_smaller_than_what_is_left_out() {
        for width in [0.02, 0.15, 0.5, 3.0] {
            for reach in [1.0, 400.0, 1.0e5] {
                let law = ResolvedPresence { shape: PresenceShape::Logistic, reach, width };
                let cut = reach * (1.0 + LOGISTIC_CUTOFF_WIDTHS * width);
                for k in 0..4000 {
                    let d = cut * (1.0 + k as f64 * 2.5e-4);
                    let term = law.log_probability(d, false).0;
                    assert!(-LOGISTIC_LEFT_OUT < term && term <= 0.0, "width {width} reach {reach}: {term} at {d}");
                }
            }
        }
    }

    /// An agent near a corner of a site of 600 by 400 m with its three links and 400 failed attempts spread over the
    /// site, under both law shapes with a reach of 6 m, so that most failed attempts lie beyond the cutoff from most cells.
    /// Against the same evaluator without the cutoff, the arithmetic of every term: under the Gaussian law every value and
    /// the bounds of every block are bitwise the same, and under the logistic law every value lies above the sum of every
    /// term by less than 1e-12 a failed attempt, and the bounds of random blocks hold both. The adaptive posterior keeps
    /// its certificate under both.
    #[test]
    fn failures_beyond_the_cutoff_keep_values_and_bounds_certified() {
        let m = model(0.28, 0.31, 8.0, 2.5);
        let grid = Grid::new(&area(0.0, 60_000.0, 0.0, 40_000.0), 100.0).unwrap();
        let (nx, ny) = (grid.xs.len(), grid.ys.len());
        let truth = [1_500.0, 1_200.0];
        let present: Vec<[f64; 3]> = vec![[1_000.0, 900.0, 250.0], [2_000.0, 1_100.0, 240.0], [1_300.0, 1_700.0, 260.0]];
        let tables: Vec<LinkTable> =
            present.iter().map(|a| table_at(((truth[0] - a[0]).powi(2) + (truth[1] - a[1]).powi(2) + (a[2] - 100.0).powi(2)).sqrt(), &m)).collect();
        let links: Vec<([f64; 3], &LinkTable)> = present.iter().copied().zip(&tables).collect();
        let mut state = 17_u64;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let missing: Vec<[f64; 3]> = (0..400).map(|_| [60_000.0 * uniform(), 40_000.0 * uniform(), 200.0 + 100.0 * uniform()]).collect();
        for shape in [PresenceShape::Gaussian, PresenceShape::Logistic] {
            let law = Some(ResolvedPresence { shape, reach: 600.0, width: 0.15 });
            let evaluator = CellEvaluator::new(&grid, &links, &present, &missing, &[80.0, 120.0], &[-0.7, -0.7], law);
            assert!(evaluator.beyond_reach(), "{shape:?}");
            let mut every = CellEvaluator::new(&grid, &links, &present, &missing, &[80.0, 120.0], &[-0.7, -0.7], law);
            every.failures = None;
            let values = exhaustive(&evaluator);
            let every_values = exhaustive(&every);
            for (c, (value, exact)) in values.iter().zip(&every_values).enumerate() {
                assert_eq!(*exact, value_with_every_term(&every, c % nx, c / nx), "cell {c}");
                match shape {
                    PresenceShape::Gaussian => assert_eq!(value.to_bits(), exact.to_bits(), "cell {c}: {value} against {exact}"),
                    PresenceShape::Logistic => {
                        let rounding = 1e-12 * (1.0 + exact.abs());
                        let allowance = missing.len() as f64 * LOGISTIC_LEFT_OUT;
                        assert!(exact - rounding <= *value && *value <= exact + allowance + rounding, "cell {c}: {value} against {exact}");
                    }
                }
            }
            for _ in 0..300 {
                let (w, h) = (1 + (uniform() * 64.0) as usize, 1 + (uniform() * 64.0) as usize);
                let (i0, j0) = ((uniform() * (nx - w + 1) as f64) as usize, (uniform() * (ny - h + 1) as f64) as usize);
                let (low, high) = evaluator.bounds(i0, i0 + w - 1, j0, j0 + h - 1);
                if shape == PresenceShape::Gaussian {
                    let (every_low, every_high) = every.bounds(i0, i0 + w - 1, j0, j0 + h - 1);
                    assert_eq!((low.to_bits(), high.to_bits()), (every_low.to_bits(), every_high.to_bits()), "block ({i0}, {j0}) of {w} by {h}");
                }
                for j in j0..j0 + h {
                    for i in i0..i0 + w {
                        let (value, exact) = (values[j * nx + i], every_values[j * nx + i]);
                        assert!(low <= value.min(exact) && value.max(exact) <= high, "{shape:?} ({i}, {j}): {value} {exact} outside [{low}, {high}]");
                    }
                }
            }
            let support = adaptive(&evaluator, GUARD_NODES);
            let comparison = compare(&support, &values);
            assert!(!support.guarded && comparison.kept >= comparison.leading, "{shape:?}: {comparison:?}");
            if shape == PresenceShape::Gaussian {
                assert_eq!(support.leaves, adaptive(&every, GUARD_NODES).leaves);
            }
        }
    }

    /// Where every failed attempt lies within the cutoff of every cell, the evaluator keeps the arithmetic without it:
    /// on a room, and on the large site of the test above with a reach whose cutoff exceeds the site, every value is
    /// bitwise the sum of every term.
    #[test]
    fn a_grid_within_the_cutoff_of_every_failure_keeps_every_term() {
        let (grid, tables, anchors) = agents();
        for (name, evaluator) in evaluators(&grid, &tables, &anchors) {
            assert!(!evaluator.beyond_reach(), "{name}");
        }
        let grid = Grid::new(&area(0.0, 60_000.0, 0.0, 40_000.0), 500.0).unwrap();
        let (nx, ny) = (grid.xs.len(), grid.ys.len());
        let missing: Vec<[f64; 3]> = (0..60).map(|k| [1_000.0 * k as f64, 600.0 * k as f64, 250.0]).collect();
        for shape in [PresenceShape::Gaussian, PresenceShape::Logistic] {
            let law = Some(ResolvedPresence { shape, reach: 20_000.0, width: 0.15 });
            let evaluator = CellEvaluator::new(&grid, &[], &[[500.0, 500.0, 250.0]], &missing, &[100.0], &[0.0], law);
            assert!(!evaluator.beyond_reach(), "{shape:?}");
            for c in 0..nx * ny {
                assert_eq!(evaluator.value(c % nx, c / nx).to_bits(), value_with_every_term(&evaluator, c % nx, c / nx).to_bits(), "{shape:?} cell {c}");
            }
        }
    }

    /// A node takes the 32 bytes GUARD_NODES counts, so a posterior holds at most 2^23 of them, 256 MiB, and twice that
    /// while it refines one level into the next.
    #[test]
    fn a_node_takes_32_bytes() {
        assert_eq!(std::mem::size_of::<Node>(), 32);
    }

    #[test]
    fn a_uniform_draw_below_a_count_is_uniform() {
        let mut rng = Rng(42);
        let mut counts = [0usize; 3];
        for _ in 0..300_000 {
            counts[rng.below(3) as usize] += 1;
        }
        assert!(counts.iter().all(|c| (*c as f64 - 100_000.0).abs() < 1500.0), "{counts:?}");
        assert_eq!(Rng(1).below(1), 0);
        let mut large = Rng(9);
        assert!((0..1000).all(|_| large.below(u64::MAX - 5) < u64::MAX - 5));
    }

    /// Four anchors around a room, presence on, and several height nodes: the bounds of random blocks hold the value
    /// of every cell of the block, for both law shapes and with present and missing anchors.
    #[test]
    fn node_bounds_hold_every_cell_of_their_block() {
        let tables: Vec<LinkTable> = [430.0, 610.0, 820.0].iter().map(|d| table_at(*d, &model(0.28, 0.31, 8.0, 2.5))).collect();
        let anchors = [[0.0, 0.0, 250.0], [1500.0, 0.0, 240.0], [1500.0, 1000.0, 260.0], [0.0, 1000.0, 250.0]];
        let grid = Grid::new(&area(-200.0, 1700.0, -150.0, 1150.0), 13.0).unwrap();
        let (nx, ny) = (grid.xs.len(), grid.ys.len());
        let mut state = 3_u64;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        for shape in [PresenceShape::Gaussian, PresenceShape::Logistic] {
            for law in [None, Some(ResolvedPresence { shape, reach: 700.0, width: 0.15 })] {
                let links: Vec<([f64; 3], &LinkTable)> = anchors[..3].iter().copied().zip(&tables).collect();
                let evaluator = CellEvaluator::new(&grid, &links, &anchors[..3], &anchors[3..], &[0.0, 80.0, 160.0], &[-1.2, -0.9, -1.2], law);
                for _ in 0..300 {
                    let (w, h) = (1 + (uniform() * 40.0) as usize, 1 + (uniform() * 40.0) as usize);
                    let i0 = (uniform() * (nx - w) as f64) as usize;
                    let j0 = (uniform() * (ny - h) as f64) as usize;
                    let (low, high) = evaluator.bounds(i0, i0 + w - 1, j0, j0 + h - 1);
                    assert!(low.is_finite() && high.is_finite());
                    for j in j0..j0 + h {
                        for i in i0..i0 + w {
                            let value = evaluator.value(i, j);
                            assert!(low <= value && value <= high, "{value} at ({i}, {j}) outside [{low}, {high}]");
                        }
                    }
                }
            }
        }
    }

    /// Every scanned block of link.rs, the fits of Bern, TIERS, IDLab and Bregar with early detections, both offset
    /// shapes and independent packets among them, as the table of two links of an agent with a third anchor it failed
    /// to reach, over two height nodes, on a grid of 64 cells a side that spans the table and its continuation on
    /// both sides. The bounds of 40 random blocks of up to 16 by 16 cells, without presence and under both law shapes,
    /// hold the value of every cell of the block.
    #[test]
    fn node_bounds_hold_every_cell_for_every_scanned_block() {
        let blocks = crate::link::tests::scanned_blocks();
        let check = |(y, s, m): &(Vec<f64>, Vec<f64>, BlockModel)| {
            let table = LinkTable::from_packets(y, s, m, log_t_norm_for(m.nu));
            let reach = y.iter().copied().fold(0.0, f64::max) + 300.0;
            let anchors = [[0.0, 0.0, 250.0], [0.35 * reach, -0.2 * reach, 230.0], [-0.3 * reach, 0.4 * reach, 260.0]];
            let grid = Grid::new(&area(-reach, reach, -reach, reach), 2.0 * reach / 64.0).unwrap();
            let (nx, ny) = (grid.xs.len(), grid.ys.len());
            let mut state = y.iter().fold(17_u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3));
            let mut uniform = move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1_u64 << 53) as f64
            };
            let links = [(anchors[0], &table), (anchors[1], &table)];
            for law in [None, Some(PresenceShape::Gaussian), Some(PresenceShape::Logistic)] {
                let law = law.map(|shape| ResolvedPresence { shape, reach: 0.8 * reach, width: 0.15 });
                let evaluator = CellEvaluator::new(&grid, &links, &anchors[..2], &anchors[2..], &[0.0, 120.0], &[-0.7, -0.7], law);
                for _ in 0..40 {
                    let (w, h) = (1 + (uniform() * 16.0) as usize, 1 + (uniform() * 16.0) as usize);
                    let (i0, j0) = ((uniform() * (nx - w + 1) as f64) as usize, (uniform() * (ny - h + 1) as f64) as usize);
                    let (low, high) = evaluator.bounds(i0, i0 + w - 1, j0, j0 + h - 1);
                    assert!(low.is_finite() && high.is_finite(), "{low} {high} for {y:?} {m:?}");
                    for j in j0..j0 + h {
                        for i in i0..i0 + w {
                            let value = evaluator.value(i, j);
                            assert!(low <= value && value <= high, "{value} at ({i}, {j}) outside [{low}, {high}] for {y:?} {m:?} {law:?}");
                        }
                    }
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

    /// An agent with three links in a room: the peak's value is the largest of the exhaustive reference and its cell
    /// the first of that value, and a forced guard reports itself.
    #[test]
    fn the_peak_is_the_first_cell_of_the_largest_value() {
        let m = model(0.28, 0.31, 8.0, 2.5);
        let truth: [f64; 2] = [610.0, 380.0];
        let anchors: [[f64; 3]; 4] = [[0.0, 0.0, 250.0], [1500.0, 0.0, 250.0], [700.0, 1000.0, 250.0], [1500.0, 1000.0, 250.0]];
        let tables: Vec<LinkTable> =
            anchors[..3].iter().map(|a| table_at(((truth[0] - a[0]).powi(2) + (truth[1] - a[1]).powi(2) + (a[2] - 100.0).powi(2)).sqrt(), &m)).collect();
        let links: Vec<([f64; 3], &LinkTable)> = anchors[..3].iter().copied().zip(&tables).collect();
        let heights = [(60.0, -1.1), (100.0, -1.1), (140.0, -1.1)];
        let law = Some(ResolvedPresence { shape: PresenceShape::Gaussian, reach: 900.0, width: 0.15 });
        let box_ = area(-300.0, 1800.0, -300.0, 1300.0);
        let grid = Grid::new(&box_, 7.0).unwrap();
        let evaluator = CellEvaluator::new(&grid, &links, &anchors[..3], &anchors[3..], &[60.0, 100.0, 140.0], &[-1.1, -1.1, -1.1], law);
        let peak = peak_of(&evaluator, GUARD_NODES).unwrap();
        assert_eq!(posterior_peak(&links, &anchors[..3], &anchors[3..], &heights, law.as_ref(), &box_, 7.0), Some(Point2::new(peak.x, peak.y)));
        let values = exhaustive(&evaluator);
        let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let first = values.iter().position(|v| *v == best).unwrap();
        assert_eq!(peak.value.to_bits(), best.to_bits());
        assert_eq!((peak.x, peak.y), (grid.xs[first % grid.xs.len()], grid.ys[first / grid.xs.len()]));
        assert!(!peak.guarded && peak.cells < values.len() / 20, "{} cells of {}", peak.cells, values.len());
        assert!((peak.x - truth[0]).hypot(peak.y - truth[1]) < 30.0, "{peak:?}");
        let tight = peak_of(&evaluator, 5).unwrap();
        assert!(tight.guarded);
        // no peak for a box the grid refuses, nor where no value exceeds minus infinity
        assert_eq!(posterior_peak(&links, &[], &[], &heights, None, &area(0.0, 1.0e7, 0.0, 10.0), 1.0), None);
        assert_eq!(posterior_peak(&links, &[], &[], &[], None, &box_, 7.0), None);
    }

    /// What a support keeps and leaves out against the exhaustive reference `values` of its evaluator.
    #[derive(Debug)]
    pub(crate) struct Comparison {
        /// Kept cells, those of single cell leaves, those of coarse leaves.
        pub(crate) kept: usize,
        pub(crate) finest: usize,
        pub(crate) coarse: usize,
        /// Cells of the smallest leading set with 1 - EPSILON of the exact mass.
        pub(crate) leading: usize,
        /// The natural log of the exact mass left out and of the whole exact mass.
        pub(crate) dropped: f64,
        pub(crate) total: f64,
    }

    /// Compares a support with the exhaustive reference and asserts the certificate: every single cell leaf carries
    /// bitwise its reference value, every cell of a coarse leaf lies within the leaf's bounds, the certified dropped
    /// mass is at least the exact one and the reported share at least the exact share, and without the guard every
    /// cell of the smallest leading set with 1 - EPSILON of the exact mass is kept and the reported share is at most
    /// EPSILON.
    pub(crate) fn compare(support: &Support, values: &[f64]) -> Comparison {
        let (nx, ny) = support.shape;
        assert_eq!(values.len(), nx * ny);
        let mut kept = vec![false; nx * ny];
        let (mut finest, mut coarse) = (0, 0);
        for leaf in &support.leaves {
            let (i1, j1) = leaf.last(nx, ny);
            for j in leaf.j as usize..=j1 {
                for i in leaf.i as usize..=i1 {
                    let c = j * nx + i;
                    assert!(!kept[c], "cell ({i}, {j}) kept twice");
                    kept[c] = true;
                    if leaf.level == 0 {
                        assert_eq!(values[c].to_bits(), leaf.upper.to_bits(), "cell ({i}, {j})");
                        finest += 1;
                    } else {
                        assert!(leaf.lower <= values[c] && values[c] <= leaf.upper, "cell ({i}, {j}): {} outside {leaf:?}", values[c]);
                        coarse += 1;
                    }
                }
            }
        }
        let (mut total, mut dropped) = (LogSum::default(), LogSum::default());
        for (c, value) in values.iter().enumerate() {
            total.add(*value);
            if !kept[c] {
                dropped.add(*value);
            }
        }
        let (total, exact_dropped) = (total.value(), dropped.value());
        assert!(
            exact_dropped == f64::NEG_INFINITY || exact_dropped <= support.dropped + 1e-9 * (1.0 + support.dropped.abs()),
            "exact dropped {exact_dropped} above the certified {}",
            support.dropped
        );
        // the smallest leading set, every cell at or above its last value
        let mut order: Vec<usize> = (0..values.len()).filter(|c| !values[*c].is_nan()).collect();
        order.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
        let mut held = LogSum::default();
        let mut leading = 0;
        for &c in &order {
            held.add(values[c]);
            leading += 1;
            if held.value() >= total + (1.0 - EPSILON).ln() {
                break;
            }
        }
        let last = values[order[leading - 1]];
        if !support.guarded {
            assert!(values.iter().enumerate().all(|(c, v)| v.is_nan() || *v < last || kept[c]), "a cell of the leading set was left out");
        }
        let exact_share = if exact_dropped == f64::NEG_INFINITY { 0.0 } else { (exact_dropped - total).exp() };
        assert!(exact_share <= support.dropped_share() * (1.0 + 1e-9) + 1e-300, "exact share {exact_share} above {}", support.dropped_share());
        if !support.guarded {
            assert!(support.dropped_share() <= EPSILON, "{}", support.dropped_share());
        }
        Comparison { kept: finest + coarse, finest, coarse, leading, dropped: exact_dropped, total }
    }

    /// The agents of the tests on a 1900 by 1300 cm room at 13 cm: a ring around one anchor under a wide offset,
    /// three links and a failed attempt under either law shape over three height nodes, and the presence pattern of
    /// one success and two failures alone.
    fn agents() -> (Grid, Vec<LinkTable>, Vec<[f64; 3]>) {
        let grid = Grid::new(&area(-200.0, 1700.0, -150.0, 1150.0), 13.0).unwrap();
        let mut tables: Vec<LinkTable> = [430.0, 610.0, 820.0].iter().map(|d| table_at(*d, &model(0.28, 0.31, 8.0, 2.5))).collect();
        tables.push(table_at(640.0, &model(1.0, 0.05, 2.0, 8.0)));
        let anchors = vec![[0.0, 0.0, 250.0], [1500.0, 0.0, 240.0], [1500.0, 1000.0, 260.0], [0.0, 1000.0, 250.0], [700.0, 500.0, 200.0]];
        (grid, tables, anchors)
    }

    fn evaluators<'a>(grid: &'a Grid, tables: &'a [LinkTable], anchors: &[[f64; 3]]) -> Vec<(&'static str, CellEvaluator<'a>)> {
        let heights = [0.0, 80.0, 160.0];
        let weights = [-1.2, -0.9, -1.2];
        let law = |shape| Some(ResolvedPresence { shape, reach: 700.0, width: 0.15 });
        let room: Vec<([f64; 3], &LinkTable)> = anchors[..3].iter().copied().zip(tables).collect();
        vec![
            ("ring", CellEvaluator::new(grid, &[(anchors[4], &tables[3])], &[anchors[4]], &[], &[0.0], &[0.0], None)),
            ("room gaussian", CellEvaluator::new(grid, &room, &anchors[..3], &anchors[3..4], &heights, &weights, law(PresenceShape::Gaussian))),
            ("room logistic", CellEvaluator::new(grid, &room, &anchors[..3], &anchors[3..4], &heights, &weights, law(PresenceShape::Logistic))),
            ("presence only", CellEvaluator::new(grid, &[], &anchors[4..5], &anchors[..2], &heights, &weights, law(PresenceShape::Gaussian))),
        ]
    }

    #[test]
    fn the_adaptive_posterior_keeps_every_cell_that_counts_with_its_exact_value() {
        let (grid, tables, anchors) = agents();
        for (name, evaluator) in evaluators(&grid, &tables, &anchors) {
            let values = exhaustive(&evaluator);
            let support = adaptive(&evaluator, GUARD_NODES);
            let comparison = compare(&support, &values);
            assert!(!support.guarded, "{name}");
            // the work is a fraction of the grid's, where the posterior is not flat
            println!("{name}: {comparison:?}, {} nodes, {} cells of {}", support.nodes, support.cells, values.len());
            assert!(support.cells < values.len(), "{name}");
            // the kept cells hold at least the smallest leading set, and the single cells lie within delta of the best
            // value seen, at most FLAT_GAP below the best cell
            let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let delta = (values.len() as f64 / EPSILON).ln();
            assert!(comparison.kept >= comparison.leading, "{name}");
            let near = values.iter().filter(|v| **v >= best - delta - FLAT_GAP).count();
            assert!(comparison.finest <= near, "{name}: {} single cells, {near} within delta", comparison.finest);
            // the kept mass is the log sum of the reference values of the kept cells in canonical order, bitwise
            let (nx, _) = evaluator.shape();
            let mut reference = LogSum::default();
            support.for_each_cell(|i, j| reference.add(values[j * nx + i]));
            let mut cells = 0;
            assert_eq!(kept_mass(&evaluator, &support, &mut cells).to_bits(), reference.value().to_bits(), "{name}");
            assert_eq!(cells, comparison.coarse, "{name}");
        }
    }

    /// A coarse leaf whose upper bound lies a nat below the best value of its cells, which the certified bounds exclude.
    /// Its draws accepted every cell above the bound as if it lay at the bound, so the best cell was drawn at 6 in 100
    /// where its exact share is 14, and nothing told. The draw that reaches such a cell ends with None, and debug builds
    /// assert it.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "lies above the upper bound of its coarse leaf"))]
    fn a_value_above_its_leafs_upper_bound_ends_the_draws() {
        let table = table_at(300.0, &model(0.28, 0.31, 8.0, 2.5));
        let grid = Grid::new(&area(0.0, 160.0, 0.0, 160.0), 20.0).unwrap();
        let evaluator = CellEvaluator::new(&grid, &[([0.0, 0.0, 0.0], &table)], &[], &[], &[0.0], &[0.0], None);
        let values = exhaustive(&evaluator);
        let (nx, ny) = evaluator.shape();
        let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bottom = values.iter().copied().fold(f64::INFINITY, f64::min);
        let leaf = Node { lower: bottom, upper: top - 1.0, ..Node::root(nx, ny) };
        let support = Support { leaves: vec![leaf], shape: (nx, ny), dropped: f64::NEG_INFINITY, kept: top, guarded: false, nodes: 0, cells: 0 };
        let sampler = Sampler::new(&support).unwrap();
        let mut stream = Rng(11);
        let draws = 400_000;
        let mut counts = vec![0u32; nx * ny];
        for _ in 0..draws {
            let Some((i, j)) = sampler.draw(Some(&evaluator), &mut stream, &mut 0) else { return };
            counts[j * nx + i] += 1;
        }
        // every draw gave a cell, so the draws must follow the exact law, which a bound below a value cannot give
        let best = values.iter().position(|v| *v == top).unwrap();
        let mut total = LogSum::default();
        values.iter().for_each(|v| total.add(*v));
        let (exact, drawn) = ((top - total.value()).exp(), f64::from(counts[best]) / draws as f64);
        assert!((drawn - exact).abs() < 0.01, "the best cell was drawn at {drawn:.4} for an exact share of {exact:.4}, and nothing told");
    }

    /// A guard far below the posterior's needs drops what it cannot hold with its certified mass and says so.
    #[test]
    fn a_tiny_guard_is_reported_and_its_mass_certified() {
        let (grid, tables, anchors) = agents();
        for (name, evaluator) in evaluators(&grid, &tables, &anchors) {
            let values = exhaustive(&evaluator);
            let support = adaptive(&evaluator, 40);
            assert!(support.guarded, "{name}");
            assert!(support.leaves.len() <= 40, "{name}");
            let comparison = compare(&support, &values);
            assert!(support.dropped_share() > 0.0 && support.dropped_share() <= 1.0, "{name}: {comparison:?}");
            assert!(Sampler::new(&support).is_some() || support.leaves.is_empty());
        }
    }

    /// With coarse leaves forced by a gap of two and a half nats, a million draws of the room and of the presence
    /// pattern alone match the exact law of the kept cells: the chi-square statistic over bins of consecutive cells
    /// with at least 40 expected draws stays within five standard deviations of its degrees of freedom.
    #[test]
    fn draws_follow_the_exact_law_of_the_kept_cells() {
        let (grid, tables, anchors) = agents();
        for (name, evaluator) in evaluators(&grid, &tables, &anchors).into_iter().skip(1).step_by(2) {
            let (nx, ny) = evaluator.shape();
            let support = adaptive_with(&evaluator, GUARD_NODES, 2.5);
            let values = exhaustive(&evaluator);
            let comparison = compare(&support, &values);
            assert!(comparison.coarse > 0, "{name}: {comparison:?}");
            let sampler = Sampler::new(&support).unwrap();
            let mut stream = Rng(20260927);
            let draws = 1_000_000;
            let mut counts = vec![0u32; nx * ny];
            let mut cells = 0;
            for _ in 0..draws {
                let (i, j) = sampler.draw(Some(&evaluator), &mut stream, &mut cells).unwrap();
                counts[j * nx + i] += 1;
            }
            let mut kept = LogSum::default();
            support.for_each_cell(|i, j| kept.add(values[j * nx + i]));
            let kept = kept.value();
            let (mut statistic, mut bins, mut expected, mut observed) = (0.0, 0, 0.0, 0.0);
            support.for_each_cell(|i, j| {
                let c = j * nx + i;
                expected += draws as f64 * (values[c] - kept).exp();
                observed += f64::from(counts[c]);
                if expected >= 40.0 {
                    statistic += (observed - expected).powi(2) / expected;
                    bins += 1;
                    (expected, observed) = (0.0, 0.0);
                }
            });
            let freedom = (bins - 1) as f64;
            println!("{name}: chi-square {statistic:.1} over {bins} bins, {cells} evaluations for {draws} draws");
            assert!(statistic < freedom + 5.0 * (2.0 * freedom).sqrt(), "{name}: {statistic} over {bins} bins");
            assert!(counts.iter().enumerate().all(|(c, n)| *n == 0 || !values[c].is_nan()));
        }
    }
}
