//! The start of the anchor aided solver with the block range model.
//!
//! The start is the geometric initializer's layout in which the peak of an agent's own anchor posterior replaces its
//! lateration wherever its block range links identify its position. They do when they reach at least MIN_PEAK_ANCHORS
//! horizontal anchor places that do not all lie within PEAK_LINE_DISTANCE_CM of one line. Each such agent is a target of
//! its component's alignment by its peak and sits there, and the laterations of the other agents stay targets of the
//! alignment alone, as in the initializer (`initializer::layout_with_overrides`). The anchor posterior is the anchor only
//! estimator's posterior of the agent's anchor links and presence opportunities, its height marginalized over the height
//! nodes. The peak is the cell of largest posterior value on a grid of PEAK_STEP_CM over the union of the squares around
//! the agent's linked anchors, each square of half side the upper end of its link's table, found by the branch and bound
//! of the adaptive posterior (`posterior::posterior_peak`).
//!
//! Every stage of the solve runs the multistart of its start and, as after the geometric initializer, one more start
//! that places every agent with a lateration of its own there and keeps the others where the stage starts
//! (`lateration_start`). So an agent whose peak replaced its lateration still gets a start at that lateration.

use std::cmp::Ordering;

#[cfg(test)]
use crate::link::FarTable;
use crate::link::LinkTable;
use crate::presence::{PresenceProfile, ResolvedPresence};
use crate::{CalibrationModel, EngineError, PlanarArea, Point2, PreparedProblem, ReachRule, SolveConfig};

use super::aac;

/// Side in centimetres of the grid cells on which the peak of an agent's anchor posterior is searched.
pub(crate) const PEAK_STEP_CM: f64 = 1.0;

/// The fewest horizontal anchor places an agent's block range links must reach for the peak of its anchor posterior to
/// replace its lateration in the start. One place leaves a ring of positions and two leave a mirrored pair, of which the
/// peak can take the wrong one, so the lateration of an agent with fewer places stays a target of the alignment alone
/// and the agent keeps its place in the aligned embedding. Anchors at one horizontal position count once whatever their
/// heights, since anchors stacked above one another leave the same ring or mirror as one of them.
pub(crate) const MIN_PEAK_ANCHORS: usize = 3;

/// The distance in centimetres from one line within which an agent's horizontal anchor places must not all lie for the
/// peak of its anchor posterior to replace its lateration. Places that all lie close to one line, anchors along one wall
/// for instance, give a position and its mirror image across that line nearly the same posterior value, and the peak
/// can take the mirror. The places pass when their smallest width over all directions exceeds twice this distance
/// (`minimum_width`), and the lateration of an agent whose places do not pass stays, as for fewer places.
pub(crate) const PEAK_LINE_DISTANCE_CM: f64 = 100.0;

/// One anchor link of an agent, the anchor's position and height and the block likelihood table of the link.
pub(crate) type AnchorLink<'a> = ([f64; 3], &'a LinkTable);

/// The start layout of the anchor aided solver with the block range model and the presence law of the scenario,
/// resolved with the final stage's `graph`. The quantile reach comes from the successful links before any layout. The
/// maximum likelihood reach judges the attempt outcomes at the layout with the range only peaks, which leave out every
/// presence term, and the peaks are then taken again under its law.
pub(crate) fn start(
    problem: &PreparedProblem,
    config: &SolveConfig,
    calibration: CalibrationModel,
    graph: &aac::Graph,
    presence: &PresenceProfile,
) -> Result<(Vec<Point2>, Option<ResolvedPresence>), EngineError> {
    let layout = |law: Option<&ResolvedPresence>| {
        crate::initializer::layout_with_overrides(problem, config, calibration, &peak_overrides(graph, law))
    };
    match presence.reach {
        ReachRule::Quantile => {
            let law = presence.resolve(&graph.successful);
            Ok((layout(law.as_ref())?, law))
        }
        ReachRule::Ml => {
            let range_only = layout(None)?;
            match graph.resolve_presence(presence, &range_only) {
                Some(law) => Ok((layout(Some(&law))?, Some(law))),
                None => Ok((range_only, None)),
            }
        }
    }
}

/// The override of every agent's lateration under `law`, the peak of its anchor posterior when its block range links
/// identify its position (`identifies`). None for every other agent and for an agent without a cell of finite value,
/// whose lateration stays.
pub(crate) fn peak_overrides(graph: &aac::Graph, law: Option<&ResolvedPresence>) -> Vec<Option<Point2>> {
    (0..graph.agents)
        .map(|agent| {
            let (links, present, missing) = agent_inputs(graph, agent);
            if !identifies(&links) {
                return None;
            }
            agent_peak(&links, &present, &missing, &graph.heights, law, &peak_area(&links))
        })
        .collect()
}

/// Whether an agent's anchor `links` identify its position for the start. Their horizontal anchor places number at least
/// MIN_PEAK_ANCHORS and do not all lie within PEAK_LINE_DISTANCE_CM of one line, so their smallest width over all
/// directions exceeds twice that distance.
pub(super) fn identifies(links: &[AnchorLink<'_>]) -> bool {
    let places = horizontal_places(links);
    places.len() >= MIN_PEAK_ANCHORS && minimum_width(&places) > 2.0 * PEAK_LINE_DISTANCE_CM
}

/// The distinct horizontal positions of the anchors of `links`, anchors at one position counting once whatever their
/// heights, in the order of x and then y.
pub(super) fn horizontal_places(links: &[AnchorLink<'_>]) -> Vec<[f64; 2]> {
    let mut places: Vec<[f64; 2]> = links.iter().map(|(anchor, _)| [anchor[0], anchor[1]]).collect();
    places.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    places.dedup();
    places
}

/// The smallest width of `points` over all directions, the least distance between two parallel lines that hold every
/// point between them, and zero for fewer than three points or points on one line. The points all lie within half this
/// width of the middle line of the narrowest such strip, and no line comes closer to all of them. The narrowest strip
/// has one side along an edge of the convex hull, so the width is the least over the hull's edges of the largest
/// distance of a hull vertex from the edge's line. `points` follow the order of x and then y, without repeats.
pub(super) fn minimum_width(points: &[[f64; 2]]) -> f64 {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return 0.0;
    }
    (0..hull.len())
        .map(|k| {
            let (a, b) = (hull[k], hull[(k + 1) % hull.len()]);
            let farthest = hull.iter().map(|p| cross(a, b, *p).abs()).fold(0.0, f64::max);
            farthest / (b[0] - a[0]).hypot(b[1] - a[1])
        })
        .fold(f64::INFINITY, f64::min)
}

/// The vertices of the convex hull of `points`, which follow the order of x and then y without repeats, counterclockwise
/// and without the points inside its edges (Andrew's monotone chain). Fewer than three vertices for points on one line.
fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut hull = half_hull(points.iter().copied());
    hull.extend(half_hull(points.iter().rev().copied()));
    hull
}

/// One chain of the monotone chain, the lower hull of points in increasing order or the upper hull of points in
/// decreasing order, without its last point, which begins the other chain.
fn half_hull(points: impl Iterator<Item = [f64; 2]>) -> Vec<[f64; 2]> {
    let mut chain: Vec<[f64; 2]> = Vec::new();
    for p in points {
        while chain.len() >= 2 && cross(chain[chain.len() - 2], chain[chain.len() - 1], p) <= 0.0 {
            chain.pop();
        }
        chain.push(p);
    }
    chain.pop();
    chain
}

/// Twice the signed area of the triangle `a`, `b`, `p`, positive when `p` lies to the left of the line from `a` to `b`.
fn cross(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// The peak of one agent's anchor posterior on the grid of PEAK_STEP_CM over `area`, the one call of the search, by the
/// branch and bound of the adaptive posterior (`posterior::posterior_peak`). The peak is the centre of the first cell of
/// largest value, in rows from the lowest y and within a row from the lowest x, every value bitwise the anchor only
/// estimator's. None where no cell has a value above minus infinity or the grid refuses the area.
pub(super) fn agent_peak(
    links: &[AnchorLink<'_>],
    present: &[[f64; 3]],
    missing: &[[f64; 3]],
    heights: &[(f64, f64)],
    law: Option<&ResolvedPresence>,
    area: &PlanarArea,
) -> Option<Point2> {
    crate::posterior::posterior_peak(links, present, missing, heights, law, area, PEAK_STEP_CM)
}

/// The block range links of one agent's anchor pairs with their anchors, and the anchors of its present and missing
/// presence opportunities, in canonical order. The links follow the anchor geometry and then their packets, and the
/// opportunities follow the anchor geometry, so that neither the anchor identifiers nor the order of the request can
/// change a peak.
pub(super) fn agent_inputs(graph: &aac::Graph, agent: usize) -> (Vec<AnchorLink<'_>>, Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut links: Vec<(AnchorLink<'_>, &[(f64, f64)])> = Vec::new();
    let (mut present, mut missing) = (Vec::new(), Vec::new());
    for &k in &graph.anchor_links[agent] {
        let pair = &graph.pairs[k];
        let anchor = graph.anchors[pair.right - graph.agents];
        if let Some(link) = &pair.range {
            if let Some(table) = &link.table {
                links.push(((anchor, table), &link.packets));
            }
        }
        if pair.opportunity {
            if pair.observed {
                present.push(anchor);
            } else {
                missing.push(anchor);
            }
        }
    }
    links.sort_by(|(left, left_packets), (right, right_packets)| {
        geometry(&left.0).cmp(&geometry(&right.0)).then_with(|| packet_order(left_packets, right_packets))
    });
    present.sort_by_key(geometry);
    missing.sort_by_key(geometry);
    (links.into_iter().map(|(link, _)| link).collect(), present, missing)
}

/// The canonical key of an anchor, its geometry.
fn geometry(anchor: &[f64; 3]) -> [u64; 3] {
    anchor.map(f64::to_bits)
}

/// The canonical order of the packets of two links at one place, by distance and then scale packet by packet, and
/// then by their count.
fn packet_order(left: &[(f64, f64)], right: &[(f64, f64)]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(a, b)| a.0.to_bits().cmp(&b.0.to_bits()).then_with(|| a.1.to_bits().cmp(&b.1.to_bits())))
        .find(|order| order.is_ne())
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.len().cmp(&right.len()))
}

/// The area of an agent's peak search, the union of the squares around the anchors of its `links`, each square of
/// half side the upper end of its link's table.
pub(super) fn peak_area(links: &[AnchorLink<'_>]) -> PlanarArea {
    let empty = PlanarArea {
        minimum_x_cm: f64::INFINITY,
        maximum_x_cm: f64::NEG_INFINITY,
        minimum_y_cm: f64::INFINITY,
        maximum_y_cm: f64::NEG_INFINITY,
    };
    links.iter().fold(empty, |area, (anchor, table)| {
        let half = table.range().1;
        PlanarArea {
            minimum_x_cm: area.minimum_x_cm.min(anchor[0] - half),
            maximum_x_cm: area.maximum_x_cm.max(anchor[0] + half),
            minimum_y_cm: area.minimum_y_cm.min(anchor[1] - half),
            maximum_y_cm: area.maximum_y_cm.max(anchor[1] + half),
        }
    })
}

/// The peak of an agent's anchor posterior over `area` at `step` by evaluating every cell (`CellPosterior`), the
/// independent reference of the tests for the branch and bound search the start calls. The peak is the centre of the
/// cell of largest value, the first such cell in the grid's order, row by row from the lowest y and within a row from
/// the lowest x. A cell whose value is not a number or minus infinity is never the peak, and None means that no cell
/// has another value. `links` holds the agent's anchor links and `present` and `missing` the anchors of its presence
/// opportunities, each in canonical order, `heights` the height nodes with their normalized log weights, and `law` the
/// presence law, None for the range only posterior.
#[cfg(test)]
pub(super) fn exhaustive_peak(
    links: &[AnchorLink<'_>],
    present: &[[f64; 3]],
    missing: &[[f64; 3]],
    heights: &[(f64, f64)],
    law: Option<&ResolvedPresence>,
    area: &PlanarArea,
    step: f64,
) -> Option<Point2> {
    let mut posterior = CellPosterior::new(links, present, missing, heights, law, area, step);
    let mut values = vec![0.0; posterior.xs.len()];
    let mut best: Option<(f64, Point2)> = None;
    for r in 0..posterior.ys.len() {
        posterior.row(r, &mut values);
        for (i, value) in values.iter().enumerate() {
            if *value > best.map_or(f64::NEG_INFINITY, |(top, _)| top) {
                best = Some((*value, Point2::new(posterior.xs[i], posterior.ys[r])));
            }
        }
    }
    best.map(|(_, peak)| peak)
}

/// Every cell of an agent's anchor posterior over an area, with the anchor only estimator's grid and arithmetic.
///
/// The cell centres follow that grid. Each axis starts half a step inside its lower bound and adds the step while
/// below the upper bound, or holds the middle of an axis shorter than half a step. Every link's table has its
/// continuation tabulated over the slant distances from its anchor to the centres at the height nodes, a centimetre
/// wider on either side. At every cell and height node the link terms add to zero in link order, the presence terms
/// of the present and then the missing anchors add as one sum, and the node sums plus their log weights combine by
/// the log sum exp in node order. A transcription of the anchor only grid kept apart from the adaptive posterior, the
/// reference of the tests.
#[cfg(test)]
pub(super) struct CellPosterior<'a> {
    pub(super) xs: Vec<f64>,
    pub(super) ys: Vec<f64>,
    /// Every link with its table, whose continuation is tabulated over the grid.
    links: Vec<(Offsets, FarTable<'a>)>,
    /// The anchors of the presence opportunities, the present ones first, and their outcomes.
    opportunities: Vec<Offsets>,
    outcomes: Vec<bool>,
    log_weights: Vec<f64>,
    law: Option<ResolvedPresence>,
    /// One row of terms per height node and the slant distances of the opportunities at one cell.
    terms: Vec<Vec<f64>>,
    distances: Vec<f64>,
}

/// An anchor with the squared offsets from it along x of every grid column and along z of every height node.
#[cfg(test)]
struct Offsets {
    anchor: [f64; 3],
    dx2: Vec<f64>,
    dz2: Vec<f64>,
}

#[cfg(test)]
impl Offsets {
    fn new(anchor: &[f64; 3], xs: &[f64], zs: &[f64]) -> Self {
        Self {
            anchor: *anchor,
            dx2: xs.iter().map(|x| (x - anchor[0]) * (x - anchor[0])).collect(),
            dz2: zs.iter().map(|z| (z - anchor[2]) * (z - anchor[2])).collect(),
        }
    }

    /// The squared offset along y of a grid row at `y`.
    fn dy2(&self, y: f64) -> f64 {
        (y - self.anchor[1]) * (y - self.anchor[1])
    }
}

#[cfg(test)]
impl<'a> CellPosterior<'a> {
    pub(super) fn new(
        links: &[AnchorLink<'a>],
        present: &[[f64; 3]],
        missing: &[[f64; 3]],
        heights: &[(f64, f64)],
        law: Option<&ResolvedPresence>,
        area: &PlanarArea,
        step: f64,
    ) -> Self {
        let axis = |low: f64, high: f64| {
            let mut centres = Vec::new();
            let mut v = low + step / 2.0;
            while v < high {
                centres.push(v);
                v += step;
            }
            if centres.is_empty() {
                centres.push((low + high) / 2.0);
            }
            centres
        };
        let xs = axis(area.minimum_x_cm, area.maximum_x_cm);
        let ys = axis(area.minimum_y_cm, area.maximum_y_cm);
        let zs: Vec<f64> = heights.iter().map(|(z, _)| *z).collect();
        let links = links
            .iter()
            .map(|(anchor, table)| {
                let (nearest, farthest) = slant_span(&xs, &ys, anchor, &zs);
                (Offsets::new(anchor, &xs, &zs), table.far(nearest, farthest))
            })
            .collect();
        let opportunities: Vec<Offsets> = present.iter().chain(missing).map(|anchor| Offsets::new(anchor, &xs, &zs)).collect();
        let outcomes = present.iter().map(|_| true).chain(missing.iter().map(|_| false)).collect();
        let terms = vec![vec![0.0; xs.len()]; zs.len()];
        let distances = vec![0.0; opportunities.len()];
        Self {
            xs,
            ys,
            links,
            opportunities,
            outcomes,
            log_weights: heights.iter().map(|(_, w)| *w).collect(),
            law: law.copied(),
            terms,
            distances,
        }
    }

    /// The value of every cell of row `r` into `values`, the log posterior of the cell marginalized over the height
    /// nodes, up to the normalization.
    pub(super) fn row(&mut self, r: usize, values: &mut [f64]) {
        let Self { ys, links, opportunities, outcomes, log_weights, law, terms, distances, .. } = self;
        let y = ys[r];
        let link_dy2: Vec<f64> = links.iter().map(|(offsets, _)| offsets.dy2(y)).collect();
        let presence_dy2: Vec<f64> = opportunities.iter().map(|offsets| offsets.dy2(y)).collect();
        for (node, row) in terms.iter_mut().enumerate() {
            row.fill(0.0);
            for ((offsets, table), dy2) in links.iter().zip(&link_dy2) {
                let dz2 = offsets.dz2[node];
                for (term, dx2) in row.iter_mut().zip(&offsets.dx2) {
                    *term += table.eval((dx2 + dy2 + dz2).sqrt());
                }
            }
            if let Some(law) = law {
                for (i, term) in row.iter_mut().enumerate() {
                    for ((distance, offsets), dy2) in distances.iter_mut().zip(opportunities.iter()).zip(&presence_dy2) {
                        *distance = (offsets.dx2[i] + dy2 + offsets.dz2[node]).sqrt();
                    }
                    *term += detection_loglik(distances, outcomes, law);
                }
            }
        }
        for (i, value) in values.iter_mut().enumerate() {
            *value = (0..log_weights.len()).map(|node| terms[node][i] + log_weights[node]).fold(f64::NEG_INFINITY, log_add_exp);
        }
    }
}

/// The nearest and the farthest slant distance from `anchor` to a cell centre at a height node, a centimetre wider on
/// either side, as the anchor only grid spans them.
#[cfg(test)]
fn slant_span(xs: &[f64], ys: &[f64], anchor: &[f64; 3], heights: &[f64]) -> (f64, f64) {
    let (x_low, x_high) = (xs[0], xs[xs.len() - 1]);
    let (y_low, y_high) = (ys[0], ys[ys.len() - 1]);
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

/// The presence log likelihood of the opportunities at `dists`, `present` whether each produced ranges, as the anchor
/// only estimator sums it.
#[cfg(test)]
fn detection_loglik(dists: &[f64], present: &[bool], law: &ResolvedPresence) -> f64 {
    dists.iter().zip(present).map(|(d, p)| law.log_probability(*d, *p).0).sum()
}

/// `ln(exp(a) + exp(b))` as the anchor only estimator computes it.
#[cfg(test)]
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
