use std::cmp::Ordering;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};
use thiserror::Error;

pub const INPUT_SHA256: &str = "65c077bb1aa9b8793d9b206a406ecfd574047ad345fbafdbf93cfcdfad26fa21";
pub const FITTER_PROTOCOL_SHA256: &str =
    "4721cec2b8e7178a756db2e94ee5c20bfe6b852161248ad55b80af884589b4e1";
pub const ROW_HEADER: &[u8] = b"SOLID.BERN.CALIBRATION.ROWS.V1\n";
pub const TRAINING_SCANS: [i64; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
pub const SCAN_COUNTS: [u64; 11] = [
    174_363, 181_526, 176_367, 179_537, 175_715, 174_897, 174_111, 180_338, 174_896, 182_267,
    173_869,
];

const RECORD_BYTES: usize = 80;
const SOURCE_ROW_COUNT: usize = 2_124_073;
const MAD_MULTIPLIER: f64 = 1.4826;
const MAD_FLOOR: f64 = 1e-12;
const IRLS_TOLERANCE: f64 = 1e-10;
const IRLS_LIMIT: usize = 100;
const SCALE_MIN_CM: f64 = 8.0;
const SCALE_MAX_CM: f64 = 55.0;
const RUNTIME_TOF_FLOOR_NS: f64 = 1.0;
const RUNTIME_TOF_MAX_NS: f64 = 200.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationRow {
    pub source_ordinal: u64,
    pub scan_id: i64,
    pub timestamp: i64,
    pub listening_period: i64,
    pub poller_id: i64,
    pub poller_pos_id: i64,
    pub responder_id: i64,
    pub responder_pos_id: i64,
    pub tof_ns: f64,
    pub distance_cm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RowManifest {
    pub bytes_sha256: String,
    pub rows: Vec<CalibrationRow>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationFit {
    pub range_slope_cm_per_ns: f64,
    pub range_offset_cm: f64,
    pub range_iterations: u32,
    pub range_final_residual_center_cm: f64,
    pub range_final_mad_sigma_cm: f64,
    pub scale_slope: f64,
    pub scale_intercept_cm: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedSample {
    pub distance_cm: f64,
    pub scale_cm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitReceipt {
    pub fit_id: String,
    pub fit_scan_ids: Vec<i64>,
    pub held_scan_id: Option<i64>,
    pub row_count: u64,
    pub artifact_path: String,
    pub artifact_sha256: String,
    pub row_manifest_path: String,
    pub row_manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationManifest {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub fits: Vec<FitReceipt>,
}

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("calibration row manifest has an invalid header or length")]
    InvalidRowManifest,
    #[error("calibration row manifest contains an invalid value or order")]
    InvalidRow,
    #[error("calibration row manifest has the wrong scans or row count")]
    InvalidPartition,
    #[error("weighted affine input is invalid or singular")]
    InvalidAffineInput,
    #[error("range IRLS did not converge")]
    IrlsNonconvergence,
    #[error("the fitted range slope is not positive")]
    NonpositiveRangeSlope,
    #[error("calibration arithmetic produced a nonfinite value")]
    NonfiniteArithmetic,
    #[error("calibration output already exists")]
    OutputExists,
    #[error("calibration count or byte length overflowed")]
    Overflow,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub fn read_row_manifest(path: &Path) -> Result<RowManifest, CalibrationError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    parse_row_manifest(&bytes)
}

pub fn parse_row_manifest(bytes: &[u8]) -> Result<RowManifest, CalibrationError> {
    let prefix = ROW_HEADER
        .len()
        .checked_add(8)
        .ok_or(CalibrationError::Overflow)?;
    if bytes.len() < prefix || &bytes[..ROW_HEADER.len()] != ROW_HEADER {
        return Err(CalibrationError::InvalidRowManifest);
    }
    let row_count = u64::from_le_bytes(
        bytes[ROW_HEADER.len()..prefix]
            .try_into()
            .map_err(|_| CalibrationError::InvalidRowManifest)?,
    );
    let row_count_usize = usize::try_from(row_count).map_err(|_| CalibrationError::Overflow)?;
    let expected = row_count_usize
        .checked_mul(RECORD_BYTES)
        .and_then(|length| length.checked_add(prefix))
        .ok_or(CalibrationError::Overflow)?;
    if bytes.len() != expected {
        return Err(CalibrationError::InvalidRowManifest);
    }

    let mut rows = Vec::with_capacity(row_count_usize);
    let mut seen_source_ordinals = vec![false; SOURCE_ROW_COUNT];
    for record in bytes[prefix..].chunks_exact(RECORD_BYTES) {
        let row = decode_row(record)?;
        let source_ordinal =
            usize::try_from(row.source_ordinal).map_err(|_| CalibrationError::InvalidRow)?;
        let seen = seen_source_ordinals
            .get_mut(source_ordinal)
            .ok_or(CalibrationError::InvalidRow)?;
        if *seen {
            return Err(CalibrationError::InvalidRow);
        }
        *seen = true;
        if rows
            .last()
            .is_some_and(|previous| row_order(previous, &row) != Ordering::Less)
        {
            return Err(CalibrationError::InvalidRow);
        }
        rows.push(row);
    }
    Ok(RowManifest {
        bytes_sha256: hex_digest(bytes),
        rows,
    })
}

pub fn fit_calibration(rows: &[CalibrationRow]) -> Result<CalibrationFit, CalibrationError> {
    fit_calibration_with_limit(rows, IRLS_LIMIT)
}

fn fit_calibration_with_limit(
    rows: &[CalibrationRow],
    iteration_limit: usize,
) -> Result<CalibrationFit, CalibrationError> {
    if rows.len() < 100
        || rows.iter().any(|row| {
            !row.tof_ns.is_finite()
                || row.tof_ns < 0.0
                || !row.distance_cm.is_finite()
                || row.distance_cm < 0.0
        })
        || rows
            .windows(2)
            .any(|pair| row_order(&pair[0], &pair[1]) != Ordering::Less)
    {
        return Err(CalibrationError::InvalidRow);
    }
    let x = rows.iter().map(|row| row.tof_ns).collect::<Vec<_>>();
    let y = rows.iter().map(|row| row.distance_cm).collect::<Vec<_>>();
    let mut weights = vec![1.0; rows.len()];
    let (mut slope, mut offset) = weighted_affine(&x, &y, &weights)?;
    let mut final_iterations = None;

    for iteration in 1..=iteration_limit {
        let residuals = residuals(&x, &y, slope, offset)?;
        weights = one_mad_weights(&residuals)?.0;
        let (next_slope, next_offset) = weighted_affine(&x, &y, &weights)?;
        let difference = (next_slope - slope).abs().max((next_offset - offset).abs());
        slope = next_slope;
        offset = next_offset;
        if difference < IRLS_TOLERANCE {
            final_iterations =
                Some(u32::try_from(iteration).map_err(|_| CalibrationError::Overflow)?);
            break;
        }
    }
    let range_iterations = final_iterations.ok_or(CalibrationError::IrlsNonconvergence)?;
    if slope <= 0.0 {
        return Err(CalibrationError::NonpositiveRangeSlope);
    }

    let final_residuals = residuals(&x, &y, slope, offset)?;
    let final_center = median(&final_residuals)?;
    let final_deviations = final_residuals
        .iter()
        .map(|residual| finite((residual - final_center).abs()))
        .collect::<Result<Vec<_>, _>>()?;
    let final_sigma = finite(MAD_MULTIPLIER * median(&final_deviations)?)?.max(MAD_FLOOR);
    let (scale_slope, scale_intercept_cm) = scale_fit(rows, slope, offset)?;

    Ok(CalibrationFit {
        range_slope_cm_per_ns: positive_zero(slope),
        range_offset_cm: positive_zero(offset),
        range_iterations,
        range_final_residual_center_cm: positive_zero(final_center),
        range_final_mad_sigma_cm: positive_zero(final_sigma),
        scale_slope: positive_zero(scale_slope),
        scale_intercept_cm: positive_zero(scale_intercept_cm),
    })
}

fn one_mad_weights(residuals: &[f64]) -> Result<(Vec<f64>, f64, f64), CalibrationError> {
    let center = median(residuals)?;
    let deviations = residuals
        .iter()
        .map(|residual| finite((residual - center).abs()))
        .collect::<Result<Vec<_>, _>>()?;
    let sigma = finite(MAD_MULTIPLIER * median(&deviations)?)?.max(MAD_FLOOR);
    let weights = deviations
        .into_iter()
        .map(|deviation| {
            let standardized = finite(deviation / sigma)?;
            if standardized <= 1.0 {
                Ok(1.0)
            } else {
                finite(1.0 / standardized)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((weights, center, sigma))
}

pub fn convert_sample(
    fit: CalibrationFit,
    tof_ns: f64,
) -> Result<CalibratedSample, CalibrationError> {
    if !tof_ns.is_finite() || !(0.0..=RUNTIME_TOF_MAX_NS).contains(&tof_ns) {
        return Err(CalibrationError::InvalidRow);
    }
    let x = tof_ns.max(RUNTIME_TOF_FLOOR_NS);
    let linear_range = finite(fit.range_slope_cm_per_ns * x)?;
    let linear_range = finite(linear_range + fit.range_offset_cm)?;
    let distance_cm = positive_zero(linear_range.max(0.0));
    let (scale_cm, _) = scale_and_derivative(fit, distance_cm)?;
    Ok(CalibratedSample {
        distance_cm,
        scale_cm,
    })
}

pub fn scale_and_derivative(
    fit: CalibrationFit,
    distance_cm: f64,
) -> Result<(f64, f64), CalibrationError> {
    if !distance_cm.is_finite() || distance_cm < 0.0 {
        return Err(CalibrationError::InvalidRow);
    }
    let linear_scale = finite(fit.scale_slope * distance_cm)?;
    let linear_scale = finite(linear_scale + fit.scale_intercept_cm)?;
    let (scale, derivative) = if linear_scale <= SCALE_MIN_CM {
        (SCALE_MIN_CM, 0.0)
    } else if linear_scale >= SCALE_MAX_CM {
        (SCALE_MAX_CM, 0.0)
    } else {
        (linear_scale, fit.scale_slope)
    };
    Ok((positive_zero(scale), positive_zero(derivative)))
}

pub fn fit_all(calibration_directory: &Path) -> Result<CalibrationManifest, CalibrationError> {
    fs::create_dir_all(calibration_directory)?;
    let manifest_path = calibration_directory.join("manifest.json");
    let artifact_paths = TRAINING_SCANS
        .into_iter()
        .map(Some)
        .chain([None])
        .map(|held_scan| calibration_directory.join(format!("{}.json", fit_id(held_scan),)))
        .collect::<Vec<_>>();
    let mut output_paths = artifact_paths.clone();
    output_paths.push(manifest_path.clone());
    preflight_outputs(&output_paths)?;

    let mut receipts = Vec::with_capacity(12);
    let mut artifacts = Vec::with_capacity(12);
    for held_scan in TRAINING_SCANS.into_iter().map(Some).chain([None]) {
        let fit_id = fit_id(held_scan);
        let fit_scans = TRAINING_SCANS
            .into_iter()
            .filter(|scan| Some(*scan) != held_scan)
            .collect::<Vec<_>>();
        let row_manifest_name = format!("{fit_id}.rows.bin");
        let row_manifest = read_row_manifest(&calibration_directory.join(&row_manifest_name))?;
        validate_partition(&row_manifest.rows, &fit_scans)?;
        let fit = fit_calibration(&row_manifest.rows)?;
        let artifact_name = format!("{fit_id}.json");
        let artifact = artifact_bytes(&fit_id, &fit_scans, held_scan, &row_manifest, fit)?;
        let artifact_path = calibration_directory.join(&artifact_name);
        artifacts.push((artifact_path, artifact.clone()));
        receipts.push(FitReceipt {
            fit_id,
            fit_scan_ids: fit_scans,
            held_scan_id: held_scan,
            row_count: u64::try_from(row_manifest.rows.len())
                .map_err(|_| CalibrationError::Overflow)?,
            artifact_path: artifact_name,
            artifact_sha256: hex_digest(&artifact),
            row_manifest_path: row_manifest_name,
            row_manifest_sha256: row_manifest.bytes_sha256,
        });
    }
    let bytes = manifest_bytes(&receipts);
    artifacts.push((manifest_path, bytes.clone()));
    publish_all(calibration_directory, &artifacts)?;
    Ok(CalibrationManifest {
        sha256: hex_digest(&bytes),
        bytes,
        fits: receipts,
    })
}

fn fit_id(held_scan: Option<i64>) -> String {
    held_scan.map_or_else(
        || "full-training".to_owned(),
        |scan| format!("outer-holdout-scan-{scan:02}"),
    )
}

fn decode_row(record: &[u8]) -> Result<CalibrationRow, CalibrationError> {
    let mut fields = record.chunks_exact(8);
    let source_ordinal = decode_u64(fields.next())?;
    let scan_id = decode_i64(fields.next())?;
    let timestamp = decode_i64(fields.next())?;
    let listening_period = decode_i64(fields.next())?;
    let poller_id = decode_i64(fields.next())?;
    let poller_pos_id = decode_i64(fields.next())?;
    let responder_id = decode_i64(fields.next())?;
    let responder_pos_id = decode_i64(fields.next())?;
    let tof_bits = decode_u64(fields.next())?;
    let distance_bits = decode_u64(fields.next())?;
    let tof_ns = f64::from_bits(tof_bits);
    let distance_cm = f64::from_bits(distance_bits);
    if fields.next().is_some()
        || tof_bits != canonical_bits(tof_ns)
        || distance_bits != canonical_bits(distance_cm)
        || !tof_ns.is_finite()
        || tof_ns < 0.0
        || !distance_cm.is_finite()
        || distance_cm < 0.0
    {
        return Err(CalibrationError::InvalidRow);
    }
    Ok(CalibrationRow {
        source_ordinal,
        scan_id,
        timestamp,
        listening_period,
        poller_id,
        poller_pos_id,
        responder_id,
        responder_pos_id,
        tof_ns,
        distance_cm,
    })
}

fn decode_u64(field: Option<&[u8]>) -> Result<u64, CalibrationError> {
    Ok(u64::from_le_bytes(
        field
            .ok_or(CalibrationError::InvalidRowManifest)?
            .try_into()
            .map_err(|_| CalibrationError::InvalidRowManifest)?,
    ))
}

fn decode_i64(field: Option<&[u8]>) -> Result<i64, CalibrationError> {
    Ok(i64::from_le_bytes(
        field
            .ok_or(CalibrationError::InvalidRowManifest)?
            .try_into()
            .map_err(|_| CalibrationError::InvalidRowManifest)?,
    ))
}

fn row_order(left: &CalibrationRow, right: &CalibrationRow) -> Ordering {
    left.scan_id
        .cmp(&right.scan_id)
        .then_with(|| left.timestamp.cmp(&right.timestamp))
        .then_with(|| left.listening_period.cmp(&right.listening_period))
        .then_with(|| left.poller_id.cmp(&right.poller_id))
        .then_with(|| left.poller_pos_id.cmp(&right.poller_pos_id))
        .then_with(|| left.responder_id.cmp(&right.responder_id))
        .then_with(|| left.responder_pos_id.cmp(&right.responder_pos_id))
        .then_with(|| canonical_bits(left.tof_ns).cmp(&canonical_bits(right.tof_ns)))
        .then_with(|| canonical_bits(left.distance_cm).cmp(&canonical_bits(right.distance_cm)))
        .then_with(|| left.source_ordinal.cmp(&right.source_ordinal))
}

fn weighted_affine(x: &[f64], y: &[f64], weights: &[f64]) -> Result<(f64, f64), CalibrationError> {
    if x.is_empty() || x.len() != y.len() || x.len() != weights.len() {
        return Err(CalibrationError::InvalidAffineInput);
    }
    let mut weight_sum = Neumaier::default();
    let mut x_sum = Neumaier::default();
    let mut y_sum = Neumaier::default();
    for ((&x_value, &y_value), &weight) in x.iter().zip(y).zip(weights) {
        if !x_value.is_finite() || !y_value.is_finite() || !weight.is_finite() || weight < 0.0 {
            return Err(CalibrationError::InvalidAffineInput);
        }
        let x_term = finite(weight * x_value)?;
        let y_term = finite(weight * y_value)?;
        weight_sum.add(weight)?;
        x_sum.add(x_term)?;
        y_sum.add(y_term)?;
    }
    let total_weight = weight_sum.finish()?;
    if total_weight <= 0.0 {
        return Err(CalibrationError::InvalidAffineInput);
    }
    let x_bar = finite(x_sum.finish()? / total_weight)?;
    let y_bar = finite(y_sum.finish()? / total_weight)?;
    let mut sxx = Neumaier::default();
    let mut sxy = Neumaier::default();
    for ((&x_value, &y_value), &weight) in x.iter().zip(y).zip(weights) {
        let dx = finite(x_value - x_bar)?;
        let dy = finite(y_value - y_bar)?;
        let square = finite(dx * dx)?;
        let sxx_term = finite(weight * square)?;
        let cross = finite(dx * dy)?;
        let sxy_term = finite(weight * cross)?;
        sxx.add(sxx_term)?;
        sxy.add(sxy_term)?;
    }
    let sxx = sxx.finish()?;
    if sxx <= 0.0 {
        return Err(CalibrationError::InvalidAffineInput);
    }
    let slope = finite(sxy.finish()? / sxx)?;
    let product = finite(slope * x_bar)?;
    let offset = finite(y_bar - product)?;
    Ok((positive_zero(slope), positive_zero(offset)))
}

fn residuals(x: &[f64], y: &[f64], slope: f64, offset: f64) -> Result<Vec<f64>, CalibrationError> {
    x.iter()
        .zip(y)
        .map(|(&x_value, &y_value)| {
            let prediction = finite(slope * x_value)?;
            let prediction = finite(prediction + offset)?;
            finite(y_value - prediction)
        })
        .collect()
}

fn scale_fit(
    rows: &[CalibrationRow],
    slope: f64,
    offset: f64,
) -> Result<(f64, f64), CalibrationError> {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| {
        canonical_bits(left.distance_cm)
            .cmp(&canonical_bits(right.distance_cm))
            .then_with(|| row_order(left, right))
    });
    let lengths = bin_lengths(ordered.len())?;
    let mut distances = Vec::with_capacity(100);
    let mut scales = Vec::with_capacity(100);
    let mut start = 0_usize;
    for length in lengths {
        let end = start
            .checked_add(length)
            .ok_or(CalibrationError::Overflow)?;
        let rows = &ordered[start..end];
        let bin_distances = rows.iter().map(|row| row.distance_cm).collect::<Vec<_>>();
        let bin_residuals = rows
            .iter()
            .map(|row| {
                let fitted = finite(slope * row.tof_ns)?;
                let fitted = finite(fitted + offset)?;
                finite(fitted - row.distance_cm)
            })
            .collect::<Result<Vec<_>, CalibrationError>>()?;
        let center = median(&bin_residuals)?;
        let deviations = bin_residuals
            .iter()
            .map(|residual| finite((residual - center).abs()))
            .collect::<Result<Vec<_>, _>>()?;
        distances.push(median(&bin_distances)?);
        scales.push(finite(MAD_MULTIPLIER * median(&deviations)?)?);
        start = end;
    }
    if start != ordered.len() {
        return Err(CalibrationError::InvalidAffineInput);
    }
    weighted_affine(&distances, &scales, &[1.0; 100])
}

fn bin_lengths(row_count: usize) -> Result<[usize; 100], CalibrationError> {
    if row_count < 100 {
        return Err(CalibrationError::InvalidAffineInput);
    }
    let quotient = row_count / 100;
    let remainder = row_count % 100;
    Ok(std::array::from_fn(|bin| {
        quotient + usize::from(bin < remainder)
    }))
}

fn median(values: &[f64]) -> Result<f64, CalibrationError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(CalibrationError::InvalidAffineInput);
    }
    let mut ordered = values
        .iter()
        .copied()
        .map(positive_zero)
        .collect::<Vec<_>>();
    ordered.sort_unstable_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    let result = if ordered.len() % 2 == 0 {
        let lower = ordered[middle - 1];
        let difference = finite(ordered[middle] - lower)?;
        let half = finite(difference / 2.0)?;
        finite(lower + half)?
    } else {
        ordered[middle]
    };
    Ok(positive_zero(result))
}

fn validate_partition(rows: &[CalibrationRow], scans: &[i64]) -> Result<(), CalibrationError> {
    let expected = scans.iter().try_fold(0_u64, |total, scan| {
        let index = TRAINING_SCANS
            .iter()
            .position(|candidate| candidate == scan)
            .ok_or(CalibrationError::InvalidPartition)?;
        total
            .checked_add(SCAN_COUNTS[index])
            .ok_or(CalibrationError::Overflow)
    })?;
    if u64::try_from(rows.len()).map_err(|_| CalibrationError::Overflow)? != expected
        || rows.iter().any(|row| !scans.contains(&row.scan_id))
    {
        return Err(CalibrationError::InvalidPartition);
    }
    for scan in scans {
        let actual = u64::try_from(rows.iter().filter(|row| row.scan_id == *scan).count())
            .map_err(|_| CalibrationError::Overflow)?;
        if actual != SCAN_COUNTS[(*scan - 1) as usize] {
            return Err(CalibrationError::InvalidPartition);
        }
    }
    Ok(())
}

fn artifact_bytes(
    fit_id: &str,
    fit_scans: &[i64],
    held_scan: Option<i64>,
    row_manifest: &RowManifest,
    fit: CalibrationFit,
) -> Result<Vec<u8>, CalibrationError> {
    let scans = fit_scans
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let held = held_scan.map_or_else(|| "null".to_owned(), |scan| scan.to_string());
    let row_count =
        u64::try_from(row_manifest.rows.len()).map_err(|_| CalibrationError::Overflow)?;
    let json = format!(
        "{{\"schema\":\"solid.bern.calibration.v1\",\"fit_id\":\"{fit_id}\",\"fit_scan_ids\":[{scans}],\"held_scan_id\":{held},\"input_sha256\":\"{INPUT_SHA256}\",\"row_manifest_sha256\":\"{}\",\"row_count\":{row_count},\"range_slope_cm_per_ns_bits\":\"{}\",\"range_offset_cm_bits\":\"{}\",\"range_iterations\":{},\"range_final_residual_center_cm_bits\":\"{}\",\"range_final_mad_sigma_cm_bits\":\"{}\",\"scale_slope_bits\":\"{}\",\"scale_intercept_cm_bits\":\"{}\",\"scale_min_cm_bits\":\"4020000000000000\",\"scale_max_cm_bits\":\"404b800000000000\",\"runtime_tof_floor_ns_bits\":\"3ff0000000000000\",\"runtime_tof_max_ns_bits\":\"4069000000000000\",\"fitter_protocol_sha256\":\"{FITTER_PROTOCOL_SHA256}\"}}\n",
        row_manifest.bytes_sha256,
        float_bits(fit.range_slope_cm_per_ns),
        float_bits(fit.range_offset_cm),
        fit.range_iterations,
        float_bits(fit.range_final_residual_center_cm),
        float_bits(fit.range_final_mad_sigma_cm),
        float_bits(fit.scale_slope),
        float_bits(fit.scale_intercept_cm),
    );
    Ok(json.into_bytes())
}

fn manifest_bytes(receipts: &[FitReceipt]) -> Vec<u8> {
    let entries = receipts
        .iter()
        .map(|receipt| {
            let scans = receipt
                .fit_scan_ids
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let held = receipt
                .held_scan_id
                .map_or_else(|| "null".to_owned(), |scan| scan.to_string());
            format!(
                "{{\"fit_id\":\"{}\",\"fit_scan_ids\":[{scans}],\"held_scan_id\":{held},\"row_count\":{},\"artifact_path\":\"{}\",\"artifact_sha256\":\"{}\",\"row_manifest_path\":\"{}\",\"row_manifest_sha256\":\"{}\"}}",
                receipt.fit_id,
                receipt.row_count,
                receipt.artifact_path,
                receipt.artifact_sha256,
                receipt.row_manifest_path,
                receipt.row_manifest_sha256,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema\":\"solid.bern.calibration.manifest.v1\",\"fitter_protocol_sha256\":\"{FITTER_PROTOCOL_SHA256}\",\"input_sha256\":\"{INPUT_SHA256}\",\"fits\":[{entries}]}}\n"
    )
    .into_bytes()
}

fn preflight_outputs(paths: &[PathBuf]) -> Result<(), CalibrationError> {
    if paths
        .iter()
        .any(|path| path.exists() || pending_path(path).exists())
    {
        Err(CalibrationError::OutputExists)
    } else {
        Ok(())
    }
}

fn publish_all(directory: &Path, outputs: &[(PathBuf, Vec<u8>)]) -> Result<(), CalibrationError> {
    let paths = outputs
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    preflight_outputs(&paths)?;

    let mut pending_paths = Vec::with_capacity(outputs.len());
    for (path, bytes) in outputs {
        let pending = pending_path(path);
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)
        {
            Ok(file) => file,
            Err(error) => {
                cleanup_paths(&pending_paths);
                return Err(error.into());
            }
        };
        pending_paths.push(pending);
        let result = (|| {
            file.write_all(bytes)?;
            file.sync_all()?;
            Ok::<(), std::io::Error>(())
        })();
        if let Err(error) = result {
            cleanup_paths(&pending_paths);
            return Err(error.into());
        }
    }

    let mut published_paths = Vec::with_capacity(outputs.len());
    for ((path, _), pending) in outputs.iter().zip(&pending_paths) {
        if let Err(error) = fs::hard_link(pending, path) {
            cleanup_paths(&published_paths);
            cleanup_paths(&pending_paths);
            return Err(error.into());
        }
        published_paths.push(path.clone());
    }

    for pending in &pending_paths {
        if let Err(error) = fs::remove_file(pending) {
            cleanup_paths(&published_paths);
            cleanup_paths(&pending_paths);
            return Err(error.into());
        }
    }
    if let Err(error) = File::open(directory).and_then(|handle| handle.sync_all()) {
        cleanup_paths(&published_paths);
        return Err(error.into());
    }
    Ok(())
}

fn cleanup_paths(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn pending_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".pending");
    PathBuf::from(name)
}

#[derive(Clone, Copy, Debug, Default)]
struct Neumaier {
    sum: f64,
    correction: f64,
}

impl Neumaier {
    fn add(&mut self, value: f64) -> Result<(), CalibrationError> {
        let trial = finite(self.sum + value)?;
        let correction = if self.sum.abs() >= value.abs() {
            let difference = finite(self.sum - trial)?;
            finite(difference + value)?
        } else {
            let difference = finite(value - trial)?;
            finite(difference + self.sum)?
        };
        self.correction = finite(self.correction + correction)?;
        self.sum = trial;
        Ok(())
    }

    fn finish(self) -> Result<f64, CalibrationError> {
        finite(self.sum + self.correction)
    }
}

fn finite(value: f64) -> Result<f64, CalibrationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CalibrationError::NonfiniteArithmetic)
    }
}

fn positive_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn canonical_bits(value: f64) -> u64 {
    positive_zero(value).to_bits()
}

fn float_bits(value: f64) -> String {
    format!("{:016x}", canonical_bits(value))
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    use super::*;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let ordinal = NEXT_TEST_DIRECTORY.fetch_add(1, AtomicOrdering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "solid-bern-calibration-{}-{ordinal}",
                std::process::id(),
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn row(index: u64, x: f64, y: f64) -> CalibrationRow {
        CalibrationRow {
            source_ordinal: index,
            scan_id: 1,
            timestamp: i64::try_from(index).unwrap(),
            listening_period: 0,
            poller_id: 1,
            poller_pos_id: 1,
            responder_id: 2,
            responder_pos_id: 2,
            tof_ns: x,
            distance_cm: y,
        }
    }

    fn encode_rows(rows: &[CalibrationRow]) -> Vec<u8> {
        let mut bytes = ROW_HEADER.to_vec();
        bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
        for row in rows {
            bytes.extend_from_slice(&row.source_ordinal.to_le_bytes());
            for value in [
                row.scan_id,
                row.timestamp,
                row.listening_period,
                row.poller_id,
                row.poller_pos_id,
                row.responder_id,
                row.responder_pos_id,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes.extend_from_slice(&canonical_bits(row.tof_ns).to_le_bytes());
            bytes.extend_from_slice(&canonical_bits(row.distance_cm).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn row_manifest_round_trip_is_exact() {
        let rows = vec![row(0, -0.0, 1.0), row(1, 1.0, 3.0)];
        let bytes = encode_rows(&rows);
        let decoded = parse_row_manifest(&bytes).unwrap();
        assert_eq!(decoded.rows[0].tof_ns.to_bits(), 0);
        assert_eq!(decoded.rows[1], rows[1]);
        assert_eq!(decoded.bytes_sha256, hex_digest(&bytes));
    }

    #[test]
    fn malformed_length_order_and_nonfinite_rows_are_rejected() {
        let mut bytes = encode_rows(&[row(1, 1.0, 3.0), row(0, 0.0, 1.0)]);
        assert!(matches!(
            parse_row_manifest(&bytes),
            Err(CalibrationError::InvalidRow)
        ));
        bytes.truncate(bytes.len() - 1);
        assert!(matches!(
            parse_row_manifest(&bytes),
            Err(CalibrationError::InvalidRowManifest)
        ));
        let invalid = row(0, f64::NAN, 1.0);
        assert!(matches!(
            parse_row_manifest(&encode_rows(&[invalid])),
            Err(CalibrationError::InvalidRow)
        ));

        let duplicate_ordinal = vec![row(0, 0.0, 1.0), row(0, 1.0, 3.0)];
        assert!(matches!(
            parse_row_manifest(&encode_rows(&duplicate_ordinal)),
            Err(CalibrationError::InvalidRow)
        ));
    }

    #[test]
    fn row_manifest_rejects_noncanonical_negative_zero_bits() {
        const PREFIX_BYTES: usize = 39;
        const TOF_OFFSET: usize = PREFIX_BYTES + 64;
        const DISTANCE_OFFSET: usize = PREFIX_BYTES + 72;
        for offset in [TOF_OFFSET, DISTANCE_OFFSET] {
            let mut bytes = encode_rows(&[row(0, 0.0, 0.0)]);
            bytes[offset..offset + 8].copy_from_slice(&(-0.0_f64).to_bits().to_le_bytes());
            assert!(matches!(
                parse_row_manifest(&bytes),
                Err(CalibrationError::InvalidRow)
            ));
        }
    }

    #[test]
    fn median_handles_odd_even_duplicates_and_signed_zero() {
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
        assert_eq!(median(&[2.0, 2.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(&[-0.0, 0.0]).unwrap().to_bits(), 0);
    }

    #[test]
    fn weighted_affine_recovers_an_exact_line() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(weighted_affine(&x, &y, &[1.0; 4]).unwrap(), (2.0, 1.0));
        assert!(weighted_affine(&[1.0, 1.0], &[2.0, 3.0], &[1.0; 2]).is_err());
    }

    #[test]
    fn weighted_affine_matches_the_frozen_cancellation_oracle() {
        let x = [0.1, 100_000_000.25, -99_999_999.5, 3.75];
        let y = [7.0, 300_000_001.0, -299_999_998.0, 12.25];
        let weights = [1.0e-8, 1.0, 1.0, 1.0e8];
        let (slope, offset) = weighted_affine(&x, &y, &weights).unwrap();
        assert_eq!(slope.to_bits(), 0x4007_ffff_ffd5_0ce2);
        assert_eq!(offset.to_bits(), 0x3fef_ffff_fbce_4220);
    }

    #[test]
    fn gross_outlier_receives_the_frozen_one_mad_weight() {
        let (weights, center, sigma) = one_mad_weights(&[0.0, 0.0, 0.0, 100.0]).unwrap();
        assert_eq!(center.to_bits(), 0);
        assert_eq!(sigma.to_bits(), MAD_FLOOR.to_bits());
        assert_eq!(&weights[..3], &[1.0, 1.0, 1.0]);
        assert_eq!(weights[3].to_bits(), (MAD_FLOOR / 100.0).to_bits());
    }

    #[test]
    fn runtime_conversion_uses_floor_distance_and_scale_clamps() {
        let fit = CalibrationFit {
            range_slope_cm_per_ns: 2.0,
            range_offset_cm: -3.0,
            range_iterations: 1,
            range_final_residual_center_cm: 0.0,
            range_final_mad_sigma_cm: 1.0,
            scale_slope: 10.0,
            scale_intercept_cm: 0.0,
        };
        assert_eq!(
            convert_sample(fit, 0.0).unwrap(),
            CalibratedSample {
                distance_cm: 0.0,
                scale_cm: 8.0
            }
        );
        assert_eq!(
            convert_sample(fit, 2.0).unwrap(),
            CalibratedSample {
                distance_cm: 1.0,
                scale_cm: 10.0
            }
        );
        assert_eq!(convert_sample(fit, 10.0).unwrap().scale_cm, 55.0);
        assert!(convert_sample(fit, 201.0).is_err());
        assert_eq!(scale_and_derivative(fit, 0.8).unwrap(), (8.0, 0.0));
        assert_eq!(scale_and_derivative(fit, 1.0).unwrap(), (10.0, 10.0));
        assert_eq!(scale_and_derivative(fit, 5.5).unwrap(), (55.0, 0.0));
        assert_eq!(convert_sample(fit, -0.0).unwrap().distance_cm.to_bits(), 0);
        assert!(convert_sample(fit, 200.0).is_ok());
        assert!(convert_sample(fit, f64::from_bits(200.0_f64.to_bits() + 1)).is_err());
    }

    #[test]
    fn equal_count_bins_assign_remainders_to_the_first_bins() {
        assert_eq!(bin_lengths(100).unwrap(), [1; 100]);
        let lengths = bin_lengths(103).unwrap();
        assert_eq!(&lengths[..3], &[2, 2, 2]);
        assert!(lengths[3..].iter().all(|length| *length == 1));
        assert!(bin_lengths(99).is_err());
    }

    #[test]
    fn scale_fit_matches_the_frozen_coefficient_oracle() {
        let mut rows = Vec::with_capacity(200);
        for bin in 0..100_u64 {
            let distance = 200.0 + bin as f64;
            let deviation = 2.0 + 0.005 * distance;
            rows.push(row(2 * bin, distance - deviation, distance));
            rows.push(row(2 * bin + 1, distance + deviation, distance));
        }
        let (slope, offset) = scale_fit(&rows, 1.0, 0.0).unwrap();
        assert_eq!(slope.to_bits(), 0x3f7e_5d18_090b_41b0);
        assert_eq!(offset.to_bits(), 0x4007_b8ba_c710_cb10);
    }

    #[test]
    fn normative_constant_bits_are_exact() {
        assert_eq!(MAD_MULTIPLIER.to_bits(), 0x3ff7_b8ba_c710_cb29);
        assert_eq!(MAD_FLOOR.to_bits(), 0x3d71_9799_812d_ea11);
        assert_eq!(IRLS_TOLERANCE.to_bits(), 0x3ddb_7cdf_d9d7_bdbb);
        assert_eq!(SCALE_MIN_CM.to_bits(), 0x4020_0000_0000_0000);
        assert_eq!(SCALE_MAX_CM.to_bits(), 0x404b_8000_0000_0000);
        assert_eq!(RUNTIME_TOF_MAX_NS.to_bits(), 0x4069_0000_0000_0000);
    }

    #[test]
    fn calibration_artifact_encoding_has_exact_key_order_and_bits() {
        let row_manifest = RowManifest {
            bytes_sha256: "00".repeat(32),
            rows: vec![row(0, 1.0, 2.0)],
        };
        let fit = CalibrationFit {
            range_slope_cm_per_ns: 2.0,
            range_offset_cm: -3.0,
            range_iterations: 4,
            range_final_residual_center_cm: 0.0,
            range_final_mad_sigma_cm: 1.0,
            scale_slope: 0.5,
            scale_intercept_cm: 7.0,
        };
        let encoded = artifact_bytes("full-training", &[1, 2], None, &row_manifest, fit).unwrap();
        let expected = format!(
            "{{\"schema\":\"solid.bern.calibration.v1\",\"fit_id\":\"full-training\",\"fit_scan_ids\":[1,2],\"held_scan_id\":null,\"input_sha256\":\"{INPUT_SHA256}\",\"row_manifest_sha256\":\"{}\",\"row_count\":1,\"range_slope_cm_per_ns_bits\":\"4000000000000000\",\"range_offset_cm_bits\":\"c008000000000000\",\"range_iterations\":4,\"range_final_residual_center_cm_bits\":\"0000000000000000\",\"range_final_mad_sigma_cm_bits\":\"3ff0000000000000\",\"scale_slope_bits\":\"3fe0000000000000\",\"scale_intercept_cm_bits\":\"401c000000000000\",\"scale_min_cm_bits\":\"4020000000000000\",\"scale_max_cm_bits\":\"404b800000000000\",\"runtime_tof_floor_ns_bits\":\"3ff0000000000000\",\"runtime_tof_max_ns_bits\":\"4069000000000000\",\"fitter_protocol_sha256\":\"{FITTER_PROTOCOL_SHA256}\"}}\n",
            "00".repeat(32),
        );
        assert_eq!(encoded, expected.into_bytes());
    }

    #[test]
    fn calibration_manifest_encoding_has_exact_key_order() {
        let receipt = FitReceipt {
            fit_id: "outer-holdout-scan-01".to_owned(),
            fit_scan_ids: vec![2, 3],
            held_scan_id: Some(1),
            row_count: 7,
            artifact_path: "outer-holdout-scan-01.json".to_owned(),
            artifact_sha256: "11".repeat(32),
            row_manifest_path: "outer-holdout-scan-01.rows.bin".to_owned(),
            row_manifest_sha256: "22".repeat(32),
        };
        let bytes = manifest_bytes(&[receipt]);
        let expected = format!(
            "{{\"schema\":\"solid.bern.calibration.manifest.v1\",\"fitter_protocol_sha256\":\"{FITTER_PROTOCOL_SHA256}\",\"input_sha256\":\"{INPUT_SHA256}\",\"fits\":[{{\"fit_id\":\"outer-holdout-scan-01\",\"fit_scan_ids\":[2,3],\"held_scan_id\":1,\"row_count\":7,\"artifact_path\":\"outer-holdout-scan-01.json\",\"artifact_sha256\":\"{}\",\"row_manifest_path\":\"outer-holdout-scan-01.rows.bin\",\"row_manifest_sha256\":\"{}\"}}]}}\n",
            "11".repeat(32),
            "22".repeat(32),
        );
        assert_eq!(bytes, expected.into_bytes());
    }

    #[test]
    fn late_output_or_pending_file_prevents_every_publication() {
        for blocked_path in ["outer-holdout-scan-11.json", "full-training.json.pending"] {
            let directory = TestDirectory::new();
            fs::write(directory.path().join(blocked_path), b"occupied").unwrap();
            assert!(matches!(
                fit_all(directory.path()),
                Err(CalibrationError::OutputExists)
            ));
            for held_scan in TRAINING_SCANS.into_iter().map(Some).chain([None]) {
                let path = directory.path().join(format!("{}.json", fit_id(held_scan)));
                if path.file_name().unwrap() != "outer-holdout-scan-11.json" {
                    assert!(!path.exists());
                }
            }
            assert!(!directory.path().join("manifest.json").exists());
        }
    }

    #[test]
    fn multi_file_publication_is_exact_and_rejects_overwrite() {
        let directory = TestDirectory::new();
        let outputs = [
            (directory.path().join("first.json"), b"first\n".to_vec()),
            (directory.path().join("second.json"), b"second\n".to_vec()),
        ];
        publish_all(directory.path(), &outputs).unwrap();
        assert_eq!(fs::read(&outputs[0].0).unwrap(), outputs[0].1);
        assert_eq!(fs::read(&outputs[1].0).unwrap(), outputs[1].1);
        assert!(!pending_path(&outputs[0].0).exists());
        assert!(!pending_path(&outputs[1].0).exists());
        assert!(matches!(
            publish_all(directory.path(), &outputs),
            Err(CalibrationError::OutputExists)
        ));
    }

    #[test]
    fn zero_iteration_limit_proves_nonconvergence_is_an_error() {
        let rows = (0..100)
            .map(|index| row(index, index as f64, 2.0 * index as f64 + 1.0))
            .collect::<Vec<_>>();
        assert!(matches!(
            fit_calibration_with_limit(&rows, 0),
            Err(CalibrationError::IrlsNonconvergence)
        ));
    }

    #[test]
    fn exact_line_fit_and_required_failure_modes_are_deterministic() {
        let exact = (0..100)
            .map(|index| row(index, index as f64, 2.0 * index as f64 + 1.0))
            .collect::<Vec<_>>();
        let fit = fit_calibration(&exact).unwrap();
        assert_eq!(fit.range_slope_cm_per_ns, 2.0);
        assert_eq!(fit.range_offset_cm, 1.0);

        let descending = (0..100)
            .map(|index| row(index, index as f64, 200.0 - index as f64))
            .collect::<Vec<_>>();
        assert!(matches!(
            fit_calibration(&descending),
            Err(CalibrationError::NonpositiveRangeSlope)
        ));

        let constant_distance = (0..100)
            .map(|index| row(index, index as f64, 10.0))
            .collect::<Vec<_>>();
        assert!(matches!(
            scale_fit(&constant_distance, 1.0, 0.0),
            Err(CalibrationError::InvalidAffineInput)
        ));
    }
}
