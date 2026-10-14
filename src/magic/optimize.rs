//! Deterministic, quality-first coordinate search over measured operator maxima.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{analysis::{ImageAnalysis, MaskAtlas}, operators::{self, OperatorParams}, validate::{self, QualityReport}};
use crate::config::LinearImage;

pub const SEARCH_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricGain {
    /// A missing baseline measurement disables the corresponding operator.
    pub provenance: String,
    pub baseline: Option<f32>,
    pub candidate: Option<f32>,
    pub delta_sigma: Option<f32>,
    pub bounded_gain: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTrial {
    /// Fractions of the caller's already measured, variant-scaled maxima.
    pub scales: OperatorParams,
    pub parameters: OperatorParams,
    pub quality: QualityReport,
    pub contrast: MetricGain,
    pub chroma: MetricGain,
    pub objective: f32,
    pub preference_nudge: f32,
    pub rejection_causes: Vec<String>,
    pub cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchReport {
    pub version: u32,
    pub cache_key: String,
    pub trials: Vec<SearchTrial>,
    pub winner: usize,
    pub winner_parameters: OperatorParams,
}

pub struct SearchResult {
    pub image: LinearImage,
    pub report: SearchReport,
}

fn components(p: OperatorParams) -> [f32; 4] { [p.tone, p.detail, p.sharpen, p.color] }
fn vector(v: [f32; 4]) -> OperatorParams { OperatorParams { tone: v[0], detail: v[1], sharpen: v[2], color: v[3] } }
fn luminance(image: &LinearImage, i: usize, n: usize) -> f32 {
    if image.channels == 1 { image.data[i] } else {
        (image.data[i] + image.data[n+i] + image.data[2*n+i]) / 3.0
    }
}

/// Paired differences avoid mistaking unchanged image texture for a measured gain.
fn measured_gain(reference: &LinearImage, candidate: &LinearImage, masks: &MaskAtlas, contrast: bool, available: bool) -> MetricGain {
    let provenance = if contrast { "significant-signal horizontal high-pass, paired standard error" } else { "significant-signal RGB channel spread, paired standard error" };
    let mut metric = MetricGain { provenance: provenance.into(), baseline: None, candidate: None, delta_sigma: None, bounded_gain: 0.0 };
    if !available { return metric; }
    let n = reference.width * reference.height;
    let mut count = 0u64;
    let (mut sum_base, mut sum_candidate, mut mean_delta, mut m2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    // A fixed sampling stride keeps the score bounded in work and reproducible.
    let stride = n.div_ceil(131_072).max(1);
    for i in (0..n).step_by(stride) {
        if reference.mask[i] == 0 || masks.signal[i] <= 0.6 || (contrast && i % reference.width == 0) { continue; }
        let (before, after) = if contrast {
            if reference.mask[i-1] == 0 { continue; }
            ((luminance(reference, i, n) - luminance(reference, i-1, n)).abs(),
             (luminance(candidate, i, n) - luminance(candidate, i-1, n)).abs())
        } else {
            let range = |image: &LinearImage| {
                let a = [image.data[i], image.data[n+i], image.data[2*n+i]];
                a.into_iter().fold(f32::NEG_INFINITY, f32::max) - a.into_iter().fold(f32::INFINITY, f32::min)
            };
            (range(reference), range(candidate))
        };
        if !before.is_finite() || !after.is_finite() { continue; }
        count += 1;
        let delta = (after - before) as f64;
        let shift = delta - mean_delta;
        mean_delta += shift / count as f64;
        m2 += shift * (delta - mean_delta);
        sum_base += before as f64;
        sum_candidate += after as f64;
    }
    if count < 16 { return metric; }
    let base = sum_base / count as f64;
    let current = sum_candidate / count as f64;
    metric.baseline = Some(base as f32);
    metric.candidate = Some(current as f32);
    let stderr = (m2 / (count - 1) as f64 / count as f64).sqrt();
    metric.delta_sigma = if stderr > 0.0 { Some((mean_delta / stderr).clamp(-f32::MAX as f64, f32::MAX as f64) as f32) } else if mean_delta > 0.0 { Some(f32::MAX) } else { Some(0.0) };
    // Never reward changes below three standard errors, or measurements without
    // a nonzero baseline. The normalized improvement cannot exceed unity.
    if base > 1e-8 && mean_delta > 0.0 && mean_delta >= 3.0 * stderr {
        metric.bounded_gain = (mean_delta / base).clamp(0.0, 1.0) as f32;
    }
    metric
}

fn better(new: &SearchTrial, old: &SearchTrial) -> bool {
    if !new.quality.passed { return false; }
    if !old.quality.passed { return true; }
    let a = new.objective + new.preference_nudge;
    let b = old.objective + old.preference_nudge;
    if (a - b).abs() > 1e-7 { return a > b; }
    // Prefer smaller overall changes, then lexicographic coordinate order.
    let n = components(new.scales);
    let o = components(old.scales);
    let sum_n: f32 = n.iter().sum();
    let sum_o: f32 = o.iter().sum();
    if (sum_n - sum_o).abs() > 1e-7 { return sum_n < sum_o; }
    n < o
}

/// `maximum` is already multiplied by recipe quality, variant and strength.
/// `cache_key` includes the parent's source digest, recipe/version and crop.
/// No persistent pixel cache is retained; duplicate parameter vectors within
/// this invocation reuse their evaluated trial and never consume another gate.
pub fn search(image: &LinearImage, analysis: &ImageAnalysis, masks: &MaskAtlas, maximum: OperatorParams, prior: Option<f32>, cache_key: &str) -> SearchResult {
    let available = [analysis.multiscale_contrast.iter().any(|x| x.is_some()),
        analysis.multiscale_contrast.iter().any(|x| x.is_some()),
        analysis.multiscale_contrast.iter().any(|x| x.is_some()) && analysis.star_fwhm.is_some() && analysis.stars.fwhm_confidence >= 0.35,
        image.channels == 3 && analysis.color_stats.signal_chroma.is_some()];
    let maximum = if maximum.is_valid() { components(maximum) } else { [0.0; 4] };
    let maximum: [f32; 4] = std::array::from_fn(|i| if available[i] { maximum[i] } else { 0.0 });
    let prior = prior.filter(|p| p.is_finite() && (0.0..=1.0).contains(p));
    let mut report = SearchReport { version: SEARCH_VERSION, cache_key: cache_key.into(),
        trials: Vec::with_capacity(21), winner: 0, winner_parameters: OperatorParams::default() };
    let mut cache = HashMap::<[u32; 4], usize>::new();
    let mut winner_image = image.clone();
    let mut winner_scales = [0.0; 4];
    // Baseline plus twelve fixed-order coordinate trials. The second pass
    // probes two neighbors per coordinate around the first-pass winner.
    let mut vectors = Vec::with_capacity(21);
    vectors.push([0.0; 4]);
    for axis in 0..4 { for level in [0.0, 0.5, 1.0] {
        let mut scales = [1.0; 4];
        scales[axis] = level;
        vectors.push(scales);
    }}
    for index in 0..21 {
        if index == 13 {
            for axis in 0..4 { for level in [0.0, 1.0] {
                let mut scales = winner_scales;
                scales[axis] = (winner_scales[axis] + level) * 0.5;
                vectors.push(scales);
            }}
        }
        let scales = vectors[index];
        let params = vector(std::array::from_fn(|i| maximum[i] * scales[i]));
        let key = components(params).map(f32::to_bits);
        if let Some(&original) = cache.get(&key) {
            let mut trial = report.trials[original].clone();
            trial.scales = vector(scales);
            trial.cached = true;
            report.trials.push(trial);
            continue;
        }
        let candidate = if params == OperatorParams::default() { image.clone() }
            else { operators::apply_candidate(image, analysis, masks, &params) };
        let quality = validate::validate_candidate(image, &candidate, analysis, masks);
        let (contrast, chroma) = if quality.passed {
            (measured_gain(image, &candidate, masks, true, available[0]),
             measured_gain(image, &candidate, masks, false, available[3]))
        } else {
            (MetricGain { provenance: "rejected by quality gate; contrast not scored".into(), baseline: None, candidate: None, delta_sigma: None, bounded_gain: 0.0 },
             MetricGain { provenance: "rejected by quality gate; chroma not scored".into(), baseline: None, candidate: None, delta_sigma: None, bounded_gain: 0.0 })
        };
        let objective = if quality.passed { 0.6 * contrast.bounded_gain + 0.4 * chroma.bounded_gain } else { 0.0 };
        // Preference only breaks near-equal safe measured outcomes; cannot turn
        // an unsafe candidate into a winner or override a >=0.001 objective gain.
        let preference_nudge = if quality.passed { prior.map_or(0.0, |p| {
            let distance = components(vector(scales)).iter().zip(maximum).filter(|(_, m)| *m > 0.0)
                .map(|(s, _)| (*s - p).abs()).sum::<f32>() / 4.0;
            0.0001 * (1.0 - distance)
        }) } else { 0.0 };
        let trial = SearchTrial { scales: vector(scales), parameters: params, rejection_causes: quality.reasons.clone(), quality,
            contrast, chroma, objective, preference_nudge, cached: false };
        let trial_index = report.trials.len();
        if report.trials.is_empty() || better(&trial, &report.trials[report.winner]) {
            report.winner = trial_index;
            report.winner_parameters = params;
            winner_scales = scales;
            winner_image = candidate;
        }
        cache.insert(key, trial_index);
        report.trials.push(trial);
    }
    SearchResult { image: winner_image, report }
}
