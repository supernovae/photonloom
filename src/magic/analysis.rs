//! Deterministic measurements on a bounded, plane-contiguous Magic preview.
//! All coordinates and mask values refer to this preview, not the original science frame.
use crate::config::LinearImage;
use crate::register::{detect_stars, robust_median_mad, DetectedStar};
use serde::{Deserialize, Serialize};

const VERSION: u32 = 1;
const SAMPLE_LIMIT: usize = 131_072;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    pub count: usize,
    /// Bins span the observed finite range [min, max]; a constant input uses bin zero.
    pub histogram: Vec<u64>,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub p01: Option<f32>,
    pub median: Option<f32>,
    pub p99: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackgroundStats {
    pub sky_level: Option<f32>,
    pub spatial_spread: Option<f32>,
    pub sky_fraction: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoiseStats {
    pub sky_mad_sigma: Option<f32>,
    pub high_pass_sigma: Option<f32>,
    pub combined_sigma: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClippingStats {
    /// Display-domain values at 0/1, where the transfer actually defines those endpoints.
    pub shadow_fraction: Option<f32>,
    pub highlight_fraction: Option<f32>,
    pub per_channel_shadow: Vec<Option<f32>>,
    pub per_channel_highlight: Vec<Option<f32>>,
    pub clipped_star_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StarStats {
    /// Sorted in deterministic descending flux order, limited by register::detect_stars.
    pub unsaturated: Vec<DetectedStar>,
    pub saturated_centers: Vec<[f32; 2]>,
    pub median_fwhm: Option<f32>,
    pub median_eccentricity: Option<f32>,
    pub fwhm_confidence: f32,
    pub original_to_preview_scale: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorStats {
    pub median_rgb: Option<[f32; 3]>,
    pub median_chroma: Option<f32>,
    pub signal_chroma: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructureStats {
    pub significant_fraction: Option<f32>,
    pub core_fraction: Option<f32>,
    pub edge_fraction: Option<f32>,
    /// No morphology or subject class is inferred from image statistics alone.
    pub subject: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalysisMetadata {
    pub exposure_seconds: Option<f32>,
    pub filter: Option<String>,
    pub camera: Option<String>,
    pub binning: Option<[u32; 2]>,
    pub gain: Option<f32>,
    pub wcs: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaskReferences {
    pub version: u32,
    /// Relative paths only; populated when the session persists bounded debug masks.
    pub stars: Option<String>,
    pub star_halos: Option<String>,
    pub sky: Option<String>,
    pub signal: Option<String>,
    pub core: Option<String>,
    pub low_snr: Option<String>,
    pub edges: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAnalysis {
    pub version: u32,
    pub domain: super::InputDomain,
    pub width: usize,
    pub height: usize,
    pub valid_fraction: f32,
    pub channel_stats: Vec<Distribution>,
    pub luminance_stats: Distribution,
    pub background: BackgroundStats,
    pub noise: NoiseStats,
    pub noise_sigma: Option<f32>,
    pub signal_snr: Option<f32>,
    pub dynamic_range: Option<f32>,
    pub clipping: ClippingStats,
    pub stars: StarStats,
    pub star_fwhm: Option<f32>,
    /// Median absolute local contrast at radii 1, 2, 4 pixels, respectively.
    pub multiscale_contrast: [Option<f32>; 3],
    pub color_stats: ColorStats,
    pub structure: StructureStats,
    pub metadata: AnalysisMetadata,
    pub measurement_confidence: f32,
    pub confidence: f32,
    pub unavailable: Vec<String>,
    pub masks: MaskReferences,
}

#[derive(Debug, Clone)]
pub struct MaskAtlas {
    pub width: usize,
    pub height: usize,
    pub stars: Vec<f32>,
    pub star_halos: Vec<f32>,
    pub sky: Vec<f32>,
    pub signal: Vec<f32>,
    pub core: Vec<f32>,
    pub low_snr: Vec<f32>,
    pub edges: Vec<f32>,
}

impl MaskAtlas {
    fn new(width: usize, height: usize) -> Self {
        let zeros = || vec![0.0; width * height];
        Self { width, height, stars: zeros(), star_halos: zeros(), sky: zeros(), signal: zeros(), core: zeros(), low_snr: zeros(), edges: zeros() }
    }
}

fn quantile(sorted: &[f32], q: f32) -> Option<f32> {
    if sorted.is_empty() { return None; }
    let pos = q * (sorted.len() - 1) as f32;
    let a = pos.floor() as usize;
    let b = (a + 1).min(sorted.len() - 1);
    Some(sorted[a] + (sorted[b] - sorted[a]) * pos.fract())
}

fn median(mut values: Vec<f32>) -> Option<f32> {
    values.sort_by(f32::total_cmp);
    quantile(&values, 0.5)
}

fn sigma(values: &[f32]) -> Option<f32> {
    if values.len() < 16 { return None; }
    let mut v = values.to_vec();
    v.sort_by(f32::total_cmp);
    let center = quantile(&v, 0.5)?;
    for x in &mut v { *x = (*x - center).abs(); }
    v.sort_by(f32::total_cmp);
    let estimate = quantile(&v, 0.5)? * 1.4826;
    (estimate.is_finite() && estimate > 1e-8).then_some(estimate)
}

fn distribution(values: &[f32], valid: &[u8]) -> Distribution {
    let mut sample = Vec::new();
    let mut count = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let step = values.len().div_ceil(SAMPLE_LIMIT).max(1);
    for (i, &value) in values.iter().enumerate() {
        if valid.get(i).copied().unwrap_or(0) == 0 || !value.is_finite() { continue; }
        count += 1;
        min = min.min(value);
        max = max.max(value);
        if (count - 1) % step == 0 { sample.push(value); }
    }
    let mut histogram = vec![0u64; 256];
    if count != 0 {
        for (i, &v) in values.iter().enumerate() {
            if valid[i] == 0 { continue; }
            let bin = if max > min { (((v - min) / (max - min) * 255.0).clamp(0.0, 255.0)) as usize } else { 0 };
            histogram[bin] += 1;
        }
    }
    sample.sort_by(f32::total_cmp);
    Distribution { count, histogram, min: (count > 0).then_some(min), max: (count > 0).then_some(max), p01: quantile(&sample, 0.01), median: quantile(&sample, 0.5), p99: quantile(&sample, 0.99) }
}

fn smooth_at(lum: &[f32], valid: &[u8], w: usize, h: usize, x: usize, y: usize, radius: usize) -> f32 {
    let mut sum = 0.0;
    let mut weight = 0.0;
    let r = radius as isize;
    for dy in -r..=r {
        let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
        for dx in -r..=r {
            let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
            let i = yy * w + xx;
            if valid[i] == 0 { continue; }
            // Separable triangular kernel, reflected at the image boundary.
            let k = (radius + 1 - dx.unsigned_abs()) * (radius + 1 - dy.unsigned_abs());
            sum += lum[i] * k as f32;
            weight += k as f32;
        }
    }
    if weight == 0.0 { lum[y * w + x] } else { sum / weight }
}

/// Analyze a pre-sized preview. Invalid or nonfinite RGB components exclude the entire pixel.
/// The caller provides trusted FITS metadata separately; this routine never invents WCS or identity.
pub fn analyze_image(image: &LinearImage, domain: super::InputDomain) -> (ImageAnalysis, MaskAtlas) {
    let w = image.width;
    let h = image.height;
    let n = w.checked_mul(h).unwrap_or(0);
    let mut atlas = MaskAtlas::new(w, h);
    let mut valid = vec![0u8; n];
    let mut luminance = vec![0.0f32; n];
    let planes: Vec<&[f32]> = (0..image.channels.min(3))
        .filter_map(|c| image.data.get(c * n..(c + 1) * n)).collect();
    let rgb = planes.len() == 3;
    let supported = rgb || planes.len() == 1;
    for i in 0..n {
        if !supported || image.mask.get(i).copied().unwrap_or(0) == 0 || planes.iter().any(|p| !p[i].is_finite()) { continue; }
        let l = if rgb { 0.2126 * planes[0][i] + 0.7152 * planes[1][i] + 0.0722 * planes[2][i] } else { planes[0][i] };
        if l.is_finite() { luminance[i] = l; valid[i] = 1; }
    }
    let valid_count = valid.iter().filter(|&&x| x != 0).count();
    let channel_stats: Vec<_> = planes.iter().map(|p| distribution(p, &valid)).collect();
    let luminance_stats = distribution(&luminance, &valid);
    let (global_med, global_mad) = robust_median_mad(&luminance, &valid);
    let initial_sigma = (global_mad * 1.4826).max(0.0);
    let mut sky_values = Vec::new();
    let mut sky_count = 0usize;
    let sample_step = n.div_ceil(SAMPLE_LIMIT).max(1);
    if valid_count >= 16 && initial_sigma > 1e-8 {
        let cut = global_med + initial_sigma * 1.5;
        for i in 0..n {
            if valid[i] != 0 && luminance[i] <= cut {
                sky_count += 1;
                if (sky_count - 1) % sample_step == 0 { sky_values.push(luminance[i]); }
            }
        }
    }
    let sky_level = median(sky_values.clone());
    let sky_mad_sigma = sigma(&sky_values);
    let mut sky_residual = Vec::new();
    if let Some(sky) = sky_level {
        let upper = sky + 2.0 * sky_mad_sigma.unwrap_or(initial_sigma);
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if valid[i] == 0 || luminance[i] > upper || valid[i-1] == 0 || valid[i+1] == 0 || valid[i-w] == 0 || valid[i+w] == 0 { continue; }
                if i % sample_step == 0 {
                    sky_residual.push(luminance[i] - (luminance[i-1] + luminance[i+1] + luminance[i-w] + luminance[i+w]) * 0.25);
                }
            }
        }
    }
    // White-noise sigma of a pixel minus the average of four independent neighbors is sqrt(1.25) sigma.
    let high_pass_sigma = sigma(&sky_residual).map(|s| s / 1.25f32.sqrt());
    let noise_sigma = match (sky_mad_sigma, high_pass_sigma) {
        (Some(a), Some(b)) => Some(a.min(b.max(a * 0.25))),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        _ => None,
    };
    let noise = NoiseStats { sky_mad_sigma, high_pass_sigma, combined_sigma: noise_sigma };
    let mut tile_levels = Vec::new();
    if let (Some(sky), Some(ns)) = (sky_level, noise_sigma) {
        for ty in 0..8 {
            for tx in 0..8 {
                let mut tile = Vec::new();
                for y in (ty * h / 8)..((ty + 1) * h / 8) {
                    for x in (tx * w / 8)..((tx + 1) * w / 8) {
                        let i = y * w + x;
                        if valid[i] != 0 && luminance[i] < sky + 2.0 * ns { tile.push(luminance[i]); }
                    }
                }
                if tile.len() >= 16 { if let Some(v) = median(tile) { tile_levels.push(v); } }
            }
        }
    }
    let background = BackgroundStats { sky_level, spatial_spread: sigma(&tile_levels), sky_fraction: if valid_count == 0 { 0.0 } else { sky_count as f32 / valid_count as f32 } };

    let star_image = LinearImage { width: w, height: h, channels: 1, data: luminance.clone(), mask: valid.clone(), bayer: None, exposure: image.exposure };
    let mut detected = if noise_sigma.is_some() && w >= 9 && h >= 9 { detect_stars(&star_image) } else { Vec::new() };
    // The shared detector uses its own robust background and may regard a clipped peak as an
    // ordinary star on display-range inputs. Exclude those from all PSF estimates.
    let display_endpoints = matches!(&domain, super::InputDomain::Display);
    let linear_saturation = (!display_endpoints && luminance_stats.max.unwrap_or(0.0) > 60_000.0).then_some(62_000.0);
    detected.retain(|s| {
        let peak = s.peak + global_med;
        (!display_endpoints || peak < 1.0 - 1.0 / 65535.0)
            && linear_saturation.is_none_or(|limit| peak < limit)
    });
    let mut fwhms: Vec<f32> = detected.iter().filter(|s| s.fwhm.is_finite() && s.fwhm >= 1.0 && s.fwhm <= 12.0 && s.eccentricity < 0.7).map(|s| s.fwhm).collect();
    let ecc = median(detected.iter().map(|s| s.eccentricity).filter(|v| v.is_finite()).collect());
    let fwhm = if fwhms.len() >= 3 { median(std::mem::take(&mut fwhms)) } else { None };
    let fwhm_confidence = if fwhm.is_some() { (detected.len() as f32 / 15.0).min(1.0) * (1.0 - ecc.unwrap_or(1.0)).clamp(0.0, 1.0) } else { 0.0 };

    // Connected saturated plateaus: only high, locally outstanding blobs (not a clipped
    // background or a single hot pixel) count as stars. Scan top-to-bottom for stable IDs.
    let mut saturated_centers = Vec::new();
    if (display_endpoints || linear_saturation.is_some()) && w > 2 && h > 2 && noise_sigma.is_some() {
        let mut seen = vec![false; n];
        let threshold = if display_endpoints { 1.0 - 1.0 / 65535.0 } else { linear_saturation.unwrap_or(f32::INFINITY) };
        let sky = sky_level.unwrap_or(0.0);
        for i in 0..n {
            if seen[i] || valid[i] == 0 || luminance[i] < threshold || luminance[i] < sky + 5.0 * noise_sigma.unwrap_or(0.0) { continue; }
            let mut queue = vec![i];
            seen[i] = true;
            let mut pos = 0;
            let mut sx = 0usize;
            let mut sy = 0usize;
            while pos < queue.len() {
                let p = queue[pos]; pos += 1;
                sx += p % w; sy += p / w;
                let x = p % w; let y = p / w;
                for (nx, ny) in [(x.saturating_sub(1), y), ((x+1).min(w-1), y), (x, y.saturating_sub(1)), (x, (y+1).min(h-1))] {
                    let j = ny*w + nx;
                    if !seen[j] && valid[j] != 0 && luminance[j] >= threshold {
                        seen[j] = true; queue.push(j);
                    }
                }
            }
            if (2..=256).contains(&queue.len()) { saturated_centers.push([sx as f32 / queue.len() as f32, sy as f32 / queue.len() as f32]); }
        }
    }
    let clipping = if valid_count != 0 && display_endpoints {
        let mut shadow = Vec::new(); let mut highlight = Vec::new();
        for p in &planes {
            shadow.push(Some(p.iter().enumerate().filter(|(i, v)| valid[*i] != 0 && **v <= 0.0).count() as f32 / valid_count as f32));
            highlight.push(Some(p.iter().enumerate().filter(|(i, v)| valid[*i] != 0 && **v >= 1.0).count() as f32 / valid_count as f32));
        }
        ClippingStats { shadow_fraction: Some(shadow.iter().flatten().sum::<f32>() / planes.len() as f32), highlight_fraction: Some(highlight.iter().flatten().sum::<f32>() / planes.len() as f32), per_channel_shadow: shadow, per_channel_highlight: highlight, clipped_star_count: saturated_centers.len() }
    } else { ClippingStats { per_channel_shadow: vec![None; planes.len()], per_channel_highlight: vec![None; planes.len()], clipped_star_count: saturated_centers.len(), ..Default::default() } };
    let stars = StarStats { unsaturated: detected, saturated_centers, median_fwhm: fwhm, median_eccentricity: ecc, fwhm_confidence, original_to_preview_scale: None };

    let mut significant = Vec::new();
    let mut three_scale = [Vec::new(), Vec::new(), Vec::new()];
    let mut chromas = Vec::new(); let mut signal_chromas = Vec::new();
    let mut significant_count = 0usize; let mut core_count = 0usize; let mut edge_count = 0usize;
    let contrast_step = n.div_ceil(SAMPLE_LIMIT).max(1);
    if let (Some(sky), Some(ns)) = (sky_level, noise_sigma) {
        for y in 0..h {
            for x in 0..w {
                let i = y*w+x;
                if valid[i] == 0 { continue; }
                let z = ((luminance[i] - sky) / ns).max(0.0);
                let signal = ((z - 2.0) / 5.0).clamp(0.0, 1.0);
                atlas.signal[i] = signal;
                atlas.sky[i] = (1.0 - signal).clamp(0.0, 1.0);
                atlas.low_snr[i] = ((5.0 - z) / 5.0).clamp(0.0, 1.0);
                atlas.core[i] = ((z - 12.0) / 18.0).clamp(0.0, 1.0);
                if z > 3.0 { if i % contrast_step == 0 { significant.push(z); } significant_count += 1; }
                if z > 12.0 { core_count += 1; }
                if x > 3 && y > 3 && x+4 < w && y+4 < h {
                    let high = (luminance[i] - smooth_at(&luminance, &valid, w, h, x, y, 1)).abs();
                    atlas.edges[i] = ((high / ns - 1.0) / 6.0).clamp(0.0, 1.0) * signal;
                    if z > 3.0 && i % contrast_step == 0 {
                        three_scale[0].push(high);
                        for (scale, radius) in [(1, 2), (2, 4)] {
                            three_scale[scale].push((luminance[i] - smooth_at(&luminance, &valid, w, h, x, y, radius)).abs());
                        }
                    }
                }
                if atlas.edges[i] >= 0.5 { edge_count += 1; }
                if rgb {
                    let lo = planes.iter().map(|p| p[i]).fold(f32::INFINITY, f32::min);
                    let hi = planes.iter().map(|p| p[i]).fold(f32::NEG_INFINITY, f32::max);
                    let chroma = ((hi - lo) / luminance[i].abs().max(ns)).clamp(0.0, 10.0);
                    if i % contrast_step == 0 { chromas.push(chroma); if z > 3.0 { signal_chromas.push(chroma); } }
                }
            }
        }
        for s in &stars.unsaturated {
            paint_star(&mut atlas, &valid, s.x, s.y, s.fwhm.clamp(1.0, 12.0));
        }
        for &[x, y] in &stars.saturated_centers { paint_star(&mut atlas, &valid, x, y, fwhm.unwrap_or(3.0).clamp(2.0, 8.0)); }
        for i in 0..n {
            if valid[i] == 0 { continue; }
            let protection = atlas.star_halos[i];
            atlas.sky[i] *= 1.0 - protection;
            atlas.signal[i] *= 1.0 - atlas.stars[i];
            atlas.core[i] *= 1.0 - atlas.stars[i];
            atlas.edges[i] *= 1.0 - protection;
        }
    }
    let signal_snr = if significant.len() >= 16 { median(significant) } else { None };
    let contrast = three_scale.map(|v| if v.len() >= 16 { median(v) } else { None });
    let median_rgb = if rgb {
        match (channel_stats[0].median, channel_stats[1].median, channel_stats[2].median) {
            (Some(r), Some(g), Some(b)) => Some([r, g, b]),
            _ => None,
        }
    } else { None };
    let color_stats = ColorStats { median_rgb, median_chroma: median(chromas), signal_chroma: median(signal_chromas) };
    let structure = StructureStats { significant_fraction: (valid_count > 0).then_some(significant_count as f32 / valid_count.max(1) as f32), core_fraction: (valid_count > 0).then_some(core_count as f32 / valid_count.max(1) as f32), edge_fraction: (valid_count > 0).then_some(edge_count as f32 / valid_count.max(1) as f32), subject: None };
    let dynamic_range = match (luminance_stats.p99, sky_level, noise_sigma) { (Some(hi), Some(lo), Some(ns)) if hi > lo => Some((hi - lo) / ns), _ => None };
    let mut unavailable = Vec::new();
    if !supported { unavailable.push("unsupported channel layout".into()); }
    if valid_count < 16 { unavailable.push("insufficient finite, unmasked pixels".into()); }
    if noise_sigma.is_none() { unavailable.push("sky/noise undefined (flat or insufficient sky)".into()); }
    if signal_snr.is_none() { unavailable.push("significant structure SNR undefined".into()); }
    if fwhm.is_none() { unavailable.push("reliable unsaturated stellar FWHM unavailable; sharpening disabled".into()); }
    unavailable.push("subject classification unavailable without verified catalog evidence".into());
    let measurement_confidence = if signal_snr.is_some() && noise_sigma.is_some() {
        (valid_count as f32 / 4096.0).min(1.0)
            * (background.sky_fraction * 2.0).min(1.0)
            * (significant_count as f32 / 32.0).min(1.0)
            * (if fwhm.is_some() { 0.7 + 0.3 * fwhm_confidence } else { 0.35 })
    } else { 0.0 };
    let analysis = ImageAnalysis { version: VERSION, domain, width: w, height: h, valid_fraction: if n > 0 { valid_count as f32 / n as f32 } else { 0.0 }, channel_stats, luminance_stats, background, noise, noise_sigma, signal_snr, dynamic_range, clipping, stars, star_fwhm: fwhm, multiscale_contrast: contrast, color_stats, structure, metadata: AnalysisMetadata::default(), measurement_confidence, confidence: measurement_confidence, unavailable, masks: MaskReferences { version: VERSION, ..Default::default() } };
    (analysis, atlas)
}

fn paint_star(atlas: &mut MaskAtlas, valid: &[u8], x: f32, y: f32, fwhm: f32) {
    let sigma = (fwhm / 2.3548).max(0.5);
    let radius = (sigma * 6.0).ceil() as isize;
    let x0 = ((x as isize) - radius).max(0) as usize;
    let y0 = ((y as isize) - radius).max(0) as usize;
    let x1 = ((x as isize) + radius).min(atlas.width as isize - 1) as usize;
    let y1 = ((y as isize) + radius).min(atlas.height as isize - 1) as usize;
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            let i = yy * atlas.width + xx;
            if valid[i] == 0 { continue; }
            let r2 = (xx as f32 - x).powi(2) + (yy as f32 - y).powi(2);
            atlas.stars[i] = atlas.stars[i].max((-r2 / (2.0 * sigma * sigma)).exp());
            atlas.star_halos[i] = atlas.star_halos[i].max((-r2 / (18.0 * sigma * sigma)).exp());
        }
    }
}
