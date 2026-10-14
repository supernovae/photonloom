//! Bounded display-space operators. Every branch starts from the same baseline.
use crate::config::LinearImage;
use serde::{Deserialize, Serialize};
use super::analysis::{ImageAnalysis, MaskAtlas};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct OperatorParams {
    /// Actual (not nominal) maximum fractional midtone displacement, in [0, .12].
    pub tone: f32,
    /// Actual Laplacian band gain, in [0, .15].
    pub detail: f32,
    /// Actual noise-gated high-frequency gain, in [0, .08].
    pub sharpen: f32,
    /// Actual chroma gain above unity, in [0, .15].
    pub color: f32,
}

impl OperatorParams {
    pub fn is_valid(self) -> bool {
        [(self.tone, 0.12), (self.detail, 0.15), (self.sharpen, 0.08), (self.color, 0.15)]
            .into_iter().all(|(v, max)| v.is_finite() && (0.0..=max).contains(&v))
    }
}

#[inline]
fn weight(mask: &[f32], i: usize) -> f32 {
    mask.get(i).copied().filter(|v| v.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0)
}

#[inline]
fn lum(image: &LinearImage, i: usize, n: usize) -> f32 {
    if image.channels == 1 { image.data[i] } else {
        0.2126 * image.data[i] + 0.7152 * image.data[n+i] + 0.0722 * image.data[2*n+i]
    }
}

/// Endpoint-preserving asinh stretch (zero strength is bitwise neutral).
pub fn luminance_asinh(x: f32, strength: f32) -> f32 {
    if !x.is_finite() { return 0.0; }
    if !strength.is_finite() { return x.clamp(0.0, 1.0); }
    let x = x.clamp(0.0, 1.0);
    let s = strength.clamp(0.0, 1.0);
    if s == 0.0 { return x; }
    let shape = 4.0 * s;
    x + s * ((shape * x).asinh() / shape.asinh() - x)
}

/// Generalized hyperbolic-sine, normalized so black and white stay fixed.
/// A convex blend with the identity guarantees monotonicity for finite valid parameters.
pub fn luminance_ghs(x: f32, shape: f32, pivot: f32, strength: f32) -> f32 {
    if !x.is_finite() { return 0.0; }
    let x = x.clamp(0.0, 1.0);
    if !shape.is_finite() || !pivot.is_finite() || !strength.is_finite() { return x; }
    let s = strength.clamp(0.0, 1.0);
    let a = shape.clamp(0.0, 12.0);
    let p = pivot.clamp(0.01, 0.99);
    if s == 0.0 || a < 1e-4 { return x; }
    let lo = (-a * p).asinh();
    let hi = (a * (1.0 - p)).asinh();
    let mapped = ((a * (x - p)).asinh() - lo) / (hi - lo);
    (x + s * (mapped - x)).clamp(0.0, 1.0)
}

/// Positive, endpoint-fixed midtone curve. Limits faint-signal blackpoint shifts.
pub fn masked_monotonic_curve(x: f32, gain: f32, mask: f32) -> f32 {
    if !x.is_finite() { return 0.0; }
    let x = x.clamp(0.0, 1.0);
    if !gain.is_finite() || !mask.is_finite() { return x; }
    let a = gain.clamp(0.0, 0.12) * mask.clamp(0.0, 1.0) * 1.78;
    // Vanishes quadratically near x=0 to keep sky noise quiet while lifting
    // midtone nebulosity; strictly monotonic throughout [0, 1] for valid gains.
    let inv = 1.0 - x;
    (x + a * x * inv * (x / (x + 0.16)) * 5.4).clamp(0.0, 1.0)
}

#[inline]
fn reflected(pos: isize, size: usize) -> usize {
    if size <= 1 { return 0; }
    let period = 2 * (size - 1) as isize;
    let p = pos.rem_euclid(period);
    if p < size as isize { p as usize } else { (period-p) as usize }
}

/// Separable reflected Gaussian; invalid neighbors do not bleed into valid pixels.
fn gaussian(src: &[f32], valid: &[u8], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if w == 0 || h == 0 { return Vec::new(); }
    let radius = ((sigma * 3.0).ceil() as usize).clamp(1, 24);
    let kernel: Vec<f32> = (0..=radius).map(|j| (-0.5*(j as f32/sigma).powi(2)).exp()).collect();
    let mut horizontal = vec![0.0; src.len()];
    let mut counts = vec![0.0; src.len()];
    for y in 0..h { for x in 0..w {
        let i = y*w+x;
        for d in -(radius as isize)..=radius as isize {
            let j = y*w+reflected(x as isize+d,w);
            if valid[j] != 0 && src[j].is_finite() && (0.0..=1.0).contains(&src[j]) {
                let k = kernel[d.unsigned_abs()];
                horizontal[i] += k*src[j]; counts[i] += k;
            }
        }
    }}
    let mut out = vec![0.0; src.len()];
    for y in 0..h { for x in 0..w {
        let i = y*w+x;
        let mut sum = 0.0;
        let mut count = 0.0;
        for d in -(radius as isize)..=radius as isize {
            let j = reflected(y as isize+d,h)*w+x;
            let k = kernel[d.unsigned_abs()];
            sum += k*horizontal[j]; count += k*counts[j];
        }
        out[i] = if count > 0.0 { sum/count } else { src[i] };
    }}
    out
}

/// Three-scale Laplacian with reflected edges; strength is constrained by measured noise.
pub fn pyramid_detail(image: &LinearImage, masks: &MaskAtlas, noise: Option<f32>, fwhm: Option<f32>, gain: f32) -> Vec<f32> {
    let Some(n) = image.width.checked_mul(image.height) else { return Vec::new(); };
    if n == 0 || image.mask.len() != n || !matches!(image.channels, 1|3) || image.data.len() != n.saturating_mul(image.channels) || !gain.is_finite() || gain <= 0.0 || !noise.is_some_and(|v| v.is_finite() && v > 0.0) { return vec![0.0;n]; }
    let noise = noise.unwrap();
    let source: Vec<f32> = (0..n).map(|i| lum(image,i,n)).collect();
    let base = fwhm.filter(|v| v.is_finite() && (1.0..=32.0).contains(v)).unwrap_or(2.0);
    let mut previous = source.clone();
    let mut correction = vec![0.0;n];
    for (scale, band_gain) in [(0.5, 0.35), (1.2, 0.35), (2.8, 0.30)] {
        let blurred = gaussian(&previous,&image.mask,image.width,image.height,(base*scale).clamp(0.6,12.0));
        for i in 0..n {
            if image.mask[i] == 0 || !source[i].is_finite() { continue; }
            let band = previous[i]-blurred[i];
            let significance = ((band.abs()-0.75*noise)/(2.0*noise)).clamp(0.0,1.0);
            let protect = (1.0-weight(&masks.stars,i)).min(1.0-weight(&masks.star_halos,i))
                * (1.0-weight(&masks.sky,i)) * (1.0-weight(&masks.low_snr,i))
                * (1.0-weight(&masks.core,i));
            correction[i] += band*significance*protect*band_gain*gain.clamp(0.0,0.15)*4.5;
        }
        previous = blurred;
    }
    correction
}

/// High-frequency correction, unavailable without a trustworthy measured stellar PSF.
pub fn edge_noise_aware_sharpen(image: &LinearImage, masks: &MaskAtlas, noise: Option<f32>, fwhm: Option<f32>, gain: f32) -> Vec<f32> {
    let Some(n) = image.width.checked_mul(image.height) else { return Vec::new(); };
    let mut correction = vec![0.0;n];
    let Some(psf) = fwhm.filter(|v| v.is_finite() && (1.0..=32.0).contains(v)) else { return correction; };
    let Some(noise) = noise.filter(|v| v.is_finite() && *v > 0.0) else { return correction; };
    if image.mask.len() != n || image.data.len() != n.saturating_mul(image.channels) || !matches!(image.channels, 1|3) || !gain.is_finite() || gain <= 0.0 { return correction; }
    let source: Vec<f32> = (0..n).map(|i| lum(image,i,n)).collect();
    let blur = gaussian(&source,&image.mask,image.width,image.height,(psf/2.355).clamp(0.6,8.0));
    for i in 0..n {
        if image.mask[i] == 0 || !source[i].is_finite() { continue; }
        let residual = source[i]-blur[i];
        let significance = ((residual.abs()-1.2*noise)/(2.5*noise)).clamp(0.0,1.0);
        let protect = (1.0-weight(&masks.stars,i)).min(1.0-weight(&masks.star_halos,i))
            * (1.0-weight(&masks.core,i)) * (1.0-weight(&masks.sky,i))
            * (1.0-weight(&masks.low_snr,i));
        correction[i] = residual*significance*protect*gain.clamp(0.0,0.08)*3.5;
    }
    correction
}

/// Mix an adjustment with a bounded, feathered spatial weight.
pub fn apply_masks(original: f32, adjusted: f32, mask: f32) -> f32 {
    if !original.is_finite() || !adjusted.is_finite() || !mask.is_finite() { return original; }
    original + (adjusted-original)*mask.clamp(0.0,1.0)
}

/// Preserve chromatic hue by scaling all three offsets from luminance identically.
/// Common gamut compression avoids per-channel hue clipping.
pub fn weighted_color(rgb: [f32;3], target: f32, saturation: f32) -> [f32;3] {
    if !rgb.iter().all(|v| v.is_finite()) || !target.is_finite() || !saturation.is_finite() { return rgb; }
    let y = 0.2126*rgb[0]+0.7152*rgb[1]+0.0722*rgb[2];
    if !(0.0..=1.0).contains(&y) { return rgb; }
    let target = target.clamp(0.0,1.0);
    let chroma = saturation.clamp(0.0,1.65);
    let delta = [(rgb[0]-y)*chroma,(rgb[1]-y)*chroma,(rgb[2]-y)*chroma];
    let hi = if rgb.iter().all(|&v| v < 1.0) && target < 1.0 { (1.0 - 1e-5f32).max(target) } else { 1.0 };
    let lo = if rgb.iter().all(|&v| v > 0.0) && target > 0.0 { (1e-5f32).min(target) } else { 0.0 };
    let mut gamut: f32 = 1.0;
    for d in delta { if d > 0.0 { gamut = gamut.min((hi-target)/d); }
        else if d < 0.0 { gamut = gamut.min((lo-target)/d); } }
    gamut = gamut.clamp(0.0, 1.0);
    [target+gamut*delta[0],target+gamut*delta[1],target+gamut*delta[2]]
}

/// A valid neutral vector is an exact clone, including masked and nonfinite payloads.
/// Invalid parameters or atlas geometry fail closed to the unchanged baseline.
pub fn apply_candidate(image: &LinearImage, analysis: &ImageAnalysis, masks: &MaskAtlas, params: &OperatorParams) -> LinearImage {
    let mut output = image.clone();
    let Some(n) = image.width.checked_mul(image.height) else { return output; };
    if !params.is_valid() || !matches!(image.channels,1|3) || image.data.len()!=n.saturating_mul(image.channels)
        || image.mask.len()!=n || masks.width!=image.width || masks.height!=image.height
        || [&masks.stars,&masks.star_halos,&masks.sky,&masks.signal,&masks.core,&masks.low_snr,&masks.edges].iter().any(|m|m.len()!=n || m.iter().any(|v|!v.is_finite() || !(0.0..=1.0).contains(v)))
        || n==0 { return output; }
    if *params == OperatorParams::default() { return output; }
    let reliable_noise = analysis.noise_sigma.filter(|v|v.is_finite() && *v>0.0);
    let reliable_fwhm = analysis.star_fwhm.filter(|v|v.is_finite() && (1.0..=32.0).contains(v) && analysis.stars.fwhm_confidence.is_finite() && analysis.stars.fwhm_confidence >= 0.35);
    let detail = (params.detail>0.0 && reliable_noise.is_some())
        .then(|| pyramid_detail(image,masks,reliable_noise,reliable_fwhm,params.detail));
    let sharpen = (params.sharpen>0.0 && reliable_fwhm.is_some() && reliable_noise.is_some())
        .then(|| edge_noise_aware_sharpen(image,masks,reliable_noise,reliable_fwhm,params.sharpen));
    for i in 0..n {
        if image.mask[i]==0 || (0..image.channels).any(|c|!image.data[c*n+i].is_finite()) { continue; }
        let y = lum(image,i,n);
        if !(0.0..=1.0).contains(&y) { continue; }
        let star = weight(&masks.stars,i).max(weight(&masks.star_halos,i));
        let tone_mask = (1.0-star).powi(2)*(1.0-weight(&masks.core,i)*0.8);
        let tone = masked_monotonic_curve(y,params.tone,tone_mask);
        let change = (detail.as_ref().map_or(0.0,|v|v[i])+sharpen.as_ref().map_or(0.0,|v|v[i])).clamp(-0.08,0.08);
        let target = if change == 0.0 && tone == y { y } else {
            (tone+change).clamp(if y>0.0 { 1e-5 } else { 0.0 },if y<1.0 { 1.0-1e-5 } else {1.0})
        };
        if image.channels==1 { output.data[i]=target; continue; }
        let color_gate = (1.0-star)*(1.0-weight(&masks.low_snr,i))*weight(&masks.signal,i);
        let lum_boost = if target > y && y > 1e-3 { (target / y).powf(0.55).clamp(1.0, 1.35) } else { 1.0 };
        let saturation = lum_boost * (1.0 + params.color * 2.5 * color_gate);
        let rgb = [image.data[i],image.data[n+i],image.data[2*n+i]];
        let mapped = weighted_color(rgb,target,saturation);
        for c in 0..3 { output.data[c*n+i]=mapped[c]; }
    }
    output
}
