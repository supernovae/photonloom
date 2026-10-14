//! Display-only, bounded RGB finishing. Never apply this to stacked linear science frames.

use anyhow::{Result, bail};
use clap::ValueEnum;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{AcceleratorMode, LinearImage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum FinishPreset {
    #[default]
    Balanced,
    Neutral,
    Vivid,
    DarkNebula,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum StarMode {
    #[default]
    Standard,
    Starless,
    Reduced,
    StarsOnly,
}

impl StarMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Starless => "starless",
            Self::Reduced => "reduced",
            Self::StarsOnly => "stars-only",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FinishOptions {
    pub preset: FinishPreset,
    /// Display exposure in stops, from -4 to +4.
    pub exposure: Option<f32>,
    /// Midtone-pivot curve exponent (1 = identity).
    pub contrast: Option<f32>,
    /// Chrominance multiplier (1 = identity).
    pub saturation: Option<f32>,
    /// Conventional gamma (>1 brightens midtones).
    pub gamma: Option<f32>,
    /// Noise-gated dark detail strength (0 = disabled).
    pub local_contrast: Option<f32>,
    /// Star layer mode: standard, starless, reduced, or stars-only.
    #[serde(default)]
    pub star_mode: StarMode,
    /// Star screen-reblend weight in [0.0, 1.0] (0.0 = starless, 1.0 = full stars).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub star_blend: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResolvedFinish {
    pub preset: FinishPreset,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gamma: f32,
    pub local_contrast: f32,
    #[serde(default)]
    pub star_mode: StarMode,
    #[serde(default = "default_star_blend")]
    pub star_blend: f32,
}

const fn default_star_blend() -> f32 {
    1.0
}

impl FinishPreset {
    pub const fn options() -> [Self; 4] {
        [Self::Balanced, Self::Neutral, Self::Vivid, Self::DarkNebula]
    }

    pub const fn resolved_defaults(self) -> ResolvedFinish {
        let (contrast, saturation, gamma, local_contrast) = match self {
            Self::Neutral => (1.0, 1.0, 1.0, 0.0),
            Self::Balanced => (1.08, 1.08, 1.10, 0.0),
            Self::Vivid => (1.16, 1.18, 1.08, 0.0),
            Self::DarkNebula => (1.08, 1.08, 1.12, 0.18),
        };
        ResolvedFinish {
            preset: self,
            exposure: 0.0,
            contrast,
            saturation,
            gamma,
            local_contrast,
            star_mode: StarMode::Standard,
            star_blend: 1.0,
        }
    }
}

impl FinishOptions {
    pub fn resolve(&self) -> Result<ResolvedFinish> {
        let mut resolved = self.preset.resolved_defaults();
        for (name, value, min, max) in [
            ("exposure", self.exposure, -4.0, 4.0),
            ("contrast", self.contrast, 0.5, 1.8),
            ("saturation", self.saturation, 0.0, 1.8),
            ("gamma", self.gamma, 0.5, 2.0),
            ("local_contrast", self.local_contrast, 0.0, 0.3),
            ("star_blend", self.star_blend, 0.0, 1.0),
        ] {
            if let Some(v) = value {
                if !v.is_finite() || !(min..=max).contains(&v) {
                    bail!("{name} must be finite and between {min} and {max}");
                }
            }
        }
        resolved.exposure = self.exposure.unwrap_or(resolved.exposure);
        resolved.contrast = self.contrast.unwrap_or(resolved.contrast);
        resolved.saturation = self.saturation.unwrap_or(resolved.saturation);
        resolved.gamma = self.gamma.unwrap_or(resolved.gamma);
        resolved.local_contrast = self.local_contrast.unwrap_or(resolved.local_contrast);
        let (star_mode, star_blend) = match (self.star_mode, self.star_blend) {
            (StarMode::Starless, _) => (StarMode::Starless, 0.0),
            (StarMode::StarsOnly, _) => (StarMode::StarsOnly, 1.0),
            (StarMode::Reduced, Some(b)) if b <= 0.0 => (StarMode::Starless, 0.0),
            (StarMode::Reduced, Some(b)) => (StarMode::Reduced, b),
            (StarMode::Reduced, None) => (StarMode::Reduced, 0.45),
            (StarMode::Standard, Some(b)) if b <= 0.0 => (StarMode::Starless, 0.0),
            (StarMode::Standard, Some(b)) if b < 1.0 => (StarMode::Reduced, b),
            (StarMode::Standard, _) => (StarMode::Standard, 1.0),
        };
        resolved.star_mode = star_mode;
        resolved.star_blend = star_blend;
        Ok(resolved)
    }
}

#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

#[inline]
fn tone(y: f32, params: &ResolvedFinish, exposure: f32, inverse_gamma: f32) -> f32 {
    let x = y.clamp(0.0, 1.0);
    // Rational highlight shoulder keeps both endpoints fixed while exposure shifts midtones.
    let exposed = if exposure == 1.0 {
        x
    } else {
        exposure * x / (1.0 + (exposure - 1.0) * x)
    };
    let pivot = 0.18;
    let curved = if params.contrast == 1.0 {
        exposed
    } else if exposed <= pivot {
        pivot * (exposed / pivot).powf(params.contrast)
    } else {
        1.0 - (1.0 - pivot) * ((1.0 - exposed) / (1.0 - pivot)).powf(params.contrast)
    };
    if params.gamma == 1.0 {
        curved.clamp(0.0, 1.0)
    } else {
        curved.clamp(0.0, 1.0).powf(inverse_gamma)
    }
}

/// Alter only valid display RGB pixels. The original mask, exposure metadata and dimensions remain intact.
/// Neutral is an exact no-op, including pixels with out-of-gamut or masked payload values.
pub fn apply_finish(image: &mut LinearImage, options: &FinishOptions) -> Result<ResolvedFinish> {
    apply_finish_with_accelerator(image, options, AcceleratorMode::Auto)
}

/// Applies display finishing with optional GPU-offloaded star separation (`starless`, `reduced`, `stars-only`).
pub fn apply_finish_with_accelerator(
    image: &mut LinearImage,
    options: &FinishOptions,
    accelerator: AcceleratorMode,
) -> Result<ResolvedFinish> {
    let params = options.resolve()?;
    let npix = image.width.checked_mul(image.height);
    let expected = npix.and_then(|n| n.checked_mul(3));
    if image.channels != 3 || expected != Some(image.data.len()) || npix != Some(image.mask.len()) {
        bail!("display finishing requires a valid planar RGB image and mask");
    }
    let npix = npix.unwrap();
    match params.star_mode {
        StarMode::Standard => {
            apply_curves_in_place(image, &params, npix);
        }
        StarMode::Starless => {
            let sep = crate::stars::separate_stars(image, accelerator)?;
            image.data = sep.starless.data;
            apply_curves_in_place(image, &params, npix);
        }
        StarMode::Reduced => {
            let sep = crate::stars::separate_stars(image, accelerator)?;
            image.data = sep.starless.data;
            apply_curves_in_place(image, &params, npix);
            crate::stars::screen_blend_stars(image, &sep.stars, params.star_blend);
        }
        StarMode::StarsOnly => {
            let sep = crate::stars::separate_stars(image, accelerator)?;
            image.data = sep.stars.data;
            apply_curves_in_place(image, &params, npix);
        }
    }
    Ok(params)
}

fn apply_curves_in_place(image: &mut LinearImage, params: &ResolvedFinish, npix: usize) {
    if params.exposure == 0.0
        && params.contrast == 1.0
        && params.saturation == 1.0
        && params.gamma == 1.0
        && params.local_contrast == 0.0
    {
        return;
    }
    let exposure = 2.0f32.powf(params.exposure);
    let inverse_gamma = 1.0 / params.gamma;
    // Only the optional local operation needs a luminance buffer; no RGB image copies.
    let local_luma = if params.local_contrast > 0.0 && npix > 0 {
        Some(local_dark_correction(image, params.local_contrast))
    } else {
        None
    };
    let (red, rest) = image.data.split_at_mut(npix);
    let (green, blue) = rest.split_at_mut(npix);
    red.par_iter_mut()
        .zip(green.par_iter_mut())
        .zip(blue.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((r, g), b))| {
            if image.mask[i] == 0 || !r.is_finite() || !g.is_finite() || !b.is_finite() {
                return;
            }
            let y = luminance(*r, *g, *b);
            if y <= 0.0 {
                return;
            }
            let input = y.clamp(0.0, 1.0);
            let local = local_luma.as_ref().map_or(0.0, |correction| correction[i]);
            let target = (tone(input, params, exposure, inverse_gamma) - local).clamp(0.0, 1.0);
            // Keep the original hue: adjust luminance once, scale chroma uniformly,
            // then shrink chroma as a group if any channel reaches the RGB gamut edge.
            let chroma_scale = params.saturation * (target / y);
            let delta = [
                (*r - y) * chroma_scale,
                (*g - y) * chroma_scale,
                (*b - y) * chroma_scale,
            ];
            let mut gamut = 1.0f32;
            for &d in &delta {
                if d > 0.0 {
                    gamut = gamut.min((1.0 - target) / d);
                } else if d < 0.0 {
                    gamut = gamut.min(-target / d);
                }
            }
            *r = (target + gamut * delta[0]).clamp(0.0, 1.0);
            *g = (target + gamut * delta[1]).clamp(0.0, 1.0);
            *b = (target + gamut * delta[2]).clamp(0.0, 1.0);
        });
    // curves applied in-place
}

/// Local means are computed using two running-sum box passes (O(pixels)), with
/// masked pixels excluded and bright stars Winsorized before they enter the mean.
/// Adjacent-pixel residuals estimate the noise floor; depressions below it stay untouched.
fn local_dark_correction(image: &LinearImage, strength: f32) -> Vec<f32> {
    let w = image.width;
    let h = image.height;
    let n = w * h;
    let (r, rest) = image.data.split_at(n);
    let (g, b) = rest.split_at(n);
    let lum: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| luminance(r[i], g[i], b[i]).clamp(0.0, 1.0))
        .collect();
    let step = (n / 131_072).max(1);
    let mut sample: Vec<f32> = (0..n)
        .step_by(step)
        .filter(|&i| image.mask[i] != 0 && lum[i].is_finite())
        .map(|i| lum[i])
        .collect();
    if sample.is_empty() {
        return vec![0.0; n];
    }
    sample.sort_unstable_by(f32::total_cmp);
    let median = sample[sample.len() / 2];
    let mut residuals: Vec<f32> = (0..n)
        .step_by(step)
        .filter(|&i| i % w > 0 && image.mask[i] != 0 && image.mask[i - 1] != 0)
        .map(|i| (lum[i] - lum[i - 1]).abs())
        .filter(|v| v.is_finite())
        .collect();
    if residuals.is_empty() {
        return vec![0.0; n];
    }
    residuals.sort_unstable_by(f32::total_cmp);
    let noise = residuals[residuals.len() / 2] * (1.0 / (0.67449 * std::f32::consts::SQRT_2));
    // Star rejection uses robust global upper quantile rather than a fixed intensity threshold.
    let star_cap = sample[(sample.len() * 9 / 10).min(sample.len() - 1)].max(median + 3.0 * noise);
    let radius = (w.max(h) / 64).clamp(8, 256);
    let mut row_sum = vec![0.0f32; n];
    let mut row_count = vec![0.0f32; n];
    row_sum
        .par_chunks_mut(w)
        .zip(row_count.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (out_sum, out_count))| {
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            let row = y * w;
            for x in 0..=radius.min(w - 1) {
                if image.mask[row + x] != 0 && lum[row + x].is_finite() {
                    sum += lum[row + x].min(star_cap);
                    count += 1.0;
                }
            }
            for x in 0..w {
                if x > 0 && x + radius < w {
                    let add = x + radius;
                    if image.mask[row + add] != 0 && lum[row + add].is_finite() {
                        sum += lum[row + add].min(star_cap);
                        count += 1.0;
                    }
                }
                if x > radius {
                    let remove = x - radius - 1;
                    if image.mask[row + remove] != 0 && lum[row + remove].is_finite() {
                        sum -= lum[row + remove].min(star_cap);
                        count -= 1.0;
                    }
                }
                out_sum[x] = sum;
                out_count[x] = count;
            }
        });
    let mut correction = vec![0.0; n];
    let mut sums = vec![0.0f32; w];
    let mut counts = vec![0.0f32; w];
    for initial in 0..=radius.min(h - 1) {
        for x in 0..w {
            sums[x] += row_sum[initial * w + x];
            counts[x] += row_count[initial * w + x];
        }
    }
    for y in 0..h {
        let add = (y + radius < h).then_some((y + radius) * w);
        let remove = y.checked_sub(radius + 1).map(|prev| prev * w);
        for x in 0..w {
            if y > 0 {
                if let Some(next) = add {
                    sums[x] += row_sum[next + x];
                    counts[x] += row_count[next + x];
                }
            }
            if let Some(prev) = remove {
                sums[x] -= row_sum[prev + x];
                counts[x] -= row_count[prev + x];
            }
            let i = y * w + x;
            if image.mask[i] == 0 || !lum[i].is_finite() || counts[x] <= 0.0 {
                continue;
            }
            let depression = (sums[x] / counts[x] - lum[i]).max(0.0);
            // Smooth gate: noise-scale fluctuations in a flat sky cannot become dust.
            let significant = (depression - 3.0 * noise).max(0.0);
            let gate = significant / (significant + 2.0 * noise + 1e-6);
            correction[i] = (strength * significant * gate).min(0.3 * lum[i]);
        }
    }
    correction
}
