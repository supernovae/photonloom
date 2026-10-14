use crate::config::{
    CalibrationSummary, Channel, EditParameters, GalleryManifest, LinearImage, MixSpec,
    OmittedVariantRecord, ProcessOptions, ScienceArtifact, TargetGallery, VariantArtifact,
};
use crate::crop::{CropRect, crop_image};
use crate::finish_style::{FinishOptions, ResolvedFinish, apply_finish};
use crate::fits::{inspect_fits_header, read_fits_linear};
use crate::integrate::{TargetIntegrationOutput, build_science_artifacts};
use crate::register::{detect_stars, robust_median_mad};
use anyhow::{Context, Result, bail};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

/// Removes smooth sky background gradient from each plane of `img` using a star-masked 8x8 tile grid
/// and robust 2D quadratic polynomial surface fit.
pub fn remove_smooth_background(img: &LinearImage) -> LinearImage {
    let mut out = img.clone();
    remove_background_in_place(&mut out);
    out
}

fn remove_background_in_place(out: &mut LinearImage) {
    let coeffs = fit_background_coefficients(out);
    subtract_background_model(out, &coeffs, (0, 0), (out.width, out.height));
}

/// Robust full-image quadratic sky fit; coefficients are in normalized image
/// coordinates and can be applied without seams to native tiles.
pub fn fit_background_coefficients(img: &LinearImage) -> Vec<[f64; 6]> {
    let (w, h) = (img.width, img.height);
    let mut models = vec![[0.0; 6]; img.channels];
    if w < 8 || h < 8 { return models; }
    let gx_cnt = 8usize.min(w / 4).max(2);
    let gy_cnt = 8usize.min(h / 4).max(2);
    for (c, coeffs) in models.iter_mut().enumerate() {
        let src_plane = img.plane(c);
        let (global_med, global_mad) = robust_median_mad(src_plane, &img.mask);
        let global_sigma = (global_mad * 1.4826).max(1e-5);
        let star_clip = global_med + 2.5 * global_sigma;
        let mut samples = Vec::with_capacity(gx_cnt * gy_cnt);
        let mut tile_vals = Vec::new();
        for ty in 0..gy_cnt {
            let y0 = (ty * h) / gy_cnt;
            let y1 = ((ty + 1) * h) / gy_cnt;
            for tx in 0..gx_cnt {
                let x0 = (tx * w) / gx_cnt;
                let x1 = ((tx + 1) * w) / gx_cnt;
                tile_vals.clear();
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = y * w + x;
                        if img.mask[p] != 0 {
                            let v = src_plane[p];
                            if v.is_finite() && v <= star_clip { tile_vals.push(v); }
                        }
                    }
                }
                if tile_vals.len() >= 4 {
                    let middle = tile_vals.len() / 2;
                    tile_vals.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
                    let med = f64::from(tile_vals[tile_vals.len() / 2]);
                    let cx = (0.5 * ((x0 + x1) as f64) / (w as f64)) * 2.0 - 1.0;
                    let cy = (0.5 * ((y0 + y1) as f64) / (h as f64)) * 2.0 - 1.0;
                    samples.push((cx, cy, med));
                }
            }
        }
        *coeffs = fit_quadratic_2d(&samples, f64::from(global_med));
    }
    models
}

pub fn subtract_background_model(img: &mut LinearImage, coeffs: &[[f64; 6]],
    origin: (usize, usize), full: (usize, usize)) {
    if full.0 < 8 || full.1 < 8 { return; }
    let (w, h) = (img.width, img.height);
    let npix = w * h;
    for (c, &coeffs) in coeffs.iter().enumerate().take(img.channels) {
        let dst_plane = &mut img.data[c * npix..(c + 1) * npix];
        for y in 0..h {
            let v = ((origin.1 + y) as f64 + 0.5) / full.1 as f64 * 2.0 - 1.0;
            for x in 0..w {
                let p = y * w + x;
                if img.mask[p] == 0 { dst_plane[p] = 0.0; continue; }
                let u = ((origin.0 + x) as f64 + 0.5) / full.0 as f64 * 2.0 - 1.0;
                let bg = (coeffs[0] + coeffs[1] * u + coeffs[2] * v
                    + coeffs[3] * u * u + coeffs[4] * u * v + coeffs[5] * v * v) as f32;
                dst_plane[p] -= bg;
            }
        }
    }
}

fn fit_quadratic_2d(samples: &[(f64, f64, f64)], fallback_c0: f64) -> [f64; 6] {
    if samples.len() < 6 {
        return [fallback_c0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }

    let mut weights = vec![1.0f64; samples.len()];
    let mut coeffs = [fallback_c0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // 3 IRLS Huber iterations for robustness against bright nebulosity tiles
    for _iter in 0..3 {
        let mut ata = [[0.0f64; 7]; 6];
        for (&(u, v, z), &w) in samples.iter().zip(weights.iter()) {
            let basis = [1.0, u, v, u * u, u * v, v * v];
            for r in 0..6 {
                ata[r][6] += w * basis[r] * z;
                for c in 0..6 {
                    ata[r][c] += w * basis[r] * basis[c];
                }
            }
        }
        // Small ridge regularization on quadratic terms
        for i in 1..6 {
            ata[i][i] += 1e-4;
        }
        if let Some(sol) = solve_6x6(ata) {
            coeffs = sol;
        } else {
            return [fallback_c0, 0.0, 0.0, 0.0, 0.0, 0.0];
        }

        // Update Huber weights
        let mut res: Vec<f64> = samples
            .iter()
            .map(|&(u, v, z)| {
                let pred = coeffs[0]
                    + coeffs[1] * u
                    + coeffs[2] * v
                    + coeffs[3] * u * u
                    + coeffs[4] * u * v
                    + coeffs[5] * v * v;
                (z - pred).abs()
            })
            .collect();
        let mut sorted_res = res.clone();
        sorted_res.sort_by(|a, b| a.total_cmp(b));
        let scale = (sorted_res[sorted_res.len() / 2] * 1.4826).max(1e-4);
        for (w, r) in weights.iter_mut().zip(res.drain(..)) {
            *w = if r <= 1.5 * scale {
                1.0
            } else {
                (1.5 * scale / r).max(0.05)
            };
        }
    }

    coeffs
}

fn solve_6x6(mut m: [[f64; 7]; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let mut pivot = col;
        for row in (col + 1)..6 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        if m[pivot][col].abs() < 1e-11 {
            return None;
        }
        m.swap(col, pivot);
        let div = m[col][col];
        for c in col..7 {
            m[col][c] /= div;
        }
        for row in 0..6 {
            if row != col {
                let factor = m[row][col];
                for c in col..7 {
                    m[row][c] -= factor * m[col][c];
                }
            }
        }
    }
    Some([m[0][6], m[1][6], m[2][6], m[3][6], m[4][6], m[5][6]])
}

fn convolve_5x5(src: &[f32], w: usize, h: usize, kernel: &[[f32; 5]; 5]) -> Vec<f32> {
    let mut dst = vec![0.0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let yi = y as isize;
        for x in 0..w {
            let xi = x as isize;
            let mut acc = 0.0f32;
            for ky in 0..5 {
                let ny = (yi + ky as isize - 2).clamp(0, (h as isize) - 1) as usize;
                for kx in 0..5 {
                    let nx = (xi + kx as isize - 2).clamp(0, (w as isize) - 1) as usize;
                    acc += kernel[ky][kx] * src[ny * w + nx];
                }
            }
            row[x] = acc;
        }
    });
    dst
}

/// Bounded PSF-based Richardson-Lucy deconvolution on linear luminance with noise-aware early stop
/// and stellar-mask core protection.
pub fn deconvolve_luminance_rl(lum: &LinearImage) -> LinearImage {
    let w = lum.width;
    let h = lum.height;
    let npix = w * h;
    if w < 12 || h < 12 {
        return lum.clone();
    }

    let stars = detect_stars(lum);
    let mut fwhms: Vec<f32> = stars.iter().map(|s| s.fwhm).collect();
    fwhms.sort_by(|a, b| a.total_cmp(b));
    let med_fwhm = if fwhms.is_empty() {
        2.2
    } else {
        fwhms[fwhms.len() / 2].clamp(1.6, 4.5)
    };
    let sigma_psf = (med_fwhm / 2.3548).clamp(0.65, 1.6);

    let mut psf = [[0.0f32; 5]; 5];
    let mut psf_sum = 0.0f32;
    for ky in 0..5 {
        let dy = (ky as f32) - 2.0;
        for kx in 0..5 {
            let dx = (kx as f32) - 2.0;
            let val = (-(dx * dx + dy * dy) / (2.0 * sigma_psf * sigma_psf)).exp();
            psf[ky][kx] = val;
            psf_sum += val;
        }
    }
    for ky in 0..5 {
        for kx in 0..5 {
            psf[ky][kx] /= psf_sum;
        }
    }

    let src = lum.plane(0);
    let (med, mad) = robust_median_mad(src, &lum.mask);
    let sigma_bg = (mad * 1.4826).max(1e-4);

    // Shift to strictly positive domain for RL
    let shift = (-med + 5.0 * sigma_bg).max(1.0);
    let obs: Vec<f32> = src.iter().map(|&v| (v + shift).max(1e-4)).collect();

    // Build stellar core protection + noise gate weight per pixel
    let mut alpha = vec![0.0f32; npix];
    for p in 0..npix {
        if lum.mask[p] == 0 {
            continue;
        }
        let snr = ((src[p] - med) / sigma_bg).max(0.0);
        // Only deconvolve structures significantly above background noise
        let noise_gate = ((snr - 2.0) / 4.0).clamp(0.0, 1.0);
        alpha[p] = 0.35 * noise_gate;
    }
    // Protect bright star cores from ringing
    for s in &stars {
        let sx = s.x.round() as isize;
        let sy = s.y.round() as isize;
        for dy in -2isize..=2 {
            for dx in -2isize..=2 {
                let nx = sx + dx;
                let ny = sy + dy;
                if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                    let r2 = (dx * dx + dy * dy) as f32;
                    let core_damp = (r2 / 5.0).clamp(0.15, 1.0);
                    alpha[(ny as usize) * w + (nx as usize)] *= core_damp;
                }
            }
        }
    }

    let mut est = obs.clone();
    let mut prev_rms = f32::INFINITY;

    for _iter in 0..6 {
        let mut ratio = convolve_5x5(&est, w, h, &psf);
        for p in 0..npix {
            ratio[p] = (obs[p] / ratio[p].max(1e-5)).clamp(0.5, 2.0);
        }
        let corr = convolve_5x5(&ratio, w, h, &psf);
        drop(ratio);
        let mut step_sq_sum = 0.0f32;
        let mut active_cnt = 0usize;
        for p in 0..npix {
            let a = alpha[p];
            if a > 1e-3 {
                let next_v = est[p] * ((1.0 - a) + a * corr[p]);
                let diff = next_v - est[p];
                step_sq_sum += diff * diff;
                active_cnt += 1;
                est[p] = next_v;
            }
        }
        if active_cnt == 0 {
            break;
        }
        let step_rms = (step_sq_sum / active_cnt as f32).sqrt();
        // Noise-aware early stop
        if step_rms < 0.08 * sigma_bg || step_rms > 1.05 * prev_rms {
            break;
        }
        prev_rms = step_rms;
    }

    let mut out = LinearImage {
        width: w,
        height: h,
        channels: 1,
        data: est,
        mask: lum.mask.clone(),
        bayer: None,
        exposure: lum.exposure,
    };
    for p in 0..npix {
        if out.mask[p] != 0 {
            out.data[p] -= shift;
        }
    }
    out
}

/// Applies gentle 3x3 chrominance denoise while preserving full luminance detail.
pub fn gentle_chroma_denoise(rgb: &mut LinearImage) {
    if rgb.channels != 3 || rgb.width < 3 || rgb.height < 3 {
        return;
    }
    let w = rgb.width;
    let h = rgb.height;
    let npix = w * h;

    // Preserve original chroma while writing smoothed results directly to RGB.
    let mut cr = vec![0.0f32; npix];
    let mut cb = vec![0.0f32; npix];

    for p in 0..npix {
        let r = rgb.data[p];
        let g = rgb.data[npix + p];
        let b = rgb.data[2 * npix + p];
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        cr[p] = r - y;
        cb[p] = b - y;
    }

    let mut buf_r = [0.0f32; 9];
    let mut buf_b = [0.0f32; 9];

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let p = y * w + x;
            if rgb.mask[p] == 0 {
                continue;
            }
            let mut idx = 0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let np = ((y as isize + dy) as usize) * w + ((x as isize + dx) as usize);
                    buf_r[idx] = cr[np];
                    buf_b[idx] = cb[np];
                    idx += 1;
                }
            }
            buf_r.sort_by(|a, b| a.total_cmp(b));
            buf_b.sort_by(|a, b| a.total_cmp(b));
            let lum = 0.2126 * rgb.data[p]
                + 0.7152 * rgb.data[npix + p]
                + 0.0722 * rgb.data[2 * npix + p];
            let r = lum + 0.5 * cr[p] + 0.5 * buf_r[4];
            let b = lum + 0.5 * cb[p] + 0.5 * buf_b[4];
            let g = (lum - 0.2126 * r - 0.0722 * b) / 0.7152;
            rgb.data[p] = r;
            rgb.data[npix + p] = g;
            rgb.data[2 * npix + p] = b;
        }
    }
}

/// Normalizes a single-channel background-extracted linear image into a consistent dynamic range unit
/// (`bg_median -> 0.0`, `1.0 -> ~25*sigma`) so multi-channel narrowband palettes combine cleanly.
fn normalize_mono_channel(ch_img: &LinearImage) -> LinearImage {
    let mut bg_sub = remove_smooth_background(ch_img);
    let (med, mad) = robust_median_mad(bg_sub.plane(0), &bg_sub.mask);
    let sigma = (mad * 1.4826).max(1e-4);

    // Find 99.5th percentile of positive signal to scale channel dynamic range
    let mut pos: Vec<f32> = bg_sub
        .plane(0)
        .iter()
        .zip(bg_sub.mask.iter())
        .step_by((bg_sub.pixel_count() / 131_072).max(1))
        .filter(|&(_, &m)| m != 0)
        .map(|(&v, _)| (v - med).max(0.0))
        .collect();
    let p99 = if pos.is_empty() {
        10.0 * sigma
    } else {
        let index = (pos.len() * 995 / 1000).min(pos.len() - 1);
        let (_, percentile, _) = pos.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
        (*percentile).max(8.0 * sigma)
    };

    drop(pos);
    for (v, &m) in bg_sub.data.iter_mut().zip(bg_sub.mask.iter()) {
        if m != 0 {
            *v = (*v - med + 0.25 * sigma) / p99;
        } else {
            *v = 0.0;
        }
    }
    bg_sub
}

/// Sample-calibrated normalization shared by all native tiles in one channel.
/// Fitted once on the bounded raw preview, never independently per tile.
pub struct MagicMonoModel {
    background: [f64; 6],
    median: f32,
    sigma: f32,
    white: f32,
}

impl MagicMonoModel {
    pub fn fit(sample: &LinearImage) -> Self {
        let background = fit_background_coefficients(sample)[0];
        let mut sub = sample.clone();
        subtract_background_model(&mut sub, &[background], (0, 0), (sample.width, sample.height));
        let (median, mad) = robust_median_mad(sub.plane(0), &sub.mask);
        let sigma = (mad * 1.4826).max(1e-4);
        let mut pos: Vec<f32> = sub.plane(0).iter().zip(&sub.mask)
            .step_by((sub.pixel_count() / 131_072).max(1))
            .filter(|&(_, &m)| m != 0).map(|(&v, _)| (v - median).max(0.0)).collect();
        let white = if pos.is_empty() { 10.0 * sigma } else {
            let index = (pos.len() * 995 / 1000).min(pos.len() - 1);
            let (_, p, _) = pos.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
            (*p).max(8.0 * sigma)
        };
        Self { background, median, sigma, white }
    }

    pub fn apply(&self, img: &mut LinearImage, origin: (usize, usize), full: (usize, usize)) {
        subtract_background_model(img, &[self.background], origin, full);
        for (value, &valid) in img.data.iter_mut().zip(&img.mask) {
            *value = if valid != 0 { (*value - self.median + 0.25 * self.sigma) / self.white } else { 0.0 };
        }
    }
}

/// A fixed global RGB display transfer from bounded raw samples. Uses the
/// established Magic background, star-neutral WB and arcsinh/shoulder formula.
pub struct MagicStretchModel {
    background: Vec<[f64; 6]>,
    medians: [f32; 3],
    white_balance: [f32; 3],
    black_guard: f32,
    white: f32,
}

impl MagicStretchModel {
    pub fn fit(sample: &LinearImage, palette_style: &str) -> Self {
        let background = fit_background_coefficients(sample);
        let mut sub = sample.clone();
        subtract_background_model(&mut sub, &background, (0, 0), (sample.width, sample.height));
        let mut medians = [0.0f32; 3];
        for (c, median) in medians.iter_mut().enumerate().take(sub.channels.min(3)) {
            *median = robust_median_mad(sub.plane(c), &sub.mask).0;
        }
        let mut white_balance = [1.0; 3];
        if sub.channels == 3 && palette_style == "natural" {
            let stars = detect_stars(&sub);
            if stars.len() >= 5 {
                let mut sums = [0.0f32; 3];
                let n = sub.pixel_count();
                for s in stars.iter().take(25) {
                    let p = (s.y.round() as usize).min(sub.height - 1) * sub.width
                        + (s.x.round() as usize).min(sub.width - 1);
                    for c in 0..3 { sums[c] += (sub.data[c * n + p] - medians[c]).max(0.0); }
                }
                let avg = (sums[0] + sums[1] + sums[2]) / 3.0;
                for c in 0..3 {
                    if avg > 1e-3 && sums[c] > 1e-3 { white_balance[c] = (avg / sums[c]).clamp(0.7, 1.4); }
                }
            }
        }
        let n = sub.pixel_count();
        for c in 0..sub.channels.min(3) {
            for p in 0..n { sub.data[c * n + p] = (sub.data[c * n + p] - medians[c]) * white_balance[c]; }
        }
        let lum = sub.to_luminance();
        let (l_med, l_mad) = robust_median_mad(lum.plane(0), &lum.mask);
        let l_sigma = (l_mad * 1.4826).max(1e-5);
        let mut values: Vec<f32> = lum.plane(0).iter().zip(&lum.mask)
            .step_by((n / 131_072).max(1)).filter(|&(_, &m)| m != 0)
            .map(|(&v, _)| (v - l_med).max(0.0)).collect();
        let white = if values.is_empty() { 20.0 * l_sigma } else {
            let index = (values.len() * 996 / 1000).min(values.len() - 1);
            let (_, p, _) = values.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
            (*p).max(10.0 * l_sigma)
        };
        Self { background, medians, white_balance,
            black_guard: l_med - 0.8 * l_sigma, white }
    }

    pub fn apply(&self, mut img: LinearImage, origin: (usize, usize), full: (usize, usize)) -> LinearImage {
        subtract_background_model(&mut img, &self.background, origin, full);
        let n = img.pixel_count();
        let mut out = LinearImage::new(img.width, img.height, 3);
        out.mask = img.mask.clone();
        out.exposure = img.exposure;
        let asinh_beta = 35.0f32.asinh();
        for p in 0..n {
            if img.mask[p] == 0 { continue; }
            let mut rgb = [0.0f32; 3];
            for c in 0..3 {
                let src = if img.channels == 1 { 0 } else { c };
                rgb[c] = (((img.data[src * n + p] - self.medians[src]) * self.white_balance[src]
                    - self.black_guard) / self.white).max(0.0);
            }
            let lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).max(1e-6);
            let stretched = (35.0 * lum).asinh() / asinh_beta;
            let scale = (stretched / (1.0 + 0.18 * stretched)) / lum;
            for c in 0..3 { out.data[c * n + p] = (rgb[c] * scale).clamp(0.0, 1.0); }
        }
        out
    }
}

/// Combines 3 normalized mono planes into an RGB `LinearImage`, applies RL luminance deconvolution,
/// star-neutral white balance, arcsinh stretch, and optional `EditParameters`.
pub fn stretch_and_balance_rgb(
    raw_rgb: LinearImage,
    palette_style: &str,
    edits: Option<&EditParameters>,
) -> LinearImage {
    stretch_rgb_core(raw_rgb, palette_style, edits, true)
}

/// Magic starts with the same sky neutralization and stretch, without deconvolution
/// or palette grading; its measured, gated operators own those adjustments.
pub fn stretch_magic_baseline(raw_rgb: LinearImage, palette_style: &str) -> LinearImage {
    stretch_rgb_core(raw_rgb, palette_style, None, false)
}

fn stretch_rgb_core(
    mut raw_rgb: LinearImage,
    palette_style: &str,
    edits: Option<&EditParameters>,
    legacy_grade: bool,
) -> LinearImage {
    let w = raw_rgb.width;
    let h = raw_rgb.height;
    let npix = w * h;

    remove_background_in_place(&mut raw_rgb);
    let mut rgb = raw_rgb;

    // Background neutralization and channel balancing
    let mut medians = [0.0f32; 3];
    let mut sigmas = [1e-4f32; 3];
    for c in 0..rgb.channels.min(3) {
        let (m, mad) = robust_median_mad(rgb.plane(c), &rgb.mask);
        medians[c] = m;
        sigmas[c] = (mad * 1.4826).max(1e-5);
    }

    if rgb.channels == 3 {
        // Star-neutral white balance for broadband natural RGB
        let stars = detect_stars(&rgb);
        let mut star_sums = [0.0f32; 3];
        if stars.len() >= 5 && palette_style == "natural" {
            for s in stars.iter().take(25) {
                let sx = (s.x.round() as usize).min(w - 1);
                let sy = (s.y.round() as usize).min(h - 1);
                let p = sy * w + sx;
                for c in 0..3 {
                    star_sums[c] += (rgb.data[c * npix + p] - medians[c]).max(0.0);
                }
            }
        }
        let avg_star = (star_sums[0] + star_sums[1] + star_sums[2]) / 3.0;
        for c in 0..3 {
            let wb = if avg_star > 1e-3 && star_sums[c] > 1e-3 {
                (avg_star / star_sums[c]).clamp(0.7, 1.4)
            } else {
                1.0
            };
            let m = medians[c];
            let plane = rgb.plane_mut(c);
            for p in 0..npix {
                plane[p] = (plane[p] - m) * wb;
            }
        }

        if legacy_grade {
            // Existing manual finish retains its historical deconvolution and grade.
            let lum_before = rgb.to_luminance();
            let lum_deconv = deconvolve_luminance_rl(&lum_before);
            for p in 0..npix {
                if rgb.mask[p] == 0 {
                    continue;
                }
                let yb = lum_before.data[p];
                let ya = lum_deconv.data[p];
                if yb > 1e-4 && ya > 0.0 {
                    let ratio = (ya / yb).clamp(0.75, 1.45);
                    rgb.data[p] *= ratio;
                    rgb.data[npix + p] *= ratio;
                    rgb.data[2 * npix + p] *= ratio;
                }
            }
            apply_palette_grading(&mut rgb, palette_style);
            gentle_chroma_denoise(&mut rgb);
        }
    } else if rgb.channels == 1 && legacy_grade {
        rgb = deconvolve_luminance_rl(&rgb);
    }

    // Arcsinh stretch with clipped-black guard and soft highlight compression
    let lum_ref = rgb.to_luminance();
    let (l_med, l_mad) = robust_median_mad(lum_ref.plane(0), &lum_ref.mask);
    let l_sigma = (l_mad * 1.4826).max(1e-5);

    let mut pos_vals: Vec<f32> = lum_ref
        .plane(0)
        .iter()
        .zip(lum_ref.mask.iter())
        .step_by((lum_ref.pixel_count() / 131_072).max(1))
        .filter(|&(_, &m)| m != 0)
        .map(|(&v, _)| (v - l_med).max(0.0))
        .collect();
    let white_pt = if pos_vals.is_empty() {
        20.0 * l_sigma
    } else {
        let index = (pos_vals.len() * 996 / 1000).min(pos_vals.len() - 1);
        let (_, percentile, _) = pos_vals.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
        (*percentile).max(10.0 * l_sigma)
    };
    drop(pos_vals);
    drop(lum_ref);

    let black_guard = l_med - 0.8 * l_sigma;
    // Display exposure and saturation are applied once by the parametric finish.
    let exp_mult = 1.0f32;
    let sat_mult = 1.0f32;
    let temp_shift = edits.and_then(|e| e.temperature).unwrap_or(0.0);

    let beta = 35.0f32;
    let asinh_beta = beta.asinh();

    let mut stretched = LinearImage::new(w, h, 3);
    stretched.mask = rgb.mask.clone();
    stretched.exposure = rgb.exposure;

    for p in 0..npix {
        if rgb.mask[p] == 0 {
            continue;
        }
        let (mut r, mut g, mut b) = if rgb.channels == 3 {
            (
                ((rgb.data[p] - black_guard) / white_pt).max(0.0),
                ((rgb.data[npix + p] - black_guard) / white_pt).max(0.0),
                ((rgb.data[2 * npix + p] - black_guard) / white_pt).max(0.0),
            )
        } else {
            let v = ((rgb.data[p] - black_guard) / white_pt).max(0.0);
            (v, v, v)
        };

        r *= exp_mult * (1.0 + 0.22 * temp_shift);
        g *= exp_mult * (1.0 + 0.03 * temp_shift);
        b *= exp_mult * (1.0 - 0.22 * temp_shift);

        let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).max(1e-6);
        let stretched_lum = (beta * lum).asinh() / asinh_beta;
        // Soft highlight shoulder
        let comp_lum = stretched_lum / (1.0 + 0.18 * stretched_lum);
        let scale = comp_lum / lum;

        r *= scale;
        g *= scale;
        b *= scale;

        // Saturation adjustment around luminance
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        r = (y + sat_mult * (r - y)).clamp(0.0, 1.0);
        g = (y + sat_mult * (g - y)).clamp(0.0, 1.0);
        b = (y + sat_mult * (b - y)).clamp(0.0, 1.0);

        stretched.data[p] = r;
        stretched.data[npix + p] = g;
        stretched.data[2 * npix + p] = b;
    }

    stretched
}

fn apply_palette_grading(rgb: &mut LinearImage, palette_style: &str) {
    let npix = rgb.width * rgb.height;
    match palette_style {
        "hubble" => {
            // Tuned SHO Hubble Gold & Cyan (SCNR green balance toward warm gold and cyan-blue)
            for p in 0..npix {
                let s2 = rgb.data[p].max(0.0);
                let ha = rgb.data[npix + p].max(0.0);
                let o3 = rgb.data[2 * npix + p].max(0.0);
                let r = 0.72 * s2 + 0.38 * ha;
                let g = 0.55 * ha + 0.45 * o3;
                let b = 1.08 * o3;
                // Cap excess green above average of R and B
                let max_g = 0.58 * (r + b);
                rgb.data[p] = r;
                rgb.data[npix + p] = g.min(max_g.max(0.35 * g));
                rgb.data[2 * npix + p] = b;
            }
        }
        "jwst-inspired" => {
            // Artistic JWST NIRCam-inspired warm amber/copper & deep sapphire styling
            for p in 0..npix {
                let r0 = rgb.data[p].max(0.0);
                let g0 = rgb.data[npix + p].max(0.0);
                let b0 = rgb.data[2 * npix + p].max(0.0);
                let r = 0.85 * r0 + 0.35 * g0;
                let g = 0.52 * g0 + 0.32 * r0 + 0.20 * b0;
                let b = 1.15 * b0 + 0.15 * g0;
                rgb.data[p] = r;
                rgb.data[npix + p] = g;
                rgb.data[2 * npix + p] = b;
            }
        }
        "hubble-inspired" => {
            // Artistic Hubble-inspired gold/teal styling from broadband RGB
            for p in 0..npix {
                let r0 = rgb.data[p].max(0.0);
                let g0 = rgb.data[npix + p].max(0.0);
                let b0 = rgb.data[2 * npix + p].max(0.0);
                let r = 1.05 * r0 + 0.20 * g0;
                let g = 0.65 * g0 + 0.35 * b0;
                let b = 1.12 * b0 + 0.15 * g0;
                rgb.data[p] = r;
                rgb.data[npix + p] = g;
                rgb.data[2 * npix + p] = b;
            }
        }
        _ => {}
    }
}

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let x = v.clamp(0.0, 1.0);
    let srgb = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

#[inline]
fn linear_to_u16(v: f32) -> u16 {
    let x = v.clamp(0.0, 1.0);
    (x * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
}

/// Supplies one interleaved RGB float row and its validity mask. The callback
/// may reuse its buffers; neither output codec keeps a reference to them.
pub fn write_variant_image_rows(
    width: usize,
    height: usize,
    tiff_dest: &Path,
    png_dest: &Path,
    mut row: impl FnMut(usize, &mut [f32], &mut [u8]) -> Result<()>,
) -> Result<()> {
    let w = u32::try_from(width).context("TIFF width exceeds u32")?;
    let h = u32::try_from(height).context("TIFF height exceeds u32")?;
    let _ = width.checked_mul(height).and_then(|n| n.checked_mul(13)).context("Image dimensions overflow")?;
    if width == 0 || height == 0 { bail!("Cannot write empty image"); }
    let step = width.max(height).div_ceil(2048);
    let pw = width.div_ceil(step);
    let ph = height.div_ceil(step);
    let preview_len = pw.checked_mul(ph).context("PNG preview dimensions overflow")?;
    let mut sums = vec![[0.0f32; 3]; preview_len];
    let mut counts = vec![0usize; preview_len];
    let mut rgb = vec![0.0f32; width.checked_mul(3).context("RGB row overflow")?];
    let mut mask = vec![0u8; width];
    let parent = tiff_dest.parent().context("TIFF destination has no parent")?;
    std::fs::create_dir_all(parent)?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut writer = std::io::BufWriter::new(temp.as_file_mut());
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut writer)
            .context("Failed to initialize 16-bit TIFF encoder")?;
        let mut image = encoder.new_image::<tiff::encoder::colortype::RGB16>(w, h)?;
        image.rows_per_strip(32)?;
        let mut samples = Vec::with_capacity(width.checked_mul(32).and_then(|n| n.checked_mul(3)).context("TIFF strip overflow")?);
        let mut y = 0usize;
        while image.next_strip_sample_count() != 0 {
            let count = image.next_strip_sample_count() as usize / 3;
            let rows = count / width;
            samples.clear();
            for _ in 0..rows {
                row(y, &mut rgb, &mut mask)?;
                for x in 0..width {
                    for c in 0..3 { samples.push(linear_to_u16(rgb[3 * x + c])); }
                    if mask[x] != 0 {
                        let dst = (y / step) * pw + x / step;
                        for c in 0..3 { sums[dst][c] += rgb[3 * x + c]; }
                        counts[dst] += 1;
                    }
                }
                y += 1;
            }
            image.write_strip(&samples)?;
        }
        image.finish()?;
        drop(encoder);
        std::io::Write::flush(&mut writer)?;
    }
    temp.as_file().sync_all()?;
    temp.persist(tiff_dest).map_err(|e| e.error)?;
    // Area-average only the bounded linear-light preview before PNG encoding.
    let mut png_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(pw as u32, ph as u32);
    for y in 0..ph {
        for x in 0..pw {
            let p = y * pw + x;
            let color = sums[p].map(|v| linear_to_srgb_u8(v / counts[p].max(1) as f32));
            png_img.put_pixel(x as u32, y as u32, Rgb(color));
        }
    }
    let mut png_cursor = Cursor::new(Vec::new());
    png_img.write_to(&mut png_cursor, image::ImageFormat::Png)
        .context("Failed to encode sRGB PNG preview")?;
    crate::atomic_write_file(png_dest, &png_cursor.into_inner())?;
    Ok(())
}

/// Writes unchanged legacy pixels through the bounded strip/row encoder.
pub fn write_variant_images(
    stretched_rgb: &LinearImage,
    tiff_dest: &Path,
    png_dest: &Path,
) -> Result<()> {
    let width = stretched_rgb.width;
    let npix = width.checked_mul(stretched_rgb.height).context("Image dimensions overflow")?;
    if stretched_rgb.channels != 3 || stretched_rgb.data.len() != npix.checked_mul(3).context("Image dimensions overflow")?
        || stretched_rgb.mask.len() != npix { bail!("Expected complete RGB image"); }
    write_variant_image_rows(width, stretched_rgb.height, tiff_dest, png_dest, |y, rgb, mask| {
        let offset = y * width;
        mask.copy_from_slice(&stretched_rgb.mask[offset..offset + width]);
        for x in 0..width {
            for c in 0..3 { rgb[x * 3 + c] = stretched_rgb.data[c * npix + offset + x]; }
        }
        Ok(())
    })
}

fn combine_three_planes(
    r_plane: &LinearImage,
    g_plane: &LinearImage,
    b_plane: &LinearImage,
) -> LinearImage {
    let w = r_plane.width;
    let h = r_plane.height;
    let npix = w * h;
    let mut out = LinearImage::new(w, h, 3);
    out.exposure = r_plane.exposure + g_plane.exposure + b_plane.exposure;
    out.data[0..npix].copy_from_slice(r_plane.plane(0));
    out.data[npix..2 * npix].copy_from_slice(g_plane.plane(0));
    out.data[2 * npix..3 * npix].copy_from_slice(b_plane.plane(0));
    for p in 0..npix {
        out.mask[p] = r_plane.mask[p] & g_plane.mask[p] & b_plane.mask[p];
    }
    out
}

/// Builds normalized single-channel linear maps from a channel integration table.
/// If `RgbCfa` is present and separate `R`/`G`/`B` filters are absent, exposes extracted R, G, B planes for RGB/custom mixing,
/// while NEVER synthesizing narrowband `Ha`/`OIII`/`SII`.
pub fn build_normalized_channel_map(
    raw_channels: &BTreeMap<Channel, &LinearImage>,
) -> BTreeMap<Channel, LinearImage> {
    let mut map = BTreeMap::new();
    for (&ch, img) in raw_channels {
        if ch == Channel::RgbCfa && img.channels == 3 {
            if !raw_channels.contains_key(&Channel::R) {
                map.insert(Channel::R, normalize_mono_channel(&img.extract_channel(0)));
            }
            if !raw_channels.contains_key(&Channel::G) {
                map.insert(Channel::G, normalize_mono_channel(&img.extract_channel(1)));
            }
            if !raw_channels.contains_key(&Channel::B) {
                map.insert(Channel::B, normalize_mono_channel(&img.extract_channel(2)));
            }
        } else {
            map.insert(ch, normalize_mono_channel(img));
        }
    }
    map
}

/// Renders a single named palette or custom mix from a target's linear channel map.
pub fn compose_palette_linear(
    palette_id: &str,
    raw_channels: &BTreeMap<Channel, &LinearImage>,
    norm_channels: &BTreeMap<Channel, LinearImage>,
    mix_spec: Option<&MixSpec>,
) -> Result<(LinearImage, Vec<String>, String, String, bool)> {
    match palette_id {
        "natural" => {
            if let Some(rgb_cfa) = raw_channels.get(&Channel::RgbCfa) {
                Ok((
                    (**rgb_cfa).clone(),
                    vec!["RGB".to_string()],
                    "Natural Color (RGB)".to_string(),
                    "Broadband true-color calibrated integration with star-neutral white balance."
                        .to_string(),
                    false,
                ))
            } else if let (Some(r), Some(g), Some(b)) = (
                norm_channels.get(&Channel::R),
                norm_channels.get(&Channel::G),
                norm_channels.get(&Channel::B),
            ) {
                Ok((
                    combine_three_planes(r, g, b),
                    vec!["R".to_string(), "G".to_string(), "B".to_string()],
                    "Natural Color (R/G/B)".to_string(),
                    "Broadband R, G, B filter integration with star-neutral white balance."
                        .to_string(),
                    false,
                ))
            } else {
                bail!("Requires RGB CFA or R, G, and B filter integrations")
            }
        }
        "hoo" => {
            let (Some(ha), Some(o3)) = (
                norm_channels.get(&Channel::Ha),
                norm_channels.get(&Channel::Oiii),
            ) else {
                bail!("Requires both Ha and OIII narrowband filter integrations")
            };
            Ok((
                combine_three_planes(ha, o3, o3),
                vec!["Ha".to_string(), "OIII".to_string()],
                "HOO Bicolor (Ha → R, OIII → G+B)".to_string(),
                "Dual-band emission mapping: Hydrogen-alpha in red, doubly-ionized Oxygen in green and blue.".to_string(),
                false,
            ))
        }
        "sho" => {
            let (Some(s2), Some(ha), Some(o3)) = (
                norm_channels.get(&Channel::Sii),
                norm_channels.get(&Channel::Ha),
                norm_channels.get(&Channel::Oiii),
            ) else {
                bail!("Requires SII, Ha, and OIII narrowband filter integrations")
            };
            Ok((
                combine_three_planes(s2, ha, o3),
                vec!["SII".to_string(), "Ha".to_string(), "OIII".to_string()],
                "SHO Narrowband (SII → R, Ha → G, OIII → B)".to_string(),
                "Three-band emission mapping ordered by wavelength: Sulfur-II, Hydrogen-alpha, Oxygen-III.".to_string(),
                false,
            ))
        }
        "osh" => {
            let (Some(o3), Some(s2), Some(ha)) = (
                norm_channels.get(&Channel::Oiii),
                norm_channels.get(&Channel::Sii),
                norm_channels.get(&Channel::Ha),
            ) else {
                bail!("Requires OIII, SII, and Ha narrowband filter integrations")
            };
            Ok((
                combine_three_planes(o3, s2, ha),
                vec!["OIII".to_string(), "SII".to_string(), "Ha".to_string()],
                "OSH Narrowband (OIII → R, SII → G, Ha → B)".to_string(),
                "Inverted emission palette emphasizing Oxygen-III in red and Hydrogen-alpha in blue.".to_string(),
                false,
            ))
        }
        "hubble" => {
            let (Some(s2), Some(ha), Some(o3)) = (
                norm_channels.get(&Channel::Sii),
                norm_channels.get(&Channel::Ha),
                norm_channels.get(&Channel::Oiii),
            ) else {
                bail!("Requires SII, Ha, and OIII narrowband filter integrations")
            };
            Ok((
                combine_three_planes(s2, ha, o3),
                vec!["SII".to_string(), "Ha".to_string(), "OIII".to_string()],
                "Hubble Palette (Tuned SHO)".to_string(),
                "Tuned SHO narrowband balance with SCNR green moderation for gold-and-cyan emission contrast.".to_string(),
                false,
            ))
        }
        "jwst-inspired" => {
            if let (Some(s2), Some(ha), Some(o3)) = (
                norm_channels.get(&Channel::Sii),
                norm_channels.get(&Channel::Ha),
                norm_channels.get(&Channel::Oiii),
            ) {
                Ok((
                    combine_three_planes(s2, ha, o3),
                    vec!["SII".to_string(), "Ha".to_string(), "OIII".to_string()],
                    "JWST-Inspired (Artistic Interpretation from SHO)".to_string(),
                    "Artistic interpretation: NIRCam-inspired copper/amber and deep sapphire tone mapping from SII/Ha/OIII. Not actual Webb infrared band measurements.".to_string(),
                    true,
                ))
            } else if let Ok((nat_img, used, _, _, _)) =
                compose_palette_linear("natural", raw_channels, norm_channels, None)
            {
                Ok((
                    nat_img,
                    used,
                    "JWST-Inspired (Artistic Interpretation from RGB)".to_string(),
                    "Artistic interpretation: NIRCam-inspired warm amber and sapphire color styling from broadband RGB. Not actual Webb infrared band measurements.".to_string(),
                    true,
                ))
            } else {
                bail!("Requires either SHO narrowband or RGB broadband integrations")
            }
        }
        "hubble-inspired" => {
            let (nat_img, used, _, _, _) =
                compose_palette_linear("natural", raw_channels, norm_channels, None)?;
            Ok((
                nat_img,
                used,
                "Hubble-Inspired (Artistic Interpretation from RGB)".to_string(),
                "Artistic interpretation: Hubble-inspired gold and cyan tone styling derived from broadband RGB. Not actual narrowband emission measurements.".to_string(),
                true,
            ))
        }
        id if id == "monochrome" || id.starts_with("monochrome-") => {
            let channel = if let Some(name) = id.strip_prefix("monochrome-") {
                Channel::from_token(name)
                    .filter(|ch| *ch != Channel::RgbCfa && ch.as_str().eq_ignore_ascii_case(name))
                    .context("Unknown monochrome filter")?
            } else if let Some((&ch, _)) = raw_channels.iter().find(|(ch, _)| **ch == Channel::L) {
                ch
            } else if raw_channels.len() == 1 {
                *raw_channels.keys().next().unwrap()
            } else {
                bail!("Choose a filter-specific monochrome palette")
            };
            if channel == Channel::RgbCfa {
                bail!("RGB CFA is a color integration, not a monochrome filter")
            }
            let lum = norm_channels
                .get(&channel)
                .context("Monochrome filter is not present")?;
            let name = channel.as_str();
            Ok((
                combine_three_planes(lum, lum, lum),
                vec![name.to_string()],
                format!("Monochrome ({name} filter)"),
                format!(
                    "Calibrated {name} filter integration displayed in grayscale (not RGB color)."
                ),
                false,
            ))
        }
        "custom" => {
            let Some(spec) = mix_spec else {
                bail!("No custom --mix specification provided")
            };
            for req_ch in spec.required_channels() {
                if !norm_channels.contains_key(&req_ch) {
                    bail!(
                        "Custom mix '{}' requires channel '{}', which is not present in the input frames",
                        spec.raw,
                        req_ch
                    );
                }
            }
            let required = spec.required_channels();
            let first = required.first().context("Custom mix has no channels")?;
            let sample = norm_channels
                .get(first)
                .context("Custom mix channel is unavailable")?;
            let w = sample.width;
            let h = sample.height;
            let npix = w * h;
            let mut out = LinearImage::new(w, h, 3);
            out.exposure = sample.exposure;

            for (c_out, terms) in spec.expressions.iter().enumerate() {
                for term in terms {
                    if term.weight <= 0.0 {
                        continue;
                    }
                    let src_ch = &norm_channels[&term.channel];
                    let src_plane = src_ch.plane(0);
                    for p in 0..npix {
                        out.data[c_out * npix + p] += term.weight * src_plane[p];
                        if src_ch.mask[p] == 0 {
                            out.mask[p] = 0;
                        }
                    }
                }
            }
            let used: Vec<String> = spec
                .required_channels()
                .into_iter()
                .map(|c| c.as_str().to_string())
                .collect();
            Ok((
                out,
                used,
                format!("Custom Mix ({})", spec.raw),
                format!("User-defined linear channel mix: {}.", spec.raw),
                false,
            ))
        }
        other => bail!("Unsupported palette identifier '{}'", other),
    }
}

fn used_science<'a>(
    science: &'a [ScienceArtifact],
    used: &'a [String],
) -> impl Iterator<Item = &'a ScienceArtifact> {
    science.iter().filter(move |s| {
        used.contains(&s.channel)
            || (s.channel == "RGB"
                && used.iter().any(|band| {
                    matches!(band.as_str(), "R" | "G" | "B")
                        && !science.iter().any(|other| other.channel == *band)
                }))
    })
}

fn display_crop(science: &[ScienceArtifact], used: &[String], trim: bool) -> Option<CropRect> {
    if trim {
        used_science(science, used).find_map(|s| s.recommended_crop)
    } else {
        None
    }
}

fn finish_options_from_resolved(f: ResolvedFinish) -> FinishOptions {
    FinishOptions {
        preset: f.preset,
        exposure: Some(f.exposure),
        contrast: Some(f.contrast),
        saturation: Some(f.saturation),
        gamma: Some(f.gamma),
        local_contrast: Some(f.local_contrast),
    }
}

fn canonical_palette(palette: &str) -> Result<String> {
    let lower = palette.to_ascii_lowercase();
    if let Some(band) = lower.strip_prefix("monochrome-") {
        let channel = Channel::from_token(band).context("Unknown monochrome filter")?;
        Ok(format!("monochrome-{}", channel.as_str()))
    } else {
        Ok(lower)
    }
}

/// Renders all feasible gallery variants for a target from present channels.
#[allow(clippy::too_many_arguments)]
pub fn render_target_gallery(
    target_id: &str,
    target_label: &str,
    cal_summary: &CalibrationSummary,
    mut stack_output: TargetIntegrationOutput,
    run_dir: &Path,
    run_rel_dir: &str,
    options: &ProcessOptions,
    mix_spec: Option<&MixSpec>,
    warnings: &mut Vec<String>,
) -> Result<TargetGallery> {
    // FITS exports are already complete. Retain full-canvas metadata before
    // cropping these owned in-memory copies for display-only finishing.
    let science_integrations = build_science_artifacts(&stack_output.integrations);
    if options.trim_edges {
        for ci in stack_output.integrations.values_mut() {
            if let Some(crop) = ci.recommended_crop {
                crop_image(&mut ci.image, &crop)?;
            }
        }
    }
    // All integrated bands remain available for standalone grayscale, including unaligned ones.
    let raw_channels: BTreeMap<_, _> = stack_output
        .integrations
        .iter()
        .map(|(&ch, ci)| (ch, &ci.image))
        .collect();
    let norm_channels = build_normalized_channel_map(&raw_channels);

    // If `--mix` was explicitly passed, validate that all required channels exist in this target
    if let Some(spec) = mix_spec {
        for req_ch in spec.required_channels() {
            if !norm_channels.contains_key(&req_ch) {
                bail!(
                    "Custom --mix '{}' requires channel '{}', which is missing from target '{}' (present: {:?})",
                    spec.raw,
                    req_ch,
                    target_label,
                    norm_channels.keys().map(|c| c.as_str()).collect::<Vec<_>>()
                );
            }
        }
    }

    let has_sho = norm_channels.contains_key(&Channel::Sii)
        && norm_channels.contains_key(&Channel::Ha)
        && norm_channels.contains_key(&Channel::Oiii);
    let has_only_mono = raw_channels.len() == 1 && !raw_channels.contains_key(&Channel::RgbCfa);

    let mut candidate_palettes = vec![
        "natural".to_string(),
        "hoo".to_string(),
        "sho".to_string(),
        "osh".to_string(),
        "hubble".to_string(),
        "jwst-inspired".to_string(),
    ];
    if !has_sho {
        candidate_palettes.push("hubble-inspired".to_string());
    }
    for &ch in raw_channels.keys() {
        if ch != Channel::RgbCfa {
            candidate_palettes.push(if has_only_mono {
                "monochrome".to_string()
            } else {
                format!("monochrome-{}", ch.as_str())
            });
        }
    }
    if mix_spec.is_some() {
        candidate_palettes.push("custom".to_string());
    }

    let mut variants = Vec::new();
    let mut omitted_variants = Vec::new();

    for pal_id in candidate_palettes {
        // A lone mono filter must never be interpreted as a color palette.
        if has_only_mono && pal_id != "monochrome" && pal_id != "custom" {
            omitted_variants.push(OmittedVariantRecord {
                palette: pal_id,
                reason: "Only one monochrome filter is present; omitting color synthesis."
                    .to_string(),
            });
            continue;
        }

        // A failed cross-filter registration cannot be hidden behind an artistic preset
        // or custom mix. A native CFA image needs no cross-filter registration.
        if !stack_output.cross_filter_aligned
            && (matches!(pal_id.as_str(), "hoo" | "sho" | "osh" | "hubble")
                || (pal_id == "custom"
                    && mix_spec.is_some_and(|spec| spec.required_channels().len() > 1))
                || (matches!(
                    pal_id.as_str(),
                    "natural" | "hubble-inspired" | "jwst-inspired"
                ) && !raw_channels.contains_key(&Channel::RgbCfa))
                || (pal_id == "jwst-inspired" && has_sho))
        {
            omitted_variants.push(OmittedVariantRecord {
                palette: pal_id,
                reason: "Omitted because cross-filter star alignment failed.".to_string(),
            });
            continue;
        }

        match compose_palette_linear(&pal_id, &raw_channels, &norm_channels, mix_spec) {
            Ok((mut linear_rgb, used_channels, title, description, artistic)) => {
                linear_rgb.exposure = used_science(&science_integrations, &used_channels)
                    .map(|s| s.total_exposure_s)
                    .sum();
                let mut stretched = stretch_and_balance_rgb(linear_rgb, &pal_id, None);
                let finish = apply_finish(&mut stretched, &options.finish)?;
                let crop = display_crop(&science_integrations, &used_channels, options.trim_edges);
                let tiff_name = format!("{}-{}.tiff", target_id, pal_id);
                let png_name = format!("{}-{}.png", target_id, pal_id);
                let tiff_path = run_dir.join(&tiff_name);
                let png_path = run_dir.join(&png_name);
                write_variant_images(&stretched, &tiff_path, &png_path)?;

                let provenance = format!(
                    "Channels: {} | Total exposure: {:.1}s | Calibrated & integrated linear float",
                    used_channels.join("+"),
                    used_science(&science_integrations, &used_channels)
                        .map(|s| s.total_exposure_s)
                        .sum::<f32>()
                );

                variants.push(VariantArtifact {
                    id: pal_id.clone(),
                    title,
                    palette: pal_id.clone(),
                    description,
                    artistic_interpretation: artistic,
                    ai_enhanced: false,
                    tiff_path: format!("{run_rel_dir}/{tiff_name}"),
                    png_path: format!("{run_rel_dir}/{png_name}"),
                    channels_used: used_channels,
                    provenance,
                    restore_info: None,
                    finish: Some(finish),
                    crop,
                    magic: None,
                });
            }
            Err(reason) => {
                omitted_variants.push(OmittedVariantRecord {
                    palette: pal_id,
                    reason: format!("{reason:#}"),
                });
            }
        }
    }

    // Opt-in local neural restoration (`--restore`) on finished RGB preserving original variants and science FITS
    if options.restore {
        let base_var = variants
            .iter()
            .find(|v| v.id == "natural")
            .or_else(|| variants.first())
            .cloned()
            .context("No finished variant available for --restore pass")?;

        let gallery_root = run_dir.parent().and_then(|p| p.parent()).unwrap_or(run_dir);
        let input_tiff = gallery_root.join(&base_var.tiff_path);
        let restored_id = format!("{}-restored", base_var.id);
        let restored_tiff_name = format!("{}-{}.tiff", target_id, restored_id);
        let restored_png_name = format!("{}-{}.png", target_id, restored_id);
        let restored_tiff_path = run_dir.join(&restored_tiff_name);
        let restored_png_path = run_dir.join(&restored_png_name);

        let restore_meta = crate::models::run_cosmic_clarity_restore(
            &input_tiff,
            &restored_tiff_path,
            &restored_png_path,
            0.35,
        )?;

        variants.push(VariantArtifact {
            id: restored_id.clone(),
            title: format!("{} (AI-Enhanced Restoration)", base_var.title),
            palette: restored_id,
            description: format!(
                "AI-enhanced star-shape and denoise pass on finished {} using pinned MIT Cosmic Clarity CNNs (strength {:.2}). Preserves original variants and scientific FITS; not scientifically measured detail.",
                base_var.title, restore_meta.strength
            ),
            artistic_interpretation: true,
            ai_enhanced: true,
            tiff_path: format!("{run_rel_dir}/{restored_tiff_name}"),
            png_path: format!("{run_rel_dir}/{restored_png_name}"),
            channels_used: base_var.channels_used.clone(),
            provenance: format!(
                "{} | Cosmic Clarity rev {} (SHA256 verified)",
                base_var.provenance,
                &restore_meta.source_revision[..12]
            ),
            restore_info: Some(restore_meta),
            finish: base_var.finish,
            crop: base_var.crop,
            magic: None,
        });
    }

    if variants.is_empty() {
        bail!(
            "Target '{}' produced no feasible gallery variants",
            target_label
        );
    }

    // Prefer measured RGB, then the complete SHO set, then HOO, then a stable filter priority.
    let primary_variant = ["natural", "sho", "hoo"]
        .into_iter()
        .find(|&id| variants.iter().any(|v| v.id == id))
        .map(str::to_string)
        .or_else(|| {
            [
                Channel::L,
                Channel::Ha,
                Channel::Oiii,
                Channel::Sii,
                Channel::R,
                Channel::G,
                Channel::B,
            ]
            .into_iter()
            .find_map(|ch| {
                variants
                    .iter()
                    .find(|v| v.channels_used == [ch.as_str()] && v.id.starts_with("monochrome"))
                    .map(|v| v.id.clone())
            })
        })
        .unwrap_or_else(|| variants[0].id.clone());

    let selected_variant = if let Some(req_pal) = &options.palette {
        let req_lower = req_pal.to_ascii_lowercase();
        if let Some(found) = variants.iter().find(|v| {
            v.id.eq_ignore_ascii_case(&req_lower) || v.palette.eq_ignore_ascii_case(&req_lower)
        }) {
            found.id.clone()
        } else {
            warnings.push(format!(
                "Requested initial palette '{}' is not available for target '{}'; selecting primary variant '{}'",
                req_pal, target_label, primary_variant
            ));
            primary_variant.clone()
        }
    } else if mix_spec.is_some() && variants.iter().any(|v| v.id == "custom") {
        "custom".to_string()
    } else {
        primary_variant.clone()
    };

    Ok(TargetGallery {
        target_id: target_id.to_string(),
        target_label: target_label.to_string(),
        about: crate::catalog::lookup_target(target_label),
        primary_variant,
        selected_variant,
        science_integrations,
        variants,
        omitted_variants,
        calibration_summary: cal_summary.clone(),
        magic_sessions: Vec::new(),
    })
}

/// Reloads stored linear FITS integrations from a completed gallery's run directory and rerenders
/// a preset/custom mix or edited variant without repeating calibration or stacking.
pub fn rerender_variant_from_stored_fits(
    gallery_dir: &Path,
    manifest: &mut GalleryManifest,
    palette: Option<&str>,
    mix: Option<&str>,
    finish_override: Option<&FinishOptions>,
    trim_override: Option<bool>,
    edits: Option<&EditParameters>,
    variant_suffix: Option<&str>,
) -> Result<VariantArtifact> {
    let target = manifest
        .targets
        .first_mut()
        .context("Gallery manifest contains no targets")?;
    if target.about.is_none() {
        target.about = crate::catalog::lookup_target(&target.target_label);
    }
    let previous = target
        .variants
        .iter()
        .find(|v| v.id == target.selected_variant);
    let mut finish_options = finish_override
        .copied()
        .or_else(|| {
            previous
                .and_then(|v| v.finish)
                .map(finish_options_from_resolved)
        })
        .unwrap_or_default();
    let trim_edges =
        trim_override.unwrap_or_else(|| previous.map(|v| v.crop.is_some()).unwrap_or(true));
    if let Some(edits) = edits {
        finish_options.exposure = edits.exposure.or(finish_options.exposure);
        finish_options.saturation = edits.saturation.or(finish_options.saturation);
    }
    finish_options.resolve()?;

    let mut raw_channels = BTreeMap::new();
    for sci in &target.science_integrations {
        let Some(ch) = Channel::from_token(&sci.channel) else {
            continue;
        };
        let fits_path = gallery_dir.join(&sci.fits_path);
        let meta = inspect_fits_header(&fits_path)?;
        let frame = crate::fits::build_fits_frame(
            &fits_path,
            meta,
            crate::config::FrameRole::Light,
            Some(ch),
        );
        let mut img = read_fits_linear(&frame)?;
        img.exposure = sci.total_exposure_s;
        if trim_edges {
            if let Some(crop) = sci.recommended_crop {
                crop_image(&mut img, &crop)?;
            }
        }
        raw_channels.insert(ch, img);
    }
    let raw_channels: BTreeMap<_, _> = raw_channels.iter().map(|(&ch, img)| (ch, img)).collect();
    let norm_channels = build_normalized_channel_map(&raw_channels);

    let mix_spec = if let Some(m) = mix {
        Some(MixSpec::parse(m)?)
    } else {
        None
    };

    let base_palette = if mix_spec.is_some() {
        "custom".to_string()
    } else if let Some(p) = palette.or_else(|| edits.and_then(|e| e.palette.as_deref())) {
        canonical_palette(p)?
    } else {
        canonical_palette(
            target
                .variants
                .iter()
                .find(|v| v.id == target.selected_variant)
                .map(|v| v.palette.as_str())
                .unwrap_or(target.selected_variant.as_str()),
        )?
    };

    // Stored FITS retain unaligned standalone filters; a rerender must not
    // reintroduce a cross-filter mix that the initial gallery omitted.
    let alignment_failed = target.omitted_variants.iter().any(|omitted| {
        omitted
            .reason
            .contains("cross-filter star alignment failed")
    });
    if alignment_failed
        && (matches!(base_palette.as_str(), "hoo" | "sho" | "osh" | "hubble")
            || (base_palette == "custom"
                && mix_spec
                    .as_ref()
                    .is_some_and(|spec| spec.required_channels().len() > 1))
            || (matches!(
                base_palette.as_str(),
                "natural" | "jwst-inspired" | "hubble-inspired"
            ) && !raw_channels.contains_key(&Channel::RgbCfa))
            || (base_palette == "jwst-inspired"
                && [Channel::Sii, Channel::Ha, Channel::Oiii]
                    .iter()
                    .all(|ch| norm_channels.contains_key(ch))))
    {
        bail!("Cannot combine filters: cross-filter star alignment failed")
    }

    let (mut linear_rgb, used_channels, base_title, mut description, artistic) =
        compose_palette_linear(
            &base_palette,
            &raw_channels,
            &norm_channels,
            mix_spec.as_ref(),
        )?;

    linear_rgb.exposure = used_science(&target.science_integrations, &used_channels)
        .map(|s| s.total_exposure_s)
        .sum();
    let mut stretched = stretch_and_balance_rgb(linear_rgb, &base_palette, edits);
    let finish = apply_finish(&mut stretched, &finish_options)?;
    let crop = display_crop(&target.science_integrations, &used_channels, trim_edges);

    let reuse_base = edits.is_none()
        && variant_suffix.is_some_and(|suffix| {
            clap::ValueEnum::to_possible_value(&finish.preset)
                .is_some_and(|value| value.get_name() == suffix)
        })
        && base_palette != "custom"
        && target.variants.iter().any(|v| {
            v.id == base_palette && v.finish == Some(finish) && v.crop == crop && !v.ai_enhanced
        });
    let (var_id, title) = if reuse_base {
        (base_palette.clone(), base_title)
    } else if let Some(suffix) = variant_suffix {
        (
            format!("{}-{}", base_palette, suffix),
            format!("{} ({})", base_title, suffix),
        )
    } else {
        (base_palette.clone(), base_title)
    };

    if let Some(ed) = edits {
        let mut parts = Vec::new();
        if let Some(exp) = ed.exposure {
            parts.push(format!("exposure {:+.2} stops", exp));
        }
        if let Some(temp) = ed.temperature {
            parts.push(format!("temperature {:+.2}", temp));
        }
        if let Some(sat) = ed.saturation {
            parts.push(format!("saturation {:.2}x", sat));
        }
        if !parts.is_empty() {
            description = format!("{} Edited: {}.", description, parts.join(", "));
        }
    }

    let run_rel_dir = &manifest.run_dir;
    let run_dir = gallery_dir.join(run_rel_dir);
    let tiff_name = format!("{}-{}.tiff", target.target_id, var_id);
    let png_name = format!("{}-{}.png", target.target_id, var_id);
    write_variant_images(
        &stretched,
        &run_dir.join(&tiff_name),
        &run_dir.join(&png_name),
    )?;

    let artifact = VariantArtifact {
        id: var_id.clone(),
        title,
        palette: base_palette,
        description,
        artistic_interpretation: artistic,
        ai_enhanced: false,
        tiff_path: format!("{run_rel_dir}/{tiff_name}"),
        png_path: format!("{run_rel_dir}/{png_name}"),
        channels_used: used_channels.clone(),
        provenance: format!(
            "Rerendered from linear science FITS ({}) without re-stacking",
            used_channels.join("+")
        ),
        restore_info: None,
        finish: Some(finish),
        crop,
        magic: None,
    };

    if let Some(existing) = target.variants.iter_mut().find(|v| v.id == var_id) {
        *existing = artifact.clone();
    } else {
        target.variants.push(artifact.clone());
    }
    target.selected_variant = var_id;

    // Persist updated manifest and HTML atomically
    let manifest_bytes = serde_json::to_vec_pretty(manifest)?;
    crate::atomic_write_file(&run_dir.join("manifest.json"), &manifest_bytes)?;
    let run_html = generate_static_gallery_html(manifest, true)?;
    crate::atomic_write_file(&run_dir.join("index.html"), run_html.as_bytes())?;
    crate::atomic_write_file(&gallery_dir.join("manifest.json"), &manifest_bytes)?;
    let root_html = generate_static_gallery_html(manifest, false)?;
    crate::atomic_write_file(&gallery_dir.join("index.html"), root_html.as_bytes())?;

    Ok(artifact)
}

fn rel_asset_path(path: &str, run_dir_prefix: &str, inside_run_dir: bool) -> String {
    if inside_run_dir {
        let prefix = format!("{run_dir_prefix}/");
        path.strip_prefix(&prefix).unwrap_or(path).to_string()
    } else {
        path.to_string()
    }
}

/// Generates a self-contained, offline-openable responsive `index.html` for the gallery.
pub fn generate_static_gallery_html(
    manifest: &GalleryManifest,
    inside_run_dir: bool,
) -> Result<String> {
    let mut html = String::with_capacity(16384);
    let run_link = if inside_run_dir {
        "index.html".to_string()
    } else {
        format!("{}/index.html", manifest.run_dir)
    };

    html.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photonloom — From photons to masterpieces</title>
<style>
:root {
  --bg: #0b0e14;
  --panel: #131822;
  --border: #242d3d;
  --fg: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --badge-art: #d29922;
  --badge-ai: #a371f7;
  --badge-sci: #3fb950;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.5;
}
header {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}
header h1 { margin: 0; font-size: 1.4rem; }
header p { margin: 0.25rem 0 0; color: var(--muted); font-size: 0.9rem; }
main { max-width: 1280px; margin: 0 auto; padding: 1.5rem 2rem; }
.meta-bar {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1.2rem;
  margin-bottom: 1.5rem;
  font-size: 0.88rem;
  color: var(--muted);
}
.warnings {
  background: rgba(210, 153, 34, 0.12);
  border: 1px solid var(--badge-art);
  border-radius: 8px;
  padding: 0.8rem 1.2rem;
  margin-bottom: 1.5rem;
}
.target-section { margin-bottom: 2.5rem; }
.target-about { grid-column: 1 / -1; color: var(--muted); margin: 0; }
.target-about a { color: var(--accent); }
.variants-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.25rem;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.card img {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: contain;
  background: #05070a;
}
.card-body { padding: 1rem 1.2rem; flex: 1; display: flex; flex-direction: column; }
.card-title { font-weight: 600; font-size: 1.05rem; margin: 0 0 0.4rem; }
.badges { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
.badge {
  font-size: 0.72rem;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-weight: 600;
  text-transform: uppercase;
}
.badge.sci { background: rgba(63, 185, 80, 0.18); color: var(--badge-sci); }
.badge.art { background: rgba(210, 153, 34, 0.18); color: var(--badge-art); }
.badge.ai { background: rgba(163, 113, 247, 0.18); color: var(--badge-ai); }
.desc { font-size: 0.86rem; color: var(--muted); margin: 0 0 0.75rem; flex: 1; }
.prov { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.85rem; }
.actions { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.btn {
  text-decoration: none;
  font-size: 0.82rem;
  padding: 0.4rem 0.75rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  color: var(--fg);
  background: #1c2331;
}
.btn.primary { background: var(--accent); color: #05070a; border-color: var(--accent); font-weight: 600; }
.science-box, .omitted-box {
  margin-top: 1.25rem;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.2rem;
  font-size: 0.86rem;
}
</style>
</head>
<body>
<header>
  <div>
    <h1>Photonloom — From photons to masterpieces</h1>
    <p>Offline-openable Astrophotography Science &amp; Gallery Report</p>
  </div>
  <div>
    <a class="btn" href=""#,
    );
    html.push_str(&html_escape(&run_link));
    html.push_str(r#"">Run Snapshot ("#);
    html.push_str(&html_escape(&manifest.run_id));
    html.push_str(
        r#")</a>
  </div>
</header>
<main>
"#,
    );

    html.push_str(&format!(
        r#"<div class="meta-bar">
  <strong>Input:</strong> {} &nbsp;|&nbsp;
  <strong>Run ID:</strong> {} &nbsp;|&nbsp;
  <strong>Accelerator:</strong> {} (requested: {})
</div>"#,
        html_escape(&manifest.canonical_input),
        html_escape(&manifest.run_id),
        html_escape(&manifest.accelerator_used),
        html_escape(&manifest.accelerator_requested),
    ));

    if !manifest.warnings.is_empty() {
        html.push_str(r#"<div class="warnings"><strong>Processing Warnings:</strong><ul>"#);
        for w in &manifest.warnings {
            html.push_str(&format!("<li>{}</li>", html_escape(w)));
        }
        html.push_str("</ul></div>");
    }

    for target in &manifest.targets {
        html.push_str(&format!(
            r#"<section class="target-section">
  <h2>Target: {}</h2>
  <div class="variants-grid">"#,
            html_escape(&target.target_label)
        ));
        if let Some(about) = &target.about {
            if about
                .wikipedia_url
                .starts_with("https://en.wikipedia.org/wiki/")
            {
                html.push_str(&format!(
                    r#"<p class="target-about">{} <a href="{}" target="_blank" rel="noopener noreferrer">{} on Wikipedia</a></p>"#,
                    html_escape(&about.summary),
                    html_escape(&about.wikipedia_url),
                    html_escape(&about.name),
                ));
            }
        }

        for var in &target.variants {
            let selected_cls = if var.id == target.selected_variant {
                "card selected"
            } else {
                "card"
            };
            let png_href = rel_asset_path(&var.png_path, &manifest.run_dir, inside_run_dir);
            let tiff_href = rel_asset_path(&var.tiff_path, &manifest.run_dir, inside_run_dir);

            html.push_str(&format!(
                r#"<article class="{}" data-variant="{}">
      <img src="{}" alt="{}" loading="lazy">
      <div class="card-body">
        <h3 class="card-title">{}</h3>
        <div class="badges">
          <span class="badge sci">Palette: {}</span>"#,
                selected_cls,
                html_escape(&var.id),
                html_escape(&png_href),
                html_escape(&var.title),
                html_escape(&var.title),
                html_escape(&var.palette),
            ));
            if var.artistic_interpretation {
                html.push_str(r#"<span class="badge art">Artistic Interpretation</span>"#);
            }
            if var.ai_enhanced {
                html.push_str(r#"<span class="badge ai">AI-Enhanced (Not Measured Detail)</span>"#);
            }
            if let Some(finish) = var.finish {
                html.push_str(&format!(
                    r#"<span class="badge art">Display finish: {:?}</span>"#,
                    finish.preset,
                ));
            }
            if let Some(crop) = var.crop {
                html.push_str(&format!(
                    r#"<span class="badge">Display crop: {}×{}</span>"#,
                    crop.width, crop.height,
                ));
            }
            html.push_str(&format!(
                r#"</div>
        <p class="desc">{}</p>
        <div class="prov">{}</div>
        <div class="actions">
          <a class="btn primary" href="{}" download>Download 16-bit TIFF</a>
          <a class="btn" href="{}" target="_blank">Preview sRGB PNG</a>
        </div>
      </div>
    </article>"#,
                html_escape(&var.description),
                html_escape(&var.provenance),
                html_escape(&tiff_href),
                html_escape(&png_href),
            ));
        }

        html.push_str("</div>");

        // Science FITS integrations
        html.push_str(
            r#"<div class="science-box">
    <strong>Linear 32-bit Float Science Integrations (FITS):</strong>
    <ul>"#,
        );
        for sci in &target.science_integrations {
            let fits_href = rel_asset_path(&sci.fits_path, &manifest.run_dir, inside_run_dir);
            html.push_str(&format!(
                r#"<li><strong>Channel {}:</strong> <a href="{}" download>{}</a> ({}×{}, {:.1}s total, {} accepted, {} rejected)</li>"#,
                html_escape(&sci.channel),
                html_escape(&fits_href),
                html_escape(&sci.fits_path),
                sci.width,
                sci.height,
                sci.total_exposure_s,
                sci.accepted_frames.len(),
                sci.rejected_frames.len()
            ));
        }
        html.push_str("</ul></div>");

        if !target.omitted_variants.is_empty() {
            html.push_str(
                r#"<div class="omitted-box">
    <strong>Omitted Palettes:</strong>
    <ul>"#,
            );
            for om in &target.omitted_variants {
                html.push_str(&format!(
                    "<li><strong>{}:</strong> {}</li>",
                    html_escape(&om.palette),
                    html_escape(&om.reason)
                ));
            }
            html.push_str("</ul></div>");
        }

        html.push_str("</section>");
    }

    html.push_str("</main></body></html>\n");
    Ok(html)
}

fn html_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}
