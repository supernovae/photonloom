//! GPU-offloaded and parallel CPU starless nebula separation, harmonic star inpainting,
//! and standard unscreen/screen star re-composition workflows.

use crate::config::{AcceleratorMode, LinearImage};
use crate::gpu::GpuStacker;
use crate::register::{detect_stars, robust_median_mad};
use anyhow::{Result, bail};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::sync::LazyLock;

/// Multiscale jump-flood / à-trous harmonic relaxation schedule (even length = 24 passes).
/// Coarse passes (stride 4) propagate surrounding nebula continuum across wide star cores,
/// medium passes (stride 2) bridge intermediate skirts, and fine passes (stride 1) ensure
/// C1 pixel-level smoothness.
pub const STARLESS_DIFFUSION_STEPS: [u32; 24] = [
    4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
];

const TILE_CORE: usize = 512;
const TILE_HALO: usize = 32;

static SHARED_GPU_STACKER: LazyLock<Mutex<Option<GpuStacker>>> =
    LazyLock::new(|| Mutex::new(None));

#[derive(Debug, Clone)]
pub struct StarSeparation {
    /// Starless nebula/galaxy/background image with stars inpainted by local continuum.
    pub starless: LinearImage,
    /// Extracted stars-only layer (unscreened in display domain, subtracted in linear domain).
    pub stars: LinearImage,
    /// Continuous stellar support mask in `[0.0, 1.0]` (`width * height`).
    pub star_mask: Vec<f32>,
    /// Number of distinct stellar peaks/blobs masked.
    pub detected_star_count: usize,
    /// True if the harmonic star-inpainting passes executed on a hardware GPU.
    pub used_gpu: bool,
    /// Name of the hardware GPU adapter if GPU offloading was used.
    pub adapter_name: Option<String>,
}

/// CPU implementation of the 24-pass multiscale harmonic star-inpainting kernel.
/// Executes the exact same mathematical recurrence as `STARLESS_SHADER_WGSL` in `src/gpu.rs`
/// and `starless_diffuse_step` in `src/hip_stack.hip`.
pub fn inpaint_starless_tile_cpu(
    width: usize,
    height: usize,
    channels: usize,
    orig_values: &[f32],
    seed_values: &[f32],
    star_mask: &[f32],
    valid_mask: &[u32],
) -> Vec<f32> {
    let np = width * height;
    let total = np * channels;
    if np == 0 || channels == 0 || orig_values.len() != total || seed_values.len() != total {
        return orig_values.to_vec();
    }
    let mut ping = seed_values.to_vec();
    let mut pong = vec![0.0f32; total];
    let w_i = width as isize;
    let h_i = height as isize;

    for (iter, &step) in STARLESS_DIFFUSION_STEPS.iter().enumerate() {
        let s = step as isize;
        let (src, dst) = if iter % 2 == 0 {
            (&ping[..], &mut pong[..])
        } else {
            (&pong[..], &mut ping[..])
        };

        dst.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row_global, dst_row)| {
                let c = row_global / height;
                let y = (row_global % height) as isize;
                let plane_offset = c * np;
                let row_offset = (y as usize) * width;

                for x in 0..width {
                    let p = row_offset + x;
                    let idx = plane_offset + p;
                    let orig = orig_values[idx];
                    let alpha = star_mask[p].clamp(0.0, 1.0);
                    if valid_mask[p] == 0 || alpha <= 1e-5 {
                        dst_row[x] = orig;
                        continue;
                    }
                    let x_i = x as isize;
                    let mut sum_v = 0.0f32;
                    let mut sum_w = 0.0f32;

                    for dy in -1isize..=1 {
                        let ny = (y + dy * s).clamp(0, h_i - 1) as usize;
                        let ny_row = ny * width;
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = (x_i + dx * s).clamp(0, w_i - 1) as usize;
                            let np_idx = ny_row + nx;
                            if valid_mask[np_idx] != 0 {
                                let w_dir = if dx == 0 || dy == 0 { 0.15f32 } else { 0.10f32 };
                                let m_nbr = star_mask[np_idx].clamp(0.0, 1.0);
                                let w = w_dir * (1.0 - 0.5 * m_nbr);
                                sum_v += w * src[plane_offset + np_idx];
                                sum_w += w;
                            }
                        }
                    }

                    let avg = if sum_w > 0.0 {
                        sum_v / sum_w
                    } else {
                        src[idx]
                    };
                    let cand = (1.0 - alpha) * orig + alpha * avg;
                    dst_row[x] = orig.min(cand);
                }
            });
    }

    ping
}

/// Separable O(N) weighted box filter over `src` (`width * height`) using per-pixel `weights`.
fn weighted_box_smooth(
    src: &[f32],
    weights: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = width * height;
    if n == 0 || radius == 0 {
        return (src.to_vec(), weights.to_vec());
    }
    let r = radius.min(width.saturating_sub(1)).max(1);
    let mut h_sum = vec![0.0f32; n];
    let mut h_wt = vec![0.0f32; n];

    h_sum
        .par_chunks_mut(width)
        .zip(h_wt.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (row_s, row_w))| {
            let row = y * width;
            let mut s = 0.0f32;
            let mut w = 0.0f32;
            for x in 0..=r.min(width - 1) {
                let wt = weights[row + x];
                if wt > 0.0 && src[row + x].is_finite() {
                    s += src[row + x] * wt;
                    w += wt;
                }
            }
            for x in 0..width {
                if x > 0 && x + r < width {
                    let add = row + x + r;
                    let wt = weights[add];
                    if wt > 0.0 && src[add].is_finite() {
                        s += src[add] * wt;
                        w += wt;
                    }
                }
                if x > r {
                    let rem = row + x - r - 1;
                    let wt = weights[rem];
                    if wt > 0.0 && src[rem].is_finite() {
                        s -= src[rem] * wt;
                        w -= wt;
                    }
                }
                row_s[x] = s;
                row_w[x] = w;
            }
        });

    let ry = radius.min(height.saturating_sub(1)).max(1);
    let win_area = ((2 * r + 1) * (2 * ry + 1)) as f32;
    let mut out_val = vec![0.0f32; n];
    let mut out_cov = vec![0.0f32; n];
    let mut col_s = vec![0.0f32; width];
    let mut col_w = vec![0.0f32; width];

    for y in 0..=ry.min(height - 1) {
        let row = y * width;
        for x in 0..width {
            col_s[x] += h_sum[row + x];
            col_w[x] += h_wt[row + x];
        }
    }
    for y in 0..height {
        let add_row = (y + ry < height).then_some((y + ry) * width);
        let rem_row = y.checked_sub(ry + 1).map(|py| py * width);
        let row = y * width;
        for x in 0..width {
            if y > 0 {
                if let Some(ar) = add_row {
                    col_s[x] += h_sum[ar + x];
                    col_w[x] += h_wt[ar + x];
                }
            }
            if let Some(rr) = rem_row {
                col_s[x] -= h_sum[rr + x];
                col_w[x] -= h_wt[rr + x];
            }
            let w = col_w[x];
            out_val[row + x] = if w > 1e-6 {
                col_s[x] / w
            } else {
                src[row + x]
            };
            out_cov[row + x] = (w / win_area).clamp(0.0, 1.0);
        }
    }

    (out_val, out_cov)
}

#[derive(Clone, Copy)]
struct StarFootprint {
    x: f32,
    y: f32,
    fwhm: f32,
    snr: f32,
    core_radius: f32,
}

/// Detects unsaturated PSF stars, multiscale compact wavelet point-sources, and saturated
/// bright-star plateaus/halos to build a smooth continuous star mask `M_star(x, y) in [0, 1]`.
pub fn build_star_mask(image: &LinearImage) -> (Vec<f32>, usize) {
    let w = image.width;
    let h = image.height;
    let n = w.checked_mul(h).unwrap_or(0);
    if w < 7 || h < 7 || n == 0 || !matches!(image.channels, 1 | 3) {
        return (vec![0.0; n], 0);
    }

    // Build combined detection signal: luminance + narrowband channel peak so colored
    // single-band halos (e.g. OIII blue/teal halos or Ha red spikes) are captured.
    let mut det_sig = vec![0.0f32; n];
    let mut valid_u8 = vec![0u8; n];
    let mut valid_f32 = vec![0.0f32; n];
    for p in 0..n {
        if image.mask.get(p).copied().unwrap_or(0) == 0 {
            continue;
        }
        if image.channels == 1 {
            let v = image.data[p];
            if v.is_finite() {
                det_sig[p] = v;
                valid_u8[p] = 1;
                valid_f32[p] = 1.0;
            }
        } else {
            let r = image.data[p];
            let g = image.data[n + p];
            let b = image.data[2 * n + p];
            if r.is_finite() && g.is_finite() && b.is_finite() {
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let peak_ch = r.max(g).max(b);
                det_sig[p] = 0.55 * lum + 0.45 * peak_ch;
                valid_u8[p] = 1;
                valid_f32[p] = 1.0;
            }
        }
    }

    let valid_count = valid_u8.iter().filter(|&&v| v != 0).count();
    if valid_count < 16 {
        return (vec![0.0; n], 0);
    }

    let (global_med, global_mad) = robust_median_mad(&det_sig, &valid_u8);
    let global_sigma = (global_mad * 1.4826).max(1e-5);

    // Estimate true high-frequency noise floor from adjacent horizontal/vertical differences
    // so bright nebulosity variations do not inflate the stellar detection threshold.
    let step = (n / 131_072).max(1);
    let mut hf_diffs = Vec::with_capacity(n / step + 1);
    let sky_cap = global_med + 2.5 * global_sigma;
    for i in (0..n).step_by(step) {
        if i % w > 0
            && valid_u8[i] != 0
            && valid_u8[i - 1] != 0
            && det_sig[i] <= sky_cap
            && det_sig[i - 1] <= sky_cap
        {
            hf_diffs.push((det_sig[i] - det_sig[i - 1]).abs());
        }
    }
    let noise_sigma = if hf_diffs.len() >= 16 {
        let mid = hf_diffs.len() / 2;
        hf_diffs.select_nth_unstable_by(mid, f32::total_cmp);
        let hf_sigma = hf_diffs[mid] * (1.0 / (0.67449 * std::f32::consts::SQRT_2));
        global_sigma.min(hf_sigma.max(global_sigma * 0.12)).max(1e-5)
    } else {
        global_sigma
    };

    // Pass 1 local backgrounds at medium (r=8) and wide (r=18) radii with initial star cap.
    let star_cap = global_med + 4.0 * global_sigma;
    let capped_weights: Vec<f32> = (0..n)
        .map(|p| {
            if valid_u8[p] == 0 {
                0.0
            } else if det_sig[p] > star_cap {
                0.08
            } else {
                1.0
            }
        })
        .collect();
    let (bg_med, _) = weighted_box_smooth(&det_sig, &capped_weights, w, h, 8.min(w / 3).max(2));
    let (bg_wide, _) = weighted_box_smooth(&det_sig, &capped_weights, w, h, 18.min(w / 2).max(3));

    let mut footprints = Vec::<StarFootprint>::new();

    // 1. Unsaturated PSF stars from standard detector
    let det_image = LinearImage {
        width: w,
        height: h,
        channels: 1,
        data: det_sig.clone(),
        mask: valid_u8.clone(),
        bayer: None,
        exposure: image.exposure,
    };
    for s in detect_stars(&det_image) {
        if s.fwhm.is_finite() && s.fwhm >= 0.8 && s.fwhm <= 18.0 && s.eccentricity <= 0.78 {
            let p = (s.y.round() as usize).min(h - 1) * w + (s.x.round() as usize).min(w - 1);
            let local_excess = (det_sig[p] - bg_med[p]).max(s.peak);
            let snr = (local_excess / noise_sigma).max(5.0);
            footprints.push(StarFootprint {
                x: s.x,
                y: s.y,
                fwhm: s.fwhm.clamp(1.4, 12.0),
                snr,
                core_radius: (s.fwhm * 0.75).clamp(1.2, 6.0),
            });
        }
    }

    // 2. Local multiscale morphological / wavelet stellar point-source detector.
    // Captures stars embedded in bright nebulosity and fainter field stars while rejecting
    // extended nebula ridges via an 8-compass-direction drop-off test.
    let border = 3usize;
    let dirs: [(isize, isize); 8] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ];
    let local_fps: Vec<StarFootprint> = (border..(h - border))
        .into_par_iter()
        .flat_map_iter(|y| {
            let mut row_fps = Vec::new();
            for x in border..(w - border) {
                let p = y * w + x;
                if valid_u8[p] == 0 {
                    continue;
                }
                let val = det_sig[p];
                let local_bg = bg_med[p].min(bg_wide[p]);
                let excess = val - local_bg;
                if excess < 4.0 * noise_sigma {
                    continue;
                }

                let mut is_max = true;
                let mut elevated_nbrs = 0usize;
                'nbr: for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let np = ((y as isize + dy) as usize) * w + ((x as isize + dx) as usize);
                        let nv = det_sig[np];
                        if nv > val || (nv == val && (dy < 0 || (dy == 0 && dx < 0))) {
                            is_max = false;
                            break 'nbr;
                        }
                        if nv - local_bg > 1.3 * noise_sigma {
                            elevated_nbrs += 1;
                        }
                    }
                }
                if !is_max || elevated_nbrs < 1 {
                    continue;
                }

                let mut drop_dirs = 0usize;
                for &(dx, dy) in &dirs {
                    let mut dropped = false;
                    for dist in [3isize, 5, 8] {
                        let nx = (x as isize + dx * dist).clamp(0, w as isize - 1) as usize;
                        let ny = (y as isize + dy * dist).clamp(0, h as isize - 1) as usize;
                        let n_excess = (det_sig[ny * w + nx] - bg_wide[ny * w + nx]).max(0.0);
                        if n_excess < 0.55 * excess {
                            dropped = true;
                            break;
                        }
                    }
                    if dropped {
                        drop_dirs += 1;
                    }
                }
                if drop_dirs < 6 {
                    continue;
                }

                let mut sum_w = 0.0f32;
                let mut sum_x = 0.0f32;
                let mut sum_y = 0.0f32;
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        let nx = (x as isize + dx) as usize;
                        let ny = (y as isize + dy) as usize;
                        let wt = (det_sig[ny * w + nx] - local_bg - noise_sigma).max(0.0);
                        sum_w += wt;
                        sum_x += wt * (nx as f32);
                        sum_y += wt * (ny as f32);
                    }
                }
                if sum_w <= 0.0 {
                    continue;
                }
                let cx = sum_x / sum_w;
                let cy = sum_y / sum_w;
                let mut m_r2 = 0.0f32;
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        let nx = (x as isize + dx) as usize;
                        let ny = (y as isize + dy) as usize;
                        let wt = (det_sig[ny * w + nx] - local_bg - noise_sigma).max(0.0);
                        let r2 = (nx as f32 - cx).powi(2) + (ny as f32 - cy).powi(2);
                        m_r2 += wt * r2;
                    }
                }
                let fwhm = (2.3548 * (0.5 * m_r2 / sum_w).max(0.15).sqrt()).clamp(1.3, 9.0);
                row_fps.push(StarFootprint {
                    x: cx,
                    y: cy,
                    fwhm,
                    snr: (excess / noise_sigma).max(4.0),
                    core_radius: (fwhm * 0.75).clamp(1.2, 5.0),
                });
            }
            row_fps
        })
        .collect();
    footprints.extend(local_fps);

    // 3. Saturated / bright star cores and halos
    let max_sig = det_sig
        .iter()
        .zip(&valid_u8)
        .filter(|&(_, &m)| m != 0)
        .map(|(&v, _)| v)
        .fold(0.0f32, f32::max);
    let bright_thresh = if max_sig <= 1.05 {
        0.92f32.min(global_med + 10.0 * global_sigma)
    } else {
        (global_med + 12.0 * global_sigma).min(max_sig * 0.85)
    };
    let mut visited = vec![false; n];
    for p in 0..n {
        if visited[p]
            || valid_u8[p] == 0
            || det_sig[p] < bright_thresh
            || det_sig[p] - bg_wide[p] < 6.0 * noise_sigma
        {
            continue;
        }
        let mut queue = vec![p];
        visited[p] = true;
        let mut head = 0usize;
        let mut sx = 0usize;
        let mut sy = 0usize;
        let mut min_x = p % w;
        let mut max_x = min_x;
        let mut min_y = p / w;
        let mut max_y = min_y;
        let mut peak_excess = 0.0f32;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            let cx = cur % w;
            let cy = cur / w;
            sx += cx;
            sy += cy;
            min_x = min_x.min(cx);
            max_x = max_x.max(cx);
            min_y = min_y.min(cy);
            max_y = max_y.max(cy);
            peak_excess = peak_excess.max(det_sig[cur] - bg_wide[cur]);
            if queue.len() > 1600 {
                break;
            }
            for (nx, ny) in [
                (cx.saturating_sub(1), cy),
                ((cx + 1).min(w - 1), cy),
                (cx, cy.saturating_sub(1)),
                (cx, (cy + 1).min(h - 1)),
            ] {
                let np = ny * w + nx;
                if !visited[np]
                    && valid_u8[np] != 0
                    && det_sig[np] >= bright_thresh * 0.92
                    && det_sig[np] - bg_wide[np] >= 4.5 * noise_sigma
                {
                    visited[np] = true;
                    queue.push(np);
                }
            }
        }
        let area = queue.len();
        let span_x = (max_x - min_x + 1) as f32;
        let span_y = (max_y - min_y + 1) as f32;
        let aspect = span_x.max(span_y) / span_x.min(span_y).max(1.0);
        if (1..=1200).contains(&area) && aspect <= 2.4 {
            let cx = sx as f32 / area as f32;
            let cy = sy as f32 / area as f32;
            let r_blob = ((area as f32) / std::f32::consts::PI).sqrt().max(1.5);
            footprints.push(StarFootprint {
                x: cx,
                y: cy,
                fwhm: (r_blob * 2.2).clamp(2.5, 16.0),
                snr: (peak_excess / noise_sigma).max(12.0),
                core_radius: (r_blob * 1.4).clamp(2.0, 14.0),
            });
        }
    }

    // Paint radial star support profiles into `raw_mask`
    let mut raw_mask = vec![0.0f32; n];
    for fp in &footprints {
        let sigma = (fp.fwhm / 2.3548).clamp(0.65, 7.0);
        let prom = (1.0 + 0.24 * (1.0 + fp.snr / 8.0).ln()).clamp(1.0, 2.8);
        let r_core = fp.core_radius.max(1.55 * sigma * prom).clamp(1.2, 16.0);
        let r_outer = (3.8 * sigma * prom).max(r_core + 2.2).clamp(3.0, 32.0);
        let x0 = ((fp.x - r_outer).floor() as isize).max(0) as usize;
        let y0 = ((fp.y - r_outer).floor() as isize).max(0) as usize;
        let x1 = ((fp.x + r_outer).ceil() as isize).min(w as isize - 1) as usize;
        let y1 = ((fp.y + r_outer).ceil() as isize).min(h as isize - 1) as usize;
        let span = (r_outer - r_core).max(1.0);

        for yy in y0..=y1 {
            let dy = yy as f32 - fp.y;
            for xx in x0..=x1 {
                let p = yy * w + xx;
                if valid_u8[p] == 0 {
                    continue;
                }
                let dx = xx as f32 - fp.x;
                let r = (dx * dx + dy * dy).sqrt();
                if r <= r_core {
                    raw_mask[p] = 1.0;
                } else if r < r_outer {
                    let t = ((r - r_core) / span).clamp(0.0, 1.0);
                    let w_skirt =
                        (-2.6 * t * t).exp() * 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                    raw_mask[p] = raw_mask[p].max(w_skirt);
                }
            }
        }
    }

    if footprints.is_empty() {
        return (raw_mask, 0);
    }

    // Smooth star mask with a 1-px weighted box pass for C1 feather transitions
    let (mut smoothed, _) = weighted_box_smooth(&raw_mask, &valid_f32, w, h, 1);
    for p in 0..n {
        if valid_u8[p] == 0 {
            smoothed[p] = 0.0;
        } else {
            // Preserve 1.0 at star cores while keeping feathered skirts smooth
            smoothed[p] = smoothed[p].max(raw_mask[p] * 0.92).clamp(0.0, 1.0);
            if smoothed[p] < 1e-3 {
                smoothed[p] = 0.0;
            }
        }
    }

    (smoothed, footprints.len())
}

/// Computes the initial multiscale star-excluded continuum seed for all channels of `image`.
pub fn build_continuum_seed(image: &LinearImage, star_mask: &[f32]) -> Vec<f32> {
    let w = image.width;
    let h = image.height;
    let n = w * h;
    let channels = image.channels;
    if n == 0 || channels == 0 {
        return image.data.clone();
    }

    // Weight unmasked pixels strongly and star-core pixels near zero.
    let bg_weights: Vec<f32> = (0..n)
        .map(|p| {
            if image.mask[p] == 0 {
                0.0
            } else {
                let inv = (1.0 - star_mask[p].clamp(0.0, 1.0)).max(0.0);
                (inv * inv * inv).max(1e-5)
            }
        })
        .collect();

    let r_small = 5.min(w / 3).min(h / 3).max(1);
    let r_med = 13.min(w / 2).min(h / 2).max(2);
    let r_large = 26.min(w.saturating_sub(1)).min(h.saturating_sub(1)).max(3);

    let mut seed = vec![0.0f32; n * channels];
    for c in 0..channels {
        let plane = image.plane(c);
        let (s_small, cov_small) = weighted_box_smooth(plane, &bg_weights, w, h, r_small);
        let (s_med, cov_med) = weighted_box_smooth(plane, &bg_weights, w, h, r_med);
        let (s_large, _) = weighted_box_smooth(plane, &bg_weights, w, h, r_large);
        let out_plane = &mut seed[c * n..(c + 1) * n];
        for p in 0..n {
            let orig = plane[p];
            let alpha = star_mask[p].clamp(0.0, 1.0);
            if image.mask[p] == 0 || !orig.is_finite() || alpha <= 1e-5 {
                out_plane[p] = orig;
                continue;
            }
            let w1 = (cov_small[p] * 4.0).clamp(0.0, 1.0);
            let w2 = (cov_med[p] * 3.0).clamp(0.0, 1.0);
            let cont_med = w2 * s_med[p] + (1.0 - w2) * s_large[p];
            let continuum = w1 * s_small[p] + (1.0 - w1) * cont_med;
            let blended = (1.0 - alpha) * orig + alpha * orig.min(continuum);
            out_plane[p] = orig.min(blended);
        }
    }
    seed
}

fn is_display_range(image: &LinearImage) -> bool {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let step = (image.data.len() / 65_536).max(1);
    let n = image.pixel_count().max(1);
    for (idx, &v) in image.data.iter().enumerate().step_by(step) {
        if image.mask[idx % n] != 0 && v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    min_v >= -0.05 && max_v <= 1.05
}

/// Extracts the stars-only layer from `original` and `starless`.
/// Uses standard astrophotography Unscreen (`(I - S) / (1 - S)`) in `[0, 1]` display space
/// and linear subtraction (`max(I - S, 0)`) in linear space.
pub fn extract_stars_layer(original: &LinearImage, starless: &LinearImage) -> LinearImage {
    let w = original.width;
    let h = original.height;
    let c = original.channels;
    let n = w * h;
    let display = is_display_range(original);
    let mut stars = LinearImage::new(w, h, c);
    stars.mask = original.mask.clone();
    stars.exposure = original.exposure;

    for ch in 0..c {
        let orig_p = original.plane(ch);
        let s_less = starless.plane(ch);
        let out_p = stars.plane_mut(ch);
        for i in 0..n {
            if original.mask[i] == 0 || !orig_p[i].is_finite() || !s_less[i].is_finite() {
                out_p[i] = 0.0;
                continue;
            }
            let diff = (orig_p[i] - s_less[i]).max(0.0);
            out_p[i] = if display {
                let denom = (1.0 - s_less[i].clamp(0.0, 0.9999)).max(1e-4);
                (diff / denom).clamp(0.0, 1.0)
            } else {
                diff
            };
        }
    }
    stars
}

/// Blends `stars` back into `starless` at `blend` weight in `[0.0, 1.0]` using Screen blending
/// (`1 - (1 - S) * (1 - w * stars)`) in display space or additive blending in linear space.
pub fn screen_blend_stars(starless: &mut LinearImage, stars: &LinearImage, blend: f32) {
    let w_blend = blend.clamp(0.0, 1.0);
    if w_blend <= 0.0
        || starless.width != stars.width
        || starless.height != stars.height
        || starless.channels != stars.channels
    {
        return;
    }
    let n = starless.pixel_count();
    let display = is_display_range(starless);
    let channels = starless.channels;
    let mask = &starless.mask;
    for ch in 0..channels {
        let s_plane = &mut starless.data[ch * n..(ch + 1) * n];
        let st_plane = stars.plane(ch);
        for i in 0..n {
            if mask[i] == 0 || !s_plane[i].is_finite() || !st_plane[i].is_finite() {
                continue;
            }
            let s_val = s_plane[i];
            let st_val = (st_plane[i] * w_blend).max(0.0);
            s_plane[i] = if display {
                let a = s_val.clamp(0.0, 1.0);
                let b = st_val.clamp(0.0, 1.0);
                (1.0 - (1.0 - a) * (1.0 - b)).clamp(0.0, 1.0)
            } else {
                s_val + st_val
            };
        }
    }
}

/// Runs tiled harmonic star inpainting using an optional `GpuStacker` (with 32-px reflected/clamped
/// tile halos for large frames) and falls back to `inpaint_starless_tile_cpu` when `stacker` is `None`.
pub fn separate_stars_with_stacker(
    image: &LinearImage,
    mut stacker: Option<&mut GpuStacker>,
) -> Result<StarSeparation> {
    let w = image.width;
    let h = image.height;
    let channels = image.channels;
    let n = w.checked_mul(h).unwrap_or(0);
    if !matches!(channels, 1 | 3)
        || image.data.len() != n.saturating_mul(channels)
        || image.mask.len() != n
    {
        bail!("Star separation requires a valid 1-channel or 3-channel LinearImage");
    }

    let (star_mask, detected_star_count) = build_star_mask(image);
    if detected_star_count == 0 || star_mask.iter().all(|&m| m <= 1e-5) {
        let mut empty_stars = LinearImage::new(w, h, channels);
        empty_stars.mask = image.mask.clone();
        empty_stars.exposure = image.exposure;
        let adapter_name = stacker.as_ref().map(|s| s.adapter_name.clone());
        return Ok(StarSeparation {
            starless: image.clone(),
            stars: empty_stars,
            star_mask,
            detected_star_count: 0,
            used_gpu: stacker.is_some(),
            adapter_name,
        });
    }

    let seed = build_continuum_seed(image, &star_mask);
    let mut starless = image.clone();
    let mut used_gpu = false;
    let mut adapter_name = None;

    for y0 in (0..h).step_by(TILE_CORE) {
        for x0 in (0..w).step_by(TILE_CORE) {
            let tw = (w - x0).min(TILE_CORE);
            let th = (h - y0).min(TILE_CORE);
            let hx0 = x0.saturating_sub(TILE_HALO);
            let hy0 = y0.saturating_sub(TILE_HALO);
            let hx1 = (x0 + tw + TILE_HALO).min(w);
            let hy1 = (y0 + th + TILE_HALO).min(h);
            let hw = hx1 - hx0;
            let hh = hy1 - hy0;
            let h_np = hw * hh;

            let mut t_orig = vec![0.0f32; h_np * channels];
            let mut t_seed = vec![0.0f32; h_np * channels];
            let mut t_mask = vec![0.0f32; h_np];
            let mut t_valid = vec![0u32; h_np];

            for ty in 0..hh {
                let src_row = (hy0 + ty) * w + hx0;
                let dst_row = ty * hw;
                t_mask[dst_row..dst_row + hw].copy_from_slice(&star_mask[src_row..src_row + hw]);
                for tx in 0..hw {
                    let p = src_row + tx;
                    let valid = image.mask[p] != 0
                        && (0..channels).all(|c| image.data[c * n + p].is_finite());
                    t_valid[dst_row + tx] = u32::from(valid);
                }
                for c in 0..channels {
                    let s_off = c * n + src_row;
                    let d_off = c * h_np + dst_row;
                    t_orig[d_off..d_off + hw].copy_from_slice(&image.data[s_off..s_off + hw]);
                    t_seed[d_off..d_off + hw].copy_from_slice(&seed[s_off..s_off + hw]);
                }
            }

            let inpainted = if let Some(gpu) = stacker.as_deref_mut() {
                let out = gpu.inpaint_starless_tile(
                    hw, hh, channels, &t_orig, &t_seed, &t_mask, &t_valid,
                )?;
                used_gpu = true;
                if adapter_name.is_none() {
                    adapter_name = Some(gpu.adapter_name.clone());
                }
                out
            } else {
                inpaint_starless_tile_cpu(hw, hh, channels, &t_orig, &t_seed, &t_mask, &t_valid)
            };

            // Copy interior tile back to `starless`
            let inner_x = x0 - hx0;
            let inner_y = y0 - hy0;
            for ty in 0..th {
                let src_row = (inner_y + ty) * hw + inner_x;
                let dst_row = (y0 + ty) * w + x0;
                for c in 0..channels {
                    starless.data[c * n + dst_row..c * n + dst_row + tw]
                        .copy_from_slice(&inpainted[c * h_np + src_row..c * h_np + src_row + tw]);
                }
            }
        }
    }

    let stars = extract_stars_layer(image, &starless);
    Ok(StarSeparation {
        starless,
        stars,
        star_mask,
        detected_star_count,
        used_gpu,
        adapter_name,
    })
}

/// Separates `image` into a starless nebula image and a stars-only layer, offloading the
/// 24-pass multiscale harmonic inpainting to the GPU when available (`AcceleratorMode::Auto` or `Gpu`)
/// and falling back to parallel CPU (`rayon`) when `AcceleratorMode::Cpu` or no GPU is present.
pub fn separate_stars(image: &LinearImage, accelerator: AcceleratorMode) -> Result<StarSeparation> {
    match accelerator {
        AcceleratorMode::Cpu => separate_stars_with_stacker(image, None),
        AcceleratorMode::Gpu => {
            let mut guard = SHARED_GPU_STACKER.lock();
            if guard.is_none() {
                *guard = Some(GpuStacker::try_new()?);
            }
            separate_stars_with_stacker(image, guard.as_mut())
        }
        AcceleratorMode::Auto => {
            let mut guard = SHARED_GPU_STACKER.lock();
            if guard.is_none() {
                if let Ok(gpu) = GpuStacker::try_new() {
                    *guard = Some(gpu);
                }
            }
            if guard.is_some() {
                if let Ok(sep) = separate_stars_with_stacker(image, guard.as_mut()) {
                    return Ok(sep);
                }
            }
            drop(guard);
            separate_stars_with_stacker(image, None)
        }
    }
}

/// Convenience helper returning only the starless `LinearImage`.
pub fn remove_stars(image: &LinearImage, accelerator: AcceleratorMode) -> Result<LinearImage> {
    Ok(separate_stars(image, accelerator)?.starless)
}
