use crate::calibrate::{SpooledFrame, TargetCalibrationOutput, read_spooled_image};
use crate::config::{
    AcceleratorMode, Channel, FileFormat, LinearImage, PhaseProgress, ProcessOptions,
    RejectedFrameRecord, ScienceArtifact,
};
use crate::fits::{WcsTan, inspect_fits_header, write_fits_linear};
use crate::gpu::GpuStacker;
use crate::register::{
    AffineTransform, DetectedStar, FrameQuality, detect_stars, evaluate_spooled_frame_quality,
    match_star_fields, warp_linear_image_lanczos3, warp_spooled_tile_lanczos3,
};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ChannelIntegration {
    pub channel: Channel,
    pub image: LinearImage,
    pub aligned_to_target: bool,
    pub total_exposure_s: f32,
    pub accepted_frames: Vec<String>,
    pub rejected_frames: Vec<RejectedFrameRecord>,
    pub fits_rel_path: String,
    /// Original sharpest reference light: this is the native pixel grid of the integration.
    pub reference_frame_path: Option<PathBuf>,
    pub reference_wcs: Option<WcsTan>,
    /// Present only on the exported target-reference grid; no unmeasured cross-filter WCS.
    pub wcs: Option<WcsTan>,
    /// Geometric full-coverage footprint, independent of isolated bad pixels.
    pub valid_region: Vec<(f64, f64)>,
    pub recommended_crop: Option<crate::crop::CropRect>,
}

#[derive(Debug, Clone)]
pub struct TargetIntegrationOutput {
    pub target_id: String,
    pub target_label: String,
    pub width: usize,
    pub height: usize,
    pub integrations: BTreeMap<Channel, ChannelIntegration>,
    pub warnings: Vec<String>,
    pub rejected_frames: Vec<RejectedFrameRecord>,
    pub used_gpu: bool,
    pub cross_filter_aligned: bool,
}

/// CPU implementation of per-pixel weighted/clipped tile integration.
/// Executes the exact same mathematical algorithm (exposure/inverse-variance weighting +
/// 3-iteration 3-sigma Winsorized clipping) as the WGSL compute shader in `src/gpu.rs`.
pub fn integrate_tile_cpu(
    num_pixels: usize,
    num_frames: usize,
    tile_values: &[f32],
    tile_masks_u32: &[u32],
    weights: &[f32],
) -> (Vec<f32>, Vec<u8>) {
    let mut out_values = vec![0.0f32; num_pixels];
    let mut out_masks = vec![0u8; num_pixels];

    out_values
        .par_iter_mut()
        .zip(out_masks.par_iter_mut())
        .enumerate()
        .with_min_len(1024)
        .for_each_init(
            || {
                (
                    Vec::<f32>::with_capacity(num_frames),
                    Vec::<f32>::with_capacity(num_frames),
                    Vec::<f32>::with_capacity(num_frames),
                    Vec::<f32>::with_capacity(num_frames),
                )
            },
            |(vals, wts, sorted, devs), (p, (out_v, out_m))| {
                vals.clear();
                wts.clear();

                for k in 0..num_frames {
                    let idx = k * num_pixels + p;
                    if tile_masks_u32[idx] != 0 {
                        vals.push(tile_values[idx]);
                        wts.push(weights[k]);
                    }
                }

                let m = vals.len();
                if m == 0 {
                    *out_v = 0.0;
                    *out_m = 0;
                    return;
                }
                if m == 1 {
                    *out_v = vals[0];
                    *out_m = 1;
                    return;
                }
                if m == 2 {
                    let wsum = wts[0] + wts[1];
                    *out_v = if wsum > 0.0 {
                        (wts[0] * vals[0] + wts[1] * vals[1]) / wsum
                    } else {
                        0.5 * (vals[0] + vals[1])
                    };
                    *out_m = 1;
                    return;
                }

                sorted.clear();
                sorted.extend_from_slice(vals);
                sorted.sort_unstable_by(|a, b| a.total_cmp(b));
                let half = m / 2;
                let mut mu = if (m & 1) == 1 {
                    sorted[half]
                } else {
                    0.5 * (sorted[half - 1] + sorted[half])
                };

                devs.clear();
                devs.extend(sorted.iter().map(|&v| (v - mu).abs()));
                devs.sort_unstable_by(|a, b| a.total_cmp(b));
                let mad = if (m & 1) == 1 {
                    devs[half]
                } else {
                    0.5 * (devs[half - 1] + devs[half])
                };

                let mut sigma = (1.4826 * mad).max(1e-4 * mu.abs().max(1.0));

                for _ in 0..3 {
                    let low = mu - 3.0 * sigma;
                    let high = mu + 3.0 * sigma;
                    let mut w_sum = 0.0f32;
                    let mut wy_sum = 0.0f32;
                    for i in 0..m {
                        let u = vals[i].clamp(low, high);
                        let w = wts[i];
                        w_sum += w;
                        wy_sum += w * u;
                    }
                    if w_sum <= 0.0 {
                        break;
                    }
                    let mu_new = wy_sum / w_sum;
                    let mut var_sum = 0.0f32;
                    for i in 0..m {
                        let u = vals[i].clamp(low, high);
                        let diff = u - mu_new;
                        var_sum += wts[i] * diff * diff;
                    }
                    let var_val = var_sum / w_sum;
                    mu = mu_new;
                    sigma = (1.134 * var_val.max(0.0).sqrt()).max(1e-4 * mu.abs().max(1.0));
                }

                let final_low = mu - 3.0 * sigma;
                let final_high = mu + 3.0 * sigma;
                let mut inlier_w_sum = 0.0f32;
                let mut inlier_val_sum = 0.0f32;
                for i in 0..m {
                    let v = vals[i];
                    if v >= final_low && v <= final_high {
                        let w = wts[i];
                        inlier_w_sum += w;
                        inlier_val_sum += w * v;
                    }
                }

                *out_v = if inlier_w_sum > 0.0 {
                    inlier_val_sum / inlier_w_sum
                } else {
                    mu
                };
                *out_m = 1;
            },
        );

    (out_values, out_masks)
}

/// Bound host tile buffers, worker clipping scratch and concurrent warp patches.
fn cpu_tile_dim(num_frames: usize) -> Result<usize> {
    const BUDGET: usize = 256 * 1024 * 1024;
    let workers = rayon::current_num_threads();
    let fixed_bytes = num_frames
        .checked_mul(16 * workers + 4)
        .context("CPU clipping scratch size overflow")?;
    let per_pixel = num_frames
        .checked_mul(8)
        .and_then(|n| n.checked_add(20 * workers + 8))
        .context("CPU tile size overflow")?;
    let pixels = BUDGET.checked_sub(fixed_bytes).unwrap_or(0) / per_pixel.max(1);
    if pixels == 0 {
        bail!(
            "Cannot fit one integration pixel with {num_frames} frames and {workers} workers in the 256 MiB tile budget; reduce RAYON_NUM_THREADS or use GPU integration"
        );
    }
    Ok(((pixels as f64).sqrt().floor() as usize).min(512))
}

struct RegisteredSub {
    spooled: SpooledFrame,
    transform: AffineTransform,
}

/// Registers and integrates the calibrated frames for one channel.
pub fn integrate_target_channels(
    target_id: &str,
    target_label: &str,
    cal_output: &TargetCalibrationOutput,
    options: &ProcessOptions,
    progress_range: (f32, f32),
) -> Result<TargetIntegrationOutput> {
    let accelerator = options.accelerator;
    let width = cal_output.width;
    let height = cal_output.height;
    let npix = width * height;

    let mut warnings = Vec::new();
    let mut all_rejected = Vec::new();
    let mut used_gpu = false;

    // Only initialize a GPU when this channel can actually dispatch on it.
    let max_subs_in_channel = cal_output
        .spooled_by_channel
        .values()
        .map(|v| v.len())
        .max()
        .unwrap_or(0);
    let want_gpu = match accelerator {
        AcceleratorMode::Cpu => false,
        AcceleratorMode::Gpu => true,
        AcceleratorMode::Auto => max_subs_in_channel >= 3 && npix >= 8_000_000,
    };

    let mut gpu_stacker: Option<GpuStacker> = if want_gpu {
        match GpuStacker::try_new() {
            Ok(stacker) => {
                options.report_progress(
                    &format!("Using GPU: {}", stacker.adapter_name),
                    progress_range.0,
                );
                Some(stacker)
            }
            Err(e) => {
                if accelerator == AcceleratorMode::Gpu {
                    return Err(e.context(
                        "Forced GPU acceleration (--accelerator gpu) failed: no compatible GPU adapter/device available",
                    ));
                }
                warnings.push(format!(
                    "Auto GPU acceleration unavailable ({:#}); falling back to CPU integration",
                    e
                ));
                None
            }
        }
    } else {
        None
    };

    // Calibration and demosaicing preserve native pixel geometry. Do not attach WCS
    // if the selected original has different dimensions or is not a FITS light.
    let original_wcs = |frame: &crate::config::Frame| {
        if frame.format != FileFormat::Fits || frame.width != width || frame.height != height {
            return None;
        }
        inspect_fits_header(&frame.path).ok().and_then(|meta| {
            (meta.width == width && meta.height == height).then_some(meta.wcs).flatten()
        })
    };

    let mut integrations: BTreeMap<Channel, ChannelIntegration> = BTreeMap::new();

    for (&channel, spooled_frames) in &cal_output.spooled_by_channel {
        if spooled_frames.is_empty() {
            continue;
        }

        let mut ch_rejected = Vec::new();

        // Single-light input passes through calibrated as linear integration (warn no rejection benefit)
        if spooled_frames.len() == 1 {
            let only = &spooled_frames[0];
            warnings.push(format!(
                "Target '{}' channel {}: single light frame '{}'; passing through calibrated linear frame without outlier rejection",
                target_label,
                channel,
                only.original_frame.path.display()
            ));
            let img = read_spooled_image(&only.spool_path)?;

            integrations.insert(
                channel,
                ChannelIntegration {
                    channel,
                    image: img,
                    aligned_to_target: true,
                    total_exposure_s: only.exposure,
                    accepted_frames: vec![only.original_frame.path.to_string_lossy().to_string()],
                    rejected_frames: Vec::new(),
                    reference_frame_path: Some(only.original_frame.path.clone()),
                    reference_wcs: original_wcs(&only.original_frame),
                    wcs: None,
                    fits_rel_path: String::new(),
                    valid_region: crate::crop::frame_region(
                        width,
                        height,
                        [AffineTransform::identity()],
                    ),
                    recommended_crop: None,
                },
            );
            continue;
        }

        // Evaluate star quality for each frame in this channel
        let mut qualities: Vec<Option<FrameQuality>> = Vec::with_capacity(spooled_frames.len());
        let mut valid_fwhms = Vec::new();
        let stage = format!("Measuring native stars ({channel})");
        let quality_end = progress_range.0 + 0.20 * (progress_range.1 - progress_range.0);
        let mut quality_progress = PhaseProgress::new(
            options,
            &stage,
            spooled_frames.len(),
            (progress_range.0, quality_end),
        );
        for sf in spooled_frames {
            let q = evaluate_spooled_frame_quality(sf)?;
            if q.star_count >= 12 {
                valid_fwhms.push(q.median_fwhm);
                qualities.push(Some(q));
            } else {
                let rec = RejectedFrameRecord {
                    path: sf.original_frame.path.to_string_lossy().to_string(),
                    reason: format!(
                        "Insufficient unsaturated stars detected ({}, minimum 12 required)",
                        q.star_count
                    ),
                };
                ch_rejected.push(rec.clone());
                all_rejected.push(rec);
                qualities.push(None);
            }
            quality_progress.report(qualities.len());
        }

        if valid_fwhms.is_empty() {
            bail!(
                "Target '{}' channel {}: all {} light frames failed star detection (<12 stars)",
                target_label,
                channel,
                spooled_frames.len()
            );
        }
        valid_fwhms.sort_by(|a, b| a.total_cmp(b));
        let group_median_fwhm = valid_fwhms[valid_fwhms.len() / 2].max(1.5);

        // Filter out trailing/poor FWHM frames and choose the sharpest frame as reference
        let mut ref_idx: Option<usize> = None;
        let mut best_score = -1.0f32;
        for (idx, q_opt) in qualities.iter_mut().enumerate() {
            if let Some(q) = q_opt {
                if q.median_eccentricity > 0.85 || q.median_fwhm > 2.4 * group_median_fwhm.max(2.5)
                {
                    let rec = RejectedFrameRecord {
                        path: spooled_frames[idx]
                            .original_frame
                            .path
                            .to_string_lossy()
                            .to_string(),
                        reason: format!(
                            "Trailing or degraded PSF (median FWHM {:.2} px, eccentricity {:.2})",
                            q.median_fwhm, q.median_eccentricity
                        ),
                    };
                    ch_rejected.push(rec.clone());
                    all_rejected.push(rec);
                    *q_opt = None;
                    continue;
                }
                if best_score < 0.0 || q.sharpness_score > best_score * 1.05 {
                    best_score = q.sharpness_score;
                    ref_idx = Some(idx);
                }
            }
        }

        let Some(ref_idx) = ref_idx else {
            bail!(
                "Target '{}' channel {}: no sharp reference frame passed quality checks",
                target_label,
                channel
            );
        };
        let ref_stars = qualities[ref_idx].as_ref().unwrap().stars.clone();

        // Register all remaining frames against ref_stars
        let mut registered_subs: Vec<RegisteredSub> = Vec::with_capacity(spooled_frames.len());
        for (idx, sf) in spooled_frames.iter().enumerate() {
            let Some(q) = &qualities[idx] else {
                continue;
            };
            if idx == ref_idx {
                registered_subs.push(RegisteredSub {
                    spooled: sf.clone(),
                    transform: AffineTransform::identity(),
                });
                continue;
            }
            match match_star_fields(&ref_stars, &q.stars, width, height) {
                Ok(m) => {
                    registered_subs.push(RegisteredSub {
                        spooled: sf.clone(),
                        transform: m.transform,
                    });
                }
                Err(err) => {
                    let rec = RejectedFrameRecord {
                        path: sf.original_frame.path.to_string_lossy().to_string(),
                        reason: format!("{err:#}"),
                    };
                    ch_rejected.push(rec.clone());
                    all_rejected.push(rec);
                }
            }
        }

        // For multi-light input with no matching pair, fail group rather than treating failures as successful integration
        if registered_subs.len() < 2 {
            bail!(
                "Target '{}' channel {}: multi-light input ({} frames) had no matching registered pair",
                target_label,
                channel,
                spooled_frames.len()
            );
        }

        // Verify common overlap >= 50% of reference area
        let mut common_overlap = 0usize;
        let grid_steps = 32usize;
        for gy in 0..grid_steps {
            let ry = ((gy as f32) + 0.5) * (height as f32) / (grid_steps as f32);
            for gx in 0..grid_steps {
                let rx = ((gx as f32) + 0.5) * (width as f32) / (grid_steps as f32);
                let all_in = registered_subs.iter().all(|sub| {
                    let (sx, sy) = sub.transform.ref_to_src(rx, ry);
                    sx >= 0.0 && sx < width as f32 && sy >= 0.0 && sy < height as f32
                });
                if all_in {
                    common_overlap += 1;
                }
            }
        }
        let overlap_frac = (common_overlap as f32) / ((grid_steps * grid_steps) as f32);
        if overlap_frac < 0.50 {
            bail!(
                "Target '{}' channel {}: common overlap across registered frames is {:.1}% (< 50% minimum)",
                target_label,
                channel,
                overlap_frac * 100.0
            );
        }

        // Compute normalized exposure / inverse-background-variance weights
        let num_reg = registered_subs.len();
        let mut weights: Vec<f32> = registered_subs
            .iter()
            .map(|sub| {
                let var = (sub.spooled.bg_sigma * sub.spooled.bg_sigma).max(1e-6);
                (sub.spooled.exposure.max(1e-3)) / var
            })
            .collect();
        let w_sum: f32 = weights.iter().sum::<f32>().max(1e-6);
        for w in &mut weights {
            *w /= w_sum;
        }

        let channels_count = registered_subs[0].spooled.channels;
        let mut integrated_img = LinearImage::new(width, height, channels_count);
        let total_exposure_s: f32 = registered_subs.iter().map(|s| s.spooled.exposure).sum();
        integrated_img.exposure = total_exposure_s;

        let cpu_dim = cpu_tile_dim(num_reg)?;
        let tile_dim = if let Some(gs) = &gpu_stacker {
            let gpu_dim = gs.clamped_tile_dim(num_reg);
            if gpu_dim == 0 {
                if accelerator == AcceleratorMode::Gpu {
                    bail!(
                        "GPU cannot fit a single integration pixel for {num_reg} frames within its memory/buffer limits"
                    );
                }
                warnings.push(format!(
                    "GPU memory/buffer limits cannot fit {num_reg} frames; using bounded CPU tiles"
                ));
                gpu_stacker = None;
                cpu_dim
            } else {
                gpu_dim.min(cpu_dim)
            }
        } else {
            cpu_dim
        };

        // Stream 512x512 tiles so RAM scales with tile * frame_count
        let mut tile_values_buf = Vec::new();
        let mut tile_masks_u32_buf = Vec::new();
        let total_tiles = width.div_ceil(tile_dim) * height.div_ceil(tile_dim) * channels_count;
        let stage = format!("Integrating {channel} ({num_reg} frames, {tile_dim}px tiles)");
        let mut tile_progress = PhaseProgress::new(
            options,
            &stage,
            total_tiles,
            (quality_end, progress_range.1),
        );
        let mut completed_tiles = 0;

        for c in 0..channels_count {
            for ty0 in (0..height).step_by(tile_dim) {
                let ty1 = (ty0 + tile_dim).min(height);
                let th = ty1 - ty0;
                for tx0 in (0..width).step_by(tile_dim) {
                    let tx1 = (tx0 + tile_dim).min(width);
                    let tw = tx1 - tx0;
                    let tile_pixels = tw * th;

                    tile_values_buf.resize(num_reg * tile_pixels, 0.0f32);
                    tile_masks_u32_buf.resize(num_reg * tile_pixels, 0u32);

                    // Copy each warped tile directly into its frame slot, avoiding
                    // retaining all warped images alongside the integration buffers.
                    tile_values_buf
                        .par_chunks_mut(tile_pixels)
                        .zip(tile_masks_u32_buf.par_chunks_mut(tile_pixels))
                        .zip(registered_subs.par_iter())
                        .try_for_each(|((values, masks), sub)| -> Result<()> {
                            let (w_data, w_mask) = warp_spooled_tile_lanczos3(
                                &sub.spooled,
                                c,
                                &sub.transform,
                                tx0,
                                ty0,
                                tx1,
                                ty1,
                            )?;
                            values.copy_from_slice(&w_data);
                            for (mask, &valid) in masks.iter_mut().zip(&w_mask) {
                                *mask = u32::from(valid != 0);
                            }
                            Ok(())
                        })?;

                    // Run tile integration on GPU if active, with fallback to CPU on auto mode
                    let (out_tile_vals, out_tile_mask) = if let Some(gs) = gpu_stacker.as_mut() {
                        if accelerator == AcceleratorMode::Gpu || num_reg >= 3 {
                            match gs.integrate_tile(
                                tile_pixels,
                                num_reg,
                                &tile_values_buf,
                                &tile_masks_u32_buf,
                                &weights,
                            ) {
                                Ok(res) => {
                                    used_gpu = true;
                                    res
                                }
                                Err(err) => {
                                    if accelerator == AcceleratorMode::Gpu {
                                        return Err(err.context("GPU tile integration failed"));
                                    }
                                    warnings.push(format!(
                                        "GPU tile dispatch failed ({err:#}); falling back to CPU"
                                    ));
                                    gpu_stacker = None;
                                    integrate_tile_cpu(
                                        tile_pixels,
                                        num_reg,
                                        &tile_values_buf,
                                        &tile_masks_u32_buf,
                                        &weights,
                                    )
                                }
                            }
                        } else {
                            integrate_tile_cpu(
                                tile_pixels,
                                num_reg,
                                &tile_values_buf,
                                &tile_masks_u32_buf,
                                &weights,
                            )
                        }
                    } else {
                        integrate_tile_cpu(
                            tile_pixels,
                            num_reg,
                            &tile_values_buf,
                            &tile_masks_u32_buf,
                            &weights,
                        )
                    };

                    // Copy integrated tile into output image plane
                    let plane_base = c * npix;
                    for ly in 0..th {
                        let dst_row = (ty0 + ly) * width + tx0;
                        let src_row = ly * tw;
                        integrated_img.data[plane_base + dst_row..plane_base + dst_row + tw]
                            .copy_from_slice(&out_tile_vals[src_row..src_row + tw]);
                        if c == 0 {
                            integrated_img.mask[dst_row..dst_row + tw]
                                .copy_from_slice(&out_tile_mask[src_row..src_row + tw]);
                        } else {
                            for lx in 0..tw {
                                integrated_img.mask[dst_row + lx] &= out_tile_mask[src_row + lx];
                            }
                        }
                    }
                    completed_tiles += 1;
                    tile_progress.report(completed_tiles);
                }
            }
        }

        let accepted_paths: Vec<String> = registered_subs
            .iter()
            .map(|s| s.spooled.original_frame.path.to_string_lossy().to_string())
            .collect();

        integrations.insert(
            channel,
            ChannelIntegration {
                channel,
                image: integrated_img,
                aligned_to_target: true,
                total_exposure_s,
                accepted_frames: accepted_paths,
                reference_frame_path: Some(spooled_frames[ref_idx].original_frame.path.clone()),
                reference_wcs: original_wcs(&spooled_frames[ref_idx].original_frame),
                wcs: None,
                rejected_frames: ch_rejected,
                fits_rel_path: String::new(),
                valid_region: crate::crop::frame_region(
                    width,
                    height,
                    registered_subs.iter().map(|sub| sub.transform),
                ),
                recommended_crop: None,
            },
        );
    }
    Ok(TargetIntegrationOutput {
        target_id: target_id.to_string(),
        target_label: target_label.to_string(),
        width,
        height,
        integrations,
        warnings,
        rejected_frames: all_rejected,
        used_gpu,
        cross_filter_aligned: true,
    })
}

/// Aligns completed channel integrations to a common filter and exports the final FITS files.
pub fn finalize_target_channels(
    mut output: TargetIntegrationOutput,
    run_dir: &Path,
    run_rel_dir: &str,
) -> Result<TargetIntegrationOutput> {
    let width = output.width;
    let height = output.height;
    let target_id = &output.target_id;
    let target_label = &output.target_label;
    let integrations = &mut output.integrations;
    let warnings = &mut output.warnings;
    let mut cross_filter_aligned = true;
    let ref_channel = [
        Channel::Ha, Channel::L, Channel::R, Channel::RgbCfa,
        Channel::G, Channel::Oiii, Channel::Sii, Channel::B,
    ]
    .into_iter()
    .find(|ch| integrations.contains_key(ch))
    .or_else(|| integrations.keys().next().copied());
    for (&channel, entry) in integrations.iter_mut() {
        entry.wcs = (Some(channel) == ref_channel).then_some(entry.reference_wcs).flatten();
    }
    let channel_keys: Vec<Channel> = integrations.keys().copied().collect();
    if channel_keys.len() > 1 {
        let ref_channel = ref_channel.expect("nonempty integrations have a reference channel");
        let ref_stars: Vec<DetectedStar> = detect_stars(&integrations[&ref_channel].image);
        for &ch in &channel_keys {
            if ch == ref_channel {
                continue;
            }
            let ch_img = &integrations[&ch].image;
            let ch_stars = detect_stars(ch_img);
            match match_star_fields(&ref_stars, &ch_stars, width, height) {
                Ok(m) => {
                    let warped = warp_linear_image_lanczos3(ch_img, &m.transform);
                    if let Some(entry) = integrations.get_mut(&ch) {
                        entry.image = warped;
                        entry.valid_region = crate::crop::warp_region(
                            &entry.valid_region,
                            &m.transform,
                            width,
                            height,
                        );
                        entry.aligned_to_target = true;
                    }
                }
                Err(err) => {
                    cross_filter_aligned = false;
                    warnings.push(format!(
                        "Cross-filter alignment of {} to reference {} failed ({err:#}); mixed multi-filter palettes requiring {} will be omitted",
                        ch, ref_channel, ch
                    ));
                    if let Some(entry) = integrations.get_mut(&ch) {
                        entry.aligned_to_target = false;
                    }
                }
            }
        }
    }
    let common_region = crate::crop::intersect_regions(
        width,
        height,
        integrations
            .values()
            .filter(|ci| ci.aligned_to_target)
            .map(|ci| ci.valid_region.as_slice()),
    );
    let common_crop = crate::crop::largest_crop(width, height, &common_region);
    for entry in integrations.values_mut() {
        entry.recommended_crop = if entry.aligned_to_target {
            common_crop
        } else {
            crate::crop::largest_crop(width, height, &entry.valid_region)
        };
    }
    for (&channel, entry) in integrations {
        let fits_name = format!("integrated-{}-{}.fits", target_id, channel.as_str());
        let fits_path = run_dir.join(&fits_name);
        write_fits_linear(
            &fits_path,
            &entry.image,
            channel.as_str(),
            target_label,
            entry.total_exposure_s,
            entry.accepted_frames.len(),
            entry.rejected_frames.len(),
            entry.wcs.as_ref(),
        )
        .with_context(|| format!("Failed to export integrated FITS '{}'", fits_path.display()))?;
        entry.fits_rel_path = format!("{run_rel_dir}/{fits_name}");
    }
    output.cross_filter_aligned = cross_filter_aligned;
    Ok(output)
}

pub fn build_science_artifacts(
    integrations: &BTreeMap<Channel, ChannelIntegration>,
) -> Vec<ScienceArtifact> {
    integrations
        .values()
        .map(|ci| ScienceArtifact {
            channel: ci.channel.as_str().to_string(),
            fits_path: ci.fits_rel_path.clone(),
            width: ci.image.width,
            height: ci.image.height,
            total_exposure_s: ci.total_exposure_s,
            accepted_frames: ci.accepted_frames.clone(),
            rejected_frames: ci.rejected_frames.clone(),
            recommended_crop: ci.recommended_crop,
            wcs: ci.wcs,
        })
        .collect()
}
