use crate::config::{
    BayerPattern, CalibrationSummary, Channel, FileFormat, FrameMeta, FrameRole, LinearImage,
    PhaseProgress, ProcessOptions,
};
use crate::scan::{TargetGroup, max_parallel_frames};
use crate::{fits, raw, register};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const SPOOL_MAGIC: &[u8; 8] = b"PLSP0001";
const SPOOL_HEADER_BYTES: u64 = 32;

#[derive(Debug, Clone)]
pub struct SpooledFrame {
    pub original_frame: FrameMeta,
    pub spool_path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub exposure: f32,
    pub bg_sigma: f32,
}

#[derive(Debug, Clone)]
pub struct TargetCalibrationOutput {
    pub target_id: String,
    pub target_label: String,
    pub width: usize,
    pub height: usize,
    pub spooled_by_channel: BTreeMap<Channel, Vec<SpooledFrame>>,
    pub summary: CalibrationSummary,
}

pub fn write_spooled_image(path: &Path, img: &LinearImage, bg_sigma: f32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)
        .with_context(|| format!("Failed to create spool file '{}'", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(SPOOL_MAGIC)?;
    writer.write_all(&(img.width as u32).to_le_bytes())?;
    writer.write_all(&(img.height as u32).to_le_bytes())?;
    writer.write_all(&(img.channels as u32).to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?;
    writer.write_all(&img.exposure.to_le_bytes())?;
    writer.write_all(&bg_sigma.to_le_bytes())?;

    let float_bytes: &[u8] = bytemuck::cast_slice(&img.data);
    writer.write_all(float_bytes)?;
    writer.write_all(&img.mask)?;
    writer.flush()?;
    Ok(())
}

pub fn read_spooled_image(path: &Path) -> Result<LinearImage> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open spool file '{}'", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hdr = [0u8; 32];
    reader.read_exact(&mut hdr)?;
    if &hdr[0..8] != SPOOL_MAGIC {
        bail!("Invalid spool file header in '{}'", path.display());
    }
    let width = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let channels = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    let exposure = f32::from_le_bytes(hdr[24..28].try_into().unwrap());

    let npix = width * height;
    let mut data = vec![0.0f32; npix * channels];
    let float_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    reader.read_exact(float_bytes)?;

    let mut mask = vec![0u8; npix];
    reader.read_exact(&mut mask)?;

    Ok(LinearImage {
        width,
        height,
        channels,
        data,
        mask,
        bayer: None,
        exposure,
    })
}

/// Reads a rectangular region `[x0..x1, y0..y1]` of a single channel `c` and its mask from a spooled file
/// without loading the full image into RAM.
pub fn read_spooled_region(
    path: &Path,
    width: usize,
    height: usize,
    c: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> Result<(Vec<f32>, Vec<u8>)> {
    let rw = x1 - x0;
    let rh = y1 - y0;
    let mut out_data = vec![0.0f32; rw * rh];
    let mut out_mask = vec![0u8; rw * rh];

    let mut file = File::open(path)
        .with_context(|| format!("Failed to open spool file '{}'", path.display()))?;
    let plane_offset = SPOOL_HEADER_BYTES + ((c * height * width) * 4) as u64;
    let mask_base = SPOOL_HEADER_BYTES + ((height * width * (c + 1).max(1)) * 4) as u64;
    // Wait: mask is after ALL channels (`channels * height * width * 4`). Read channels from header!
    let mut hdr = [0u8; 32];
    file.read_exact(&mut hdr)?;
    let channels = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    let real_mask_base = SPOOL_HEADER_BYTES + (channels * height * width * 4) as u64;
    let _ = mask_base;

    for row in 0..rh {
        let sy = y0 + row;
        let f_off = plane_offset + ((sy * width + x0) * 4) as u64;
        file.seek(SeekFrom::Start(f_off))?;
        let dst_slice = &mut out_data[row * rw..(row + 1) * rw];
        file.read_exact(bytemuck::cast_slice_mut(dst_slice))?;
    }

    for row in 0..rh {
        let sy = y0 + row;
        let m_off = real_mask_base + (sy * width + x0) as u64;
        file.seek(SeekFrom::Start(m_off))?;
        let dst_mask = &mut out_mask[row * rw..(row + 1) * rw];
        file.read_exact(dst_mask)?;
    }

    Ok((out_data, out_mask))
}

fn load_raw_or_fits(frame: &FrameMeta) -> Result<LinearImage> {
    match frame.format {
        FileFormat::Fits => fits::read_fits_linear(frame),
        FileFormat::Raw => raw::read_raw_mosaic(frame),
    }
}

fn same_geometry_and_sensor(a: &FrameMeta, b: &FrameMeta) -> bool {
    a.format == b.format
        && a.sensor == b.sensor
        && a.width == b.width
        && a.height == b.height
        && a.channels == b.channels
        && a.bayer == b.bayer
        && a.binning == b.binning
}

fn compatible_gain(a: &FrameMeta, b: &FrameMeta) -> bool {
    match (a.gain_or_iso, b.gain_or_iso) {
        (Some(ga), Some(gb)) => (ga - gb).abs() <= 0.02 * ga.abs().max(1.0),
        (None, None) => true,
        _ => false,
    }
}

fn compatible_temp(a: &FrameMeta, b: &FrameMeta) -> bool {
    match (a.temperature, b.temperature) {
        (Some(ta), Some(tb)) => (ta - tb).abs() <= 2.0,
        (None, None) => true,
        _ => false,
    }
}

fn compatible_exposure_5pct(ea: Option<f32>, eb: Option<f32>) -> bool {
    match (ea, eb) {
        (Some(a), Some(b)) if a > 0.0 && b > 0.0 => ((a - b).abs() / a.max(b)) <= 0.05,
        (None, None) => true,
        _ => false,
    }
}

/// Metadata-only calibration availability using the same master matching rules as calibration.
pub(crate) fn matching_counts(light: &FrameMeta, frames: &[FrameMeta]) -> (usize, usize, usize, usize) {
    let bias = frames.iter().filter(|f| f.role == FrameRole::Bias
        && same_geometry_and_sensor(f, light) && compatible_gain(f, light)).count();
    let darks = frames.iter().filter(|f| f.role == FrameRole::Dark
        && same_geometry_and_sensor(f, light) && compatible_gain(f, light)
        && compatible_temp(f, light)
        && (compatible_exposure_5pct(f.exposure, light.exposure)
            || (f.exposure.unwrap_or(0.0) > 0.0 && light.exposure.unwrap_or(0.0) > 0.0
                && (light.format == FileFormat::Raw || bias > 0)))).count();
    let flats: Vec<_> = frames.iter().filter(|f| f.role == FrameRole::Flat
        && same_geometry_and_sensor(f, light)
        && (f.channel == light.channel || (light.channel == Some(Channel::RgbCfa) && f.bayer.is_some()))).collect();
    let flat_darks = flats.first().map_or(0, |flat| frames.iter().filter(|f|
        f.role == FrameRole::FlatDark && same_geometry_and_sensor(f, light)
        && compatible_gain(f, flat) && compatible_temp(f, flat)
        && compatible_exposure_5pct(f.exposure, flat.exposure)).count());
    (bias, darks, flats.len(), flat_darks)
}

/// Combines calibration frames into a master using median combination with 3-MAD outlier rejection for >= 3 frames.
pub fn combine_calibration_frames(frames: &[FrameMeta]) -> Result<LinearImage> {
    if frames.is_empty() {
        bail!("Cannot combine empty calibration frame list");
    }
    if frames.len() == 1 {
        return load_raw_or_fits(&frames[0]);
    }
    let mut loaded = Vec::with_capacity(frames.len());
    for f in frames {
        loaded.push(load_raw_or_fits(f)?);
    }
    let first = &loaded[0];
    let width = first.width;
    let height = first.height;
    let channels = first.channels;
    let total_len = width * height * channels;
    let npix = width * height;

    let mut out_data = vec![0.0f32; total_len];
    let mut out_mask = vec![1u8; npix];

    if loaded.len() == 2 {
        for i in 0..total_len {
            out_data[i] = 0.5 * (loaded[0].data[i] + loaded[1].data[i]);
        }
        for p in 0..npix {
            out_mask[p] = loaded[0].mask[p] & loaded[1].mask[p];
        }
    } else {
        let n = loaded.len();
        let mut vals = Vec::with_capacity(n);
        let mut devs = Vec::with_capacity(n);
        for i in 0..total_len {
            let p = i % npix;
            vals.clear();
            for img in &loaded {
                if img.mask[p] != 0 && img.data[i].is_finite() {
                    vals.push(img.data[i]);
                }
            }
            if vals.is_empty() {
                out_data[i] = 0.0;
                out_mask[p] = 0;
                continue;
            }
            vals.sort_by(|a, b| a.total_cmp(b));
            let med = vals[vals.len() / 2];
            if vals.len() >= 3 {
                devs.clear();
                for &v in &vals {
                    devs.push((v - med).abs());
                }
                devs.sort_by(|a, b| a.total_cmp(b));
                let mad = devs[devs.len() / 2] * 1.4826;
                let thresh = 3.0 * mad.max(1e-4);
                let mut sum = 0.0f32;
                let mut cnt = 0usize;
                for &v in &vals {
                    if (v - med).abs() <= thresh {
                        sum += v;
                        cnt += 1;
                    }
                }
                out_data[i] = if cnt > 0 { sum / cnt as f32 } else { med };
            } else {
                out_data[i] = med;
            }
        }
    }

    let avg_exp = frames.iter().filter_map(|f| f.exposure).sum::<f32>() / (frames.len() as f32);

    Ok(LinearImage {
        width,
        height,
        channels,
        data: out_data,
        mask: out_mask,
        bayer: first.bayer,
        exposure: if avg_exp > 0.0 {
            avg_exp
        } else {
            first.exposure
        },
    })
}

/// Normalizes a master flat (after flat-dark or bias subtraction) per CFA phase or per plane.
pub fn normalize_flat_master(flat: &mut LinearImage) -> Result<()> {
    let w = flat.width;
    let h = flat.height;
    let npix = w * h;

    if let Some(_bayer) = flat.bayer {
        // Normalize each of the 4 (y&1, x&1) Bayer phases independently by its median
        for py in 0..2 {
            for px in 0..2 {
                let mut phase_vals = Vec::with_capacity(npix / 4 + 1);
                for y in (py..h).step_by(2) {
                    for x in (px..w).step_by(2) {
                        let idx = y * w + x;
                        let v = flat.data[idx];
                        if flat.mask[idx] != 0 && v.is_finite() && v > 1e-4 {
                            phase_vals.push(v);
                        }
                    }
                }
                if phase_vals.is_empty() {
                    bail!("Flat master has no positive finite pixels in Bayer phase ({px}, {py})");
                }
                phase_vals.sort_by(|a, b| a.total_cmp(b));
                let med = phase_vals[phase_vals.len() / 2];
                for y in (py..h).step_by(2) {
                    for x in (px..w).step_by(2) {
                        let idx = y * w + x;
                        let v = flat.data[idx];
                        if flat.mask[idx] != 0 && v.is_finite() && v > 1e-4 * med {
                            flat.data[idx] = v / med;
                        } else {
                            flat.data[idx] = 1.0;
                            flat.mask[idx] = 0;
                        }
                    }
                }
            }
        }
    } else {
        for c in 0..flat.channels {
            let offset = c * npix;
            let mut vals: Vec<f32> = (0..npix)
                .filter(|&p| {
                    flat.mask[p] != 0
                        && flat.data[offset + p].is_finite()
                        && flat.data[offset + p] > 1e-4
                })
                .map(|p| flat.data[offset + p])
                .collect();
            if vals.is_empty() {
                bail!("Flat master has no positive finite pixels in channel {}", c);
            }
            vals.sort_by(|a, b| a.total_cmp(b));
            let med = vals[vals.len() / 2];
            for p in 0..npix {
                let v = flat.data[offset + p];
                if flat.mask[p] != 0 && v.is_finite() && v > 1e-4 * med {
                    flat.data[offset + p] = v / med;
                } else {
                    flat.data[offset + p] = 1.0;
                    flat.mask[p] = 0;
                }
            }
        }
    }
    Ok(())
}

/// Interpolates isolated residual hot/cold pixels using same-CFA neighbors (step 2 for Bayer, step 1 for mono).
pub fn repair_hot_pixels(img: &mut LinearImage) {
    let w = img.width;
    let h = img.height;
    if w < 5 || h < 5 {
        return;
    }
    let step: isize = if img.bayer.is_some() { 2 } else { 1 };
    let npix = w * h;

    let mask = &img.mask;
    for c in 0..img.channels {
        let plane_offset = c * npix;
        let orig = img.data[plane_offset..plane_offset + npix].to_vec();
        let (_, global_mad) = register::robust_median_mad(&orig, mask);
        let global_sigma = (global_mad * 1.4826).max(1e-4);

        img.data[plane_offset..plane_offset + npix]
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                let mut neighbors = [0.0f32; 8];
                for x in 0..w {
                    let p = y * w + x;
                    if mask[p] == 0 {
                        continue;
                    }
                    let val = orig[p];
                    let mut n_cnt = 0usize;
                    for dy in [-step, 0, step] {
                        for dx in [-step, 0, step] {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = x as isize + dx;
                            let ny = y as isize + dy;
                            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                                let np = (ny as usize) * w + (nx as usize);
                                if mask[np] != 0 {
                                    neighbors[n_cnt] = orig[np];
                                    n_cnt += 1;
                                }
                            }
                        }
                    }
                    if n_cnt < 4 {
                        continue;
                    }
                    let slice = &mut neighbors[..n_cnt];
                    slice.sort_by(|a, b| a.total_cmp(b));
                    let med = slice[n_cnt / 2];

                    let mut mad_arr = [0.0f32; 8];
                    for i in 0..n_cnt {
                        mad_arr[i] = (slice[i] - med).abs();
                    }
                    mad_arr[..n_cnt].sort_by(|a, b| a.total_cmp(b));
                    let local_sigma = (mad_arr[n_cnt / 2] * 1.4826).max(global_sigma);

                    // Check 4-connected immediate neighbors to avoid clipping real peaked star cores
                    let mut orth_max_excess = 0.0f32;
                    for (ox, oy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                        let nx = x as isize + ox;
                        let ny = y as isize + oy;
                        if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                            let np = (ny as usize) * w + (nx as usize);
                            let ex = (orig[np] - med).max(0.0);
                            if ex > orth_max_excess {
                                orth_max_excess = ex;
                            }
                        }
                    }

                    let diff = val - med;
                    // Isolated hot pixel or extreme negative dead pixel
                    let is_hot = diff > 6.0 * local_sigma && diff > 3.5 * orth_max_excess;
                    let is_cold = diff < -6.0 * local_sigma;
                    if is_hot || is_cold {
                        row[x] = med;
                    }
                }
            });
    }
}

/// Malvar-He-Cutler (2004) 5x5 linear gradient-corrected Bayer demosaicing.
/// Preserves linear float RGB; only called *after* mosaic calibration.
pub fn demosaic_malvar_he_cutler(mosaic: &LinearImage, bayer: BayerPattern) -> Result<LinearImage> {
    if mosaic.channels != 1 {
        bail!("demosaic_malvar_he_cutler expects a 1-channel mosaic image");
    }
    let w = mosaic.width;
    let h = mosaic.height;
    let src = &mosaic.data;

    let sample = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, (w as isize) - 1) as usize;
        let cy = y.clamp(0, (h as isize) - 1) as usize;
        src[cy * w + cx]
    };

    let mut out = LinearImage::new(w, h, 3);
    out.mask = mosaic.mask.clone();
    out.exposure = mosaic.exposure;
    out.bayer = None;

    let npix = w * h;
    let (r_plane, rest) = out.data.split_at_mut(npix);
    let (g_plane, b_plane) = rest.split_at_mut(npix);

    for y in 0..h {
        let yi = y as isize;
        for x in 0..w {
            let xi = x as isize;
            let idx = y * w + x;
            if out.mask[idx] == 0 {
                continue;
            }
            let c0 = sample(xi, yi);
            let color = bayer.color_at(x, y);

            // Cross & neighborhood values
            let n = sample(xi, yi - 1);
            let s = sample(xi, yi + 1);
            let w1 = sample(xi - 1, yi);
            let e1 = sample(xi + 1, yi);

            let n2 = sample(xi, yi - 2);
            let s2 = sample(xi, yi + 2);
            let w2 = sample(xi - 2, yi);
            let e2 = sample(xi + 2, yi);

            let nw = sample(xi - 1, yi - 1);
            let ne = sample(xi + 1, yi - 1);
            let sw = sample(xi - 1, yi + 1);
            let se = sample(xi + 1, yi + 1);

            let (mut r, mut g, mut b) = match color {
                0 => {
                    // Red site
                    let g_val = 0.125 * (4.0 * c0 + 2.0 * (n + s + w1 + e1) - (n2 + s2 + w2 + e2));
                    let b_val =
                        0.125 * (6.0 * c0 + 2.0 * (nw + ne + sw + se) - 1.5 * (n2 + s2 + w2 + e2));
                    (c0, g_val, b_val)
                }
                2 => {
                    // Blue site
                    let g_val = 0.125 * (4.0 * c0 + 2.0 * (n + s + w1 + e1) - (n2 + s2 + w2 + e2));
                    let r_val =
                        0.125 * (6.0 * c0 + 2.0 * (nw + ne + sw + se) - 1.5 * (n2 + s2 + w2 + e2));
                    (r_val, g_val, c0)
                }
                _ => {
                    // Green site
                    let horiz_val = 0.125
                        * (5.0 * c0 + 4.0 * (w1 + e1) - (nw + ne + sw + se + w2 + e2)
                            + 0.5 * (n2 + s2));
                    let vert_val = 0.125
                        * (5.0 * c0 + 4.0 * (n + s) - (nw + ne + sw + se + n2 + s2)
                            + 0.5 * (w2 + e2));
                    if bayer.is_green_in_red_row(x, y) {
                        (horiz_val, c0, vert_val)
                    } else {
                        (vert_val, c0, horiz_val)
                    }
                }
            };

            // Clamp to local neighborhood bounds to prevent negative ringing around high-contrast stars
            let g_min = n.min(s).min(w1).min(e1).min(c0);
            let g_max = n.max(s).max(w1).max(e1).max(c0);
            let d_min = nw.min(ne).min(sw).min(se).min(g_min);
            let d_max = nw.max(ne).max(sw).max(se).max(g_max);

            r = r.clamp(d_min, d_max);
            g = g.clamp(g_min, g_max);
            b = b.clamp(d_min, d_max);

            r_plane[idx] = r;
            g_plane[idx] = g;
            b_plane[idx] = b;
        }
    }

    Ok(out)
}

/// Calibrates one channel's light frames, spooling only that channel to `staging_dir`.
pub fn calibrate_target_group(
    group: &TargetGroup,
    channel: Channel,
    calibration_frames: &[FrameMeta],
    staging_dir: &Path,
    options: &ProcessOptions,
    progress_range: (f32, f32),
) -> Result<TargetCalibrationOutput> {
    std::fs::create_dir_all(staging_dir)?;

    let mut summary = CalibrationSummary::default();
    let sample_light = group
        .lights_by_channel
        .get(&channel)
        .and_then(|v| v.first())
        .context("TargetGroup has no light frames for channel")?;

    // 1. Find compatible Bias frames
    let bias_candidates: Vec<FrameMeta> = calibration_frames
        .iter()
        .filter(|f| f.role == FrameRole::Bias && same_geometry_and_sensor(f, sample_light))
        .cloned()
        .collect();
    let compatible_biases: Vec<FrameMeta> = bias_candidates
        .iter()
        .filter(|f| compatible_gain(f, sample_light))
        .cloned()
        .collect();

    let bias_master = if sample_light.format == FileFormat::Raw {
        // For RAW, camera black is already subtracted from lights/calibration mosaics first;
        // no additional bias when black is already accounted for.
        None
    } else if !compatible_biases.is_empty() {
        summary.used_bias = Some(format!("{} frame(s)", compatible_biases.len()));
        Some(combine_calibration_frames(&compatible_biases)?)
    } else {
        if !bias_candidates.is_empty() {
            summary.warnings.push(format!(
                "Omitted {} bias frame(s) due to mismatched gain/ISO for target '{}'",
                bias_candidates.len(),
                group.target_label
            ));
        }
        summary.missing_masters.push("bias".to_string());
        None
    };
    options.report_detail(&format!("{} · {}: bias {} (gain/ISO ±2%; RAW uses camera black level)",
        group.target_label, channel, compatible_biases.len()));

    // 2. Precompute FlatDark and Flat masters per channel
    let all_flat_darks: Vec<FrameMeta> = calibration_frames
        .iter()
        .filter(|f| f.role == FrameRole::FlatDark && same_geometry_and_sensor(f, sample_light))
        .cloned()
        .collect();
    let all_flats: Vec<FrameMeta> = calibration_frames
        .iter()
        .filter(|f| f.role == FrameRole::Flat && same_geometry_and_sensor(f, sample_light))
        .cloned()
        .collect();
    let all_darks: Vec<FrameMeta> = calibration_frames
        .iter()
        .filter(|f| f.role == FrameRole::Dark && same_geometry_and_sensor(f, sample_light))
        .cloned()
        .collect();

    let mut spooled_by_channel: BTreeMap<Channel, Vec<SpooledFrame>> = BTreeMap::new();

    for (&current_channel, lights) in &group.lights_by_channel {
        if current_channel != channel || lights.is_empty() {
            continue;
        }
        let first_light = &lights[0];

        // Build normalized flat master for this channel if available
        let channel_flats: Vec<FrameMeta> = all_flats
            .iter()
            .filter(|f| {
                f.channel == Some(channel) || (channel == Channel::RgbCfa && f.bayer.is_some())
            })
            .cloned()
            .collect();

        let flat_master = if !channel_flats.is_empty() {
            let mut raw_flat = combine_calibration_frames(&channel_flats)?;
            let ref_flat = &channel_flats[0];
            // Check for matching flat-dark per flat exposure
            let matching_flat_darks: Vec<FrameMeta> = all_flat_darks
                .iter()
                .filter(|fd| {
                    compatible_gain(fd, ref_flat)
                        && compatible_temp(fd, ref_flat)
                        && compatible_exposure_5pct(fd.exposure, ref_flat.exposure)
                })
                .cloned()
                .collect();

            if !matching_flat_darks.is_empty() {
                let fd_master = combine_calibration_frames(&matching_flat_darks)?;
                for i in 0..raw_flat.data.len() {
                    raw_flat.data[i] -= fd_master.data[i];
                }
                summary.used_flat_darks.push(format!(
                    "{}: {} frame(s)",
                    channel,
                    matching_flat_darks.len()
                ));
            } else if first_light.format == FileFormat::Fits {
                if let Some(bias) = &bias_master {
                    for i in 0..raw_flat.data.len() {
                        raw_flat.data[i] -= bias.data[i];
                    }
                }
            }
            normalize_flat_master(&mut raw_flat)?;
            summary
                .used_flats
                .push(format!("{}: {} frame(s)", channel, channel_flats.len()));
            Some(raw_flat)
        } else {
            summary.missing_masters.push(format!("flat ({channel})"));
            None
        };

        // Match dark master for this channel's lights
        let exact_darks: Vec<FrameMeta> = all_darks
            .iter()
            .filter(|d| {
                compatible_gain(d, first_light)
                    && compatible_temp(d, first_light)
                    && compatible_exposure_5pct(d.exposure, first_light.exposure)
            })
            .cloned()
            .collect();

        let scalable_darks: Vec<FrameMeta> = if exact_darks.is_empty() {
            all_darks
                .iter()
                .filter(|d| {
                    compatible_gain(d, first_light)
                        && compatible_temp(d, first_light)
                        && d.exposure.unwrap_or(0.0) > 0.0
                        && first_light.exposure.unwrap_or(0.0) > 0.0
                        && (first_light.format == FileFormat::Raw || bias_master.is_some())
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        enum DarkMode {
            Exact(LinearImage),
            Scaled { master: LinearImage, dark_exp: f32 },
            None,
        }

        let dark_mode = if !exact_darks.is_empty() {
            summary
                .used_darks
                .push(format!("{}: {} exact frame(s)", channel, exact_darks.len()));
            DarkMode::Exact(combine_calibration_frames(&exact_darks)?)
        } else if !scalable_darks.is_empty() {
            let master = combine_calibration_frames(&scalable_darks)?;
            let dark_exp = master.exposure;
            summary.used_darks.push(format!(
                "{}: {} scaled frame(s) (dark {:.1}s)",
                channel,
                scalable_darks.len(),
                dark_exp
            ));
            DarkMode::Scaled { master, dark_exp }
        } else {
            if !all_darks.is_empty() {
                let reason = format!(
                    "Omitted {} incompatible dark frame(s) for '{}' ({}): temperature (>2C diff), gain/ISO, or unscalable exposure mismatch",
                    all_darks.len(),
                    group.target_label,
                    channel
                );
                if !summary.warnings.contains(&reason) {
                    summary.warnings.push(reason);
                }
            } else {
                let warn = format!(
                    "No compatible dark master for '{}' ({}); proceeding without dark subtraction",
                    group.target_label, channel
                );
                if !summary.warnings.contains(&warn) {
                    summary.warnings.push(warn);
                }
            }
            summary.missing_masters.push(format!("dark ({channel})"));
            DarkMode::None
        };
        options.report_detail(&format!(
            "{} · {} calibration: flats {} · flat-darks {} · exact darks {} · scalable darks {} · missing {}; geometry/sensor/binning matched, dark gain ±2%, temperature ±2°C, exposure ±5%",
            group.target_label, channel, channel_flats.len(), summary.used_flat_darks.len(),
            exact_darks.len(), scalable_darks.len(),
            if summary.missing_masters.is_empty() { "none".to_string() } else { summary.missing_masters.join(", ") }
        ));

        // Restrict in-flight full-resolution images (including repair buffers)
        // independently of the global Rayon worker count. Masters are shared
        // read-only; each indexed frame writes a distinct spool file.
        let batch_size =
            max_parallel_frames(first_light.width, first_light.height, first_light.channels);
        let mut spooled_vec = Vec::with_capacity(lights.len());
        let stage = format!("Calibrating {} · {}", group.target_label, channel);
        let mut progress = PhaseProgress::new(options, &stage, lights.len(), progress_range);
        for batch in lights.chunks(batch_size).enumerate() {
            let (batch_idx, lights_batch) = batch;
            let batch_results: Vec<Result<SpooledFrame>> = lights_batch
                .par_iter()
                .enumerate()
                .map(|(offset, light_meta)| {
                    let l_idx = batch_idx * batch_size + offset;
                    let mut light = load_raw_or_fits(light_meta)?;
                    let light_exp = light_meta.exposure.unwrap_or(1.0).max(1e-6);

                    match (&dark_mode, light_meta.format) {
                        (DarkMode::Exact(dark), _) => {
                            // Both FITS (`light - raw_dark`) and RAW (`black_corrected_light - black_corrected_dark`)
                            for i in 0..light.data.len() {
                                light.data[i] -= dark.data[i];
                            }
                        }
                        (
                            DarkMode::Scaled {
                                master: dark,
                                dark_exp,
                            },
                            FileFormat::Fits,
                        ) => {
                            // FITS scaled dark when bias master exists:
                            // light - bias - (raw_dark - bias) * (light_exp / dark_exp)
                            let scale = light_exp / (*dark_exp).max(1e-6);
                            if let Some(bias) = &bias_master {
                                for i in 0..light.data.len() {
                                    let b = bias.data[i];
                                    light.data[i] = light.data[i] - b - (dark.data[i] - b) * scale;
                                }
                            }
                        }
                        (
                            DarkMode::Scaled {
                                master: dark,
                                dark_exp,
                            },
                            FileFormat::Raw,
                        ) => {
                            // RAW scaled dark: camera black already subtracted once from both light and dark
                            let scale = light_exp / (*dark_exp).max(1e-6);
                            for i in 0..light.data.len() {
                                light.data[i] -= dark.data[i] * scale;
                            }
                        }
                        (DarkMode::None, FileFormat::Fits) => {
                            if let Some(bias) = &bias_master {
                                for i in 0..light.data.len() {
                                    light.data[i] -= bias.data[i];
                                }
                            }
                        }
                        (DarkMode::None, FileFormat::Raw) => {
                            // Camera black already subtracted in read_raw_mosaic
                        }
                    }

                    // Divide by normalized flat if present
                    if let Some(flat) = &flat_master {
                        let npix = light.width * light.height;
                        for c in 0..light.channels {
                            let f_c = c.min(flat.channels - 1);
                            for p in 0..npix {
                                if flat.mask[p] == 0 {
                                    light.mask[p] = 0;
                                    light.data[c * npix + p] = 0.0;
                                } else {
                                    let fv = flat.data[f_c * npix + p];
                                    if fv.is_finite() && fv > 1e-4 {
                                        light.data[c * npix + p] /= fv;
                                    } else {
                                        light.mask[p] = 0;
                                        light.data[c * npix + p] = 0.0;
                                    }
                                }
                            }
                        }
                    }

                    // Interpolate residual hot pixels using same-CFA neighbors
                    repair_hot_pixels(&mut light);

                    // Demosaic Bayer CFA *after* mosaic calibration
                    let calibrated = if let Some(bayer) = light.bayer {
                        demosaic_malvar_he_cutler(&light, bayer)?
                    } else {
                        light
                    };

                    let (_, mad) =
                        register::robust_median_mad(calibrated.plane(0), &calibrated.mask);
                    let bg_sigma = (mad * 1.4826).max(1e-4);

                    let spool_path =
                        staging_dir.join(format!("cal_{}_{:04}.spool", channel.as_str(), l_idx));
                    write_spooled_image(&spool_path, &calibrated, bg_sigma)?;

                    Ok(SpooledFrame {
                        original_frame: light_meta.clone(),
                        spool_path,
                        width: calibrated.width,
                        height: calibrated.height,
                        channels: calibrated.channels,
                        exposure: calibrated.exposure,
                        bg_sigma,
                    })
                })
                .collect();
            for result in batch_results {
                spooled_vec.push(result?);
            }
            progress.report(spooled_vec.len());
        }

        spooled_by_channel.insert(channel, spooled_vec);
    }

    Ok(TargetCalibrationOutput {
        target_id: group.target_id.clone(),
        target_label: group.target_label.clone(),
        width: group.width,
        height: group.height,
        spooled_by_channel,
        summary,
    })
}
