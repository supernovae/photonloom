use crate::config::{
    Channel, FileFormat, FrameMeta, FrameRole, LinearImage, PhaseProgress, ProcessOptions,
    RejectedFrameRecord,
};
use crate::{fits, raw, register};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// Bound concurrent full-resolution frames independently of Rayon's worker count.
// Account for image, mask, temporary pixel buffers, and FITS decoder overhead.
pub(crate) fn max_parallel_frames(width: usize, height: usize, channels: usize) -> usize {
    const FRAME_BUDGET: usize = 2 * 1024 * 1024 * 1024;
    let bytes_per_pixel = if channels >= 3 { 40 } else { 16 };
    let frame_bytes = width
        .saturating_mul(height)
        .saturating_mul(bytes_per_pixel)
        .max(1);
    (FRAME_BUDGET / frame_bytes).clamp(1, 4)
}

const FITS_EXTENSIONS: &[&str] = &["fit", "fits", "fts"];
const RAW_EXTENSIONS: &[&str] = &["cr2", "cr3", "nef", "arw", "dng", "raf", "rw2"];

#[derive(Debug, Clone)]
pub struct ScanReport {
    pub frames: Vec<FrameMeta>,
    pub light_frames: Vec<FrameMeta>,
    pub calibration_frames: Vec<FrameMeta>,
    pub warnings: Vec<String>,
    pub rejected_frames: Vec<RejectedFrameRecord>,
}

#[derive(Debug, Clone)]
pub struct TargetGroup {
    pub target_id: String,
    pub target_label: String,
    pub sensor: String,
    pub width: usize,
    pub height: usize,
    pub binning: (u32, u32),
    pub lights_by_channel: BTreeMap<Channel, Vec<FrameMeta>>,
}

fn classify_extension(path: &Path) -> Option<FileFormat> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    if FITS_EXTENSIONS.contains(&ext.as_str()) {
        Some(FileFormat::Fits)
    } else if RAW_EXTENSIONS.contains(&ext.as_str()) {
        Some(FileFormat::Raw)
    } else {
        None
    }
}

/// True when `path` has a FITS or camera RAW extension the scanner ingests.
pub fn is_supported_frame_path(path: &Path) -> bool {
    classify_extension(path).is_some()
}

fn extract_folder_role(root: &Path, path: &Path) -> Result<Option<FrameRole>> {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut found: Option<(FrameRole, String)> = None;
    if let Some(parent) = rel.parent() {
        for comp in parent.components() {
            let s = comp.as_os_str().to_string_lossy();
            if let Some(role) = FrameRole::from_token(&s) {
                if let Some((prev_role, prev_dir)) = &found {
                    if *prev_role != role {
                        bail!(
                            "Conflicting folder roles '{}' ({}) and '{}' ({}) in path '{}'",
                            prev_dir,
                            prev_role,
                            s,
                            role,
                            path.display()
                        );
                    }
                } else {
                    found = Some((role, s.to_string()));
                }
            }
        }
    }
    Ok(found.map(|(r, _)| r))
}

fn extract_filename_role(path: &Path) -> Result<Option<FrameRole>> {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return Ok(None);
    };
    let lower = stem.to_ascii_lowercase();
    // Check flatdark / darkflat before flat or dark
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    let mut roles = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len()
            && ((tokens[i] == "flat" && tokens[i + 1].starts_with("dark"))
                || (tokens[i] == "dark" && tokens[i + 1].starts_with("flat")))
        {
            roles.push(FrameRole::FlatDark);
            i += 2;
            continue;
        }
        if let Some(r) = FrameRole::from_token(tokens[i]) {
            roles.push(r);
        }
        i += 1;
    }
    roles.dedup();
    if roles.len() > 1 {
        bail!(
            "Conflicting role tokens {:?} in filename '{}'",
            roles,
            path.display()
        );
    }
    Ok(roles.into_iter().next())
}

fn resolve_role(root: &Path, path: &Path, header_imagetyp: Option<&str>) -> Result<FrameRole> {
    let header_role = match header_imagetyp {
        Some(raw) => {
            let parsed = FrameRole::from_token(raw);
            if parsed.is_none() {
                bail!(
                    "Unrecognized FITS IMAGETYP/FRAME '{}' in '{}'",
                    raw,
                    path.display()
                );
            }
            parsed
        }
        None => None,
    };
    let folder_role = extract_folder_role(root, path)?;
    let file_role = extract_filename_role(path)?;

    if let (Some(hr), Some(fr)) = (header_role, folder_role) {
        if hr != fr {
            bail!(
                "Conflicting frame roles for '{}': FITS header says '{}' but parent folder says '{}'",
                path.display(),
                hr,
                fr
            );
        }
    }
    if let (Some(hr), Some(flr)) = (header_role, file_role) {
        if hr != flr {
            bail!(
                "Conflicting frame roles for '{}': FITS header says '{}' but filename token says '{}'",
                path.display(),
                hr,
                flr
            );
        }
    }
    if let (Some(fr), Some(flr)) = (folder_role, file_role) {
        if fr != flr {
            bail!(
                "Conflicting frame roles for '{}': parent folder says '{}' but filename token says '{}'",
                path.display(),
                fr,
                flr
            );
        }
    }

    Ok(header_role
        .or(folder_role)
        .or(file_role)
        .unwrap_or(FrameRole::Light))
}

fn extract_path_channel(root: &Path, path: &Path) -> Option<Channel> {
    let rel = path.strip_prefix(root).unwrap_or(path);
    if let Some(parent) = rel.parent() {
        for comp in parent.components().rev() {
            let s = comp.as_os_str().to_string_lossy();
            if let Some(ch) = Channel::from_token(&s) {
                return Some(ch);
            }
        }
    }
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        for tok in stem.split(|c: char| !c.is_ascii_alphanumeric() && c != 'α') {
            if tok.is_empty() {
                continue;
            }
            let lower = tok.to_lowercase();
            match lower.as_str() {
                "ha" | "halpha" | "hα" => return Some(Channel::Ha),
                "oiii" | "o3" => return Some(Channel::Oiii),
                "sii" | "s2" => return Some(Channel::Sii),
                "red" => return Some(Channel::R),
                "green" => return Some(Channel::G),
                "blue" => return Some(Channel::B),
                "lum" | "luminance" => return Some(Channel::L),
                "rgb" | "osc" => return Some(Channel::RgbCfa),
                _ => {}
            }
        }
    }
    None
}

/// Discovers and classifies all supported FITS and camera RAW files under `input`.
pub fn scan(input: &Path) -> Result<Vec<FrameMeta>> {
    Ok(scan_detailed(input)?.frames)
}

pub fn scan_detailed(input: &Path) -> Result<ScanReport> {
    if !input.exists() {
        bail!("Input path '{}' does not exist", input.display());
    }
    if !input.is_dir() {
        bail!("Input path '{}' is not a directory", input.display());
    }

    let mut candidate_paths: Vec<(PathBuf, FileFormat)> = Vec::new();
    for entry in WalkDir::new(input).follow_links(false) {
        let entry = entry.with_context(|| {
            format!(
                "Failed while traversing input directory '{}'",
                input.display()
            )
        })?;
        if entry.file_type().is_file() {
            if let Some(fmt) = classify_extension(entry.path()) {
                candidate_paths.push((entry.into_path(), fmt));
            }
        }
    }
    candidate_paths.sort_by(|a, b| a.0.cmp(&b.0));

    if candidate_paths.is_empty() {
        bail!(
            "No supported FITS (.fit/.fits/.fts) or RAW (.cr2/.cr3/.nef/.arw/.dng/.raf/.rw2) files found in '{}'",
            input.display()
        );
    }

    let mut frames = Vec::with_capacity(candidate_paths.len());
    let mut light_frames = Vec::new();
    let mut calibration_frames = Vec::new();
    let mut rejected_frames = Vec::new();

    for (path, fmt) in candidate_paths {
        let frame = match fmt {
            FileFormat::Fits => {
                let meta = match fits::inspect_fits_header(&path) {
                    Ok(meta) => meta,
                    Err(error) => {
                        rejected_frames.push(RejectedFrameRecord {
                            path: path.display().to_string(),
                            reason: format!("Unreadable or invalid FITS header: {error:#}"),
                        });
                        continue;
                    }
                };
                let role = resolve_role(input, &path, meta.imagetyp.as_deref())?;
                let header_channel = match &meta.filter {
                    Some(f) => {
                        let Some(ch) = Channel::from_token(f) else {
                            bail!(
                                "Unrecognized FILTER '{}' in FITS file '{}'",
                                f,
                                path.display()
                            );
                        };
                        Some(ch)
                    }
                    None => None,
                };
                let path_channel = extract_path_channel(input, &path);
                if let (Some(hc), Some(pc)) = (header_channel, path_channel) {
                    if hc != pc && !(hc == Channel::RgbCfa || pc == Channel::RgbCfa) {
                        bail!(
                            "Conflicting channel assignment for '{}': FITS FILTER header says '{}' but path says '{}'",
                            path.display(),
                            hc,
                            pc
                        );
                    }
                }
                let channel = match role {
                    FrameRole::Bias | FrameRole::Dark => None,
                    FrameRole::Light | FrameRole::Flat | FrameRole::FlatDark => {
                        if meta.is_rgb_cube || meta.bayerpat.is_some() {
                            Some(header_channel.unwrap_or(Channel::RgbCfa))
                        } else {
                            // Mono without filter is L only, never guessed H/O/S
                            Some(header_channel.or(path_channel).unwrap_or(Channel::L))
                        }
                    }
                };
                fits::build_fits_frame(&path, meta, role, channel)
            }
            FileFormat::Raw => {
                let meta = match raw::inspect_raw_header(&path) {
                    Ok(meta) => meta,
                    Err(error) => {
                        rejected_frames.push(RejectedFrameRecord {
                            path: path.display().to_string(),
                            reason: format!("Unreadable or unsupported RAW header: {error:#}"),
                        });
                        continue;
                    }
                };
                let role = resolve_role(input, &path, None)?;
                let path_channel = extract_path_channel(input, &path);
                let channel = match role {
                    FrameRole::Bias | FrameRole::Dark => None,
                    FrameRole::Light | FrameRole::Flat | FrameRole::FlatDark => {
                        if meta.bayer.is_some() {
                            Some(Channel::RgbCfa)
                        } else {
                            Some(path_channel.unwrap_or(Channel::L))
                        }
                    }
                };
                raw::build_raw_frame(&path, meta, role, channel)
            }
        };

        if frame.role == FrameRole::Light {
            light_frames.push(frame.clone());
        } else {
            calibration_frames.push(frame.clone());
        }
        frames.push(frame);
    }

    if light_frames.is_empty() {
        bail!(
            "No light frames found in '{}' (found {} calibration frames only)",
            input.display(),
            calibration_frames.len()
        );
    }

    Ok(ScanReport {
        frames,
        light_frames,
        calibration_frames,
        warnings: Vec::new(),
        rejected_frames,
    })
}

/// Builds an intensity-normalized 2D luminance preview from a raw/FITS light frame for cross-filter star matching.
pub fn load_normalized_preview(frame: &FrameMeta) -> Result<LinearImage> {
    let raw_img = match frame.format {
        FileFormat::Fits => fits::read_fits_linear(frame)?,
        FileFormat::Raw => raw::read_raw_mosaic(frame)?,
    };

    // Convert CFA or 3-channel to 2D luminance preview
    let lum = if raw_img.channels == 3 {
        raw_img.to_luminance()
    } else if raw_img.bayer.is_some() {
        // Quick 2x2 neighborhood box average for preview star detection before full calibration
        let w = raw_img.width;
        let h = raw_img.height;
        let mut out = LinearImage::new(w, h, 1);
        out.mask = raw_img.mask.clone();
        for y in 0..h {
            let y1 = (y + 1).min(h - 1);
            for x in 0..w {
                let x1 = (x + 1).min(w - 1);
                let v = 0.25
                    * (raw_img.data[y * w + x]
                        + raw_img.data[y * w + x1]
                        + raw_img.data[y1 * w + x]
                        + raw_img.data[y1 * w + x1]);
                out.data[y * w + x] = v;
            }
        }
        out
    } else {
        raw_img
    };

    // Intensity-normalize preview using median and MAD so star detection works uniformly across filters
    let (med, mad) = register::robust_median_mad(&lum.data, &lum.mask);
    let scale = (mad * 1.4826).max(1e-3);
    let mut norm = lum;
    norm.data
        .par_iter_mut()
        .zip(norm.mask.par_iter())
        .for_each(|(p, &m)| {
            if m != 0 {
                *p = ((*p - med) / scale).max(0.0);
            } else {
                *p = 0.0;
            }
        });
    Ok(norm)
}

/// Partitions light frames by sensor/dimensions/binning and star-field overlap (>= 12 matched stars).
/// FITS `OBJECT` is used only as a display label and tie-breaker, never as proof that two fields match.
pub fn partition_targets(
    report: &ScanReport,
    options: &ProcessOptions,
) -> Result<Vec<TargetGroup>> {
    if report.light_frames.is_empty() {
        bail!("Cannot partition targets: no light frames present");
    }

    // Group first by (sensor, width, height, binning)
    type GeometryKey = (String, usize, usize, (u32, u32));
    let mut by_geom: BTreeMap<GeometryKey, Vec<FrameMeta>> = BTreeMap::new();
    for frame in &report.light_frames {
        let key = (
            frame.sensor.clone(),
            frame.width,
            frame.height,
            frame.binning,
        );
        by_geom.entry(key).or_default().push(frame.clone());
    }

    struct Cluster {
        ref_stars: Vec<register::DetectedStar>,
        object_label: Option<String>,
        frames: Vec<FrameMeta>,
    }

    let mut target_groups = Vec::new();
    let mut target_counter = 0usize;
    let mut progress = PhaseProgress::new(
        options,
        "Checking star fields",
        report.light_frames.len(),
        (0.05, 0.15),
    );
    let mut completed = 0;

    for ((sensor, width, height, binning), frames) in by_geom {
        let mut clusters: Vec<Cluster> = Vec::new();
        let mut low_star_frames: Vec<(FrameMeta, usize)> = Vec::new();

        // Each batch keeps at most a few full-resolution images in flight. Only
        // small star lists survive into the ordered clustering pass.
        let channels = frames.iter().map(|f| f.channels).max().unwrap_or(1);
        let batch_size = max_parallel_frames(width, height, channels);
        for batch in frames.chunks(batch_size) {
            let extracted: Vec<Result<Vec<register::DetectedStar>>> = batch
                .par_iter()
                .map(|frame| {
                    let preview = load_normalized_preview(frame)?;
                    Ok(register::detect_stars(&preview))
                })
                .collect();
            for (frame, stars) in batch.iter().cloned().zip(extracted) {
                let stars = stars?;
                if stars.len() < 12 {
                    low_star_frames.push((frame, stars.len()));
                    continue;
                }

                // Try matching against existing clusters by star-field overlap (>= 12 matched stars)
                let mut best_cluster_idx: Option<usize> = None;
                let mut best_inliers = 0usize;

                for (c_idx, cluster) in clusters.iter().enumerate() {
                    if let Ok(m) =
                        register::match_star_fields(&cluster.ref_stars, &stars, width, height)
                    {
                        if m.inliers >= 12 && m.overlap_fraction >= 0.50 {
                            let object_tiebreak =
                                frame.object.is_some() && frame.object == cluster.object_label;
                            if m.inliers > best_inliers
                                || (m.inliers == best_inliers && object_tiebreak)
                            {
                                best_inliers = m.inliers;
                                best_cluster_idx = Some(c_idx);
                            }
                        }
                    }
                }

                if let Some(c_idx) = best_cluster_idx {
                    if clusters[c_idx].object_label.is_none() && frame.object.is_some() {
                        clusters[c_idx].object_label = frame.object.clone();
                    }
                    clusters[c_idx].frames.push(frame);
                } else {
                    clusters.push(Cluster {
                        ref_stars: stars,
                        object_label: frame.object.clone(),
                        frames: vec![frame],
                    });
                }
            }
            completed += batch.len();
            progress.report(completed);
        }

        // If some frames had < 12 stars in raw preview, attach them to a matching cluster if one exists
        // so registration can record them in rejected_frames with a clear diagnostic, or fail if no cluster has >=12 stars.
        if clusters.is_empty() {
            let detail = low_star_frames
                .iter()
                .map(|(f, n)| format!("{} ({} stars)", f.path.display(), n))
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "No light frames for sensor '{}' ({}x{}) contained the minimum 12 detectable stars required for star-field verification: {}",
                sensor,
                width,
                height,
                detail
            );
        }

        for (lf, _count) in low_star_frames {
            // Tie-break by OBJECT label if multiple clusters exist, else first cluster
            let idx = clusters
                .iter()
                .position(|c| c.object_label.is_some() && c.object_label == lf.object)
                .unwrap_or(0);
            clusters[idx].frames.push(lf);
        }

        for cluster in clusters {
            target_counter += 1;
            let raw_label = cluster
                .object_label
                .clone()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| format!("Target {target_counter}"));
            let target_id = sanitize_target_id(&raw_label, target_counter);
            let target_label = if target_counter > 1 && cluster.object_label.is_some() {
                format!("{raw_label} (Field {target_counter})")
            } else {
                raw_label
            };

            let mut lights_by_channel: BTreeMap<Channel, Vec<FrameMeta>> = BTreeMap::new();
            for f in cluster.frames {
                let ch = f.channel.unwrap_or(Channel::L);
                lights_by_channel.entry(ch).or_default().push(f);
            }

            target_groups.push(TargetGroup {
                target_id,
                target_label,
                sensor: sensor.clone(),
                width,
                height,
                binning,
                lights_by_channel,
            });
        }
    }

    Ok(target_groups)
}

fn sanitize_target_id(label: &str, index: usize) -> String {
    let slug: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = slug.trim_matches('-');
    let clean = if trimmed.is_empty() {
        format!("target-{index}")
    } else {
        format!("{trimmed}-{index}")
    };
    clean
}
