pub mod calibrate;
pub mod catalog;
pub mod config;
pub mod crop;
pub mod finish;
pub mod finish_style;
pub mod fits;
pub mod gpu;
#[cfg(target_os = "linux")]
mod hip;
pub mod magic;
pub mod integrate;
pub mod models;
pub mod raw;
pub mod register;
pub mod scan;
pub mod web;

use anyhow::{Context, Result, bail};
pub use config::{
    AcceleratorMode, Channel, Frame, FrameMeta, GalleryManifest, LinearImage, ProcessOptions,
};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Atomic file write using a temporary file in the same directory followed by overwrite-safe persist.
pub fn atomic_write_file(dest: &Path, contents: &[u8]) -> Result<()> {
    let parent = dest
        .parent()
        .context("Destination file has no parent directory")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create parent directory '{}'", parent.display()))?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("Failed to create temporary file in '{}'", parent.display()))?;
    tmp.write_all(contents)
        .with_context(|| format!("Failed to write temporary file for '{}'", dest.display()))?;
    tmp.as_file()
        .sync_all()
        .with_context(|| format!("Failed to sync temporary file for '{}'", dest.display()))?;
    tmp.persist(dest).map_err(|e| e.error).with_context(|| {
        format!(
            "Failed to atomically replace destination file '{}'",
            dest.display()
        )
    })?;
    Ok(())
}

/// Checks if an existing output directory is safe to use for `canonical_input`.
/// Allows if `output` does not exist, if `output` is an empty directory (or only has `lost+found`),
/// or if `output/manifest.json` exists and names the same `canonical_input`.
pub fn check_existing_output_allowed(output: &Path, canonical_input: &Path) -> Result<bool> {
    if !output.exists() {
        return Ok(false);
    }
    if !output.is_dir() {
        bail!(
            "Output path '{}' exists and is not a directory",
            output.display()
        );
    }
    let manifest_path = output.join("manifest.json");
    if manifest_path.exists() {
        let raw = std::fs::read_to_string(&manifest_path).with_context(|| {
            format!(
                "Output directory '{}' exists with unreadable manifest.json",
                output.display()
            )
        })?;
        let existing: GalleryManifest = serde_json::from_str(&raw).with_context(|| {
            format!(
                "Output directory '{}' contains an invalid manifest.json; refusing to overwrite",
                output.display()
            )
        })?;
        let canon_in_str = canonical_input.to_string_lossy();
        if existing.canonical_input != canon_in_str {
            bail!(
                "Output directory '{}' already belongs to input '{}', which does not match requested input '{}'; refusing to overwrite",
                output.display(),
                existing.canonical_input,
                canon_in_str
            );
        }
        return Ok(true);
    }

    // Check if directory is empty (or only has lost+found for mounted K8s PVCs)
    let mut non_empty = false;
    for entry in std::fs::read_dir(output)
        .with_context(|| format!("Failed to inspect output directory '{}'", output.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        if name != "lost+found" {
            non_empty = true;
            break;
        }
    }
    if non_empty {
        bail!(
            "Output directory '{}' already exists and has no matching root manifest.json; refusing to overwrite existing files",
            output.display()
        );
    }
    Ok(true)
}

/// Main processing entry point for CLI and Web runs.
pub fn process(input: &Path, output: &Path, options: &ProcessOptions) -> Result<GalleryManifest> {
    options.report_progress("Validating paths and options", 0.02);
    options.finish.resolve()?;
    let (canon_in, canon_out) = config::validate_input_output_paths(input, output)?;

    // Validate custom mix upfront before scanning/calibrating
    let mix_spec = if let Some(mix_raw) = &options.mix {
        Some(config::MixSpec::parse(mix_raw)?)
    } else {
        None
    };

    let output_existed = check_existing_output_allowed(&canon_out, &canon_in)?;

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let run_id = format!("run-{now_ms}-{}", std::process::id());
    let run_rel_dir = format!("runs/{run_id}");
    let run_dir = canon_out.join("runs").join(&run_id);
    let staging_dir = run_dir.join("staging");

    std::fs::create_dir_all(&staging_dir).with_context(|| {
        format!(
            "Failed to create run staging directory '{}'",
            staging_dir.display()
        )
    })?;

    let result = process_inner(
        &canon_in,
        &canon_out,
        &run_id,
        &run_rel_dir,
        &run_dir,
        &staging_dir,
        now_ms,
        options,
        mix_spec.as_ref(),
    );

    let staging_cleanup = std::fs::remove_dir_all(&staging_dir).with_context(|| {
        format!(
            "Failed to remove staging directory '{}'",
            staging_dir.display()
        )
    });

    match result {
        Ok(mut manifest) => {
            staging_cleanup?;
            if options.magic_view {
                for target_id in manifest.targets.iter().map(|t| t.target_id.clone()).collect::<Vec<_>>() {
                    let source = magic::MagicSource::Gallery {
                        gallery: canon_out.clone(), target_id: target_id.clone(), palette: String::new(),
                    };
                    let result = (|| -> Result<()> {
                        let mut session = magic::analyze(&source, &magic::MagicOptions { crop: options.trim_edges, ..Default::default() }, &canon_out)?;
                        magic::preview(&mut session, 0.0)?;
                        magic::preview(&mut session, 0.5)?;
                        let artifact = magic::render(&mut session, magic::MagicVariant::Enhanced, 0.5)?;
                        if let Some(target) = manifest.targets.iter_mut().find(|target| target.target_id == target_id) {
                            target.selected_variant = artifact.id;
                        }
                        Ok(())
                    })();
                    if let Err(error) = result {
                        manifest.warnings.push(format!("Optional Magic View for '{target_id}' failed: {error:#}"));
                    }
                }
                // Magic publishes independently; preserve those variants before changing selection.
                let mut published: GalleryManifest = serde_json::from_slice(&std::fs::read(canon_out.join("manifest.json"))?)?;
                for target in &mut published.targets {
                    if let Some(selected) = manifest.targets.iter().find(|t| t.target_id == target.target_id) {
                        if selected.selected_variant.starts_with("magic-") {
                            target.selected_variant = selected.selected_variant.clone();
                        }
                    }
                }
                published.warnings = manifest.warnings;
                let bytes = serde_json::to_vec_pretty(&published)?;
                atomic_write_file(&run_dir.join("manifest.json"), &bytes)?;
                atomic_write_file(&run_dir.join("index.html"), finish::generate_static_gallery_html(&published, true)?.as_bytes())?;
                atomic_write_file(&canon_out.join("manifest.json"), &bytes)?;
                atomic_write_file(&canon_out.join("index.html"), finish::generate_static_gallery_html(&published, false)?.as_bytes())?;
                manifest = published;
            }
            options.report_progress("Succeeded", 1.0);
            Ok(manifest)
        }
        Err(err) => {
            // Keep failed partial runs hidden from root gallery and clean up owned run directory.
            let run_cleanup = std::fs::remove_dir_all(&run_dir)
                .with_context(|| format!("Failed to remove failed run '{}'", run_dir.display()));
            if let Some(runs_parent) = run_dir.parent() {
                let _ = std::fs::remove_dir(runs_parent);
            }
            if !output_existed {
                let _ = std::fs::remove_dir(&canon_out);
            }
            match (staging_cleanup, run_cleanup) {
                (Err(cleanup), _) | (_, Err(cleanup)) => Err(err.context(cleanup)),
                _ => Err(err),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_inner(
    canon_in: &Path,
    canon_out: &Path,
    run_id: &str,
    run_rel_dir: &str,
    run_dir: &Path,
    staging_dir: &Path,
    now_ms: u64,
    options: &ProcessOptions,
    mix_spec: Option<&config::MixSpec>,
) -> Result<GalleryManifest> {
    options.report_progress("Scanning and classifying frames", 0.05);
    let scan_report = scan::scan_detailed(canon_in)?;

    options.report_progress(
        "Partitioning targets and matching calibration masters",
        0.05,
    );
    let groups = scan::partition_targets(&scan_report, options)?;

    let mut warnings = scan_report.warnings.clone();
    let mut rejected_frames = scan_report.rejected_frames.clone();
    let mut target_galleries = Vec::new();
    let mut actual_accelerator = "cpu".to_string();

    let num_groups = groups.len().max(1);
    for (g_idx, group) in groups.iter().enumerate() {
        let base_prog = 0.20 + 0.65 * (g_idx as f32 / num_groups as f32);
        let mut calibration_summary = config::CalibrationSummary::default();
        let mut assembled = integrate::TargetIntegrationOutput {
            target_id: group.target_id.clone(),
            target_label: group.target_label.clone(),
            width: group.width,
            height: group.height,
            integrations: Default::default(),
            warnings: Vec::new(),
            rejected_frames: Vec::new(),
            used_gpu: false,
            cross_filter_aligned: true,
        };
        let channel_span = 0.45 / num_groups as f32 / group.lights_by_channel.len().max(1) as f32;
        for (channel_idx, (&channel, lights)) in group.lights_by_channel.iter().enumerate() {
            if lights.is_empty() {
                continue;
            }
            let channel_start = base_prog + channel_idx as f32 * channel_span;
            let calibration_end = channel_start + 0.30 * channel_span;
            options.report_progress(
                &format!(
                    "Calibrating target '{}' channel {}",
                    group.target_label, channel
                ),
                channel_start,
            );
            let channel_staging = staging_dir.join(&group.target_id).join(channel.as_str());
            let mut cal_output = calibrate::calibrate_target_group(
                group,
                channel,
                &scan_report.calibration_frames,
                &channel_staging,
                options,
                (channel_start, calibration_end),
            )?;
            let summary = std::mem::take(&mut cal_output.summary);
            if calibration_summary.used_bias.is_none() {
                calibration_summary.used_bias = summary.used_bias;
            }
            calibration_summary.used_darks.extend(summary.used_darks);
            calibration_summary.used_flats.extend(summary.used_flats);
            calibration_summary
                .used_flat_darks
                .extend(summary.used_flat_darks);
            for missing in summary.missing_masters {
                if !calibration_summary.missing_masters.contains(&missing) {
                    calibration_summary.missing_masters.push(missing);
                }
            }
            for warning in summary.warnings {
                if !calibration_summary.warnings.contains(&warning) {
                    calibration_summary.warnings.push(warning);
                }
            }
            options.report_progress(
                &format!(
                    "Registering and integrating target '{}' channel {}",
                    group.target_label, channel
                ),
                calibration_end,
            );
            let channel_output = integrate::integrate_target_channels(
                &group.target_id,
                &group.target_label,
                &cal_output,
                options,
                (calibration_end, channel_start + channel_span),
            )?;
            // These files belong to this run only; free disk before calibrating the next channel.
            std::fs::remove_dir_all(&channel_staging).with_context(|| {
                format!(
                    "Failed to remove completed channel spool '{}'",
                    channel_staging.display()
                )
            })?;
            assembled.integrations.extend(channel_output.integrations);
            assembled.warnings.extend(channel_output.warnings);
            assembled
                .rejected_frames
                .extend(channel_output.rejected_frames);
            assembled.used_gpu |= channel_output.used_gpu;
        }
        let stack_output = integrate::finalize_target_channels(assembled, run_dir, run_rel_dir)?;
        if stack_output.used_gpu {
            actual_accelerator = "gpu".to_string();
        }
        warnings.extend(calibration_summary.warnings.clone());
        warnings.extend(stack_output.warnings.clone());
        rejected_frames.extend(stack_output.rejected_frames.clone());

        options.report_progress(
            &format!("Rendering gallery variants for '{}'", group.target_label),
            base_prog + 0.45 / num_groups as f32,
        );

        let target_gallery = finish::render_target_gallery(
            &group.target_id,
            &group.target_label,
            &calibration_summary,
            stack_output,
            run_dir,
            run_rel_dir,
            options,
            mix_spec,
            &mut warnings,
        )?;
        target_galleries.push(target_gallery);
    }

    if target_galleries.is_empty() || target_galleries.iter().all(|t| t.variants.is_empty()) {
        bail!("Processing produced an empty gallery with no renderable variants");
    }

    let manifest = GalleryManifest {
        version: 1,
        run_id: run_id.to_string(),
        run_dir: run_rel_dir.to_string(),
        canonical_input: canon_in.to_string_lossy().to_string(),
        canonical_output: canon_out.to_string_lossy().to_string(),
        created_at_unix_ms: now_ms,
        accelerator_requested: options.accelerator.to_string(),
        accelerator_used: actual_accelerator,
        targets: target_galleries,
        warnings,
        rejected_frames,
    };

    options.report_progress("Writing gallery manifest and HTML", 0.95);
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("Failed to serialize GalleryManifest")?;

    // Write run-level manifest and index first
    atomic_write_file(&run_dir.join("manifest.json"), &manifest_bytes)?;
    let run_html = finish::generate_static_gallery_html(&manifest, true)?;
    atomic_write_file(&run_dir.join("index.html"), run_html.as_bytes())?;

    // Write root manifest.json with temp-file/rename, then replace root index.html last
    atomic_write_file(&canon_out.join("manifest.json"), &manifest_bytes)?;
    let root_html = finish::generate_static_gallery_html(&manifest, false)?;
    atomic_write_file(&canon_out.join("index.html"), root_html.as_bytes())?;

    Ok(manifest)
}
