//! Measured, non-generative finishing of original integrations or display files.
pub mod analysis;
pub mod operators;
pub mod optimize;
pub mod planner;
pub mod preferences;
pub mod recipes;
pub mod source;
pub mod validate;

use crate::config::{GalleryManifest, MagicDecisionRef, VariantArtifact};
use crate::crop::CropRect;
use crate::finish::{generate_static_gallery_html, write_variant_images};
use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum InputDomain { Auto, Linear, Display }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum TransferFunction { Auto, Linear, Srgb }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum MagicVariant { Natural, Enhanced, Dramatic }
impl MagicVariant {
    pub fn as_str(self) -> &'static str { match self { Self::Natural => "natural", Self::Enhanced => "enhanced", Self::Dramatic => "dramatic" } }
    fn multiplier(self) -> f32 { match self { Self::Natural => 0.30, Self::Enhanced => 0.65, Self::Dramatic => 1.00 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum MagicSource {
    Gallery { gallery: PathBuf, target_id: String, palette: String },
    File { path: PathBuf, domain: InputDomain, transfer: TransferFunction },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MagicOptions {
    pub crop: bool,
    pub catalog_online: bool,
    pub magic_llm: Option<String>,
    pub magic_learn: bool,
    pub proposal: Option<planner::MagicProposal>,
    pub planner_variant: MagicVariant,
    pub planner_strength: f32,
}
impl Default for MagicOptions {
    fn default() -> Self {
        Self { crop: true, catalog_online: false, magic_llm: None, magic_learn: false, proposal: None,
            planner_variant: MagicVariant::Enhanced, planner_strength: 0.5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicDecision {
    pub version: u32,
    pub variant: MagicVariant,
    pub strength: f32,
    pub recipe_id: String,
    #[serde(default)]
    pub recipe_version: u32,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub catalog_score: f64,
    #[serde(default)]
    pub fallback_reason: Option<String>,
    pub measurement_quality: f32,
    pub nominal: operators::OperatorParams,
    pub actual: operators::OperatorParams,
    pub attempts: Vec<MagicAttempt>,
    #[serde(default)]
    pub search: Option<optimize::SearchReport>,
    #[serde(default)]
    pub proposal: Option<planner::MagicProposal>,
    pub explanation: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicAttempt {
    pub backoff: f32,
    pub quality: validate::QualityReport,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicCandidate {
    pub variant: MagicVariant,
    pub strength: f32,
    pub preview_path: String,
    pub decision: MagicDecision,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicSession {
    pub schema: u32,
    pub algorithm: String,
    pub source: MagicSource,
    pub source_digest: String,
    pub source_path: PathBuf,
    pub session_path: PathBuf,
    pub gallery_root: Option<PathBuf>,
    pub target_id: String,
    pub palette: String,
    pub channels_used: Vec<String>,
    pub dimensions: [usize; 2],
    pub crop: Option<CropRect>,
    pub domain: InputDomain,
    pub transfer: TransferFunction,
    pub assumption: Option<String>,
    #[serde(default)]
    pub subject: Option<crate::catalog::SubjectDecision>,
    #[serde(default)]
    pub catalog_online: bool,
    #[serde(default)]
    pub proposal: Option<planner::MagicProposal>,
    #[serde(default)]
    pub planner_warning: Option<String>,
    #[serde(default)]
    pub magic_learn: bool,
    #[serde(default)]
    pub preference_prior: Option<f32>,
    #[serde(default)]
    pub preference_variant: Option<MagicVariant>,
    pub analysis: analysis::ImageAnalysis,
    pub previews: Vec<MagicCandidate>,
    pub artifacts: Vec<VariantArtifact>,
}
fn persist(session: &MagicSession) -> Result<()> {
    crate::atomic_write_file(&session.session_path, &serde_json::to_vec_pretty(session)?)
}
fn sane_strength(strength: f32) -> Result<()> {
    ensure!(strength.is_finite() && (0.0..=1.0).contains(&strength), "Magic strength must be finite and in [0, 1]");
    Ok(())
}
fn loaded(session: &MagicSession) -> Result<source::PreviewSource> {
    ensure!(session.schema == 1 && session.algorithm == "magic-1", "Unsupported Magic session schema/algorithm");
    let src = source::load_preview(&session.source, session.crop.is_some())?;
    ensure!(src.digest == session.source_digest && src.target_id == session.target_id && src.palette == session.palette, "Magic source changed since analysis; existing artifacts are unchanged");
    ensure!(src.native_dimensions == session.dimensions, "Magic image geometry changed");
    Ok(src)
}
pub fn analyze(source: &MagicSource, options: &MagicOptions, output: &Path) -> Result<MagicSession> {
    sane_strength(options.planner_strength)?;
    ensure!(options.proposal.is_none() || (options.magic_llm.is_none() && !options.magic_learn),
        "Explicit Magic parameters conflict with local planner or preference learning");
    if let Some(url) = &options.magic_llm { crate::models::validate_loopback_llm_url(url)?; }
    let src = source::load_preview(source, options.crop)?;
    let output = crate::config::resolve_prospective_path(output)?;
    let dir = output.join("magic").join(&src.digest).join(&src.target_id);
    std::fs::create_dir_all(&dir)?;
    let (mut analysis, masks) = analysis::analyze_image(&src.baseline, src.domain);
    analysis.stars.original_to_preview_scale = Some(src.baseline.width as f32 / src.native_dimensions[0] as f32);
    let subject = crate::catalog::resolve_subject(src.subject_wcs.as_ref(), src.native_dimensions, &src.title, options.catalog_online);
    if let Some(name) = &subject.selected { analysis.structure.subject = Some(name.clone()); }
    let class = subject.selected.as_ref().map(|_| subject.class.as_str());
    let (recipe, _) = recipes::select(class, &analysis, &src.channels_used);
    recipe.validate()?;
    let mut planner_warning = None;
    let proposal = if let Some(proposal) = &options.proposal {
        proposal.validate_for_analysis(&recipe, &analysis, &src.channels_used)?;
        Some(proposal.clone())
    } else if let Some(url) = &options.magic_llm {
        match planner::query_local(url, &analysis, &recipe, options.planner_variant, options.planner_strength)
            .and_then(|p| { p.validate_for_analysis(&recipe, &analysis, &src.channels_used)?; Ok(p) }) {
            Ok(p) => Some(p),
            Err(error) => {
                planner_warning = Some(format!("Local planner unavailable ({error:#}); deterministic measured recipe used"));
                None
            }
        }
    } else { None };
    let (preference_prior, preference_variant) = if options.magic_learn {
        match preferences::default_path() {
            Ok(path) => {
                let (previous, warning) = preferences::load_matching(&path, true, subject.class.as_str(), &recipe.id);
                if warning.is_some() { planner_warning = warning; }
                match previous {
                    Some(p) => (Some(p.strength), Some(p.variant)),
                    None => (None, None),
                }
            }
            Err(error) => {
                planner_warning = Some(format!("Magic preference store unavailable ({error:#}); deterministic plan used"));
                (None, None)
            }
        }
    } else { (None, None) };
    let path = dir.join("session.json");
    let canonical_source = match source {
        MagicSource::Gallery { target_id, palette, .. } =>
            MagicSource::Gallery { gallery: src.source_path.clone(), target_id: target_id.clone(), palette: palette.clone() },
        MagicSource::File { domain, transfer, .. } =>
            MagicSource::File { path: src.source_path.clone(), domain: *domain, transfer: *transfer },
    };
    let mut session = MagicSession {
        schema: 1, algorithm: "magic-1".into(), source: canonical_source, source_digest: src.digest,
        source_path: src.source_path, session_path: path, gallery_root: src.gallery,
        target_id: src.target_id, palette: src.palette, channels_used: src.channels_used,
        dimensions: src.native_dimensions, crop: src.crop,
        domain: src.domain, transfer: src.transfer, assumption: src.assumption,
        subject: Some(subject), catalog_online: options.catalog_online,
        proposal, planner_warning, magic_learn: options.magic_learn, preference_prior, preference_variant,
        analysis, previews: Vec::new(), artifacts: Vec::new(),
    };
    if session.session_path.exists() {
        let previous: MagicSession = serde_json::from_slice(&std::fs::read(&session.session_path)?)?;
        ensure!(previous.schema == 1 && previous.source_digest == session.source_digest
            && previous.palette == session.palette && previous.crop == session.crop
            && previous.domain == session.domain && previous.transfer == session.transfer,
            "Magic session cache mismatch: source domain/transfer changed");
        if previous.catalog_online == session.catalog_online
            && previous.magic_learn == session.magic_learn && previous.preference_prior == session.preference_prior
            && previous.preference_variant == session.preference_variant
            && previous.proposal == session.proposal
            && previous.previews.iter().all(|p| p.decision.search.as_ref().is_some_and(|s| s.version == optimize::SEARCH_VERSION))
            && serde_json::to_value(&previous.subject)? == serde_json::to_value(&session.subject)? {
            session.previews = previous.previews;
            session.artifacts = previous.artifacts;
        }
    }
    save_masks(&dir, &mut session.analysis, &masks)?;
    persist(&session)?;
    Ok(session)
}
fn save_masks(dir: &Path, analysis: &mut analysis::ImageAnalysis, masks: &analysis::MaskAtlas) -> Result<()> {
    // Mask debug planes are bounded to preview geometry and versioned with the analysis.
    for (name, plane, slot) in [
        ("stars", &masks.stars, &mut analysis.masks.stars),
        ("star-halos", &masks.star_halos, &mut analysis.masks.star_halos),
        ("sky", &masks.sky, &mut analysis.masks.sky),
        ("signal", &masks.signal, &mut analysis.masks.signal),
        ("core", &masks.core, &mut analysis.masks.core),
        ("low-snr", &masks.low_snr, &mut analysis.masks.low_snr),
        ("edges", &masks.edges, &mut analysis.masks.edges),
    ] {
        let mut bytes = Vec::with_capacity(plane.len() * 4);
        for &value in plane { bytes.extend_from_slice(&value.clamp(0.0, 1.0).to_le_bytes()); }
        let binary_name = format!("mask-v1-{name}.f32");
        crate::atomic_write_file(&dir.join(&binary_name), &bytes)?;
        let debug: Vec<u8> = plane.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        let gray = image::GrayImage::from_raw(masks.width as u32, masks.height as u32, debug)
            .context("Invalid preview mask geometry")?;
        let mut encoded = std::io::Cursor::new(Vec::new());
        gray.write_to(&mut encoded, image::ImageFormat::Png)?;
        let png_name = format!("mask-v1-{name}.png");
        crate::atomic_write_file(&dir.join(&png_name), &encoded.into_inner())?;
        *slot = Some(png_name);
    }
    analysis.masks.version = 1;
    Ok(())
}
fn candidate(session: &MagicSession, src: &source::PreviewSource, variant: MagicVariant, strength: f32) -> Result<(crate::config::LinearImage, MagicDecision)> {
    sane_strength(strength)?;
    let (analysis, masks) = analysis::analyze_image(&src.baseline, src.domain);
    let q = analysis.signal_snr.map(|s| ((s - 3.0) / 5.0).clamp(0.0, 1.0).sqrt() * analysis.measurement_confidence).unwrap_or(0.0);
    let strength_curve = strength * (1.4 - 0.4 * strength);
    let factor = (variant.multiplier() * strength_curve * q).clamp(0.0, 1.0);
    let selected_class = session.subject.as_ref().filter(|subject| subject.selected.is_some()).map(|subject| subject.class.as_str());
    let (recipe, fallback_reason) = recipes::select(selected_class, &analysis, &session.channels_used);
    recipe.validate()?;
    let mut nominal = recipe.strengths(&analysis, factor);
    let proposal = session.proposal.as_ref().filter(|p| p.variant == variant && (p.strength - strength).abs() < 1e-6 && p.recipe_id == recipe.id);
    if let Some(proposal) = proposal {
        let scales = proposal.operator_scales;
        nominal.tone *= scales.tone;
        nominal.detail *= scales.detail;
        nominal.sharpen *= scales.sharpen;
        nominal.color *= scales.color;
    }
    let prior = (session.preference_variant == Some(variant)).then_some(session.preference_prior).flatten();
    let key = format!("{}:{}:{}:{}:{:?}:{:?}", session.source_digest, recipe.id, recipe.version,
        optimize::SEARCH_VERSION, session.crop, nominal);
    let result = optimize::search(&src.baseline, &analysis, &masks, nominal, prior, &key);
    let optimized = result.report.winner_parameters;
    let mut optimized_image = Some(result.image);
    let search = result.report;
    let mut attempts = Vec::new();
    let mut selected = src.baseline.clone();
    let mut actual = operators::OperatorParams::default();
    for backoff in [1.0, 0.5, 0.25, 0.0] {
        let params = operators::OperatorParams { tone: optimized.tone * backoff, detail: optimized.detail * backoff,
            sharpen: optimized.sharpen * backoff, color: optimized.color * backoff };
        let image = if backoff == 1.0 { optimized_image.take().context("Optimized image already used")? }
            else if backoff == 0.0 { src.baseline.clone() }
            else { operators::apply_candidate(&src.baseline, &analysis, &masks, &params) };
        let quality = validate::validate_candidate(&src.baseline, &image, &analysis, &masks);
        let passed = quality.passed;
        attempts.push(MagicAttempt { backoff, quality });
        if passed {
            selected = image;
            actual = params;
            break;
        }
    }
    let explanation = if strength == 0.0 {
        format!("Recipe {}: unmodified common baseline (strength 0.00)", recipe.id)
    } else if q == 0.0 {
        "Missing reliable significant-structure SNR; unchanged safe baseline".into()
    } else if actual == operators::OperatorParams::default() {
        format!("Recipe {}: no safely measurable gain; unchanged baseline", recipe.id)
    } else {
        format!(
            "Recipe {} ({}, {:.0}% scale): applied tone {:+.1}%, detail {:+.1}%, sharpen {:+.1}%, color {:+.1}% (SNR {:.2}, confidence {:.0}%, {} trials, sharpening {})",
            recipe.id,
            variant.as_str(),
            variant.multiplier() * strength * 100.0,
            actual.tone * 100.0,
            actual.detail * 100.0,
            actual.sharpen * 100.0,
            actual.color * 100.0,
            analysis.signal_snr.unwrap_or(0.0),
            analysis.measurement_confidence * 100.0,
            search.trials.len(),
            if actual.sharpen > 0.0 { "gated by measured PSF" } else { "disabled or unrewarded" }
        )
    };
    Ok((selected, MagicDecision { version: 3, variant, strength, recipe_id: recipe.id, recipe_version: recipe.version,
        subject: session.subject.as_ref().and_then(|s| s.selected.clone()),
        catalog_score: session.subject.as_ref().map_or(0.0, |s| s.score), fallback_reason,
        measurement_quality: q, nominal, actual, attempts, search: Some(search), proposal: proposal.cloned(), explanation }))
}
fn id(session: &MagicSession, variant: MagicVariant, strength: f32) -> String {
    let mut digest = Sha256::new();
    digest.update(session.source_digest.as_bytes());
    digest.update(session.palette.as_bytes());
    digest.update(variant.as_str().as_bytes());
    digest.update(strength.to_le_bytes());
    digest.update(recipes::RECIPE_VERSION.to_le_bytes());
    digest.update(optimize::SEARCH_VERSION.to_le_bytes());
    if let Some(proposal) = &session.proposal {
        digest.update(proposal.recipe_id.as_bytes());
        digest.update(proposal.variant.as_str().as_bytes());
        digest.update(proposal.strength.to_le_bytes());
        for value in [proposal.operator_scales.tone, proposal.operator_scales.detail, proposal.operator_scales.sharpen, proposal.operator_scales.color] {
            digest.update(value.to_le_bytes());
        }
    }
    if let (Some(prior), Some(preferred_variant)) = (session.preference_prior, session.preference_variant) {
        digest.update(preferred_variant.as_str().as_bytes());
        digest.update(prior.to_le_bytes());
    }
    if let Some(subject) = &session.subject {
        digest.update(subject.class.as_str().as_bytes());
        if let Some(selected) = &subject.selected { digest.update(selected.as_bytes()); }
    }
    format!("magic-{}-{}", variant.as_str(), &format!("{:x}", digest.finalize())[..12])
}
fn relative(session: &MagicSession, path: &Path) -> Result<String> {
    let root = session.session_path.parent().and_then(Path::parent).and_then(Path::parent).and_then(Path::parent).context("Invalid session location")?;
    Ok(path.strip_prefix(root)?.to_string_lossy().to_string())
}
pub fn preview(session: &mut MagicSession, strength: f32) -> Result<Vec<MagicCandidate>> {
    sane_strength(strength)?;
    let src = loaded(session)?;
    let dir = session.session_path.parent().context("Session has no parent")?;
    let mut results = Vec::new();
    for variant in [MagicVariant::Natural, MagicVariant::Enhanced, MagicVariant::Dramatic] {
        let (image, decision) = candidate(session, &src, variant, strength)?;
        let name = id(session, variant, strength);
        let png = dir.join(format!("{name}-preview.png"));
        let preview_tiff = dir.join(format!("{name}-preview.tiff"));
        write_variant_images(&image, &preview_tiff, &png)?;
        let item = MagicCandidate { variant, strength, preview_path: relative(session, &png)?, decision };
        session.previews.retain(|c| c.variant != variant || c.strength != strength);
        session.previews.push(item.clone());
        results.push(item);
    }
    persist(session)?;
    let root = session.session_path.parent().and_then(Path::parent).and_then(Path::parent).and_then(Path::parent).context("Invalid output root")?;
    if session.gallery_root.as_deref() != Some(root) {
        publish_standalone(session)?;
    }
    Ok(results)
}
fn accumulate_quality(total: &mut validate::QualityReport, next: &validate::QualityReport) {
    total.valid_pixels += next.valid_pixels;
    total.newly_clipped_pixels += next.newly_clipped_pixels;
    total.preexisting_clipped_pixels += next.preexisting_clipped_pixels;
    for c in 0..3 { total.newly_clipped_channels[c] += next.newly_clipped_channels[c]; }
    total.clipped_fraction = total.clipped_fraction.max(next.clipped_fraction);
    total.oversaturated_fraction = total.oversaturated_fraction.max(next.oversaturated_fraction);
    total.matched_stars += next.matched_stars;
    for (current, sample) in [
        (&mut total.sky_noise_ratio, next.sky_noise_ratio),
        (&mut total.sky_high_frequency_ratio, next.sky_high_frequency_ratio),
        (&mut total.bright_halo_overshoot_sigma, next.bright_halo_overshoot_sigma),
        (&mut total.mask_edge_seam_sigma, next.mask_edge_seam_sigma),
    ] {
        if let Some(sample) = sample {
            *current = Some(current.unwrap_or(0.0).max(sample));
        }
    }
    if let Some(sample) = next.faint_signal_retention {
        total.faint_signal_retention = Some(total.faint_signal_retention.unwrap_or(sample).min(sample));
    }
    if let Some(sample) = next.star_fwhm_ratio {
        let previous = total.star_fwhm_ratio.unwrap_or(1.0);
        if (sample - 1.0).abs() > (previous - 1.0).abs() {
            total.star_fwhm_ratio = Some(sample);
        }
    }
}

/// Reuse a single globally fitted native baseline. Every enhancement tile has
/// a reflected-halo neighborhood; no previous branch or preview is input.
fn render_native(
    session: &MagicSession, source: &source::PreviewSource,
    decision: &mut MagicDecision, tiff: &Path, png: &Path,
) -> Result<()> {
    let [width, height] = source.native_dimensions;
    let dir = session.session_path.parent().context("Session directory missing")?;
    let tiles = source::NativeTiles::from_preview(&session.source, session.crop.is_some(), source)?;
    let model = source::NativeBaselineModel::fit(&session.source, session.crop.is_some(), source)?;
    let mut baseline = source::BaselineScratch::create(dir, width, height)?;
    for y in (0..height).step_by(512) {
        for x in (0..width).step_by(512) {
            let w = (width - x).min(512);
            let h = (height - y).min(512);
            let tile = model.apply(tiles.read_tile(x, y, w, h)?, x, y)?;
            baseline.write_tile(x, y, &tile)?;
        }
    }
    let mut selected = None;
    let mut attempts = Vec::new();
    for backoff in [1.0, 0.5, 0.25, 0.0] {
        let mut output = source::BaselineScratch::create(dir, width, height)?;
        let params = operators::OperatorParams {
            tone: decision.nominal.tone * backoff,
            detail: decision.nominal.detail * backoff,
            sharpen: decision.nominal.sharpen * backoff,
            color: decision.nominal.color * backoff,
        };
        let mut rejected = None;
        let mut successful_quality: Option<validate::QualityReport> = None;
        for y in (0..height).step_by(512) {
            for x in (0..width).step_by(512) {
                let w = (width - x).min(512);
                let h = (height - y).min(512);
                let halo = 32usize;
                let left = x.saturating_sub(halo);
                let top = y.saturating_sub(halo);
                let right = (x + w + halo).min(width);
                let bottom = (y + h + halo).min(height);
                let original = baseline.read_tile(left, top, right - left, bottom - top)?;
                let (analysis, masks) = analysis::analyze_image(&original, session.domain);
                let candidate = if backoff == 0.0 { original.clone() } else {
                    operators::apply_candidate(&original, &analysis, &masks, &params)
                };
                let quality = validate::validate_candidate(&original, &candidate, &analysis, &masks);
                if !quality.passed {
                    rejected = Some(quality);
                    break;
                }
                if let Some(summary) = &mut successful_quality {
                    accumulate_quality(summary, &quality);
                } else {
                    successful_quality = Some(quality.clone());
                }
                let mut center = crate::config::LinearImage::new(w, h, 3);
                let src_n = original.pixel_count();
                let dst_n = center.pixel_count();
                for row in 0..h {
                    let start = (row + y - top) * original.width + x - left;
                    let dst = row * w;
                    center.mask[dst..dst + w].copy_from_slice(&candidate.mask[start..start + w]);
                    for c in 0..3 {
                        center.data[c * dst_n + dst..c * dst_n + dst + w]
                            .copy_from_slice(&candidate.data[c * src_n + start..c * src_n + start + w]);
                    }
                }
                output.write_tile(x, y, &center)?;
            }
            if rejected.is_some() { break; }
        }
        if let Some(quality) = rejected {
            attempts.push(MagicAttempt { backoff, quality });
            continue;
        }
        attempts.push(MagicAttempt { backoff, quality: successful_quality.context("Native quality summary unavailable")? });
        decision.actual = params;
        decision.attempts.extend(attempts);
        decision.explanation = if backoff == 0.0 {
            "Native quality gates rejected enhancements; unchanged safe full-resolution baseline".into()
        } else if backoff < 1.0 {
            format!("Native quality gates required {:.2}× backoff; {}", backoff, decision.explanation)
        } else { decision.explanation.clone() };
        selected = Some(output);
        break;
    }
    let mut selected = selected.context("Native baseline failed safety validation")?;
    selected.write_variant_images(tiff, png)
}

pub fn render(session: &mut MagicSession, variant: MagicVariant, strength: f32) -> Result<VariantArtifact> {
    sane_strength(strength)?;
    let mut src = loaded(session)?;
    let (_, mut decision) = candidate(session, &src, variant, strength)?;
    let dir = session.session_path.parent().context("Session has no parent")?;
    let name = id(session, variant, strength);
    let tiff = dir.join(format!("{name}.tiff"));
    let png = dir.join(format!("{name}.png"));
    if session.dimensions[0].checked_mul(session.dimensions[1]).context("Native dimensions overflow")? <= 2_000_000 {
        src.baseline = source::load(&session.source, session.crop.is_some())?.baseline;
        let (image, full_decision) = candidate(session, &src, variant, strength)?;
        decision = full_decision;
        write_variant_images(&image, &tiff, &png)?;
    } else {
        render_native(session, &src, &mut decision, &tiff, &png)?;
    }
    let decision_file = dir.join(format!("{name}-decision.json"));
    crate::atomic_write_file(&decision_file, &serde_json::to_vec_pretty(&decision)?)?;
    let artifact = VariantArtifact {
        id: name.clone(), title: format!("Magic {}", variant.as_str()), palette: session.palette.clone(),
        description: decision.explanation.clone(), artistic_interpretation: session.palette != "natural" && session.palette != "source-rgb", ai_enhanced: false,
        tiff_path: relative(session, &tiff)?, png_path: relative(session, &png)?, channels_used: session.channels_used.clone(),
        provenance: format!("Magic algorithm 1, source SHA-256 {}; RGB16 linear-light TIFF, sRGB PNG", session.source_digest),
        restore_info: None, finish: None, crop: session.crop,
        magic: Some(MagicDecisionRef { session: relative(session, &session.session_path)?, decision: relative(session, &decision_file)?, source_digest: session.source_digest.clone(), variant: Some(variant), strength: Some(strength) }),
    };
    session.artifacts.retain(|a| a.id != name);
    session.artifacts.push(artifact.clone());
    persist(session)?;
    let root = session.session_path.parent().and_then(Path::parent).and_then(Path::parent).and_then(Path::parent).context("Invalid output root")?;
    if session.gallery_root.as_deref() == Some(root) {
        publish_gallery(session, &artifact)?;
    } else {
        publish_standalone(session)?;
    }
    if session.magic_learn {
        let class = session.subject.as_ref().map_or("unknown", |s| s.class.as_str());
        let preference = preferences::Preference { class: class.into(), recipe_id: decision.recipe_id.clone(), variant, strength };
        if let Err(error) = preferences::default_path().and_then(|path| preferences::save(&path, &preference)) {
            session.planner_warning = Some(format!("Magic preference was not saved ({error:#})"));
            persist(session)?;
        } else if session.planner_warning.as_deref().is_some_and(|message| message.starts_with("Magic preferences unavailable:")) {
            session.planner_warning = None;
            persist(session)?;
        }
    }
    Ok(artifact)
}
fn publish_gallery(session: &MagicSession, artifact: &VariantArtifact) -> Result<()> {
    let root = session.gallery_root.as_ref().context("Not a gallery")?;
    let mut manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let target = manifest.targets.iter_mut().find(|t| t.target_id == session.target_id).context("Target changed")?;
    target.variants.retain(|v| v.id != artifact.id);
    target.variants.push(artifact.clone());
    target.selected_variant = artifact.id.clone();
    let path = artifact.magic.as_ref().context("Missing Magic decision")?.session.clone();
    if !target.magic_sessions.contains(&path) { target.magic_sessions.push(path); }
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    let run = root.join(&manifest.run_dir);
    crate::atomic_write_file(&run.join("manifest.json"), &bytes)?;
    crate::atomic_write_file(&run.join("index.html"), generate_static_gallery_html(&manifest, true)?.as_bytes())?;
    crate::atomic_write_file(&root.join("manifest.json"), &bytes)?;
    crate::atomic_write_file(&root.join("index.html"), generate_static_gallery_html(&manifest, false)?.as_bytes())?;
    Ok(())
}
fn publish_standalone(session: &MagicSession) -> Result<()> {
    let dir = session.session_path.parent().context("Missing session directory")?;
    let mut html = String::from("<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Magic View</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:1rem;background:#121720;color:#eef}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}figure{margin:0}img{max-width:100%}a{color:#a5d6ff}</style><h1>Magic View</h1><p>Original source unchanged; RGB16 linear-light TIFF and sRGB PNG.</p><section>");
    for candidate in &session.previews {
        html.push_str(&format!("<figure><img alt=\"{}\" src=\"{}\"><figcaption><strong>{}</strong>: {}</figcaption></figure>",
            candidate.variant.as_str(),
            Path::new(&candidate.preview_path).file_name().unwrap_or_default().to_string_lossy(),
            candidate.variant.as_str(), candidate.decision.explanation));
    }
    html.push_str("</section><h2>Full-resolution exports</h2><ul>");
    for artifact in &session.artifacts {
        let file = |path: &str| Path::new(path).file_name().unwrap_or_default().to_string_lossy().into_owned();
        html.push_str(&format!("<li>{}: <a href=\"{}\">RGB16 TIFF</a> · <a href=\"{}\">sRGB PNG</a> — {}</li>",
            artifact.title, file(&artifact.tiff_path), file(&artifact.png_path), artifact.description));
    }
    html.push_str("</ul>");
    crate::atomic_write_file(&dir.join("index.html"), html.as_bytes())
}
pub fn open(path: &Path) -> Result<MagicSession> {
    let path = path.canonicalize()?;
    let session: MagicSession = serde_json::from_slice(&std::fs::read(&path)?)?;
    ensure!(session.session_path == path, "Session path does not match its original location");
    loaded(&session)?;
    Ok(session)
}
