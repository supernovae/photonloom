use crate::crop::CropRect;
use crate::finish_style::{FinishOptions, ResolvedFinish};
use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum AcceleratorMode {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl fmt::Display for AcceleratorMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

pub type ProgressCallback = Arc<dyn Fn(&str, f32) + Send + Sync>;

#[derive(Clone, Serialize, Deserialize)]
pub struct ProcessOptions {
    pub accelerator: AcceleratorMode,
    pub palette: Option<String>,
    pub mix: Option<String>,
    pub restore: bool,
    #[serde(default)]
    pub magic_view: bool,
    #[serde(default)]
    pub finish: FinishOptions,
    #[serde(default = "default_trim_edges")]
    pub trim_edges: bool,
    #[serde(skip)]
    pub progress: Option<ProgressCallback>,
}

fn default_trim_edges() -> bool {
    true
}

impl Default for ProcessOptions {
    fn default() -> Self {
        Self {
            accelerator: AcceleratorMode::Auto,
            palette: None,
            mix: None,
            restore: false,
            magic_view: false,
            finish: FinishOptions::default(),
            trim_edges: true,
            progress: None,
        }
    }
}

impl fmt::Debug for ProcessOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessOptions")
            .field("accelerator", &self.accelerator)
            .field("palette", &self.palette)
            .field("mix", &self.mix)
            .field("restore", &self.restore)
            .field("magic_view", &self.magic_view)
            .field("finish", &self.finish)
            .field("trim_edges", &self.trim_edges)
            .field("has_progress", &self.progress.is_some())
            .finish()
    }
}

impl ProcessOptions {
    pub fn report_progress(&self, stage: &str, fraction: f32) {
        if let Some(cb) = &self.progress {
            cb(stage, fraction.clamp(0.0, 1.0));
        }
    }
}

/// Measured phase ETA; progress percentages are mapped into the enclosing run.
pub(crate) struct PhaseProgress<'a> {
    options: &'a ProcessOptions,
    stage: &'a str,
    total: usize,
    range: (f32, f32),
    started: std::time::Instant,
    last_report: std::time::Instant,
}

impl<'a> PhaseProgress<'a> {
    pub(crate) fn new(
        options: &'a ProcessOptions,
        stage: &'a str,
        total: usize,
        range: (f32, f32),
    ) -> Self {
        let now = std::time::Instant::now();
        Self {
            options,
            stage,
            total,
            range,
            started: now,
            last_report: now,
        }
    }

    pub(crate) fn report(&mut self, completed: usize) {
        if self.options.progress.is_none() || completed == 0 {
            return;
        }
        if completed != 1
            && completed < self.total
            && self.last_report.elapsed().as_secs_f32() < 1.0
        {
            return;
        }
        let elapsed = self.started.elapsed().as_secs_f64();
        let remaining = elapsed * self.total.saturating_sub(completed) as f64 / completed as f64;
        let seconds = remaining.ceil() as u64;
        let message = format!(
            "{}: {}/{} — phase ETA {}m {:02}s",
            self.stage,
            completed,
            self.total,
            seconds / 60,
            seconds % 60
        );
        let fraction = completed as f32 / self.total.max(1) as f32;
        self.options.report_progress(
            &message,
            self.range.0 + (self.range.1 - self.range.0) * fraction,
        );
        self.last_report = std::time::Instant::now();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Channel {
    R,
    G,
    B,
    L,
    Ha,
    Oiii,
    Sii,
    RgbCfa,
}

impl Channel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::R => "R",
            Self::G => "G",
            Self::B => "B",
            Self::L => "L",
            Self::Ha => "Ha",
            Self::Oiii => "OIII",
            Self::Sii => "SII",
            Self::RgbCfa => "RGB",
        }
    }

    pub fn from_token(token: &str) -> Option<Self> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return None;
        }
        let lower = trimmed.to_lowercase();
        let norm: String = lower
            .chars()
            .filter(|c| !matches!(c, '-' | '_' | ' ' | '.'))
            .collect();
        match norm.as_str() {
            "r" | "red" => Some(Self::R),
            "g" | "green" => Some(Self::G),
            "b" | "blue" => Some(Self::B),
            "l" | "lum" | "luminance" | "clear" | "mono" => Some(Self::L),
            "ha" | "halpha" | "hα" | "hydrogenalpha" => Some(Self::Ha),
            "oiii" | "o3" | "oxygeniii" => Some(Self::Oiii),
            "sii" | "s2" | "sulfurii" | "sulphurii" => Some(Self::Sii),
            "rgb" | "osc" | "color" | "cfa" => Some(Self::RgbCfa),
            _ => None,
        }
    }

    pub fn from_mix_channel(token: &str) -> Option<Self> {
        match Self::from_token(token) {
            Some(
                ch @ (Self::R | Self::G | Self::B | Self::L | Self::Ha | Self::Oiii | Self::Sii),
            ) => Some(ch),
            _ => None,
        }
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrameRole {
    Light,
    Dark,
    Bias,
    Flat,
    FlatDark,
}

impl FrameRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
            Self::Bias => "bias",
            Self::Flat => "flat",
            Self::FlatDark => "flatdark",
        }
    }

    pub fn from_token(token: &str) -> Option<Self> {
        let lower = token.trim().to_lowercase();
        let norm: String = lower
            .chars()
            .filter(|c| !matches!(c, '-' | '_' | ' '))
            .collect();
        match norm.as_str() {
            "light" | "lights" | "lightframe" | "science" => Some(Self::Light),
            "flatdark" | "flatdarks" | "darkflat" | "darkflats" | "flatdarkframe" => {
                Some(Self::FlatDark)
            }
            "dark" | "darks" | "darkframe" => Some(Self::Dark),
            "bias" | "biases" | "offset" | "offsets" | "biasframe" => Some(Self::Bias),
            "flat" | "flats" | "flatfield" | "flatframe" => Some(Self::Flat),
            _ => None,
        }
    }
}

impl fmt::Display for FrameRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BayerPattern {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
}

impl BayerPattern {
    pub fn from_str_token(s: &str) -> Option<Self> {
        match s.trim().to_uppercase().as_str() {
            "RGGB" => Some(Self::Rggb),
            "BGGR" => Some(Self::Bggr),
            "GRBG" => Some(Self::Grbg),
            "GBRG" => Some(Self::Gbrg),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rggb => "RGGB",
            Self::Bggr => "BGGR",
            Self::Grbg => "GRBG",
            Self::Gbrg => "GBRG",
        }
    }

    /// Returns 0 for Red, 1 for Green, 2 for Blue at `(x, y)`.
    #[inline]
    pub fn color_at(&self, x: usize, y: usize) -> usize {
        let idx = ((y & 1) << 1) | (x & 1);
        match self {
            Self::Rggb => [0, 1, 1, 2][idx],
            Self::Bggr => [2, 1, 1, 0][idx],
            Self::Grbg => [1, 0, 2, 1][idx],
            Self::Gbrg => [1, 2, 0, 1][idx],
        }
    }

    /// Returns true if `(x, y)` is a green site on a red row.
    #[inline]
    pub fn is_green_in_red_row(&self, x: usize, y: usize) -> bool {
        self.color_at(x, y) == 1 && self.color_at(x ^ 1, y) == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileFormat {
    Fits,
    Raw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub path: PathBuf,
    pub format: FileFormat,
    pub role: FrameRole,
    pub channel: Option<Channel>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub exposure: Option<f32>,
    pub temperature: Option<f32>,
    pub gain_or_iso: Option<f32>,
    pub binning: (u32, u32),
    pub bayer: Option<BayerPattern>,
    pub sensor: String,
    pub object: Option<String>,
    pub black_level: f32,
    pub white_level: f32,
    pub cblack: [f32; 4],
}

pub type FrameMeta = Frame;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearImage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// Plane-contiguous row-major data: `data[(c * height + y) * width + x]`.
    pub data: Vec<f32>,
    /// Per-pixel validity mask (`width * height`): 1 = valid, 0 = masked/invalid.
    pub mask: Vec<u8>,
    pub bayer: Option<BayerPattern>,
    pub exposure: f32,
}

impl LinearImage {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        let npix = width * height;
        Self {
            width,
            height,
            channels,
            data: vec![0.0; npix * channels],
            mask: vec![1u8; npix],
            bayer: None,
            exposure: 1.0,
        }
    }

    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    #[inline]
    pub fn idx(&self, c: usize, x: usize, y: usize) -> usize {
        (c * self.height + y) * self.width + x
    }

    #[inline]
    pub fn get(&self, c: usize, x: usize, y: usize) -> f32 {
        self.data[self.idx(c, x, y)]
    }

    #[inline]
    pub fn set(&mut self, c: usize, x: usize, y: usize, val: f32) {
        let i = self.idx(c, x, y);
        self.data[i] = val;
    }

    #[inline]
    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        self.mask[y * self.width + x] != 0
    }

    #[inline]
    pub fn set_valid(&mut self, x: usize, y: usize, valid: bool) {
        self.mask[y * self.width + x] = u8::from(valid);
    }

    pub fn plane(&self, c: usize) -> &[f32] {
        let npix = self.width * self.height;
        &self.data[c * npix..(c + 1) * npix]
    }

    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let npix = self.width * self.height;
        &mut self.data[c * npix..(c + 1) * npix]
    }

    pub fn extract_channel(&self, c: usize) -> LinearImage {
        LinearImage {
            width: self.width,
            height: self.height,
            channels: 1,
            data: self.plane(c).to_vec(),
            mask: self.mask.clone(),
            bayer: None,
            exposure: self.exposure,
        }
    }

    pub fn to_luminance(&self) -> LinearImage {
        if self.channels == 1 {
            return self.clone();
        }
        let npix = self.width * self.height;
        let mut lum = vec![0.0f32; npix];
        let r = self.plane(0);
        let g = self.plane(1);
        let b = self.plane(2);
        for i in 0..npix {
            if self.mask[i] != 0 {
                lum[i] = 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
            }
        }
        LinearImage {
            width: self.width,
            height: self.height,
            channels: 1,
            data: lum,
            mask: self.mask.clone(),
            bayer: None,
            exposure: self.exposure,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixTerm {
    pub channel: Channel,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixSpec {
    pub raw: String,
    pub expressions: [Vec<MixTerm>; 3],
}

impl MixSpec {
    /// Parses `--mix 'Ha:1,Ha:0.7+OIII:0.3,OIII:1'`.
    /// Must have 3 comma-separated output RGB expressions; each is `CHANNEL:weight[+CHANNEL:weight...]`,
    /// channel in `R,G,B,L,Ha,OIII,SII`, finite nonnegative decimal weights summing to 1 per output.
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.split(',').collect();
        if parts.len() != 3 {
            bail!(
                "Invalid --mix specification '{}': expected exactly 3 comma-separated RGB channel expressions (e.g. 'Ha:1,Ha:0.7+OIII:0.3,OIII:1'), got {}",
                spec,
                parts.len()
            );
        }
        let mut expressions: [Vec<MixTerm>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (out_idx, expr) in parts.iter().enumerate() {
            let expr = expr.trim();
            if expr.is_empty() {
                bail!(
                    "Invalid --mix specification: output channel {} is empty",
                    out_idx + 1
                );
            }
            let mut sum = 0.0f32;
            for term_str in expr.split('+') {
                let term_str = term_str.trim();
                let Some((ch_str, w_str)) = term_str.split_once(':') else {
                    bail!(
                        "Invalid --mix term '{}': expected CHANNEL:weight format",
                        term_str
                    );
                };
                let Some(channel) = Channel::from_mix_channel(ch_str) else {
                    bail!(
                        "Invalid --mix channel '{}': allowed channels are R,G,B,L,Ha,OIII,SII",
                        ch_str.trim()
                    );
                };
                let weight: f32 = w_str.trim().parse().with_context(|| {
                    format!(
                        "Invalid --mix weight '{}' in term '{}'",
                        w_str.trim(),
                        term_str
                    )
                })?;
                if !weight.is_finite() || weight < 0.0 {
                    bail!(
                        "Invalid --mix weight '{}' in term '{}': must be finite and nonnegative",
                        weight,
                        term_str
                    );
                }
                sum += weight;
                expressions[out_idx].push(MixTerm { channel, weight });
            }
            if (sum - 1.0).abs() > 1e-3 {
                bail!(
                    "Invalid --mix expression '{}': weights must sum to 1.0 per output channel (got {:.4})",
                    expr,
                    sum
                );
            }
        }
        Ok(Self {
            raw: spec.to_string(),
            expressions,
        })
    }

    pub fn required_channels(&self) -> Vec<Channel> {
        let mut req = Vec::new();
        for expr in &self.expressions {
            for term in expr {
                if term.weight > 0.0 && !req.contains(&term.channel) {
                    req.push(term.channel);
                }
            }
        }
        req
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EditParameters {
    pub palette: Option<String>,
    pub exposure: Option<f32>,
    pub saturation: Option<f32>,
    pub temperature: Option<f32>,
}

impl EditParameters {
    pub fn validate(&self, available_palettes: &[String]) -> Result<()> {
        if let Some(p) = &self.palette {
            let lower = p.to_lowercase();
            if !available_palettes
                .iter()
                .any(|ap| ap.to_lowercase() == lower)
            {
                bail!(
                    "Requested palette '{}' is not available in this gallery (available: {})",
                    p,
                    available_palettes.join(", ")
                );
            }
        }
        if let Some(exp) = self.exposure {
            if !exp.is_finite() || !(-2.0..=2.0).contains(&exp) {
                bail!(
                    "Edit exposure {} is out of bounds: must be finite in [-2.0, 2.0] stops",
                    exp
                );
            }
        }
        if let Some(sat) = self.saturation {
            if !sat.is_finite() || !(0.0..=2.0).contains(&sat) {
                bail!(
                    "Edit saturation {} is out of bounds: must be finite in [0.0, 2.0]",
                    sat
                );
            }
        }
        if let Some(temp) = self.temperature {
            if !temp.is_finite() || !(-1.0..=1.0).contains(&temp) {
                bail!(
                    "Edit temperature {} is out of bounds: must be finite in [-1.0, 1.0]",
                    temp
                );
            }
        }
        if self.palette.is_none()
            && self.exposure.is_none()
            && self.saturation.is_none()
            && self.temperature.is_none()
        {
            bail!("Edit specification did not contain any valid adjustments");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedFrameRecord {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibrationSummary {
    pub used_bias: Option<String>,
    pub used_darks: Vec<String>,
    pub used_flats: Vec<String>,
    pub used_flat_darks: Vec<String>,
    pub missing_masters: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreMetadata {
    pub sharpen_model_sha256: String,
    pub denoise_model_sha256: String,
    pub source_repo: String,
    pub source_revision: String,
    pub strength: f32,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantArtifact {
    pub id: String,
    pub title: String,
    pub palette: String,
    pub description: String,
    pub artistic_interpretation: bool,
    pub ai_enhanced: bool,
    pub tiff_path: String,
    pub png_path: String,
    pub channels_used: Vec<String>,
    pub provenance: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restore_info: Option<RestoreMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<ResolvedFinish>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop: Option<CropRect>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magic: Option<MagicDecisionRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicDecisionRef {
    pub session: String,
    pub decision: String,
    pub source_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<crate::magic::MagicVariant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmittedVariantRecord {
    pub palette: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScienceArtifact {
    pub channel: String,
    pub fits_path: String,
    pub width: usize,
    pub height: usize,
    pub total_exposure_s: f32,
    pub accepted_frames: Vec<String>,
    pub rejected_frames: Vec<RejectedFrameRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_crop: Option<CropRect>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wcs: Option<crate::fits::WcsTan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetGallery {
    pub target_id: String,
    pub target_label: String,
    pub primary_variant: String,
    pub selected_variant: String,
    pub science_integrations: Vec<ScienceArtifact>,
    pub variants: Vec<VariantArtifact>,
    pub omitted_variants: Vec<OmittedVariantRecord>,
    pub calibration_summary: CalibrationSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<crate::catalog::TargetInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub magic_sessions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryManifest {
    pub version: u32,
    pub run_id: String,
    pub run_dir: String,
    pub canonical_input: String,
    pub canonical_output: String,
    pub created_at_unix_ms: u64,
    pub accelerator_requested: String,
    pub accelerator_used: String,
    pub targets: Vec<TargetGallery>,
    pub warnings: Vec<String>,
    pub rejected_frames: Vec<RejectedFrameRecord>,
}

/// Computes the sibling default output directory `<INPUT basename>-photonloom`.
pub fn default_output_for_input(input: &Path) -> Result<PathBuf> {
    let canonical_in = std::fs::canonicalize(input).with_context(|| {
        format!(
            "Input directory '{}' does not exist or cannot be resolved",
            input.display()
        )
    })?;
    let name = canonical_in
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("observations");
    let parent = canonical_in.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent.join(format!("{name}-photonloom")))
}

/// Resolves canonical path even if the final leaf(s) do not exist yet, following symlinks on existing ancestors.
pub fn resolve_prospective_path(path: &Path) -> Result<PathBuf> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("Failed to determine current working directory")?
            .join(path)
    };
    let mut existing = abs.as_path();
    let mut tail = Vec::new();
    while !existing.exists() {
        if let Some(name) = existing.file_name() {
            tail.push(name.to_os_string());
        } else {
            break;
        }
        if let Some(parent) = existing.parent() {
            existing = parent;
        } else {
            break;
        }
    }
    let mut base = if existing.exists() {
        std::fs::canonicalize(existing)
            .with_context(|| format!("Failed to canonicalize path '{}'", existing.display()))?
    } else {
        existing.to_path_buf()
    };
    for part in tail.into_iter().rev() {
        if part == "." {
            continue;
        } else if part == ".." {
            base.pop();
        } else {
            base.push(part);
        }
    }
    Ok(base)
}

/// Validates that `input` and `output` are not equal and neither is nested inside the other.
pub fn validate_input_output_paths(input: &Path, output: &Path) -> Result<(PathBuf, PathBuf)> {
    let canon_in = std::fs::canonicalize(input).with_context(|| {
        format!(
            "Input directory '{}' does not exist or is inaccessible",
            input.display()
        )
    })?;
    if !canon_in.is_dir() {
        bail!("Input path '{}' is not a directory", canon_in.display());
    }
    let canon_out = resolve_prospective_path(output)?;
    if canon_in == canon_out {
        bail!(
            "Refusing to run: output directory '{}' is identical to input directory '{}'",
            canon_out.display(),
            canon_in.display()
        );
    }
    if canon_out.starts_with(&canon_in) {
        bail!(
            "Refusing to run: output directory '{}' is nested inside input directory '{}'",
            canon_out.display(),
            canon_in.display()
        );
    }
    if canon_in.starts_with(&canon_out) {
        bail!(
            "Refusing to run: input directory '{}' is nested inside output directory '{}'",
            canon_in.display(),
            canon_out.display()
        );
    }
    Ok((canon_in, canon_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_spec_parsing_valid_and_invalid() {
        let spec = MixSpec::parse("Ha:1,Ha:0.7+OIII:0.3,OIII:1").expect("valid mix");
        assert_eq!(spec.expressions[0].len(), 1);
        assert_eq!(spec.expressions[1].len(), 2);
        assert_eq!(spec.required_channels(), vec![Channel::Ha, Channel::Oiii]);

        assert!(MixSpec::parse("Ha:0.5,OIII:1,SII:1").is_err());
        assert!(MixSpec::parse("Ha:1,OIII:1").is_err());
        assert!(MixSpec::parse("FOO:1,OIII:1,SII:1").is_err());
        assert!(MixSpec::parse("Ha:1.2+OIII:-0.2,OIII:1,SII:1").is_err());
    }
}
