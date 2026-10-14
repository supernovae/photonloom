//! Versioned, declarative restrictions on measured finishing operators.
//! Recipes select bounds and masks; identity never overrides image measurements.
use serde::{Deserialize, Serialize};
use super::{analysis::ImageAnalysis, operators::OperatorParams};

pub const RECIPE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeSpec {
    pub id: String,
    pub version: u32,
    pub classes: Vec<String>,
    pub required_metrics: Vec<String>,
    pub required_channels: Vec<String>,
    pub operations: Vec<String>,
    pub mask_names: Vec<String>,
    /// Maximum fractional tone, detail, sharpen and chroma change at q=1.
    pub bounds: OperatorParams,
    pub safety: String,
    pub fallback: String,
}

impl RecipeSpec {
    pub fn validate(&self) -> anyhow::Result<()> {
        use anyhow::ensure;
        ensure!(self.version == RECIPE_VERSION, "Unsupported recipe version");
        ensure!(self.fallback == "general-natural", "Unknown recipe fallback");
        ensure!(self.safety == "quality-v1", "Unknown quality policy");
        ensure!(self.bounds.is_valid(), "Recipe operator bounds exceed global limits");
        ensure!(!self.operations.is_empty() && self.operations.iter().all(|op| matches!(op.as_str(), "masked-tone" | "diffuse-detail" | "star-protected-sharpen" | "signal-chroma")), "Unknown recipe operation");
        ensure!(self.mask_names.iter().all(|mask| matches!(mask.as_str(), "stars" | "star_halos" | "sky" | "signal" | "core" | "low_snr" | "edges")), "Unknown recipe mask");
        ensure!(self.required_metrics.iter().all(|metric| matches!(metric.as_str(), "noise" | "snr" | "psf" | "core_range" | "structure")), "Unknown recipe metric");
        ensure!(self.required_channels.iter().all(|channel| matches!(channel.as_str(), "rgb" | "ha" | "oiii" | "sii")), "Unknown recipe channel");
        ensure!(all().iter().any(|spec| spec.id == self.id && spec.classes == self.classes && spec.operations == self.operations && spec.mask_names == self.mask_names && spec.required_metrics == self.required_metrics && spec.required_channels == self.required_channels && spec.bounds == self.bounds), "Recipe capabilities differ from allowlisted specification");
        Ok(())
    }

    pub fn available(&self, analysis: &ImageAnalysis, channels: &[String]) -> bool {
        self.required_metrics.iter().all(|metric| match metric.as_str() {
            "snr" => analysis.signal_snr.is_some_and(|v| v >= 3.0),
            "noise" => analysis.noise_sigma.is_some_and(|v| v > 0.0),
            "psf" => analysis.star_fwhm.is_some() && analysis.stars.fwhm_confidence >= 0.35,
            "structure" => analysis.structure.significant_fraction.is_some_and(|v| v > 0.0),
            "core_range" => analysis.dynamic_range.is_some_and(|v| v >= 20.0) && analysis.structure.core_fraction.is_some_and(|v| v > 0.0),
            _ => false,
        }) && self.required_channels.iter().all(|required| {
            if required == "rgb" { channels.len() >= 3 || channels.iter().any(|v| v.eq_ignore_ascii_case("rgb")) }
            else { channels.iter().any(|v| v.eq_ignore_ascii_case(required)) }
        })
    }

    pub fn strengths(&self, analysis: &ImageAnalysis, factor: f32) -> OperatorParams {
        let noise = analysis.noise_sigma.is_some_and(|v| v.is_finite() && v > 0.0);
        let psf = noise && analysis.star_fwhm.is_some_and(|v| v.is_finite() && (1.0..=32.0).contains(&v)) && analysis.stars.fwhm_confidence >= 0.35;
        let used = |name: &str| self.operations.iter().any(|op| op == name);
        let factor = factor.clamp(0.0, 1.0);
        OperatorParams {
            tone: if used("masked-tone") { self.bounds.tone * factor } else { 0.0 },
            detail: if noise && used("diffuse-detail") { self.bounds.detail * factor } else { 0.0 },
            sharpen: if psf && used("star-protected-sharpen") { self.bounds.sharpen * factor } else { 0.0 },
            color: if noise && used("signal-chroma") { self.bounds.color * factor } else { 0.0 },
        }
    }
}

fn spec(id: &str, classes: &[&str], metrics: &[&str], channels: &[&str], operations: &[&str], masks: &[&str], bounds: [f32;4]) -> RecipeSpec {
    RecipeSpec { id: id.into(), version: RECIPE_VERSION, classes: classes.iter().map(|s| (*s).into()).collect(),
        required_metrics: metrics.iter().map(|s| (*s).into()).collect(), required_channels: channels.iter().map(|s| (*s).into()).collect(),
        operations: operations.iter().map(|s| (*s).into()).collect(), mask_names: masks.iter().map(|s| (*s).into()).collect(),
        bounds: OperatorParams { tone: bounds[0], detail: bounds[1], sharpen: bounds[2], color: bounds[3] },
        safety: "quality-v1".into(), fallback: "general-natural".into() }
}

/// Frozen, reviewed capability set. Actual gains are multiplied by measured q and variant strength.
pub fn all() -> Vec<RecipeSpec> {
    vec![
        spec("general-natural", &["unknown", "star-field", "other-galaxy", "reflection-nebula"], &[], &[], &["masked-tone", "diffuse-detail", "star-protected-sharpen", "signal-chroma"], &["stars", "star_halos", "sky", "core", "low_snr", "signal"], [0.08,0.06,0.05,0.05]),
        spec("emission-nebula", &["emission-nebula"], &["snr", "noise", "structure"], &[], &["masked-tone", "diffuse-detail", "star-protected-sharpen", "signal-chroma"], &["stars", "star_halos", "sky", "signal", "core", "low_snr"], [0.12,0.15,0.07,0.15]),
        spec("spiral-galaxy", &["spiral-galaxy"], &["snr", "noise", "structure"], &[], &["masked-tone", "diffuse-detail", "star-protected-sharpen", "signal-chroma"], &["stars", "star_halos", "sky", "signal", "core", "low_snr"], [0.10,0.12,0.06,0.08]),
        spec("star-cluster", &["open-cluster", "globular-cluster"], &["psf", "noise"], &[], &["masked-tone", "star-protected-sharpen", "signal-chroma"], &["stars", "star_halos", "sky", "signal", "core"], [0.08,0.03,0.03,0.06]),
        spec("dark-nebula", &["dark-nebula"], &["snr", "noise", "structure"], &[], &["masked-tone", "diffuse-detail", "star-protected-sharpen", "signal-chroma"], &["stars", "star_halos", "sky", "signal", "low_snr"], [0.10,0.12,0.04,0.06]),
        spec("hdr-nebula", &["emission-nebula", "reflection-nebula"], &["core_range", "snr", "noise"], &[], &["masked-tone", "diffuse-detail", "star-protected-sharpen", "signal-chroma"], &["stars", "star_halos", "sky", "signal", "core"], [0.08,0.10,0.04,0.06]),
        spec("planetary-nebula", &["planetary-nebula"], &["psf", "snr", "noise"], &[], &["masked-tone", "diffuse-detail", "star-protected-sharpen", "signal-chroma"], &["stars", "star_halos", "sky", "signal", "core"], [0.10,0.12,0.08,0.10]),
    ]
}

pub fn select(class: Option<&str>, analysis: &ImageAnalysis, channels: &[String]) -> (RecipeSpec, Option<String>) {
    let recipes = all();
    let general = &recipes[0];
    if matches!(class, Some("emission-nebula" | "reflection-nebula")) && recipes[5].available(analysis, channels) {
        return (recipes[5].clone(), None);
    }
    let Some(class) = class else { return (general.clone(), Some("No unambiguous catalog identity; general measured recipe".into())); };
    let chosen = recipes.iter().skip(1).find(|recipe| recipe.classes.iter().any(|name| name == class));
    match chosen {
        Some(recipe) if recipe.available(analysis, channels) => (recipe.clone(), None),
        Some(recipe) => (general.clone(), Some(format!("{} lacks required measured metrics or science channels; general fallback", recipe.id))),
        None => (general.clone(), Some(format!("No bounded recipe for {class}; general fallback"))),
    }
}
