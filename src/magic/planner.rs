//! Opt-in, local-only suggestions. A proposal cannot expand a measured recipe's capabilities.
use super::{analysis::ImageAnalysis, recipes::{self, RecipeSpec}, MagicVariant};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{io::Read, path::Path, time::Duration};

const MAX_PARAMETERS: u64 = 16 * 1024;
const MAX_RESPONSE: u64 = 32 * 1024;
const MAX_REQUEST: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalScales {
    pub tone: f32,
    pub detail: f32,
    pub sharpen: f32,
    pub color: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagicProposal {
    pub version: u32,
    pub recipe_id: String,
    pub variant: MagicVariant,
    pub strength: f32,
    pub operator_scales: ProposalScales,
}

impl MagicProposal {
    /// Check the frozen recipe allowlist and reject nonzero unsupported operations.
    /// The caller also checks `validate_for_analysis` against the current image.
    pub fn validate(&self, recipe: &RecipeSpec) -> Result<()> {
        ensure!(self.version == 1, "Unsupported Magic proposal schema version");
        recipe.validate()?;
        ensure!(self.recipe_id == recipe.id, "Proposal recipe differs from selected recipe");
        ensure!(self.strength.is_finite() && (0.0..=1.0).contains(&self.strength), "Invalid proposal strength");
        let values = [
            (self.operator_scales.tone, "masked-tone"),
            (self.operator_scales.detail, "diffuse-detail"),
            (self.operator_scales.sharpen, "star-protected-sharpen"),
            (self.operator_scales.color, "signal-chroma"),
        ];
        for (scale, operation) in values {
            ensure!(scale.is_finite() && (0.0..=1.0).contains(&scale), "Invalid {operation} scale");
            ensure!(scale == 0.0 || recipe.operations.iter().any(|name| name == operation), "Recipe does not permit {operation}");
        }
        Ok(())
    }

    /// A missing measured prerequisite cannot be supplied by the LLM or parameter file.
    pub fn validate_for_analysis(&self, recipe: &RecipeSpec, analysis: &ImageAnalysis, channels: &[String]) -> Result<()> {
        self.validate(recipe)?;
        ensure!(recipe.available(analysis, channels), "Recipe requires unavailable measured metrics or science channels");
        let enabled = recipe.strengths(analysis, 1.0);
        ensure!(self.operator_scales.detail == 0.0 || enabled.detail > 0.0, "Detail requires measured noise");
        ensure!(self.operator_scales.sharpen == 0.0 || enabled.sharpen > 0.0, "Sharpening requires reliable measured PSF");
        ensure!(self.operator_scales.color == 0.0 || enabled.color > 0.0, "Color requires measured noise");
        Ok(())
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Read a versioned JSON parameters file without accepting trailing data or oversized input.
/// Recipe/image-specific checks must follow after the current source has been analyzed.
pub fn load_parameters(path: &Path) -> Result<MagicProposal> {
    let file = std::fs::File::open(path).with_context(|| format!("Opening Magic parameters {}", path.display()))?;
    ensure!(file.metadata()?.len() <= MAX_PARAMETERS, "Magic parameters exceed size limit");
    let mut bytes = Vec::new();
    file.take(MAX_PARAMETERS + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= MAX_PARAMETERS, "Magic parameters exceed size limit");
    let proposal: MagicProposal = serde_json::from_slice(&bytes).context("Invalid Magic parameters JSON")?;
    let recipe = recipes::all().into_iter().find(|r| r.id == proposal.recipe_id)
        .context("Unknown Magic proposal recipe ID")?;
    proposal.validate(&recipe)?;
    Ok(proposal)
}

#[derive(Deserialize)]
struct Completion { choices: Vec<Choice> }
#[derive(Deserialize)]
struct Choice { message: Message }
#[derive(Deserialize)]
struct Message { content: String }

/// Explicitly opt-in to a local planner. Only measured aggregate scalars and reviewed
/// recipe/mask identifiers leave this process; no headers, WCS, paths, history or pixels.
pub fn query_local(llm_url: &str, analysis: &ImageAnalysis, recipe: &RecipeSpec, variant: MagicVariant, strength: f32) -> Result<MagicProposal> {
    recipe.validate()?;
    ensure!(strength.is_finite() && (0.0..=1.0).contains(&strength), "Invalid requested strength");
    // `ImageAnalysis` lacks science-channel provenance. The selected recipe must be checked
    // against real channels by the caller before invoking this function.
    ensure!(recipe.required_metrics.iter().all(|metric| match metric.as_str() {
        "snr" => analysis.signal_snr.is_some_and(|v| v.is_finite() && v >= 3.0),
        "noise" => analysis.noise_sigma.is_some_and(|v| v.is_finite() && v > 0.0),
        "psf" => analysis.star_fwhm.is_some_and(|v| v.is_finite()) && analysis.stars.fwhm_confidence >= 0.35,
        "structure" => analysis.structure.significant_fraction.is_some_and(|v| v.is_finite() && v > 0.0),
        "core_range" => analysis.dynamic_range.is_some_and(|v| v.is_finite() && v >= 20.0) && analysis.structure.core_fraction.is_some_and(|v| v.is_finite() && v > 0.0),
        _ => false,
    }), "Recipe requires unavailable measured metrics");
    ensure!(recipe.required_channels.is_empty(), "Recipe needs channel provenance unavailable to local planner");
    let base = crate::models::validate_loopback_llm_url(llm_url)?;
    ensure!(base.username().is_empty() && base.password().is_none() && base.query().is_none() && base.fragment().is_none(), "Local LLM URL must not include credentials, query or fragment");
    let endpoint = base.join("/v1/chat/completions")?;
    let number = |value: Option<f32>| value.filter(|v| v.is_finite());
    let summary = serde_json::json!({
        "noise_sigma": number(analysis.noise_sigma),
        "signal_snr": number(analysis.signal_snr),
        "dynamic_range": number(analysis.dynamic_range),
        "measurement_confidence": number(Some(analysis.measurement_confidence)),
        "star_fwhm": number(analysis.star_fwhm),
        "psf_confidence": number(Some(analysis.stars.fwhm_confidence)),
        "structure_fraction": number(analysis.structure.significant_fraction),
        "core_fraction": number(analysis.structure.core_fraction),
        "contrast_scales": analysis.multiscale_contrast.map(number),
        "signal_chroma": number(analysis.color_stats.signal_chroma),
    });
    let payload = serde_json::json!({
        "model": "local-model", "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return ONLY one JSON object with exactly version (1), recipe_id, variant, strength, operator_scales {tone,detail,sharpen,color}. All scales and strength must be numbers in [0,1]. Use only the given recipe, variant, operations and masks. Do not invent metrics or channels."},
            {"role": "user", "content": serde_json::json!({
                "recipe_id": recipe.id, "variant": variant, "strength": strength,
                "allowed_operations": recipe.operations, "allowed_masks": recipe.mask_names,
                "measurements": summary,
            }).to_string()},
        ]
    });
    let request = serde_json::to_vec(&payload)?;
    ensure!(request.len() <= MAX_REQUEST, "Local LLM request exceeds size limit");
    // Blocking reqwest is run on a dedicated thread, safe even when called from a tokio worker.
    let body = std::thread::spawn(move || -> Result<Vec<u8>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(8)).redirect(reqwest::redirect::Policy::none())
            .no_proxy().build()?;
        let response = client.post(endpoint).header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(request).send().context("Local Magic planner unavailable")?;
        ensure!(response.status().is_success(), "Local Magic planner returned HTTP {}", response.status());
        ensure!(response.content_length().is_none_or(|len| len <= MAX_RESPONSE), "Local Magic planner response exceeds size limit");
        let mut bytes = Vec::new();
        response.take(MAX_RESPONSE + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() as u64 <= MAX_RESPONSE, "Local Magic planner response exceeds size limit");
        Ok(bytes)
    }).join().map_err(|_| anyhow::anyhow!("Local Magic planner thread panicked"))??;
    let completion: Completion = serde_json::from_slice(&body).context("Invalid local Magic planner response")?;
    ensure!(completion.choices.len() == 1, "Local Magic planner must return exactly one choice");
    let content = completion.choices[0].message.content.trim();
    ensure!(content.starts_with('{') && content.ends_with('}'), "Local Magic planner must return JSON only");
    let proposal: MagicProposal = serde_json::from_str(content).context("Invalid local Magic planner proposal")?;
    proposal.validate(recipe)?;
    ensure!(proposal.variant == variant && proposal.strength == strength, "Local Magic planner changed requested variant or strength");
    // `strengths` also disables operators with missing optional measurements.
    let enabled = recipe.strengths(analysis, 1.0);
    ensure!(proposal.operator_scales.detail == 0.0 || enabled.detail > 0.0, "Detail requires measured noise");
    ensure!(proposal.operator_scales.sharpen == 0.0 || enabled.sharpen > 0.0, "Sharpening requires reliable measured PSF");
    ensure!(proposal.operator_scales.color == 0.0 || enabled.color > 0.0, "Color requires measured noise");
    Ok(proposal)
}
