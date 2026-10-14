use crate::config::{EditParameters, GalleryManifest, RestoreMetadata, VariantArtifact};
use crate::finish_style::StarMode;
use crate::finish::rerender_variant_from_stored_fits;
use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const COSMIC_CLARITY_REPO: &str = "https://github.com/setiastro/cosmicclarity";
pub const COSMIC_CLARITY_REV: &str = "423acc442724872ff6d81011962a8cbf5744a7f7";

pub const SHARPEN_MODEL_FILENAME: &str = "deep_sharp_stellar_cnn.pth";
pub const SHARPEN_MODEL_SHA256: &str =
    "c480bdf8522b2e3797525217cad9e04058c59b2a32eb8f7d1f7e90d99aafb9be";

pub const DENOISE_MODEL_FILENAME: &str = "denoise_cnn.pth";
pub const DENOISE_MODEL_SHA256: &str =
    "9867c7d822223c922d37a92b6f2d1713d37f943d963b8cfbef43fe21918b405c";

pub const EMBEDDED_RESTORE_PY: &str = include_str!("../ml/restore.py");

#[derive(Debug, Clone)]
pub struct InstalledModels {
    pub dir: PathBuf,
    pub sharpen_path: PathBuf,
    pub denoise_path: PathBuf,
}

pub fn models_data_dir() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("PHOTONLOOM_MODELS_DIR") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }
    let proj = ProjectDirs::from("io", "photonloom", "photonloom")
        .context("Failed to determine OS per-user application data directory")?;
    Ok(proj.data_local_dir().join("models"))
}

pub fn compute_sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

pub fn verify_file_sha256(path: &Path, expected_hex: &str) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "Model weight file '{}' is not installed. Run `photonloom models install` first.",
            path.display()
        )
    })?;
    let actual = compute_sha256_hex(&bytes);
    if !actual.eq_ignore_ascii_case(expected_hex) {
        bail!(
            "SHA-256 checksum mismatch for '{}': expected {}, got {}",
            path.display(),
            expected_hex,
            actual
        );
    }
    Ok(actual)
}

/// Downloads and verifies pinned MIT-licensed Cosmic Clarity weights into user app data.
pub fn install_models() -> Result<InstalledModels> {
    let dir = models_data_dir()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create model directory '{}'", dir.display()))?;

    println!(
        "Notice: Downloading optional neural restoration weights from Seti Astro Cosmic Clarity ({COSMIC_CLARITY_REPO}, revision {COSMIC_CLARITY_REV})."
    );
    println!("License: MIT License (Copyright (c) 2024 Seti Astro / Franklin Marek).");

    let dir_clone = dir.clone();
    std::thread::spawn(move || -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .context("Failed to initialize HTTPS client for model download")?;

        let assets = [
            (SHARPEN_MODEL_FILENAME, SHARPEN_MODEL_SHA256),
            (DENOISE_MODEL_FILENAME, DENOISE_MODEL_SHA256),
        ];

        for (filename, expected_sha) in assets {
            let dest = dir_clone.join(filename);
            if dest.is_file() && verify_file_sha256(&dest, expected_sha).is_ok() {
                println!("Verified existing '{filename}' ({expected_sha})");
                continue;
            }
            let url = format!(
                "https://raw.githubusercontent.com/setiastro/cosmicclarity/{COSMIC_CLARITY_REV}/{filename}"
            );
            println!("Downloading {url} ...");
            let resp = client
                .get(&url)
                .send()
                .with_context(|| format!("Network error while downloading '{url}'"))?;
            if !resp.status().is_success() {
                bail!(
                    "Failed to download '{}': HTTP {}",
                    url,
                    resp.status()
                );
            }
            let bytes = resp
                .bytes()
                .with_context(|| format!("Failed to read response body from '{url}'"))?;
            let actual_sha = compute_sha256_hex(&bytes);
            if !actual_sha.eq_ignore_ascii_case(expected_sha) {
                bail!(
                    "Refusing to install '{}': SHA-256 mismatch (expected {}, got {})",
                    filename,
                    expected_sha,
                    actual_sha
                );
            }
            crate::atomic_write_file(&dest, &bytes)?;
            println!("Installed and verified '{filename}' (SHA-256: {actual_sha})");
        }
        Ok(())
    })
    .join()
    .map_err(|_| anyhow::anyhow!("Model download thread panicked"))??;

    let license_text = format!(
        "Seti Astro Cosmic Clarity Models\nSource: {COSMIC_CLARITY_REPO}\nRevision: {COSMIC_CLARITY_REV}\nLicense: MIT\n{SHARPEN_MODEL_FILENAME}: {SHARPEN_MODEL_SHA256}\n{DENOISE_MODEL_FILENAME}: {DENOISE_MODEL_SHA256}\n"
    );
    crate::atomic_write_file(
        &dir.join("LICENSE-COSMIC-CLARITY.txt"),
        license_text.as_bytes(),
    )?;

    Ok(InstalledModels {
        sharpen_path: dir.join(SHARPEN_MODEL_FILENAME),
        denoise_path: dir.join(DENOISE_MODEL_FILENAME),
        dir,
    })
}

pub fn resolve_python_executable() -> String {
    if let Ok(py) = std::env::var("PHOTONLOOM_PYTHON") {
        let trimmed = py.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if cfg!(windows) {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

#[derive(Deserialize)]
struct HelperOutput {
    device: String,
    strength: f32,
    sharpen_sha256: String,
    denoise_sha256: String,
}

/// Runs the embedded `ml/restore.py` via `python -c` on a finished 16-bit RGB TIFF.
pub fn run_cosmic_clarity_restore(
    input_tiff: &Path,
    output_tiff: &Path,
    output_png: &Path,
    strength: f32,
) -> Result<RestoreMetadata> {
    let dir = models_data_dir()?;
    let sharpen_path = dir.join(SHARPEN_MODEL_FILENAME);
    let denoise_path = dir.join(DENOISE_MODEL_FILENAME);

    verify_file_sha256(&sharpen_path, SHARPEN_MODEL_SHA256).with_context(|| {
        format!(
            "Cannot run --restore: '{}' is missing or unverified in '{}'. Run `photonloom models install` first.",
            SHARPEN_MODEL_FILENAME,
            dir.display()
        )
    })?;
    verify_file_sha256(&denoise_path, DENOISE_MODEL_SHA256).with_context(|| {
        format!(
            "Cannot run --restore: '{}' is missing or unverified in '{}'. Run `photonloom models install` first.",
            DENOISE_MODEL_FILENAME,
            dir.display()
        )
    })?;

    let py_bin = resolve_python_executable();
    let output = Command::new(&py_bin)
        .arg("-c")
        .arg(EMBEDDED_RESTORE_PY)
        .arg("--input-tiff")
        .arg(input_tiff)
        .arg("--output-tiff")
        .arg(output_tiff)
        .arg("--output-png")
        .arg(output_png)
        .arg("--sharpen-weights")
        .arg(&sharpen_path)
        .arg("--denoise-weights")
        .arg(&denoise_path)
        .arg("--strength")
        .arg(format!("{strength:.3}"))
        .output()
        .with_context(|| {
            format!(
                "Failed to execute Python helper '{}'. Set PHOTONLOOM_PYTHON or install Python 3 with `pip install torch numpy Pillow tifffile` and run `photonloom models install`.",
                py_bin
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "--restore helper failed (exit {:?}): {}\nEnsure optional dependencies are installed (`pip install -r ml/requirements-restore.txt`) and models are installed (`photonloom models install`).",
            output.status.code(),
            stderr.trim()
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout
        .lines()
        .rev()
        .find(|l| l.trim().starts_with('{'))
        .context("Restoration helper did not return JSON metadata on stdout")?;
    let parsed: HelperOutput = serde_json::from_str(last_line)
        .context("Failed to parse JSON metadata from restoration helper")?;

    Ok(RestoreMetadata {
        sharpen_model_sha256: parsed.sharpen_sha256,
        denoise_model_sha256: parsed.denoise_sha256,
        source_repo: COSMIC_CLARITY_REPO.to_string(),
        source_revision: COSMIC_CLARITY_REV.to_string(),
        strength: parsed.strength,
        device: parsed.device,
    })
}

/// Validates that a `--local-llm` URL uses HTTP(S) and points strictly to a loopback host.
pub fn validate_loopback_llm_url(raw_url: &str) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(raw_url)
        .with_context(|| format!("Invalid --local-llm URL '{raw_url}'"))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!(
            "Unsupported URL scheme '{}' in --local-llm: only http/https on loopback is allowed",
            url.scheme()
        );
    }
    let Some(host_str) = url.host_str() else {
        bail!("--local-llm URL '{raw_url}' is missing a host");
    };
    let clean_host = host_str.trim_start_matches('[').trim_end_matches(']');
    let is_loopback = if clean_host.eq_ignore_ascii_case("localhost") {
        true
    } else if let Ok(ip) = clean_host.parse::<IpAddr>() {
        ip.is_loopback()
    } else {
        false
    };
    if !is_loopback {
        bail!(
            "Refusing non-loopback --local-llm host '{}': local LLM endpoint must bind loopback (127.0.0.1, ::1, or localhost)",
            host_str
        );
    }
    Ok(url)
}

/// Parses the documented deterministic edit grammar:
/// - Palette names: `natural`, `hoo`, `sho`, `osh`, `hubble`, `jwst-inspired`, `hubble-inspired`, `monochrome`, `custom`
/// - Exposure: `brighter`, `brighten`, `darker`, `darken` (with optional `a little`/`slightly`/`half a stop`/`1 stop`)
/// - Temperature: `warmer`, `warm`, `cooler`, `cool`
/// - Saturation: `more saturation`, `more saturated`, `less saturation`, `less saturated`, `desaturate`
/// Rejects unknown/unsupported phrases rather than pretending a model complied.
pub fn parse_deterministic_edit(
    request: &str,
    available_palettes: &[String],
) -> Result<EditParameters> {
    let lower = request.trim().to_ascii_lowercase();
    if lower.is_empty() {
        bail!("Edit request cannot be empty");
    }

    let known_palettes = [
        "jwst-inspired",
        "hubble-inspired",
        "monochrome",
        "natural",
        "hubble",
        "custom",
        "sho",
        "hoo",
        "osh",
    ];

    let mut params = EditParameters::default();

    // Check for palette mention
    for pal in known_palettes {
        if contains_word_or_phrase(&lower, pal) {
            params.palette = Some(pal.to_string());
            break;
        }
    }

    // Determine magnitude modifier if present
    let is_slight = lower.contains("a little")
        || lower.contains("slightly")
        || lower.contains("a bit")
        || lower.contains("subtle");
    let is_strong = lower.contains("much") || lower.contains("very") || lower.contains("a lot");
    let half_stop = lower.contains("half a stop") || lower.contains("0.5 stop");
    let one_stop = lower.contains("one stop") || lower.contains("1 stop");

    // Exposure grammar
    if contains_word_or_phrase(&lower, "brighter")
        || contains_word_or_phrase(&lower, "brighten")
        || contains_word_or_phrase(&lower, "increase exposure")
    {
        let delta = if half_stop {
            0.5
        } else if one_stop {
            1.0
        } else if is_slight {
            0.35
        } else if is_strong {
            0.85
        } else {
            0.5
        };
        params.exposure = Some(delta);
    } else if contains_word_or_phrase(&lower, "darker")
        || contains_word_or_phrase(&lower, "darken")
        || contains_word_or_phrase(&lower, "decrease exposure")
    {
        let delta = if half_stop {
            -0.5
        } else if one_stop {
            -1.0
        } else if is_slight {
            -0.35
        } else if is_strong {
            -0.85
        } else {
            -0.5
        };
        params.exposure = Some(delta);
    }

    // Temperature grammar
    if contains_word_or_phrase(&lower, "warmer") || contains_word_or_phrase(&lower, "warm") {
        let t = if is_slight {
            0.25
        } else if is_strong {
            0.65
        } else {
            0.40
        };
        params.temperature = Some(t);
    } else if contains_word_or_phrase(&lower, "cooler") || contains_word_or_phrase(&lower, "cool") {
        let t = if is_slight {
            -0.25
        } else if is_strong {
            -0.65
        } else {
            -0.40
        };
        params.temperature = Some(t);
    }

    // Saturation grammar
    if lower.contains("more saturation")
        || lower.contains("more saturated")
        || lower.contains("increase saturation")
    {
        params.saturation = Some(if is_slight { 1.25 } else { 1.45 });
    } else if lower.contains("less saturation")
        || lower.contains("less saturated")
        || lower.contains("decrease saturation")
        || contains_word_or_phrase(&lower, "desaturate")
    {
        params.saturation = Some(if is_slight { 0.80 } else { 0.65 });
    }


    // Star mode / star separation grammar
    if contains_word_or_phrase(&lower, "starless")
        || contains_word_or_phrase(&lower, "without stars")
        || contains_word_or_phrase(&lower, "no stars")
        || contains_word_or_phrase(&lower, "remove stars")
        || contains_word_or_phrase(&lower, "remove the stars")
    {
        params.star_mode = Some(StarMode::Starless);
        params.star_blend = Some(0.0);
    } else if contains_word_or_phrase(&lower, "stars only")
        || contains_word_or_phrase(&lower, "only stars")
    {
        params.star_mode = Some(StarMode::StarsOnly);
        params.star_blend = Some(1.0);
    } else if contains_word_or_phrase(&lower, "reduce stars")
        || contains_word_or_phrase(&lower, "reduce the stars")
        || contains_word_or_phrase(&lower, "smaller stars")
        || contains_word_or_phrase(&lower, "fewer stars")
        || contains_word_or_phrase(&lower, "less stars")
        || contains_word_or_phrase(&lower, "dim stars")
        || contains_word_or_phrase(&lower, "subdue stars")
    {
        params.star_mode = Some(StarMode::Reduced);
        params.star_blend = Some(if is_slight {
            0.65
        } else if is_strong {
            0.25
        } else {
            0.45
        });
    } else if contains_word_or_phrase(&lower, "restore stars")
        || contains_word_or_phrase(&lower, "add stars")
        || contains_word_or_phrase(&lower, "keep stars")
        || contains_word_or_phrase(&lower, "with stars")
    {
        params.star_mode = Some(StarMode::Standard);
        params.star_blend = Some(1.0);
    }
    // Check for unrecognized words outside the documented deterministic grammar
    let allowed_vocab = [
        "make",
        "the",
        "image",
        "variant",
        "palette",
        "picture",
        "photo",
        "render",
        "switch",
        "to",
        "in",
        "with",
        "and",
        "a",
        "an",
        "little",
        "bit",
        "slightly",
        "subtle",
        "much",
        "very",
        "lot",
        "by",
        "half",
        "one",
        "stop",
        "stops",
        "0",
        "5",
        "1",
        "colors",
        "color",
        "tone",
        "tones",
        "brighter",
        "brighten",
        "darker",
        "darken",
        "increase",
        "decrease",
        "exposure",
        "warmer",
        "warm",
        "cooler",
        "cool",
        "more",
        "less",
        "saturation",
        "saturated",
        "desaturate",
        "natural",
        "hoo",
        "sho",
        "osh",
        "hubble",
        "jwst",
        "inspired",
        "monochrome",
        "custom",
        "please",
        "it",
        "starless",
        "star",
        "stars",
        "remove",
        "without",
        "no",
        "reduce",
        "smaller",
        "fewer",
        "dim",
        "subdue",
        "only",
        "add",
        "keep",
        "restore",
        "from",
    ];
    for token in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if token.is_empty() {
            continue;
        }
        if !allowed_vocab.contains(&token) {
            bail!(
                "Unsupported edit request phrase '{}' (unrecognized token '{}'). Supported deterministic grammar: brighter/darker, warmer/cooler, more/less saturation, and palette names (natural, hoo, sho, osh, hubble, jwst-inspired, hubble-inspired). Provide --local-llm http://127.0.0.1:8080 for free-form local LLM translation.",
                request,
                token
            );
        }
    }

    params.validate(available_palettes)?;
    Ok(params)
}

fn contains_word_or_phrase(haystack: &str, needle: &str) -> bool {
    let norm_h: Vec<&str> = haystack
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '-')
        .filter(|s| !s.is_empty())
        .collect();
    let norm_n: Vec<&str> = needle
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '-')
        .filter(|s| !s.is_empty())
        .collect();
    if norm_n.is_empty() || norm_h.len() < norm_n.len() {
        return false;
    }
    norm_h.windows(norm_n.len()).any(|w| w == norm_n.as_slice())
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    temperature: f32,
    messages: Vec<ChatMessage>,
    response_format: ResponseFormat,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

/// Calls a user-managed loopback `llama-server` `/v1/chat/completions` endpoint and validates
/// the returned constrained JSON `{ "palette", "exposure", "saturation", "temperature" }`.
pub fn query_local_llm_edit(
    llm_url: &str,
    request: &str,
    available_palettes: &[String],
) -> Result<EditParameters> {
    let base_url = validate_loopback_llm_url(llm_url)?;
    let endpoint = base_url
        .join("/v1/chat/completions")
        .context("Failed to construct /v1/chat/completions URL")?;

    let system_prompt = format!(
        "You are a constrained astrophotography gallery parameter translator. \
Respond ONLY with a single valid JSON object matching this exact schema: \
{{\"palette\": string|null, \"exposure\": number|null, \"saturation\": number|null, \"temperature\": number|null}}. \
Available palettes: [{}]. \
Bounds: exposure in [-2.0, 2.0] stops, saturation in [0.0, 2.0] (1.0 is unchanged), temperature in [-1.0, 1.0] (positive=warmer, negative=cooler). \
If the request is unsupported or asks for something outside these parameters, return all nulls.",
        available_palettes.join(", ")
    );

    let payload = ChatCompletionRequest {
        model: "local-model".to_string(),
        temperature: 0.0,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            },
            ChatMessage {
                role: "user".to_string(),
                content: request.to_string(),
            },
        ],
        response_format: ResponseFormat {
            kind: "json_object".to_string(),
        },
    };

    let body: ChatCompletionResponse =
        std::thread::spawn(move || -> Result<ChatCompletionResponse> {
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .context("Failed to build local LLM HTTP client")?;

            let resp = client
                .post(endpoint.clone())
                .json(&payload)
                .send()
                .with_context(|| format!("Failed to reach local LLM server at '{endpoint}'"))?;

            if !resp.status().is_success() {
                bail!(
                    "Local LLM server at '{}' returned HTTP {}",
                    endpoint,
                    resp.status()
                );
            }

            resp.json()
                .context("Invalid chat completion response JSON from local LLM server")
        })
        .join()
        .map_err(|_| anyhow::anyhow!("Local LLM request thread panicked"))??;

    let content = body
        .choices
        .first()
        .map(|c| c.message.content.trim())
        .context("Local LLM response contained no choices")?;
    // Extract JSON object if wrapped in markdown code fences
    let json_slice = extract_json_object(content)?;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawLlmEdit {
        palette: Option<String>,
        exposure: Option<f32>,
        saturation: Option<f32>,
        temperature: Option<f32>,
        #[serde(default)]
        star_mode: Option<StarMode>,
        #[serde(default)]
        star_blend: Option<f32>,
    }

    let raw: RawLlmEdit = serde_json::from_str(json_slice).with_context(|| {
        format!(
            "Local LLM returned invalid or non-conforming JSON: {}",
            content
        )
    })?;

    let params = EditParameters {
        palette: raw.palette,
        exposure: raw.exposure,
        saturation: raw.saturation,
        temperature: raw.temperature,
        star_mode: raw.star_mode,
        star_blend: raw.star_blend,
    };
    params.validate(available_palettes)?;
    Ok(params)
}

fn extract_json_object(s: &str) -> Result<&str> {
    let trimmed = s.trim();
    let Some(start) = trimmed.find('{') else {
        bail!("Local LLM output did not contain a JSON object: '{trimmed}'");
    };
    let Some(end) = trimmed.rfind('}') else {
        bail!("Local LLM output did not contain a closing JSON brace: '{trimmed}'");
    };
    if end < start {
        bail!("Malformed JSON delimiters in local LLM output: '{trimmed}'");
    }
    Ok(&trimmed[start..=end])
}

/// Applies a deterministic or local-LLM edit to an existing gallery without modifying science FITS.
pub fn apply_gallery_edit(
    gallery_dir: &Path,
    request: &str,
    local_llm: Option<&str>,
) -> Result<VariantArtifact> {
    let manifest_path = gallery_dir.join("manifest.json");
    let raw_manifest = std::fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "Failed to read gallery manifest at '{}'",
            manifest_path.display()
        )
    })?;
    let mut manifest: GalleryManifest =
        serde_json::from_str(&raw_manifest).context("Failed to parse gallery manifest.json")?;

    let target = manifest
        .targets
        .first()
        .context("Gallery manifest has no targets")?;
    let available_palettes: Vec<String> =
        target.variants.iter().map(|v| v.palette.clone()).collect();

    let edits = if let Some(llm_url) = local_llm {
        query_local_llm_edit(llm_url, request, &available_palettes)?
    } else {
        parse_deterministic_edit(request, &available_palettes)?
    };

    rerender_variant_from_stored_fits(
        gallery_dir,
        &mut manifest,
        edits.palette.as_deref(),
        None,
        None,
        None,
        Some(&edits),
        Some("edited"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_grammar_and_validation() {
        let avail = vec!["natural".to_string(), "hoo".to_string(), "sho".to_string()];
        let ed = parse_deterministic_edit("make the HOO image warmer", &avail).unwrap();
        assert_eq!(ed.palette.as_deref(), Some("hoo"));
        assert!(ed.temperature.unwrap() > 0.0);

        let ed2 = parse_deterministic_edit(
            "cool the colors a little and brighten by half a stop",
            &avail,
        )
        .unwrap();
        assert_eq!(ed2.exposure, Some(0.5));
        assert!(ed2.temperature.unwrap() < 0.0);

        assert!(parse_deterministic_edit("add a flying saucer and galaxies", &avail).is_err());
        assert!(validate_loopback_llm_url("http://127.0.0.1:8080").is_ok());
        assert!(validate_loopback_llm_url("http://192.168.1.50:8080").is_err());
    }
}
