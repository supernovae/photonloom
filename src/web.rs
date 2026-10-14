use crate::config::{
    AcceleratorMode, GalleryManifest, ProcessOptions, VariantArtifact, default_output_for_input,
};
use crate::finish::rerender_variant_from_stored_fits;
use crate::finish_style::{FinishOptions, FinishPreset, ResolvedFinish};
use crate::models::{apply_gallery_edit, validate_loopback_llm_url};
use crate::magic::{self, InputDomain, MagicOptions, MagicSession, MagicSource, MagicVariant, TransferFunction};
use anyhow::{Context, Result, bail};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, StatusCode, Uri, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, mpsc};

const INDEX_HTML: &str = include_str!("web/index.html");
const STYLE_CSS: &str = include_str!("web/style.css");
const APP_JS: &str = include_str!("web/app.js");
const HISTOGRAM_JS: &str = include_str!("web/histogram.js");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunJob {
    pub id: String,
    pub status: JobStatus,
    pub stage: String,
    pub progress: f32,
    pub input: String,
    pub output: String,
    pub palette: Option<String>,
    pub accelerator: AcceleratorMode,
    pub finish: FinishOptions,
    pub trim_edges: bool,
    #[serde(default)]
    pub magic_view: bool,
    pub warnings: Vec<String>,
    pub error: Option<String>,
    pub manifest: Option<GalleryManifest>,
}

#[derive(Debug, Clone, Serialize)]
struct MagicJob {
    id: String,
    status: JobStatus,
    stage: String,
    error: Option<String>,
    session: Option<serde_json::Value>,
    #[serde(skip)]
    session_path: Option<PathBuf>,
    #[serde(skip)]
    gallery_run: Option<String>,
}

type ApiError = (StatusCode, Json<serde_json::Value>);

fn api_error(status: StatusCode, message: impl std::fmt::Display) -> ApiError {
    (status, Json(serde_json::json!({"error": message.to_string()})))
}

fn public_session(session: &MagicSession) -> serde_json::Value {
    serde_json::json!({
        "schema": session.schema, "target_id": session.target_id, "palette": session.palette,
        "dimensions": session.dimensions, "crop": session.crop, "domain": session.domain,
        "transfer": session.transfer, "assumption": session.assumption,
        "subject": session.subject, "catalog_online": session.catalog_online,
        "proposal": session.proposal, "planner_warning": session.planner_warning,
        "magic_learn": session.magic_learn, "preference_prior": session.preference_prior,
        "analysis": session.analysis, "previews": session.previews, "artifacts": session.artifacts,
    })
}

#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<Mutex<HashMap<String, RunJob>>>,
    pub job_order: Arc<Mutex<Vec<String>>>,
    pub queue_tx: mpsc::UnboundedSender<String>,
    pub output_locks: Arc<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>>,
    magic_jobs: Arc<Mutex<HashMap<String, MagicJob>>>,
    pub local_llm: Option<String>,
}

impl AppState {
    fn get_output_lock(&self, output_dir: &Path) -> Arc<AsyncMutex<()>> {
        let mut map = self.output_locks.lock();
        map.entry(output_dir.to_path_buf())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateRunRequest {
    pub input: String,
    pub output: Option<String>,
    pub palette: Option<String>,
    pub mix: Option<String>,
    pub accelerator: Option<AcceleratorMode>,
    pub restore: Option<bool>,
    pub finish: Option<FinishOptions>,
    pub trim_edges: Option<bool>,
    #[serde(default)]
    pub magic_view: bool,
}

#[derive(Debug, Deserialize)]
pub struct VariantRerenderRequest {
    pub palette: Option<String>,
    pub mix: Option<String>,
    pub finish: Option<FinishOptions>,
    pub trim_edges: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct EditRequest {
    pub request: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GalleryMagicRequest {
    target_id: Option<String>,
    palette: Option<String>,
    strength: Option<f32>,
    variant: Option<MagicVariant>,
    options: Option<MagicOptions>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileMagicRequest {
    path: String,
    output: String,
    domain: InputDomain,
    transfer: TransferFunction,
    strength: Option<f32>,
    variant: Option<MagicVariant>,
    options: Option<MagicOptions>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MagicRenderRequest {
    variant: MagicVariant,
    strength: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GalleryMagicRenderRequest {
    session_id: String,
    variant: MagicVariant,
    strength: f32,
}

#[derive(Debug, Deserialize)]
struct MagicQuery {
    target_id: Option<String>,
    palette: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VariantUpdateResponse {
    pub variant: VariantArtifact,
    pub manifest: GalleryManifest,
}

/// Validates that `--bind` resolves exclusively to loopback (`127.0.0.1`, `::1`, or `localhost`).
pub fn parse_loopback_bind(bind: &str) -> Result<SocketAddr> {
    if let Ok(addr) = bind.parse::<SocketAddr>() {
        if !addr.ip().is_loopback() {
            bail!(
                "Refusing to bind non-loopback address '{}': `photonloom serve` binds loopback only (127.0.0.1 or ::1). Use an SSH tunnel for remote access.",
                addr
            );
        }
        return Ok(addr);
    }

    let addrs: Vec<SocketAddr> = bind
        .to_socket_addrs()
        .with_context(|| format!("Invalid --bind address '{bind}'"))?
        .collect();
    if addrs.is_empty() {
        bail!("--bind address '{bind}' did not resolve to any socket address");
    }
    for addr in &addrs {
        if !addr.ip().is_loopback() {
            bail!(
                "Refusing to bind '{}': resolved to non-loopback address '{}'",
                bind,
                addr
            );
        }
    }
    Ok(addrs[0])
}

fn check_same_origin_write(
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if let Some(sec_site) = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) {
        if sec_site.eq_ignore_ascii_case("cross-site") {
            return Err((
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": "Cross-site write request rejected" })),
            ));
        }
    }

    if let Some(origin_val) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        if origin_val == "null" {
            return Err((
                StatusCode::FORBIDDEN,
                Json(
                    serde_json::json!({ "error": "Cross-origin write request from 'null' origin rejected" }),
                ),
            ));
        }
        let Ok(origin_url) = reqwest::Url::parse(origin_val) else {
            return Err((
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": "Malformed Origin header rejected" })),
            ));
        };
        let Some(host_str) = origin_url.host_str() else {
            return Err((
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": "Origin header missing host" })),
            ));
        };
        let clean = host_str.trim_start_matches('[').trim_end_matches(']');
        let is_loopback = clean.eq_ignore_ascii_case("localhost")
            || clean
                .parse::<IpAddr>()
                .map(|ip| ip.is_loopback())
                .unwrap_or(false);
        if !is_loopback {
            return Err((
                StatusCode::FORBIDDEN,
                Json(
                    serde_json::json!({ "error": format!("Cross-origin write request from '{}' rejected", origin_val) }),
                ),
            ));
        }
        let request_host = headers.get(header::HOST).and_then(|v| v.to_str().ok())
            .and_then(|host| reqwest::Url::parse(&format!("http://{host}/")).ok());
        if origin_url.scheme() != "http"
            || request_host.as_ref().map_or(true, |host| host.origin() != origin_url.origin()) {
            return Err(api_error(StatusCode::FORBIDDEN, "Origin does not match this server's host and port"));
        }
    }

    Ok(())
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(serve_index))
        .route("/run", get(serve_index))
        .route("/gallery", get(serve_index))
        .route("/magic", get(serve_index))
        .route("/edit", get(serve_index))
        .route("/static/style.css", get(serve_css))
        .route("/static/app.js", get(serve_js))
        .route("/static/histogram.js", get(serve_histogram_js))
        .route("/api/finish-presets", get(finish_presets))
        .route("/api/runs", get(list_runs).post(create_run))
        .route("/api/runs/{id}", get(get_run))
        .route("/api/runs/{id}/gallery", get(get_run_gallery))
        .route("/api/runs/{id}/magic", get(get_run_magic).post(post_run_magic))
        .route("/api/runs/{id}/magic/render", post(post_run_magic_render))
        .route("/api/magic/preferences", get(get_magic_preferences))
        .route("/api/magic/preferences/reset", post(post_magic_reset_preferences))
        .route("/api/magic/sessions", post(post_file_magic))
        .route("/api/magic/sessions/{id}", get(get_magic_job))
        .route("/api/magic/sessions/{id}/render", post(post_file_magic_render))
        .route("/api/magic/sessions/{id}/assets/{*path}", get(get_magic_asset))
        .route("/api/runs/{id}/gallery/{*path}", get(get_run_gallery_asset))
        .route("/api/runs/{id}/variants", post(post_run_variant))
        .route("/api/runs/{id}/edit", post(post_run_edit))
        .fallback(fallback_handler)
        .with_state(state)
}

async fn finish_presets() -> Json<[ResolvedFinish; 4]> {
    Json(FinishPreset::options().map(FinishPreset::resolved_defaults))
}

/// Register a finished on-disk gallery as a succeeded job; never queue its input frames.
fn load_sample_gallery(gallery_dir: &Path) -> Result<RunJob> {
    let root = std::fs::canonicalize(gallery_dir)
        .with_context(|| format!("Cannot open sample gallery '{}'", gallery_dir.display()))?;
    if !root.is_dir() {
        bail!("Sample gallery '{}' is not a directory", root.display());
    }
    let manifest_path = root.join("manifest.json");
    let raw = std::fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "Cannot read sample gallery manifest '{}'",
            manifest_path.display()
        )
    })?;
    let mut manifest: GalleryManifest = serde_json::from_str(&raw).with_context(|| {
        format!(
            "Invalid sample gallery manifest '{}'",
            manifest_path.display()
        )
    })?;
    if Path::new(&manifest.canonical_output) != root {
        bail!(
            "Sample gallery '{}' has manifest output '{}'; serve the completed gallery at its recorded output path",
            root.display(),
            manifest.canonical_output
        );
    }
    if manifest.targets.is_empty() {
        bail!(
            "Sample gallery '{}' contains no completed targets",
            root.display()
        );
    }
    for target in &mut manifest.targets {
        if target.about.is_none() {
            target.about = crate::catalog::lookup_target(&target.target_label);
        }
        if target.variants.is_empty() || target.science_integrations.is_empty() {
            bail!(
                "Sample gallery target '{}' has no finished variants or integrations",
                target.target_label
            );
        }
        if !target
            .variants
            .iter()
            .any(|v| v.id == target.selected_variant)
        {
            bail!(
                "Sample gallery target '{}' has no selected variant '{}'",
                target.target_label,
                target.selected_variant
            );
        }
        for variant in &target.variants {
            for asset in [&variant.png_path, &variant.tiff_path] {
                resolve_safe_gallery_asset(&root, asset).with_context(|| {
                    format!(
                        "Sample gallery target '{}' has missing or invalid asset '{}'",
                        target.target_label, asset
                    )
                })?;
            }
        }
        for science in &target.science_integrations {
            resolve_safe_gallery_asset(&root, &science.fits_path).with_context(|| {
                format!(
                    "Sample gallery target '{}' has missing or invalid integration '{}'",
                    target.target_label, science.fits_path
                )
            })?;
        }
    }
    Ok(RunJob {
        id: "job-1".to_string(),
        status: JobStatus::Succeeded,
        stage: "Loaded existing gallery".to_string(),
        progress: 1.0,
        input: manifest.canonical_input.clone(),
        output: manifest.canonical_output.clone(),
        palette: None,
        accelerator: match manifest.accelerator_requested.as_str() {
            "gpu" => AcceleratorMode::Gpu,
            "cpu" => AcceleratorMode::Cpu,
            _ => AcceleratorMode::Auto,
        },
        finish: FinishOptions::default(),
        trim_edges: true,
        magic_view: false,
        warnings: manifest.warnings.clone(),
        error: None,
        manifest: Some(manifest),
    })
}

pub async fn serve(
    bind: &str,
    local_llm: Option<String>,
    sample_gallery: Option<PathBuf>,
) -> Result<()> {
    let addr = parse_loopback_bind(bind)?;
    if let Some(llm) = &local_llm {
        validate_loopback_llm_url(llm)?;
    }
    let sample_job = sample_gallery
        .as_deref()
        .map(load_sample_gallery)
        .transpose()
        .context("Failed to register sample gallery")?;

    let (queue_tx, mut queue_rx) = mpsc::unbounded_channel::<String>();
    let state = AppState {
        jobs: Arc::new(Mutex::new(HashMap::new())),
        job_order: Arc::new(Mutex::new(Vec::new())),
        queue_tx,
        magic_jobs: Arc::new(Mutex::new(HashMap::new())),
        output_locks: Arc::new(Mutex::new(HashMap::new())),
        local_llm,
    };
    if let Some(job) = sample_job {
        state.job_order.lock().push(job.id.clone());
        state.jobs.lock().insert(job.id.clone(), job);
    }

    // Single active run worker loop
    let worker_state = state.clone();
    tokio::spawn(async move {
        while let Some(job_id) = queue_rx.recv().await {
            run_queued_job(&worker_state, &job_id).await;
        }
    });

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("Failed to bind local web server on '{addr}'"))?;
    let local_addr = listener.local_addr()?;
    println!("Photonloom web gallery listening on http://{local_addr}");

    axum::serve(listener, build_router(state))
        .await
        .context("Web server terminated unexpectedly")
}

async fn run_queued_job(state: &AppState, job_id: &str) {
    let (input_str, output_str, palette, accelerator, finish, trim_edges, magic_view) = {
        let mut jobs = state.jobs.lock();
        let Some(job) = jobs.get_mut(job_id) else {
            return;
        };
        job.status = JobStatus::Running;
        job.stage = "Starting run".to_string();
        job.progress = 0.01;
        (
            job.input.clone(),
            job.output.clone(),
            job.palette.clone(),
            job.accelerator,
            job.finish,
            job.trim_edges,
            job.magic_view,
        )
    };

    let input_path = PathBuf::from(&input_str);
    let resolved_output = if output_str.trim().is_empty() {
        match default_output_for_input(&input_path) {
            Ok(p) => p,
            Err(err) => {
                let mut jobs = state.jobs.lock();
                if let Some(job) = jobs.get_mut(job_id) {
                    job.status = JobStatus::Failed;
                    job.stage = "Failed".to_string();
                    job.error = Some(format!("{err:#}"));
                }
                return;
            }
        }
    } else {
        PathBuf::from(&output_str)
    };

    let resolved_output = match crate::config::resolve_prospective_path(&resolved_output) {
        Ok(path) => path,
        Err(err) => {
            if let Some(job) = state.jobs.lock().get_mut(job_id) {
                job.status = JobStatus::Failed;
                job.stage = "Failed".into();
                job.error = Some(format!("{err:#}"));
            }
            return;
        }
    };
    {
        let mut jobs = state.jobs.lock();
        if let Some(job) = jobs.get_mut(job_id) {
            job.output = resolved_output.to_string_lossy().to_string();
        }
    }

    let out_lock = state.get_output_lock(&resolved_output);
    let _guard = out_lock.lock().await;

    let jobs_arc = state.jobs.clone();
    let jid = job_id.to_string();
    let progress_cb = Arc::new(move |stage: &str, pct: f32| {
        let mut jobs = jobs_arc.lock();
        if let Some(job) = jobs.get_mut(&jid) {
            job.stage = stage.to_string();
            job.progress = pct;
        }
    });

    let options = ProcessOptions {
        accelerator,
        palette,
        mix: None,
        restore: false,
        finish,
        trim_edges,
        magic_view,
        progress: Some(progress_cb),
    };

    let in_clone = input_path.clone();
    let out_clone = resolved_output.clone();
    let res =
        tokio::task::spawn_blocking(move || crate::process(&in_clone, &out_clone, &options)).await;

    let mut jobs = state.jobs.lock();
    if let Some(job) = jobs.get_mut(job_id) {
        match res {
            Ok(Ok(manifest)) => {
                job.status = JobStatus::Succeeded;
                job.stage = "Succeeded".to_string();
                job.progress = 1.0;
                job.output = manifest.canonical_output.clone();
                job.warnings = manifest.warnings.clone();
                job.manifest = Some(manifest);
            }
            Ok(Err(err)) => {
                job.status = JobStatus::Failed;
                job.stage = "Failed".to_string();
                job.error = Some(format!("{err:#}"));
            }
            Err(join_err) => {
                job.status = JobStatus::Failed;
                job.stage = "Failed".to_string();
                job.error = Some(format!("Worker panicked: {join_err}"));
            }
        }
    }
}

async fn serve_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn serve_css() -> Response {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLE_CSS,
    )
        .into_response()
}

async fn serve_js() -> Response {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        APP_JS,
    )
        .into_response()
}

async fn serve_histogram_js() -> Response {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        HISTOGRAM_JS,
    )
        .into_response()
}

async fn fallback_handler(uri: Uri) -> Response {
    let raw_path = uri.path();
    let lower = raw_path.to_ascii_lowercase();
    if raw_path.contains("..") || lower.contains("%2e%2e") || raw_path.starts_with("/api/runs/") || raw_path.starts_with("/api/magic/") {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": "Path traversal and unauthorized asset paths are denied" })),
        )
            .into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": "Not found" })),
    )
        .into_response()
}

async fn list_runs(State(state): State<AppState>) -> Json<Vec<RunJob>> {
    let order = state.job_order.lock();
    let jobs = state.jobs.lock();
    let list: Vec<RunJob> = order
        .iter()
        .rev()
        .filter_map(|id| jobs.get(id).cloned())
        .collect();
    Json(list)
}

async fn create_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateRunRequest>,
) -> Result<(StatusCode, Json<RunJob>), (StatusCode, Json<serde_json::Value>)> {
    check_same_origin_write(&headers)?;

    let input_trimmed = req.input.trim();
    if input_trimmed.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Input path cannot be empty" })),
        ));
    }

    let id = {
        let mut order = state.job_order.lock();
        let next_num = order.len() + 1;
        let id = format!("job-{next_num}");
        order.push(id.clone());
        id
    };

    let job = RunJob {
        id: id.clone(),
        status: JobStatus::Queued,
        stage: "Queued".to_string(),
        progress: 0.0,
        input: input_trimmed.to_string(),
        output: req.output.unwrap_or_default(),
        palette: req.palette,
        accelerator: req.accelerator.unwrap_or_default(),
        finish: req.finish.unwrap_or_default(),
        trim_edges: req.trim_edges.unwrap_or(true),
        magic_view: req.magic_view,
        warnings: Vec::new(),
        error: None,
        manifest: None,
    };

    {
        let mut jobs = state.jobs.lock();
        jobs.insert(id.clone(), job.clone());
    }

    let _ = state.queue_tx.send(id);
    Ok((StatusCode::ACCEPTED, Json(job)))
}

async fn get_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<RunJob>, (StatusCode, Json<serde_json::Value>)> {
    let jobs = state.jobs.lock();
    let Some(job) = jobs.get(&id).cloned() else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("Run '{id}' not found") })),
        ));
    };
    Ok(Json(job))
}

async fn get_run_gallery(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<GalleryManifest>, (StatusCode, Json<serde_json::Value>)> {
    let jobs = state.jobs.lock();
    let Some(job) = jobs.get(&id) else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("Run '{id}' not found") })),
        ));
    };
    let Some(manifest) = &job.manifest else {
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!("Gallery for run '{id}' is not ready (status: {:?})", job.status)
            })),
        ));
    };
    Ok(Json(manifest.clone()))
}

/// Validates and resolves an asset path inside `gallery_root`, denying traversal (`..`) and symlinks.
fn resolve_safe_gallery_asset(gallery_root: &Path, rel_path: &str) -> Result<PathBuf> {
    if rel_path.is_empty() || rel_path.contains('\\') || rel_path.contains('\0') {
        bail!("Invalid asset path");
    }
    let rel = Path::new(rel_path);
    for comp in rel.components() {
        match comp {
            Component::Normal(_) => {}
            _ => bail!("Path traversal component denied in '{}'", rel_path),
        }
    }

    let canon_root =
        std::fs::canonicalize(gallery_root).context("Failed to canonicalize gallery root")?;

    // Walk component-by-component and reject any symlink
    let mut current = canon_root.clone();
    for comp in rel.components() {
        current.push(comp.as_os_str());
        let meta = std::fs::symlink_metadata(&current)
            .with_context(|| format!("Asset '{}' does not exist", rel_path))?;
        if meta.file_type().is_symlink() {
            bail!("Symlinks are denied in served gallery assets");
        }
    }

    let canon_target = std::fs::canonicalize(&current)?;
    if !canon_target.starts_with(&canon_root) || !canon_target.is_file() {
        bail!("Asset path escapes gallery root or is not a regular file");
    }
    Ok(canon_target)
}

async fn get_run_gallery_asset(
    State(state): State<AppState>,
    AxumPath((id, rel_path)): AxumPath<(String, String)>,
) -> Response {
    let gallery_root = {
        let jobs = state.jobs.lock();
        let Some(job) = jobs.get(&id) else {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "Run not found" })),
            )
                .into_response();
        };
        let Some(manifest) = &job.manifest else {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({ "error": "Gallery not ready" })),
            )
                .into_response();
        };
        PathBuf::from(&manifest.canonical_output)
    };
    if rel_path.starts_with("magic/") && !public_magic_asset(&rel_path) {
        return api_error(StatusCode::FORBIDDEN, "Magic source/session assets are private").into_response();
    }

    let safe_path = match resolve_safe_gallery_asset(&gallery_root, &rel_path) {
        Ok(p) => p,
        Err(err) => {
            let msg = format!("{err:#}");
            let status = if msg.contains("does not exist") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::FORBIDDEN
            };
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let Ok(bytes) = tokio::fs::read(&safe_path).await else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to read asset file" })),
        )
            .into_response();
    };

    let ext = safe_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let content_type = match ext.as_str() {
        "png" => "image/png",
        "tiff" | "tif" => "image/tiff",
        "fits" | "fit" | "fts" => "application/fits",
        "json" => "application/json",
        "html" => "text/html; charset=utf-8",
        _ => "application/octet-stream",
    };

    ([(header::CONTENT_TYPE, content_type)], bytes).into_response()
}

async fn post_run_variant(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(req): Json<VariantRerenderRequest>,
) -> Result<Json<VariantUpdateResponse>, (StatusCode, Json<serde_json::Value>)> {
    check_same_origin_write(&headers)?;

    let gallery_dir = {
        let jobs = state.jobs.lock();
        let Some(job) = jobs.get(&id) else {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("Run '{id}' not found") })),
            ));
        };
        let Some(m) = &job.manifest else {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({ "error": "Run has not finished yet" })),
            ));
        };
        PathBuf::from(&m.canonical_output)
    };

    let out_lock = state.get_output_lock(&gallery_dir);
    let _guard = out_lock.lock().await;
    // Read after taking the output lock so a Magic render cannot be overwritten by a stale clone.
    let mut manifest: GalleryManifest = serde_json::from_slice(
        &std::fs::read(gallery_dir.join("manifest.json")).map_err(magic_error)?,
    ).map_err(magic_error)?;

    let pal = req.palette.clone();
    let mix = req.mix.clone();
    let finish = req.finish;
    let trim_edges = req.trim_edges;
    let suffix = finish
        .map(|f| serde_json::to_value(f.preset))
        .transpose()
        .map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": error.to_string()})),
            )
        })?
        .and_then(|value| value.as_str().map(str::to_owned));
    let g_dir = gallery_dir.clone();

    let res = tokio::task::spawn_blocking(move || {
        let variant = rerender_variant_from_stored_fits(
            &g_dir,
            &mut manifest,
            pal.as_deref(),
            mix.as_deref(),
            finish.as_ref(),
            trim_edges,
            None,
            suffix.as_deref(),
        )?;
        Ok::<_, anyhow::Error>((variant, manifest))
    })
    .await;

    match res {
        Ok(Ok((variant, updated_manifest))) => {
            let mut jobs = state.jobs.lock();
            if let Some(job) = jobs.get_mut(&id) {
                job.manifest = Some(updated_manifest.clone());
            }
            Ok(Json(VariantUpdateResponse {
                variant,
                manifest: updated_manifest,
            }))
        }
        Ok(Err(err)) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{err:#}") })),
        )),
        Err(join_err) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Worker error: {join_err}") })),
        )),
    }
}

async fn post_run_edit(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(req): Json<EditRequest>,
) -> Result<Json<VariantUpdateResponse>, (StatusCode, Json<serde_json::Value>)> {
    check_same_origin_write(&headers)?;

    let gallery_dir = {
        let jobs = state.jobs.lock();
        let Some(job) = jobs.get(&id) else {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("Run '{id}' not found") })),
            ));
        };
        let Some(m) = &job.manifest else {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({ "error": "Run has not finished yet" })),
            ));
        };
        PathBuf::from(&m.canonical_output)
    };

    let out_lock = state.get_output_lock(&gallery_dir);
    let _guard = out_lock.lock().await;

    let llm = state.local_llm.clone();
    let request_text = req.request.clone();
    let g_dir = gallery_dir.clone();

    let res = tokio::task::spawn_blocking(move || {
        let variant = apply_gallery_edit(&g_dir, &request_text, llm.as_deref())?;
        let raw_manifest = std::fs::read_to_string(g_dir.join("manifest.json"))?;
        let updated: GalleryManifest = serde_json::from_str(&raw_manifest)?;
        Ok::<_, anyhow::Error>((variant, updated))
    })
    .await;

    match res {
        Ok(Ok((variant, updated_manifest))) => {
            let mut jobs = state.jobs.lock();
            if let Some(job) = jobs.get_mut(&id) {
                job.manifest = Some(updated_manifest.clone());
            }
            Ok(Json(VariantUpdateResponse {
                variant,
                manifest: updated_manifest,
            }))
        }
        Ok(Err(err)) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{err:#}") })),
        )),
        Err(join_err) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Worker error: {join_err}") })),
        )),
    }
}

fn public_magic_asset(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or("");
    (name.starts_with("magic-") && (name.ends_with(".png") || name.ends_with(".tiff") || name.ends_with("-decision.json")))
        || (name.starts_with("mask-") && name.ends_with(".png"))
}

fn magic_error(message: impl std::fmt::Display) -> ApiError {
    api_error(StatusCode::BAD_REQUEST, message)
}

fn magic_strength(strength: f32) -> Result<f32, ApiError> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(magic_error("Strength must be between 0 and 1"));
    }
    Ok(strength)
}

fn gallery_root(state: &AppState, id: &str) -> Result<PathBuf, ApiError> {
    let jobs = state.jobs.lock();
    let job = jobs.get(id).ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Run not found"))?;
    let manifest = job.manifest.as_ref().ok_or_else(|| api_error(StatusCode::CONFLICT, "Gallery not ready"))?;
    Ok(PathBuf::from(&manifest.canonical_output))
}

fn gallery_selection(root: &Path, target_id: Option<&str>, palette: Option<&str>) -> Result<(String, String), ApiError> {
    let manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))
        .map_err(magic_error)?).map_err(magic_error)?;
    let target = match target_id {
        Some(id) => manifest.targets.iter().find(|t| t.target_id == id),
        None if manifest.targets.len() == 1 => manifest.targets.first(),
        None => return Err(magic_error("Select a target for this multi-target gallery")),
    }.ok_or_else(|| magic_error("Unknown target ID"))?;
    let requested = palette.filter(|p| !p.is_empty()).unwrap_or(&target.primary_variant);
    let palette = target.variants.iter()
        .find(|v| (v.id == requested || v.palette == requested) && v.magic.is_none() && v.restore_info.is_none())
        .map(|v| v.palette.clone())
        .ok_or_else(|| magic_error("Palette unavailable from original science channels"))?;
    Ok((target.target_id.clone(), palette))
}

fn new_magic_job(state: &AppState, gallery_run: Option<String>) -> MagicJob {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = format!("magic-job-{}", NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let job = MagicJob { id: id.clone(), status: JobStatus::Queued, stage: "Waiting for output lock".into(),
        error: None, session: None, session_path: None, gallery_run };
    state.magic_jobs.lock().insert(id, job.clone());
    job
}

async fn run_magic_analysis(state: AppState, job: MagicJob, source: MagicSource, output: PathBuf, options: MagicOptions, strength: f32) {
    let lock = state.get_output_lock(&output);
    let _guard = lock.lock().await;
    if let Some(entry) = state.magic_jobs.lock().get_mut(&job.id) {
        entry.status = JobStatus::Running;
        entry.stage = "Analyzing source and generating three previews".into();
    }
    let result = tokio::task::spawn_blocking(move || {
        if let MagicSource::Gallery { gallery, target_id, palette } = &source {
            let (selected, selected_palette) = gallery_selection(gallery, Some(target_id), Some(palette))
                .map_err(|(_, Json(v))| anyhow::anyhow!("{}", v["error"].as_str().unwrap_or("Invalid gallery selection")))?;
            if *target_id != selected || *palette != selected_palette { bail!("Gallery target or palette changed"); }
        }
        let mut session = magic::analyze(&source, &options, &output)?;
        magic::preview(&mut session, 0.0)?;
        let selected_strength = session.proposal.as_ref().map_or(strength, |proposal| proposal.strength);
        if selected_strength != 0.0 { magic::preview(&mut session, selected_strength)?; }
        Ok::<_, anyhow::Error>(session)
    }).await;
    let mut jobs = state.magic_jobs.lock();
    if let Some(entry) = jobs.get_mut(&job.id) {
        match result {
            Ok(Ok(session)) => {
                entry.status = JobStatus::Succeeded;
                entry.stage = "Three previews ready".into();
                entry.session_path = Some(session.session_path.clone());
                entry.session = Some(public_session(&session));
            }
            Ok(Err(e)) => { entry.status = JobStatus::Failed; entry.stage = "Magic View failed".into(); entry.error = Some(format!("{e:#}")); }
            Err(e) => { entry.status = JobStatus::Failed; entry.stage = "Magic View failed".into(); entry.error = Some(format!("Worker failed: {e}")); }
        }
    }
}

async fn start_magic_render(state: AppState, job: MagicJob, path: PathBuf, output: PathBuf, variant: MagicVariant, strength: f32) {
    let lock = state.get_output_lock(&output);
    let _guard = lock.lock().await;
    if let Some(entry) = state.magic_jobs.lock().get_mut(&job.id) {
        entry.status = JobStatus::Running;
        entry.stage = "Rendering full-resolution Magic variant".into();
    }
    let result = tokio::task::spawn_blocking(move || {
        let mut session = magic::open(&path)?;
        magic::render(&mut session, variant, strength)?;
        Ok::<_, anyhow::Error>(session)
    }).await;
    let mut jobs = state.magic_jobs.lock();
    if let Some(entry) = jobs.get_mut(&job.id) {
        match result {
            Ok(Ok(session)) => {
                entry.status = JobStatus::Succeeded;
                entry.stage = "Full-resolution render ready".into();
                entry.session_path = Some(session.session_path.clone());
                entry.session = Some(public_session(&session));
                if let Some(run_id) = &entry.gallery_run {
                    if let Ok(bytes) = std::fs::read(output.join("manifest.json")) {
                        if let Ok(manifest) = serde_json::from_slice::<GalleryManifest>(&bytes) {
                            if let Some(run) = state.jobs.lock().get_mut(run_id) { run.manifest = Some(manifest); }
                        }
                    }
                }
            }
            Ok(Err(e)) => { entry.status = JobStatus::Failed; entry.stage = "Full render failed".into(); entry.error = Some(format!("{e:#}")); }
            Err(e) => { entry.status = JobStatus::Failed; entry.stage = "Full render failed".into(); entry.error = Some(format!("Worker failed: {e}")); }
        }
    }
}

async fn get_magic_preferences() -> Result<Json<serde_json::Value>, ApiError> {
    let path = magic::preferences::default_path().map_err(magic_error)?;
    let (preference, warning) = magic::preferences::load(&path, true);
    Ok(Json(serde_json::json!({ "preference": preference, "warning": warning })))
}

async fn post_magic_reset_preferences(headers: HeaderMap) -> Result<Json<serde_json::Value>, ApiError> {
    check_same_origin_write(&headers)?;
    let path = magic::preferences::default_path().map_err(magic_error)?;
    magic::preferences::reset(&path).map_err(magic_error)?;
    Ok(Json(serde_json::json!({ "reset": true })))
}

async fn post_run_magic(State(state): State<AppState>, headers: HeaderMap, AxumPath(id): AxumPath<String>,
    Json(req): Json<GalleryMagicRequest>) -> Result<(StatusCode, Json<MagicJob>), ApiError> {
    check_same_origin_write(&headers)?;
    let root = gallery_root(&state, &id)?;
    let strength = magic_strength(req.strength.unwrap_or(0.5))?;
    let (target_id, palette) = gallery_selection(&root, req.target_id.as_deref(), req.palette.as_deref())?;
    let job = new_magic_job(&state, Some(id));
    let mut options = req.options.unwrap_or_default();
    options.planner_strength = strength;
    options.planner_variant = req.variant.unwrap_or(MagicVariant::Enhanced);
    tokio::spawn(run_magic_analysis(state, job.clone(),
        MagicSource::Gallery { gallery: root.clone(), target_id, palette }, root,
        options, strength));
    Ok((StatusCode::ACCEPTED, Json(job)))
}

async fn post_file_magic(State(state): State<AppState>, headers: HeaderMap,
    Json(req): Json<FileMagicRequest>) -> Result<(StatusCode, Json<MagicJob>), ApiError> {
    check_same_origin_write(&headers)?;
    let strength = magic_strength(req.strength.unwrap_or(0.5))?;
    let path = std::fs::canonicalize(&req.path).map_err(magic_error)?;
    if !path.is_file() { return Err(magic_error("Input must be a regular file")); }
    if req.output.trim().is_empty() { return Err(magic_error("Output directory is required")); }
    let output = crate::config::resolve_prospective_path(Path::new(&req.output)).map_err(magic_error)?;
    let job = new_magic_job(&state, None);
    let mut options = req.options.unwrap_or_default();
    options.planner_strength = strength;
    options.planner_variant = req.variant.unwrap_or(MagicVariant::Enhanced);
    tokio::spawn(run_magic_analysis(state, job.clone(),
        MagicSource::File { path, domain: req.domain, transfer: req.transfer },
        output, options, strength));
    Ok((StatusCode::ACCEPTED, Json(job)))
}

async fn get_magic_job(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Result<Json<MagicJob>, ApiError> {
    state.magic_jobs.lock().get(&id).cloned().map(Json)
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Magic session not found"))
}

async fn get_run_magic(State(state): State<AppState>, AxumPath(id): AxumPath<String>,
    Query(query): Query<MagicQuery>) -> Result<Json<serde_json::Value>, ApiError> {
    let root = gallery_root(&state, &id)?;
    let (target, palette) = gallery_selection(&root, query.target_id.as_deref(), query.palette.as_deref())?;
    let jobs = state.magic_jobs.lock();
    let latest = jobs.values().filter(|job| job.gallery_run.as_deref() == Some(id.as_str())
        && job.status == JobStatus::Succeeded
        && job.session.as_ref().and_then(|s| s["target_id"].as_str()) == Some(target.as_str())
        && job.session.as_ref().and_then(|s| s["palette"].as_str()) == Some(palette.as_str()))
        .max_by_key(|job| job.id.strip_prefix("magic-job-").and_then(|s| s.parse::<u64>().ok()).unwrap_or(0));
    if let Some(job) = latest {
        return Ok(Json(serde_json::json!({"session": job.session, "id": job.id})));
    }
    drop(jobs);
    let manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))
        .map_err(magic_error)?).map_err(magic_error)?;
    let target = manifest.targets.iter().find(|t| t.target_id == target).unwrap();
    for reference in target.magic_sessions.iter().rev() {
        if !reference.starts_with("magic/") { continue; }
        let Ok(path) = resolve_safe_gallery_asset(&root, reference) else { continue };
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(session) = serde_json::from_slice::<MagicSession>(&bytes) {
                if session.schema == 1 && session.algorithm == "magic-1"
                    && session.palette == palette && session.target_id == target.target_id
                    && session.gallery_root.as_deref() == Some(root.as_path())
                    && !session.previews.is_empty()
                    && session.previews.iter().all(|p| {
                        p.decision.search.as_ref().is_some_and(|s| s.version == magic::optimize::SEARCH_VERSION)
                            && resolve_safe_gallery_asset(&root, &p.preview_path).is_ok_and(|f| f.is_file())
                    }) {
                    let mut job = new_magic_job(&state, Some(id.clone()));
                    job.status = JobStatus::Succeeded;
                    job.stage = "Loaded saved session".into();
                    job.session_path = Some(session.session_path.clone());
                    let public = public_session(&session);
                    job.session = Some(public.clone());
                    state.magic_jobs.lock().insert(job.id.clone(), job.clone());
                    return Ok(Json(serde_json::json!({"session": public, "id": job.id})));
                }
            }
        }
    }
    Ok(Json(serde_json::json!({"session": null})))
}

async fn post_run_magic_render(State(state): State<AppState>, headers: HeaderMap,
    AxumPath(id): AxumPath<String>, Json(req): Json<GalleryMagicRenderRequest>)
    -> Result<(StatusCode, Json<MagicJob>), ApiError> {
    check_same_origin_write(&headers)?;
    let strength = magic_strength(req.strength)?;
    let root = gallery_root(&state, &id)?;
    let source = state.magic_jobs.lock().get(&req.session_id).cloned()
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Magic session not found"))?;
    if source.gallery_run.as_deref() != Some(id.as_str()) || source.status != JobStatus::Succeeded {
        return Err(api_error(StatusCode::CONFLICT, "Magic session is not ready for this gallery"));
    }
    let path = source.session_path.ok_or_else(|| api_error(StatusCode::CONFLICT, "Session missing"))?;
    if !path.starts_with(root.join("magic")) { return Err(api_error(StatusCode::FORBIDDEN, "Session outside gallery")); }
    let job = new_magic_job(&state, Some(id));
    tokio::spawn(start_magic_render(state, job.clone(), path, root, req.variant, strength));
    Ok((StatusCode::ACCEPTED, Json(job)))
}

async fn post_file_magic_render(State(state): State<AppState>, headers: HeaderMap,
    AxumPath(id): AxumPath<String>, Json(req): Json<MagicRenderRequest>)
    -> Result<(StatusCode, Json<MagicJob>), ApiError> {
    check_same_origin_write(&headers)?;
    let strength = magic_strength(req.strength)?;
    let source = state.magic_jobs.lock().get(&id).cloned()
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Magic session not found"))?;
    if source.gallery_run.is_some() || source.status != JobStatus::Succeeded {
        return Err(api_error(StatusCode::CONFLICT, "Standalone Magic session not ready"));
    }
    let path = source.session_path.ok_or_else(|| api_error(StatusCode::CONFLICT, "Session missing"))?;
    let output = path.parent().and_then(Path::parent).and_then(Path::parent).and_then(Path::parent)
        .ok_or_else(|| magic_error("Invalid session path"))?.to_path_buf();
    let job = new_magic_job(&state, None);
    tokio::spawn(start_magic_render(state, job.clone(), path, output, req.variant, strength));
    Ok((StatusCode::ACCEPTED, Json(job)))
}

async fn get_magic_asset(State(state): State<AppState>,
    AxumPath((id, rel_path)): AxumPath<(String, String)>) -> Response {
    let source = state.magic_jobs.lock().get(&id).cloned();
    let Some(job) = source else { return api_error(StatusCode::NOT_FOUND, "Session not found").into_response() };
    if job.gallery_run.is_some() { return api_error(StatusCode::FORBIDDEN, "Use gallery assets").into_response(); }
    let Some(path) = job.session_path else { return api_error(StatusCode::CONFLICT, "Session not ready").into_response() };
    if !public_magic_asset(&rel_path) {
        return api_error(StatusCode::FORBIDDEN, "Asset type not public").into_response();
    }
    let Some(root) = path.parent() else { return api_error(StatusCode::FORBIDDEN, "Invalid session").into_response() };
    let safe = match resolve_safe_gallery_asset(root, &rel_path) {
        Ok(file) => file,
        Err(e) => return api_error(StatusCode::FORBIDDEN, e).into_response(),
    };
    let Ok(bytes) = tokio::fs::read(safe).await else { return api_error(StatusCode::NOT_FOUND, "Asset not found").into_response() };
    let mime = if rel_path.ends_with(".png") { "image/png" } else if rel_path.ends_with(".json") { "application/json" } else { "image/tiff" };
    ([(header::CONTENT_TYPE, mime)], bytes).into_response()
}
