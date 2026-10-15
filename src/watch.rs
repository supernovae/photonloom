//! Polling watcher for observation folders (including network and mounted filesystems).
use crate::config::ProcessOptions;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

const STATE_FILE: &str = ".photonloom-watch.json";

#[derive(Debug, Clone, Serialize)]
pub struct WatchConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub interval: Duration,
    pub settle: Duration,
    pub keep_runs: usize,
}

pub enum WatchPoll {
    Unchanged,
    Settling { changed_files: usize, quiet_for: Duration },
    Ready,
}

#[derive(Debug, Clone, Serialize)]
pub struct WatchSnapshot {
    pub input: String,
    pub output: String,
    pub state: String,
    pub frame_files: usize,
    pub total_bytes: u64,
    pub changed_files: usize,
    pub quiet_for_s: u64,
    pub last_change_unix_ms: Option<u64>,
    pub next_check_unix_ms: Option<u64>,
    pub interval_s: u64,
    pub settle_s: u64,
    pub keep_runs: usize,
    pub last_run: Option<WatchRunRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchRunRecord {
    pub run_id: Option<String>,
    pub finished_unix_ms: u64,
    pub elapsed_ms: u64,
    pub ok: bool,
    pub error: Option<String>,
    pub frame_files: usize,
}

#[derive(Serialize, Deserialize)]
struct SavedState {
    fingerprint_digest: String,
    last_run: WatchRunRecord,
    #[serde(default)]
    entries: Vec<Entry>,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    path: PathBuf,
    size: u64,
    mtime: Option<Duration>,
}

pub struct FolderWatcher {
    config: WatchConfig,
    observed: Vec<Entry>,
    observed_digest: String,
    ready_digest: Option<String>,
    ready_entries: Option<Vec<Entry>>,
    processed_entries: Vec<Entry>,
    processed_digest: Option<String>,
    changed_at: Option<Instant>,
    last_change_unix_ms: Option<u64>,
    next_check_unix_ms: Option<u64>,
    state: String,
    last_run: Option<WatchRunRecord>,
    total_bytes: u64,
    changed_files: usize,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

/// `YYYY-MM-DD HH:MM:SSZ` (UTC) for unattended logs; avoids a date/time dependency.
fn stamp() -> String {
    let secs = now_ms() / 1000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil-from-days (Howard Hinnant), valid for the Unix era.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn digest(entries: &[Entry]) -> String {
    let mut hasher = Sha256::new();
    for entry in entries {
        let path = entry.path.to_string_lossy();
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update(entry.size.to_le_bytes());
        match entry.mtime {
            Some(time) => {
                hasher.update([1]);
                hasher.update(time.as_secs().to_le_bytes());
                hasher.update(time.subsec_nanos().to_le_bytes());
            }
            None => hasher.update([0]),
        }
    }
    format!("{:x}", hasher.finalize())
}

fn skip_name(name: &std::ffi::OsStr) -> bool {
    let name = name.to_string_lossy();
    name.starts_with('.') || name.ends_with('~') || name.ends_with(".part") || name.ends_with(".tmp")
}

impl FolderWatcher {
    pub fn new(config: WatchConfig) -> Result<Self> {
        if config.keep_runs == 0 {
            bail!("--keep-runs must be at least 1");
        }
        if !config.input.is_dir() {
            bail!("Watch input '{}' must be an existing directory", config.input.display());
        }
        crate::config::validate_input_output_paths(&config.input, &config.output)?;
        let saved: Option<SavedState> = match std::fs::read(config.output.join(STATE_FILE)) {
            Ok(bytes) => Some(serde_json::from_slice(&bytes).context("Parsing watch state")?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("Reading watch state"),
        };
        let processed_entries = saved.as_ref().map(|state| state.entries.clone()).unwrap_or_default();
        Ok(Self {
            config,
            observed: Vec::new(),
            observed_digest: String::new(),
            ready_digest: None,
            ready_entries: None,
            processed_entries,
            processed_digest: saved.as_ref().map(|state| state.fingerprint_digest.clone()),
            changed_at: None,
            last_change_unix_ms: None,
            next_check_unix_ms: None,
            state: "idle".into(),
            last_run: saved.map(|state| state.last_run),
            total_bytes: 0,
            changed_files: 0,
        })
    }

    pub fn poll(&mut self) -> Result<WatchPoll> {
        let mut entries = Vec::new();
        for item in WalkDir::new(&self.config.input).into_iter().filter_entry(|item| {
            item.depth() == 0 || !skip_name(item.file_name())
        }) {
            let item = item.with_context(|| format!("Walking '{}'", self.config.input.display()))?;
            if !item.file_type().is_file() || !crate::scan::is_supported_frame_path(item.path()) {
                continue;
            }
            let metadata = item.metadata()?;
            entries.push(Entry {
                path: item.path().strip_prefix(&self.config.input)?.to_path_buf(),
                size: metadata.len(),
                mtime: metadata.modified().ok().and_then(|time| time.duration_since(UNIX_EPOCH).ok()),
            });
        }
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        let fingerprint = digest(&entries);
        if fingerprint != self.observed_digest {
            // First observation after (re)start: files untouched for longer than `settle`
            // are already quiet, so cron/`--once` runs need not wait out the settle window.
            let newest_age = (self.observed_digest.is_empty())
                .then(|| entries.iter().filter_map(|entry| entry.mtime).max())
                .flatten()
                .and_then(|newest| SystemTime::now().duration_since(UNIX_EPOCH + newest).ok())
                .map_or(Duration::ZERO, |age| age.min(self.config.settle));
            let now = Instant::now();
            self.changed_at = Some(now.checked_sub(newest_age).unwrap_or(now));
            self.last_change_unix_ms = Some(now_ms());
            self.observed_digest = fingerprint;
            self.ready_digest = None;
            self.ready_entries = None;
        }
        self.total_bytes = entries.iter().map(|entry| entry.size).sum();
        self.observed = entries;
        self.next_check_unix_ms = Some(now_ms().saturating_add(self.config.interval.as_millis() as u64));
        if self.observed.is_empty() || self.processed_digest.as_deref() == Some(&self.observed_digest) {
            self.changed_files = 0;
            self.state = "idle".into();
            return Ok(WatchPoll::Unchanged);
        }
        let quiet_for = self.changed_at.map_or(Duration::ZERO, |instant| instant.elapsed());
        let old: std::collections::BTreeMap<_, _> = self.processed_entries.iter().map(|entry| (&entry.path, entry)).collect();
        let new: std::collections::BTreeMap<_, _> = self.observed.iter().map(|entry| (&entry.path, entry)).collect();
        self.changed_files = self.observed.iter().filter(|entry| old.get(&entry.path) != Some(entry)).count()
            + self.processed_entries.iter().filter(|entry| !new.contains_key(&entry.path)).count();
        if quiet_for < self.config.settle {
            self.state = "settling".into();
            return Ok(WatchPoll::Settling { changed_files: self.changed_files, quiet_for });
        }
        self.ready_digest = Some(self.observed_digest.clone());
        self.state = "queued".into();
        Ok(WatchPoll::Ready)
    }

    pub fn begin(&mut self) {
        self.ready_digest.get_or_insert_with(|| self.observed_digest.clone());
        self.ready_entries = Some(self.observed.clone());
        self.state = "processing".into();
    }

    pub fn finish(&mut self, record: WatchRunRecord) -> Result<()> {
        let saved = SavedState {
            fingerprint_digest: self.ready_digest.take().context("No ready snapshot to finish")?,
            entries: self.ready_entries.take().context("No ready entries to finish")?,
            last_run: record.clone(),
        };
        crate::atomic_write_file(&self.config.output.join(STATE_FILE), &serde_json::to_vec_pretty(&saved)?)?;
        self.processed_digest = Some(saved.fingerprint_digest);
        self.processed_entries = saved.entries;
        self.changed_files = 0;
        self.last_run = Some(record);
        self.state = "idle".into();
        if let Err(error) = prune_runs(&self.config.output, self.config.keep_runs) {
            eprintln!("warning: pruning old runs in '{}' failed: {error:#}", self.config.output.display());
        }
        Ok(())
    }

    pub fn snapshot(&self) -> WatchSnapshot {
        WatchSnapshot {
            input: self.config.input.to_string_lossy().into_owned(),
            output: self.config.output.to_string_lossy().into_owned(),
            state: self.state.clone(),
            frame_files: self.observed.len(),
            total_bytes: self.total_bytes,
            changed_files: self.changed_files,
            quiet_for_s: if self.state == "settling" {
                self.changed_at.map_or(0, |instant| instant.elapsed().as_secs())
            } else { 0 },
            last_change_unix_ms: self.last_change_unix_ms,
            next_check_unix_ms: self.next_check_unix_ms,
            interval_s: self.config.interval.as_secs(),
            settle_s: self.config.settle.as_secs(),
            keep_runs: self.config.keep_runs,
            last_run: self.last_run.clone(),
        }
    }

    pub fn set_state(&mut self, state: &str) {
        self.state = state.to_owned();
    }
}

pub fn prune_runs(output: &Path, keep: usize) -> Result<Vec<PathBuf>> {
    if keep == 0 {
        bail!("keep_runs must be at least 1");
    }
    let runs = output.join("runs");
    if !runs.exists() {
        return Ok(Vec::new());
    }
    let protected = match std::fs::read(output.join("manifest.json")) {
        Ok(bytes) => {
            let manifest: serde_json::Value = serde_json::from_slice(&bytes).context("Parsing root manifest before pruning")?;
            let relative = manifest.get("run_dir").and_then(|value| value.as_str())
                .context("Root manifest has no run_dir; refusing to prune")?;
            let mut parts = Path::new(relative).components();
            match (parts.next(), parts.next(), parts.next()) {
                (Some(Component::Normal(root)), Some(Component::Normal(id)), None) if root == "runs" => Some(id.to_os_string()),
                _ => bail!("Root manifest has invalid run_dir; refusing to prune"),
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("Reading root manifest before pruning"),
    };
    let mut dirs = std::fs::read_dir(&runs)?.collect::<Result<Vec<_>, _>>()?;
    dirs.retain(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()));
    dirs.sort_by_key(|entry| std::cmp::Reverse(entry.file_name()));
    let reserved = usize::from(protected.is_some());
    let mut retained_other = 0;
    let mut removed = Vec::new();
    for entry in dirs {
        let path = entry.path();
        if protected.as_deref() == Some(entry.file_name().as_os_str()) {
            continue;
        }
        if retained_other < keep - reserved {
            retained_other += 1;
        } else {
            std::fs::remove_dir_all(&path).with_context(|| format!("Pruning '{}'", path.display()))?;
            removed.push(path);
        }
    }
    Ok(removed)
}

pub fn run(config: WatchConfig, options: ProcessOptions, once: bool) -> Result<()> {
    let mut watcher = FolderWatcher::new(config.clone())?;
    let mut first_poll = true;
    loop {
        match watcher.poll()? {
            WatchPoll::Unchanged => {
                if once {
                    println!("[{}] unchanged: no new frame snapshot since the last processed run", stamp());
                    return Ok(());
                }
                if first_poll {
                    let frames = watcher.snapshot().frame_files;
                    let what = if frames == 0 { "no frames yet".to_string() } else { format!("{frames} frames already processed") };
                    println!("[{}] idle: {what}; watching {} every {}s",
                        stamp(), config.input.display(), config.interval.as_secs());
                }
            }
            WatchPoll::Settling { changed_files, quiet_for } => {
                println!("[{}] {changed_files} files changed, waiting for {}s of quiet ({}s so far)",
                    stamp(), config.settle.as_secs(), quiet_for.as_secs());
            }
            WatchPoll::Ready => {
                watcher.begin();
                let frames = watcher.snapshot().frame_files;
                println!("[{}] processing {frames} frames from {}", stamp(), config.input.display());
                let started = Instant::now();
                let result = crate::process(&config.input, &config.output, &options);
                let record = WatchRunRecord {
                    run_id: result.as_ref().ok().map(|manifest| manifest.run_id.clone()),
                    finished_unix_ms: now_ms(),
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    ok: result.is_ok(),
                    error: result.as_ref().err().map(|error| format!("{error:#}")),
                    frame_files: frames,
                };
                watcher.finish(record)?;
                match result {
                    Ok(manifest) => println!("[{}] complete in {:.1}s; run {}; gallery: {}/index.html",
                        stamp(), started.elapsed().as_secs_f64(), manifest.run_id, config.output.display()),
                    Err(error) => eprintln!("[{}] processing failed: {error:#} (waiting for a changed snapshot)", stamp()),
                }
                if once { return Ok(()); }
            }
        }
        first_poll = false;
        std::thread::sleep(config.interval);
    }
}
