//! Explicitly opted-in, local preferences; no source metadata is stored here.
use super::{MagicVariant, recipes};
use anyhow::{Context, Result, ensure};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u32 = 1;
const FILE_NAME: &str = "magic-preferences.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preference {
    pub class: String,
    pub recipe_id: String,
    pub variant: MagicVariant,
    pub strength: f32,
}

impl Preference {
    pub fn validate(&self) -> Result<()> {
        ensure!(matches!(self.class.as_str(),
            "emission-nebula" | "reflection-nebula" | "dark-nebula" |
            "spiral-galaxy" | "other-galaxy" | "globular-cluster" |
            "open-cluster" | "planetary-nebula" | "star-field" | "unknown"),
            "Unknown preference subject class");
        ensure!(self.strength.is_finite() && (0.0..=1.0).contains(&self.strength),
            "Preference strength must be finite and in [0, 1]");
        let recipes = recipes::all();
        let recipe = recipes.iter().find(|recipe| recipe.id == self.recipe_id)
            .context("Unknown preference recipe ID")?;
        ensure!(recipe.id == "general-natural" || recipe.classes.contains(&self.class),
            "Preference recipe does not match subject class");
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredHistory {
    version: u32,
    selections: Vec<Preference>,
}

fn read_history(path: &Path) -> Result<StoredHistory> {
    let contents = std::fs::read(path).context("Could not read Magic preferences")?;
    ensure!(contents.len() <= 16 * 1024, "Magic preferences exceed size limit");
    let history: StoredHistory = serde_json::from_slice(&contents).context("Invalid Magic preferences JSON")?;
    ensure!(history.version == SCHEMA_VERSION, "Unsupported Magic preferences schema");
    ensure!(!history.selections.is_empty() && history.selections.len() <= 64, "Invalid Magic preference history length");
    for preference in &history.selections { preference.validate()?; }
    Ok(history)
}

pub fn default_path() -> Result<PathBuf> {
    let project = ProjectDirs::from("io", "photonloom", "photonloom")
        .context("Failed to determine OS per-user application data directory")?;
    Ok(project.data_local_dir().join(FILE_NAME))
}

/// An opted-out caller never opens the preferences file. Corrupt or absent data is
/// reported as a warning, never silently treated as a learned prior.
pub fn load(path: &Path, enabled: bool) -> (Option<Preference>, Option<String>) {
    if !enabled { return (None, None); }
    match read_history(path) {
        Ok(history) => (history.selections.last().cloned(), None),
        Err(error) => (None, Some(format!("Magic preferences unavailable: {error:#}"))),
    }
}

/// The newest matching class/recipe is eligible; unrelated prior subjects cannot
/// bias a different image. Disabled calls never open the local store.
pub fn load_matching(path: &Path, enabled: bool, class: &str, recipe_id: &str)
    -> (Option<Preference>, Option<String>) {
    if !enabled { return (None, None); }
    match read_history(path) {
        Ok(history) => (history.selections.into_iter().rev()
            .find(|p| p.class == class && p.recipe_id == recipe_id), None),
        Err(error) => (None, Some(format!("Magic preferences unavailable: {error:#}"))),
    }
}

/// Append an explicit full-render selection, retaining the last 64 choices.
pub fn save(path: &Path, preference: &Preference) -> Result<()> {
    preference.validate()?;
    let mut selections = if path.exists() { read_history(path)?.selections } else { Vec::new() };
    selections.push(preference.clone());
    if selections.len() > 64 { selections.drain(..selections.len() - 64); }
    let bytes = serde_json::to_vec_pretty(&StoredHistory { version: SCHEMA_VERSION, selections })?;
    crate::atomic_write_file(path, &bytes)
}

/// Delete only the app-owned preference file, never its parent directory.
pub fn reset(path: &Path) -> Result<()> {
    ensure!(path.file_name().is_some_and(|name| name == FILE_NAME),
        "Refusing to reset a file other than magic-preferences.json");
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("Failed to reset Magic preferences at '{}'", path.display())),
    }
}
