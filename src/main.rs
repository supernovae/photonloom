use anyhow::{Context, Result};
use clap::ValueEnum;
use clap::{Parser, Subcommand};
use photonloom::config::{AcceleratorMode, ProcessOptions, default_magic_output, default_output_for_input};
use photonloom::finish_style::{FinishOptions, FinishPreset, StarMode};
use photonloom::magic::{self, InputDomain, MagicOptions, MagicSource, MagicVariant, TransferFunction};
use photonloom::watch::WatchConfig;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(clap::Args, Debug)]
struct FinishArgs {
    /// Display recipe (balanced, natural, vivid, faint-nebula, or detail); linear science FITS stay unchanged
    #[arg(long, value_enum, default_value = "balanced")]
    finish_preset: FinishPreset,
    /// Exposure adjustment in -4..4 stops
    #[arg(long)]
    exposure: Option<f32>,
    /// Contrast multiplier in 0.5..1.8
    #[arg(long)]
    contrast: Option<f32>,
    /// Saturation multiplier in 0..1.8
    #[arg(long)]
    saturation: Option<f32>,
    /// Gamma in 0.5..2
    #[arg(long)]
    gamma: Option<f32>,
    /// Local contrast in 0..0.3
    #[arg(long)]
    local_contrast: Option<f32>,
    /// Keep the full display canvas rather than trimming alignment borders (default: trim)
    #[arg(long)]
    keep_edges: bool,
    /// Remove stars using GPU-offloaded multiscale harmonic star inpainting (default: off)
    #[arg(long, default_value_t = false)]
    starless: bool,
    /// Star separation mode (standard, starless, reduced, stars-only; default: standard)
    #[arg(long, value_enum)]
    star_mode: Option<StarMode>,
    /// Star screen-reblend weight in 0..1 (0 starless, 0.45 reduced, 1 full stars)
    #[arg(long)]
    star_blend: Option<f32>,
}

impl FinishArgs {
    fn options(&self) -> FinishOptions {
        let star_mode = if self.starless {
            StarMode::Starless
        } else {
            self.star_mode.unwrap_or(StarMode::Standard)
        };
        FinishOptions {
            preset: self.finish_preset,
            exposure: self.exposure,
            contrast: self.contrast,
            saturation: self.saturation,
            gamma: self.gamma,
            local_contrast: self.local_contrast,
            star_mode,
            star_blend: self.star_blend,
        }
    }
}

#[derive(clap::Args, Debug)]
struct ProcessingArgs {
    /// Compute accelerator: auto (default), cpu, or gpu
    #[arg(long, value_enum, default_value_t = AcceleratorMode::Auto)]
    accelerator: AcceleratorMode,
    /// Initial selected palette (e.g. natural, sho, hoo, osh, hubble)
    #[arg(long)]
    palette: Option<String>,
    /// Custom RGB mix (e.g. 'Ha:1,Ha:0.7+OIII:0.3,OIII:1')
    #[arg(long)]
    mix: Option<String>,
    /// Apply local Cosmic Clarity restoration (default: off)
    #[arg(long)]
    restore: bool,
    #[command(flatten)]
    finish: FinishArgs,
}

impl ProcessingArgs {
    fn options(self, magic_view: bool, progress: bool) -> ProcessOptions {
        ProcessOptions {
            accelerator: self.accelerator,
            palette: self.palette,
            mix: self.mix,
            restore: self.restore,
            magic_view,
            finish: self.finish.options(),
            trim_edges: !self.finish.keep_edges,
            progress: progress.then(cli_progress),
        }
    }
}

/// Terminal activity printer: `[ 42% 03:15] stage` plus indented detail lines.
fn cli_progress() -> photonloom::config::ProgressCallback {
    let started = Instant::now();
    Arc::new(move |event: photonloom::config::ProgressEvent<'_>| {
        let elapsed = started.elapsed().as_secs();
        match event.fraction {
            Some(pct) => eprintln!("[{:>3.0}% {:02}:{:02}] {}", pct * 100.0, elapsed / 60, elapsed % 60, event.message),
            None => eprintln!("           · {}", event.message),
        }
    })
}

#[derive(clap::Args, Debug)]
struct WatchTuning {
    /// Poll interval in seconds (default: 30; network mounts supported)
    #[arg(long, default_value_t = 30)]
    interval: u64,
    /// Required quiet period in seconds before processing (default: 120)
    #[arg(long, default_value_t = 120)]
    settle: u64,
    /// Retain newest gallery runs, always preserving the active run (minimum: 1; default: 3)
    #[arg(long, default_value_t = 3)]
    keep_runs: usize,
}

impl WatchTuning {
    fn config(self, input: PathBuf, output: Option<PathBuf>) -> Result<WatchConfig> {
        let output = output.map(Ok).unwrap_or_else(|| default_output_for_input(&input))?;
        Ok(WatchConfig {
            input,
            output,
            interval: Duration::from_secs(self.interval),
            settle: Duration::from_secs(self.settle),
            keep_runs: self.keep_runs,
        })
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "photonloom",
    version,
    about = "Photonloom — From photons to masterpieces: astrophotography calibration, integration, and gallery engine",
    after_help = "Three ways to begin:\n  Studio:       photonloom                       (opens the browser)\n  One-shot:     photonloom inspect INPUT && photonloom run INPUT\n  Unattended:   photonloom watch INPUT            (or serve --watch INPUT)"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Calibrate, register, integrate, and render a gallery from an observation folder
    Run {
        /// Input directory containing light and calibration frames
        input: PathBuf,

        /// Output gallery directory (defaults to sibling <INPUT basename>-photonloom)
        #[arg(long)]
        output: Option<PathBuf>,

        /// Generate measured Magic previews and a safely enhanced full image (default: off)
        #[arg(long)]
        magic_view: bool,
        #[command(flatten)]
        processing: ProcessingArgs,
    },

    /// Start local loopback web gallery and processing studio
    Serve {
        /// Loopback address and port (default: 127.0.0.1:8787; explicit binds disable fallback)
        #[arg(long)]
        bind: Option<String>,
        /// Open the studio in your browser (default: off for explicit serve)
        #[arg(long)]
        open: bool,
        /// Optional local llama-server URL on loopback (default: none)
        #[arg(long)]
        local_llm: Option<String>,
        /// Load an existing gallery (env PHOTONLOOM_GALLERY; default: none)
        #[arg(long, env = "PHOTONLOOM_GALLERY")]
        gallery: Option<PathBuf>,
        /// Watch INPUT for new frames while serving (default: disabled)
        #[arg(long)]
        watch: Option<PathBuf>,
        /// Gallery output for --watch (default: sibling INPUT-photonloom)
        #[arg(long)]
        output: Option<PathBuf>,
        #[command(flatten)]
        tuning: WatchTuning,
        /// Disable automatic Magic View for watched runs (default: enabled)
        #[arg(long)]
        no_magic: bool,
        #[command(flatten)]
        processing: ProcessingArgs,
    },

    /// Examine frame metadata and suggested target plan without processing pixels
    Inspect {
        /// Folder containing light and calibration frames
        input: PathBuf,
        /// Print the observation plan as JSON instead of a human summary
        #[arg(long)]
        json: bool,
    },

    /// Poll an observation folder and process each settled snapshot once
    Watch {
        /// Input directory containing light and calibration frames
        input: PathBuf,
        /// Gallery output (default: sibling INPUT-photonloom)
        #[arg(long)]
        output: Option<PathBuf>,
        #[command(flatten)]
        tuning: WatchTuning,
        /// Process one changed snapshot then exit; unchanged snapshots exit immediately
        #[arg(long)]
        once: bool,
        /// Disable automatic Magic View (default: enabled)
        #[arg(long)]
        no_magic: bool,
        #[command(flatten)]
        processing: ProcessingArgs,
    },

    /// Apply a display recipe to stored integrations without repeating stacking
    Finish {
        /// Gallery directory created by run/watch
        gallery: PathBuf,
        /// Render palette selection (default: gallery's selected palette)
        #[arg(long)]
        palette: Option<String>,
        #[command(flatten)]
        finish: FinishArgs,
    },

    /// Rerender a gallery variant using deterministic grammar or an optional local LLM
    Edit {
        /// Path to an existing Photonloom gallery output directory
        gallery: PathBuf,

        /// Natural-language edit request (e.g. "make the HOO image warmer")
        request: String,

        /// Optional local llama-server URL on loopback (e.g. http://127.0.0.1:8080)
        #[arg(long)]
        local_llm: Option<String>,
    },

    /// Adaptive measured finishing from an original gallery or an image file
    Magic {
        #[command(subcommand)]
        action: MagicAction,
    },

    /// Manage optional local neural restoration models
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
}

#[derive(Subcommand, Debug)]
enum MagicAction {
    View {
        /// Gallery directory or standalone FITS/image file
        input: PathBuf,
        /// Output gallery directory (default: input directory, or sibling STEM-magic for a file)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Gallery target ID (required if the gallery has multiple targets)
        #[arg(long)]
        target: Option<String>,
        /// Gallery palette (default: selected palette)
        #[arg(long)]
        palette: Option<String>,
        /// Input domain for standalone files (default: auto)
        #[arg(long, value_enum, default_value_t = InputDomain::Auto)]
        input_domain: InputDomain,
        /// Transfer function for standalone files (default: auto)
        #[arg(long, value_enum, default_value_t = TransferFunction::Auto)]
        transfer: TransferFunction,
        /// Full render variant (default: previews only)
        #[arg(long, value_enum, conflicts_with = "magic_parameters")]
        variant: Option<MagicVariant>,
        /// Magic enhancement strength in 0..1 (default: 0.5)
        #[arg(long, conflicts_with = "magic_parameters")]
        strength: Option<f32>,
        /// Opt in to a bounded HTTPS SIMBAD position query (only with valid WCS; default: off)
        #[arg(long, default_value_t = false)]
        catalog_online: bool,
        /// Optional loopback-only local llama-server for bounded recipe proposals (default: none)
        #[arg(long, conflicts_with = "magic_parameters")]
        magic_llm: Option<String>,
        /// Enable private local preference history after saved selections (default: off)
        #[arg(long, default_value_t = false, conflicts_with = "magic_parameters")]
        magic_learn: bool,
        /// Replay validated MagicProposal JSON instead of other tuning flags (default: none)
        #[arg(long)]
        magic_parameters: Option<PathBuf>,
        /// Produce starless Magic View candidates via multiscale inpainting (default: off)
        #[arg(long, default_value_t = false)]
        starless: bool,
        /// Star screen-reblend weight in 0..1 for Magic View (default: automatic)
        #[arg(long)]
        star_blend: Option<f32>,
    },
    Render {
        /// Session JSON written by magic view
        session_json: PathBuf,
        /// Full-image Magic render variant
        #[arg(long, value_enum)]
        variant: MagicVariant,
        /// Magic enhancement strength in 0..1 (default: 0.5)
        #[arg(long, default_value_t = 0.5)]
        strength: f32,
    },
    /// Delete only Photonloom's local Magic preference history
    ResetPreferences,
}

#[derive(Subcommand, Debug)]
enum ModelsAction {
    /// Download and SHA-256 verify pinned MIT Cosmic Clarity weights into user app data
    Install,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => serve(photonloom::web::ServeConfig {
            bind: "127.0.0.1:8787".into(),
            bind_fallback: true,
            local_llm: None,
            gallery: None,
            open_browser: true,
            watch: None,
        }),
        Some(Commands::Run { input, output, processing, magic_view }) => {
            let output_dir = output.map(Ok).unwrap_or_else(|| default_output_for_input(&input))?;
            let options = processing.options(magic_view, true);
            let started = Instant::now();
            let manifest = photonloom::process(&input, &output_dir, &options)?;
            println!("Gallery ready at {}/index.html (run {}, accelerator: {}, elapsed {:.1}s)",
                manifest.canonical_output, manifest.run_id, manifest.accelerator_used,
                started.elapsed().as_secs_f64());
            println!("Next: photonloom serve --gallery {} --open", output_dir.display());
            println!("      photonloom finish {} --finish-preset balanced", output_dir.display());
            for warning in &manifest.warnings { eprintln!("warning: {warning}"); }
            Ok(())
        }
        Some(Commands::Inspect { input, json }) => {
            let plan = photonloom::plan::inspect(&input)?;
            if json { println!("{}", serde_json::to_string_pretty(&plan)?); }
            else { println!("{plan}"); }
            Ok(())
        }
        Some(Commands::Watch { input, output, tuning, once, no_magic, processing }) => {
            photonloom::watch::run(tuning.config(input, output)?, processing.options(!no_magic, true), once)
        }
        Some(Commands::Serve { bind, open, local_llm, gallery, watch, output, tuning, no_magic, processing }) => {
            if watch.is_none() && output.is_some() {
                anyhow::bail!("--output requires --watch");
            }
            let bind_fallback = bind.is_none();
            let watch = watch.map(|input| -> Result<_> {
                Ok((tuning.config(input, output)?, processing.options(!no_magic, false)))
            }).transpose()?;
            serve(photonloom::web::ServeConfig {
                bind: bind.unwrap_or_else(|| "127.0.0.1:8787".into()),
                bind_fallback,
                local_llm,
                gallery,
                open_browser: open,
                watch,
            })
        }
        Some(Commands::Magic { action }) => match action {
            MagicAction::View { input, output, target, palette, input_domain, transfer, variant, strength,
                catalog_online, magic_llm, magic_learn, magic_parameters, starless, star_blend } => {
                let output = output.map(Ok).unwrap_or_else(|| default_magic_output(&input))?;
                let proposal = magic_parameters.as_deref().map(magic::planner::load_parameters).transpose()?;
                let requested_strength = proposal.as_ref().map(|p| p.strength).or(strength).unwrap_or(0.5);
                let requested_variant = proposal.as_ref().map(|p| p.variant).or(variant);
                let source = if input.is_dir() {
                    let manifest: photonloom::GalleryManifest = serde_json::from_slice(
                        &std::fs::read(input.join("manifest.json")).context("Reading gallery manifest")?
                    ).context("Parsing gallery manifest")?;
                    let target_id = match target {
                        Some(target_id) => target_id,
                        None if manifest.targets.len() == 1 => manifest.targets[0].target_id.clone(),
                        None => anyhow::bail!("--target is required for a gallery with multiple targets"),
                    };
                    if input_domain != InputDomain::Auto || transfer != TransferFunction::Auto {
                        anyhow::bail!("--input-domain and --transfer apply only to standalone files");
                    }
                    MagicSource::Gallery { gallery: input, target_id, palette: palette.unwrap_or_default() }
                } else {
                    if target.is_some() || palette.is_some() {
                        anyhow::bail!("--target and --palette require an original gallery");
                    }
                    MagicSource::File { path: input, domain: input_domain, transfer }
                };
                let mut session = magic::analyze(&source, &MagicOptions { catalog_online, magic_llm,
                    magic_learn, proposal, planner_strength: requested_strength,
                    planner_variant: requested_variant.unwrap_or(MagicVariant::Enhanced),
                    starless, star_blend, progress: photonloom::config::ProgressSink(Some(cli_progress())),
                    ..Default::default() }, &output)?;
                let selected_strength = session.proposal.as_ref().map_or(requested_strength, |p| p.strength);
                let selected_variant = session.proposal.as_ref().map(|p| p.variant).or(requested_variant);
                if selected_strength != 0.0 { magic::preview(&mut session, 0.0)?; }
                let candidates = magic::preview(&mut session, selected_strength)?;
                if let Some(warning) = &session.planner_warning { eprintln!("warning: {warning}"); }
                for item in &candidates { println!("{} preview: {} ({})", item.variant.as_str(), item.preview_path, item.decision.explanation); }
                if let Some(variant) = selected_variant {
                    let artifact = magic::render(&mut session, variant, selected_strength)?;
                    println!("Full Magic image: {} (TIFF {}), session {}", artifact.png_path, artifact.tiff_path, session.session_path.display());
                } else {
                    println!("Session: {}", session.session_path.display());
                }
                Ok(())
            }
            MagicAction::Render { session_json, variant, strength } => {
                let mut session = magic::open(&session_json)?;
                session.progress = photonloom::config::ProgressSink(Some(cli_progress()));
                let artifact = magic::render(&mut session, variant, strength)?;
                println!("Full Magic image: {} (TIFF {})", artifact.png_path, artifact.tiff_path);
                Ok(())
            }
            MagicAction::ResetPreferences => {
                let path = magic::preferences::default_path()?;
                magic::preferences::reset(&path)?;
                println!("Magic preference history reset");
                Ok(())
            }
        },
        Some(Commands::Finish {
            gallery,
            palette,
            finish,
        }) => {
            let mut manifest: photonloom::GalleryManifest = serde_json::from_slice(
                &std::fs::read(gallery.join("manifest.json"))
                    .context("Reading gallery manifest")?,
            )
            .context("Parsing gallery manifest")?;
            let options = finish.options();
            let name = finish
                .finish_preset
                .to_possible_value()
                .context("Unknown finish preset")?;
            let variant = photonloom::finish::rerender_variant_from_stored_fits(
                &gallery,
                &mut manifest,
                palette.as_deref(),
                None,
                Some(&options),
                Some(!finish.keep_edges),
                None,
                Some(name.get_name()),
            )?;
            println!(
                "Finished variant '{}' -> {}",
                variant.id,
                gallery.join(&variant.png_path).display()
            );
            Ok(())
        }
        Some(Commands::Edit {
            gallery,
            request,
            local_llm,
        }) => {
            let variant =
                photonloom::models::apply_gallery_edit(&gallery, &request, local_llm.as_deref())?;
            println!(
                "Created edited variant '{}' ({}) -> {}",
                variant.id, variant.title, variant.png_path
            );
            Ok(())
        }
        Some(Commands::Models { action }) => match action {
            ModelsAction::Install => {
                let paths = photonloom::models::install_models()?;
                println!(
                    "Installed verified Cosmic Clarity models to {}",
                    paths.dir.display()
                );
                Ok(())
            }
        },
    }
}

fn serve(config: photonloom::web::ServeConfig) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(photonloom::web::serve(config))
}
