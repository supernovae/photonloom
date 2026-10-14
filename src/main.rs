use anyhow::{Context, Result};
use clap::ValueEnum;
use clap::{Parser, Subcommand};
use photonloom::config::{AcceleratorMode, ProcessOptions, default_output_for_input};
use photonloom::finish_style::{FinishOptions, FinishPreset};
use photonloom::magic::{self, InputDomain, MagicOptions, MagicSource, MagicVariant, TransferFunction};
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
struct FinishArgs {
    /// Display recipe; linear science FITS are never modified
    #[arg(long, value_enum, default_value = "balanced")]
    finish_preset: FinishPreset,
    #[arg(long)]
    exposure: Option<f32>,
    #[arg(long)]
    contrast: Option<f32>,
    #[arg(long)]
    saturation: Option<f32>,
    #[arg(long)]
    gamma: Option<f32>,
    #[arg(long)]
    local_contrast: Option<f32>,
    /// Keep the full display canvas rather than trimming alignment borders
    #[arg(long)]
    keep_edges: bool,
}

impl FinishArgs {
    fn options(&self) -> FinishOptions {
        FinishOptions {
            preset: self.finish_preset,
            exposure: self.exposure,
            contrast: self.contrast,
            saturation: self.saturation,
            gamma: self.gamma,
            local_contrast: self.local_contrast,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "photonloom",
    version,
    about = "Photonloom — From photons to masterpieces: astrophotography calibration, integration, and gallery engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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

        /// Compute accelerator selection (auto, cpu, gpu)
        #[arg(long, value_enum, default_value_t = AcceleratorMode::Auto)]
        accelerator: AcceleratorMode,

        /// Initial selected palette card in the gallery (e.g. natural, sho, hoo, osh, hubble, jwst-inspired, hubble-inspired)
        #[arg(long)]
        palette: Option<String>,

        /// Custom RGB channel mix expression (e.g. 'Ha:1,Ha:0.7+OIII:0.3,OIII:1')
        #[arg(long)]
        mix: Option<String>,

        /// Apply opt-in local Cosmic Clarity star-shape & denoise restoration on finished RGB
        #[arg(long, default_value_t = false)]
        restore: bool,
        /// Generate measured Magic previews and a safely enhanced full image per target
        #[arg(long, default_value_t = false)]
        magic_view: bool,
        #[command(flatten)]
        finish: FinishArgs,
    },

    /// Start local loopback web gallery and processing UI
    Serve {
        /// Loopback address and port to bind (must be 127.0.0.1 / ::1 / localhost)
        #[arg(long, default_value = "127.0.0.1:8787")]
        bind: String,

        /// Optional local llama-server URL on loopback (e.g. http://127.0.0.1:8080)
        #[arg(long)]
        local_llm: Option<String>,

        /// Load a completed gallery into the web UI on startup (without processing its inputs)
        #[arg(long, env = "PHOTONLOOM_SAMPLE_GALLERY")]
        sample_gallery: Option<PathBuf>,
    },

    /// Apply a display recipe to stored integrations without repeating stacking
    Finish {
        gallery: PathBuf,
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
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        target: Option<String>,
        #[arg(long)]
        palette: Option<String>,
        #[arg(long, value_enum, default_value_t = InputDomain::Auto)]
        input_domain: InputDomain,
        #[arg(long, value_enum, default_value_t = TransferFunction::Auto)]
        transfer: TransferFunction,
        #[arg(long, value_enum, conflicts_with = "magic_parameters")]
        variant: Option<MagicVariant>,
        #[arg(long, conflicts_with = "magic_parameters")]
        strength: Option<f32>,
        /// Opt in to a bounded HTTPS SIMBAD position query (only with valid WCS)
        #[arg(long, default_value_t = false)]
        catalog_online: bool,
        /// Optional loopback-only local llama-server for bounded recipe proposals
        #[arg(long, conflicts_with = "magic_parameters")]
        magic_llm: Option<String>,
        /// Enable private, local preference history after explicitly saved selections
        #[arg(long, default_value_t = false, conflicts_with = "magic_parameters")]
        magic_learn: bool,
        /// Replay a versioned validated MagicProposal JSON without other tuning flags
        #[arg(long)]
        magic_parameters: Option<PathBuf>,
    },
    Render {
        session_json: PathBuf,
        #[arg(long, value_enum)]
        variant: MagicVariant,
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
        Commands::Run {
            input,
            output,
            accelerator,
            palette,
            mix,
            restore,
            finish,
            magic_view,
        } => {
            let output_dir = match output {
                Some(out) => out,
                None => default_output_for_input(&input)?,
            };
            let options = ProcessOptions {
                accelerator,
                palette,
                mix,
                restore,
                magic_view,
                finish: finish.options(),
                trim_edges: !finish.keep_edges,
                progress: Some(std::sync::Arc::new(|stage, pct| {
                    eprintln!("[{:>3.0}%] {}", pct * 100.0, stage);
                })),
            };
            let manifest = photonloom::process(&input, &output_dir, &options)?;
            println!(
                "Gallery ready at {} (run {}, accelerator: {})",
                manifest.canonical_output, manifest.run_id, manifest.accelerator_used
            );
            for w in &manifest.warnings {
                eprintln!("warning: {w}");
            }
            Ok(())
        }
        Commands::Magic { action } => match action {
            MagicAction::View { input, output, target, palette, input_domain, transfer, variant, strength,
                catalog_online, magic_llm, magic_learn, magic_parameters } => {
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
                    planner_variant: requested_variant.unwrap_or(MagicVariant::Enhanced), ..Default::default() }, &output)?;
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
        Commands::Finish {
            gallery,
            palette,
            finish,
        } => {
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
        Commands::Serve {
            bind,
            local_llm,
            sample_gallery,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(photonloom::web::serve(&bind, local_llm, sample_gallery))
        }
        Commands::Edit {
            gallery,
            request,
            local_llm,
        } => {
            let variant =
                photonloom::models::apply_gallery_edit(&gallery, &request, local_llm.as_deref())?;
            println!(
                "Created edited variant '{}' ({}) -> {}",
                variant.id, variant.title, variant.png_path
            );
            Ok(())
        }
        Commands::Models { action } => match action {
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
