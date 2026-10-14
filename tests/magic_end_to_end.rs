use photonloom::config::LinearImage;
use photonloom::fits::write_fits_linear;
use photonloom::magic::{self, InputDomain, MagicOptions, MagicSource, MagicVariant, TransferFunction};
use sha2::{Digest, Sha256};

#[test]
fn original_science_survives_independent_previews_and_replay() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("science.fits");
    let mut image = LinearImage::new(128, 128, 1);
    for y in 0..128 {
        for x in 0..128 {
            let r2 = ((x as f32 - 63.0).powi(2) + (y as f32 - 63.0).powi(2)) / 400.0;
            image.data[y * 128 + x] = 50.0 + 1000.0 * (-r2).exp() + (x % 7) as f32;
        }
    }
    write_fits_linear(&source, &image, "Ha", "test", 120.0, 1, 0, None).unwrap();
    let initial_hash = Sha256::digest(std::fs::read(&source).unwrap());
    let input = MagicSource::File { path: source.clone(), domain: InputDomain::Linear, transfer: TransferFunction::Auto };
    let mut session = magic::analyze(&input, &MagicOptions::default(), &temp.path().join("out")).unwrap();
    let preview = magic::preview(&mut session, 0.5).unwrap();
    assert_eq!(preview.len(), 3);
    assert!(preview.iter().all(|item| temp.path().join("out").join(&item.preview_path).is_file()));
    assert!(preview.iter().all(|item| item.decision.attempts.last().unwrap().quality.passed));
    let full = magic::render(&mut session, MagicVariant::Enhanced, 0.5).unwrap();
    let report = std::fs::read_to_string(session.session_path.parent().unwrap().join("index.html")).unwrap();
    assert!(report.contains(&format!("href=\"{}\"", std::path::Path::new(&full.tiff_path).file_name().unwrap().to_string_lossy())));
    assert!(report.contains("natural") && report.contains("dramatic"));
    let tiff_path = temp.path().join("out").join(&full.tiff_path);
    let bytes = std::fs::read(&tiff_path).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(&bytes)).unwrap();
    assert_eq!(decoder.dimensions().unwrap(), (128, 128));
    assert!(matches!(decoder.read_image().unwrap(), tiff::decoder::DecodingResult::U16(_)));
    let reopened = magic::open(&session.session_path).unwrap();
    assert_eq!(reopened.source_digest, session.source_digest);
    let replay = magic::render(&mut session, MagicVariant::Enhanced, 0.5).unwrap();
    assert_eq!(bytes, std::fs::read(temp.path().join("out").join(replay.tiff_path)).unwrap());
    assert_eq!(initial_hash.as_slice(), Sha256::digest(std::fs::read(&source).unwrap()).as_slice());
    let alternative = magic::render(&mut session, MagicVariant::Enhanced, 0.8).unwrap();
    assert_ne!(full.id, alternative.id);
    assert!(tiff_path.exists());
}

#[test]
fn strength_boundaries_and_unknown_raster_layout_fail_without_replacing_session() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.png");
    let img = image::ImageBuffer::from_pixel(32, 32, image::Rgb([20u8, 40, 100]));
    img.save(&input).unwrap();
    let source = MagicSource::File { path: input.clone(), domain: InputDomain::Auto, transfer: TransferFunction::Auto };
    let mut session = magic::analyze(&source, &MagicOptions::default(), &temp.path().join("out")).unwrap();
    assert!(magic::preview(&mut session, f32::NAN).is_err());
    let neutral = magic::preview(&mut session, 0.0).unwrap();
    assert!(neutral.iter().all(|candidate| candidate.decision.actual.tone == 0.0));
    std::fs::write(&input, b"changed").unwrap();
    assert!(magic::render(&mut session, MagicVariant::Natural, 0.5).is_err());
    assert!(session.session_path.exists());
}

#[test]
fn explicit_parameters_replay_full_tiff_and_conflicting_cli_controls_fail() {
    use photonloom::magic::planner::{MagicProposal, ProposalScales};
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("signal.fits");
    let mut image = LinearImage::new(96, 96, 1);
    for y in 0..96 { for x in 0..96 {
        let dx = x as f32 - 48.0;
        let dy = y as f32 - 48.0;
        image.data[y * 96 + x] = 40.0 + 500.0 * (-(dx*dx + dy*dy)/360.0).exp() + ((x*13+y*7)%19) as f32;
    }}
    write_fits_linear(&source_path, &image, "Ha", "uncataloged", 60.0, 1, 0, None).unwrap();
    let proposal = MagicProposal { version: 1, recipe_id: "general-natural".into(), variant: MagicVariant::Enhanced,
        strength: 0.5, operator_scales: ProposalScales { tone: 0.8, detail: 0.0, sharpen: 0.0, color: 0.0 } };
    let params_path = temp.path().join("parameters.json");
    std::fs::write(&params_path, proposal.to_json().unwrap()).unwrap();
    let source = MagicSource::File { path: source_path.clone(), domain: InputDomain::Linear, transfer: TransferFunction::Auto };
    let options = MagicOptions { proposal: Some(proposal.clone()), ..Default::default() };
    let mut api = magic::analyze(&source, &options, &temp.path().join("api")).unwrap();
    let preview = magic::preview(&mut api, 0.5).unwrap();
    assert_eq!(preview[1].decision.proposal.as_ref(), Some(&proposal));
    assert!(preview[1].decision.search.as_ref().unwrap().trials.len() <= 21);
    let artifact = magic::render(&mut api, MagicVariant::Enhanced, 0.5).unwrap();
    let api_tiff = std::fs::read(temp.path().join("api").join(&artifact.tiff_path)).unwrap();

    let output = temp.path().join("cli");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_photonloom"))
        .arg("magic").arg("view").arg(&source_path).arg("--output").arg(&output)
        .arg("--input-domain").arg("linear").arg("--magic-parameters").arg(&params_path).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let cli = magic::analyze(&source, &options, &output).unwrap();
    assert_eq!(cli.artifacts.len(), 1);
    assert_eq!(api_tiff, std::fs::read(output.join(&cli.artifacts[0].tiff_path)).unwrap());

    let conflict = std::process::Command::new(env!("CARGO_BIN_EXE_photonloom"))
        .arg("magic").arg("view").arg(&source_path).arg("--output").arg(&output)
        .arg("--magic-parameters").arg(&params_path).arg("--strength").arg("0.8").output().unwrap();
    assert!(!conflict.status.success());
    assert!(String::from_utf8_lossy(&conflict.stderr).contains("cannot be used with"));
    let mut wrong = proposal;
    wrong.recipe_id = "planetary-nebula".into();
    assert!(magic::analyze(&source, &MagicOptions { proposal: Some(wrong), ..Default::default() },
        &temp.path().join("invalid")).is_err());
}
