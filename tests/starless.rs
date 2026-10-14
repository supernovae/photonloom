#[path = "../examples/make_fixture.rs"]
#[allow(dead_code)]
mod make_fixture;

use photonloom::config::{AcceleratorMode, LinearImage, ProcessOptions};
use photonloom::finish_style::{FinishOptions, StarMode};
use photonloom::magic::{self, InputDomain, MagicOptions, MagicSource, MagicVariant, TransferFunction};
use photonloom::models::{apply_gallery_edit, parse_deterministic_edit};
use photonloom::stars::{screen_blend_stars, separate_stars};
use tempfile::TempDir;

#[test]
fn cpu_and_gpu_star_separation_agree_and_unscreen_roundtrips() {
    let (w, h) = (64usize, 64usize);
    let raw_plane = make_fixture::synthesize_light_plane(w, h, 0.0, 0.0, 1.0, false, false, 42);
    let mut img = LinearImage::new(w, h, 3);
    for p in 0..w * h {
        let v = ((raw_plane[p] - 50.0) / 4800.0).clamp(0.0, 1.0);
        img.data[p] = v;
        img.data[w * h + p] = (v * 0.92).clamp(0.0, 1.0);
        img.data[2 * w * h + p] = (v * 0.85).clamp(0.0, 1.0);
    }

    let cpu_sep = separate_stars(&img, AcceleratorMode::Cpu).expect("CPU star separation");
    assert!(
        cpu_sep.detected_star_count >= 12,
        "Expected >= 12 detected stars, got {}",
        cpu_sep.detected_star_count
    );

    let auto_sep = separate_stars(&img, AcceleratorMode::Auto).expect("Auto GPU/CPU star separation");
    assert_eq!(cpu_sep.detected_star_count, auto_sep.detected_star_count);
    let max_diff = cpu_sep
        .starless
        .data
        .iter()
        .zip(&auto_sep.starless.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= 1e-4,
        "CPU vs GPU starless max difference {max_diff} exceeds 1e-4"
    );

    // Verify Unscreen + Screen-blend at weight 1.0 reconstructs the original image
    let mut recombined = cpu_sep.starless.clone();
    screen_blend_stars(&mut recombined, &cpu_sep.stars, 1.0);
    let roundtrip_err = recombined
        .data
        .iter()
        .zip(&img.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        roundtrip_err <= 2e-4,
        "Unscreen + screen blend roundtrip error {roundtrip_err} exceeds 2e-4"
    );
}

#[test]
fn pipeline_run_edit_and_magic_view_support_starless_workflows() {
    let tmp = TempDir::new().expect("tempdir");
    let fix_dir = tmp.path().join("obs");
    let out_dir = tmp.path().join("gallery");
    make_fixture::generate_full_fixture(&fix_dir, 64, 64, 3).expect("generate fixture");

    let opts = ProcessOptions {
        accelerator: AcceleratorMode::Auto,
        palette: Some("hoo".to_string()),
        finish: FinishOptions {
            star_mode: StarMode::Starless,
            ..Default::default()
        },
        ..Default::default()
    };
    let manifest = photonloom::process(&fix_dir, &out_dir, &opts).expect("process with starless");
    let target = &manifest.targets[0];
    assert_eq!(target.selected_variant, "hoo-starless");

    let var_ids: Vec<&str> = target.variants.iter().map(|v| v.id.as_str()).collect();
    assert!(var_ids.contains(&"hoo"), "Base HOO variant must be preserved");
    assert!(var_ids.contains(&"hoo-starless"), "Starless HOO variant must be emitted");
    assert!(var_ids.contains(&"hoo-stars-only"), "Stars-only companion variant must be emitted");

    // Verify deterministic edit grammar for starless and star reduction
    let avail = vec!["natural".to_string(), "hoo".to_string(), "sho".to_string()];
    let parsed_sl = parse_deterministic_edit("make the SHO image starless", &avail).unwrap();
    assert_eq!(parsed_sl.palette.as_deref(), Some("sho"));
    assert_eq!(parsed_sl.star_mode, Some(StarMode::Starless));
    assert_eq!(parsed_sl.star_blend, Some(0.0));

    let parsed_red = parse_deterministic_edit("reduce stars a little in HOO", &avail).unwrap();
    assert_eq!(parsed_red.palette.as_deref(), Some("hoo"));
    assert_eq!(parsed_red.star_mode, Some(StarMode::Reduced));
    assert_eq!(parsed_red.star_blend, Some(0.65));

    let edited_var = apply_gallery_edit(&out_dir, "make the SHO image starless", None)
        .expect("apply starless edit");
    assert_eq!(edited_var.id, "sho-starless");
    assert!(out_dir.join(&edited_var.png_path).is_file());
    assert!(out_dir.join(&edited_var.tiff_path).is_file());

    // Verify Magic View starless mode produces distinct safety-gated starless previews and render
    let source = MagicSource::Gallery {
        gallery: out_dir.clone(),
        target_id: target.target_id.clone(),
        palette: "hoo".to_string(),
    };
    let mut std_session = magic::analyze(
        &source,
        &MagicOptions {
            starless: false,
            ..Default::default()
        },
        &tmp.path().join("magic-std"),
    )
    .unwrap();
    let std_art = magic::render(&mut std_session, MagicVariant::Enhanced, 0.5).unwrap();

    let mut sl_session = magic::analyze(
        &source,
        &MagicOptions {
            starless: true,
            ..Default::default()
        },
        &tmp.path().join("magic-sl"),
    )
    .unwrap();
    let sl_previews = magic::preview(&mut sl_session, 0.5).unwrap();
    assert_eq!(sl_previews.len(), 3);
    assert!(
        sl_previews
            .iter()
            .all(|p| p.decision.explanation.contains("starless nebula"))
    );
    let sl_art = magic::render(&mut sl_session, MagicVariant::Enhanced, 0.5).unwrap();
    assert_ne!(std_art.id, sl_art.id);
    assert!(sl_art.title.contains("Starless"));
    assert!(tmp.path().join("magic-sl").join(&sl_art.png_path).is_file());
}

#[test]
fn standalone_magic_view_starless_removes_stars_on_real_ctb1_crop() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("magic-hoo-enhanced.png");
    std::fs::write(&source, include_bytes!("fixtures/magic-hoo-enhanced.png")).unwrap();
    let mut session = magic::analyze(
        &MagicSource::File {
            path: source.clone(),
            domain: InputDomain::Auto,
            transfer: TransferFunction::Auto,
        },
        &MagicOptions {
            starless: true,
            ..Default::default()
        },
        &temp.path().join("out"),
    )
    .unwrap();
    let previews = magic::preview(&mut session, 0.5).unwrap();
    assert_eq!(previews.len(), 3);
    let enhanced = previews
        .iter()
        .find(|p| p.variant == MagicVariant::Enhanced)
        .unwrap();
    let orig_img = image::open(&source).unwrap().to_rgb8();
    let sl_img = image::open(temp.path().join("out").join(&enhanced.preview_path))
        .unwrap()
        .to_rgb8();
    assert_eq!(orig_img.dimensions(), sl_img.dimensions());
    assert_ne!(orig_img.as_raw(), sl_img.as_raw());
}
