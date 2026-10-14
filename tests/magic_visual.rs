use photonloom::magic::{self, InputDomain, MagicOptions, MagicSource, MagicVariant, TransferFunction};

#[test]
fn tiny_hoo_display_crop_retains_bounded_color_and_stellar_profiles() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("magic-hoo-enhanced.png");
    std::fs::write(&source, include_bytes!("fixtures/magic-hoo-enhanced.png")).unwrap();
    let mut session = magic::analyze(
        &MagicSource::File { path: source, domain: InputDomain::Auto, transfer: TransferFunction::Auto },
        &MagicOptions::default(), &temp.path().join("output"),
    ).unwrap();
    let previews = magic::preview(&mut session, 0.5).unwrap();
    let enhanced = previews.iter().find(|p| p.variant == MagicVariant::Enhanced).unwrap();
    let got = image::open(temp.path().join("output").join(&enhanced.preview_path)).unwrap().to_rgb8();
    let expected = image::load_from_memory(include_bytes!("fixtures/magic-hoo-enhanced-expected.png")).unwrap().to_rgb8();
    assert_eq!(got.dimensions(), expected.dimensions());
    let mut worst = 0u8;
    for (actual, reference) in got.as_raw().iter().zip(expected.as_raw()) {
        worst = worst.max(actual.abs_diff(*reference));
    }
    assert!(worst <= 2, "Expected bounded HOO color/stellar crop, worst RGB8 difference {worst}");
    let natural = previews.iter().find(|p| p.variant == MagicVariant::Natural).unwrap();
    let dramatic = previews.iter().find(|p| p.variant == MagicVariant::Dramatic).unwrap();
    assert!(natural.decision.actual.tone > 0.0);
    assert!(natural.decision.actual.tone < enhanced.decision.actual.tone);
    assert!(enhanced.decision.actual.tone < dramatic.decision.actual.tone);
    assert_ne!(natural.decision.explanation, enhanced.decision.explanation);
    assert_ne!(enhanced.decision.explanation, dramatic.decision.explanation);
    let nat_img = image::open(temp.path().join("output").join(&natural.preview_path)).unwrap().to_rgb8();
    let dram_img = image::open(temp.path().join("output").join(&dramatic.preview_path)).unwrap().to_rgb8();
    assert_ne!(nat_img.as_raw(), got.as_raw());
    assert_ne!(got.as_raw(), dram_img.as_raw());
}
