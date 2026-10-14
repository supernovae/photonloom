use photonloom::config::LinearImage;
use photonloom::magic::{InputDomain, analysis::analyze_image, operators::{OperatorParams, apply_candidate, luminance_ghs}, validate::validate_candidate};

#[test]
fn ghs_is_strictly_monotonic_and_anchored() {
    let mut previous = luminance_ghs(0.0, 8.0, 0.35, 1.0);
    assert_eq!(previous, 0.0);
    for i in 1..=4096 {
        let now = luminance_ghs(i as f32 / 4096.0, 8.0, 0.35, 1.0);
        assert!(now > previous, "LUT ceased being strictly monotonic at sample {i}");
        previous = now;
    }
    assert_eq!(previous, 1.0);
}

#[test]
fn starless_inputs_keep_sharpening_disabled_and_zero_strength_exact() {
    let mut image = LinearImage::new(96, 96, 3);
    for p in 0..image.pixel_count() {
        for c in 0..3 { image.data[c * 96 * 96 + p] = 0.2 + 0.001 * (p % 7) as f32; }
    }
    let (analysis, masks) = analyze_image(&image, InputDomain::Display);
    let neutral = apply_candidate(&image, &analysis, &masks, &OperatorParams::default());
    assert_eq!(neutral.data, image.data);
    assert!(validate_candidate(&image, &neutral, &analysis, &masks).passed);
    let sharpen_only = apply_candidate(&image, &analysis, &masks, &OperatorParams { sharpen: 0.08, ..Default::default() });
    assert_eq!(sharpen_only.data, image.data);
}
