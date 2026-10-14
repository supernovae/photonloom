use photonloom::config::LinearImage;
use photonloom::magic::{InputDomain, analysis::analyze_image};

#[test]
fn masked_and_flat_inputs_do_not_invent_structure_or_subject() {
    let mut flat = LinearImage::new(256, 256, 3);
    flat.data.fill(0.2);
    let (analysis, masks) = analyze_image(&flat, InputDomain::Display);
    assert_eq!(analysis.luminance_stats.histogram.len(), 256);
    assert!(analysis.star_fwhm.is_none());
    assert!(analysis.structure.subject.is_none());
    assert_eq!(masks.stars.len(), 256 * 256);
    assert!(masks.stars.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    flat.mask.fill(0);
    flat.data[0] = f32::NAN;
    let (analysis, _) = analyze_image(&flat, InputDomain::Display);
    assert!(analysis.signal_snr.is_none());
    assert_eq!(analysis.measurement_confidence, 0.0);
}

#[test]
fn deterministic_histograms_exclude_invalid_pixels() {
    let mut image = LinearImage::new(128, 128, 1);
    for y in 0..128 { for x in 0..128 {
        let p = y * 128 + x;
        image.data[p] = 0.08 + ((p * 17 % 100) as f32 - 50.0) * 0.00005;
        for &(cx, cy) in &[(24.0f32, 30.0f32), (70.0, 65.0), (102.0, 92.0)] {
            let r2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
            image.data[p] += 0.8 * (-r2 / 8.0).exp();
        }
    }}
    image.data[0] = f32::NAN;
    image.mask[0] = 0;
    let (a, _) = analyze_image(&image, InputDomain::Display);
    let (b, _) = analyze_image(&image, InputDomain::Display);
    assert_eq!(serde_json::to_vec(&a).unwrap(), serde_json::to_vec(&b).unwrap());
    assert_eq!(a.luminance_stats.count, image.pixel_count() - 1);
    assert_eq!(a.luminance_stats.histogram.iter().sum::<u64>(), (image.pixel_count() - 1) as u64);
    assert!(a.luminance_stats.max.unwrap() > a.luminance_stats.median.unwrap());
}
