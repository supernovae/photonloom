use photonloom::{config::LinearImage, magic::{InputDomain, analysis::analyze_image, recipes}};

fn measured() -> photonloom::magic::analysis::ImageAnalysis {
    let mut image = LinearImage::new(128, 128, 3);
    let n = image.pixel_count();
    for y in 0..128 { for x in 0..128 {
        let i = y*128+x;
        let signal = if (x as i32-64).pow(2)+(y as i32-64).pow(2) < 32*32 { 0.23 } else { 0.0 };
        let noise = ((x*97+y*31)%47) as f32*0.0008;
        for c in 0..3 { image.data[c*n+i] = 0.02+noise+signal; }
    }}
    analyze_image(&image, InputDomain::Display).0
}

#[test]
fn bundled_recipes_are_valid_and_have_explicit_limits() {
    let recipes = recipes::all();
    assert_eq!(recipes.len(), 7);
    for spec in recipes { spec.validate().unwrap(); assert!(spec.bounds.is_valid()); }
    assert_eq!(recipes::select(None, &measured(), &[]).0.id, "general-natural");
}

#[test]
fn unavailable_metrics_or_unknown_subject_return_general_recipe() {
    let mut analysis = measured();
    analysis.signal_snr = None;
    analysis.noise_sigma = None;
    analysis.star_fwhm = None;
    for class in ["emission-nebula", "spiral-galaxy", "planetary-nebula", "dark-nebula"] {
        let (spec, reason) = recipes::select(Some(class), &analysis, &[]);
        assert_eq!(spec.id, "general-natural");
        assert!(reason.unwrap().contains("fallback"));
    }
    assert_eq!(recipes::select(Some("unknown-type"), &analysis, &[]).0.id, "general-natural");
    let params = recipes::all()[0].strengths(&analysis, 1.0);
    assert_eq!(params.detail, 0.0);
    assert_eq!(params.sharpen, 0.0);
    assert_eq!(params.color, 0.0);
}

#[test]
fn unknown_operations_and_escalated_bounds_are_rejected() {
    let mut spec = recipes::all()[0].clone();
    spec.operations.push("invent-sii".into());
    assert!(spec.validate().is_err());
    spec = recipes::all()[0].clone();
    spec.bounds.detail = 0.151;
    assert!(spec.validate().is_err());
}
