use photonloom::{config::LinearImage, magic::{InputDomain, analysis::analyze_image, operators::OperatorParams, optimize::search}};

fn field() -> LinearImage {
    let mut image = LinearImage::new(96, 96, 3);
    let n = image.pixel_count();
    for y in 0..96 {
        for x in 0..96 {
            let i = y * 96 + x;
            let r2 = (((x as f32 - 48.0) / 24.0).powi(2) + ((y as f32 - 48.0) / 17.0).powi(2)).sqrt();
            let signal = 0.16 * (1.0 - r2).max(0.0);
            let noise = ((x * 19 + y * 37) % 23) as f32 * 0.00035;
            for c in 0..3 { image.data[c * n + i] = 0.08 + noise + signal * [1.0, 0.85, 0.7][c]; }
        }
    }
    image
}

#[test]
fn bounded_search_has_stable_winner_and_at_most_twenty_one_evaluations() {
    let image = field();
    let (analysis, masks) = analyze_image(&image, InputDomain::Display);
    let max = OperatorParams { tone: 0.08, detail: 0.06, sharpen: 0.05, color: 0.05 };
    let a = search(&image, &analysis, &masks, max, None, "digest:general-natural:v1:crop");
    let b = search(&image, &analysis, &masks, max, None, "digest:general-natural:v1:crop");
    assert_eq!(a.report.trials.len(), 21);
    assert!(a.report.trials.iter().filter(|t| !t.cached).count() <= 21);
    assert_eq!(serde_json::to_vec(&a.report).unwrap(), serde_json::to_vec(&b.report).unwrap());
    assert_eq!(a.image.data, b.image.data);
    assert!(a.report.trials[a.report.winner].quality.passed);
    assert_eq!(a.report.winner_parameters, a.report.trials[a.report.winner].parameters);
}

#[test]
fn ties_choose_neutral_vector_even_when_all_tried_vectors_duplicate() {
    let image = field();
    let (analysis, masks) = analyze_image(&image, InputDomain::Display);
    let result = search(&image, &analysis, &masks, OperatorParams::default(), None, "zero");
    assert_eq!(result.report.winner, 0);
    assert_eq!(result.report.winner_parameters, OperatorParams::default());
    assert_eq!(result.report.trials.iter().filter(|t| !t.cached).count(), 1);
    assert_eq!(result.image.data, image.data);
}

#[test]
fn unsafe_high_contrast_is_rejected_before_scoring_or_preference() {
    let image = field();
    let (mut analysis, mut masks) = analyze_image(&image, InputDomain::Display);
    // Simulate a faulty/no-sky atlas: changes are not safe to assess, even if
    // their signal-region contrast rises and an opt-in preference favors them.
    masks.sky.fill(0.0);
    masks.signal.fill(1.0);
    analysis.multiscale_contrast = [Some(0.01); 3];
    let result = search(&image, &analysis, &masks,
        OperatorParams { tone: 0.12, detail: 0.15, sharpen: 0.08, color: 0.15 }, Some(1.0), "no-sky");
    assert!(result.report.trials.iter().any(|t| !t.quality.passed && !t.rejection_causes.is_empty()));
    assert!(result.report.trials.iter().filter(|t| !t.quality.passed).all(|t| t.objective == 0.0 && t.preference_nudge == 0.0));
    assert!(result.report.trials[result.report.winner].quality.passed);
    assert_eq!(result.image.data, image.data);
}

#[test]
fn absent_measurements_disable_dependent_operators() {
    let image = field();
    let (mut analysis, masks) = analyze_image(&image, InputDomain::Display);
    analysis.multiscale_contrast = [None; 3];
    analysis.color_stats.signal_chroma = None;
    analysis.star_fwhm = None;
    let max = OperatorParams { tone: 0.08, detail: 0.06, sharpen: 0.05, color: 0.05 };
    let result = search(&image, &analysis, &masks, max, None, "missing");
    assert!(result.report.trials.iter().all(|trial| trial.parameters == OperatorParams::default()));
    assert!(result.report.trials.iter().all(|trial| trial.contrast.bounded_gain == 0.0 && trial.chroma.bounded_gain == 0.0));
    assert_eq!(result.image.data, image.data);
}
