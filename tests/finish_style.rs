use photonloom::config::LinearImage;
use photonloom::finish_style::{FinishOptions, FinishPreset, StarMode, apply_finish};

fn gray_image(width: usize, height: usize, value: f32) -> LinearImage {
    let mut image = LinearImage::new(width, height, 3);
    for c in 0..3 {
        image.plane_mut(c).fill(value);
    }
    image
}

fn set_gray(image: &mut LinearImage, x: usize, y: usize, value: f32) {
    for c in 0..3 {
        image.set(c, x, y, value);
    }
}

#[test]
fn neutral_is_exact_identity_even_for_masked_and_out_of_range_pixels() {
    let mut image = gray_image(3, 1, 0.25);
    image.set(0, 0, 0, 1.4);
    image.set(1, 0, 0, 0.02);
    image.set(2, 2, 0, -0.1);
    image.mask[1] = 0;
    image.set(0, 1, 0, f32::NAN);
    image.exposure = 730.0;
    let before = image.clone();
    let resolved = apply_finish(
        &mut image,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(resolved.exposure, 0.0);
    assert_eq!(resolved.local_contrast, 0.0);
    assert_eq!(image.width, before.width);
    assert_eq!(image.height, before.height);
    assert_eq!(image.mask, before.mask);
    assert_eq!(image.exposure, before.exposure);
    assert!(
        image
            .data
            .iter()
            .zip(before.data.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    apply_finish(&mut image, &FinishOptions::default()).unwrap();
    assert_eq!(image.mask, before.mask);
    assert_eq!(image.exposure, before.exposure);
    for c in 0..3 {
        assert_eq!(image.get(c, 1, 0).to_bits(), before.get(c, 1, 0).to_bits());
    }
}

#[test]
fn pivot_curve_is_monotonic_and_saturation_keeps_hue_within_gamut() {
    let mut image = LinearImage::new(5, 1, 3);
    for (x, v) in [0.02, 0.08, 0.18, 0.55, 0.99].into_iter().enumerate() {
        set_gray(&mut image, x, 0, v);
    }
    let options = FinishOptions {
        preset: FinishPreset::Neutral,
        contrast: Some(1.7),
        ..Default::default()
    };
    apply_finish(&mut image, &options).unwrap();
    let values: Vec<_> = (0..5).map(|x| image.get(0, x, 0)).collect();
    assert!(values.windows(2).all(|v| v[0] < v[1]), "{values:?}");
    assert!((values[2] - 0.18).abs() < 1e-5);
    assert!(values[0] < 0.02 && values[4] > 0.99);

    let mut colorful = LinearImage::new(1, 1, 3);
    colorful.set(0, 0, 0, 0.98);
    colorful.set(1, 0, 0, 0.33);
    colorful.set(2, 0, 0, 0.08);
    apply_finish(
        &mut colorful,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            exposure: Some(1.4),
            saturation: Some(1.8),
            ..Default::default()
        },
    )
    .unwrap();
    let r = colorful.get(0, 0, 0);
    let g = colorful.get(1, 0, 0);
    let b = colorful.get(2, 0, 0);
    assert!(r <= 1.0 && b >= 0.0 && r > g && g > b);
    assert!(((r - g) / (g - b) - (0.98 - 0.33) / (0.33 - 0.08)).abs() < 1e-4);
}

#[test]
fn exposure_and_gamma_have_conventional_foreground_effect() {
    let mut exposed = gray_image(1, 1, 0.25);
    apply_finish(
        &mut exposed,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            exposure: Some(1.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((exposed.get(0, 0, 0) - 0.4).abs() < 1e-5);
    let mut gamma = gray_image(1, 1, 0.25);
    apply_finish(
        &mut gamma,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            gamma: Some(2.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((gamma.get(0, 0, 0) - 0.5).abs() < 1e-5);
    let mut balanced = gray_image(1, 1, 0.08);
    apply_finish(&mut balanced, &FinishOptions::default()).unwrap();
    assert!(
        balanced.get(0, 0, 0) > 0.08,
        "balanced should gently lift faint signal"
    );
}

#[test]
fn dark_nebula_preset_deepens_real_dust_but_not_star_halos_or_flat_noise() {
    let mut scene = gray_image(96, 96, 0.28);
    for y in 42..54 {
        for x in 40..52 {
            set_gray(&mut scene, x, y, 0.16);
        }
    }
    set_gray(&mut scene, 22, 22, 0.99);
    let mut without_local = scene.clone();
    apply_finish(
        &mut without_local,
        &FinishOptions {
            preset: FinishPreset::DarkNebula,
            local_contrast: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let mut with_local = scene;
    apply_finish(
        &mut with_local,
        &FinishOptions {
            preset: FinishPreset::DarkNebula,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(with_local.get(0, 41, 45) + 0.002 < without_local.get(0, 41, 45));
    assert!((with_local.get(0, 23, 22) - without_local.get(0, 23, 22)).abs() < 0.001);
    assert!((with_local.get(0, 22, 22) - without_local.get(0, 22, 22)).abs() < 0.001);
    assert!(
        with_local.get(0, 41, 45) > 0.06,
        "dust should not be black-clipped"
    );

    let mut noise = gray_image(96, 96, 0.25);
    for y in 0..96 {
        for x in 0..96 {
            set_gray(
                &mut noise,
                x,
                y,
                if (x + y) % 2 == 0 { 0.244 } else { 0.256 },
            );
        }
    }
    let mut baseline = noise.clone();
    apply_finish(
        &mut baseline,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            ..Default::default()
        },
    )
    .unwrap();
    apply_finish(
        &mut noise,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            local_contrast: Some(0.3),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        noise
            .data
            .iter()
            .zip(baseline.data.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
}

#[test]
fn invalid_knobs_reject_nan_infinity_and_out_of_range_before_touching_pixels() {
    let invalid = [
        FinishOptions {
            exposure: Some(f32::NAN),
            ..Default::default()
        },
        FinishOptions {
            gamma: Some(f32::INFINITY),
            ..Default::default()
        },
        FinishOptions {
            contrast: Some(0.0),
            ..Default::default()
        },
        FinishOptions {
            saturation: Some(1.81),
            ..Default::default()
        },
        FinishOptions {
            local_contrast: Some(-0.01),
            ..Default::default()
        },
        FinishOptions {
            exposure: Some(4.01),
            ..Default::default()
        },
        FinishOptions {
            star_blend: Some(-0.01),
            ..Default::default()
        },
        FinishOptions {
            star_blend: Some(1.01),
            ..Default::default()
        },
        FinishOptions {
            star_blend: Some(f32::NAN),
            ..Default::default()
        },
    ];
    for options in invalid {
        let mut image = gray_image(2, 2, 0.25);
        let before = image.data.clone();
        assert!(apply_finish(&mut image, &options).is_err());
        assert_eq!(image.data, before);
    }
}

#[test]
fn starless_reduced_and_stars_only_modes_separate_stars_while_preserving_nebula() {
    let (w, h) = (96usize, 96usize);
    let mut scene = LinearImage::new(w, h, 3);
    let star_centers = [(24.0f32, 24.0f32, 0.72f32), (68.0, 32.0, 0.65), (48.0, 70.0, 0.80)];
    for y in 0..h {
        for x in 0..w {
            let xf = x as f32;
            let yf = y as f32;
            let neb = 0.14 + 0.10 * (-((xf - 48.0).powi(2) + (yf - 48.0).powi(2)) / 900.0).exp();
            let mut star = 0.0f32;
            for &(sx, sy, amp) in &star_centers {
                let r2 = (xf - sx).powi(2) + (yf - sy).powi(2);
                if r2 <= 36.0 {
                    star += amp * (-r2 / (2.0 * 1.35 * 1.35)).exp();
                }
            }
            scene.set(0, x, y, (neb * 1.15 + star).clamp(0.0, 1.0));
            scene.set(1, x, y, (neb * 0.95 + star).clamp(0.0, 1.0));
            scene.set(2, x, y, (neb * 0.85 + star).clamp(0.0, 1.0));
        }
    }

    let mut starless_img = scene.clone();
    let res_starless = apply_finish(
        &mut starless_img,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            star_mode: StarMode::Starless,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res_starless.star_mode, StarMode::Starless);
    assert_eq!(res_starless.star_blend, 0.0);

    let mut reduced_img = scene.clone();
    let res_reduced = apply_finish(
        &mut reduced_img,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            star_mode: StarMode::Reduced,
            star_blend: Some(0.45),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res_reduced.star_mode, StarMode::Reduced);
    assert!((res_reduced.star_blend - 0.45).abs() < 1e-6);

    let mut stars_only_img = scene.clone();
    let res_stars = apply_finish(
        &mut stars_only_img,
        &FinishOptions {
            preset: FinishPreset::Neutral,
            star_mode: StarMode::StarsOnly,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res_stars.star_mode, StarMode::StarsOnly);

    // At each star center: starless < reduced < original, and stars_only has strong positive signal
    for &(sx, sy, _) in &star_centers {
        let (ix, iy) = (sx as usize, sy as usize);
        let orig_v = scene.get(0, ix, iy);
        let sl_v = starless_img.get(0, ix, iy);
        let red_v = reduced_img.get(0, ix, iy);
        let st_v = stars_only_img.get(0, ix, iy);
        assert!(
            sl_v < orig_v - 0.40,
            "Star at ({ix},{iy}) was not removed: orig={orig_v}, starless={sl_v}"
        );
        assert!(
            sl_v < red_v && red_v < orig_v,
            "Expected starless ({sl_v}) < reduced ({red_v}) < original ({orig_v})"
        );
        assert!(st_v > 0.45, "Stars-only layer missing star at ({ix},{iy}): {st_v}");
    }

    // Away from stars (e.g. central nebula at (48, 48)): nebula is untouched in starless & zero in stars-only
    assert!((starless_img.get(0, 48, 48) - scene.get(0, 48, 48)).abs() < 1e-6);
    assert!(stars_only_img.get(0, 48, 48) < 1e-5);
}
