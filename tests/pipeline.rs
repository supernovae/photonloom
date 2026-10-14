#[path = "../examples/make_fixture.rs"]
#[allow(dead_code)]
mod make_fixture;

use photonloom::config::{AcceleratorMode, BayerPattern, Channel, FrameRole, ProcessOptions};
use photonloom::fits::{inspect_fits_header, read_fits_linear};
use photonloom::raw::{inspect_raw_header, read_raw_mosaic};
use photonloom::register::{detect_stars, match_star_fields, robust_median_mad};
use photonloom::scan::{partition_targets, scan_detailed};
use tempfile::TempDir;

#[test]
fn test_full_pipeline_calibration_registration_rejection_and_signal_accuracy() {
    let tmp = TempDir::new().expect("tempdir");
    let fix_dir = tmp.path().join("obs");
    let out_dir = tmp.path().join("obs-photonloom");

    make_fixture::generate_full_fixture(&fix_dir, 64, 64, 3).expect("generate fixture");

    // 1. Verify scan and registration accuracy (< 0.5 px on frame 2's +3/-2 px dither)
    let scan_rep = scan_detailed(&fix_dir).expect("scan");
    let groups = partition_targets(&scan_rep, &ProcessOptions::default()).expect("partition");
    assert_eq!(
        groups.len(),
        1,
        "All overlapping filters must group into 1 target"
    );
    let group = &groups[0];

    let ha_lights = &group.lights_by_channel[&Channel::Ha];
    assert_eq!(ha_lights.len(), 3);
    let f0 = read_fits_linear(&ha_lights[0]).expect("read f0");
    let f1 = read_fits_linear(&ha_lights[1]).expect("read f1");
    let stars0 = detect_stars(&f0);
    let stars1 = detect_stars(&f1);
    assert!(
        stars0.len() >= 15,
        "Expected >= 15 stars, got {}",
        stars0.len()
    );
    assert!(
        stars1.len() >= 15,
        "Expected >= 15 stars, got {}",
        stars1.len()
    );

    let reg = match_star_fields(&stars0, &stars1, 64, 64).expect("match star fields");
    assert!(reg.inliers >= 12);
    assert!(
        reg.rms < 0.5,
        "Registration RMS {} must be < 0.5 px",
        reg.rms
    );
    let tx_err = (reg.transform.tx - 3.0).abs();
    let ty_err = (reg.transform.ty - (-2.0)).abs();
    assert!(
        tx_err < 0.5 && ty_err < 0.5,
        "Recovered dither ({:.3}, {:.3}) must be within 0.5 px of (+3.0, -2.0)",
        reg.transform.tx,
        reg.transform.ty
    );

    // 2. Run full pipeline on CPU
    let opts = ProcessOptions {
        accelerator: AcceleratorMode::Cpu,
        palette: None,
        mix: None,
        restore: false,
        progress: None,
        ..Default::default()
    };
    let manifest = photonloom::process(&fix_dir, &out_dir, &opts).expect("process full fixture");
    assert_eq!(manifest.targets.len(), 1);
    let t = &manifest.targets[0];

    // Check master selection recorded in calibration summary
    assert!(t.calibration_summary.used_bias.is_some());
    assert!(!t.calibration_summary.used_darks.is_empty());
    assert!(!t.calibration_summary.used_flats.is_empty());

    // Check gallery variants: natural, hoo, sho, osh, hubble, jwst-inspired
    let var_ids: Vec<&str> = t.variants.iter().map(|v| v.id.as_str()).collect();
    for expected_pal in ["natural", "hoo", "sho", "osh", "hubble", "jwst-inspired"] {
        assert!(
            var_ids.contains(&expected_pal),
            "Missing expected variant '{}'; present: {:?}",
            expected_pal,
            var_ids
        );
    }

    // 3. Inspect integrated Ha science FITS for calibrated background, hot-pixel & streak rejection, and signal within 5%
    let ha_sci = t
        .science_integrations
        .iter()
        .find(|s| s.channel == "Ha")
        .expect("Ha science integration");
    let ha_fits_path = out_dir.join(&ha_sci.fits_path);
    let ha_meta = inspect_fits_header(&ha_fits_path).expect("inspect Ha FITS");
    let ha_frame = photonloom::fits::build_fits_frame(
        &ha_fits_path,
        ha_meta,
        FrameRole::Light,
        Some(Channel::Ha),
    );
    let ha_stack = read_fits_linear(&ha_frame).expect("read Ha stack");

    // Expected clean signal on unshifted reference frame (Ha channel_scale = 1.2)
    let expected_clean =
        make_fixture::synthesize_light_plane(64, 64, 0.0, 0.0, 1.2, false, false, 100);

    // Verify calibrated background median is within 5% of expected clean background (no 125 ADU bias+dark offset left)
    let (stack_bg_med, _) = robust_median_mad(ha_stack.plane(0), &ha_stack.mask);
    let (expected_bg_med, _) = robust_median_mad(&expected_clean, &ha_stack.mask);
    let bg_rel_err = (stack_bg_med - expected_bg_med).abs() / expected_bg_med.max(1.0);
    assert!(
        bg_rel_err < 0.05,
        "Calibrated background median {:.2} differs from expected {:.2} by {:.2}% (> 5%)",
        stack_bg_med,
        expected_bg_med,
        bg_rel_err * 100.0
    );

    // Verify hot pixel at (31, 27) (or its shifted position (28, 29) in ref grid) was rejected/interpolated
    for &(hx, hy) in &[(31usize, 27usize), (28usize, 29usize)] {
        let p = hy * 64 + hx;
        let diff = (ha_stack.data[p] - expected_clean[p]).abs();
        assert!(
            diff < 25.0,
            "Hot pixel residual at ({hx},{hy}) was not rejected: stack={:.1}, clean={:.1}",
            ha_stack.data[p],
            expected_clean[p]
        );
    }

    // Verify satellite streak on frame 3 (row y = 48 in frame 3 -> y = 47 in ref grid) was rejected by 3-sigma Winsorized clipping
    for sy in [47usize, 48usize] {
        let p = sy * 64 + 32;
        let diff = (ha_stack.data[p] - expected_clean[p]).abs();
        assert!(
            diff < 25.0,
            "Streak residual at (32,{sy}) was not rejected: stack={:.1}, clean={:.1}",
            ha_stack.data[p],
            expected_clean[p]
        );
    }

    // Verify overall science stack signal across valid interior pixels is within 5% of known clean signal
    let mut sum_stack = 0.0f64;
    let mut sum_clean = 0.0f64;
    for y in 6..58 {
        for x in 6..58 {
            let p = y * 64 + x;
            if ha_stack.mask[p] != 0 {
                sum_stack += f64::from(ha_stack.data[p]);
                sum_clean += f64::from(expected_clean[p]);
            }
        }
    }
    let signal_rel_err = ((sum_stack - sum_clean) / sum_clean).abs();
    assert!(
        signal_rel_err < 0.05,
        "Integrated science stack signal differs from ground truth by {:.3}% (limit 5%)",
        signal_rel_err * 100.0
    );
}

#[test]
fn test_raw_dng_mosaic_decode_and_pipeline() {
    let tmp = TempDir::new().expect("tempdir");
    let dng_dir = tmp.path().join("dng-obs");
    let out_dir = tmp.path().join("dng-out");

    make_fixture::generate_dng_fixture(&dng_dir, 64, 64).expect("generate dng fixture");

    let first_dng = dng_dir.join("lights/light_01.dng");
    let raw_meta = inspect_raw_header(&first_dng).expect("inspect DNG via LibRaw");
    assert_eq!(raw_meta.width, 64);
    assert_eq!(raw_meta.height, 64);
    assert_eq!(raw_meta.bayer, Some(BayerPattern::Rggb));
    assert!((raw_meta.black_level - 100.0).abs() < 1e-3);

    let frame = photonloom::raw::build_raw_frame(
        &first_dng,
        raw_meta,
        FrameRole::Light,
        Some(Channel::RgbCfa),
    );
    let mosaic = read_raw_mosaic(&frame).expect("unpack DNG mosaic");
    assert_eq!(mosaic.channels, 1);
    assert_eq!(mosaic.bayer, Some(BayerPattern::Rggb));

    // Verify black level (100) was subtracted during raw mosaic decode
    let (med, _) = robust_median_mad(&mosaic.data, &mosaic.mask);
    assert!(
        med > 40.0 && med < 120.0,
        "Expected black-subtracted DNG mosaic median near ~60-80 ADU, got {med}"
    );

    // Run full pipeline on DNG fixture
    let opts = ProcessOptions {
        accelerator: AcceleratorMode::Cpu,
        ..Default::default()
    };
    let manifest = photonloom::process(&dng_dir, &out_dir, &opts).expect("process DNG fixture");
    let var_ids: Vec<&str> = manifest.targets[0]
        .variants
        .iter()
        .map(|v| v.id.as_str())
        .collect();
    assert!(var_ids.contains(&"natural"));
    assert!(!var_ids.contains(&"sho"));
    assert!(!var_ids.contains(&"hoo"));
}

#[test]
fn test_bad_dark_omission_and_empty_input_failure() {
    let tmp = TempDir::new().expect("tempdir");
    let bad_dark_dir = tmp.path().join("bad-dark");
    let bad_dark_out = tmp.path().join("bad-dark-out");
    make_fixture::generate_bad_dark_fixture(&bad_dark_dir, 64, 64).expect("bad dark fixture");

    let opts = ProcessOptions {
        accelerator: AcceleratorMode::Cpu,
        ..Default::default()
    };
    let manifest =
        photonloom::process(&bad_dark_dir, &bad_dark_out, &opts).expect("process bad-dark");
    assert!(
        manifest.targets[0]
            .calibration_summary
            .used_darks
            .is_empty(),
        "Mismatched dark must not be used"
    );
    assert!(
        manifest
            .warnings
            .iter()
            .any(|w| w.to_lowercase().contains("incompatible dark")),
        "Expected warning about omitted incompatible dark, got: {:?}",
        manifest.warnings
    );

    // Empty input must fail and leave no output directory behind
    let empty_dir = tmp.path().join("empty");
    let empty_out = tmp.path().join("empty-out");
    make_fixture::generate_empty_fixture(&empty_dir).expect("empty fixture");
    assert!(photonloom::process(&empty_dir, &empty_out, &opts).is_err());
    assert!(
        !empty_out.exists(),
        "Failed run must clean up newly created output dir"
    );

    // Path safety: input == output or nested output must fail
    assert!(photonloom::process(&bad_dark_dir, &bad_dark_dir, &opts).is_err());
    assert!(photonloom::process(&bad_dark_dir, &bad_dark_dir.join("nested"), &opts).is_err());
}

#[test]
fn test_conflicting_roles_and_loopback_bind_guard() {
    let tmp = TempDir::new().expect("tempdir");
    let conflict_dir = tmp.path().join("conflict");
    make_fixture::generate_rgb_only_fixture(&conflict_dir, 64, 64).expect("rgb fixture");

    // Place a Dark Frame inside `lights/` subfolder -> must fail with conflicting role error
    let bad_file = conflict_dir.join("lights/impostor.fits");
    std::fs::create_dir_all(bad_file.parent().unwrap()).unwrap();
    std::fs::copy(conflict_dir.join("dark_rgb.fits"), &bad_file).expect("copy dark into lights");
    let err = scan_detailed(&conflict_dir).expect_err("conflicting header/folder role must fail");
    assert!(
        format!("{err:#}").to_lowercase().contains("conflicting"),
        "Expected conflicting role error, got: {err:#}"
    );

    // Loopback bind guard
    assert!(photonloom::web::parse_loopback_bind("127.0.0.1:8787").is_ok());
    assert!(photonloom::web::parse_loopback_bind("0.0.0.0:8787").is_err());
    assert!(photonloom::web::parse_loopback_bind("192.168.1.10:8787").is_err());
}

#[test]
fn test_cpu_vs_gpu_tile_kernel_equivalence() {
    let num_pixels = 64 * 64;
    let num_frames = 4;
    let weights = vec![0.25f32; num_frames];
    let mut tile_values = vec![0.0f32; num_pixels * num_frames];
    let tile_masks = vec![1u32; num_pixels * num_frames];

    for k in 0..num_frames {
        for p in 0..num_pixels {
            let mut v = 100.0 + (p as f32) * 0.1 + (k as f32) * 0.5;
            if k == 2 && p == 500 {
                v += 5000.0; // outlier spike on frame 2
            }
            tile_values[k * num_pixels + p] = v;
        }
    }

    let (cpu_vals, cpu_mask) = photonloom::integrate::integrate_tile_cpu(
        num_pixels,
        num_frames,
        &tile_values,
        &tile_masks,
        &weights,
    );
    assert!(
        cpu_vals[500] < 200.0,
        "Outlier at pixel 500 must be clipped"
    );
    assert_eq!(cpu_mask[500], 1);

    if let Ok(mut gpu) = photonloom::gpu::GpuStacker::try_new() {
        let (gpu_vals, gpu_mask) = gpu
            .integrate_tile(num_pixels, num_frames, &tile_values, &tile_masks, &weights)
            .expect("gpu integrate_tile");
        assert_eq!(cpu_mask, gpu_mask);
        for p in 0..num_pixels {
            let diff = (gpu_vals[p] - cpu_vals[p]).abs();
            let tol = 1e-4 + 1e-3 * cpu_vals[p].abs();
            assert!(
                diff <= tol,
                "Pixel {p} GPU/CPU mismatch: gpu={} cpu={} diff={} tol={}",
                gpu_vals[p],
                cpu_vals[p],
                diff,
                tol
            );
        }
    }

    // Also compare full pipeline CPU vs GPU FITS outputs when GPU adapter is available
    if photonloom::gpu::GpuStacker::try_new().is_ok() {
        let tmp = TempDir::new().expect("tempdir");
        let fix_dir = tmp.path().join("fix");
        let cpu_out = tmp.path().join("cpu-out");
        let gpu_out = tmp.path().join("gpu-out");
        make_fixture::generate_full_fixture(&fix_dir, 64, 64, 3).expect("generate fixture");

        let cpu_manifest = photonloom::process(
            &fix_dir,
            &cpu_out,
            &ProcessOptions {
                accelerator: AcceleratorMode::Cpu,
                ..Default::default()
            },
        )
        .expect("cpu process");
        let gpu_manifest = photonloom::process(
            &fix_dir,
            &gpu_out,
            &ProcessOptions {
                accelerator: AcceleratorMode::Gpu,
                ..Default::default()
            },
        )
        .expect("gpu process");
        assert_eq!(gpu_manifest.accelerator_used, "gpu");

        for (cpu_sci, gpu_sci) in cpu_manifest.targets[0]
            .science_integrations
            .iter()
            .zip(gpu_manifest.targets[0].science_integrations.iter())
        {
            assert_eq!(cpu_sci.channel, gpu_sci.channel);
            let cpu_p = cpu_out.join(&cpu_sci.fits_path);
            let gpu_p = gpu_out.join(&gpu_sci.fits_path);
            let cpu_f = photonloom::fits::build_fits_frame(
                &cpu_p,
                inspect_fits_header(&cpu_p).unwrap(),
                FrameRole::Light,
                None,
            );
            let gpu_f = photonloom::fits::build_fits_frame(
                &gpu_p,
                inspect_fits_header(&gpu_p).unwrap(),
                FrameRole::Light,
                None,
            );
            let cpu_img = read_fits_linear(&cpu_f).unwrap();
            let gpu_img = read_fits_linear(&gpu_f).unwrap();
            for (idx, (&cv, &gv)) in cpu_img.data.iter().zip(gpu_img.data.iter()).enumerate() {
                let p = idx % (cpu_img.width * cpu_img.height);
                if cpu_img.mask[p] != 0 && gpu_img.mask[p] != 0 {
                    let diff = (gv - cv).abs();
                    let tol = 1e-4 + 1e-3 * cv.abs();
                    assert!(
                        diff <= tol,
                        "Channel {} pixel {} GPU/CPU FITS mismatch: gpu={} cpu={} diff={} tol={}",
                        cpu_sci.channel,
                        idx,
                        gv,
                        cv,
                        diff,
                        tol
                    );
                }
            }
        }
    }
}
