use std::collections::BTreeMap;

use photonloom::config::{
    CalibrationSummary, Channel, GalleryManifest, LinearImage, ProcessOptions,
};
use photonloom::finish::{
    build_normalized_channel_map, compose_palette_linear, render_target_gallery,
    rerender_variant_from_stored_fits,
};
use photonloom::fits::write_fits_linear;
use photonloom::integrate::{ChannelIntegration, TargetIntegrationOutput};
use tempfile::TempDir;

fn image(seed: f32) -> LinearImage {
    let mut img = LinearImage::new(32, 32, 1);
    for y in 0..32 {
        for x in 0..32 {
            img.set(
                0,
                x,
                y,
                seed + (x * 7 + y * 3) as f32
                    + if (x + y + seed as usize) % 13 == 0 {
                        60.0
                    } else {
                        0.0
                    },
            );
        }
    }
    img
}

fn gallery(
    channels: &[Channel],
    unaligned: Option<Channel>,
    options: ProcessOptions,
) -> (
    TempDir,
    TargetIntegrationOutput,
    photonloom::config::TargetGallery,
    Vec<String>,
) {
    let tmp = TempDir::new().unwrap();
    let mut integrations = BTreeMap::new();
    for (i, &ch) in channels.iter().enumerate() {
        let img = image(i as f32 * 25.0);
        let fits_rel_path = format!("run/integrated-test-{}.fits", ch.as_str());
        write_fits_linear(
            &tmp.path().join(&fits_rel_path),
            &img,
            ch.as_str(),
            "test",
            10.0,
            1,
            0,
            None,
        )
        .unwrap();
        integrations.insert(
            ch,
            ChannelIntegration {
                channel: ch,
                image: img,
                aligned_to_target: unaligned != Some(ch),
                total_exposure_s: 10.0,
                accepted_frames: vec![format!("input-{i}")],
                rejected_frames: vec![],
                fits_rel_path,
                reference_frame_path: None,
                reference_wcs: None,
                wcs: None,
                valid_region: Vec::new(),
                recommended_crop: None,
            },
        );
    }
    let output = TargetIntegrationOutput {
        target_id: "test".into(),
        target_label: "Test".into(),
        width: 32,
        height: 32,
        integrations,
        warnings: vec![],
        rejected_frames: vec![],
        used_gpu: false,
        cross_filter_aligned: unaligned.is_none(),
    };
    std::fs::create_dir_all(tmp.path().join("run")).unwrap();
    let mut warnings = vec![];
    let result = render_target_gallery(
        "test",
        "Test",
        &CalibrationSummary::default(),
        output.clone(),
        &tmp.path().join("run"),
        "run",
        &options,
        None,
        &mut warnings,
    )
    .unwrap();
    (tmp, output, result, warnings)
}

fn variant<'a>(
    target: &'a photonloom::config::TargetGallery,
    id: &str,
) -> &'a photonloom::config::VariantArtifact {
    target
        .variants
        .iter()
        .find(|v| v.id == id)
        .unwrap_or_else(|| panic!("missing variant {id}"))
}

fn gray_png(root: &std::path::Path, variant: &photonloom::config::VariantArtifact) {
    let rgb = image::open(root.join(&variant.png_path)).unwrap().to_rgb8();
    assert!(
        rgb.pixels().all(|p| p[0] == p[1] && p[1] == p[2]),
        "{} must display in grayscale",
        variant.id
    );
    assert!(
        rgb.pixels().any(|p| p[0] != 0),
        "{} must contain actual signal",
        variant.id
    );
}

#[test]
fn complete_filter_sets_choose_measured_palette() {
    let (tmp, _, hoo, _) = gallery(
        &[Channel::Ha, Channel::Oiii],
        None,
        ProcessOptions::default(),
    );
    assert_eq!(hoo.primary_variant, "hoo");
    assert_eq!(hoo.selected_variant, "hoo");
    assert_eq!(variant(&hoo, "hoo").channels_used, ["Ha", "OIII"]);
    assert!(tmp.path().join(&variant(&hoo, "hoo").png_path).exists());
    assert!(
        hoo.variants
            .iter()
            .all(|v| !v.channels_used.contains(&"SII".to_string()))
    );

    let (_, _, sho, _) = gallery(
        &[Channel::Sii, Channel::Ha, Channel::Oiii],
        None,
        ProcessOptions::default(),
    );
    assert_eq!(sho.primary_variant, "sho");
    assert_eq!(variant(&sho, "sho").channels_used, ["SII", "Ha", "OIII"]);

    let (_, _, rgb, _) = gallery(
        &[
            Channel::R,
            Channel::G,
            Channel::B,
            Channel::Ha,
            Channel::Oiii,
            Channel::Sii,
        ],
        None,
        ProcessOptions::default(),
    );
    assert_eq!(rgb.primary_variant, "natural");
    assert_eq!(variant(&rgb, "natural").channels_used, ["R", "G", "B"]);
}

#[test]
fn partial_filter_sets_produce_honest_gray_with_stable_priority() {
    let (tmp, _, single, _) = gallery(&[Channel::Ha], None, ProcessOptions::default());
    assert_eq!(single.primary_variant, "monochrome");
    let ha = variant(&single, "monochrome");
    assert_eq!(ha.channels_used, ["Ha"]);
    assert!(ha.title.contains("Ha") && !ha.title.contains("Luminance"));
    gray_png(tmp.path(), ha);

    let (tmp, _, narrow, _) = gallery(
        &[Channel::Sii, Channel::Ha],
        None,
        ProcessOptions::default(),
    );
    assert_eq!(narrow.primary_variant, "monochrome-Ha");
    for (id, band) in [("monochrome-Ha", "Ha"), ("monochrome-SII", "SII")] {
        let v = variant(&narrow, id);
        assert_eq!(v.channels_used, [band]);
        assert!(v.title.contains(band));
        gray_png(tmp.path(), v);
    }
    assert!(
        !narrow
            .variants
            .iter()
            .any(|v| v.id == "hoo" || v.id == "sho")
    );

    let (tmp, _, partial_rgb, warnings) = gallery(
        &[Channel::R, Channel::G],
        None,
        ProcessOptions {
            palette: Some("natural".into()),
            ..ProcessOptions::default()
        },
    );
    assert_eq!(partial_rgb.primary_variant, "monochrome-R");
    assert_eq!(partial_rgb.selected_variant, "monochrome-R");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("natural") && w.contains("monochrome-R"))
    );
    for (id, band) in [("monochrome-R", "R"), ("monochrome-G", "G")] {
        let v = variant(&partial_rgb, id);
        assert_eq!(v.channels_used, [band]);
        gray_png(tmp.path(), v);
    }
}

#[test]
fn monochrome_composition_uses_only_its_named_filter_and_rerenders_from_fits() {
    let (tmp, output, target, _) = gallery(
        &[Channel::Ha, Channel::Sii],
        None,
        ProcessOptions::default(),
    );
    let raw: BTreeMap<_, _> = output
        .integrations
        .iter()
        .map(|(&ch, ci)| (ch, &ci.image))
        .collect();
    let norm = build_normalized_channel_map(&raw);
    for (id, channel) in [
        ("monochrome-Ha", Channel::Ha),
        ("monochrome-SII", Channel::Sii),
    ] {
        let (img, used, _, _, _) = compose_palette_linear(id, &raw, &norm, None).unwrap();
        assert_eq!(used, [channel.as_str()]);
        assert_eq!(img.plane(0), norm[&channel].plane(0));
        assert_eq!(img.plane(0), img.plane(1));
        assert_eq!(img.plane(1), img.plane(2));
    }
    let mut manifest = GalleryManifest {
        version: 1,
        run_id: "test".into(),
        run_dir: "run".into(),
        canonical_input: "test".into(),
        canonical_output: tmp.path().display().to_string(),
        created_at_unix_ms: 0,
        accelerator_requested: "cpu".into(),
        accelerator_used: "cpu".into(),
        targets: vec![target],
        warnings: vec![],
        rejected_frames: vec![],
    };
    assert_ne!(
        norm[&Channel::Ha].plane(0),
        norm[&Channel::Sii].plane(0),
        "filters must have distinguishable signals"
    );
    let rerendered = rerender_variant_from_stored_fits(
        tmp.path(),
        &mut manifest,
        Some("monochrome-SII"),
        None,
        None,
        None,
        None,
        Some("edited"),
    )
    .unwrap();
    assert_eq!(rerendered.channels_used, ["SII"]);
    assert_eq!(rerendered.id, "monochrome-SII-edited");
    assert_eq!(manifest.targets[0].selected_variant, rerendered.id);
    gray_png(tmp.path(), &rerendered);
}

#[test]
fn failed_cross_filter_alignment_keeps_standalone_science_and_blocks_rerender_mix() {
    let (tmp, _, target, _) = gallery(
        &[Channel::Ha, Channel::Oiii],
        Some(Channel::Oiii),
        ProcessOptions::default(),
    );
    assert_eq!(target.primary_variant, "monochrome-Ha");
    assert!(!target.variants.iter().any(|v| v.id == "hoo"));
    for band in ["Ha", "OIII"] {
        let sci = target
            .science_integrations
            .iter()
            .find(|s| s.channel == band)
            .unwrap();
        assert!(tmp.path().join(&sci.fits_path).exists());
        gray_png(tmp.path(), variant(&target, &format!("monochrome-{band}")));
    }
    let mut manifest = GalleryManifest {
        version: 1,
        run_id: "test".into(),
        run_dir: "run".into(),
        canonical_input: "test".into(),
        canonical_output: tmp.path().display().to_string(),
        created_at_unix_ms: 0,
        accelerator_requested: "cpu".into(),
        accelerator_used: "cpu".into(),
        targets: vec![target],
        warnings: vec![],
        rejected_frames: vec![],
    };
    assert!(
        rerender_variant_from_stored_fits(
            tmp.path(),
            &mut manifest,
            Some("hoo"),
            None,
            None,
            None,
            None,
            None
        )
        .is_err()
    );
    let mono = rerender_variant_from_stored_fits(
        tmp.path(),
        &mut manifest,
        Some("monochrome-OIII"),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(mono.channels_used, ["OIII"]);
}

#[test]
fn cross_filter_exports_only_reference_grid_wcs() {
    use photonloom::fits::{WcsTan, inspect_fits_header};
    use photonloom::integrate::{build_science_artifacts, finalize_target_channels};
    let (tmp, mut output, _, _) = gallery(
        &[Channel::Ha, Channel::Oiii], None, ProcessOptions::default(),
    );
    let reference = WcsTan {
        crval: [23.0, 41.0], crpix: [16.0, 16.0],
        cd: [[-0.001, 0.0], [0.0, 0.001]],
    };
    let other = WcsTan {
        crval: [120.0, -20.0], crpix: [16.0, 16.0],
        cd: [[-0.001, 0.0], [0.0, 0.001]],
    };
    output.integrations.get_mut(&Channel::Ha).unwrap().reference_wcs = Some(reference);
    output.integrations.get_mut(&Channel::Oiii).unwrap().reference_wcs = Some(other);
    let output = finalize_target_channels(output, &tmp.path().join("run"), "run").unwrap();
    let ha = &output.integrations[&Channel::Ha];
    let oiii = &output.integrations[&Channel::Oiii];
    assert_eq!(ha.wcs, Some(reference));
    assert_eq!(oiii.wcs, None);
    assert_eq!(inspect_fits_header(&tmp.path().join(&ha.fits_rel_path)).unwrap().wcs, Some(reference));
    assert_eq!(inspect_fits_header(&tmp.path().join(&oiii.fits_rel_path)).unwrap().wcs, None);
    let science = build_science_artifacts(&output.integrations);
    assert_eq!(science.iter().find(|s| s.channel == "Ha").unwrap().wcs, Some(reference));
    assert!(science.iter().find(|s| s.channel == "OIII").unwrap().wcs.is_none());
    let legacy = serde_json::json!({
        "channel":"Ha", "fits_path":"run/legacy.fits", "width":32, "height":32,
        "total_exposure_s":10.0, "accepted_frames":[], "rejected_frames":[]
    });
    let old: photonloom::config::ScienceArtifact = serde_json::from_value(legacy).unwrap();
    assert!(old.wcs.is_none());
}
