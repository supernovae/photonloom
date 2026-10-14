use photonloom::config::{LinearImage, GalleryManifest};
use photonloom::fits::write_fits_linear;
use photonloom::magic::{self, MagicOptions, MagicSource};

#[test]
fn named_gallery_target_only_uses_real_filters_and_leaves_science_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("gallery");
    let run = root.join("runs/run-test");
    std::fs::create_dir_all(&run).unwrap();
    let mut ha = LinearImage::new(64, 64, 1);
    let mut oiii = LinearImage::new(64, 64, 1);
    for i in 0..ha.data.len() { ha.data[i] = 20.0 + i as f32 / 200.0; oiii.data[i] = 25.0 + i as f32 / 400.0; }
    write_fits_linear(&run.join("ha.fits"), &ha, "Ha", "target", 120.0, 2, 0, None).unwrap();
    write_fits_linear(&run.join("oiii.fits"), &oiii, "OIII", "target", 120.0, 2, 0, None).unwrap();
    let sci = |channel: &str, name: &str| serde_json::json!({
        "channel":channel,"fits_path":format!("runs/run-test/{name}"),"width":64,"height":64,
        "total_exposure_s":120.0,"accepted_frames":[],"rejected_frames":[]
    });
    let variant = serde_json::json!({
        "id":"hoo","title":"HOO","palette":"hoo","description":"two real bands",
        "artistic_interpretation":false,"ai_enhanced":false,"tiff_path":"runs/run-test/unused.tiff",
        "png_path":"runs/run-test/unused.png","channels_used":["Ha","OIII"],"provenance":"science"
    });
    let target = serde_json::json!({
        "target_id":"target-1","target_label":"Target 1","primary_variant":"hoo","selected_variant":"hoo",
        "science_integrations":[sci("Ha","ha.fits"),sci("OIII","oiii.fits")],"variants":[variant],
        "omitted_variants":[],"calibration_summary":{"used_bias":null,"used_darks":[],"used_flats":[],
            "used_flat_darks":[],"missing_masters":[],"warnings":[]}
    });
    let manifest = serde_json::json!({
        "version":1,"run_id":"run-test","run_dir":"runs/run-test","canonical_input":"/unused",
        "canonical_output":root.to_str().unwrap(),"created_at_unix_ms":0,
        "accelerator_requested":"cpu","accelerator_used":"cpu","targets":[target],
        "warnings":[],"rejected_frames":[]
    });
    std::fs::write(root.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    let _manifest: GalleryManifest = serde_json::from_value(manifest).unwrap();
    let source = |palette: &str| MagicSource::Gallery { gallery: root.clone(), target_id: "target-1".into(), palette: palette.into() };
    assert!(magic::analyze(&source("sho"), &MagicOptions::default(), &root).is_err());
    let session = magic::analyze(&source(""), &MagicOptions::default(), &root).unwrap();
    assert_eq!(session.palette, "hoo");
    assert_eq!(session.channels_used, vec!["Ha", "OIII"]);
    assert!(session.analysis.structure.subject.is_none());
}
