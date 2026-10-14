use photonloom::catalog::{SubjectClass, resolve_subject};
use photonloom::fits::WcsTan;

fn field(ra: f64, dec: f64, degrees_per_pixel: f64) -> WcsTan {
    WcsTan {
        crval: [ra, dec], crpix: [301.0, 301.0],
        cd: [[-degrees_per_pixel, 0.0], [0.0, degrees_per_pixel]],
    }
}

#[test]
fn m31_extent_outweighs_background_galaxies_and_maps_spiral() {
    let result = resolve_subject(Some(&field(10.68479,41.26906,0.003)), [601,601], "M31 HOO",false);
    assert_eq!(result.selected.as_deref(),Some("NGC0224"));
    assert_eq!(result.class,SubjectClass::SpiralGalaxy);
    assert!(result.score >= 0.6);
    assert_eq!(result.candidates[0].aliases[0],"M31");
    assert!(result.candidates[0].coverage > 0.0);
}

#[test]
fn orion_and_pleiades_have_distinct_catalog_types() {
    let orion = resolve_subject(Some(&field(83.81867,-5.38967,0.002)),[601,601],"M42",false);
    assert_eq!(orion.selected.as_deref(),Some("NGC1976"));
    assert_eq!(orion.class,SubjectClass::EmissionNebula);
    let pleiades = resolve_subject(Some(&field(56.86917,24.10528,0.003)),[601,601],"M45",false);
    assert_eq!(pleiades.selected.as_deref(),Some("Mel022"));
    assert_eq!(pleiades.class,SubjectClass::OpenCluster);
}

#[test]
fn missing_wcs_does_not_guess_sky_or_query_when_opted_in() {
    let result = resolve_subject(None,[601,601],"M31",true);
    assert_eq!(result.class,SubjectClass::Unknown);
    assert_eq!(result.selected,None);
    assert_eq!(result.candidates[0].score,0.4);
    assert_eq!(result.candidates[0].coverage,0.0);
    assert!(result.fallback_warning.as_deref().unwrap().contains("No validated WCS"));
    let ambiguous = resolve_subject(None,[601,601],"Abell 85",true);
    assert!(ambiguous.candidates.is_empty());
    assert_eq!(ambiguous.selected,None);
}

#[test]
fn far_off_center_title_cannot_override_sky_footprint() {
    let result = resolve_subject(Some(&field(10.68479,41.26906,0.002)),[601,601],"M42",false);
    assert_ne!(result.selected.as_deref(),Some("NGC1976"));
    assert!(result.candidates.iter().all(|c| c.designation != "NGC1976"));
}

#[test]
fn ra_wrap_and_polar_wcs_remain_finite() {
    let wrap = resolve_subject(Some(&field(359.99,0.0,0.002)),[601,601],"",false);
    assert!(wrap.candidates.iter().all(|c| c.score.is_finite() && (0.0..=1.0).contains(&c.score)));
    let pole = resolve_subject(Some(&field(12.0,89.95,0.0001)),[601,601],"",false);
    assert!(pole.candidates.iter().all(|c| c.score.is_finite()));
}

#[test]
fn invalid_wcs_does_not_invent_coordinates() {
    let invalid = WcsTan { cd: [[0.0;2];2], ..field(10.0,41.0,0.002) };
    let result = resolve_subject(Some(&invalid),[601,601],"M31",false);
    assert_eq!(result.selected,None);
    assert!(result.candidates.is_empty());
}
