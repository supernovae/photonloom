use photonloom::calibrate::{SpooledFrame, write_spooled_image};
use photonloom::config::{Channel, FrameRole, LinearImage};
use photonloom::fits::{build_fits_frame, inspect_fits_header, write_fits_linear};
use photonloom::register::evaluate_spooled_frame_quality;

#[test]
fn large_frame_quality_measures_native_star_width_and_centroids() {
    let temp = tempfile::tempdir().unwrap();
    let mut image = LinearImage::new(3072, 256, 1);
    image.data.fill(100.0);
    let centers: Vec<_> = (0..20)
        .map(|i| (35.25 + i as f32 * 137.0, 40.4 + (i % 3) as f32 * 65.0))
        .collect();
    for &(cx, cy) in &centers {
        for y in cy as usize - 6..=cy as usize + 6 {
            for x in cx as usize - 6..=cx as usize + 6 {
                let radius_sq = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                image.data[y * image.width + x] +=
                    1000.0 * (-radius_sq / (2.0 * 1.25f32.powi(2))).exp();
            }
        }
    }
    let fits = temp.path().join("light.fits");
    write_fits_linear(&fits, &image, "Ha", "quality", 300.0, 1, 0, None).unwrap();
    let frame = build_fits_frame(
        &fits,
        inspect_fits_header(&fits).unwrap(),
        FrameRole::Light,
        Some(Channel::Ha),
    );
    let spool = temp.path().join("light.spool");
    write_spooled_image(&spool, &image, 1.0).unwrap();
    let quality = evaluate_spooled_frame_quality(&SpooledFrame {
        original_frame: frame,
        spool_path: spool,
        width: image.width,
        height: image.height,
        channels: 1,
        exposure: 300.0,
        bg_sigma: 1.0,
    })
    .unwrap();
    assert_eq!(quality.star_count, centers.len());
    assert!(
        (2.7..3.1).contains(&quality.median_fwhm),
        "Native ~2.94px PSF reported as {}",
        quality.median_fwhm
    );
    for (cx, cy) in centers {
        assert!(
            quality
                .stars
                .iter()
                .any(|s| (s.x - cx).abs() < 0.08 && (s.y - cy).abs() < 0.08),
            "Missing precise centroid at ({cx},{cy})"
        );
    }
}

#[test]
fn tan_wcs_cd_wrap_crop_and_fits_roundtrip() {
    use photonloom::fits::WcsTan;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("reference.fits");
    let image = LinearImage::new(32, 24, 1);
    let wcs = WcsTan {
        crval: [359.95, 20.0],
        crpix: [11.0, 9.0],
        cd: [[-0.02, 0.003], [0.001, 0.02]],
    };
    write_fits_linear(&path, &image, "Ha", "reference", 60.0, 1, 0, Some(&wcs)).unwrap();
    let parsed = inspect_fits_header(&path).unwrap().wcs.unwrap();
    assert_eq!(parsed, wcs);
    let center = parsed.pixel_to_sky(10.0, 8.0).unwrap();
    assert!((center.0 - 359.95).abs() < 1e-10 && (center.1 - 20.0).abs() < 1e-10);
    let across = parsed.pixel_to_sky(0.0, 8.0).unwrap();
    assert!(across.0 < 1.0, "RA failed to wrap: {}", across.0);
    let cropped = parsed.shifted_for_crop(4, 5);
    assert_eq!(cropped.crpix, [7.0, 4.0]);
    assert_eq!(cropped.pixel_to_sky(6.0, 3.0), Some(center));
    assert_eq!(parsed.footprint([32, 24]).unwrap().len(), 8);
    assert!(parsed.pixel_to_sky(f64::NAN, 0.0).is_none());
}

#[test]
fn tan_wcs_pc_cdelt_near_pole_and_invalid_headers() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("pc.fits");
    let image = LinearImage::new(40, 30, 1);
    write_fits_linear(&path, &image, "Ha", "pole", 10.0, 1, 0, None).unwrap();
    assert!(inspect_fits_header(&path).unwrap().wcs.is_none());
    {
        let mut file = fitsio::FitsFile::edit(&path).unwrap();
        let hdu = file.primary_hdu().unwrap();
        for (key, value) in [
            ("CRVAL1", 12.0), ("CRVAL2", 89.8), ("CRPIX1", 20.0), ("CRPIX2", 15.0),
            ("PC1_1", 0.0), ("PC1_2", -1.0), ("PC2_1", 1.0), ("PC2_2", 0.0),
            ("CDELT1", 0.01), ("CDELT2", 0.02),
        ] {
            hdu.write_key(&mut file, key, value).unwrap();
        }
        hdu.write_key(&mut file, "CTYPE1", "RA---TAN").unwrap();
        hdu.write_key(&mut file, "CTYPE2", "DEC--TAN").unwrap();
    }
    let parsed = inspect_fits_header(&path).unwrap().wcs.unwrap();
    assert_eq!(parsed.cd, [[0.0, -0.01], [0.02, 0.0]]);
    let (ra, dec) = parsed.pixel_to_sky(19.0, 14.0).unwrap();
    assert!((ra - 12.0).abs() < 1e-10 && (dec - 89.8).abs() < 1e-10);
    assert!(parsed.footprint([40, 30]).unwrap().iter().all(|(ra, dec)| {
        (0.0..360.0).contains(ra) && (-90.0..=90.0).contains(dec)
    }));
    {
        let mut file = fitsio::FitsFile::edit(&path).unwrap();
        file.primary_hdu().unwrap().write_key(&mut file, "A_ORDER", 2_i64).unwrap();
    }
    assert!(inspect_fits_header(&path).unwrap().wcs.is_none());
    {
        let mut file = fitsio::FitsFile::edit(&path).unwrap();
        file.primary_hdu().unwrap().write_key(&mut file, "CTYPE1", "RA---TAN-SIP").unwrap();
    }
    assert!(inspect_fits_header(&path).unwrap().wcs.is_none());
}

#[test]
fn tan_wcs_rejects_singular_cd_and_sip_projection_independently() {
    use photonloom::fits::WcsTan;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bad.fits");
    let image = LinearImage::new(8, 8, 1);
    let good = WcsTan {
        crval: [360.0, -45.0], crpix: [4.0, 4.0],
        cd: [[0.01, 0.0], [0.0, 0.01]],
    };
    let singular = WcsTan { cd: [[0.01, 0.02], [0.02, 0.04]], ..good };
    assert!(write_fits_linear(&path, &image, "Ha", "bad", 1.0, 1, 0, Some(&singular)).is_err());
    write_fits_linear(&path, &image, "Ha", "good", 1.0, 1, 0, Some(&good)).unwrap();
    assert_eq!(inspect_fits_header(&path).unwrap().wcs, Some(good));
    assert!(good.pixel_to_sky(3.0, 3.0).unwrap().0 < 1e-10);
    write_fits_linear(&path, &image, "Ha", "sip", 1.0, 1, 0, None).unwrap();
    {
        let mut file = fitsio::FitsFile::edit(&path).unwrap();
        let hdu = file.primary_hdu().unwrap();
        for (key, value) in [
            ("CRVAL1", 0.0), ("CRVAL2", -45.0), ("CRPIX1", 4.0), ("CRPIX2", 4.0),
            ("CD1_1", 0.01), ("CD1_2", 0.0), ("CD2_1", 0.0), ("CD2_2", 0.01),
        ] { hdu.write_key(&mut file, key, value).unwrap(); }
        hdu.write_key(&mut file, "CTYPE1", "RA---TAN-SIP").unwrap();
        hdu.write_key(&mut file, "CTYPE2", "DEC--TAN").unwrap();
    }
    assert!(inspect_fits_header(&path).unwrap().wcs.is_none());
}
