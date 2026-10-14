use photonloom::register::{AffineTransform, DetectedStar, match_star_fields};

const WIDTH: usize = 2400;
const HEIGHT: usize = 1800;

fn star(x: f32, y: f32, flux: f32) -> DetectedStar {
    DetectedStar {
        x,
        y,
        flux,
        peak: flux,
        fwhm: 3.0,
        eccentricity: 0.1,
    }
}

// Non-lattice coordinates, distributed across the full frame. Sorting by flux
// emulates the detector's brightness ordering, not a correspondence ordering.
fn field(seed: u32) -> Vec<DetectedStar> {
    let mut stars: Vec<_> = (0..34)
        .map(|i| {
            let n = i as u32;
            let x = 130.0
                + ((n * (683 + seed * 19) + n * n * (71 + seed * 13) + seed * 127) % 2090) as f32;
            let y = 110.0
                + ((n * (431 + seed * 23) + n * n * (193 + seed * 17) + seed * 89) % 1560) as f32;
            star(x, y, (34 - i) as f32)
        })
        .collect();
    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    stars
}

fn assert_transform(actual: AffineTransform, expected: AffineTransform) {
    for (x, y) in [(200.0, 180.0), (1200.0, 900.0), (2150.0, 1600.0)] {
        let (ax, ay) = actual.ref_to_src(x, y);
        let (ex, ey) = expected.ref_to_src(x, y);
        assert!(
            (ax - ex).hypot(ay - ey) < 0.5,
            "wrong transform at ({x}, {y}): {actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn recovers_field_with_missing_brightest_star_and_reordered_brightness() {
    let reference = field(0);
    let expected = AffineTransform {
        a00: 0.996,
        a01: -0.087,
        tx: 93.0,
        a10: 0.087,
        a11: 0.996,
        ty: -56.0,
    };
    let mut source: Vec<_> = reference
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 0 && *i != 2 && *i != 5 && *i != 8)
        .map(|(i, s)| {
            let (x, y) = expected.ref_to_src(s.x, s.y);
            // Subpixel centroid changes and drastic brightness reordering.
            star(
                x + (i % 3) as f32 * 0.06 - 0.06,
                y + (i % 5) as f32 * 0.04 - 0.08,
                ((i * 17) % 37 + 1) as f32,
            )
        })
        .collect();
    source.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    let matched =
        match_star_fields(&reference, &source, WIDTH, HEIGHT).expect("overlapping star field");
    assert!(matched.inliers >= 25, "{matched:?}");
    assert!(matched.rms < 0.2, "{matched:?}");
    assert_transform(matched.transform, expected);
}

#[test]
fn recovers_meridian_flip_with_missing_bright_stars() {
    let reference = field(0);
    let expected = AffineTransform {
        a00: -1.0,
        a01: 0.0,
        tx: 2420.0,
        a10: 0.0,
        a11: -1.0,
        ty: 1805.0,
    };
    let mut source: Vec<_> = reference
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 0 && *i != 1 && *i != 4)
        .map(|(i, s)| {
            let (x, y) = expected.ref_to_src(s.x, s.y);
            star(
                x + (i % 3) as f32 * 0.04,
                y - (i % 4) as f32 * 0.04,
                ((i * 23) % 41 + 1) as f32,
            )
        })
        .collect();
    source.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    let matched =
        match_star_fields(&reference, &source, WIDTH, HEIGHT).expect("rotated star field");
    assert!(matched.inliers >= 25, "{matched:?}");
    assert!(matched.rms < 0.2, "{matched:?}");
    assert_transform(matched.transform, expected);
}

#[test]
fn unrelated_field_is_not_registered() {
    let reference = field(0);
    let other = field(17);
    assert!(match_star_fields(&reference, &other, WIDTH, HEIGHT).is_err());
}
