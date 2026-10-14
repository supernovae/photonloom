use photonloom::{
    config::{BayerPattern, LinearImage},
    crop::{CropRect, crop_image, frame_region, intersect_regions, largest_crop, warp_region},
    register::AffineTransform,
};

fn translation(tx: f32, ty: f32) -> AffineTransform {
    AffineTransform {
        tx,
        ty,
        ..AffineTransform::identity()
    }
}

#[test]
fn translated_frames_trim_only_unsupported_pixel_centers() {
    let region = frame_region(
        20,
        18,
        [AffineTransform::identity(), translation(2.0, -1.0)],
    );
    assert_eq!(
        largest_crop(20, 18, &region),
        Some(CropRect {
            original_width: 20,
            original_height: 18,
            x: 1,
            y: 4,
            width: 14,
            height: 12,
        })
    );
    assert_eq!(
        largest_crop(20, 18, &frame_region(20, 18, [AffineTransform::identity()])),
        Some(CropRect {
            original_width: 20,
            original_height: 18,
            x: 0,
            y: 0,
            width: 20,
            height: 18
        })
    );
}

#[test]
fn rotated_footprint_finds_maximum_rectangle_not_uniform_border() {
    let theta = std::f32::consts::FRAC_PI_4;
    let (s, c) = theta.sin_cos();
    let tf = AffineTransform {
        a00: c,
        a01: -s,
        a10: s,
        a11: c,
        tx: 20.0 * (1.0 - c + s),
        ty: 20.0 * (1.0 - s - c),
    };
    let footprint = frame_region(41, 41, [tf]);
    let crop = largest_crop(41, 41, &footprint).expect("rotation leaves a center footprint");
    let inside = |x: usize, y: usize| {
        let x = x as f64;
        let y = y as f64;
        let sx = tf.a00 as f64 * x + tf.a01 as f64 * y + tf.tx as f64;
        let sy = tf.a10 as f64 * x + tf.a11 as f64 * y + tf.ty as f64;
        (3.0..=37.0).contains(&sx) && (3.0..=37.0).contains(&sy)
    };
    for y in crop.y..crop.y + crop.height {
        for x in crop.x..crop.x + crop.width {
            assert!(inside(x, y), "unsupported pixel ({x},{y}) in {crop:?}");
        }
    }
    // Exhaustive independent rectangle search on the small test canvas.
    let mut optimum = 0;
    for y0 in 0..41 {
        for y1 in y0..41 {
            for x0 in 0..41 {
                for x1 in x0..41 {
                    if [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
                        .into_iter()
                        .all(|(x, y)| inside(x, y))
                    {
                        optimum = optimum.max((x1 - x0 + 1) * (y1 - y0 + 1));
                    }
                }
            }
        }
    }
    assert_eq!(crop.width * crop.height, optimum);
    assert!(crop.width < 35 && crop.height < 35);
}

#[test]
fn separately_aligned_filters_share_only_their_common_supported_footprint() {
    let red = frame_region(30, 20, [translation(2.0, 0.0)]);
    let blue = frame_region(30, 20, [translation(-2.0, 0.0)]);
    let blue_in_red = warp_region(&blue, &translation(1.0, 0.0), 30, 20);
    let common = intersect_regions(30, 20, [&red[..], &blue_in_red[..]]);
    assert_eq!(
        largest_crop(30, 20, &common),
        Some(CropRect {
            original_width: 30,
            original_height: 20,
            x: 7,
            y: 6,
            width: 18,
            height: 8,
        })
    );
    let singular = AffineTransform {
        a00: 0.0,
        a11: 0.0,
        ..AffineTransform::identity()
    };
    assert!(warp_region(&blue, &singular, 30, 20).is_empty());
    assert!(largest_crop(30, 20, &intersect_regions(30, 20, [&red[..], &[][..]])).is_none());
}

#[test]
fn interior_dead_pixels_do_not_change_geometric_crop() {
    let mut image = LinearImage::new(20, 18, 1);
    image.mask[9 * 20 + 10] = 0;
    let footprint = frame_region(image.width, image.height, [translation(2.0, -1.0)]);
    let crop = largest_crop(image.width, image.height, &footprint).unwrap();
    assert!(10 >= crop.x && 10 < crop.x + crop.width && 9 >= crop.y && 9 < crop.y + crop.height);
    crop_image(&mut image, &crop).unwrap();
    assert_eq!(image.mask[(9 - crop.y) * crop.width + 10 - crop.x], 0);
}

#[test]
fn crop_reindexes_every_plane_and_mask_without_changing_exposure() {
    let mut image = LinearImage::new(7, 5, 3);
    image.exposure = 123.5;
    image.mask[2 * 7 + 3] = 0;
    for c in 0..3 {
        for y in 0..5 {
            for x in 0..7 {
                image.set(c, x, y, (c * 100 + y * 10 + x) as f32);
            }
        }
    }
    let crop = CropRect {
        original_width: 7,
        original_height: 5,
        x: 2,
        y: 1,
        width: 3,
        height: 3,
    };
    crop_image(&mut image, &crop).unwrap();
    assert_eq!(
        (image.width, image.height, image.channels, image.exposure),
        (3, 3, 3, 123.5)
    );
    for c in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(image.get(c, x, y), (c * 100 + (y + 1) * 10 + x + 2) as f32);
            }
        }
    }
    assert_eq!(image.mask[1 * 3 + 1], 0);
    assert_eq!(image.mask.iter().filter(|&&v| v == 0).count(), 1);
}

#[test]
fn invalid_crop_fails_without_mutating_image() {
    let mut image = LinearImage::new(7, 5, 2);
    let original = image.clone();
    for crop in [
        CropRect {
            original_width: 8,
            original_height: 5,
            x: 0,
            y: 0,
            width: 7,
            height: 5,
        },
        CropRect {
            original_width: 7,
            original_height: 5,
            x: 6,
            y: 2,
            width: 2,
            height: 2,
        },
        CropRect {
            original_width: 7,
            original_height: 5,
            x: usize::MAX,
            y: 0,
            width: 1,
            height: 1,
        },
        CropRect {
            original_width: 7,
            original_height: 5,
            x: 0,
            y: 0,
            width: 0,
            height: 5,
        },
    ] {
        assert!(crop_image(&mut image, &crop).is_err());
        assert_eq!(image.data, original.data);
        assert_eq!(image.mask, original.mask);
        assert_eq!(
            (image.width, image.height),
            (original.width, original.height)
        );
    }
}

#[test]
fn cropped_mosaic_keeps_each_pixels_original_filter_color() {
    for original in [
        BayerPattern::Rggb,
        BayerPattern::Bggr,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
    ] {
        for x_offset in 0..2 {
            for y_offset in 0..2 {
                let mut mosaic = LinearImage::new(8, 8, 1);
                mosaic.bayer = Some(original);
                let rectangle = CropRect {
                    original_width: 8,
                    original_height: 8,
                    x: x_offset,
                    y: y_offset,
                    width: 6,
                    height: 6,
                };
                crop_image(&mut mosaic, &rectangle).unwrap();
                let phase = mosaic.bayer.unwrap();
                for y in 0..6 {
                    for x in 0..6 {
                        assert_eq!(
                            phase.color_at(x, y),
                            original.color_at(x + x_offset, y + y_offset)
                        );
                    }
                }
            }
        }
    }
}
