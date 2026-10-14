//! Conservative display-only crop geometry in pixel-center coordinates.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use crate::{config::LinearImage, register::AffineTransform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CropRect {
    pub original_width: usize,
    pub original_height: usize,
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

fn canvas(width: usize, height: usize) -> Vec<(f64, f64)> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let x = (width - 1) as f64;
    let y = (height - 1) as f64;
    vec![(0.0, 0.0), (x, 0.0), (x, y), (0.0, y)]
}

fn finite_transform(tf: &AffineTransform) -> bool {
    [tf.a00, tf.a01, tf.tx, tf.a10, tf.a11, tf.ty]
        .iter()
        .all(|v| v.is_finite())
}

// Clip a convex polygon to a*x + b*y <= c. Intersections are evaluated in f64;
// input transforms remain f32 because that is the registration representation.
fn clip(poly: &[(f64, f64)], a: f64, b: f64, c: f64) -> Vec<(f64, f64)> {
    if poly.is_empty() || ![a, b, c].iter().all(|v| v.is_finite()) {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 1);
    let mut prev = *poly.last().unwrap();
    let mut prev_d = a * prev.0 + b * prev.1 - c;
    for &point in poly {
        let d = a * point.0 + b * point.1 - c;
        if !d.is_finite() || !prev_d.is_finite() {
            return Vec::new();
        }
        if (prev_d <= 0.0) != (d <= 0.0) {
            let t = prev_d / (prev_d - d);
            let cross = (
                prev.0 + t * (point.0 - prev.0),
                prev.1 + t * (point.1 - prev.1),
            );
            if !cross.0.is_finite() || !cross.1.is_finite() {
                return Vec::new();
            }
            out.push(cross);
        }
        if d <= 0.0 {
            out.push(point);
        }
        prev = point;
        prev_d = d;
    }
    out
}

fn clip_canvas(mut poly: Vec<(f64, f64)>, width: usize, height: usize) -> Vec<(f64, f64)> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    for (a, b, c) in [
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, (width - 1) as f64),
        (0.0, 1.0, (height - 1) as f64),
    ] {
        poly = clip(&poly, a, b, c);
    }
    poly
}

/// Common fully supported footprint of registered frames. The identity path
/// copies pixels without interpolation; every other transform reserves the full
/// three-pixel Lanczos support on each source edge.
pub fn frame_region(
    width: usize,
    height: usize,
    transforms: impl IntoIterator<Item = AffineTransform>,
) -> Vec<(f64, f64)> {
    let mut region = canvas(width, height);
    if region.is_empty() {
        return region;
    }
    for tf in transforms {
        if !finite_transform(&tf) {
            return Vec::new();
        }
        let margin = if tf.is_identity() { 0.0 } else { 3.0 };
        let max_x = (width - 1) as f64 - margin;
        let max_y = (height - 1) as f64 - margin;
        if max_x < margin || max_y < margin {
            return Vec::new();
        }
        for (a, b, c) in [
            (-(tf.a00 as f64), -(tf.a01 as f64), tf.tx as f64 - margin),
            (tf.a00 as f64, tf.a01 as f64, max_x - tf.tx as f64),
            (-(tf.a10 as f64), -(tf.a11 as f64), tf.ty as f64 - margin),
            (tf.a10 as f64, tf.a11 as f64, max_y - tf.ty as f64),
        ] {
            region = clip(&region, a, b, c);
            if region.is_empty() {
                return region;
            }
        }
    }
    region
}

/// Map a source footprint into the target grid through the inverse of the
/// target-to-source affine. Singular and nonfinite transforms have no footprint.
pub fn warp_region(
    source_region: &[(f64, f64)],
    target_to_source: &AffineTransform,
    width: usize,
    height: usize,
) -> Vec<(f64, f64)> {
    if source_region.is_empty() || !finite_transform(target_to_source) {
        return Vec::new();
    }
    if target_to_source.is_identity() {
        return clip_canvas(source_region.to_vec(), width, height);
    }
    // A non-identity filter alignment samples a Lanczos-3 neighbourhood in
    // the source filter. Erode its footprint by that support before mapping.
    // A half-plane a*x+b*y<=c is safe for the entire 3px square only when
    // a*x+b*y<=c-3*(|a|+|b|).
    let signed_area: f64 = source_region
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| {
            let (nx, ny) = source_region[(i + 1) % source_region.len()];
            x * ny - nx * y
        })
        .sum();
    if !signed_area.is_finite() || signed_area == 0.0 {
        return Vec::new();
    }
    let sign = signed_area.signum();
    let mut supported = source_region.to_vec();
    for (i, &(x, y)) in source_region.iter().enumerate() {
        let (nx, ny) = source_region[(i + 1) % source_region.len()];
        let (a, b) = (sign * (ny - y), sign * (x - nx));
        supported = clip(&supported, a, b, a * x + b * y - 3.0 * (a.abs() + b.abs()));
        if supported.is_empty() {
            return supported;
        }
    }
    let tf = target_to_source;
    let (a, b, c, d) = (tf.a00 as f64, tf.a01 as f64, tf.a10 as f64, tf.a11 as f64);
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant == 0.0 {
        return Vec::new();
    }
    let mut mapped = Vec::with_capacity(supported.len());
    for &(sx, sy) in &supported {
        let x = (d * (sx - tf.tx as f64) - b * (sy - tf.ty as f64)) / determinant;
        let y = (-c * (sx - tf.tx as f64) + a * (sy - tf.ty as f64)) / determinant;
        if !x.is_finite() || !y.is_finite() {
            return Vec::new();
        }
        mapped.push((x, y));
    }
    clip_canvas(mapped, width, height)
}

/// Intersect convex filter footprints, constrained to the output canvas.
/// An empty input footprint means no geometrically safe common crop.
pub fn intersect_regions<'a>(
    width: usize,
    height: usize,
    regions: impl IntoIterator<Item = &'a [(f64, f64)]>,
) -> Vec<(f64, f64)> {
    let mut result = canvas(width, height);
    for polygon in regions {
        if polygon.is_empty()
            || polygon
                .iter()
                .any(|&(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Vec::new();
        }
        // Accept either winding direction. A line segment is not a footprint.
        let area: f64 = polygon
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let next = polygon[(i + 1) % polygon.len()];
                x * next.1 - next.0 * y
            })
            .sum();
        if !area.is_finite() || area == 0.0 {
            return Vec::new();
        }
        let sign = if area > 0.0 { 1.0 } else { -1.0 };
        for (i, &(x, y)) in polygon.iter().enumerate() {
            let (nx, ny) = polygon[(i + 1) % polygon.len()];
            let (dx, dy) = (nx - x, ny - y);
            result = clip(&result, sign * dy, -sign * dx, sign * (dy * x - dx * y));
            if result.is_empty() {
                return result;
            }
        }
    }
    result
}

/// Maximum-area axis-aligned rectangle of integer pixel centers in a convex
/// footprint. Rasterize only one scanline at a time; the histogram and stack
/// need O(width) memory and the work is O(width * height).
pub fn largest_crop(width: usize, height: usize, region: &[(f64, f64)]) -> Option<CropRect> {
    if width == 0
        || height == 0
        || region.is_empty()
        || region
            .iter()
            .any(|&(x, y)| !x.is_finite() || !y.is_finite())
    {
        return None;
    }
    let mut heights = vec![0usize; width];
    let mut stack = Vec::<(usize, usize)>::with_capacity(width);
    let mut best: Option<CropRect> = None;
    let mut best_area = 0usize;
    for y in 0..height {
        let row = y as f64;
        let (mut left, mut right) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..region.len() {
            let (x0, y0) = region[i];
            let (x1, y1) = region[(i + 1) % region.len()];
            if y0 == y1 {
                if row == y0 {
                    left = left.min(x0.min(x1));
                    right = right.max(x0.max(x1));
                }
            } else if row >= y0.min(y1) && row <= y0.max(y1) {
                let x = x0 + (row - y0) / (y1 - y0) * (x1 - x0);
                left = left.min(x);
                right = right.max(x);
            }
        }
        let (start, end) = if left <= right {
            (
                left.max(0.0).ceil() as usize,
                right.min((width - 1) as f64).floor() as usize,
            )
        } else {
            (1, 0)
        };
        stack.clear();
        for x in 0..=width {
            let h = if x < width {
                if x >= start && x <= end {
                    heights[x] += 1;
                } else {
                    heights[x] = 0;
                }
                heights[x]
            } else {
                0
            };
            let mut from = x;
            while let Some(&(begin, old_h)) = stack.last() {
                if old_h <= h {
                    break;
                }
                stack.pop();
                from = begin;
                let w = x - begin;
                let area = w * old_h;
                let candidate = CropRect {
                    original_width: width,
                    original_height: height,
                    x: begin,
                    y: y + 1 - old_h,
                    width: w,
                    height: old_h,
                };
                if area > best_area
                    || (area == best_area
                        && best.is_none_or(|old| {
                            (candidate.y, candidate.x, candidate.height, candidate.width)
                                < (old.y, old.x, old.height, old.width)
                        }))
                {
                    best_area = area;
                    best = Some(candidate);
                }
            }
            if h > 0 && (stack.last().is_none_or(|&(_, previous)| previous < h)) {
                stack.push((from, h));
            }
        }
    }
    best
}

/// Crop planes and mask together, preserving exposure and the Bayer color phase
/// at the new pixel origin. Reject malformed rectangles before changing data.
pub fn crop_image(image: &mut LinearImage, crop: &CropRect) -> Result<()> {
    ensure!(
        crop.original_width == image.width && crop.original_height == image.height,
        "crop canvas does not match image dimensions"
    );
    ensure!(
        crop.width > 0
            && crop.height > 0
            && crop.x <= image.width
            && crop.y <= image.height
            && crop.width <= image.width - crop.x
            && crop.height <= image.height - crop.y,
        "crop rectangle is out of bounds"
    );
    let pixels = image
        .width
        .checked_mul(image.height)
        .ok_or_else(|| anyhow::anyhow!("image dimensions overflow"))?;
    let data_len = pixels
        .checked_mul(image.channels)
        .ok_or_else(|| anyhow::anyhow!("image planes overflow"))?;
    ensure!(
        image.mask.len() == pixels && image.data.len() == data_len,
        "malformed image planes"
    );
    if crop.x == 0 && crop.y == 0 && crop.width == image.width && crop.height == image.height {
        return Ok(());
    }
    let new_pixels = crop
        .width
        .checked_mul(crop.height)
        .ok_or_else(|| anyhow::anyhow!("crop dimensions overflow"))?;
    let mut data = Vec::with_capacity(
        new_pixels
            .checked_mul(image.channels)
            .ok_or_else(|| anyhow::anyhow!("crop planes overflow"))?,
    );
    for plane in image.data.chunks_exact(pixels) {
        for row in crop.y..crop.y + crop.height {
            let start = row * image.width + crop.x;
            data.extend_from_slice(&plane[start..start + crop.width]);
        }
    }
    let mut mask = Vec::with_capacity(new_pixels);
    for row in crop.y..crop.y + crop.height {
        let start = row * image.width + crop.x;
        mask.extend_from_slice(&image.mask[start..start + crop.width]);
    }
    if let Some(mut pattern) = image.bayer {
        use crate::config::BayerPattern::*;
        if crop.x & 1 != 0 {
            pattern = match pattern {
                Rggb => Grbg,
                Bggr => Gbrg,
                Grbg => Rggb,
                Gbrg => Bggr,
            };
        }
        if crop.y & 1 != 0 {
            pattern = match pattern {
                Rggb => Gbrg,
                Bggr => Grbg,
                Grbg => Bggr,
                Gbrg => Rggb,
            };
        }
        image.bayer = Some(pattern);
    }
    image.data = data;
    image.mask = mask;
    image.width = crop.width;
    image.height = crop.height;
    Ok(())
}
