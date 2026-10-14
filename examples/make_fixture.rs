use anyhow::{Context, Result};
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};
use std::path::{Path, PathBuf};

/// 18 deterministic unsaturated stars in normalized coordinates `(u, v) in [0.16, 0.84]` and relative flux.
pub const FIXTURE_STARS: [(f32, f32, f32); 18] = [
    (0.20, 0.22, 1.00),
    (0.36, 0.19, 0.85),
    (0.55, 0.23, 0.92),
    (0.73, 0.20, 0.78),
    (0.25, 0.36, 0.88),
    (0.44, 0.34, 0.95),
    (0.64, 0.38, 0.82),
    (0.80, 0.35, 0.74),
    (0.19, 0.52, 0.90),
    (0.38, 0.50, 0.86),
    (0.58, 0.53, 0.96),
    (0.76, 0.49, 0.80),
    (0.23, 0.68, 0.84),
    (0.42, 0.66, 0.91),
    (0.61, 0.70, 0.77),
    (0.79, 0.67, 0.89),
    (0.31, 0.81, 0.83),
    (0.68, 0.82, 0.87),
];

fn deterministic_noise(seed: u64, x: usize, y: usize) -> f32 {
    let mut z = seed
        .wrapping_add((x as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add((y as u64).wrapping_mul(0xBF58476D1CE4E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let u = ((z & 0xFFFFFF) as f32) / (0xFFFFFF as f32);
    (u - 0.5) * 1.6
}

fn write_fits_2d(
    path: &Path,
    width: usize,
    height: usize,
    pixels: &[f32],
    imagetyp: &str,
    filter: Option<&str>,
    bayerpat: Option<&str>,
    exptime: f32,
    ccd_temp: f32,
    gain: f32,
    object: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let dims = [height, width];
    let desc = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &dims,
    };
    let mut fits = FitsFile::create(path)
        .with_custom_primary(&desc)
        .overwrite()
        .open()
        .with_context(|| format!("Failed to create FITS '{}'", path.display()))?;
    let hdu = fits.primary_hdu()?;
    hdu.write_key(&mut fits, "IMAGETYP", imagetyp)?;
    if let Some(f) = filter {
        hdu.write_key(&mut fits, "FILTER", f)?;
    }
    if let Some(bp) = bayerpat {
        hdu.write_key(&mut fits, "BAYERPAT", bp)?;
    }
    hdu.write_key(&mut fits, "EXPTIME", f64::from(exptime))?;
    hdu.write_key(&mut fits, "CCD-TEMP", f64::from(ccd_temp))?;
    hdu.write_key(&mut fits, "GAIN", f64::from(gain))?;
    hdu.write_key(&mut fits, "INSTRUME", "PhotonloomCam")?;
    hdu.write_key(&mut fits, "OBJECT", object)?;
    hdu.write_image(&mut fits, pixels)?;
    Ok(())
}

fn build_vignette_flat(width: usize, height: usize, bias: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    let cx = (width as f32) * 0.5;
    let cy = (height as f32) * 0.5;
    let r_max2 = cx * cx + cy * cy;
    for y in 0..height {
        for x in 0..width {
            let dx = (x as f32) - cx;
            let dy = (y as f32) - cy;
            let vig = 1.0 - 0.12 * ((dx * dx + dy * dy) / r_max2);
            out[y * width + x] = bias + 10000.0 * vig;
        }
    }
    out
}

fn build_dark_frame(width: usize, height: usize, bias: f32, dark_current: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let pattern = ((x * 7 + y * 13) % 11) as f32 * 0.2;
            out[y * width + x] = bias + dark_current + pattern;
        }
    }
    out
}

/// Synthesizes a clean signal plane (stars + smooth sky background + faint emission nebula)
/// shifted by `(dx_shift, dy_shift)` in source coordinates (`x_src = x_ref + dx_shift`).
#[allow(clippy::too_many_arguments)]
pub fn synthesize_light_plane(
    width: usize,
    height: usize,
    dx_shift: f32,
    dy_shift: f32,
    channel_scale: f32,
    add_hot_pixel: bool,
    add_streak: bool,
    seed: u64,
) -> Vec<f32> {
    let w_f = width as f32;
    let h_f = height as f32;
    let sigma = 1.15f32;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);

    // Precompute star positions in this frame: `x_src = u * w + dx_shift`, `y_src = v * h + dy_shift`
    let star_coords: Vec<(f32, f32, f32)> = FIXTURE_STARS
        .iter()
        .map(|&(u, v, flux)| {
            (
                u * w_f + dx_shift,
                v * h_f + dy_shift,
                flux * 4500.0 * channel_scale,
            )
        })
        .collect();

    let mut img = vec![0.0f32; width * height];
    let neb_cx = 0.50 * w_f + dx_shift;
    let neb_cy = 0.50 * h_f + dy_shift;
    let neb_r2 = (0.22 * w_f).powi(2);

    for y in 0..height {
        let yf = y as f32;
        for x in 0..width {
            let xf = x as f32;
            // Smooth sky background gradient
            let sky = 50.0 + 12.0 * (xf / w_f) + 8.0 * (yf / h_f);
            // Soft central nebulosity
            let ndx = xf - neb_cx;
            let ndy = yf - neb_cy;
            let neb = 80.0 * channel_scale * (-(ndx * ndx + ndy * ndy) / (2.0 * neb_r2)).exp();

            let mut star_val = 0.0f32;
            for &(sx, sy, peak) in &star_coords {
                let dx = xf - sx;
                let dy = yf - sy;
                let d2 = dx * dx + dy * dy;
                if d2 <= 36.0 {
                    star_val += peak * (-d2 * inv_2s2).exp();
                }
            }

            let noise = deterministic_noise(seed, x, y);
            img[y * width + x] = sky + neb + star_val + noise;
        }
    }

    if add_hot_pixel && width > 32 && height > 32 {
        img[27 * width + 31] += 28000.0;
    }

    if add_streak && height > 50 {
        let sy = (height * 3) / 4;
        for x in (width / 6)..((width * 5) / 6) {
            img[sy * width + x] += 6500.0;
        }
    }

    img
}

fn apply_raw_sensor_effects(
    clean_signal: &[f32],
    dark: &[f32],
    flat: &[f32],
    bias: f32,
) -> Vec<f32> {
    let mut flat_net: Vec<f32> = flat.iter().map(|&f| f - bias).collect();
    flat_net.sort_by(|a, b| a.total_cmp(b));
    let flat_norm_base = flat_net[flat_net.len() / 2].max(1.0);
    clean_signal
        .iter()
        .enumerate()
        .map(|(i, &sig)| {
            let vig = (flat[i] - bias) / flat_norm_base;
            sig * vig + dark[i]
        })
        .collect()
}

fn build_rggb_mosaic_from_clean(
    width: usize,
    height: usize,
    dx_shift: f32,
    dy_shift: f32,
    add_hot_pixel: bool,
    add_streak: bool,
    seed: u64,
) -> Vec<f32> {
    let r = synthesize_light_plane(
        width, height, dx_shift, dy_shift, 1.05, false, add_streak, seed,
    );
    let g = synthesize_light_plane(
        width,
        height,
        dx_shift,
        dy_shift,
        1.00,
        false,
        add_streak,
        seed + 1,
    );
    let b = synthesize_light_plane(
        width,
        height,
        dx_shift,
        dy_shift,
        0.95,
        false,
        add_streak,
        seed + 2,
    );

    let mut mosaic = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let p = y * width + x;
            let c = match (y & 1, x & 1) {
                (0, 0) => r[p],
                (0, 1) | (1, 0) => g[p],
                _ => b[p],
            };
            mosaic[p] = c;
        }
    }
    if add_hot_pixel && width > 32 && height > 32 {
        mosaic[27 * width + 31] += 28000.0;
    }
    mosaic
}

/// Writes a minimal valid uncompressed 16-bit RGGB Bayer DNG file readable by LibRaw.
fn write_minimal_bayer_dng(
    path: &Path,
    width: u32,
    height: u32,
    mosaic: &[f32],
    black: u16,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let make_bytes = b"Photonloom\0\0"; // 12 bytes
    let model_bytes = b"SyntheticDNG\0\0"; // 14 bytes

    // TIFF header (8 bytes) + IFD0 (2 + 16*12 + 4 = 198 bytes) + extra values (26 bytes) = 232 bytes
    let num_entries: u16 = 16;
    let ifd_offset: u32 = 8;
    let extra_offset: u32 = ifd_offset + 2 + u32::from(num_entries) * 12 + 4;
    let make_offset = extra_offset;
    let model_offset = make_offset + make_bytes.len() as u32;
    let pixel_offset = (model_offset + model_bytes.len() as u32 + 15) & !15;
    let byte_count = width * height * 2;

    let mut buf = Vec::with_capacity(pixel_offset as usize + byte_count as usize);
    // Little-endian TIFF header
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&ifd_offset.to_le_bytes());

    buf.extend_from_slice(&num_entries.to_le_bytes());

    let mut push_tag = |tag: u16, field_type: u16, count: u32, val_or_offset: u32| {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&field_type.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&val_or_offset.to_le_bytes());
    };

    // Tags must be in strictly ascending order
    push_tag(254, 4, 1, 0); // NewSubfileType = 0 (Full-res RAW)
    push_tag(256, 4, 1, width); // ImageWidth
    push_tag(257, 4, 1, height); // ImageLength
    push_tag(258, 3, 1, 16); // BitsPerSample = 16
    push_tag(259, 3, 1, 1); // Compression = 1 (Uncompressed)
    push_tag(262, 3, 1, 32803); // PhotometricInterpretation = CFA (32803)
    push_tag(271, 2, make_bytes.len() as u32, make_offset); // Make
    push_tag(272, 2, model_bytes.len() as u32, model_offset); // Model
    push_tag(273, 4, 1, pixel_offset); // StripOffsets
    push_tag(277, 3, 1, 1); // SamplesPerPixel = 1
    push_tag(278, 4, 1, height); // RowsPerStrip
    push_tag(279, 4, 1, byte_count); // StripByteCounts
    push_tag(284, 3, 1, 1); // PlanarConfiguration = 1
    push_tag(33421, 3, 2, 0x0002_0002); // CFARepeatPatternDim = [2, 2]
    push_tag(33422, 1, 4, u32::from_le_bytes([0, 1, 1, 2])); // CFAPattern = RGGB
    push_tag(50706, 1, 4, u32::from_le_bytes([1, 4, 0, 0])); // DNGVersion = 1.4.0.0

    // Note: BlackLevel is set via DNG baseline or we can include 50714 & 50717 if we bump num_entries to 18!
    // Let's include 50714 (BlackLevel) and 50717 (WhiteLevel) by rebuilding with 18 entries:
    buf.clear();
    let num_entries: u16 = 18;
    let extra_offset: u32 = ifd_offset + 2 + u32::from(num_entries) * 12 + 4;
    let make_offset = extra_offset;
    let model_offset = make_offset + make_bytes.len() as u32;
    let pixel_offset = (model_offset + model_bytes.len() as u32 + 15) & !15;

    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&ifd_offset.to_le_bytes());
    buf.extend_from_slice(&num_entries.to_le_bytes());

    let mut push_tag = |tag: u16, field_type: u16, count: u32, val_or_offset: u32| {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&field_type.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&val_or_offset.to_le_bytes());
    };

    push_tag(254, 4, 1, 0);
    push_tag(256, 4, 1, width);
    push_tag(257, 4, 1, height);
    push_tag(258, 3, 1, 16);
    push_tag(259, 3, 1, 1);
    push_tag(262, 3, 1, 32803);
    push_tag(271, 2, make_bytes.len() as u32, make_offset);
    push_tag(272, 2, model_bytes.len() as u32, model_offset);
    push_tag(273, 4, 1, pixel_offset);
    push_tag(277, 3, 1, 1);
    push_tag(278, 4, 1, height);
    push_tag(279, 4, 1, byte_count);
    push_tag(284, 3, 1, 1);
    push_tag(33421, 3, 2, 0x0002_0002);
    push_tag(33422, 1, 4, u32::from_le_bytes([0, 1, 1, 2]));
    push_tag(50706, 1, 4, u32::from_le_bytes([1, 4, 0, 0]));
    push_tag(50714, 3, 1, u32::from(black)); // BlackLevel
    push_tag(50717, 3, 1, 65000); // WhiteLevel

    // Next IFD offset = 0
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(make_bytes);
    buf.extend_from_slice(model_bytes);
    while (buf.len() as u32) < pixel_offset {
        buf.push(0);
    }

    for &v in mosaic {
        let u16_val = (v + f32::from(black)).round().clamp(0.0, 65000.0) as u16;
        buf.extend_from_slice(&u16_val.to_le_bytes());
    }

    std::fs::write(path, &buf)?;
    Ok(())
}

pub fn generate_full_fixture(
    root: &Path,
    width: usize,
    height: usize,
    frames_per_ch: usize,
) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    std::fs::create_dir_all(root)?;

    let bias_val = 100.0f32;
    let dark_current = 25.0f32;
    let bias = vec![bias_val; width * height];
    let dark = build_dark_frame(width, height, bias_val, dark_current);
    let flat = build_vignette_flat(width, height, bias_val);

    // Write FITS calibration masters (mono and RGB CFA)
    write_fits_2d(
        &root.join("calibration/bias_master.fits"),
        width,
        height,
        &bias,
        "Bias Frame",
        None,
        None,
        0.0,
        -10.0,
        100.0,
        "Calibration",
    )?;
    write_fits_2d(
        &root.join("calibration/dark_master.fits"),
        width,
        height,
        &dark,
        "Dark Frame",
        None,
        None,
        120.0,
        -10.0,
        100.0,
        "Calibration",
    )?;

    for (ch_idx, filter) in ["Ha", "OIII", "SII"].iter().enumerate() {
        write_fits_2d(
            &root.join(format!("calibration/flat_{filter}.fits")),
            width,
            height,
            &flat,
            "Flat Frame",
            Some(filter),
            None,
            1.5,
            -10.0,
            100.0,
            "Calibration",
        )?;

        let ch_scale = match *filter {
            "Ha" => 1.2,
            "OIII" => 0.95,
            _ => 0.85,
        };

        for f_idx in 0..frames_per_ch {
            // Frame index 1 (2nd frame) has known (+3.0, -2.0) px dither
            let (dx, dy) = match f_idx {
                0 => (0.0, 0.0),
                1 => (3.0, -2.0),
                2 => (-1.0, 1.0),
                k => (((k % 3) as f32) - 1.0, ((k % 2) as f32) - 0.5),
            };
            let add_hot = f_idx == 1;
            let add_streak = f_idx == 2;
            let seed = ((ch_idx as u64) + 1) * 100 + (f_idx as u64);
            let clean =
                synthesize_light_plane(width, height, dx, dy, ch_scale, add_hot, add_streak, seed);
            let raw_pixels = apply_raw_sensor_effects(&clean, &dark, &flat, bias_val);
            write_fits_2d(
                &root.join(format!(
                    "lights/{filter}/light_{filter}_{:02}.fits",
                    f_idx + 1
                )),
                width,
                height,
                &raw_pixels,
                "Light Frame",
                Some(filter),
                None,
                120.0,
                -10.0,
                100.0,
                "Rosette Nebula",
            )?;
        }
    }

    // Also include RGGB Bayer RGB lights and calibration in the full fixture
    write_fits_2d(
        &root.join("calibration/bias_rgb.fits"),
        width,
        height,
        &bias,
        "Bias Frame",
        None,
        Some("RGGB"),
        0.0,
        -10.0,
        100.0,
        "Calibration",
    )?;
    write_fits_2d(
        &root.join("calibration/dark_rgb.fits"),
        width,
        height,
        &dark,
        "Dark Frame",
        None,
        Some("RGGB"),
        120.0,
        -10.0,
        100.0,
        "Calibration",
    )?;
    write_fits_2d(
        &root.join("calibration/flat_rgb.fits"),
        width,
        height,
        &flat,
        "Flat Frame",
        Some("RGB"),
        Some("RGGB"),
        1.5,
        -10.0,
        100.0,
        "Calibration",
    )?;

    for f_idx in 0..frames_per_ch {
        let (dx, dy) = match f_idx {
            0 => (0.0, 0.0),
            1 => (3.0, -2.0),
            2 => (-1.0, 1.0),
            k => (((k % 3) as f32) - 1.0, 0.0),
        };
        let clean_mosaic = build_rggb_mosaic_from_clean(
            width,
            height,
            dx,
            dy,
            f_idx == 1,
            f_idx == 2,
            900 + f_idx as u64,
        );
        let raw_mosaic = apply_raw_sensor_effects(&clean_mosaic, &dark, &flat, bias_val);
        write_fits_2d(
            &root.join(format!("lights/RGB/light_rgb_{:02}.fits", f_idx + 1)),
            width,
            height,
            &raw_mosaic,
            "Light Frame",
            Some("RGB"),
            Some("RGGB"),
            120.0,
            -10.0,
            100.0,
            "Rosette Nebula",
        )?;
    }

    Ok(())
}

pub fn generate_stress_fixture(
    root: &Path,
    width: usize,
    height: usize,
    frames: usize,
) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    std::fs::create_dir_all(root)?;

    let bias_val = 100.0f32;
    let dark_current = 20.0f32;
    let bias = vec![bias_val; width * height];
    let dark = build_dark_frame(width, height, bias_val, dark_current);
    let flat = build_vignette_flat(width, height, bias_val);

    write_fits_2d(
        &root.join("bias.fits"),
        width,
        height,
        &bias,
        "Bias Frame",
        None,
        None,
        0.0,
        -10.0,
        100.0,
        "Calibration",
    )?;
    write_fits_2d(
        &root.join("dark.fits"),
        width,
        height,
        &dark,
        "Dark Frame",
        None,
        None,
        120.0,
        -10.0,
        100.0,
        "Calibration",
    )?;
    write_fits_2d(
        &root.join("flat_Ha.fits"),
        width,
        height,
        &flat,
        "Flat Frame",
        Some("Ha"),
        None,
        1.5,
        -10.0,
        100.0,
        "Calibration",
    )?;

    for f_idx in 0..frames {
        let (dx, dy) = match f_idx {
            0 => (0.0, 0.0),
            1 => (3.0, -2.0),
            k => (((k % 3) as f32) - 1.0, ((k % 2) as f32)),
        };
        let clean = synthesize_light_plane(
            width,
            height,
            dx,
            dy,
            1.0,
            f_idx == 1,
            f_idx == 2,
            5000 + f_idx as u64,
        );
        let raw_pixels = apply_raw_sensor_effects(&clean, &dark, &flat, bias_val);
        write_fits_2d(
            &root.join(format!("light_Ha_{:02}.fits", f_idx + 1)),
            width,
            height,
            &raw_pixels,
            "Light Frame",
            Some("Ha"),
            None,
            120.0,
            -10.0,
            100.0,
            "Cygnus Wall",
        )?;
    }

    Ok(())
}

pub fn generate_rgb_only_fixture(root: &Path, width: usize, height: usize) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    std::fs::create_dir_all(root)?;

    let bias_val = 100.0f32;
    let dark = build_dark_frame(width, height, bias_val, 25.0);
    let flat = build_vignette_flat(width, height, bias_val);

    write_fits_2d(
        &root.join("dark_rgb.fits"),
        width,
        height,
        &dark,
        "Dark Frame",
        None,
        Some("RGGB"),
        120.0,
        -10.0,
        100.0,
        "Calibration",
    )?;
    write_fits_2d(
        &root.join("flat_rgb.fits"),
        width,
        height,
        &flat,
        "Flat Frame",
        Some("RGB"),
        Some("RGGB"),
        1.5,
        -10.0,
        100.0,
        "Calibration",
    )?;

    for f_idx in 0..3 {
        let (dx, dy) = match f_idx {
            0 => (0.0, 0.0),
            1 => (3.0, -2.0),
            _ => (-1.0, 1.0),
        };
        let clean_mosaic = build_rggb_mosaic_from_clean(
            width,
            height,
            dx,
            dy,
            f_idx == 1,
            f_idx == 2,
            2000 + f_idx as u64,
        );
        let raw_mosaic = apply_raw_sensor_effects(&clean_mosaic, &dark, &flat, bias_val);
        write_fits_2d(
            &root.join(format!("light_rgb_{:02}.fits", f_idx + 1)),
            width,
            height,
            &raw_mosaic,
            "Light Frame",
            Some("RGB"),
            Some("RGGB"),
            120.0,
            -10.0,
            100.0,
            "Andromeda M31",
        )?;
    }
    Ok(())
}

pub fn generate_bad_dark_fixture(root: &Path, width: usize, height: usize) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    std::fs::create_dir_all(root)?;

    // Dark has mismatched temperature (+15 C vs -10 C) and gain (400 vs 100)
    let bad_dark = vec![5000.0f32; width * height];
    write_fits_2d(
        &root.join("dark_bad.fits"),
        width,
        height,
        &bad_dark,
        "Dark Frame",
        None,
        Some("RGGB"),
        120.0,
        15.0,
        400.0,
        "Calibration",
    )?;

    for f_idx in 0..3 {
        let (dx, dy) = if f_idx == 1 { (3.0, -2.0) } else { (0.0, 0.0) };
        let clean_mosaic =
            build_rggb_mosaic_from_clean(width, height, dx, dy, false, false, 3000 + f_idx as u64);
        write_fits_2d(
            &root.join(format!("light_rgb_{:02}.fits", f_idx + 1)),
            width,
            height,
            &clean_mosaic,
            "Light Frame",
            Some("RGB"),
            Some("RGGB"),
            120.0,
            -10.0,
            100.0,
            "Orion M42",
        )?;
    }
    Ok(())
}

pub fn generate_empty_fixture(root: &Path) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    std::fs::create_dir_all(root)?;
    Ok(())
}

pub fn generate_dng_fixture(root: &Path, width: usize, height: usize) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    std::fs::create_dir_all(root)?;

    for f_idx in 0..3 {
        let (dx, dy) = match f_idx {
            0 => (0.0, 0.0),
            1 => (3.0, -2.0),
            _ => (-1.0, 1.0),
        };
        let clean_mosaic = build_rggb_mosaic_from_clean(
            width,
            height,
            dx,
            dy,
            f_idx == 1,
            f_idx == 2,
            4000 + f_idx as u64,
        );
        write_minimal_bayer_dng(
            &root.join(format!("lights/light_{:02}.dng", f_idx + 1)),
            width as u32,
            height as u32,
            &clean_mosaic,
            100,
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let base_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/photonloom-fixture"));

    let mut custom_size: Option<usize> = None;
    let mut custom_frames: Option<usize> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                if let Some(v) = args.get(i + 1) {
                    custom_size = Some(v.parse().context("Invalid --size value")?);
                    i += 2;
                    continue;
                }
            }
            "--frames" => {
                if let Some(v) = args.get(i + 1) {
                    custom_frames = Some(v.parse().context("Invalid --frames value")?);
                    i += 2;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    if custom_size.is_some() || custom_frames.is_some() {
        let size = custom_size.unwrap_or(64);
        let frames = custom_frames.unwrap_or(3);
        generate_stress_fixture(&base_path, size, size, frames)?;
        println!(
            "Generated stress fixture at {} ({}x{}, {} frames)",
            base_path.display(),
            size,
            size,
            frames
        );
        return Ok(());
    }

    let size = 64usize;
    generate_full_fixture(&base_path, size, size, 3)?;

    let base_str = base_path.to_string_lossy();
    let rgb_path = PathBuf::from(format!("{base_str}-rgb"));
    let bad_dark_path = PathBuf::from(format!("{base_str}-bad-dark"));
    let empty_path = PathBuf::from(format!("{base_str}-empty"));
    let dng_path = PathBuf::from(format!("{base_str}-dng"));

    generate_rgb_only_fixture(&rgb_path, size, size)?;
    generate_bad_dark_fixture(&bad_dark_path, size, size)?;
    generate_empty_fixture(&empty_path)?;
    generate_dng_fixture(&dng_path, size, size)?;

    println!("Generated fixtures:");
    println!("  - {}", base_path.display());
    println!("  - {}", rgb_path.display());
    println!("  - {}", bad_dark_path.display());
    println!("  - {}", empty_path.display());
    println!("  - {}", dng_path.display());

    Ok(())
}
