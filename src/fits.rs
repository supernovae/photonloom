use crate::config::{BayerPattern, Channel, FileFormat, Frame, FrameRole, LinearImage};
use anyhow::{Context, Result, bail};
use fitsio::FitsFile;
use fitsio::hdu::{FitsHdu, HduInfo};
use fitsio::images::{ImageDescription, ImageType};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Undistorted, two-dimensional celestial TAN projection. FITS CRPIX is one-based.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WcsTan {
    pub crval: [f64; 2],
    pub crpix: [f64; 2],
    pub cd: [[f64; 2]; 2],
}

impl WcsTan {
    pub fn is_valid(&self) -> bool {
        self.crval[0].is_finite()
            && self.crval[1].is_finite()
            && (-90.0..=90.0).contains(&self.crval[1])
            && self.crpix.iter().all(|v| v.is_finite())
            && self.cd.iter().flatten().all(|v| v.is_finite())
            && {
                let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
                let norm = self.cd.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
                norm > 0.0 && det.is_finite() && det.abs() > 1e-12 * norm * norm
            }
    }

    /// Converts a zero-based pixel center to J2000-style RA/Dec degrees; wraps RA into [0, 360).
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !self.is_valid() || !x.is_finite() || !y.is_finite() {
            return None;
        }
        let dx = x + 1.0 - self.crpix[0];
        let dy = y + 1.0 - self.crpix[1];
        let xi = (self.cd[0][0] * dx + self.cd[0][1] * dy).to_radians();
        let eta = (self.cd[1][0] * dx + self.cd[1][1] * dy).to_radians();
        let dec0 = self.crval[1].to_radians();
        let (sin0, cos0) = dec0.sin_cos();
        let denom = cos0 - eta * sin0;
        let ra = (self.crval[0].to_radians() + xi.atan2(denom)).to_degrees().rem_euclid(360.0);
        let dec = (sin0 + eta * cos0).atan2(denom.hypot(xi)).to_degrees();
        (ra.is_finite() && dec.is_finite()).then_some((ra, dec))
    }

    /// Shifts the reference pixel into a display crop without altering the stored science grid.
    pub fn shifted_for_crop(&self, x: usize, y: usize) -> Self {
        let mut shifted = *self;
        shifted.crpix[0] -= x as f64;
        shifted.crpix[1] -= y as f64;
        shifted
    }

    /// Validated variant of `shifted_for_crop` for externally supplied offsets.
    pub fn cropped(&self, x_offset: usize, y_offset: usize) -> Option<Self> {
        let shifted = self.shifted_for_crop(x_offset, y_offset);
        shifted.is_valid().then_some(shifted)
    }

    /// Sky coordinates of four corners and four edge midpoints of a pixel-center footprint.
    pub fn footprint(&self, dimensions: [usize; 2]) -> Option<[(f64, f64); 8]> {
        let [width, height] = dimensions;
        if width == 0 || height == 0 {
            return None;
        }
        let right = (width - 1) as f64;
        let bottom = (height - 1) as f64;
        let coords = [
            (0.0, 0.0), (right * 0.5, 0.0), (right, 0.0),
            (right, bottom * 0.5), (right, bottom), (right * 0.5, bottom),
            (0.0, bottom), (0.0, bottom * 0.5),
        ];
        let mut result = [(0.0, 0.0); 8];
        for (out, (x, y)) in result.iter_mut().zip(coords) {
            *out = self.pixel_to_sky(x, y)?;
        }
        Some(result)
    }
}

#[derive(Debug, Clone)]
pub struct FitsHeaderMeta {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub imagetyp: Option<String>,
    pub filter: Option<String>,
    pub exptime: Option<f32>,
    pub ccd_temp: Option<f32>,
    pub gain: Option<f32>,
    pub binning: (u32, u32),
    pub bayerpat: Option<BayerPattern>,
    pub instrume: Option<String>,
    pub object: Option<String>,
    pub is_rgb_cube: bool,
    pub wcs: Option<WcsTan>,
}

fn read_str_key(fits: &mut FitsFile, hdu: &FitsHdu, key: &str) -> Option<String> {
    hdu.read_key::<String>(fits, key)
        .ok()
        .map(|s| s.trim().trim_matches('\'').trim().to_string())
        .filter(|s| !s.is_empty())
}

fn read_f32_key(fits: &mut FitsFile, hdu: &FitsHdu, key: &str) -> Option<f32> {
    if let Ok(v) = hdu.read_key::<f64>(fits, key) {
        if v.is_finite() {
            return Some(v as f32);
        }
    }
    if let Ok(v) = hdu.read_key::<f32>(fits, key) {
        if v.is_finite() {
            return Some(v);
        }
    }
    if let Ok(v) = hdu.read_key::<i64>(fits, key) {
        return Some(v as f32);
    }
    if let Some(s) = read_str_key(fits, hdu, key) {
        if let Ok(v) = s.parse::<f32>() {
            if v.is_finite() {
                return Some(v);
            }
        }
    }
    None
}

fn read_u32_key(fits: &mut FitsFile, hdu: &FitsHdu, key: &str) -> Option<u32> {
    if let Ok(v) = hdu.read_key::<i64>(fits, key) {
        if v > 0 {
            return Some(v as u32);
        }
    }
    if let Some(s) = read_str_key(fits, hdu, key) {
        if let Ok(v) = s.parse::<u32>() {
            if v > 0 {
                return Some(v);
            }
        }
    }
    None
}

fn read_f64_key(fits: &mut FitsFile, hdu: &FitsHdu, key: &str) -> Option<f64> {
    hdu.read_key::<f64>(fits, key)
        .ok()
        .or_else(|| read_str_key(fits, hdu, key)?.parse().ok())
        .filter(|v| v.is_finite())
}

fn read_tan_wcs(fits: &mut FitsFile, hdu: &FitsHdu) -> Option<WcsTan> {
    if read_str_key(fits, hdu, "CTYPE1")?.to_ascii_uppercase() != "RA---TAN"
        || read_str_key(fits, hdu, "CTYPE2")?.to_ascii_uppercase() != "DEC--TAN"
    {
        return None;
    }
    for axis in 1..=2 {
        if let Some(unit) = read_str_key(fits, hdu, &format!("CUNIT{axis}")) {
            if !unit.eq_ignore_ascii_case("deg") {
                return None;
            }
        }
    }
    // No distortion representation is preserved by this linear TAN model.
    for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER", "CPDIS1", "CPDIS2",
                "DP1", "DP2", "D2IMDIS1", "D2IMDIS2", "DET2IM1", "DET2IM2"] {
        if hdu.read_key::<String>(fits, key).is_ok()
            || hdu.read_key::<f64>(fits, key).is_ok() {
            return None;
        }
    }
    for axis in 1..=2 {
        for degree in 0..=99 {
            if hdu.read_key::<String>(fits, &format!("PV{axis}_{degree}")).is_ok()
                || hdu.read_key::<f64>(fits, &format!("PV{axis}_{degree}")).is_ok() {
                return None;
            }
        }
    }
    let crval = [read_f64_key(fits, hdu, "CRVAL1")?, read_f64_key(fits, hdu, "CRVAL2")?];
    let crpix = [read_f64_key(fits, hdu, "CRPIX1")?, read_f64_key(fits, hdu, "CRPIX2")?];
    let cd_keys = [
        read_f64_key(fits, hdu, "CD1_1"), read_f64_key(fits, hdu, "CD1_2"),
        read_f64_key(fits, hdu, "CD2_1"), read_f64_key(fits, hdu, "CD2_2"),
    ];
    let cd = if cd_keys.iter().any(Option::is_some) {
        [[cd_keys[0]?, cd_keys[1]?], [cd_keys[2]?, cd_keys[3]?]]
    } else {
        let pc_keys = [
            read_f64_key(fits, hdu, "PC1_1"), read_f64_key(fits, hdu, "PC1_2"),
            read_f64_key(fits, hdu, "PC2_1"), read_f64_key(fits, hdu, "PC2_2"),
        ];
        if pc_keys.iter().all(Option::is_none) {
            return None;
        }
        let cdelt1 = read_f64_key(fits, hdu, "CDELT1")?;
        let cdelt2 = read_f64_key(fits, hdu, "CDELT2")?;
        [[pc_keys[0].unwrap_or(1.0) * cdelt1, pc_keys[1].unwrap_or(0.0) * cdelt1],
         [pc_keys[2].unwrap_or(0.0) * cdelt2, pc_keys[3].unwrap_or(1.0) * cdelt2]]
    };
    let wcs = WcsTan { crval, crpix, cd };
    wcs.is_valid().then_some(wcs)
}

pub fn inspect_fits_header(path: &Path) -> Result<FitsHeaderMeta> {
    let mut fits = FitsFile::open(path)
        .with_context(|| format!("Failed to open FITS file '{}'", path.display()))?;
    let hdu = fits
        .primary_hdu()
        .with_context(|| format!("Failed to read primary HDU in '{}'", path.display()))?;

    let (width, height, channels, is_rgb_cube) = match &hdu.info {
        HduInfo::ImageInfo { shape, .. } => match shape.as_slice() {
            // fitsio reverses NAXIS dimensions to [NAXIS2 (height), NAXIS1 (width)]
            [h, w] if *w > 0 && *h > 0 => (*w, *h, 1usize, false),
            [c, h, w] if *w > 0 && *h > 0 => {
                if *c != 3 {
                    bail!(
                        "Unsupported 3D FITS cube in '{}' with {} planes (only explicit 3-plane RGB cubes are supported)",
                        path.display(),
                        c
                    );
                }
                let ctype3 = read_str_key(&mut fits, &hdu, "CTYPE3")
                    .unwrap_or_default()
                    .to_uppercase();
                let colortyp = read_str_key(&mut fits, &hdu, "COLORTYP")
                    .or_else(|| read_str_key(&mut fits, &hdu, "COLORSPC"))
                    .unwrap_or_default()
                    .to_uppercase();
                let explicit_rgb =
                    matches!(ctype3.as_str(), "RGB" | "COLOR" | "CHANNEL" | "RGB-COLOR")
                        || matches!(colortyp.as_str(), "RGB" | "SRGB");
                if !explicit_rgb {
                    bail!(
                        "Rejected 3D FITS cube in '{}': 3D FITS cubes are only supported when the RGB channel axis is explicitly declared via CTYPE3='RGB' or COLORTYP='RGB'",
                        path.display()
                    );
                }
                (*w, *h, 3usize, true)
            }
            other => {
                bail!(
                    "Unsupported FITS image dimensions {:?} in '{}' (expected 2D image or explicit 3D RGB cube)",
                    other,
                    path.display()
                );
            }
        },
        _ => bail!(
            "Primary HDU in '{}' does not contain a valid FITS image",
            path.display()
        ),
    };

    let imagetyp = read_str_key(&mut fits, &hdu, "IMAGETYP")
        .or_else(|| read_str_key(&mut fits, &hdu, "FRAME"))
        .or_else(|| read_str_key(&mut fits, &hdu, "OBSTYPE"));

    let filter = read_str_key(&mut fits, &hdu, "FILTER");
    let exptime = read_f32_key(&mut fits, &hdu, "EXPTIME")
        .or_else(|| read_f32_key(&mut fits, &hdu, "EXPOSURE"));
    let ccd_temp = read_f32_key(&mut fits, &hdu, "CCD-TEMP")
        .or_else(|| read_f32_key(&mut fits, &hdu, "SET-TEMP"))
        .or_else(|| read_f32_key(&mut fits, &hdu, "CCDTEMP"));
    let gain = read_f32_key(&mut fits, &hdu, "GAIN")
        .or_else(|| read_f32_key(&mut fits, &hdu, "ISO"))
        .or_else(|| read_f32_key(&mut fits, &hdu, "EGAIN"));

    let xbin = read_u32_key(&mut fits, &hdu, "XBINNING")
        .or_else(|| read_u32_key(&mut fits, &hdu, "BINNING"))
        .unwrap_or(1);
    let ybin = read_u32_key(&mut fits, &hdu, "YBINNING").unwrap_or(xbin);

    let bayerpat = if let Some(bpat_str) = read_str_key(&mut fits, &hdu, "BAYERPAT")
        .or_else(|| read_str_key(&mut fits, &hdu, "COLORTYP"))
    {
        if is_rgb_cube
            || bpat_str.eq_ignore_ascii_case("RGB")
            || bpat_str.eq_ignore_ascii_case("MONO")
        {
            None
        } else {
            let Some(parsed) = BayerPattern::from_str_token(&bpat_str) else {
                bail!(
                    "Unsupported CFA/Bayer pattern '{}' in FITS file '{}' (only RGGB, BGGR, GRBG, GBRG are supported)",
                    bpat_str,
                    path.display()
                );
            };
            Some(parsed)
        }
    } else {
        None
    };

    let instrume = read_str_key(&mut fits, &hdu, "INSTRUME")
        .or_else(|| read_str_key(&mut fits, &hdu, "TELESCOP"));
    let object = read_str_key(&mut fits, &hdu, "OBJECT");
    let wcs = read_tan_wcs(&mut fits, &hdu);


    Ok(FitsHeaderMeta {
        width,
        height,
        channels,
        imagetyp,
        filter,
        exptime,
        ccd_temp,
        gain,
        binning: (xbin, ybin),
        bayerpat,
        instrume,
        object,
        is_rgb_cube,
        wcs,

    })
}

pub fn build_fits_frame(
    path: &Path,
    meta: FitsHeaderMeta,
    role: FrameRole,
    channel: Option<Channel>,
) -> Frame {
    let sensor = meta.instrume.unwrap_or_else(|| {
        format!(
            "FITS-{}x{}-bin{}x{}",
            meta.width, meta.height, meta.binning.0, meta.binning.1
        )
    });
    Frame {
        path: path.to_path_buf(),
        format: FileFormat::Fits,
        role,
        channel,
        width: meta.width,
        height: meta.height,
        channels: meta.channels,
        exposure: meta.exptime,
        temperature: meta.ccd_temp,
        gain_or_iso: meta.gain,
        binning: meta.binning,
        bayer: meta.bayerpat,
        sensor,
        object: meta.object,
        black_level: 0.0,
        white_level: 65535.0,
        cblack: [0.0; 4],
    }
}

/// Decodes the primary 2D FITS image (or explicit 3D RGB cube) into a row-major `LinearImage` (f32).
/// Honors `BSCALE`/`BZERO` automatically via CFITSIO `fits_read_img` with `TFLOAT`.
pub fn read_fits_linear(frame: &Frame) -> Result<LinearImage> {
    let meta = inspect_fits_header(&frame.path)?;
    let mut fits = FitsFile::open(&frame.path)
        .with_context(|| format!("Failed to open FITS file '{}'", frame.path.display()))?;
    let hdu = fits
        .primary_hdu()
        .with_context(|| format!("Failed to access primary HDU in '{}'", frame.path.display()))?;

    let mut pixels: Vec<f32> = hdu.read_image(&mut fits).with_context(|| {
        format!(
            "Failed to read FITS pixel data from '{}'",
            frame.path.display()
        )
    })?;

    let expected = meta.width * meta.height * meta.channels;
    if pixels.len() != expected {
        bail!(
            "Unexpected pixel count in '{}': got {}, expected {}",
            frame.path.display(),
            pixels.len(),
            expected
        );
    }

    let npix = meta.width * meta.height;
    let mut mask = vec![1u8; npix];
    for p in 0..npix {
        for c in 0..meta.channels {
            let idx = c * npix + p;
            if !pixels[idx].is_finite() {
                pixels[idx] = 0.0;
                mask[p] = 0;
            }
        }
    }

    Ok(LinearImage {
        width: meta.width,
        height: meta.height,
        channels: meta.channels,
        data: pixels,
        mask,
        bayer: meta.bayerpat,
        exposure: frame.exposure.unwrap_or(1.0).max(1e-6),
    })
}

/// Writes a 32-bit float FITS image (2D for single-channel, 3D explicit RGB cube for 3-channel)
/// with provenance header keywords. Uses atomic temp-file/rename to avoid partial FITS files.
pub fn write_fits_linear(
    dest: &Path,
    img: &LinearImage,
    channel_name: &str,
    object_name: &str,
    total_exposure: f32,
    accepted_count: usize,
    rejected_count: usize,
    wcs: Option<&WcsTan>,
) -> Result<()> {
    let parent = dest
        .parent()
        .context("FITS destination path has no parent directory")?;
    std::fs::create_dir_all(parent)?;

    let tmp_dir = tempfile::Builder::new()
        .prefix(".fits-tmp-")
        .tempdir_in(parent)?;
    let tmp_fits_path = tmp_dir.path().join("out.fits");

    let dims_2d = [img.height, img.width];
    let dims_3d = [img.channels, img.height, img.width];
    let dimensions: &[usize] = if img.channels == 1 {
        &dims_2d
    } else if img.channels == 3 {
        &dims_3d
    } else {
        bail!("Unsupported channel count {} for FITS export", img.channels);
    };
    if wcs.is_some_and(|wcs| !wcs.is_valid()) {
        bail!("Invalid TAN WCS for FITS export");
    }

    let desc = ImageDescription {
        data_type: ImageType::Float,
        dimensions,
    };

    {
        let mut fits = FitsFile::create(&tmp_fits_path)
            .with_custom_primary(&desc)
            .overwrite()
            .open()
            .with_context(|| {
                format!(
                    "Failed to create temporary FITS file '{}'",
                    tmp_fits_path.display()
                )
            })?;
        let hdu = fits.primary_hdu()?;
        hdu.write_key(&mut fits, "IMAGETYP", "Light Frame")?;
        hdu.write_key(&mut fits, "FILTER", channel_name)?;
        hdu.write_key(&mut fits, "OBJECT", object_name)?;
        hdu.write_key(&mut fits, "EXPTIME", f64::from(total_exposure))?;
        hdu.write_key(&mut fits, "NCOMBINE", accepted_count as i64)?;
        hdu.write_key(&mut fits, "NREJECT", rejected_count as i64)?;
        if img.channels == 3 {
            hdu.write_key(&mut fits, "CTYPE3", "RGB")?;
            hdu.write_key(&mut fits, "COLORTYP", "RGB")?;
        }
        if let Some(bayer) = img.bayer {
            hdu.write_key(&mut fits, "BAYERPAT", bayer.as_str())?;
        }
        if let Some(wcs) = wcs {
            hdu.write_key(&mut fits, "CTYPE1", "RA---TAN")?;
            hdu.write_key(&mut fits, "CTYPE2", "DEC--TAN")?;
            hdu.write_key(&mut fits, "CUNIT1", "deg")?;
            hdu.write_key(&mut fits, "CUNIT2", "deg")?;
            for (key, value) in [
                ("CRVAL1", wcs.crval[0]), ("CRVAL2", wcs.crval[1]),
                ("CRPIX1", wcs.crpix[0]), ("CRPIX2", wcs.crpix[1]),
                ("CD1_1", wcs.cd[0][0]), ("CD1_2", wcs.cd[0][1]),
                ("CD2_1", wcs.cd[1][0]), ("CD2_2", wcs.cd[1][1]),
            ] {
                hdu.write_key(&mut fits, key, value)?;
            }
        }
        hdu.write_image(&mut fits, &img.data)
            .with_context(|| format!("Failed to write FITS pixels to '{}'", dest.display()))?;
    }

    std::fs::File::open(&tmp_fits_path)?
        .sync_all()
        .with_context(|| {
            format!(
                "Failed to sync temporary FITS '{}'",
                tmp_fits_path.display()
            )
        })?;
    std::fs::rename(&tmp_fits_path, dest)
        .with_context(|| format!("Failed to atomically replace FITS '{}'", dest.display()))?;
    Ok(())
}
