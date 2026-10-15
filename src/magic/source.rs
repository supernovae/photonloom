use super::{InputDomain, MagicSource, TransferFunction};
use crate::config::{Channel, FrameRole, GalleryManifest, LinearImage};
use crate::crop::{CropRect, crop_image};
use crate::finish::{build_normalized_channel_map, compose_palette_linear, stretch_magic_baseline};
use crate::fits::{build_fits_frame, inspect_fits_header, read_fits_linear};
use crate::fits::WcsTan;
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use fitsio::FitsFile;
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use crate::finish::{MagicMonoModel, MagicStretchModel};

pub struct LoadedSource {
    pub baseline: LinearImage,
    pub digest: String,
    pub target_id: String,
    pub palette: String,
    pub channels_used: Vec<String>,
    pub crop: Option<CropRect>,
    pub domain: InputDomain,
    pub transfer: TransferFunction,
    pub assumption: Option<String>,
    pub gallery: Option<PathBuf>,
    pub source_path: PathBuf,
}

fn hash_file(path: &Path, digest: &mut Sha256) -> Result<()> {
    let mut file = std::fs::File::open(path).with_context(|| format!("Opening source {}", path.display()))?;
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 { break; }
        digest.update(&buffer[..count]);
    }
    Ok(())
}

fn checked_asset(root: &Path, rel: &str) -> Result<PathBuf> {
    let path = Path::new(rel);
    ensure!(path.components().all(|component| matches!(component, Component::Normal(_))), "Invalid gallery asset path: {rel}");
    let canon = root.join(path).canonicalize()?;
    ensure!(canon.starts_with(root), "Gallery asset escapes its root: {rel}");
    Ok(canon)
}

fn display_fits(path: &Path, domain: InputDomain) -> Result<(LinearImage, InputDomain, Option<String>)> {
    let meta = inspect_fits_header(path)?;
    let frame = build_fits_frame(path, meta.clone(), FrameRole::Light, None);
    let image = read_fits_linear(&frame)?;
    let resolved = match domain {
        InputDomain::Auto if meta.imagetyp.as_deref().is_some_and(|s| s.to_lowercase().contains("display")) => InputDomain::Display,
        InputDomain::Auto if meta.imagetyp.as_deref().is_some_and(|s| s.to_lowercase().contains("stack") || s.to_lowercase().contains("light")) => InputDomain::Linear,
        InputDomain::Auto => {
            let (min, max) = image.data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            if min < -0.001 || max > 1.2 { InputDomain::Linear } else { InputDomain::Display }
        }
        other => other,
    };
    let warning = (domain == InputDomain::Auto && resolved == InputDomain::Display).then(|| "FITS display domain assumed from bounded samples; override with --input-domain linear if unstretched".into());
    let rgb = if resolved == InputDomain::Linear {
        stretch_magic_baseline(image, if meta.channels == 1 { "monochrome" } else { "natural" })
    } else { triplet(image) };
    Ok((rgb, resolved, warning))
}

fn triplet(image: LinearImage) -> LinearImage {
    if image.channels == 3 { return image; }
    let n = image.pixel_count();
    let mut rgb = LinearImage::new(image.width, image.height, 3);
    for plane in 0..3 { rgb.data[plane * n..(plane + 1) * n].copy_from_slice(&image.data); }
    rgb.mask = image.mask;
    rgb
}

fn srgb_inverse(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn display_raster(path: &Path, transfer: TransferFunction) -> Result<LinearImage> {
    let dyn_img = image::ImageReader::open(path)?.with_guessed_format()?.decode()
        .with_context(|| format!("Decoding image {}", path.display()))?;
    let (w, h) = (dyn_img.width() as usize, dyn_img.height() as usize);
    let n = w.checked_mul(h).context("Image dimensions overflow")?;
    ensure!(n > 0 && n.checked_mul(13).is_some(), "Invalid raster dimensions");
    let mut out = LinearImage::new(w, h, 3);
    match dyn_img {
        image::DynamicImage::ImageRgb16(img) => {
            for (p, rgb) in img.pixels().enumerate() {
                for c in 0..3 {
                    let v = rgb[c] as f32 / 65535.0;
                    out.data[c * n + p] = if transfer == TransferFunction::Srgb { srgb_inverse(v) } else { v };
                }
            }
        }
        image::DynamicImage::ImageRgb8(img) => {
            for (p, rgb) in img.pixels().enumerate() {
                for c in 0..3 {
                    let v = rgb[c] as f32 / 255.0;
                    out.data[c * n + p] = if transfer == TransferFunction::Srgb { srgb_inverse(v) } else { v };
                }
            }
        }
        _ => bail!("Unsupported raster layout: expected RGB8 or RGB16 without alpha"),
    }
    Ok(out)
}

fn matches_gallery_root(manifest_canonical_output: &str, canonical_root: &Path) -> bool {
    let recorded = Path::new(manifest_canonical_output);
    recorded == canonical_root || recorded.canonicalize().ok().as_deref() == Some(canonical_root)
}

fn adjacent_gallery_transfer(path: &Path) -> bool {
    let Some(root) = path.parent().and_then(Path::parent).and_then(Path::parent) else { return false };
    let Ok(bytes) = std::fs::read(root.join("manifest.json")) else { return false };
    let Ok(manifest) = serde_json::from_slice::<GalleryManifest>(&bytes) else { return false };
    let Ok(canon_root) = root.canonicalize() else { return false };
    if !matches_gallery_root(&manifest.canonical_output, &canon_root) { return false }
    manifest.targets.iter().flat_map(|t| &t.variants).any(|variant| {
        checked_asset(root, &variant.tiff_path).ok().as_deref() == Some(path)
    })
}

pub fn load(source: &MagicSource, crop_enabled: bool) -> Result<LoadedSource> {
    match source {
        MagicSource::Gallery { gallery, target_id, palette } => {
            let root = gallery.canonicalize().with_context(|| format!("Opening gallery {}", gallery.display()))?;
            let manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
            ensure!(matches_gallery_root(&manifest.canonical_output, &root), "Gallery root disagrees with manifest canonical output");
            ensure!(Path::new(target_id).components().count() == 1
                && target_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                && !target_id.is_empty(), "Invalid gallery target ID");
            let target = manifest.targets.iter().find(|t| &t.target_id == target_id).context("Target ID not in gallery")?;
            let pal = if palette.is_empty() {
                target.variants.iter().find(|v| v.id == target.primary_variant && v.restore_info.is_none() && v.magic.is_none())
                    .map(|v| v.palette.as_str()).context("Primary variant is not an original science palette")?
            } else { palette.as_str() };
            ensure!(target.variants.iter().any(|v| v.palette == pal && v.restore_info.is_none() && v.magic.is_none()), "No usable real palette '{pal}' on named target");
            let mut digest = Sha256::new();
            digest.update(b"magic-source-1\0");
            digest.update(pal.as_bytes());
            digest.update([u8::from(crop_enabled)]);
            let mut images = BTreeMap::new();
            let mut crop = None;
            for sci in &target.science_integrations {
                let Some(ch) = Channel::from_token(&sci.channel) else { continue };
                let path = checked_asset(&root, &sci.fits_path)?;
                digest.update(sci.channel.as_bytes());
                hash_file(&path, &mut digest)?;
                let meta = inspect_fits_header(&path)?;
                let frame = build_fits_frame(&path, meta, FrameRole::Light, Some(ch));
                let mut image = read_fits_linear(&frame)?;
                if crop_enabled {
                    if let Some(bounds) = sci.recommended_crop {
                        crop_image(&mut image, &bounds)?;
                        crop = Some(bounds);
                    }
                }
                images.insert(ch, image);
            }
            ensure!(!images.is_empty(), "Gallery lacks readable science FITS");
            let refs: BTreeMap<_, _> = images.iter().map(|(&c, image)| (c, image)).collect();
            let norm = build_normalized_channel_map(&refs);
            let (linear, channels_used, _, _, _) = compose_palette_linear(pal, &refs, &norm, None)?;
            let baseline = stretch_magic_baseline(linear, pal);
            let source_path = root.clone();
            Ok(LoadedSource { baseline, digest: format!("{:x}", digest.finalize()), target_id: target_id.clone(), palette: pal.to_owned(), channels_used, crop, domain: InputDomain::Linear, transfer: TransferFunction::Linear, assumption: None, gallery: Some(root), source_path })
        }
        MagicSource::File { path, domain, transfer } => {
            let path = path.canonicalize().with_context(|| format!("Opening input {}", path.display()))?;
            ensure!(path.is_file(), "Magic input must be a regular file");
            let mut digest = Sha256::new();
            hash_file(&path, &mut digest)?;
            let digest = format!("{:x}", digest.finalize());
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            let (baseline, domain, transfer, assumption) = match ext.as_str() {
                "fit" | "fits" | "fts" => {
                    ensure!(*transfer == TransferFunction::Auto || *transfer == TransferFunction::Linear, "FITS uses linear transfer; --transfer srgb is unsupported");
                    let (image, resolved, assumption) = display_fits(&path, *domain)?;
                    (image, resolved, TransferFunction::Linear, assumption)
                }
                "png" | "tif" | "tiff" => {
                    ensure!(*domain == InputDomain::Auto || *domain == InputDomain::Display, "Raster images have display domain; cannot assume an unstretched linear integration");
                    let known_linear = ext != "png" && adjacent_gallery_transfer(&path);
                    let resolved = match transfer { TransferFunction::Auto if known_linear => TransferFunction::Linear, TransferFunction::Auto => TransferFunction::Srgb, t => *t };
                    let assumption = (*transfer == TransferFunction::Auto && !known_linear && ext != "png").then(|| "Untagged external TIFF assumed sRGB; use --transfer linear if it is linear-light".into());
                    (display_raster(&path, resolved)?, InputDomain::Display, resolved, assumption)
                }
                _ => bail!("Unsupported Magic input: expected FITS, RGB TIFF or RGB PNG"),
            };
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
            let slug = stem.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect::<String>();
            let slug = slug.trim_matches('-');
            let target_id = format!("{}-{}", if slug.is_empty() { "image" } else { slug }, &digest[..12]);
            Ok(LoadedSource { baseline, digest, target_id, palette: if ext == "fit" || ext == "fits" || ext == "fts" { "source-fits" } else { "source-rgb" }.into(), channels_used: vec!["source".into()], crop: None, domain, transfer, assumption, gallery: None, source_path: path })
        }
    }
}

#[derive(Clone, Copy)]
struct Window { x: usize, y: usize, width: usize, height: usize }

fn checked_window(rect: Window, width: usize, height: usize) -> Result<()> {
    ensure!(rect.width > 0 && rect.height > 0
        && rect.x.checked_add(rect.width).is_some_and(|v| v <= width)
        && rect.y.checked_add(rect.height).is_some_and(|v| v <= height)
        && rect.width.checked_mul(rect.height).and_then(|n| n.checked_mul(13)).is_some(),
        "Invalid or overflowing source window");
    Ok(())
}

/// Decode only source pixels in a native rectangle, preserving original RGB
/// samples and validity. TIFF chunks and PNG rows are decoded individually.
fn visit_window(path: &Path, transfer: TransferFunction, rect: Window,
    mut pixel: impl FnMut(usize, usize, [f32; 3], bool)) -> Result<usize> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "fits" | "fit" | "fts" => {
            let meta = inspect_fits_header(path)?;
            checked_window(rect, meta.width, meta.height)?;
            let mut file = FitsFile::open(path)?;
            let hdu = file.primary_hdu()?;
            for y in rect.y..rect.y + rect.height {
                let mut planes = [Vec::new(), Vec::new(), Vec::new()];
                for (c, plane) in planes.iter_mut().enumerate().take(meta.channels) {
                    *plane = if meta.channels == 1 {
                        hdu.read_region(&mut file, &[&(rect.x..rect.x + rect.width), &(y..y + 1)])?
                    } else {
                        hdu.read_region(&mut file, &[&(rect.x..rect.x + rect.width), &(y..y + 1), &(c..c + 1)])?
                    };
                    ensure!(plane.len() == rect.width, "Truncated FITS row");
                }
                for x in 0..rect.width {
                    let p = x;
                    let rgb = [planes[0][p], planes[if meta.channels == 1 { 0 } else { 1 }][p],
                        planes[if meta.channels == 1 { 0 } else { 2 }][p]];
                    pixel(rect.x + x, y, rgb, rgb.iter().all(|v| v.is_finite()));
                }
            }
            Ok(meta.channels)
        }
        "tif" | "tiff" => {
            let mut decoder = Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?))?;
            let (width, height) = decoder.dimensions()?;
            let (width, height) = (width as usize, height as usize);
            checked_window(rect, width, height)?;
            ensure!(matches!(decoder.colortype()?, tiff::ColorType::RGB(8) | tiff::ColorType::RGB(16)),
                "Unsupported TIFF layout: expected RGB8 or RGB16");
            let (cw, ch) = decoder.chunk_dimensions();
            let (cw, ch) = (cw as usize, ch as usize);
            ensure!(cw > 0 && ch > 0 && cw.checked_mul(ch).and_then(|n| n.checked_mul(6))
                .is_some_and(|n| n <= 64 * 1024 * 1024), "TIFF chunk exceeds 64 MiB decoder budget");
            let across = width.div_ceil(cw);
            let count = match decoder.get_chunk_type() {
                ChunkType::Strip => decoder.strip_count()? as usize,
                ChunkType::Tile => decoder.tile_count()? as usize,
            };
            for index in 0..count {
                let (ox, oy) = match decoder.get_chunk_type() {
                    ChunkType::Strip => (0, index * ch),
                    ChunkType::Tile => ((index % across) * cw, (index / across) * ch),
                };
                if ox >= rect.x + rect.width || ox + cw <= rect.x
                    || oy >= rect.y + rect.height || oy + ch <= rect.y { continue; }
                let (dw, dh) = decoder.chunk_data_dimensions(index as u32);
                let (dw, dh) = (dw as usize, dh as usize);
                let decoded = decoder.read_chunk(index as u32)?;
                let scale = match &decoded {
                    DecodingResult::U8(_) => 1.0 / 255.0,
                    DecodingResult::U16(_) => 1.0 / 65535.0,
                    _ => bail!("Unsupported TIFF RGB sample depth"),
                };
                for y in oy.max(rect.y)..(oy + dh).min(rect.y + rect.height) {
                    for x in ox.max(rect.x)..(ox + dw).min(rect.x + rect.width) {
                        let i = ((y - oy) * dw + x - ox) * 3;
                        let sample = |c| match &decoded {
                            DecodingResult::U8(v) => v[i + c] as f32,
                            DecodingResult::U16(v) => v[i + c] as f32,
                            _ => unreachable!(),
                        };
                        let mut rgb = [sample(0) * scale, sample(1) * scale, sample(2) * scale];
                        if transfer == TransferFunction::Srgb { rgb = rgb.map(srgb_inverse); }
                        pixel(x, y, rgb, true);
                    }
                }
            }
            Ok(3)
        }
        "png" => {
            let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
            decoder.set_transformations(png::Transformations::IDENTITY);
            let mut reader = decoder.read_info()?;
            let info = reader.info();
            let (width, height) = (info.width as usize, info.height as usize);
            checked_window(rect, width, height)?;
            ensure!(info.color_type == png::ColorType::Rgb
                && matches!(info.bit_depth, png::BitDepth::Eight | png::BitDepth::Sixteen),
                "Unsupported PNG layout: expected RGB8 or RGB16");
            ensure!(!info.interlaced, "Interlaced PNG requires full-frame reconstruction; use non-interlaced RGB PNG");
            ensure!(width.checked_mul(6).is_some_and(|n| n <= 64 * 1024 * 1024),
                "PNG row exceeds decoder budget");
            let depth = if info.bit_depth == png::BitDepth::Eight { 1 } else { 2 };
            for y in 0..rect.y + rect.height {
                let row = reader.next_row()?.context("Truncated PNG")?;
                if y < rect.y { continue; }
                let bytes = row.data();
                for x in rect.x..rect.x + rect.width {
                    let i = x * 3 * depth;
                    let mut rgb = [0.0; 3];
                    for c in 0..3 {
                        rgb[c] = if depth == 1 { bytes[i + c] as f32 / 255.0 }
                            else { u16::from_be_bytes([bytes[i + c * 2], bytes[i + c * 2 + 1]]) as f32 / 65535.0 };
                    }
                    if transfer == TransferFunction::Srgb { rgb = rgb.map(srgb_inverse); }
                    pixel(x, y, rgb, true);
                }
            }
            Ok(3)
        }
        _ => bail!("Unsupported Magic source format"),
    }
}

fn window_image(path: &Path, transfer: TransferFunction, rect: Window, step: usize) -> Result<LinearImage> {
    let (w, h) = (rect.width.div_ceil(step), rect.height.div_ceil(step));
    let n = w.checked_mul(h).and_then(|v| v.checked_mul(13)).context("Window image overflow")? / 13;
    let mut image = LinearImage::new(w, h, 3);
    let mut counts = vec![0u64; n];
    let channels = visit_window(path, transfer, rect, |x, y, rgb, valid| {
        if !valid { return; }
        let p = ((y - rect.y) / step) * w + (x - rect.x) / step;
        counts[p] += 1;
        for c in 0..3 { image.data[c * n + p] += rgb[c]; }
    })?;
    for (p, &count) in counts.iter().enumerate() {
        image.mask[p] = u8::from(count != 0);
        for c in 0..3 { image.data[c * n + p] /= count.max(1) as f32; }
    }
    if channels == 1 { image.data.truncate(n); image.channels = 1; }
    Ok(image)
}

/// Source metadata and a display baseline of at most 2048 pixels per axis.
/// The fields mirror LoadedSource so callers can migrate without ambiguity.
pub struct PreviewSource {
    pub baseline: LinearImage,
    pub digest: String,
    pub target_id: String,
    pub palette: String,
    pub channels_used: Vec<String>,
    pub crop: Option<CropRect>,
    pub domain: InputDomain,
    pub transfer: TransferFunction,
    pub assumption: Option<String>,
    pub gallery: Option<PathBuf>,
    pub source_path: PathBuf,
    pub native_dimensions: [usize; 2],
    pub title: String,
    pub subject_wcs: Option<WcsTan>,
}

pub fn load_preview(source: &MagicSource, crop_enabled: bool) -> Result<PreviewSource> {
    match source {
        MagicSource::Gallery { gallery, target_id, palette } => {
            let root = gallery.canonicalize()?;
            let manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
            ensure!(matches_gallery_root(&manifest.canonical_output, &root), "Gallery root disagrees with manifest");
            ensure!(!target_id.is_empty() && target_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'), "Invalid target ID");
            let target = manifest.targets.iter().find(|t| &t.target_id == target_id).context("Unknown target ID")?;
            let pal = if palette.is_empty() {
                target.variants.iter().find(|v| v.id == target.primary_variant && v.restore_info.is_none() && v.magic.is_none())
                    .map(|v| v.palette.as_str()).context("Primary variant is not an original science palette")?
            } else { palette.as_str() };
            ensure!(target.variants.iter().any(|v| v.palette == pal && v.restore_info.is_none() && v.magic.is_none()), "No usable real palette '{pal}'");
            let mut digest = Sha256::new();
            digest.update(b"magic-source-1\0");
            digest.update(pal.as_bytes());
            digest.update([u8::from(crop_enabled)]);
            let mut images = BTreeMap::new();
            let mut crop = None;
            let mut dimensions = None;
            let mut subject_wcs = None;
            for sci in &target.science_integrations {
                let Some(ch) = Channel::from_token(&sci.channel) else { continue };
                let path = checked_asset(&root, &sci.fits_path)?;
                digest.update(sci.channel.as_bytes());
                hash_file(&path, &mut digest)?;
                if pal == "hoo" && !matches!(ch, Channel::Ha | Channel::Oiii) { continue; }
                let meta = inspect_fits_header(&path)?;
                if subject_wcs.is_none() {
                    let reference = sci.wcs.or(meta.wcs);
                    subject_wcs = match (reference, crop_enabled.then_some(sci.recommended_crop).flatten()) {
                        (Some(wcs), Some(bounds)) => Some(wcs.shifted_for_crop(bounds.x, bounds.y)),
                        (wcs, None) => wcs,
                        (None, Some(_)) => None,
                    };
                }
                let rect = if crop_enabled {
                    if let Some(bounds) = sci.recommended_crop {
                        ensure!(bounds.original_width == meta.width && bounds.original_height == meta.height,
                            "Gallery crop does not match science FITS");
                        crop = Some(bounds);
                        Window { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height }
                    } else { Window { x: 0, y: 0, width: meta.width, height: meta.height } }
                } else { Window { x: 0, y: 0, width: meta.width, height: meta.height } };
                checked_window(rect, meta.width, meta.height)?;
                ensure!(dimensions.is_none_or(|d| d == [rect.width, rect.height]), "Science channel dimensions/crops differ");
                dimensions = Some([rect.width, rect.height]);
                let step = rect.width.max(rect.height).div_ceil(2048).max(1);
                images.insert(ch, window_image(&path, TransferFunction::Linear, rect, step)?);
            }
            let native_dimensions = dimensions.context("Gallery lacks readable science FITS")?;
            let refs: BTreeMap<_, _> = images.iter().map(|(&ch, image)| (ch, image)).collect();
            let norm = build_normalized_channel_map(&refs);
            let (linear, channels_used, _, _, _) = compose_palette_linear(pal, &refs, &norm, None)?;
            Ok(PreviewSource { baseline: stretch_magic_baseline(linear, pal), digest: format!("{:x}", digest.finalize()),
                target_id: target_id.clone(), palette: pal.into(), channels_used, crop, domain: InputDomain::Linear,
                transfer: TransferFunction::Linear, assumption: None, gallery: Some(root.clone()), source_path: root, native_dimensions,
                title: target.target_label.clone(), subject_wcs })
        }
        MagicSource::File { path, domain, transfer } => {
            let path = path.canonicalize()?;
            ensure!(path.is_file(), "Magic input must be a regular file");
            let mut digest = Sha256::new();
            hash_file(&path, &mut digest)?;
            let digest = format!("{:x}", digest.finalize());
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            let fits = matches!(ext.as_str(), "fits" | "fit" | "fts");
            ensure!(fits || matches!(ext.as_str(), "tiff" | "tif" | "png"), "Unsupported Magic input format");
            ensure!(!fits || *transfer != TransferFunction::Srgb, "FITS uses linear transfer");
            ensure!(fits || *domain != InputDomain::Linear, "Raster input has display domain");
            let meta = if fits { Some(inspect_fits_header(&path)?) } else { None };
            let native_dimensions = if let Some(m) = &meta { [m.width, m.height] } else if ext == "png" {
                let reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path)?)).read_info()?;
                [reader.info().width as usize, reader.info().height as usize]
            } else {
                let mut decoder = Decoder::new(std::io::BufReader::new(std::fs::File::open(&path)?))?;
                let (w, h) = decoder.dimensions()?;
                [w as usize, h as usize]
            };
            let [width, height] = native_dimensions;
            let rect = Window { x: 0, y: 0, width, height };
            checked_window(rect, width, height)?;
            let resolved_transfer = if fits { TransferFunction::Linear } else if *transfer == TransferFunction::Auto {
                if ext != "png" && adjacent_gallery_transfer(&path) { TransferFunction::Linear } else { TransferFunction::Srgb }
            } else { *transfer };
            let assumption = (!fits && ext != "png" && *transfer == TransferFunction::Auto && resolved_transfer == TransferFunction::Srgb)
                .then(|| "Untagged external TIFF assumed sRGB; use --transfer linear if it is linear-light".into());
            let step = width.max(height).div_ceil(2048).max(1);
            let mut baseline = window_image(&path, resolved_transfer, rect, step)?;
            let resolved_domain = if let Some(meta) = &meta {
                match domain {
                    InputDomain::Auto if meta.imagetyp.as_deref().is_some_and(|s| s.to_lowercase().contains("display")) => InputDomain::Display,
                    InputDomain::Auto if meta.imagetyp.as_deref().is_some_and(|s| s.to_lowercase().contains("stack") || s.to_lowercase().contains("light")) => InputDomain::Linear,
                    InputDomain::Auto if baseline.data.iter().any(|&v| v < -0.001 || v > 1.2) => InputDomain::Linear,
                    InputDomain::Auto => InputDomain::Display,
                    other => *other,
                }
            } else { InputDomain::Display };
            if resolved_domain == InputDomain::Linear {
                baseline = stretch_magic_baseline(baseline, if meta.as_ref().is_some_and(|m| m.channels == 1) { "monochrome" } else { "natural" });
            } else { baseline = triplet(baseline); }
            let warning = assumption.or_else(|| (*domain == InputDomain::Auto && fits && resolved_domain == InputDomain::Display)
                .then(|| "FITS display domain assumed from bounded samples; override with --input-domain linear if unstretched".into()));
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
            let slug = stem.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect::<String>();
            let slug = slug.trim_matches('-');
            let title = meta.as_ref().and_then(|m| m.object.clone()).unwrap_or_else(|| stem.to_string());
            Ok(PreviewSource { baseline, digest: digest.clone(), target_id: format!("{}-{}", if slug.is_empty() { "image" } else { slug }, &digest[..12]),
                palette: if fits { "source-fits" } else { "source-rgb" }.into(), channels_used: vec!["source".into()],
                crop: None, domain: resolved_domain, transfer: resolved_transfer, assumption: warning,
                gallery: None, source_path: path, native_dimensions, title,
                subject_wcs: meta.and_then(|m| m.wcs) })
        }
    }
}

/// Raw native pixels from a rectangle; a tile is NOT a globally normalized or
/// stretched display baseline. Native global stretch must be derived by the
/// caller before using these tiles for a full-resolution Magic candidate.
pub struct NativeTiles {
    source: MagicSource,
    pub dimensions: [usize; 2],
    pub transfer: TransferFunction,
    palette: String,
    gallery_sources: Vec<(Channel, PathBuf, usize, usize)>,
}

pub fn load_native_tiles(source: &MagicSource, crop_enabled: bool) -> Result<NativeTiles> {
    let preview = load_preview(source, crop_enabled)?;
    NativeTiles::from_preview(source, crop_enabled, &preview)
}

impl NativeTiles {
    /// Reuse verified preview geometry without re-hashing the source.
    pub fn from_preview(source: &MagicSource, crop_enabled: bool, preview: &PreviewSource) -> Result<Self> {
        let mut gallery_sources = Vec::new();
        if let MagicSource::Gallery { gallery, target_id, .. } = source {
            let root = gallery.canonicalize()?;
            let manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
            let target = manifest.targets.iter().find(|t| &t.target_id == target_id).context("Unknown target ID")?;
            for sci in &target.science_integrations {
                let Some(ch) = Channel::from_token(&sci.channel) else { continue };
                if preview.palette == "hoo" && !matches!(ch, Channel::Ha | Channel::Oiii) { continue; }
                let path = checked_asset(&root, &sci.fits_path)?;
                let (x, y) = if crop_enabled { sci.recommended_crop.map(|r| (r.x, r.y)).unwrap_or((0, 0)) } else { (0, 0) };
                gallery_sources.push((ch, path, x, y));
            }
        }
        Ok(Self { source: source.clone(), dimensions: preview.native_dimensions,
            transfer: preview.transfer, palette: preview.palette.clone(), gallery_sources })
    }

    pub fn read_tile(&self, x: usize, y: usize, width: usize, height: usize) -> Result<LinearImage> {
        checked_window(Window { x, y, width, height }, self.dimensions[0], self.dimensions[1])?;
        match &self.source {
            MagicSource::File { path, .. } =>
                window_image(path, self.transfer, Window { x, y, width, height }, 1),
            MagicSource::Gallery { .. } => {
                let mut images = BTreeMap::new();
                for (ch, path, ox, oy) in &self.gallery_sources {
                    images.insert(*ch, window_image(path, TransferFunction::Linear,
                        Window { x: ox + x, y: oy + y, width, height }, 1)?);
                }
                let refs: BTreeMap<_, _> = images.iter().map(|(&ch, image)| (ch, image)).collect();
                // No tile-local normalization: applying it here creates seams.
                let unnormalized = images.clone();
                let (image, _, _, _, _) = compose_palette_linear(&self.palette, &refs, &unnormalized, None)?;
                Ok(image)
            }
        }
    }
}

/// Interleaved RGB f32 and validity byte, 13 bytes/pixel in disk scratch.
/// Keeps native baselines out of RAM while full-size tile operators execute.
pub struct BaselineScratch {
    file: tempfile::NamedTempFile,
    width: usize,
    height: usize,
}

impl BaselineScratch {
    pub fn create(directory: &Path, width: usize, height: usize) -> Result<Self> {
        let len = width.checked_mul(height).and_then(|n| n.checked_mul(13))
            .context("Native scratch dimensions overflow")?;
        ensure!(width > 0 && height > 0, "Native scratch cannot be empty");
        std::fs::create_dir_all(directory)?;
        let file = tempfile::NamedTempFile::new_in(directory)?;
        file.as_file().set_len(u64::try_from(len)?)?;
        // set_len can reserve only sparse metadata on Unix. Reserve actual
        // blocks so an insufficient filesystem fails before any tile work.
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            unsafe extern "C" {
                fn posix_fallocate(fd: std::os::raw::c_int, offset: i64, len: i64) -> std::os::raw::c_int;
            }
            let size = i64::try_from(len).context("Scratch size exceeds platform file offset")?;
            // SAFETY: fd remains open, offset and length are checked i64, and
            // posix_fallocate does not retain the supplied file descriptor.
            let code = unsafe { posix_fallocate(file.as_file().as_raw_fd(), 0, size) };
            if code != 0 {
                return Err(std::io::Error::from_raw_os_error(code))
                    .with_context(|| format!("Reserving {len} bytes of native scratch in {}", directory.display()));
            }
        }
        Ok(Self { file, width, height })
    }

    pub fn write_tile(&mut self, x: usize, y: usize, tile: &LinearImage) -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        ensure!(tile.channels == 3, "Scratch requires RGB native tiles");
        checked_window(Window { x, y, width: tile.width, height: tile.height }, self.width, self.height)?;
        let n = tile.width.checked_mul(tile.height).context("Scratch tile overflow")?;
        ensure!(tile.data.len() == n * 3 && tile.mask.len() == n, "Incomplete native tile");
        let mut row = vec![0u8; tile.width.checked_mul(13).context("Scratch row overflow")?];
        for ry in 0..tile.height {
            for rx in 0..tile.width {
                let p = ry * tile.width + rx;
                for c in 0..3 { row[rx * 13 + c * 4..rx * 13 + c * 4 + 4]
                    .copy_from_slice(&tile.data[c * n + p].to_le_bytes()); }
                row[rx * 13 + 12] = tile.mask[p];
            }
            let offset = ((y + ry) * self.width + x).checked_mul(13).context("Scratch offset overflow")?;
            self.file.as_file_mut().seek(SeekFrom::Start(u64::try_from(offset)?))?;
            self.file.as_file_mut().write_all(&row)?;
        }
        Ok(())
    }

    pub fn read_tile(&mut self, x: usize, y: usize, width: usize, height: usize) -> Result<LinearImage> {
        use std::io::{Read, Seek, SeekFrom};
        checked_window(Window { x, y, width, height }, self.width, self.height)?;
        let mut image = LinearImage::new(width, height, 3);
        let n = width * height;
        let mut row = vec![0u8; width.checked_mul(13).context("Scratch row overflow")?];
        for ry in 0..height {
            let offset = ((y + ry) * self.width + x).checked_mul(13).context("Scratch offset overflow")?;
            self.file.as_file_mut().seek(SeekFrom::Start(u64::try_from(offset)?))?;
            self.file.as_file_mut().read_exact(&mut row)?;
            for rx in 0..width {
                let p = ry * width + rx;
                for c in 0..3 {
                    let start = rx * 13 + c * 4;
                    image.data[c * n + p] = f32::from_le_bytes(row[start..start + 4].try_into()?);
                }
                image.mask[p] = row[rx * 13 + 12];
            }
        }
        Ok(image)
    }

    pub fn write_variant_images(&mut self, tiff_dest: &Path, png_dest: &Path) -> Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        let width = self.width;
        let mut bytes = vec![0u8; width.checked_mul(13).context("Scratch row overflow")?];
        crate::finish::write_variant_image_rows(width, self.height, tiff_dest, png_dest,
            |y, rgb, mask| {
                let offset = y.checked_mul(width).and_then(|n| n.checked_mul(13)).context("Scratch offset overflow")?;
                self.file.as_file_mut().seek(SeekFrom::Start(u64::try_from(offset)?))?;
                self.file.as_file_mut().read_exact(&mut bytes)?;
                for x in 0..width {
                    for c in 0..3 {
                        let start = x * 13 + c * 4;
                        rgb[x * 3 + c] = f32::from_le_bytes(bytes[start..start + 4].try_into()?);
                    }
                    mask[x] = bytes[x * 13 + 12];
                }
                Ok(())
            })
    }
}

/// Sample-derived global parameters, reused at absolute native coordinates.
/// Native tiles are never fitted independently.
pub struct NativeBaselineModel {
    dimensions: [usize; 2],
    slots: [Option<Channel>; 3],
    mono: BTreeMap<Channel, MagicMonoModel>,
    stretch: Option<MagicStretchModel>,
}

impl NativeBaselineModel {
    /// Caller verifies the source digest before fitting and rendering.
    pub fn fit(source: &MagicSource, crop_enabled: bool, preview: &PreviewSource) -> Result<Self> {
        let dimensions = preview.native_dimensions;
        if let MagicSource::File { path, .. } = source {
            if preview.domain == InputDomain::Display {
                return Ok(Self { dimensions, slots: [None; 3], mono: BTreeMap::new(), stretch: None });
            }
            let step = dimensions[0].max(dimensions[1]).div_ceil(2048).max(1);
            let raw = window_image(path, preview.transfer, Window {
                x: 0, y: 0, width: dimensions[0], height: dimensions[1] }, step)?;
            let style = if raw.channels == 1 { "monochrome" } else { "natural" };
            return Ok(Self { dimensions, slots: [None; 3], mono: BTreeMap::new(),
                stretch: Some(MagicStretchModel::fit(&raw, style)) });
        }
        let MagicSource::Gallery { gallery, target_id, .. } = source else { unreachable!() };
        let root = gallery.canonicalize()?;
        let manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let target = manifest.targets.iter().find(|t| &t.target_id == target_id).context("Unknown target ID")?;
        let mut raw = BTreeMap::new();
        for sci in &target.science_integrations {
            let Some(ch) = Channel::from_token(&sci.channel) else { continue };
            if preview.palette == "hoo" && !matches!(ch, Channel::Ha | Channel::Oiii) { continue; }
            let path = checked_asset(&root, &sci.fits_path)?;
            let meta = inspect_fits_header(&path)?;
            let (x, y, w, h) = if crop_enabled {
                sci.recommended_crop.map(|r| (r.x, r.y, r.width, r.height))
                    .unwrap_or((0, 0, meta.width, meta.height))
            } else { (0, 0, meta.width, meta.height) };
            ensure!([w, h] == dimensions, "Native calibration channels do not share source geometry");
            let step = w.max(h).div_ceil(2048).max(1);
            raw.insert(ch, window_image(&path, TransferFunction::Linear, Window {
                x, y, width: w, height: h }, step)?);
        }
        let mut mono = BTreeMap::new();
        let mut normalized = BTreeMap::new();
        for (&ch, image) in &raw {
            if image.channels == 1 {
                let model = MagicMonoModel::fit(image);
                let mut sample = image.clone();
                model.apply(&mut sample, (0, 0), (image.width, image.height));
                normalized.insert(ch, sample);
                mono.insert(ch, model);
            }
        }
        let refs: BTreeMap<_, _> = raw.iter().map(|(&ch, image)| (ch, image)).collect();
        let (composed, _, _, _, _) = compose_palette_linear(&preview.palette, &refs, &normalized, None)?;
        let slots = match preview.palette.as_str() {
            "hoo" => [Some(Channel::Ha), Some(Channel::Oiii), Some(Channel::Oiii)],
            "sho" | "hubble" => [Some(Channel::Sii), Some(Channel::Ha), Some(Channel::Oiii)],
            "osh" => [Some(Channel::Oiii), Some(Channel::Sii), Some(Channel::Ha)],
            "natural" | "hubble-inspired" if raw.contains_key(&Channel::RgbCfa) => [None; 3],
            "natural" | "hubble-inspired" => [Some(Channel::R), Some(Channel::G), Some(Channel::B)],
            "monochrome" => {
                let ch = if raw.contains_key(&Channel::L) { Channel::L } else {
                    ensure!(raw.len() == 1, "Ambiguous monochrome channel");
                    *raw.keys().next().context("No monochrome channel")?
                };
                [Some(ch); 3]
            }
            id if id.starts_with("monochrome-") => {
                let ch = Channel::from_token(id.trim_start_matches("monochrome-"))
                    .context("Unknown monochrome filter")?;
                [Some(ch); 3]
            }
            _ => bail!("Native calibration unsupported for palette '{}'; no unsafe tile-local fallback", preview.palette),
        };
        Ok(Self { dimensions, slots, mono,
            stretch: Some(MagicStretchModel::fit(&composed, &preview.palette)) })
    }

    /// `tile` is the raw result of NativeTiles::read_tile at `(x,y)`.
    pub fn apply(&self, mut tile: LinearImage, x: usize, y: usize) -> Result<LinearImage> {
        checked_window(Window { x, y, width: tile.width, height: tile.height },
            self.dimensions[0], self.dimensions[1])?;
        let Some(stretch) = &self.stretch else { return Ok(triplet(tile)); };
        if self.slots.iter().any(Option::is_some) {
            ensure!(tile.channels == 3, "Gallery native tile is not RGB");
            let n = tile.pixel_count();
            for c in 0..3 {
                if let Some(ch) = self.slots[c] {
                    let model = self.mono.get(&ch).context("Missing native filter calibration")?;
                    let mut plane = tile.extract_channel(c);
                    model.apply(&mut plane, (x, y), (self.dimensions[0], self.dimensions[1]));
                    tile.data[c * n..(c + 1) * n].copy_from_slice(&plane.data);
                }
            }
        }
        Ok(stretch.apply(tile, (x, y), (self.dimensions[0], self.dimensions[1])))
    }
}
