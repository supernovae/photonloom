use crate::config::{BayerPattern, Channel, FileFormat, Frame, FrameRole, LinearImage};
use anyhow::{Context, Result, bail};
use rsraw::RawImage;
use rsraw_sys as sys;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct RawHeaderMeta {
    pub width: usize,
    pub height: usize,
    pub raw_width: usize,
    pub raw_height: usize,
    pub top_margin: usize,
    pub left_margin: usize,
    pub bayer: Option<BayerPattern>,
    pub make: String,
    pub model: String,
    pub iso: Option<f32>,
    pub shutter: Option<f32>,
    pub black_level: f32,
    pub white_level: f32,
    pub cblack: [f32; 4],
}

fn resolve_bayer_from_libraw(raw: &mut RawImage, path: &Path) -> Result<Option<BayerPattern>> {
    let data_ref: &sys::libraw_data_t = raw.as_ref();
    if data_ref.idata.is_foveon != 0 {
        bail!(
            "Unsupported Foveon sensor in RAW file '{}': only Bayer or monochrome sensors are supported",
            path.display()
        );
    }
    let filters = data_ref.idata.filters;
    if filters == 9 {
        bail!(
            "Unsupported X-Trans CFA in RAW file '{}': X-Trans sensors are not supported",
            path.display()
        );
    }
    if filters == 1 {
        bail!(
            "Unsupported Leaf CatchLight CFA in RAW file '{}'",
            path.display()
        );
    }
    if filters == 0 {
        if data_ref.idata.colors == 1 {
            return Ok(None);
        }
        bail!(
            "Unsupported pre-demosaiced or linear multi-channel RAW file '{}' (colors={}, filters=0)",
            path.display(),
            data_ref.idata.colors
        );
    }
    if filters < 1000 {
        bail!(
            "Unsupported non-Bayer CFA filter mask ({}) in RAW file '{}'",
            filters,
            path.display()
        );
    }

    let cdesc_bytes: Vec<u8> = data_ref
        .idata
        .cdesc
        .iter()
        .take(4)
        .map(|&c| (c as u8).to_ascii_uppercase())
        .collect();
    let top = i32::from(data_ref.sizes.top_margin);
    let left = i32::from(data_ref.sizes.left_margin);
    let raw_ptr: *mut sys::libraw_data_t = raw.as_mut();

    let mut pat_chars = [b'?'; 4];
    for (idx, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].iter().enumerate() {
        let c_idx = unsafe { sys::libraw_COLOR(raw_ptr, top + dy, left + dx) };
        if !(0..4).contains(&c_idx) {
            bail!(
                "Invalid CFA color index {} returned by LibRaw for '{}'",
                c_idx,
                path.display()
            );
        }
        let ch = cdesc_bytes[c_idx as usize];
        pat_chars[idx] = match ch {
            b'R' => b'R',
            b'G' | b'E' => b'G', // LibRaw sometimes labels second green as G or E; verify RGBG
            b'B' => b'B',
            other => {
                bail!(
                    "Unsupported non-RGB CFA channel '{}' in RAW file '{}'",
                    other as char,
                    path.display()
                );
            }
        };
    }
    if cdesc_bytes.contains(&b'E')
        || cdesc_bytes.contains(&b'C')
        || cdesc_bytes.contains(&b'M')
        || cdesc_bytes.contains(&b'Y')
    {
        bail!(
            "Unsupported non-RGB CFA descriptor '{}' in RAW file '{}'",
            String::from_utf8_lossy(&cdesc_bytes),
            path.display()
        );
    }
    let pat_str = std::str::from_utf8(&pat_chars).unwrap_or("");
    let Some(bayer) = BayerPattern::from_str_token(pat_str) else {
        bail!(
            "Unsupported Bayer pattern '{}' in RAW file '{}'",
            pat_str,
            path.display()
        );
    };
    Ok(Some(bayer))
}

pub fn inspect_raw_header(path: &Path) -> Result<RawHeaderMeta> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read RAW file '{}'", path.display()))?;
    let mut raw = RawImage::open(&bytes)
        .map_err(|e| anyhow::anyhow!("LibRaw failed to open '{}': {:?}", path.display(), e))?;

    let bayer = resolve_bayer_from_libraw(&mut raw, path)?;
    let make = raw.make().trim().to_string();
    let model = raw.model().trim().to_string();
    let iso = {
        let v = raw.iso_speed();
        if v > 0 { Some(v as f32) } else { None }
    };
    let shutter = {
        let s = raw.shutter();
        if s.is_finite() && s > 0.0 {
            Some(s)
        } else {
            None
        }
    };

    let data_ref: &sys::libraw_data_t = raw.as_ref();
    let width = usize::from(data_ref.sizes.width);
    let height = usize::from(data_ref.sizes.height);
    let raw_width = usize::from(data_ref.sizes.raw_width);
    let raw_height = usize::from(data_ref.sizes.raw_height);
    let top_margin = usize::from(data_ref.sizes.top_margin);
    let left_margin = usize::from(data_ref.sizes.left_margin);

    if width == 0
        || height == 0
        || top_margin + height > raw_height
        || left_margin + width > raw_width
    {
        bail!(
            "Invalid active area dimensions in RAW file '{}' (active={}x{} at +{},+{}; raw={}x{})",
            path.display(),
            width,
            height,
            left_margin,
            top_margin,
            raw_width,
            raw_height
        );
    }

    let black_level = data_ref.color.black as f32;
    let white_level = if data_ref.color.maximum > 0 {
        data_ref.color.maximum as f32
    } else {
        65535.0
    };
    let cblack = [
        data_ref.color.cblack[0] as f32,
        data_ref.color.cblack[1] as f32,
        data_ref.color.cblack[2] as f32,
        data_ref.color.cblack[3] as f32,
    ];

    Ok(RawHeaderMeta {
        width,
        height,
        raw_width,
        raw_height,
        top_margin,
        left_margin,
        bayer,
        make,
        model,
        iso,
        shutter,
        black_level,
        white_level,
        cblack,
    })
}

pub fn build_raw_frame(
    path: &Path,
    meta: RawHeaderMeta,
    role: FrameRole,
    channel: Option<Channel>,
) -> Frame {
    let sensor = if !meta.make.is_empty() || !meta.model.is_empty() {
        format!("{} {}", meta.make, meta.model).trim().to_string()
    } else {
        format!("RAW-{}x{}", meta.width, meta.height)
    };
    let resolved_channel = channel.or(match meta.bayer {
        Some(_) => Some(Channel::RgbCfa),
        None => Some(Channel::L),
    });

    Frame {
        path: path.to_path_buf(),
        format: FileFormat::Raw,
        role,
        channel: resolved_channel,
        width: meta.width,
        height: meta.height,
        channels: 1,
        exposure: meta.shutter,
        temperature: None,
        gain_or_iso: meta.iso,
        binning: (1, 1),
        bayer: meta.bayer,
        sensor,
        object: None,
        black_level: meta.black_level,
        white_level: meta.white_level,
        cblack: meta.cblack,
    }
}

/// Unpacks native mosaic pixels via `rsraw::RawImage::open`/`unpack` and `rsraw_sys` rawdata pointer,
/// subtracting camera black level (`black + cblack[cfa]`) on the active sensor area.
/// Never calls `RawImage::process`.
pub fn read_raw_mosaic(frame: &Frame) -> Result<LinearImage> {
    let bytes = std::fs::read(&frame.path)
        .with_context(|| format!("Failed to read RAW file '{}'", frame.path.display()))?;
    let mut raw = RawImage::open(&bytes).map_err(|e| {
        anyhow::anyhow!("LibRaw failed to open '{}': {:?}", frame.path.display(), e)
    })?;
    let bayer = resolve_bayer_from_libraw(&mut raw, &frame.path)?;
    raw.unpack().map_err(|e| {
        anyhow::anyhow!(
            "LibRaw failed to unpack mosaic for '{}': {:?}",
            frame.path.display(),
            e
        )
    })?;

    let data_ref: &sys::libraw_data_t = raw.as_ref();
    let width = usize::from(data_ref.sizes.width);
    let height = usize::from(data_ref.sizes.height);
    let raw_width = usize::from(data_ref.sizes.raw_width);
    let raw_height = usize::from(data_ref.sizes.raw_height);
    let top_margin = usize::from(data_ref.sizes.top_margin);
    let left_margin = usize::from(data_ref.sizes.left_margin);
    let pitch_elems = if data_ref.sizes.raw_pitch > 0 {
        (data_ref.sizes.raw_pitch as usize) / std::mem::size_of::<u16>()
    } else {
        raw_width
    };

    let raw_ptr = data_ref.rawdata.raw_image;
    if raw_ptr.is_null() {
        bail!(
            "LibRaw unpack did not produce a native Bayer/mono mosaic buffer (raw_image is null) for '{}'",
            frame.path.display()
        );
    }

    let total_raw_elems = pitch_elems.max(raw_width) * raw_height;
    let raw_slice = unsafe { std::slice::from_raw_parts(raw_ptr, total_raw_elems) };

    let base_black = data_ref.color.black as f32;
    let cblack = [
        data_ref.color.cblack[0] as f32,
        data_ref.color.cblack[1] as f32,
        data_ref.color.cblack[2] as f32,
        data_ref.color.cblack[3] as f32,
    ];
    let cblack_grid_w = data_ref.color.cblack[4] as usize;
    let cblack_grid_h = data_ref.color.cblack[5] as usize;

    let cblack_all = data_ref.color.cblack;
    let raw_mut_ptr: *mut sys::libraw_data_t = raw.as_mut();
    let mut color_lut = [[0usize; 2]; 2];
    for dy in 0..2 {
        for dx in 0..2 {
            let c = unsafe {
                sys::libraw_COLOR(
                    raw_mut_ptr,
                    (top_margin + dy) as i32,
                    (left_margin + dx) as i32,
                )
            };
            color_lut[dy][dx] = (c.clamp(0, 3)) as usize;
        }
    }

    let mut data = vec![0.0f32; width * height];
    let mask = vec![1u8; width * height];

    for y in 0..height {
        let ry = top_margin + y;
        let row_offset = ry * pitch_elems;
        for x in 0..width {
            let rx = left_margin + x;
            let raw_val = f32::from(raw_slice[row_offset + rx]);
            let c_idx = color_lut[y & 1][x & 1];
            let mut black = base_black + cblack[c_idx];
            if cblack_grid_w > 0 && cblack_grid_h > 0 && cblack_grid_w * cblack_grid_h <= 4098 {
                let grid_idx = 6 + (ry % cblack_grid_h) * cblack_grid_w + (rx % cblack_grid_w);
                black += cblack_all[grid_idx] as f32;
            }
            data[y * width + x] = raw_val - black;
        }
    }

    Ok(LinearImage {
        width,
        height,
        channels: 1,
        data,
        mask,
        bayer,
        exposure: frame.exposure.unwrap_or(1.0).max(1e-6),
    })
}
