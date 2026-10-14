use crate::calibrate::{SpooledFrame, read_spooled_region};
use crate::config::LinearImage;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DetectedStar {
    pub x: f32,
    pub y: f32,
    pub flux: f32,
    pub peak: f32,
    pub fwhm: f32,
    pub eccentricity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameQuality {
    pub stars: Vec<DetectedStar>,
    pub star_count: usize,
    pub median_fwhm: f32,
    pub median_eccentricity: f32,
    pub sharpness_score: f32,
}

/// 2D affine transform mapping reference coordinates `(x_ref, y_ref)` to source frame coordinates `(x_src, y_src)`:
/// `x_src = a00 * x_ref + a01 * y_ref + tx`
/// `y_src = a10 * x_ref + a11 * y_ref + ty`
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffineTransform {
    pub a00: f32,
    pub a01: f32,
    pub tx: f32,
    pub a10: f32,
    pub a11: f32,
    pub ty: f32,
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl AffineTransform {
    pub fn identity() -> Self {
        Self {
            a00: 1.0,
            a01: 0.0,
            tx: 0.0,
            a10: 0.0,
            a11: 1.0,
            ty: 0.0,
        }
    }

    pub fn is_identity(&self) -> bool {
        (self.a00 - 1.0).abs() < 1e-5
            && self.a01.abs() < 1e-5
            && self.tx.abs() < 1e-4
            && self.a10.abs() < 1e-5
            && (self.a11 - 1.0).abs() < 1e-5
            && self.ty.abs() < 1e-4
    }

    #[inline]
    pub fn ref_to_src(&self, x_ref: f32, y_ref: f32) -> (f32, f32) {
        (
            self.a00 * x_ref + self.a01 * y_ref + self.tx,
            self.a10 * x_ref + self.a11 * y_ref + self.ty,
        )
    }

    pub fn determinant(&self) -> f32 {
        self.a00 * self.a11 - self.a01 * self.a10
    }
}

#[derive(Debug, Clone)]
pub struct RegistrationMatch {
    pub transform: AffineTransform,
    pub inliers: usize,
    pub rms: f32,
    pub overlap_fraction: f32,
}

/// Computes median and unscaled MAD of valid finite pixels.
/// Subsamples deterministically if `data.len() > 262_144` to bound CPU and memory.
pub fn robust_median_mad(data: &[f32], mask: &[u8]) -> (f32, f32) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let m_len = mask.len().max(1);
    let step = (n / 131_072).max(1);
    let mut vals = Vec::with_capacity(n / step + 1);
    for i in (0..n).step_by(step) {
        let m = mask[i % m_len];
        let v = data[i];
        if m != 0 && v.is_finite() {
            vals.push(v);
        }
    }
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mid = vals.len() / 2;
    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let med = vals[mid];

    for v in &mut vals {
        *v = (*v - med).abs();
    }
    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let mad = vals[mid];
    (med, mad)
}

/// Detects unsaturated local-maxima stars above `median + 5 * MAD` with sub-pixel centroid, FWHM, and eccentricity.
pub fn detect_stars(img: &LinearImage) -> Vec<DetectedStar> {
    let w = img.width;
    let h = img.height;
    if w < 9 || h < 9 {
        return Vec::new();
    }
    let luminance = (img.channels >= 3).then(|| img.to_luminance());
    let lum = luminance.as_ref().unwrap_or(img);
    let plane = lum.plane(0);
    let (med, mad) = robust_median_mad(plane, &lum.mask);
    let sigma = (mad * 1.4826).max(1e-4);
    let threshold = med + 5.0 * sigma;
    let neighbor_req_thresh = med + 1.2 * sigma;

    // Determine saturation ceiling from peak distribution
    let max_val = plane
        .par_iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(|| 0.0f32, f32::max);
    let sat_limit = if max_val > 60000.0 {
        62000.0
    } else {
        f32::INFINITY
    };

    let border = 3usize;
    // Indexed stripes preserve scan order for equal-flux stable-sort ties.
    const ROWS_PER_STRIPE: usize = 32;
    let stripes: Vec<Vec<DetectedStar>> = (0..(h - 2 * border).div_ceil(ROWS_PER_STRIPE))
        .into_par_iter()
        .map(|stripe| {
            let mut candidates = Vec::new();
            let y_start = border + stripe * ROWS_PER_STRIPE;
            let y_end = (y_start + ROWS_PER_STRIPE).min(h - border);
            for y in y_start..y_end {
                for x in border..(w - border) {
                    let idx = y * w + x;
                    if lum.mask[idx] == 0 {
                        continue;
                    }
                    let c_val = plane[idx];
                    if !c_val.is_finite() || c_val <= threshold || c_val >= sat_limit {
                        continue;
                    }

                    // Local maximum in 3x3 neighborhood and not a single-pixel hot spike
                    let mut is_local_max = true;
                    let mut elevated_neighbors = 0usize;
                    'nbr: for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let ny = (y as isize + dy) as usize;
                            let nx = (x as isize + dx) as usize;
                            let nv = plane[ny * w + nx];
                            if nv > c_val || (nv == c_val && (dy < 0 || (dy == 0 && dx < 0))) {
                                is_local_max = false;
                                break 'nbr;
                            }
                            if nv > neighbor_req_thresh {
                                elevated_neighbors += 1;
                            }
                        }
                    }
                    if !is_local_max || elevated_neighbors < 2 {
                        continue;
                    }

                    // Compute flux-weighted centroid and second moments in a 5x5 / 7x7 window
                    let r = 3isize;
                    let mut sum_w = 0.0f32;
                    let mut sum_x = 0.0f32;
                    let mut sum_y = 0.0f32;
                    let mut saturated = false;

                    for dy in -r..=r {
                        for dx in -r..=r {
                            let ny = (y as isize + dy) as usize;
                            let nx = (x as isize + dx) as usize;
                            let p_idx = ny * w + nx;
                            if lum.mask[p_idx] == 0 {
                                continue;
                            }
                            let pv = plane[p_idx];
                            if pv >= sat_limit {
                                saturated = true;
                                break;
                            }
                            let w_val = (pv - med).max(0.0);
                            if w_val > 1.5 * sigma {
                                sum_w += w_val;
                                sum_x += w_val * (nx as f32);
                                sum_y += w_val * (ny as f32);
                            }
                        }
                    }
                    if saturated || sum_w <= 0.0 {
                        continue;
                    }
                    let cx = sum_x / sum_w;
                    let cy = sum_y / sum_w;

                    let mut m_xx = 0.0f32;
                    let mut m_yy = 0.0f32;
                    let mut m_xy = 0.0f32;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let ny = (y as isize + dy) as usize;
                            let nx = (x as isize + dx) as usize;
                            let w_val = (plane[ny * w + nx] - med).max(0.0);
                            if w_val > 1.5 * sigma {
                                let rx = (nx as f32) - cx;
                                let ry = (ny as f32) - cy;
                                m_xx += w_val * rx * rx;
                                m_yy += w_val * ry * ry;
                                m_xy += w_val * rx * ry;
                            }
                        }
                    }
                    m_xx /= sum_w;
                    m_yy /= sum_w;
                    m_xy /= sum_w;

                    let trace = m_xx + m_yy;
                    let diff = m_xx - m_yy;
                    let disc = (diff * diff + 4.0 * m_xy * m_xy).sqrt();
                    let lambda_max = (0.5 * (trace + disc)).max(0.05);
                    let lambda_min = (0.5 * (trace - disc)).max(0.02);

                    let fwhm = 2.3548 * (0.5 * trace).max(0.04).sqrt();
                    let eccentricity = (1.0 - (lambda_min / lambda_max).clamp(0.0, 1.0)).sqrt();
                    let peak_above_bg = c_val - med;
                    let horiz_ridge = (plane[y * w + (x - 3)] - med) > 0.55 * peak_above_bg
                        && (plane[y * w + (x + 3)] - med) > 0.55 * peak_above_bg;
                    let vert_ridge = (plane[(y - 3) * w + x] - med) > 0.55 * peak_above_bg
                        && (plane[(y + 3) * w + x] - med) > 0.55 * peak_above_bg;
                    if eccentricity > 0.85 || horiz_ridge || vert_ridge {
                        continue;
                    }
                    candidates.push(DetectedStar {
                        x: cx,
                        y: cy,
                        flux: sum_w,
                        peak: c_val - med,
                        fwhm,
                        eccentricity,
                    });
                }
            }
            candidates
        })
        .collect();
    let mut raw_candidates = Vec::new();
    for stripe in stripes {
        raw_candidates.extend(stripe);
    }

    // Sort brightest first and suppress close duplicates (< 3.5 px)
    raw_candidates.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    let mut stars: Vec<DetectedStar> = Vec::with_capacity(raw_candidates.len().min(100));
    for cand in raw_candidates {
        let too_close = stars.iter().any(|s| {
            let dx = s.x - cand.x;
            let dy = s.y - cand.y;
            dx * dx + dy * dy < 12.25
        });
        if !too_close {
            stars.push(cand);
            if stars.len() >= 100 {
                break;
            }
        }
    }
    stars
}

/// Measures native-resolution stellar centroids and PSF moments.
/// Holds one monochrome frame at a time; max-pooled thumbnails cannot measure FWHM
/// or provide reliable subpixel coordinates for undersampled stars in wide fields.
pub fn evaluate_spooled_frame_quality(spooled: &SpooledFrame) -> Result<FrameQuality> {
    let (data, mask) = read_spooled_region(
        &spooled.spool_path,
        spooled.width,
        spooled.height,
        0,
        0,
        0,
        spooled.width,
        spooled.height,
    )?;
    let image = LinearImage {
        width: spooled.width,
        height: spooled.height,
        channels: 1,
        data,
        mask,
        bayer: None,
        exposure: spooled.exposure,
    };
    let stars = detect_stars(&image);

    let star_count = stars.len();
    if star_count == 0 {
        return Ok(FrameQuality {
            stars,
            star_count: 0,
            median_fwhm: 99.0,
            median_eccentricity: 1.0,
            sharpness_score: 0.0,
        });
    }

    let mut fwhms: Vec<f32> = stars.iter().map(|s| s.fwhm).collect();
    fwhms.sort_by(|a, b| a.total_cmp(b));
    let median_fwhm = fwhms[fwhms.len() / 2];

    let mut eccs: Vec<f32> = stars.iter().map(|s| s.eccentricity).collect();
    eccs.sort_by(|a, b| a.total_cmp(b));
    let median_eccentricity = eccs[eccs.len() / 2];

    let sharpness_score =
        (star_count as f32).sqrt() / (median_fwhm.max(0.5) * (1.0 + median_eccentricity));

    Ok(FrameQuality {
        stars,
        star_count,
        median_fwhm,
        median_eccentricity,
        sharpness_score,
    })
}

#[derive(Clone, Copy)]
struct StarTriangle {
    i: usize,
    j: usize,
    k: usize,
    r1: f32, // a / c
    r2: f32, // b / c
    perim: f32,
}

fn build_triangles(stars: &[DetectedStar], max_stars: usize) -> Vec<StarTriangle> {
    let n = stars.len().min(max_stars);
    let mut tris = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let d_ij =
                    ((stars[i].x - stars[j].x).powi(2) + (stars[i].y - stars[j].y).powi(2)).sqrt();
                let d_jk =
                    ((stars[j].x - stars[k].x).powi(2) + (stars[j].y - stars[k].y).powi(2)).sqrt();
                let d_ki =
                    ((stars[k].x - stars[i].x).powi(2) + (stars[k].y - stars[i].y).powi(2)).sqrt();

                // Order vertices so p0 is opposite shortest side, p2 is opposite longest side
                let mut sides = [(d_jk, i), (d_ki, j), (d_ij, k)];
                sides.sort_by(|a, b| a.0.total_cmp(&b.0));
                let a = sides[0].0;
                let b = sides[1].0;
                let c = sides[2].0;
                if c < 4.0 || a < 1.5 {
                    continue;
                }
                let r1 = a / c;
                let r2 = b / c;
                // Avoid nearly degenerate collinear triangles
                if a + b <= 1.02 * c {
                    continue;
                }
                tris.push(StarTriangle {
                    i: sides[0].1,
                    j: sides[1].1,
                    k: sides[2].1,
                    r1,
                    r2,
                    perim: a + b + c,
                });
            }
        }
    }
    tris
}

fn solve_3x3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut pivot = col;
        for row in (col + 1)..3 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        if m[pivot][col].abs() < 1e-9 {
            return None;
        }
        m.swap(col, pivot);
        let div = m[col][col];
        for c in col..4 {
            m[col][c] /= div;
        }
        for row in 0..3 {
            if row != col {
                let factor = m[row][col];
                for c in col..4 {
                    m[row][c] -= factor * m[col][c];
                }
            }
        }
    }
    Some([m[0][3], m[1][3], m[2][3]])
}

fn fit_affine_from_pairs(pairs: &[((f32, f32), (f32, f32))]) -> Option<AffineTransform> {
    if pairs.len() < 3 {
        return None;
    }
    let mut ata = [[0.0f64; 3]; 3];
    let mut atbx = [0.0f64; 3];
    let mut atby = [0.0f64; 3];

    for &((rx, ry), (sx, sy)) in pairs {
        let row = [f64::from(rx), f64::from(ry), 1.0];
        let bx = f64::from(sx);
        let by = f64::from(sy);
        for r in 0..3 {
            atbx[r] += row[r] * bx;
            atby[r] += row[r] * by;
            for c in 0..3 {
                ata[r][c] += row[r] * row[c];
            }
        }
    }

    let mx = [
        [ata[0][0], ata[0][1], ata[0][2], atbx[0]],
        [ata[1][0], ata[1][1], ata[1][2], atbx[1]],
        [ata[2][0], ata[2][1], ata[2][2], atbx[2]],
    ];
    let my = [
        [ata[0][0], ata[0][1], ata[0][2], atby[0]],
        [ata[1][0], ata[1][1], ata[1][2], atby[1]],
        [ata[2][0], ata[2][1], ata[2][2], atby[2]],
    ];

    let sx_params = solve_3x3(mx)?;
    let sy_params = solve_3x3(my)?;

    let tf = AffineTransform {
        a00: sx_params[0] as f32,
        a01: sx_params[1] as f32,
        tx: sx_params[2] as f32,
        a10: sy_params[0] as f32,
        a11: sy_params[1] as f32,
        ty: sy_params[2] as f32,
    };
    let det = tf.determinant();
    if !det.is_finite() || det < 0.8 || det > 1.25 {
        return None;
    }
    Some(tf)
}

fn evaluate_inliers(
    ref_stars: &[DetectedStar],
    src_stars: &[DetectedStar],
    tf: &AffineTransform,
    tol_px: f32,
) -> (Vec<((f32, f32), (f32, f32))>, f32) {
    let mut used_src = vec![false; src_stars.len()];
    let mut inliers = Vec::with_capacity(ref_stars.len());
    let rms = evaluate_inliers_into(
        ref_stars,
        src_stars,
        tf,
        tol_px,
        &mut used_src,
        &mut inliers,
    );
    (inliers, rms)
}

fn evaluate_inliers_into(
    ref_stars: &[DetectedStar],
    src_stars: &[DetectedStar],
    tf: &AffineTransform,
    tol_px: f32,
    used_src: &mut [bool],
    inliers: &mut Vec<((f32, f32), (f32, f32))>,
) -> f32 {
    used_src.fill(false);
    inliers.clear();
    let tol_sq = tol_px * tol_px;
    let mut sq_err_sum = 0.0f32;

    for r in ref_stars {
        let (px, py) = tf.ref_to_src(r.x, r.y);
        let mut best_j = None;
        let mut best_d2 = tol_sq;
        for (j, s) in src_stars.iter().enumerate() {
            if used_src[j] {
                continue;
            }
            let dx = s.x - px;
            let dy = s.y - py;
            let d2 = dx * dx + dy * dy;
            if d2 <= best_d2 {
                best_d2 = d2;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used_src[j] = true;
            inliers.push(((r.x, r.y), (src_stars[j].x, src_stars[j].y)));
            sq_err_sum += best_d2;
        }
    }

    let rms = if inliers.is_empty() {
        f32::INFINITY
    } else {
        (sq_err_sum / inliers.len() as f32).sqrt()
    };
    rms
}

pub fn compute_overlap_fraction(tf: &AffineTransform, width: usize, height: usize) -> f32 {
    let w = width as f32;
    let h = height as f32;
    let steps = 32usize;
    let mut inside = 0usize;
    let total = steps * steps;
    for sy in 0..steps {
        let ry = ((sy as f32) + 0.5) * h / (steps as f32);
        for sx in 0..steps {
            let rx = ((sx as f32) + 0.5) * w / (steps as f32);
            let (x_src, y_src) = tf.ref_to_src(rx, ry);
            if x_src >= 0.0 && x_src < w && y_src >= 0.0 && y_src < h {
                inside += 1;
            }
        }
    }
    (inside as f32) / (total as f32)
}

/// Matches two star fields using geometric star triangles and RANSAC affine transform fitting.
/// Rejects transforms with `< 12` inlier stars or `> 2.0` px RMS.
pub fn match_star_fields(
    ref_stars: &[DetectedStar],
    src_stars: &[DetectedStar],
    width: usize,
    height: usize,
) -> Result<RegistrationMatch> {
    if ref_stars.len() < 12 || src_stars.len() < 12 {
        bail!(
            "Insufficient stars for triangle RANSAC alignment: ref={}, target={} (minimum 12 required)",
            ref_stars.len(),
            src_stars.len()
        );
    }

    let mut hypotheses: Vec<AffineTransform> = Vec::with_capacity(256);
    hypotheses.push(AffineTransform::identity());

    // 1. Pairwise translation hypotheses from top 20 stars
    let n_trans = ref_stars.len().min(20);
    let m_trans = src_stars.len().min(20);
    for r in &ref_stars[..n_trans] {
        for s in &src_stars[..m_trans] {
            hypotheses.push(AffineTransform {
                a00: 1.0,
                a01: 0.0,
                tx: s.x - r.x,
                a10: 0.0,
                a11: 1.0,
                ty: s.y - r.y,
            });
        }
    }

    // Compare triangle descriptors across the entire brightness-ranked sample.
    // Iteration-order truncation favors triangles containing the brightest reference
    // star, which may be saturated or absent in another exposure. Indexing source
    // triangles by their shortest/longest-side ratio keeps this bounded without
    // enumerating every pair of triangles.
    let ref_tris = build_triangles(ref_stars, 32);
    let mut src_tris = build_triangles(src_stars, 32);
    src_tris.sort_by(|a, b| a.r1.total_cmp(&b.r1));

    let mut proposals = Vec::new();
    for rt in &ref_tris {
        let start = src_tris.partition_point(|st| st.r1 < rt.r1 - 0.025);
        let mut nearest: Vec<(f32, usize)> = Vec::with_capacity(4);
        for (offset, st) in src_tris[start..].iter().enumerate() {
            if st.r1 > rt.r1 + 0.025 {
                break;
            }
            let dr2 = (rt.r2 - st.r2).abs();
            let dp = (rt.perim - st.perim).abs() / rt.perim.max(st.perim);
            if dr2 > 0.025 || dp > 0.05 {
                continue;
            }
            let dr1 = rt.r1 - st.r1;
            let score = (dr1 * dr1 + dr2 * dr2 + dp * dp) / rt.perim.sqrt();
            let index = start + offset;
            let pos = nearest.partition_point(|&(other, _)| other <= score);
            if pos < 4 {
                nearest.insert(pos, (score, index));
                nearest.truncate(4);
            }
        }
        for (score, index) in nearest {
            proposals.push((score, *rt, src_tris[index]));
        }
    }
    proposals.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Spread the evaluation budget among different stars, rather than letting
    // thousands of near-identical descriptors of one bright star fill it.
    let mut ref_uses = vec![0usize; ref_stars.len().min(32)];
    let mut src_uses = vec![0usize; src_stars.len().min(32)];
    for (_, rt, st) in proposals {
        if [rt.i, rt.j, rt.k].iter().any(|&i| ref_uses[i] >= 120)
            || [st.i, st.j, st.k].iter().any(|&i| src_uses[i] >= 120)
        {
            continue;
        }
        let pairs = [
            (
                (ref_stars[rt.i].x, ref_stars[rt.i].y),
                (src_stars[st.i].x, src_stars[st.i].y),
            ),
            (
                (ref_stars[rt.j].x, ref_stars[rt.j].y),
                (src_stars[st.j].x, src_stars[st.j].y),
            ),
            (
                (ref_stars[rt.k].x, ref_stars[rt.k].y),
                (src_stars[st.k].x, src_stars[st.k].y),
            ),
        ];
        if let Some(tf) = fit_affine_from_pairs(&pairs) {
            hypotheses.push(tf);
            for i in [rt.i, rt.j, rt.k] {
                ref_uses[i] += 1;
            }
            for i in [st.i, st.j, st.k] {
                src_uses[i] += 1;
            }
            if hypotheses.len() >= 1200 {
                break;
            }
        }
    }

    // Score hypotheses in parallel with reusable worker buffers. Resolve exact
    // ties by original index so reference selection remains deterministic.
    let best = hypotheses
        .par_iter()
        .enumerate()
        .with_min_len(16)
        .map_init(
            || {
                (
                    vec![false; src_stars.len()],
                    Vec::with_capacity(ref_stars.len()),
                )
            },
            |(used, inliers), (index, candidate)| {
                let rms =
                    evaluate_inliers_into(ref_stars, src_stars, candidate, 2.5, used, inliers);
                (index, inliers.len(), rms)
            },
        )
        .max_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| b.0.cmp(&a.0))
        })
        .context("Registration has no hypotheses")?;
    let mut best_tf = hypotheses[best.0];
    let (mut best_inliers, _) = evaluate_inliers(ref_stars, src_stars, &best_tf, 2.5);

    if best_inliers.len() < 12 {
        bail!(
            "Star-field RANSAC failed: found only {} inlier stars (minimum 12 required)",
            best_inliers.len()
        );
    }

    // Least-squares refinement on inliers (2 iterations)
    for _ in 0..2 {
        if let Some(refined) = fit_affine_from_pairs(&best_inliers) {
            let (inl, _rms) = evaluate_inliers(ref_stars, src_stars, &refined, 2.0);
            if inl.len() >= 12 {
                best_tf = refined;
                best_inliers = inl;
            }
        }
    }

    // Re-evaluate final inliers at strict 2.0 px threshold
    let (final_inliers, final_rms) = evaluate_inliers(ref_stars, src_stars, &best_tf, 2.0);
    if final_inliers.len() < 12 {
        bail!(
            "Star-field registration rejected: {} inliers within 2.0 px (minimum 12 required)",
            final_inliers.len()
        );
    }
    if !final_rms.is_finite() || final_rms > 2.0 {
        bail!(
            "Star-field registration rejected: RMS residual {:.3} px exceeds 2.0 px limit",
            final_rms
        );
    }

    let overlap_fraction = compute_overlap_fraction(&best_tf, width, height);
    if overlap_fraction < 0.50 {
        bail!(
            "Star-field registration rejected: overlap fraction {:.1}% is below minimum 50%",
            overlap_fraction * 100.0
        );
    }

    Ok(RegistrationMatch {
        transform: best_tf,
        inliers: final_inliers.len(),
        rms: final_rms,
        overlap_fraction,
    })
}

#[inline]
fn sinc(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 1e-5 {
        1.0
    } else {
        let pix = std::f32::consts::PI * x;
        pix.sin() / pix
    }
}

#[inline]
fn lanczos3_weight(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Warps a tile `[tx0..tx1, ty0..ty1]` of channel `c` from a spooled frame into the reference grid
/// using Lanczos-3 interpolation. Reads only the needed source bounding box from the spool file.
#[allow(clippy::too_many_arguments)]
pub fn warp_spooled_tile_lanczos3(
    spooled: &SpooledFrame,
    c: usize,
    tf: &AffineTransform,
    tx0: usize,
    ty0: usize,
    tx1: usize,
    ty1: usize,
) -> Result<(Vec<f32>, Vec<u8>)> {
    let tw = tx1 - tx0;
    let th = ty1 - ty0;
    let full_w = spooled.width;
    let full_h = spooled.height;

    if tf.is_identity() {
        return read_spooled_region(&spooled.spool_path, full_w, full_h, c, tx0, ty0, tx1, ty1);
    }

    // Compute source bounding box from the 4 corners of the reference tile
    let corners = [
        (tx0 as f32, ty0 as f32),
        ((tx1 - 1) as f32, ty0 as f32),
        (tx0 as f32, (ty1 - 1) as f32),
        ((tx1 - 1) as f32, (ty1 - 1) as f32),
    ];
    let mut min_sx = f32::INFINITY;
    let mut max_sx = f32::NEG_INFINITY;
    let mut min_sy = f32::INFINITY;
    let mut max_sy = f32::NEG_INFINITY;
    for (rx, ry) in corners {
        let (sx, sy) = tf.ref_to_src(rx, ry);
        min_sx = min_sx.min(sx);
        max_sx = max_sx.max(sx);
        min_sy = min_sy.min(sy);
        max_sy = max_sy.max(sy);
    }

    // Expand by Lanczos-3 support radius (3 px + margin)
    let sx0_i = (min_sx.floor() as isize - 3).clamp(0, full_w as isize) as usize;
    let sy0_i = (min_sy.floor() as isize - 3).clamp(0, full_h as isize) as usize;
    let sx1_i = (max_sx.ceil() as isize + 4).clamp(0, full_w as isize) as usize;
    let sy1_i = (max_sy.ceil() as isize + 4).clamp(0, full_h as isize) as usize;

    if sx0_i >= sx1_i || sy0_i >= sy1_i {
        return Ok((vec![0.0f32; tw * th], vec![0u8; tw * th]));
    }

    let (src_patch, src_mask) = read_spooled_region(
        &spooled.spool_path,
        full_w,
        full_h,
        c,
        sx0_i,
        sy0_i,
        sx1_i,
        sy1_i,
    )?;
    let pw = sx1_i - sx0_i;

    let mut out_data = vec![0.0f32; tw * th];
    let mut out_mask = vec![0u8; tw * th];

    for ly in 0..th {
        let ry = (ty0 + ly) as f32;
        for lx in 0..tw {
            let rx = (tx0 + lx) as f32;
            let (sx, sy) = tf.ref_to_src(rx, ry);

            if sx < 0.0 || sx > (full_w - 1) as f32 || sy < 0.0 || sy > (full_h - 1) as f32 {
                continue;
            }

            let ix_center = sx.floor() as isize;
            let iy_center = sy.floor() as isize;

            let mut wx_arr = [0.0f32; 6];
            let mut wy_arr = [0.0f32; 6];
            for k in 0..6 {
                let dx = sx - ((ix_center - 2 + k as isize) as f32);
                let dy = sy - ((iy_center - 2 + k as isize) as f32);
                wx_arr[k] = lanczos3_weight(dx);
                wy_arr[k] = lanczos3_weight(dy);
            }

            let mut sum_val = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut valid_samples = 0usize;

            for (ky, &wy) in wy_arr.iter().enumerate() {
                let ny = iy_center - 2 + ky as isize;
                if ny < sy0_i as isize || ny >= sy1_i as isize {
                    continue;
                }
                let py = (ny as usize) - sy0_i;
                for (kx, &wx) in wx_arr.iter().enumerate() {
                    let nx = ix_center - 2 + kx as isize;
                    if nx < sx0_i as isize || nx >= sx1_i as isize {
                        continue;
                    }
                    let px = (nx as usize) - sx0_i;
                    let p_idx = py * pw + px;
                    if src_mask[p_idx] != 0 {
                        let w = wx * wy;
                        sum_val += w * src_patch[p_idx];
                        sum_w += w;
                        valid_samples += 1;
                    }
                }
            }

            if valid_samples >= 4 && sum_w.abs() > 1e-4 {
                out_data[ly * tw + lx] = sum_val / sum_w;
                out_mask[ly * tw + lx] = 1;
            }
        }
    }

    Ok((out_data, out_mask))
}

/// Warps an in-memory `LinearImage` into a reference grid using `tf` with Lanczos-3 interpolation.
/// Used to align finished filter integrations to the target reference before palette combination.
pub fn warp_linear_image_lanczos3(img: &LinearImage, tf: &AffineTransform) -> LinearImage {
    if tf.is_identity() {
        return img.clone();
    }
    let w = img.width;
    let h = img.height;
    let npix = w * h;
    let mut out = LinearImage::new(w, h, img.channels);
    out.exposure = img.exposure;
    out.mask.fill(0);

    for y in 0..h {
        let ry = y as f32;
        for x in 0..w {
            let rx = x as f32;
            let (sx, sy) = tf.ref_to_src(rx, ry);
            if sx < 0.0 || sx > (w - 1) as f32 || sy < 0.0 || sy > (h - 1) as f32 {
                continue;
            }
            let ix_center = sx.floor() as isize;
            let iy_center = sy.floor() as isize;

            let mut wx_arr = [0.0f32; 6];
            let mut wy_arr = [0.0f32; 6];
            for k in 0..6 {
                wx_arr[k] = lanczos3_weight(sx - ((ix_center - 2 + k as isize) as f32));
                wy_arr[k] = lanczos3_weight(sy - ((iy_center - 2 + k as isize) as f32));
            }

            let dst_p = y * w + x;
            for c in 0..img.channels {
                let plane = img.plane(c);
                let mut sum_val = 0.0f32;
                let mut sum_w = 0.0f32;
                let mut valid_samples = 0usize;
                for (ky, &wy) in wy_arr.iter().enumerate() {
                    let ny = iy_center - 2 + ky as isize;
                    if ny < 0 || ny >= h as isize {
                        continue;
                    }
                    for (kx, &wx) in wx_arr.iter().enumerate() {
                        let nx = ix_center - 2 + kx as isize;
                        if nx < 0 || nx >= w as isize {
                            continue;
                        }
                        let src_p = (ny as usize) * w + (nx as usize);
                        if img.mask[src_p] != 0 {
                            let weight = wx * wy;
                            sum_val += weight * plane[src_p];
                            sum_w += weight;
                            valid_samples += 1;
                        }
                    }
                }
                if valid_samples >= 4 && sum_w.abs() > 1e-4 {
                    out.data[c * npix + dst_p] = sum_val / sum_w;
                    out.mask[dst_p] = 1;
                }
            }
        }
    }
    out
}
