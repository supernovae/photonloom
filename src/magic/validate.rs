//! Measured, fail-closed safety gates for Magic display candidates.
use crate::{config::LinearImage, register::detect_stars};
use serde::{Deserialize, Serialize};
use super::analysis::{ImageAnalysis, MaskAtlas};

pub const QUALITY_VERSION: u32 = 1;
const CLIP_FRACTION: f32 = 0.001;
const NOISE_RATIO: f32 = 1.2;
const FWHM_RATIO: f32 = 0.10;
const FAINT_RETENTION: f32 = 0.95;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub version: u32,
    pub passed: bool,
    pub reasons: Vec<String>,
    pub valid_pixels: usize,
    pub newly_clipped_pixels: usize,
    pub newly_clipped_channels: [usize; 3],
    pub clipped_fraction: f32,
    pub preexisting_clipped_pixels: usize,
    pub sky_noise_ratio: Option<f32>,
    pub sky_high_frequency_ratio: Option<f32>,
    pub matched_stars: usize,
    pub star_fwhm_ratio: Option<f32>,
    pub bright_halo_overshoot_sigma: Option<f32>,
    pub faint_signal_retention: Option<f32>,
    pub mask_edge_seam_sigma: Option<f32>,
    pub oversaturated_fraction: f32,
    pub max_new_clipping_fraction: f32,
    pub max_sky_noise_ratio: f32,
    pub max_fwhm_change_fraction: f32,
    pub max_halo_overshoot_sigma: f32,
    pub min_faint_signal_retention: f32,
    pub max_mask_edge_seam_sigma: f32,
}

impl Default for QualityReport {
    fn default() -> Self {
        Self { version: QUALITY_VERSION, passed: false, reasons: Vec::new(), valid_pixels:0,
            newly_clipped_pixels:0, newly_clipped_channels:[0;3],clipped_fraction:0.0,
            preexisting_clipped_pixels:0,sky_noise_ratio:None,sky_high_frequency_ratio:None,
            matched_stars:0,star_fwhm_ratio:None,bright_halo_overshoot_sigma:None,
            faint_signal_retention:None,mask_edge_seam_sigma:None,oversaturated_fraction:0.0,
            max_new_clipping_fraction:CLIP_FRACTION,max_sky_noise_ratio:NOISE_RATIO,
            max_fwhm_change_fraction:FWHM_RATIO,max_halo_overshoot_sigma:3.0,
            min_faint_signal_retention:FAINT_RETENTION,max_mask_edge_seam_sigma:2.0 }
    }
}

fn luminance(image: &LinearImage, i: usize, n: usize) -> f32 {
    if image.channels==1 { image.data[i] } else {
        0.2126*image.data[i]+0.7152*image.data[n+i]+0.0722*image.data[2*n+i]
    }
}

fn mask_at(mask: &[f32], i: usize) -> f32 {
    mask.get(i).copied().filter(|v|v.is_finite()).unwrap_or(0.0).clamp(0.0,1.0)
}

fn median(v: &mut [f32]) -> Option<f32> {
    if v.is_empty() { return None; }
    let mid = v.len()/2;
    v.select_nth_unstable_by(mid,|a,b|a.total_cmp(b));
    Some(v[mid])
}
fn mad(v: &mut [f32]) -> Option<f32> {
    let med = median(v)?;
    for x in v.iter_mut() { *x=(*x-med).abs(); }
    median(v).map(|m|m*1.4826)
}
fn ratio(a: f32,b: f32) -> Option<f32> {
    if b>1e-8 && a.is_finite() && b.is_finite() {Some(a/b)} else {None}
}

/// Compares the same valid coordinates; a byte-identical safe baseline passes even
/// when star/noise/faint measurements cannot be defined on a flat or masked frame.
pub fn validate_candidate(reference: &LinearImage, candidate: &LinearImage, analysis: &ImageAnalysis, masks: &MaskAtlas) -> QualityReport {
    let mut report = QualityReport::default();
    let Some(n) = reference.width.checked_mul(reference.height) else {
        report.reasons.push("image dimensions overflow".into()); return report;
    };
    if reference.width!=candidate.width || reference.height!=candidate.height || reference.channels!=candidate.channels
        || !matches!(reference.channels,1|3) || reference.data.len()!=n.saturating_mul(reference.channels)
        || candidate.data.len()!=reference.data.len() || reference.mask.len()!=n || candidate.mask!=reference.mask
        || masks.width!=reference.width || masks.height!=reference.height
        || [&masks.stars,&masks.star_halos,&masks.sky,&masks.signal,&masks.core,&masks.low_snr,&masks.edges].iter().any(|m|m.len()!=n) {
        report.reasons.push("image or mask dimensions/validity differ".into()); return report;
    }
    let mut changed=false;
    let mut eligible=0usize;
    let mut oversaturated=0usize;
    let mut sky_ref=Vec::new(); let mut sky_out=Vec::new();
    let mut hf_ref=Vec::new(); let mut hf_out=Vec::new();
    let mut faint_samples=Vec::new();
    let mut seam: f32=0.0;
    let w=reference.width; let h=reference.height;
    let step=(n/131_072).max(1);
    for i in 0..n {
        if reference.mask[i]==0 {continue;}
        let mut valid=true;
        let mut previously_clipped=false;
        let mut now_clipped=false;
        for c in 0..reference.channels {
            let before=reference.data[c*n+i]; let after=candidate.data[c*n+i];
            if !before.is_finite() || !after.is_finite() { valid=false; break; }
            if before!=after {changed=true;}
            if before<=0.0 || before>=1.0 {previously_clipped=true;}
            else if after<=0.0 || after>=1.0 {now_clipped=true;report.newly_clipped_channels[c]+=1;}
            if before<1.0 && after>=1.0 {oversaturated+=1;}
        }
        if !valid {report.reasons.push("nonfinite valid image pixel".into());continue;}
        report.valid_pixels+=1;
        if previously_clipped {report.preexisting_clipped_pixels+=1;} else {eligible+=1;if now_clipped {report.newly_clipped_pixels+=1;}}
        if i%step!=0 {continue;}
        let a=luminance(reference,i,n); let b=luminance(candidate,i,n);
        if mask_at(&masks.sky,i)>0.8 && mask_at(&masks.stars,i)<0.1 && mask_at(&masks.star_halos,i)<0.1 {
            sky_ref.push(a); sky_out.push(b);
            if w>1 && i%w>0 && reference.mask[i-1]!=0 && mask_at(&masks.sky,i-1)>0.8 {
                hf_ref.push(a-luminance(reference,i-1,n));hf_out.push(b-luminance(candidate,i-1,n));
            }
        }
        if mask_at(&masks.signal,i)>0.6 && mask_at(&masks.core,i)<0.2 && a>0.0 && a<0.3 {
            faint_samples.push((a,b));
        }
        if i%w>0 && mask_at(&masks.edges,i)>0.5 && reference.mask[i-1]!=0 {
            let ref_grad=(a-luminance(reference,i-1,n)).abs();
            let cand_grad=b-luminance(candidate,i-1,n);
            let difference=((cand_grad-(a-luminance(reference,i-1,n))).abs()-0.35*ref_grad).max(0.0);
            seam=seam.max(difference);
        }
    }
    if !report.reasons.is_empty() {return report;}
    report.clipped_fraction=report.newly_clipped_pixels as f32/eligible.max(1) as f32;
    report.oversaturated_fraction=oversaturated as f32/(eligible.max(1)*reference.channels) as f32;
    if !changed { report.passed=true; return report; }
    if report.valid_pixels==0 {report.reasons.push("no valid pixels to assess".into());return report;}
    if report.clipped_fraction>CLIP_FRACTION || report.newly_clipped_channels.iter().any(|&v|v as f32/eligible.max(1) as f32>CLIP_FRACTION) {
        report.reasons.push("new shadow/highlight clipping exceeds 0.1%".into());
    }
    if report.oversaturated_fraction>CLIP_FRACTION {
        report.reasons.push("new channel saturation exceeds 0.1%".into());
    }
    let sky_level=median(&mut sky_ref);
    let sky_output_level=median(&mut sky_out);
    if let (Some(a),Some(b))=(mad(&mut sky_ref),mad(&mut sky_out)) {report.sky_noise_ratio=ratio(b,a);}
    if let (Some(a),Some(b))=(mad(&mut hf_ref),mad(&mut hf_out)) {report.sky_high_frequency_ratio=ratio(b,a);}
    for (name,value) in [("sky noise",report.sky_noise_ratio),("sky high-frequency residual",report.sky_high_frequency_ratio)] {
        match value { Some(v) if v<=NOISE_RATIO=>(), Some(_)=>report.reasons.push(format!("{name} exceeds 1.2x baseline")),
            None=>report.reasons.push(format!("{name} unavailable; enhancement disabled")) }
    }
    if let (Some(background),Some(output_background))=(sky_level,sky_output_level) {
        let noise_floor=analysis.noise_sigma.filter(|v|v.is_finite() && *v>0.0).unwrap_or(0.0);
        let mut faint_ref=0.0f64;let mut faint_out=0.0f64;
        for (a,b) in faint_samples {
            let signal=a-background;
            if signal>noise_floor {
                faint_ref+=signal as f64;
                faint_out+=(b-output_background).max(0.0) as f64;
            }
        }
        if faint_ref>1e-8 {report.faint_signal_retention=Some((faint_out/faint_ref) as f32);}
    }
    match report.faint_signal_retention {
        Some(v) if v>=FAINT_RETENTION=>(),Some(_)=>report.reasons.push("faint-signal retention below 95%".into()),
        None=>report.reasons.push("faint-signal retention unavailable; enhancement disabled".into()),
    }
    let sigma=analysis.noise_sigma.filter(|v|v.is_finite() && *v>1e-8);
    if let Some(noise)=sigma {report.mask_edge_seam_sigma=Some(seam/noise);}
    match report.mask_edge_seam_sigma {
        Some(v) if v<=2.0=>(),Some(_)=>report.reasons.push("mask-edge seam exceeds 2 sigma".into()),
        None=>report.reasons.push("mask-edge noise unavailable; enhancement disabled".into()),
    }
    // Refit only unsaturated stars at corresponding original coordinates. A
    // candidate star elsewhere cannot substitute for the measured source star.
    if analysis.star_fwhm.is_some_and(|f|f.is_finite() && f>=1.0 && f<=32.0)
        && analysis.stars.fwhm_confidence.is_finite() && analysis.stars.fwhm_confidence>=0.35 {
        let before=detect_stars(reference);let after=detect_stars(candidate);
        let mut ratios=Vec::new();let mut halo: f32=0.0;
        for star in before.iter().take(256) {
            if let Some(other)=after.iter().find(|other| {
                (star.x-other.x).abs()<1.5 && (star.y-other.y).abs()<1.5
                    && other.fwhm.is_finite() && star.fwhm.is_finite() && star.fwhm>0.0
            }) {
                ratios.push(other.fwhm/star.fwhm);
                if let Some(noise)=sigma {
                    let r=star.fwhm.clamp(2.0,12.0);
                    let x0=(star.x as isize-(3.5*r) as isize).max(0) as usize;
                    let x1=(star.x as usize+(3.5*r) as usize).min(w.saturating_sub(1));
                    let y0=(star.y as isize-(3.5*r) as isize).max(0) as usize;
                    let y1=(star.y as usize+(3.5*r) as usize).min(h.saturating_sub(1));
                    let mut inner_max: f32=0.0;
                    let mut outer_shifts=Vec::new();
                    for y in y0..=y1 {for x in x0..=x1 {
                        let radius=((x as f32-star.x).powi(2)+(y as f32-star.y).powi(2)).sqrt();
                        let i=y*w+x;if reference.mask[i]==0 {continue;}
                        let delta=luminance(candidate,i,n)-luminance(reference,i,n);
                        if (r..=2.5*r).contains(&radius) {
                            inner_max=inner_max.max(delta);
                        } else if (2.5*r..=3.5*r).contains(&radius) {
                            outer_shifts.push(delta);
                        }
                    }}
                    let bg_shift=median(&mut outer_shifts).unwrap_or(0.0).max(0.0);
                    halo=halo.max(((inner_max-bg_shift)/noise).max(0.0));
                }
            }
        }
        report.matched_stars=ratios.len();
        report.star_fwhm_ratio=median(&mut ratios);
        if report.matched_stars>0 {report.bright_halo_overshoot_sigma=sigma.map(|_|halo);}
    }
    // No reliable PSF forbids positive, localized star-region enhancement, but
    // does not reject a starless image's smooth protected tone change outright.
    let star_change=(0..n).any(|i|reference.mask[i]!=0 && mask_at(&masks.stars,i)>0.5
        && luminance(candidate,i,n)>luminance(reference,i,n)+1e-5);
    if star_change {
        match report.star_fwhm_ratio {
            Some(v) if (v-1.0).abs()<=FWHM_RATIO=>(),Some(_)=>report.reasons.push("matched stellar FWHM changes by more than 10%".into()),
            None=>report.reasons.push("matched stellar FWHM unavailable; star enhancement disabled".into()),
        }
        if report.bright_halo_overshoot_sigma.is_none() {
            report.reasons.push("star halo measurement unavailable; star enhancement disabled".into());
        }
    } else if let Some(v)=report.star_fwhm_ratio {if (v-1.0).abs()>FWHM_RATIO {
        report.reasons.push("matched stellar FWHM changes by more than 10%".into());
    }}
    if report.bright_halo_overshoot_sigma.is_some_and(|v|v>3.0) {
        report.reasons.push("bright-star halo overshoot exceeds 3 sigma".into());
    }
    report.passed=report.reasons.is_empty();
    report
}
