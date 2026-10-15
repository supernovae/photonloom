//! Fast header-only estimate of a run; star-field partitioning happens during processing.
use crate::calibrate::matching_counts;
use crate::config::{Channel, FrameRole, RejectedFrameRecord, default_output_for_input};
use crate::{finish, scan};
use anyhow::Result;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct ObservationPlan {
    pub input: String,
    pub default_output: String,
    pub totals: FrameCounts,
    pub targets: Vec<PlannedTarget>,
    pub feasible_palettes: Vec<String>,
    pub warnings: Vec<String>,
    pub rejected_files: Vec<RejectedFrameRecord>,
    pub scan_ms: u64,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct FrameCounts {
    pub lights: usize,
    pub darks: usize,
    pub flats: usize,
    pub bias: usize,
    pub flat_darks: usize,
    pub unreadable: usize,
}
#[derive(Debug, Clone, Serialize)]
pub struct PlannedTarget {
    pub label: String,
    pub sensor: String,
    pub width: usize,
    pub height: usize,
    pub channels: Vec<PlannedChannel>,
    pub feasible_palettes: Vec<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct PlannedChannel {
    pub channel: String,
    pub frames: usize,
    pub total_exposure_s: f64,
    pub calibration: PlannedCalibration,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct PlannedCalibration {
    pub bias: usize,
    pub darks: usize,
    pub flats: usize,
    pub flat_darks: usize,
    pub missing: Vec<String>,
}

pub fn inspect(input: &Path) -> Result<ObservationPlan> {
    let started = Instant::now();
    let report = scan::scan_detailed(input)?;
    let mut totals = FrameCounts::default();
    for frame in &report.frames {
        match frame.role {
            FrameRole::Light => totals.lights += 1,
            FrameRole::Dark => totals.darks += 1,
            FrameRole::Flat => totals.flats += 1,
            FrameRole::Bias => totals.bias += 1,
            FrameRole::FlatDark => totals.flat_darks += 1,
        }
    }
    totals.unreadable = report.rejected_frames.len();
    type Key = (String, String, usize, usize, (u32, u32));
    let mut groups: BTreeMap<Key, BTreeMap<Channel, Vec<&crate::Frame>>> = BTreeMap::new();
    for frame in &report.light_frames {
        let label = frame.object.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("Unlabelled target");
        groups.entry((label.to_string(), frame.sensor.clone(), frame.width, frame.height, frame.binning))
            .or_default().entry(frame.channel.unwrap_or(Channel::L)).or_default().push(frame);
    }
    let mut palette_set = BTreeSet::new();
    let targets = groups.into_iter().map(|((label, sensor, width, height, _), bands)| {
        let palettes = finish::feasible_palettes(&bands.keys().copied().collect::<Vec<_>>());
        palette_set.extend(palettes.iter().cloned());
        let channels = bands.into_iter().map(|(channel, frames)| {
            let representative = frames[0];
            let (bias, darks, flats, flat_darks) = matching_counts(representative, &report.calibration_frames);
            let mut missing = Vec::new();
            if bias == 0 && representative.format != crate::config::FileFormat::Raw {
                missing.push(format!("bias for {channel}"));
            }
            if flats == 0 { missing.push(format!("flats for {channel}")); }
            if darks == 0 {
                let exposure = representative.exposure.map_or("unknown exposure".to_string(), |s| format!("{s}s"));
                let temperature = representative.temperature.map_or("unknown temperature".to_string(), |c| format!("{c:.0}°C"));
                missing.push(format!("darks matching {exposure} @ {temperature}"));
            }
            PlannedChannel { channel: channel.to_string(), frames: frames.len(),
                total_exposure_s: frames.iter().map(|f| f.exposure.unwrap_or(0.0) as f64).sum(),
                calibration: PlannedCalibration { bias, darks, flats, flat_darks, missing } }
        }).collect();
        PlannedTarget { label, sensor, width, height, channels, feasible_palettes: palettes }
    }).collect();
    let mut warnings = report.warnings;
    warnings.push("This header-only estimate groups by OBJECT, sensor and geometry; star-field partitioning during run may split or merge targets.".into());
    Ok(ObservationPlan { input: input.display().to_string(), default_output: default_output_for_input(input)?.display().to_string(),
        totals, targets, feasible_palettes: palette_set.into_iter().collect(), warnings,
        rejected_files: report.rejected_frames, scan_ms: started.elapsed().as_millis() as u64 })
}

impl fmt::Display for ObservationPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Observation plan · {} (scanned in {} ms)", self.input, self.scan_ms)?;
        writeln!(f, "Lights {} · darks {} · flats {} · bias {} · flat-darks {} · unreadable {}",
            self.totals.lights, self.totals.darks, self.totals.flats, self.totals.bias, self.totals.flat_darks, self.totals.unreadable)?;
        for target in &self.targets {
            writeln!(f, "\n{} · {} · {}×{}", target.label, target.sensor, target.width, target.height)?;
            writeln!(f, "  Channel       Frames  Integration  Calibration")?;
            for ch in &target.channels {
                let minutes = (ch.total_exposure_s / 60.0).round() as u64;
                let status = if ch.calibration.missing.is_empty() { "✓".into() } else { format!("missing {}", ch.calibration.missing.join(", ")) };
                writeln!(f, "  {:<13} {:>6}  {:>5}:{:02}      {}", ch.channel, ch.frames, minutes / 60, minutes % 60, status)?;
            }
            writeln!(f, "  Palettes: {}", target.feasible_palettes.join(", "))?;
        }
        for warning in &self.warnings { writeln!(f, "Warning: {warning}")?; }
        for rejected in &self.rejected_files { writeln!(f, "Unreadable: {} — {}", rejected.path, rejected.reason)?; }
        write!(f, "Next: photonloom run {}", self.input)
    }
}
