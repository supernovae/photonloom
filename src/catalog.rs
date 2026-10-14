use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetInfo {
    pub name: String,
    pub wikipedia_url: String,
    pub summary: String,
}

struct Entry {
    aliases: &'static [&'static str],
    name: &'static str,
    wikipedia_url: &'static str,
    summary: &'static str,
}

// Each entry's designation and one-sentence description are checked against its linked
// Wikipedia article. Abell 85 alone is deliberately not an alias: it also names a galaxy cluster.
const TARGETS: &[Entry] = &[
    Entry {
        aliases: &["CTB1", "CTB 1", "G116.9+00.2", "G116.9+00.1"],
        name: "CTB 1 (Medulla Nebula)",
        wikipedia_url: "https://en.wikipedia.org/wiki/CTB_1",
        summary: "CTB 1 is a supernova remnant in Cassiopeia whose glowing shell marks the aftermath of a stellar explosion.",
    },
    Entry {
        aliases: &[
            "M31",
            "M 31",
            "MESSIER 31",
            "NGC224",
            "NGC 224",
            "ANDROMEDA GALAXY",
        ],
        name: "Andromeda Galaxy (M31)",
        wikipedia_url: "https://en.wikipedia.org/wiki/Andromeda_Galaxy",
        summary: "The Andromeda Galaxy is a barred spiral galaxy and the nearest major galaxy to the Milky Way.",
    },
    Entry {
        aliases: &[
            "M42",
            "M 42",
            "MESSIER 42",
            "NGC1976",
            "NGC 1976",
            "ORION NEBULA",
        ],
        name: "Orion Nebula (M42)",
        wikipedia_url: "https://en.wikipedia.org/wiki/Orion_Nebula",
        summary: "The Orion Nebula is a nearby star-forming cloud visible in the sword of Orion.",
    },
    Entry {
        aliases: &["M45", "M 45", "MESSIER 45", "PLEIADES", "SEVEN SISTERS"],
        name: "Pleiades (M45)",
        wikipedia_url: "https://en.wikipedia.org/wiki/Pleiades",
        summary: "The Pleiades are a nearby open star cluster in Taurus, surrounded by blue reflection nebulosity.",
    },
    Entry {
        aliases: &["NGC2237", "NGC 2237", "ROSETTE NEBULA"],
        name: "Rosette Nebula",
        wikipedia_url: "https://en.wikipedia.org/wiki/Rosette_Nebula",
        summary: "The Rosette Nebula is a star-forming region in Monoceros surrounding the young star cluster NGC 2244.",
    },
    Entry {
        aliases: &[
            "VEIL NEBULA",
            "CYGNUS LOOP",
            "NGC6960",
            "NGC 6960",
            "NGC6992",
            "NGC 6992",
        ],
        name: "Veil Nebula (Cygnus Loop)",
        wikipedia_url: "https://en.wikipedia.org/wiki/Veil_Nebula",
        summary: "The Veil Nebula traces the bright visible arcs of the Cygnus Loop, a supernova remnant in Cygnus.",
    },
    Entry {
        aliases: &["NGC7000", "NGC 7000", "NORTH AMERICA NEBULA"],
        name: "North America Nebula (NGC 7000)",
        wikipedia_url: "https://en.wikipedia.org/wiki/North_America_Nebula",
        summary: "The North America Nebula is an emission nebula in Cygnus named for its continent-like outline.",
    },
    Entry {
        aliases: &[
            "M1",
            "M 1",
            "MESSIER 1",
            "NGC1952",
            "NGC 1952",
            "CRAB NEBULA",
        ],
        name: "Crab Nebula (M1)",
        wikipedia_url: "https://en.wikipedia.org/wiki/Crab_Nebula",
        summary: "The Crab Nebula is a supernova remnant in Taurus associated with a stellar explosion observed in 1054.",
    },
];

// Separators in object labels, filenames, and headers can vary without changing the name.
fn separator(byte: u8) -> bool {
    byte.is_ascii_whitespace() || byte == b'_' || byte == b'-'
}

fn matches_alias(title: &str, alias: &str) -> bool {
    let text = title.as_bytes();
    let name = alias.as_bytes();
    for start in 0..text.len() {
        if start > 0 && text[start - 1].is_ascii_alphanumeric() {
            continue;
        }
        let (mut t, mut n) = (start, 0);
        while n < name.len() && t < text.len() {
            if separator(name[n]) {
                if !separator(text[t]) {
                    break;
                }
                while n < name.len() && separator(name[n]) {
                    n += 1;
                }
                while t < text.len() && separator(text[t]) {
                    t += 1;
                }
            } else if text[t].eq_ignore_ascii_case(&name[n]) {
                t += 1;
                n += 1;
            } else {
                break;
            }
        }
        if n == name.len() && (t == text.len() || !text[t].is_ascii_alphanumeric()) {
            return true;
        }
    }
    false
}

/// Return a curated story only when a title contains one unambiguous known designation.
/// Unknown targets and titles naming multiple targets have no associated story.
pub fn lookup_target(title: &str) -> Option<TargetInfo> {
    let mut found = None;
    for entry in TARGETS {
        if entry
            .aliases
            .iter()
            .any(|alias| matches_alias(title, alias))
        {
            if found.is_some() {
                return None;
            }
            found = Some(entry);
        }
    }
    found.map(|entry| TargetInfo {
        name: entry.name.to_owned(),
        wikipedia_url: entry.wikipedia_url.to_owned(),
        summary: entry.summary.to_owned(),
    })
}

use crate::fits::WcsTan;
use std::sync::OnceLock;
use std::time::Duration;

/// Catalog identity; HDR nebulosity needs an independent measured-core assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SubjectClass {
    EmissionNebula,
    ReflectionNebula,
    DarkNebula,
    SpiralGalaxy,
    OtherGalaxy,
    GlobularCluster,
    OpenCluster,
    PlanetaryNebula,
    StarField,
    Unknown,
}

impl SubjectClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmissionNebula => "emission-nebula",
            Self::ReflectionNebula => "reflection-nebula",
            Self::DarkNebula => "dark-nebula",
            Self::SpiralGalaxy => "spiral-galaxy",
            Self::OtherGalaxy => "other-galaxy",
            Self::GlobularCluster => "globular-cluster",
            Self::OpenCluster => "open-cluster",
            Self::PlanetaryNebula => "planetary-nebula",
            Self::StarField => "star-field",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectCandidate {
    pub designation: String,
    pub aliases: Vec<String>,
    pub class: SubjectClass,
    pub score: f64,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub major_arcmin: Option<f64>,
    pub minor_arcmin: Option<f64>,
    pub position_angle_deg: Option<f64>,
    pub catalog_type: String,
    pub hubble: Option<String>,
    pub source: String,
    /// Approximate fraction of catalog ellipse intersecting the image footprint.
    pub coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectDecision {
    pub candidates: Vec<SubjectCandidate>,
    pub class: SubjectClass,
    pub score: f64,
    pub selected: Option<String>,
    pub fallback_warning: Option<String>,
}

#[derive(Debug, Clone)]
struct Position {
    designation: String,
    aliases: Vec<String>,
    kind: String,
    hubble: Option<String>,
    ra: f64,
    dec: f64,
    major: Option<f64>,
    minor: Option<f64>,
    angle: Option<f64>,
    source: &'static str,
}

fn positive(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0)
}

fn parse_positions(bytes: &[u8]) -> Result<Vec<Position>, csv::Error> {
    let mut reader = csv::ReaderBuilder::new().flexible(false).from_reader(bytes);
    let mut positions = Vec::new();
    for row in reader.records() {
        let row = row?;
        let field = |index| row.get(index).unwrap_or("").trim();
        let invalid = || csv::Error::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData, format!("Invalid catalog coordinates or extent for {}", field(0))));
        let ra = field(2).parse::<f64>().map_err(|_|invalid())?;
        let dec = field(3).parse::<f64>().map_err(|_|invalid())?;
        if !ra.is_finite() || !(0.0..360.0).contains(&ra) || !dec.is_finite()
            || !(-90.0..=90.0).contains(&dec) || field(0).is_empty()
        {
            return Err(invalid());
        }
        if [4,5,6].into_iter().any(|i| !field(i).is_empty()
            && field(i).parse::<f64>().is_err()) {
            return Err(invalid());
        }
        let major = positive(field(4));
        let minor = positive(field(5)).or(major);
        let angle = field(6).parse::<f64>().ok().filter(|v| v.is_finite() && (0.0..180.0).contains(v));
        let aliases = [field(8), field(9)].into_iter().filter(|s| !s.is_empty())
            .flat_map(|s| s.split(',')).map(str::trim).filter(|s| !s.is_empty())
            .map(str::to_owned).collect();
        positions.push(Position {
            designation: field(0).to_owned(), aliases, kind: field(1).to_owned(),
            ra, dec, major, minor, angle, hubble: (!field(7).is_empty()).then(|| field(7).to_owned()),
            source: "OpenNGC",
        });
    }
    positions.sort_by(|a, b| a.designation.cmp(&b.designation));
    positions.dedup_by(|a, b| a.designation == b.designation);
    Ok(positions)
}

fn catalog() -> &'static [Position] {
    static CATALOG: OnceLock<Vec<Position>> = OnceLock::new();
    CATALOG.get_or_init(|| parse_positions(include_bytes!("../data/catalog/openngc-positions.csv"))
        .expect("bundled OpenNGC CSV must be valid"))
}

fn classify(kind: &str, hubble: Option<&str>) -> SubjectClass {
    match kind {
        "HII" | "EmN" | "SNR" | "Cl+N" => SubjectClass::EmissionNebula,
        "RfN" => SubjectClass::ReflectionNebula,
        "DrkN" => SubjectClass::DarkNebula,
        "PN" => SubjectClass::PlanetaryNebula,
        "GCl" => SubjectClass::GlobularCluster,
        "OCl" => SubjectClass::OpenCluster,
        "G" => {
            if hubble.is_some_and(|h| {
                let h = h.trim().to_ascii_uppercase();
                h.starts_with('S') && !h.starts_with("S0") && !h.starts_with("S-")
                    && !h.starts_with("S?")
            }) {
                SubjectClass::SpiralGalaxy
            } else {
                SubjectClass::OtherGalaxy
            }
        }
        "*" | "**" | "*Ass" => SubjectClass::StarField,
        _ => SubjectClass::Unknown,
    }
}

// Tangent-plane projection (arcminutes) about the image center. atan2 handles RA=0
// and avoids dividing by cos(dec) at the poles.
fn project(origin: (f64, f64), point: (f64, f64)) -> Option<(f64, f64)> {
    let (ra0, dec0) = (origin.0.to_radians(), origin.1.to_radians());
    let (ra, dec) = (point.0.to_radians(), point.1.to_radians());
    let delta = ra - ra0;
    let cosc = dec0.sin() * dec.sin() + dec0.cos() * dec.cos() * delta.cos();
    if cosc <= 0.0 || !cosc.is_finite() { return None; }
    let scale = 180.0 * 60.0 / std::f64::consts::PI / cosc;
    Some((scale * dec.cos() * delta.sin(),
          scale * (dec0.cos() * dec.sin() - dec0.sin() * dec.cos() * delta.cos())))
}

fn inside(point: (f64, f64), polygon: &[(f64, f64); 8]) -> bool {
    let mut sign = 0i8;
    for index in 0..8 {
        let a = polygon[index];
        let b = polygon[(index + 1) % 8];
        let cross = (b.0 - a.0) * (point.1 - a.1) - (b.1 - a.1) * (point.0 - a.0);
        if cross.abs() < 1e-9 { continue; }
        let next = if cross > 0.0 { 1 } else { -1 };
        if sign != 0 && next != sign { return false; }
        sign = next;
    }
    true
}

fn evidence(position: &Position, title: &str) -> f64 {
    if position.designation.eq_ignore_ascii_case("Mel022") && matches_alias(title, "M45") {
        return 0.4;
    }
    if matches_alias(title, &position.designation)
        || position.aliases.iter().any(|alias| matches_alias(title, alias)) {
        0.4
    } else { 0.0 }
}

fn score_position(position: &Position, origin: (f64, f64), polygon: &[(f64, f64); 8],
                  radius: f64, title: &str) -> Option<SubjectCandidate> {
    let center = project(origin, (position.ra, position.dec))?;
    let major = position.major.unwrap_or(0.0);
    let minor = position.minor.unwrap_or(major);
    let half_major = major * 0.5;
    let half_minor = minor * 0.5;
    let distance = center.0.hypot(center.1);
    if distance > radius + half_major.max(half_minor) + 1.0 { return None; }
    let title_score = evidence(position, title);
    let coverage = if major > 0.0 {
        // Fixed area quadrature estimates ellipse/footprint intersection, including
        // large off-center nebulae whose center falls outside the frame.
        let angle = position.angle.unwrap_or(0.0).to_radians();
        let (sin, cos) = angle.sin_cos();
        let mut hits = 0;
        let mut total = 0;
        for u in -8..=8 {
            for v in -8..=8 {
                let (u, v) = (u as f64 / 8.0, v as f64 / 8.0);
                if u*u + v*v > 1.0 { continue; }
                total += 1;
                let p = (center.0 + u*half_major*cos + v*half_minor*sin,
                         center.1 - u*half_major*sin + v*half_minor*cos);
                if inside(p, polygon) { hits += 1; }
            }
        }
        hits as f64 / total as f64
    } else if inside(center, polygon) { 1.0 } else { 0.0 };
    if coverage == 0.0 { return None; }
    let image_area = polygon.iter().enumerate().map(|(i, p)| {
        let next = polygon[(i+1)%8]; p.0*next.1-next.0*p.1
    }).sum::<f64>().abs() * 0.5;
    let object_area = std::f64::consts::PI * half_major * half_minor;
    // Reject tiny background galaxies as primary subjects of wide-field frames.
    let area_fit = if major == 0.0 { 0.12 }
        else { (object_area / image_area.max(1e-10)).sqrt().clamp(0.0, 1.0) };
    let overlap = (coverage * (area_fit * 2.0).min(1.0)).sqrt();
    let centrality = (1.0 - distance / radius.max(1e-10)).clamp(0.0, 1.0);
    let score = (0.70 * overlap + 0.08 * centrality + 0.22 * title_score / 0.4)
        .clamp(0.0, 1.0);
    Some(SubjectCandidate {
        designation: position.designation.clone(), aliases: position.aliases.clone(),
        class: classify(&position.kind, position.hubble.as_deref()), score,
        ra_deg: position.ra, dec_deg: position.dec,
        major_arcmin: position.major, minor_arcmin: position.minor,
        position_angle_deg: position.angle, catalog_type: position.kind.clone(),
        hubble: position.hubble.clone(), source: position.source.to_owned(), coverage,
    })
}

fn rank(positions: impl IntoIterator<Item = Position>, wcs: &WcsTan,
        dimensions: [usize; 2], title: &str) -> Vec<SubjectCandidate> {
    let [width, height] = dimensions;
    if width == 0 || height == 0 { return Vec::new(); }
    let xy = [
        (0.0, 0.0), ((width-1) as f64 / 2.0, 0.0), ((width-1) as f64, 0.0),
        ((width-1) as f64, (height-1) as f64 / 2.0),
        ((width-1) as f64, (height-1) as f64),
        ((width-1) as f64 / 2.0, (height-1) as f64),
        (0.0, (height-1) as f64), (0.0, (height-1) as f64 / 2.0)
    ];
    let Some(origin) = wcs.pixel_to_sky((width-1) as f64 / 2.0, (height-1) as f64 / 2.0) else {
        return Vec::new();
    };
    let Some(polygon) = xy.map(|(x,y)| wcs.pixel_to_sky(x,y)
        .and_then(|p| project(origin,p))).into_iter().collect::<Option<Vec<_>>>() else {
        return Vec::new();
    };
    let polygon: [(f64,f64);8] = polygon.try_into().expect("eight footprint points");
    let radius = polygon.iter().map(|p| p.0.hypot(p.1)).fold(0.0, f64::max);
    let mut candidates: Vec<_> = positions.into_iter()
        .filter_map(|p| score_position(&p, origin, &polygon, radius, title)).collect();
    candidates.sort_by(|a,b| b.score.total_cmp(&a.score).then_with(|| a.designation.cmp(&b.designation)));
    candidates.truncate(20);
    candidates
}

const SIMBAD_TAP: &str = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync";
const MAX_RESPONSE: u64 = 256 * 1024;

fn online_positions(wcs: &WcsTan, dimensions: [usize;2]) -> Result<Vec<Position>, String> {
    online_positions_with(wcs, dimensions, |query| {
        use std::io::Read;
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none()).build().map_err(|e| e.to_string())?;
        let response = client.post(SIMBAD_TAP)
            .form(&[("REQUEST","doQuery"),("LANG","ADQL"),("FORMAT","csv"),("QUERY",query)])
            .send().map_err(|e| e.to_string())?
            .error_for_status().map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        response.take(MAX_RESPONSE+1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        Ok(bytes)
    })
}

fn online_positions_with(wcs: &WcsTan, dimensions: [usize;2],
                         fetch: impl FnOnce(&str) -> Result<Vec<u8>, String>) -> Result<Vec<Position>, String> {
    let [width,height] = dimensions;
    if width == 0 || height == 0 { return Ok(Vec::new()); }
    let center = wcs.pixel_to_sky((width-1) as f64 / 2.0,(height-1) as f64 / 2.0)
        .ok_or("Invalid WCS center")?;
    let corners = [(0.0,0.0),((width-1) as f64,0.0),
                   (0.0,(height-1) as f64),((width-1) as f64,(height-1) as f64)];
    let mut radius: f64 = 0.0;
    for (x,y) in corners {
        let point = wcs.pixel_to_sky(x,y).ok_or("Invalid WCS footprint")?;
        let p = project(center,point).ok_or("Footprint extends beyond tangent hemisphere")?;
        radius = radius.max(p.0.hypot(p.1) / 60.0);
    }
    let radius = radius.clamp(0.0001, 5.0);
    let query = format!("SELECT TOP 100 main_id,ra,dec,otype FROM basic WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{:.8},{:.8},{:.8}))=1", center.0,center.1,radius);
    let bytes = fetch(&query)?;
    if bytes.len() as u64 > MAX_RESPONSE { return Err("SIMBAD CSV exceeded 256 KiB".into()); }
    parse_online_csv(&bytes)
}

fn parse_online_csv(bytes: &[u8]) -> Result<Vec<Position>, String> {
    let mut reader = csv::ReaderBuilder::new().from_reader(bytes);
    let headers = reader.headers().map_err(|e| e.to_string())?;
    if headers.iter().collect::<Vec<_>>() != ["main_id","ra","dec","otype"] {
        return Err("Unexpected SIMBAD CSV columns".into());
    }
    let mut positions = Vec::new();
    for row in reader.records().take(101) {
        let row = row.map_err(|e| e.to_string())?;
        if positions.len() == 100 { return Err("SIMBAD exceeded TOP 100".into()); }
        let ra = row[1].parse::<f64>().map_err(|e| e.to_string())?;
        let dec = row[2].parse::<f64>().map_err(|e| e.to_string())?;
        if !ra.is_finite() || !(0.0..360.0).contains(&ra)
            || !dec.is_finite() || !(-90.0..=90.0).contains(&dec) {
            return Err("Invalid SIMBAD coordinates".into());
        }
        let kind = match row[3].trim() {
            "HII" | "Em*" => "HII", "PN" => "PN", "G" => "G", "GlC" => "GCl",
            "OpC" => "OCl", _ => "unknown"
        };
        positions.push(Position {
            designation: row[0].trim().to_owned(), aliases: Vec::new(), kind: kind.into(),
            hubble: None, ra, dec, major: None, minor: None, angle: None, source: "SIMBAD",
        });
    }
    if positions.is_empty() { return Err("SIMBAD returned zero hits".into()); }
    Ok(positions)
}

/// Offline by default. Online requests only occur with both explicit opt-in and valid WCS.
pub fn resolve_subject(wcs: Option<&WcsTan>, dimensions: [usize;2], title: &str,
                       online: bool) -> SubjectDecision {
    resolve_subject_with(wcs, dimensions, title, online, online_positions)
}

fn resolve_subject_with(wcs: Option<&WcsTan>, dimensions: [usize;2], title: &str,
                        online: bool,
                        fetch: impl FnOnce(&WcsTan,[usize;2]) -> Result<Vec<Position>,String>) -> SubjectDecision {
    let mut warning = None;
    let mut candidates = if let Some(wcs) = wcs {
        let mut offline = rank(catalog().iter().cloned(),wcs,dimensions,title);
        if online {
            match fetch(wcs,dimensions) {
                Ok(positions) => {
                    offline.extend(rank(positions,wcs,dimensions,title));
                    offline.sort_by(|a,b| b.score.total_cmp(&a.score)
                        .then_with(|| a.designation.cmp(&b.designation)));
                    let mut seen = std::collections::HashSet::new();
                    offline.retain(|candidate| seen.insert(candidate.designation.clone()));
                    offline.truncate(20);
                }
                Err(err) => warning = Some(format!("Online catalog unavailable ({err}); using offline catalog")),
            }
        }
        offline
    } else {
        warning = Some("No validated WCS; title is identity evidence only, not a sky position".into());
        let mut matches: Vec<_> = catalog().iter().filter(|p| evidence(p,title) > 0.0)
            .map(|p| SubjectCandidate {
                designation:p.designation.clone(), aliases:p.aliases.clone(),
                class:classify(&p.kind,p.hubble.as_deref()), score:0.4,
                ra_deg:p.ra,dec_deg:p.dec,major_arcmin:p.major,minor_arcmin:p.minor,
                position_angle_deg:p.angle,catalog_type:p.kind.clone(),hubble:p.hubble.clone(),
                source:p.source.into(),coverage:0.0,
            }).collect();
        matches.sort_by(|a,b| a.designation.cmp(&b.designation));
        matches.truncate(20);
        matches
    };
    // "Abell 85" designates both an SNR and a galaxy cluster; neither a title
    // nor unrelated catalog entries resolve that conflict.
    if matches_alias(title,"Abell 85") && !title.contains("CTB")
        && !title.contains("G116.9") && wcs.is_none() {
        candidates.clear();
    }
    let best = candidates.first();
    let selected = best.filter(|best| best.score >= 0.6
        && best.score - candidates.get(1).map_or(0.0, |next| next.score) >= 0.15);
    if selected.is_none() && warning.is_none() {
        warning = Some("Catalog identity uncertain: low score or close competing objects; using general-natural".into());
    }
    SubjectDecision {
        class:selected.map_or(SubjectClass::Unknown,|best|best.class),
        score:best.map_or(0.0,|best|best.score),
        selected:selected.map(|best|best.designation.clone()),
        candidates, fallback_warning:warning,
    }
}

#[cfg(test)]
mod online_tests {
    use super::*;

    fn wcs() -> WcsTan {
        WcsTan { crval: [83.81867,-5.38967], crpix: [301.0,301.0],
                 cd: [[-0.002,0.0],[0.0,0.002]] }
    }

    #[test]
    fn opt_in_query_is_numeric_bounded_and_mockable() {
        let positions = online_positions_with(&wcs(),[601,601],|query| {
            assert!(query.contains("SELECT TOP 100"));
            assert!(query.contains("CIRCLE('ICRS',83."));
            assert!(!query.contains("M42"));
            Ok(b"main_id,ra,dec,otype\nSIMBAD M42,83.81867,-5.38967,HII\n".to_vec())
        }).unwrap();
        assert_eq!(positions.len(),1);
        assert_eq!(positions[0].source,"SIMBAD");
        assert_eq!(classify(&positions[0].kind,None),SubjectClass::EmissionNebula);
    }

    #[test]
    fn bad_csv_empty_results_and_transport_error_are_rejected() {
        assert!(online_positions_with(&wcs(),[601,601],|_| Ok(b"<html>failure</html>".to_vec())).is_err());
        assert!(online_positions_with(&wcs(),[601,601],|_| Ok(b"main_id,ra,dec,otype\n".to_vec())).is_err());
        assert!(online_positions_with(&wcs(),[601,601],|_| Err("timeout".into())).is_err());
        assert!(online_positions_with(&wcs(),[601,601],|_| Ok(vec![b'a';MAX_RESPONSE as usize + 1])).is_err());
        assert!(online_positions_with(&wcs(),[601,601],|_| {
            Ok(b"main_id,ra,dec,otype\nbad,NaN,0,G\n".to_vec())
        }).is_err());
    }
    #[test]
    fn failed_opt_in_keeps_offline_selection_and_no_wcs_never_fetches() {
        let fallback = resolve_subject_with(Some(&wcs()),[601,601],"M42",true,
            |_,_| Err("timeout".into()));
        assert_eq!(fallback.selected.as_deref(),Some("NGC1976"));
        assert!(fallback.fallback_warning.as_deref().unwrap().contains("timeout"));
        let absent = resolve_subject_with(None,[601,601],"M42",true,
            |_,_| panic!("a missing WCS must never trigger a network request"));
        assert_eq!(absent.selected,None);
    }
}
