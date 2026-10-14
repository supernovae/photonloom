use photonloom::magic::{MagicVariant, preferences::{Preference, load, reset, save}};
use std::path::Path;

fn preference() -> Preference {
    Preference {
        class: "emission-nebula".into(),
        recipe_id: "emission-nebula".into(),
        variant: MagicVariant::Enhanced,
        strength: 0.5,
    }
}

fn file(dir: &Path) -> std::path::PathBuf {
    dir.join("magic-preferences.json")
}

#[test]
fn opted_in_selection_round_trips_and_reset_only_removes_preference_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = file(temp.path());
    let unrelated = temp.path().join("other-data.json");
    std::fs::write(&unrelated, b"keep").unwrap();
    save(&path, &preference()).unwrap();
    let (loaded, warning) = load(&path, true);
    assert_eq!(loaded, Some(preference()));
    assert_eq!(warning, None);
    reset(&path).unwrap();
    reset(&path).unwrap();
    assert!(!path.exists());
    assert_eq!(std::fs::read(&unrelated).unwrap(), b"keep");
    let (loaded, warning) = load(&path, true);
    assert_eq!(loaded, None);
    assert!(warning.unwrap().contains("Could not read"));
    assert!(reset(&unrelated).is_err());
    assert!(unrelated.exists());
}

#[test]
fn opted_out_never_reads_existing_or_corrupt_preferences() {
    let temp = tempfile::tempdir().unwrap();
    let path = file(temp.path());
    save(&path, &preference()).unwrap();
    assert_eq!(load(&path, false), (None, None));
    std::fs::write(&path, b"not json").unwrap();
    assert_eq!(load(&path, false), (None, None));
}

#[test]
fn malformed_schema_unknown_fields_and_invalid_values_warn_without_prior() {
    let temp = tempfile::tempdir().unwrap();
    let path = file(temp.path());
    for contents in [
        "not json",
        r#"{"version":2,"selections":[{"class":"unknown","recipe_id":"general-natural","variant":"natural","strength":0.5}]}"#,
        r#"{"version":1,"selections":[{"class":"unknown","recipe_id":"general-natural","variant":"natural","strength":1.1}]}"#,
        r#"{"version":1,"selections":[{"class":"unknown","recipe_id":"not-a-recipe","variant":"natural","strength":0.5}]}"#,
        r#"{"version":1,"selections":[{"class":"unknown","recipe_id":"emission-nebula","variant":"natural","strength":0.5}]}"#,
        r#"{"version":1,"selections":[{"class":"unknown","recipe_id":"general-natural","variant":"natural","strength":0.5,"source_path":"private"}]}"#,
        r#"{"version":1,"selections":[]}"#,
    ] {
        std::fs::write(&path, contents).unwrap();
        let (prior, warning) = load(&path, true);
        assert!(prior.is_none());
        assert!(warning.is_some());
    }
}
#[test]
fn saved_json_contains_only_schema_and_selected_preferences() {
    let temp = tempfile::tempdir().unwrap();
    let path = file(temp.path());
    save(&path, &preference()).unwrap();
    let contents = std::fs::read(&path).unwrap();
    let object = serde_json::from_slice::<serde_json::Value>(&contents).unwrap();
    let object = object.as_object().unwrap();
    let mut fields: Vec<_> = object.keys().map(String::as_str).collect();
    fields.sort_unstable();
    assert_eq!(fields, ["selections", "version"]);
    assert_eq!(object["version"], 1);
    let selection = object["selections"][0].as_object().unwrap();
    let mut fields: Vec<_> = selection.keys().map(String::as_str).collect();
    fields.sort_unstable();
    assert_eq!(fields, ["class", "recipe_id", "strength", "variant"]);
    assert_eq!(selection["variant"], "enhanced");
    assert!(!String::from_utf8(contents).unwrap().contains(temp.path().to_str().unwrap()));
}

#[test]
fn history_preserves_prior_subjects_and_is_bounded_to_recent_selections() {
    use photonloom::magic::preferences::load_matching;
    let temp = tempfile::tempdir().unwrap();
    let path = file(temp.path());
    let galaxy = Preference {
        class: "spiral-galaxy".into(),
        recipe_id: "spiral-galaxy".into(),
        variant: MagicVariant::Natural,
        strength: 0.3,
    };
    save(&path, &preference()).unwrap();
    save(&path, &galaxy).unwrap();
    assert_eq!(load_matching(&path, true, "emission-nebula", "emission-nebula").0, Some(preference()));
    assert_eq!(load_matching(&path, true, "spiral-galaxy", "spiral-galaxy").0, Some(galaxy.clone()));
    assert_eq!(load_matching(&path, false, "spiral-galaxy", "spiral-galaxy"), (None, None));
    for i in 0..64 {
        save(&path, &Preference { strength: i as f32 / 64.0, ..galaxy.clone() }).unwrap();
    }
    assert_eq!(load_matching(&path, true, "emission-nebula", "emission-nebula").0, None);
    assert_eq!(load_matching(&path, true, "spiral-galaxy", "spiral-galaxy").0,
        Some(Preference { strength: 63.0 / 64.0, ..galaxy }));
    let content: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(content["selections"].as_array().unwrap().len(), 64);
}

#[test]
fn invalid_preference_does_not_replace_existing_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = file(temp.path());
    save(&path, &preference()).unwrap();
    let original = std::fs::read(&path).unwrap();
    let invalid = Preference { strength: f32::NAN, ..preference() };
    assert!(save(&path, &invalid).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
}
