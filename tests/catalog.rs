use photonloom::catalog::lookup_target;

#[test]
fn ctb1_header_links_supernova_remnant_not_galaxy_cluster() {
    let info = lookup_target("ABELL 85 SNR G116.9+00.2 (Field2)").unwrap();
    assert_eq!(info.name, "CTB 1 (Medulla Nebula)");
    assert_eq!(info.wikipedia_url, "https://en.wikipedia.org/wiki/CTB_1");
    assert!(info.summary.contains("supernova remnant"));
    assert_eq!(
        lookup_target("CTB_1").unwrap().wikipedia_url,
        info.wikipedia_url
    );
    assert_eq!(
        lookup_target("CTB-1").unwrap().wikipedia_url,
        info.wikipedia_url
    );
    assert_eq!(
        lookup_target("G116.9+00.1").unwrap().wikipedia_url,
        info.wikipedia_url
    );
}

#[test]
fn abell_85_without_specific_snr_identifier_is_ambiguous() {
    for title in [
        "Abell 85",
        "ABELL_85 (Field2)",
        "Abell 85 galaxy cluster",
        "unknown field",
    ] {
        assert!(
            lookup_target(title).is_none(),
            "unexpected story for {title}"
        );
    }
}

#[test]
fn aliases_support_spaced_and_separated_labels() {
    let cases = [
        ("M31 (Field2)", "Andromeda Galaxy (M31)"),
        ("m_31 data", "Andromeda Galaxy (M31)"),
        ("NGC-1976 HOO", "Orion Nebula (M42)"),
        ("M45", "Pleiades (M45)"),
        ("NGC2237", "Rosette Nebula"),
        ("Cygnus_Loop", "Veil Nebula (Cygnus Loop)"),
        ("NGC 7000 (Field2)", "North America Nebula (NGC 7000)"),
        ("Crab-Nebula", "Crab Nebula (M1)"),
    ];
    for (title, expected) in cases {
        let info = lookup_target(title).unwrap_or_else(|| panic!("missing story for {title}"));
        assert_eq!(info.name, expected);
        assert!(
            info.wikipedia_url
                .starts_with("https://en.wikipedia.org/wiki/")
        );
        assert!(!info.summary.is_empty());
    }
}

#[test]
fn identifier_boundaries_prevent_false_positives() {
    for title in [
        "M310",
        "xM31",
        "M31x",
        "NGC70000",
        "NGC 70000",
        "NGC22370",
        "NGC 22370",
        "CTB10",
        "G116.9+00.20",
        "M10",
        "NGC19520",
        "northamericanebula",
    ] {
        assert!(
            lookup_target(title).is_none(),
            "unexpected story for {title}"
        );
    }
}

#[test]
fn labels_naming_multiple_distinct_targets_are_not_guessed() {
    assert!(lookup_target("M31 M42").is_none());
}
