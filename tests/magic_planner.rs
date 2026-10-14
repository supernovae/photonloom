use photonloom::{config::LinearImage, magic::{analysis::{self, ImageAnalysis}, planner::{load_parameters, query_local, MagicProposal, ProposalScales}, recipes, InputDomain, MagicVariant}};
use std::{io::{Read, Write}, net::TcpListener, time::Duration};

fn analysis() -> ImageAnalysis {
    let mut img = LinearImage::new(16, 16, 3);
    for (i, value) in img.data.iter_mut().enumerate() { *value = 0.1 + (i % 41) as f32 * 0.003; }
    analysis::analyze_image(&img, InputDomain::Display).0
}

fn proposal() -> MagicProposal {
    MagicProposal { version: 1, recipe_id: "general-natural".into(), variant: MagicVariant::Enhanced,
        strength: 0.5, operator_scales: ProposalScales { tone: 0.5, detail: 0.0, sharpen: 0.0, color: 0.0 } }
}

#[test]
fn schema_scales_recipe_and_measurement_capabilities_are_enforced() {
    let general = recipes::all().remove(0);
    let good = proposal();
    good.validate(&general).unwrap();
    assert_eq!(serde_json::from_str::<MagicProposal>(&good.to_json().unwrap()).unwrap(), good);
    for invalid in [
        serde_json::json!({"version": 2, "recipe_id":"general-natural", "variant":"enhanced", "strength":0.5,"operator_scales":{"tone":0.5,"detail":0.0,"sharpen":0.0,"color":0.0}}),
        serde_json::json!({"version": 1, "recipe_id":"invented", "variant":"enhanced", "strength":0.5,"operator_scales":{"tone":0.5,"detail":0.0,"sharpen":0.0,"color":0.0}}),
        serde_json::json!({"version": 1, "recipe_id":"general-natural", "variant":"enhanced", "strength":2.0,"operator_scales":{"tone":0.5,"detail":0.0,"sharpen":0.0,"color":0.0}}),
        serde_json::json!({"version": 1, "recipe_id":"general-natural", "variant":"enhanced", "strength":0.5,"operator_scales":{"tone":1.01,"detail":0.0,"sharpen":0.0,"color":0.0}}),
        serde_json::json!({"version": 1, "recipe_id":"general-natural", "variant":"enhanced", "strength":0.5,"operator_scales":{"tone":0.5,"detail":0.0,"sharpen":0.0,"color":0.0}, "extra":"reject"}),
        serde_json::json!({"version": 1, "recipe_id":"general-natural", "variant":"enhanced", "strength":0.5,"operator_scales":{"tone":0.5,"detail":0.0,"sharpen":0.0,"color":0.0,"other":0.2}}),
    ] {
        let parsed = serde_json::from_value::<MagicProposal>(invalid);
        assert!(parsed.is_err() || parsed.unwrap().validate(&general).is_err());
    }
    let mut no_detail = recipes::all().into_iter().find(|r| r.id == "star-cluster").unwrap();
    let mut bad = good.clone();
    bad.recipe_id = no_detail.id.clone();
    bad.operator_scales.detail = 0.2;
    assert!(bad.validate(&no_detail).is_err());
    bad.operator_scales.detail = 0.0;
    assert!(bad.validate_for_analysis(&no_detail, &analysis(), &[]).is_err());
    no_detail.required_channels.push("sii".into());
    assert!(bad.validate(&no_detail).is_err());
    let mut absent = good.clone();
    absent.operator_scales.sharpen = 0.2;
    assert!(absent.validate_for_analysis(&general, &analysis(), &[]).is_err());
}

#[test]
fn parameter_files_are_bounded_and_validated() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("parameters.json");
    std::fs::write(&path, proposal().to_json().unwrap()).unwrap();
    assert_eq!(load_parameters(&path).unwrap(), proposal());
    std::fs::write(&path, " ".repeat(16 * 1024 + 1)).unwrap();
    assert!(load_parameters(&path).is_err());
    std::fs::write(&path, "{\"version\":2}").unwrap();
    assert!(load_parameters(&path).is_err());
}

fn mock(response_content: &str) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let content = serde_json::to_string(response_content).unwrap();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(4))).unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let size = stream.read(&mut buffer).unwrap();
            assert!(size > 0);
            request.extend_from_slice(&buffer[..size]);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                let length: usize = headers.lines().find_map(|line| line.strip_prefix("content-length: "))
                    .unwrap().parse().unwrap();
                if request.len() >= end + 4 + length { break; }
            }
        }
        let body = format!("{{\"choices\":[{{\"message\":{{\"content\":{content}}}}}]}}");
        let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        stream.write_all(response.as_bytes()).unwrap();
        String::from_utf8(request).unwrap()
    });
    (url, handle)
}

#[test]
fn local_planner_sends_only_aggregate_metrics_when_explicitly_called() {
    let mut measured = analysis();
    measured.metadata.wcs = Some("PRIVATE_SKY_COORDINATES".into());
    measured.metadata.camera = Some("PRIVATE_CAMERA_ID".into());
    measured.masks.sky = Some("PRIVATE_IMAGE_PATH".into());
    let recipe = recipes::all().remove(0);
    let (url, server) = mock(&proposal().to_json().unwrap());
    // Merely constructing analysis and opting into a recipe does not contact the server.
    assert!(query_local("https://public.example", &measured, &recipe, MagicVariant::Enhanced, 0.5).is_err());
    let chosen = query_local(&url, &measured, &recipe, MagicVariant::Enhanced, 0.5).unwrap();
    assert_eq!(chosen, proposal());
    let request = server.join().unwrap();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    for secret in ["PRIVATE_SKY_COORDINATES", "PRIVATE_CAMERA_ID", "PRIVATE_IMAGE_PATH", "history", "pixel"] {
        assert!(!request.contains(secret), "request leaked {secret}");
    }
    assert!(request.contains("general-natural"));
    assert!(request.contains("allowed_masks"));
}

#[test]
fn invalid_local_proposals_are_rejected_without_accepting_partial_json() {
    let measured = analysis();
    let recipe = recipes::all().remove(0);
    for content in ["```json\n{}\n```", "{\"version\":1}",
        "{\"version\":1,\"recipe_id\":\"general-natural\",\"variant\":\"enhanced\",\"strength\":0.5,\"operator_scales\":{\"tone\":1.5,\"detail\":0,\"sharpen\":0,\"color\":0}}",
        "{\"version\":1,\"recipe_id\":\"general-natural\",\"variant\":\"dramatic\",\"strength\":0.5,\"operator_scales\":{\"tone\":0.5,\"detail\":0,\"sharpen\":0,\"color\":0}}"] {
        let (url, server) = mock(content);
        assert!(query_local(&url, &measured, &recipe, MagicVariant::Enhanced, 0.5).is_err());
        server.join().unwrap();
    }
}
