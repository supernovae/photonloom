use photonloom::watch::{FolderWatcher, WatchConfig, WatchPoll, WatchRunRecord, prune_runs};
use std::time::Duration;

fn config(root: &std::path::Path, settle: Duration) -> WatchConfig {
    WatchConfig {
        input: root.join("input"),
        output: root.join("output"),
        interval: Duration::from_millis(20),
        settle,
        keep_runs: 2,
    }
}

fn completed(frame_files: usize) -> WatchRunRecord {
    WatchRunRecord {
        run_id: None,
        finished_unix_ms: 123,
        elapsed_ms: 10,
        ok: false,
        error: Some("dummy frames are intentionally not decoded".into()),
        frame_files,
    }
}

#[test]
fn settled_fingerprint_persists_across_restarts_and_ignores_partial_files() {
    let temp = tempfile::tempdir().unwrap();
    let cfg = config(temp.path(), Duration::from_millis(500));
    std::fs::create_dir_all(&cfg.input).unwrap();
    std::fs::write(cfg.input.join("a.fits"), b"one").unwrap();
    std::fs::write(cfg.input.join(".hidden.fits"), b"ignored").unwrap();
    std::fs::write(cfg.input.join("b.fits.part"), b"ignored").unwrap();
    let mut watcher = FolderWatcher::new(cfg.clone()).unwrap();
    assert!(!cfg.output.exists(), "polling must not create output");
    assert!(matches!(watcher.poll().unwrap(), WatchPoll::Settling { changed_files: 1, .. }));
    assert_eq!(watcher.snapshot().frame_files, 1);
    assert_eq!(watcher.snapshot().changed_files, 1);
    std::thread::sleep(Duration::from_millis(550));
    assert!(matches!(watcher.poll().unwrap(), WatchPoll::Ready));
    watcher.begin();
    watcher.finish(completed(1)).unwrap();
    assert!(cfg.output.join(".photonloom-watch.json").is_file());
    let mut restarted = FolderWatcher::new(cfg.clone()).unwrap();
    assert!(matches!(restarted.poll().unwrap(), WatchPoll::Unchanged));
    assert_eq!(restarted.snapshot().changed_files, 0);
    std::fs::write(cfg.input.join("b.fits"), b"new").unwrap();
    assert!(matches!(restarted.poll().unwrap(), WatchPoll::Settling { changed_files: 1, .. }));
    assert_eq!(restarted.snapshot().changed_files, 1);
    std::thread::sleep(Duration::from_millis(550));
    assert!(matches!(restarted.poll().unwrap(), WatchPoll::Ready));
    restarted.begin();
    restarted.finish(completed(2)).unwrap();
    assert_eq!(restarted.snapshot().changed_files, 0);
    assert!(matches!(FolderWatcher::new(cfg).unwrap().poll().unwrap(), WatchPoll::Unchanged));
}

#[test]
fn new_metadata_resets_quiet_timer() {
    let temp = tempfile::tempdir().unwrap();
    let cfg = config(temp.path(), Duration::from_millis(500));
    std::fs::create_dir_all(&cfg.input).unwrap();
    std::fs::write(cfg.input.join("a.fits"), b"one").unwrap();
    let mut watcher = FolderWatcher::new(cfg.clone()).unwrap();
    assert!(matches!(watcher.poll().unwrap(), WatchPoll::Settling { .. }));
    std::thread::sleep(Duration::from_millis(200));
    std::fs::write(cfg.input.join("a.fits"), b"different size").unwrap();
    assert!(matches!(watcher.poll().unwrap(), WatchPoll::Settling { changed_files: 1, quiet_for } if quiet_for < Duration::from_millis(150)));
    std::thread::sleep(Duration::from_millis(200));
    assert!(matches!(watcher.poll().unwrap(), WatchPoll::Settling { .. }));
    std::thread::sleep(Duration::from_millis(350));
    assert!(matches!(watcher.poll().unwrap(), WatchPoll::Ready));
}

#[test]
fn prune_preserves_manifest_run_even_when_it_is_oldest() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path();
    for id in ["run-001", "run-002", "run-003", "run-004"] {
        std::fs::create_dir_all(output.join("runs").join(id)).unwrap();
    }
    std::fs::write(output.join("manifest.json"), r#"{"run_dir":"runs/run-001"}"#).unwrap();
    let removed = prune_runs(output, 2).unwrap();
    assert_eq!(removed.len(), 2);
    assert!(output.join("runs/run-001").is_dir());
    assert!(output.join("runs/run-004").is_dir());
    assert!(!output.join("runs/run-002").exists());
    assert!(!output.join("runs/run-003").exists());
    prune_runs(output, 1).unwrap();
    assert!(output.join("runs/run-001").exists());
    assert!(!output.join("runs/run-004").exists());
}
