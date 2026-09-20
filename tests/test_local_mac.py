"""Mac local ergonomics: MediaPipe default, folder support, safe devices.

Covers the local-processing cleanup without touching dataset-quality logic,
the Modal GPU path, or RetinaFace availability.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from yt2avsr.config import AppConfig, load_config
from yt2avsr.pipeline import Pipeline
from yt2avsr.transcribe import resolve_device


def _local_cfg(tmp_path: Path, **overrides) -> AppConfig:
    cfg = AppConfig(workspace=tmp_path / "data")
    cfg.sources.processed_file = tmp_path / "processed_sources.txt"
    cfg.processing.max_workers = 1
    # Point the external-repo preflight at a fake repo so tests don't need
    # the real clone; production behavior (clear error when missing) is
    # covered by test_preflight_missing_external_repo below.
    fake_repo = tmp_path / "external" / "auto_avsr"
    (fake_repo / "preparation").mkdir(parents=True)
    cfg.auto_avsr.repo_dir = fake_repo
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _noop_preflight(monkeypatch) -> None:
    monkeypatch.setattr(Pipeline, "_preflight_local", lambda self: None)


# 1. Local processing without --config selects MediaPipe.
def test_default_config_selects_mediapipe() -> None:
    assert load_config(None).auto_avsr.detector == "mediapipe"
    assert AppConfig().auto_avsr.detector == "mediapipe"
    assert load_config(Path("configs/default.yaml")).auto_avsr.detector == "mediapipe"


def test_cli_local_uses_mediapipe_when_no_config(tmp_path, monkeypatch, capsys) -> None:
    import yt2avsr.cli as cli_mod

    captured: dict = {}

    class FakePipeline:
        def __init__(self, cfg, **kwargs):
            captured["cfg"] = cfg

        def process_local(self, path):
            return {"id": "x"}

    monkeypatch.setattr(cli_mod, "Pipeline", FakePipeline)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    cli_mod.local(video)
    assert captured["cfg"].auto_avsr.detector == "mediapipe"


# 2. Explicit --config override still works.
def test_explicit_retina_config_override() -> None:
    assert load_config(Path("configs/retina_1080p.yaml")).auto_avsr.detector == "retinaface"
    assert load_config(Path("configs/retinaface.yaml")).auto_avsr.detector == "retinaface"


# 3. RetinaFace is not required for the local setup path.
def test_setup_once_does_not_require_retinaface() -> None:
    lines = Path("scripts/setup_once.sh").read_text(encoding="utf-8").splitlines()
    invocations = [
        line for line in lines
        if "setup-retinaface" in line and not line.strip().startswith(("echo", "#"))
    ]
    assert invocations == []
    script = "\n".join(lines)
    assert "setup-external" in script
    assert "setup-whisper" in script


def test_retinaface_command_still_available() -> None:
    from yt2avsr.cli import setup_retinaface

    assert callable(setup_retinaface)


# 4. process-local handles a single file.
def test_process_local_single_file(tmp_path, monkeypatch) -> None:
    cfg = _local_cfg(tmp_path)
    _noop_preflight(monkeypatch)
    seen: list = []
    monkeypatch.setattr(Pipeline, "_process_item", lambda self, item: seen.append(item["id"]))
    monkeypatch.setattr("yt2avsr.pipeline.rebuild", lambda workspace: [])
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-video")

    item = Pipeline(cfg).process_local(video)

    assert item["id"] == "video"
    assert seen == ["video"]


# 5. process-local handles a folder with multiple videos.
def test_process_local_folder_processes_each_video(tmp_path, monkeypatch) -> None:
    cfg = _local_cfg(tmp_path)
    _noop_preflight(monkeypatch)
    seen: list = []
    monkeypatch.setattr(Pipeline, "_process_item", lambda self, item: seen.append(item["id"]))
    monkeypatch.setattr("yt2avsr.pipeline.rebuild", lambda workspace: [])
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "b.mkv").write_bytes(b"fake-b")
    (folder / "a.mp4").write_bytes(b"fake-a")
    (folder / "notes.txt").write_bytes(b"not a video")

    items = Pipeline(cfg).process_local(folder)

    assert [i["id"] for i in items] == ["a", "b"]
    assert seen == ["a", "b"]


def test_process_local_empty_folder_errors(tmp_path, monkeypatch) -> None:
    import pytest

    cfg = _local_cfg(tmp_path)
    _noop_preflight(monkeypatch)
    monkeypatch.setattr("yt2avsr.pipeline.rebuild", lambda workspace: [])
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "notes.txt").write_bytes(b"not a video")
    with pytest.raises(ValueError, match="No supported video files"):
        Pipeline(cfg).process_local(folder)


def test_process_local_logs_detector_and_device(tmp_path, monkeypatch, capsys) -> None:
    cfg = _local_cfg(tmp_path)
    _noop_preflight(monkeypatch)
    monkeypatch.setattr(Pipeline, "_process_item", lambda self, item: None)
    monkeypatch.setattr("yt2avsr.pipeline.rebuild", lambda workspace: [])
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")

    Pipeline(cfg).process_local(video)

    out = capsys.readouterr().out
    assert "detector=mediapipe" in out
    assert "whisper_device=" in out
    assert "whisper_compute=" in out


# Preflight: clear errors for missing ffmpeg / external repo.
def test_preflight_missing_ffmpeg_errors_clearly(tmp_path, monkeypatch) -> None:
    import pytest

    cfg = _local_cfg(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        Pipeline(cfg)._preflight_local()


def test_preflight_missing_external_repo_errors_clearly(tmp_path) -> None:
    import pytest

    cfg = AppConfig(workspace=tmp_path / "data")
    cfg.auto_avsr.repo_dir = tmp_path / "does-not-exist"
    with pytest.raises(RuntimeError, match="setup-external"):
        Pipeline(cfg)._preflight_local()


# 4b. macOS device safety: faster-whisper has no MPS backend.
def test_resolve_device_never_returns_mps() -> None:
    assert resolve_device("mps") == ("cpu", "int8")
    assert resolve_device("MPS") == ("cpu", "int8")
    device, compute = resolve_device("auto")
    assert device in {"cuda", "cpu"}
    assert compute in {"float16", "int8"}
    if device == "cpu":
        assert compute == "int8"


# 6. Modal/RetinaFace behavior is not regressed.
def test_modal_still_defaults_to_retina_preset() -> None:
    from yt2avsr.cli import modal_cmd

    default_config = inspect.signature(modal_cmd).parameters["config"].default
    assert default_config == "configs/retina_1080p.yaml"
    assert load_config(Path(str(default_config))).auto_avsr.detector == "retinaface"


def test_default_config_keeps_quality_thresholds() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.quality.min_asr_confidence == 0.72
    assert cfg.transcription.model == "large-v3-turbo"
    assert cfg.auto_avsr.output_size == 96


# --- Team CLI config helper: bare commands == --config configs/default.yaml ---

EXPECTED_HF_REPO = "avsr-tr-ekip/avsr-tr-dataset"


def test_load_cli_config_defaults_to_default_yaml() -> None:
    from yt2avsr.cli import load_cli_config

    cfg = load_cli_config(None)
    assert cfg.auto_avsr.detector == "mediapipe"
    assert cfg.normalization.max_height == 1080
    assert "1080" in cfg.download.format
    assert cfg.cloud.repo_id == EXPECTED_HF_REPO


def test_load_cli_config_explicit_wins(tmp_path) -> None:
    from yt2avsr.cli import load_cli_config

    custom = tmp_path / "custom.yaml"
    custom.write_text("extends: retina_1080p.yaml\n", encoding="utf-8")
    # extends resolves relative to the custom file's dir; use absolute instead.
    custom.write_text(
        "auto_avsr:\n  detector: retinaface\n", encoding="utf-8"
    )
    assert load_cli_config(custom).auto_avsr.detector == "retinaface"


def test_load_cli_config_falls_back_without_default_yaml(tmp_path, monkeypatch) -> None:
    import yt2avsr.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_packaged_default_config", lambda: None)
    cfg = cli_mod.load_cli_config(None)
    # Code defaults still local-first (MediaPipe), just without default.yaml values.
    assert cfg.auto_avsr.detector == "mediapipe"
    assert cfg.cloud.repo_id is None


def test_process_both_sources_uses_default_yaml_without_config(
    tmp_path, monkeypatch, capsys
) -> None:
    import yt2avsr.cli as cli_mod

    (tmp_path / "sources_no_voiceover.txt").write_text(
        "https://www.youtube.com/watch?v=team_workflow_a\n"
    )
    (tmp_path / "sources_voiceover.txt").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "sync_processed_from_hf", lambda *a, **k: [])

    captured: dict = {}

    class FakePipeline:
        def __init__(self, cfg, **kwargs):
            captured.setdefault("cfgs", []).append(cfg)

        def process_sources_file(self, path, already_partitioned=False):
            captured.setdefault("ran", []).append(path.name)

    monkeypatch.setattr(cli_mod, "Pipeline", FakePipeline)
    cli_mod.process_both_sources()

    cfg = captured["cfgs"][0]
    assert cfg.auto_avsr.detector == "mediapipe"
    assert cfg.normalization.max_height == 1080
    assert "1080" in cfg.download.format
    assert cfg.cloud.repo_id == EXPECTED_HF_REPO


def test_push_data_uses_default_yaml_repo(tmp_path, monkeypatch) -> None:
    import yt2avsr.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    pushed: dict = {}
    monkeypatch.setattr(
        "yt2avsr.cloud.push",
        lambda workspace, repo_id, **kwargs: pushed.update(repo_id=repo_id) or "ok",
    )
    monkeypatch.setattr(cli_mod, "append_processed_sources", lambda *a, **k: 0)

    cli_mod.push_data()

    assert pushed["repo_id"] == EXPECTED_HF_REPO


def test_sync_processed_uses_default_yaml_repo(tmp_path, monkeypatch) -> None:
    import yt2avsr.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    synced: dict = {}
    monkeypatch.setattr(
        cli_mod,
        "sync_processed_from_hf",
        lambda *a, **k: synced.update(k) or [],
    )

    cli_mod.sync_processed()

    assert synced["repo_id"] == EXPECTED_HF_REPO


# --- Runtime log/preflight runs once per workspace, incl. shard flow ---

def test_runtime_announced_once_across_pipelines(tmp_path, monkeypatch, capsys) -> None:
    cfg = _local_cfg(tmp_path)
    preflights: list = []
    monkeypatch.setattr(
        Pipeline, "_preflight_local", lambda self: preflights.append(1)
    )

    p1 = Pipeline(cfg)
    p2 = Pipeline(cfg, profile="voiceover")
    p1._announce_runtime_once()
    p2._announce_runtime_once()

    out = capsys.readouterr().out
    assert out.count("[local] detector=mediapipe") == 1
    assert len(preflights) == 1


def test_preflight_fails_when_only_one_binary_present(tmp_path, monkeypatch) -> None:
    import pytest

    cfg = _local_cfg(tmp_path)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None
    )
    with pytest.raises(RuntimeError, match="ffmpeg"):
        Pipeline(cfg)._preflight_local()


# --- Whisper device/compute hardening ---

def test_resolve_device_cuda_without_cuda_falls_back(monkeypatch) -> None:
    import yt2avsr.transcribe as tr_mod

    monkeypatch.setattr(tr_mod, "_cuda_available", lambda: False)
    assert tr_mod.resolve_device("cuda") == ("cpu", "int8")
    assert tr_mod.resolve_device("auto") == ("cpu", "int8")

    monkeypatch.setattr(tr_mod, "_cuda_available", lambda: True)
    assert tr_mod.resolve_device("cuda") == ("cuda", "float16")
    assert tr_mod.resolve_device("auto") == ("cuda", "float16")


def test_resolve_compute_guards_cpu(capsys) -> None:
    from yt2avsr.transcribe import resolve_compute

    assert resolve_compute("cpu", "auto") == "int8"
    assert resolve_compute("cpu", "int8") == "int8"
    assert resolve_compute("cpu", "float32") == "float32"
    assert resolve_compute("cuda", "auto") == "float16"
    assert resolve_compute("cuda", "float16") == "float16"
    # GPU-only compute on CPU falls back with a warning.
    assert resolve_compute("cpu", "float16") == "int8"
    assert "int8" in capsys.readouterr().out
