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
    assert resolve_device("cuda") == ("cuda", "float16")
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
