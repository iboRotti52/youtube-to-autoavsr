import numpy as np
import pytest
from yt2avsr.config import VisualQualityConfig
from yt2avsr.visual_quality import VisualQualityResult, analyze_visual_quality


def test_visual_quality_result_dataclass():
    res = VisualQualityResult(
        status="accepted",
        mouth_visible_ratio=0.95,
        scene_cut_ratio=0.0,
        static_speech_ratio=None,
        speech_mouth_motion_ratio=None,
        lip_sync_correlation=None,
        mouth_opening_correlation=None,
        max_missing_run_seconds=0.1,
        unstable_landmark_ratio=0.02,
        reasons=[],
        lip_sync_checked=False,
    )
    d = res.to_dict()
    assert d["status"] == "accepted"
    assert d["lip_sync_checked"] is False
    assert d["static_speech_ratio"] is None
    assert d["lip_sync_correlation"] is None
    assert d["mouth_visible_ratio"] == 0.95


def test_empty_video_visual_quality(tmp_path):
    # Empty or unreadable video file
    dummy_video = tmp_path / "empty.mp4"
    dummy_video.write_bytes(b"")
    dummy_audio = tmp_path / "empty.wav"
    dummy_audio.write_bytes(b"")

    cfg = VisualQualityConfig()
    with pytest.raises(RuntimeError, match="Cannot open visual quality input"):
        analyze_visual_quality(dummy_video, dummy_audio, cfg, verify_lip_sync=False)


def test_quality_config_save_source_clip_default():
    from yt2avsr.config import QualityConfig
    qc = QualityConfig()
    assert qc.save_source_clip is False


def test_analyze_visual_quality_slice_params(tmp_path):
    dummy_video = tmp_path / "nonexistent.mp4"
    cfg = VisualQualityConfig()
    with pytest.raises(RuntimeError, match="Cannot open visual quality input"):
        analyze_visual_quality(
            dummy_video,
            None,
            cfg,
            verify_lip_sync=False,
            start_seconds=1.5,
            duration_seconds=3.0,
        )


def test_transcription_config_num_workers():
    from yt2avsr.config import TranscriptionConfig
    tc = TranscriptionConfig()
    assert tc.num_workers == 1


def test_model_lock_defined():
    import threading
    from yt2avsr.transcribe import _MODEL_LOCK
    assert isinstance(_MODEL_LOCK, type(threading.Lock()))
