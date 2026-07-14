from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field

class DownloadConfig(BaseModel):
    format: str = "bv*[height>=720]+ba/b[height>=720]/best"
    cookies_from_browser: str | None = None
    playlist_end: int | None = None
    subtitle_languages: list[str] = ["tr", "tr-TR"]
    use_automatic_youtube_captions: bool = True
    remote_components: list[str] = ["ejs:github"]
    js_runtime: str = "deno"
    retries: int = 10
    fragment_retries: int = 10
    socket_timeout: int = 30

class NormalizationConfig(BaseModel):
    fps: int = 25
    audio_sample_rate: int = 16000
    max_height: int = 1080

class TranscriptionConfig(BaseModel):
    model: str = "medium"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    min_silence_duration_ms: int = 500
    min_word_probability: float = 0.55
    min_segment_probability: float = 0.72
    max_no_speech_probability: float = 0.45
    use_whisper_when_no_manual_subtitles: bool = True

class SegmentationConfig(BaseModel):
    min_duration: float = 2.0
    max_duration: float = 16.0
    pad_seconds: float = 0.12
    max_gap_seconds: float = 0.65
    min_words: int = 2

class ActiveSpeakerConfig(BaseModel):
    enabled: bool = True
    backend: str = "av_sync"
    min_score: float = 0.18
    min_track_coverage: float = 0.80
    face_detection_confidence: float = 0.55
    max_faces: int = 6
    track_iou_threshold: float = 0.25
    sample_every_n_frames: int = 5
    analysis_width: int = 480
    skip_when_single_face: bool = True
    single_face_probe_frames: int = 40

class TalkNetConfig(BaseModel):
    repo_dir: Path = Path("external/TalkNet-ASD")
    python_executable: Path = Path("external/talknet-venv/bin/python")
    min_speaking_probability: float = 0.60
    accept_min_speaking_ratio: float = 0.65
    review_min_speaking_ratio: float = 0.40
    keep_workdirs: bool = False


class AutoAVSRConfig(BaseModel):
    repo_dir: Path = Path("external/auto_avsr")
    detector: str = "retinaface"
    device: str = "auto"
    output_size: int = 96
    strict: bool = True

class VisualQualityConfig(BaseModel):
    enabled: bool = True
    sample_every_n_frames: int = 1
    accept_min_mouth_visible_ratio: float = 0.88
    review_min_mouth_visible_ratio: float = 0.68
    accept_max_scene_cut_ratio: float = 0.015
    review_max_scene_cut_ratio: float = 0.050
    accept_max_static_speech_ratio: float = 0.18
    review_max_static_speech_ratio: float = 0.42
    accept_max_missing_run_seconds: float = 0.35
    review_max_missing_run_seconds: float = 1.00
    accept_max_unstable_landmark_ratio: float = 0.12
    review_max_unstable_landmark_ratio: float = 0.35
    scene_cut_hist_threshold: float = 0.58
    mouth_motion_floor: float = 0.018
    audio_activity_quantile: float = 0.55
    landmark_jump_threshold: float = 0.18
    min_face_detection_confidence: float = 0.50
    min_tracking_confidence: float = 0.50

class QualityConfig(BaseModel):
    min_asr_confidence: float = 0.72
    min_active_speaker_score: float = 0.18
    min_face_coverage: float = 0.80
    min_sharpness: float = 18.0

class CloudConfig(BaseModel):
    # Shared private Hugging Face dataset repo, e.g. "my-team/avsr-tr-dataset".
    repo_id: str | None = None
    private: bool = True

class AppConfig(BaseModel):
    workspace: Path = Path("data")
    language: str = "tr"
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    active_speaker: ActiveSpeakerConfig = Field(default_factory=ActiveSpeakerConfig)
    talknet: TalkNetConfig = Field(default_factory=TalkNetConfig)
    auto_avsr: AutoAVSRConfig = Field(default_factory=AutoAVSRConfig)
    visual_quality: VisualQualityConfig = Field(default_factory=VisualQualityConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)

def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
