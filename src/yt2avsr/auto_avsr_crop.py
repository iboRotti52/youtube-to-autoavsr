from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
from .config import AutoAVSRConfig

@dataclass
class CropMetrics:
    face_coverage: float
    sharpness: float

_CACHE = {}
_LANDMARK_CACHE = {}


class NoUsableFaceError(RuntimeError):
    """Raised when Auto-AVSR cannot find a face suitable for mouth cropping."""


def _detect_landmarks(detector, frames: np.ndarray) -> list:
    """Allow face-free batches while scanning a longer source video."""
    try:
        return list(detector(frames))
    except AssertionError as exc:
        # The official MediaPipe detector uses this assertion when a batch has
        # frames but none of them contains a detectable face.
        if "Cannot detect any frames in the video" not in str(exc):
            raise
        return [None] * int(frames.shape[0])


def _device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _get_components(cfg: AutoAVSRConfig):
    repo = cfg.repo_dir.resolve()
    key = (str(repo), cfg.detector, _device(cfg.device))
    if key in _CACHE:
        return _CACHE[key]

    if not (repo / "preparation").exists():
        raise RuntimeError(
            f"Official Auto-AVSR repository not found at {repo}. "
            "Run: yt2avsr setup-external"
        )
    sys.path.insert(0, str(repo))
    try:
        if cfg.detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
        elif cfg.detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess
        else:
            raise ValueError(f"Unsupported official detector: {cfg.detector}")
        if cfg.detector == "retinaface":
            detector = LandmarksDetector(device=_device(cfg.device))
        else:
            detector = LandmarksDetector()
        processor = VideoProcess(convert_gray=False)
        _CACHE[key] = (detector, processor)
        return detector, processor
    finally:
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)

def _read_rgb_frames(source: Path) -> tuple[np.ndarray, float]:
    """Read a clip as an (T, H, W, 3) RGB uint8 array, matching what the official
    Auto-AVSR pipeline feeds its detector. Uses OpenCV so it works on any
    torchvision version (newer ones dropped torchvision.io.read_video)."""
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames), float(fps) or 25.0


def _probe_frame_count(source: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    cap.release()
    if frame_count > 0:
        return frame_count, float(fps) or 25.0

    frames, fallback_fps = _read_rgb_frames(source)
    return int(frames.shape[0]), fallback_fps


def _read_rgb_frame_slice(source: Path, start_frame: int, length: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frames = []
    while len(frames) < length:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames), float(fps) or 25.0


def _read_cached_landmarks(source: Path, cfg: AutoAVSRConfig) -> tuple[list, float]:
    source = source.resolve()
    stat = source.stat()
    key = (
        str(source),
        stat.st_mtime_ns,
        stat.st_size,
        cfg.detector,
        _device(cfg.device),
    )
    if key in _LANDMARK_CACHE:
        return _LANDMARK_CACHE[key]

    detector, _ = _get_components(cfg)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    landmarks = []
    batch = []
    batch_size = 250

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(batch) >= batch_size:
                landmarks.extend(_detect_landmarks(detector, np.asarray(batch)))
                batch.clear()

        if batch:
            landmarks.extend(_detect_landmarks(detector, np.asarray(batch)))
    finally:
        cap.release()

    if not landmarks:
        raise RuntimeError("Source video has no frames for landmark cache")

    _LANDMARK_CACHE[key] = (landmarks, float(fps) or 25.0)
    return _LANDMARK_CACHE[key]


def _copy_landmarks(landmarks: Sequence, start: int, length: int) -> list:
    selected = list(landmarks[start:start + length])
    if len(selected) < length:
        selected.extend([None] * (length - len(selected)))
    return [None if lm is None else np.array(lm, copy=True) for lm in selected]


def crop_with_official_auto_avsr(source: Path, output: Path,
                                 cfg: AutoAVSRConfig,
                                 *,
                                 landmark_source: Path | None = None,
                                 start_seconds: float | None = None) -> CropMetrics:
    detector, processor = _get_components(cfg)

    if landmark_source is not None and start_seconds is not None:
        source_frame_count, fps = _probe_frame_count(source)
        if source_frame_count == 0:
            raise RuntimeError("Source clip has no frames")
        cached_landmarks, landmark_fps = _read_cached_landmarks(landmark_source, cfg)
        start_frame = max(0, int(round(start_seconds * landmark_fps)))
        frames, _ = _read_rgb_frame_slice(
            landmark_source, start_frame, source_frame_count
        )
        landmarks = _copy_landmarks(cached_landmarks, start_frame, frames.shape[0])
    else:
        frames, fps = _read_rgb_frames(source)
        if frames.shape[0] == 0:
            raise RuntimeError("Source clip has no frames")
        landmarks = _detect_landmarks(detector, frames)

    if frames.shape[0] == 0:
        raise RuntimeError("Source clip has no frames")

    if landmarks is None or all(lm is None for lm in landmarks):
        raise NoUsableFaceError("Official Auto-AVSR detector returned no landmarks")

    sequence = processor(frames, landmarks)
    if sequence is None or len(sequence) == 0:
        raise NoUsableFaceError("Official Auto-AVSR created an empty mouth video")

    seq = np.asarray(sequence)
    if seq.ndim == 3:  # grayscale -> add a channel dim
        seq = np.repeat(seq[..., None], 3, axis=-1)
    if seq.dtype != np.uint8:
        seq = np.clip(seq, 0, 255).astype(np.uint8)

    output.parent.mkdir(parents=True, exist_ok=True)
    h, w = seq.shape[1], seq.shape[2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    for frame in seq:  # OpenCV writes BGR
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    sharp = [
        float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        for f in seq
    ]
    return CropMetrics(
        face_coverage=1.0,
        sharpness=round(float(np.mean(sharp)) if sharp else 0.0, 3),
    )
