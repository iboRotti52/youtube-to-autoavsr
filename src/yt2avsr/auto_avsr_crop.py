from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .config import AutoAVSRConfig

@dataclass
class CropMetrics:
    face_coverage: float
    sharpness: float

_CACHE = {}

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
        detector = LandmarksDetector(device=_device(cfg.device))
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


def crop_with_official_auto_avsr(source: Path, output: Path,
                                 cfg: AutoAVSRConfig) -> CropMetrics:
    detector, processor = _get_components(cfg)

    frames, fps = _read_rgb_frames(source)
    if frames.shape[0] == 0:
        raise RuntimeError("Source clip has no frames")

    landmarks = detector(frames)
    if landmarks is None or all(lm is None for lm in landmarks):
        raise RuntimeError("Official Auto-AVSR detector returned no landmarks")

    sequence = processor(frames, landmarks)
    if sequence is None or len(sequence) == 0:
        raise RuntimeError("Official Auto-AVSR created an empty mouth video")

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
