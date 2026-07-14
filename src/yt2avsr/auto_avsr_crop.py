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

def crop_with_official_auto_avsr(source: Path, output: Path,
                                 cfg: AutoAVSRConfig) -> CropMetrics:
    detector, processor = _get_components(cfg)
    try:
        landmarks = detector(str(source))
        if landmarks is None:
            raise RuntimeError("Official Auto-AVSR detector returned no landmarks")
        output.parent.mkdir(parents=True, exist_ok=True)
        processor(str(source), landmarks, str(output))
    except TypeError as exc:
        raise RuntimeError(
            "The installed Auto-AVSR API differs from the supported release."
        ) from exc

    cap = cv2.VideoCapture(str(output))
    sharp, frames = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    cap.release()
    if frames == 0:
        raise RuntimeError("Official Auto-AVSR created an empty mouth video")
    return CropMetrics(
        face_coverage=1.0,
        sharpness=round(float(np.mean(sharp)) if sharp else 0.0, 3),
    )
