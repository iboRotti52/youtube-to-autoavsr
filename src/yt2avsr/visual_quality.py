from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import wave

import cv2
import mediapipe as mp
import numpy as np

from .config import VisualQualityConfig


MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
         78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
         191, 80, 81, 82, 13, 312, 311, 310, 415]


@dataclass
class VisualQualityResult:
    status: str
    mouth_visible_ratio: float
    scene_cut_ratio: float
    static_speech_ratio: float
    max_missing_run_seconds: float
    unstable_landmark_ratio: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _read_audio_activity(path: Path, fps: float, frames: int, quantile: float) -> np.ndarray:
    """Return per-video-frame speech activity using only signal energy."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if width != 2:
        return np.ones(frames, dtype=bool)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    samples_per_frame = max(1, int(rate / fps))
    energy = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        chunk = audio[i * samples_per_frame:(i + 1) * samples_per_frame]
        if chunk.size:
            energy[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-8))

    nonzero = energy[energy > 0]
    if not nonzero.size:
        return np.zeros(frames, dtype=bool)
    threshold = float(np.quantile(nonzero, quantile))
    return energy >= threshold


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hb = cv2.calcHist([b], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def _longest_false_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def analyze_visual_quality(
    video_path: Path,
    audio_path: Path,
    cfg: VisualQualityConfig,
    verify_lip_sync: bool = True,
) -> VisualQualityResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open visual quality input: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return VisualQualityResult(
            status="rejected",
            mouth_visible_ratio=0.0,
            scene_cut_ratio=1.0,
            static_speech_ratio=1.0,
            max_missing_run_seconds=999.0,
            unstable_landmark_ratio=1.0,
            reasons=["empty_video"],
        )

    audio_active = _read_audio_activity(
        audio_path, fps, len(frames), cfg.audio_activity_quantile
    )

    visible: list[bool] = []
    mouth_motion = np.zeros(len(frames), dtype=np.float32)
    unstable = np.zeros(len(frames), dtype=bool)
    cuts = np.zeros(len(frames), dtype=bool)
    previous_hsv = None
    previous_mouth = None
    previous_geometry = None

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=cfg.min_face_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )

    try:
        for index, frame in enumerate(frames):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if previous_hsv is not None:
                cuts[index] = _hist_distance(previous_hsv, hsv) >= cfg.scene_cut_hist_threshold
            previous_hsv = hsv

            if index % max(1, cfg.sample_every_n_frames) != 0:
                visible.append(visible[-1] if visible else False)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                visible.append(False)
                previous_mouth = None
                previous_geometry = None
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            xs = np.array([landmarks[i].x * w for i in MOUTH], dtype=np.float32)
            ys = np.array([landmarks[i].y * h for i in MOUTH], dtype=np.float32)

            # No skin-tone or demographic assumptions: only geometry and temporal continuity.
            inside = bool(
                np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)) and
                xs.min() >= 0 and ys.min() >= 0 and xs.max() < w and ys.max() < h
            )
            width = float(xs.max() - xs.min()) if inside else 0.0
            height = float(ys.max() - ys.min()) if inside else 0.0
            geometry = np.array([
                float(xs.mean() / max(w, 1)),
                float(ys.mean() / max(h, 1)),
                width / max(w, 1),
                height / max(h, 1),
            ], dtype=np.float32)

            geometry_ok = inside and width >= 8 and height >= 3
            visible.append(geometry_ok)
            if not geometry_ok:
                previous_mouth = None
                previous_geometry = None
                continue

            if previous_geometry is not None:
                jump = float(np.linalg.norm(geometry - previous_geometry))
                unstable[index] = jump > cfg.landmark_jump_threshold
            previous_geometry = geometry

            pad_x, pad_y = width * 0.35, height * 0.65
            x1 = max(0, int(xs.min() - pad_x))
            x2 = min(w, int(xs.max() + pad_x))
            y1 = max(0, int(ys.min() - pad_y))
            y2 = min(h, int(ys.max() + pad_y))
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                visible[-1] = False
                previous_mouth = None
                continue

            gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 32))
            if previous_mouth is not None:
                mouth_motion[index] = float(
                    np.mean(cv2.absdiff(gray, previous_mouth))
                ) / 255.0
            previous_mouth = gray
    finally:
        mesh.close()

    visible_array = np.asarray(visible, dtype=bool)
    # Propagate sampled decisions only within the configured sampling stride.
    if len(visible_array) < len(frames):
        visible_array = np.pad(
            visible_array, (0, len(frames) - len(visible_array)), constant_values=False
        )

    speech_indices = audio_active & visible_array
    static_speech = speech_indices & (mouth_motion < cfg.mouth_motion_floor)

    mouth_visible_ratio = float(np.mean(visible_array))
    scene_cut_ratio = float(np.mean(cuts))
    static_speech_ratio = (
        float(np.sum(static_speech) / max(1, np.sum(speech_indices)))
        if np.any(speech_indices) else 1.0
    )
    max_missing_run_seconds = _longest_false_run(visible_array.tolist()) / fps
    unstable_landmark_ratio = float(np.mean(unstable[visible_array])) if np.any(visible_array) else 1.0

    # Lip/mouth visibility, scene-cut and occlusion checks run in BOTH profiles.
    accept_checks = [
        mouth_visible_ratio >= cfg.accept_min_mouth_visible_ratio,
        scene_cut_ratio <= cfg.accept_max_scene_cut_ratio,
        max_missing_run_seconds <= cfg.accept_max_missing_run_seconds,
        unstable_landmark_ratio <= cfg.accept_max_unstable_landmark_ratio,
    ]
    review_checks = [
        mouth_visible_ratio >= cfg.review_min_mouth_visible_ratio,
        scene_cut_ratio <= cfg.review_max_scene_cut_ratio,
        max_missing_run_seconds <= cfg.review_max_missing_run_seconds,
        unstable_landmark_ratio <= cfg.review_max_unstable_landmark_ratio,
    ]

    # Lip-sync (audio active but mouth static) only matters for voiceover sources,
    # where it flags external narration. Skipped for no_voiceover so natural pauses
    # don't drop otherwise-good segments.
    if verify_lip_sync:
        accept_checks.append(static_speech_ratio <= cfg.accept_max_static_speech_ratio)
        review_checks.append(static_speech_ratio <= cfg.review_max_static_speech_ratio)

    status = "accepted" if all(accept_checks) else "review" if all(review_checks) else "rejected"
    reasons: list[str] = []

    if mouth_visible_ratio < cfg.accept_min_mouth_visible_ratio:
        reasons.append("mouth_not_visible_enough")
    if scene_cut_ratio > cfg.accept_max_scene_cut_ratio:
        reasons.append("scene_changes")
    if verify_lip_sync and static_speech_ratio > cfg.accept_max_static_speech_ratio:
        reasons.append("external_voice_or_dubbing")
    if max_missing_run_seconds > cfg.accept_max_missing_run_seconds:
        reasons.append("long_mouth_missing_interval")
    if unstable_landmark_ratio > cfg.accept_max_unstable_landmark_ratio:
        reasons.append("unstable_or_occluded_landmarks")

    return VisualQualityResult(
        status=status,
        mouth_visible_ratio=round(mouth_visible_ratio, 5),
        scene_cut_ratio=round(scene_cut_ratio, 5),
        static_speech_ratio=round(static_speech_ratio, 5),
        max_missing_run_seconds=round(max_missing_run_seconds, 3),
        unstable_landmark_ratio=round(unstable_landmark_ratio, 5),
        reasons=reasons,
    )
