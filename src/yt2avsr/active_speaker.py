from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

from .config import ActiveSpeakerConfig

@dataclass
class ActiveSpeakerResult:
    video_path: Path
    score: float
    coverage: float
    track_id: int
    skipped_single_face: bool = False

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union

def _resize_for_analysis(frame, width):
    if frame.shape[1] <= width:
        return frame, 1.0
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale))), scale

def _detect(detector, frame):
    small, scale = _resize_for_analysis(frame, 480)
    result = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    boxes = []
    if result.detections:
        h, w = small.shape[:2]
        for det in result.detections:
            bb = det.location_data.relative_bounding_box
            x1, y1 = max(0, int(bb.xmin*w)), max(0, int(bb.ymin*h))
            x2, y2 = min(w, int((bb.xmin+bb.width)*w)), min(h, int((bb.ymin+bb.height)*h))
            if x2 > x1 and y2 > y1:
                inv = 1.0 / scale
                boxes.append(tuple(int(v * inv) for v in (x1, y1, x2, y2)))
    return boxes

def _probe_single_face(source, cfg):
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max(1, cfg.single_face_probe_frames))
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=cfg.face_detection_confidence)
    max_count, observed = 0, 0
    index = 0
    while observed < cfg.single_face_probe_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        max_count = max(max_count, len(_detect(detector, frame)))
        observed += 1
        index += step
    detector.close()
    cap.release()
    return observed > 0 and max_count <= 1

def select_active_speaker(source: Path, output: Path,
                          cfg: ActiveSpeakerConfig, fps: int) -> ActiveSpeakerResult:
    # Talking-head optimization: no ASD needed when sparse probing never finds
    # more than one concurrent face. Reuse the original clip directly.
    if cfg.skip_when_single_face and _probe_single_face(source, cfg):
        return ActiveSpeakerResult(source, 1.0, 1.0, 0, True)

    cap = cv2.VideoCapture(str(source))
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=cfg.face_detection_confidence)

    sampled = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, cfg.sample_every_n_frames) == 0:
            boxes = _detect(detector, frame)[:cfg.max_faces]
            sampled.append((frame_index, boxes, frame))
        frame_index += 1
    detector.close()
    cap.release()

    if not sampled:
        raise RuntimeError(f"No frames in {source}")

    tracks = []
    for fi, boxes, frame in sampled:
        used = set()
        for track in tracks:
            prev = track["boxes"][-1][1] if track["boxes"] else None
            candidates = [(_iou(prev,b), idx,b) for idx,b in enumerate(boxes) if idx not in used]
            best = max(candidates, default=(0,None,None))
            if prev and best[0] >= cfg.track_iou_threshold:
                track["boxes"].append((fi,best[2],frame)); used.add(best[1])
        for idx, box in enumerate(boxes):
            if idx not in used:
                tracks.append({"boxes":[(fi,box,frame)]})

    if not tracks:
        raise RuntimeError("No face track found")

    total_sampled = len(sampled)
    best = None
    for tid, track in enumerate(tracks):
        motion, prev_gray = [], None
        for fi, box, frame in track["boxes"]:
            x1,y1,x2,y2 = box
            my1 = y1 + int((y2-y1)*0.52)
            roi = frame[my1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64,32))
            if prev_gray is not None:
                motion.append(float(np.mean(cv2.absdiff(gray,prev_gray))) / 255.0)
            prev_gray = gray
        coverage = len(track["boxes"]) / total_sampled
        score = (float(np.mean(motion)) if motion else 0.0)
        score *= min(1.0, coverage / max(cfg.min_track_coverage,1e-6))
        candidate = (score, coverage, tid, track)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    score, coverage, tid, track = best
    sampled_boxes = {fi: box for fi, box, _ in track["boxes"]}

    # Stream a masked full-resolution output; use nearest sampled track box.
    cap = cv2.VideoCapture(str(source))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    last_box = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in sampled_boxes:
            last_box = sampled_boxes[index]
        canvas = np.zeros_like(frame)
        if last_box:
            x1,y1,x2,y2 = last_box
            px, py = int((x2-x1)*0.18), int((y2-y1)*0.18)
            x1,y1=max(0,x1-px),max(0,y1-py)
            x2,y2=min(w,x2+px),min(h,y2+py)
            canvas[y1:y2,x1:x2]=frame[y1:y2,x1:x2]
        writer.write(canvas)
        index += 1
    writer.release()
    cap.release()
    return ActiveSpeakerResult(output, round(score,5), round(coverage,5), tid, False)
