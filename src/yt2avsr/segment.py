from __future__ import annotations

from typing import Any

from .config import SegmentationConfig
from .utils import normalize_text


def make_segments(words: list[dict[str, Any]], cfg: SegmentationConfig) -> list[dict[str, Any]]:
    if not words:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for word in words:
        if not current:
            current = [word]
            continue

        proposed_duration = float(word["end"]) - float(current[0]["start"])
        gap = float(word["start"]) - float(current[-1]["end"])
        sentence_break = current[-1]["word"].endswith((".", "!", "?"))

        if proposed_duration > cfg.max_duration or (
            gap > cfg.max_gap_seconds and
            float(current[-1]["end"]) - float(current[0]["start"]) >= cfg.min_duration
        ) or (
            sentence_break and
            float(current[-1]["end"]) - float(current[0]["start"]) >= cfg.min_duration
        ):
            groups.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        groups.append(current)

    merged: list[list[dict[str, Any]]] = []
    for group in groups:
        duration = float(group[-1]["end"]) - float(group[0]["start"])
        if merged and (duration < cfg.min_duration or len(group) < cfg.min_words):
            combined_duration = float(group[-1]["end"]) - float(merged[-1][0]["start"])
            if combined_duration <= cfg.max_duration:
                merged[-1].extend(group)
                continue
        merged.append(group)

    result: list[dict[str, Any]] = []
    for index, group in enumerate(merged):
        if len(group) < cfg.min_words:
            continue
        raw_start = float(group[0]["start"])
        raw_end = float(group[-1]["end"])
        start = max(0.0, raw_start - cfg.pad_seconds)
        end = raw_end + cfg.pad_seconds
        duration = end - start
        if duration < cfg.min_duration or duration > cfg.max_duration + 2 * cfg.pad_seconds:
            continue
        text = normalize_text(" ".join(w["word"] for w in group))
        confidence = sum(min(float(w.get("probability", 0.0)), float(w.get("segment_confidence", 1.0))) for w in group) / len(group)
        result.append(
            {
                "segment_id": f"{index:06d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "text": text,
                "word_count": len(group),
                "asr_confidence": round(confidence, 5),
            }
        )
    return result
