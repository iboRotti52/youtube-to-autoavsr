from __future__ import annotations

import re
from typing import Any

from .config import SegmentationConfig
from .utils import normalize_text

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")


def remove_adjacent_repeated_sentences(text: str) -> str:
    sentences = [m.group(0).strip() for m in _SENTENCE.finditer(text)]
    if not sentences:
        return normalize_text(text)

    result: list[str] = []
    previous_key = ""
    for sentence in sentences:
        key = normalize_text(sentence).casefold().strip(".!?")
        if key and key == previous_key:
            continue
        result.append(sentence)
        previous_key = key
    return normalize_text(" ".join(result))


def make_segments(
    words: list[dict[str, Any]],
    cfg: SegmentationConfig,
    cut_times: list[float] | None = None,
    strict_sentence_boundaries: bool = False,
) -> list[dict[str, Any]]:
    if not words:
        return []

    cuts = sorted(cut_times) if cut_times else []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for word in words:
        if not current:
            current = [word]
            continue

        proposed_duration = float(word["end"]) - float(current[0]["start"])
        gap = float(word["start"]) - float(current[-1]["end"])
        sentence_break = current[-1]["word"].endswith((".", "!", "?"))

        # Bu kelimeyi eklemek segmentin süresine bir sahne kesmesi sokuyor mu?
        # (kesme, önceki kelimenin sonu ile bu kelimenin sonu arasında ise).
        # Öyleyse segmenti kesmeden ÖNCE kapat; kesme iki grup arasına düşer,
        # hiçbir klip bir cut'ı kapsamaz.
        crosses_cut = cfg.split_on_scene_cut and any(
            float(current[-1]["end"]) <= c <= float(word["end"]) for c in cuts
        )

        if crosses_cut or proposed_duration > cfg.max_duration or (
            strict_sentence_boundaries and sentence_break
        ) or (
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
        if (
            not strict_sentence_boundaries
            and merged
            and (duration < cfg.min_duration or len(group) < cfg.min_words)
        ):
            combined_duration = float(group[-1]["end"]) - float(merged[-1][0]["start"])
            # Araya sahne kesmesi giriyorsa birleştirme; aksi halde klip bir
            # cut'ı kapsar ve akış yine bozulur.
            spans_cut = cfg.split_on_scene_cut and any(
                float(merged[-1][-1]["end"]) <= c <= float(group[0]["start"])
                for c in cuts
            )
            if combined_duration <= cfg.max_duration and not spans_cut:
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
        # Padding bir sahne kesmesini geçmemeli; aksi halde klip yine cut içerir.
        if cuts:
            before = [c for c in cuts if c <= raw_start]
            after = [c for c in cuts if c >= raw_end]
            if before:
                start = max(start, before[-1])
            if after:
                end = min(end, after[0])
        duration = end - start
        if duration < cfg.min_duration or duration > cfg.max_duration + 2 * cfg.pad_seconds:
            continue
        text = remove_adjacent_repeated_sentences(
            normalize_text(" ".join(w["word"] for w in group))
        )
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
