from __future__ import annotations
from pathlib import Path
from typing import Any
from faster_whisper import WhisperModel
from .config import TranscriptionConfig
from .utils import normalize_text, write_json

def resolve_device(device: str) -> tuple[str, str]:
    if device != "auto":
        return device, "float16" if device == "cuda" else "int8"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"

def transcribe(video: Path, words_output: Path, language: str,
               cfg: TranscriptionConfig) -> list[dict[str, Any]]:
    device, default_compute = resolve_device(cfg.device)
    compute_type = default_compute if cfg.compute_type == "auto" else cfg.compute_type
    model = WhisperModel(cfg.model, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(video), language=language, beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
        vad_parameters={"min_silence_duration_ms": cfg.min_silence_duration_ms},
        word_timestamps=True, condition_on_previous_text=False,
        temperature=0.0,
    )
    words, segment_rows = [], []
    for segment in segments:
        no_speech = float(segment.no_speech_prob or 0.0)
        avg_logprob = float(segment.avg_logprob or -99.0)
        segment_probability = min(1.0, max(0.0, 1.0 + avg_logprob / 4.0))
        segment_rows.append({
            "start": segment.start, "end": segment.end,
            "avg_logprob": avg_logprob, "no_speech_probability": no_speech,
            "confidence": segment_probability,
        })
        if no_speech > cfg.max_no_speech_probability:
            continue
        for word in segment.words or []:
            token = normalize_text(word.word)
            probability = float(word.probability or 0.0)
            if not token or word.start is None or word.end is None:
                continue
            words.append({
                "word": token, "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
                "probability": round(probability, 5),
                "segment_confidence": round(segment_probability, 5),
            })
    write_json(words_output, {
        "source": "whisper",
        "model": cfg.model,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segment_rows,
        "words": words,
    })
    return words
