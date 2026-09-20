from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import threading
from typing import Any

from faster_whisper import WhisperModel
from tqdm import tqdm

from .config import TranscriptionConfig
from .utils import normalize_text, write_json


import os
_MODEL_LOCK = threading.Lock()


def _ensure_cuda_libs():
    """Ensure CUDA runtime libraries (cublas, cudnn) from pip packages are preloaded."""
    try:
        import ctypes
        import site
        for sp in site.getsitepackages():
            for sub in ("cublas", "cudnn"):
                lib_dir = os.path.join(sp, "nvidia", sub, "lib")
                if os.path.isdir(lib_dir):
                    for f in sorted(os.listdir(lib_dir)):
                        if ".so" in f:
                            try:
                                ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass
    except Exception:
        pass


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ctranslate2 CPU backend'inde güvenli compute türleri. float16/bfloat16 gibi
# GPU türleri CPU'da ya desteklenmez ya yavaştır; açıkça verilse bile int8'e
# düşülür (uyarıyla).
_CPU_SAFE_COMPUTE = {"int8", "int8_float32", "int8_float16", "int8_bfloat16", "float32"}


def resolve_device(device: str) -> tuple[str, str]:
    # faster-whisper has no MPS backend, so never force MPS: Mac falls back
    # to CPU/int8 which is explicit and supported.
    normalized = (device or "auto").strip().lower()
    if normalized == "mps":
        print(
            "[whisper] device='mps' is not supported by faster-whisper; "
            "falling back to device='cpu' compute='int8'.",
            flush=True,
        )
        return "cpu", "int8"
    if normalized == "cuda":
        if _cuda_available():
            return "cuda", "float16"
        print(
            "[whisper] device='cuda' istendi ama CUDA bulunamadi; "
            "device='cpu' compute='int8' kullaniliyor.",
            flush=True,
        )
        return "cpu", "int8"
    if normalized != "auto":
        return normalized, "int8"
    if _cuda_available():
        return "cuda", "float16"
    return "cpu", "int8"


def resolve_compute(device: str, compute_setting: str) -> str:
    """Resolve compute_type, guarding CPU against GPU-only compute types."""
    if compute_setting != "auto":
        compute = compute_setting
    else:
        compute = "float16" if device == "cuda" else "int8"
    if device == "cpu" and compute not in _CPU_SAFE_COMPUTE:
        print(
            f"[whisper] compute_type='{compute}' CPU'da desteklenmiyor; "
            "compute='int8' kullaniliyor.",
            flush=True,
        )
        return "int8"
    return compute


@lru_cache(maxsize=4)
def _load_model_cached(
    model_name: str, device: str, compute_type: str, num_workers: int
) -> WhisperModel:
    if device == "cuda":
        _ensure_cuda_libs()
    print(
        f"[whisper] loading model={model_name} device={device} compute={compute_type} num_workers={num_workers}",
        flush=True,
    )
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers,
    )


def _load_model(
    model_name: str, device: str, compute_type: str, num_workers: int = 1
) -> WhisperModel:
    with _MODEL_LOCK:
        return _load_model_cached(model_name, device, compute_type, num_workers)


def words_to_text(words: list[dict[str, Any]]) -> str:
    return normalize_text(" ".join(str(word.get("word", "")) for word in words))


def words_confidence(words: list[dict[str, Any]]) -> float:
    if not words:
        return 0.0
    scores = [
        min(
            float(word.get("probability", 0.0)),
            float(word.get("segment_confidence", 1.0)),
        )
        for word in words
    ]
    return round(sum(scores) / len(scores), 5)


def transcript_similarity(first: str, second: str) -> float:
    def comparable(text: str) -> str:
        return normalize_text(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)).casefold()

    left, right = comparable(first), comparable(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return round(SequenceMatcher(None, left.split(), right.split()).ratio(), 5)


def transcribe(video: Path, words_output: Path, language: str,
               cfg: TranscriptionConfig) -> list[dict[str, Any]]:
    device, _ = resolve_device(cfg.device)
    compute_type = resolve_compute(device, cfg.compute_type)
    num_workers = getattr(cfg, "num_workers", 1)
    try:
        model = _load_model(cfg.model, device, compute_type, num_workers=num_workers)
    except Exception as exc:
        raise RuntimeError(
            f"Whisper model '{cfg.model}' yüklenemedi ({exc}). "
            "İnternet bağlantını kontrol edip önceden indirmeyi dene: "
            "ytavsr setup-whisper --config configs/default.yaml"
        ) from exc
    print(f"[whisper] transcribing {video}", flush=True)
    segments, info = model.transcribe(
        str(video), language=language, beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
        vad_parameters={"min_silence_duration_ms": cfg.min_silence_duration_ms},
        word_timestamps=True, condition_on_previous_text=False,
        temperature=0.0,
    )
    words, segment_rows = [], []
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    progress_total = duration if duration > 0 else None
    last_progress = 0.0
    progress = tqdm(
        total=progress_total,
        desc="Whisper transcript",
        unit="sec",
        dynamic_ncols=True,
    )
    try:
        for segment in segments:
            if progress_total is not None:
                current_progress = min(progress_total, float(segment.end or 0.0))
                progress.update(max(0.0, current_progress - last_progress))
                last_progress = current_progress
            else:
                progress.update(1)

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
        if progress_total is not None and last_progress < progress_total:
            progress.update(progress_total - last_progress)
    finally:
        progress.close()
    write_json(words_output, {
        "source": "whisper",
        "model": cfg.model,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segment_rows,
        "words": words,
    })
    print(
        f"[whisper] done: {len(segment_rows)} segments, {len(words)} words",
        flush=True,
    )
    return words
