from __future__ import annotations
import html
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .utils import normalize_text, write_json

_TIME = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[.,](?P<ms>\d{3})")
_TAGS = re.compile(r"<[^>]+>")

def _seconds(token: str) -> float:
    m = _TIME.search(token)
    if not m:
        raise ValueError(token)
    return int(m["h"]) * 3600 + int(m["m"]) * 60 + int(m["s"]) + int(m["ms"]) / 1000

def _choose(entries_by_language: dict[str, Any], languages: list[str]):
    for lang in languages:
        entries = entries_by_language.get(lang) or []
        preferred = sorted(
            entries,
            key=lambda e: (
                0 if e.get("ext") == "vtt" else
                1 if e.get("ext") == "srt" else 2
            ),
        )
        for entry in preferred:
            if entry.get("url") and entry.get("ext") in {"vtt", "srt"}:
                return lang, entry["url"], entry.get("ext")
    return None

def choose_manual_subtitle(info: dict[str, Any], languages: list[str]):
    return _choose(info.get("subtitles") or {}, languages)

def choose_automatic_subtitle(info: dict[str, Any], languages: list[str]):
    return _choose(info.get("automatic_captions") or {}, languages)

def download_text(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        output.write_bytes(response.read())

def parse_vtt(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cues, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue
        left, right = [x.strip().split()[0] for x in line.split("-->", 1)]
        start, end = _seconds(left), _seconds(right)
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = normalize_text(html.unescape(_TAGS.sub("", " ".join(text_lines))))
        if text:
            cues.append({"start": start, "end": end, "text": text})
        i += 1
    return deduplicate(cues)

def deduplicate(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result, previous = [], ""
    for cue in cues:
        original = cue["text"]
        text = original
        if text == previous:
            continue
        if previous and text.startswith(previous):
            text = normalize_text(text[len(previous):])
        if text:
            result.append({**cue, "text": text})
            previous = original
    return result

def cues_to_words(cues: list[dict[str, Any]], probability: float) -> list[dict[str, Any]]:
    words = []
    for cue in cues:
        tokens = cue["text"].split()
        if not tokens:
            continue
        span = max(0.01, cue["end"] - cue["start"])
        step = span / len(tokens)
        for idx, token in enumerate(tokens):
            words.append({
                "word": token,
                "start": round(cue["start"] + idx * step, 3),
                "end": round(cue["start"] + (idx + 1) * step, 3),
                "probability": probability,
            })
    return words

def save_youtube_transcript(
    subtitle_path: Path,
    output: Path,
    language: str,
    *,
    automatic: bool,
) -> list[dict[str, Any]]:
    cues = parse_vtt(subtitle_path)
    # Automatic captions receive a conservative synthetic confidence so they
    # can be routed to review by downstream quality rules when desired.
    probability = 0.82 if automatic else 1.0
    words = cues_to_words(cues, probability)
    write_json(output, {
        "source": "youtube_auto_captions" if automatic else "youtube_manual_subtitles",
        "language": language,
        "language_probability": 1.0,
        "words": words,
    })
    return words
