from __future__ import annotations

import json
from pathlib import Path

from .config import NormalizationConfig
from .utils import require_binary, run


def normalize(source: Path, output: Path, cfg: NormalizationConfig) -> None:
    require_binary("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={cfg.fps},"
        f"scale='if(gt(ih,{cfg.max_height}),-2,iw)':"
        f"'if(gt(ih,{cfg.max_height}),{cfg.max_height},ih)',"
        "setsar=1"
    )
    run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ac", "1", "-ar", str(cfg.audio_sample_rate),
            "-movflags", "+faststart",
            str(output),
        ]
    )


def extract_clip(
    source: Path,
    start: float,
    end: float,
    video_output: Path,
    audio_output: Path,
    cfg: NormalizationConfig,
) -> None:
    require_binary("ffmpeg")
    duration = max(0.01, end - start)
    video_output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-r", str(cfg.fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ac", "1", "-ar", str(cfg.audio_sample_rate),
            "-movflags", "+faststart",
            str(video_output),
        ]
    )
    run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-vn", "-ac", "1", "-ar", str(cfg.audio_sample_rate),
            "-c:a", "pcm_s16le", str(audio_output),
        ]
    )


def probe_duration(path: Path) -> float:
    require_binary("ffprobe")
    raw = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture=True,
    )
    return float(json.loads(raw)["format"]["duration"])
