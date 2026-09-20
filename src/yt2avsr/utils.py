from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_ffmpeg() -> str | None:
    """Ensure ffmpeg and ffprobe are available on PATH, trying venv and static_ffmpeg if needed."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return shutil.which("ffmpeg")
    venv_bin = Path(sys.prefix) / "bin"
    if (venv_bin / "ffmpeg").exists():
        os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")
        if shutil.which("ffmpeg"):
            return shutil.which("ffmpeg")
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        if shutil.which("ffmpeg"):
            return shutil.which("ffmpeg")
    except Exception:
        pass
    return shutil.which("ffmpeg")


ensure_ffmpeg()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        ensure_ffmpeg()
    if shutil.which(name) is None:
        if name == "ffmpeg":
            raise RuntimeError(
                "Required executable not found on PATH: ffmpeg (ffprobe ile birlikte). "
                "Kur: macOS -> 'brew install ffmpeg' | Ubuntu -> "
                "'sudo apt install -y ffmpeg' | Windows -> 'winget install Gyan.FFmpeg'."
            )
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def run(cmd: list[str], *, capture: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip() if capture else ""
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")
    return (proc.stdout or "").strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if cleaned:
        return cleaned[:100]
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text
