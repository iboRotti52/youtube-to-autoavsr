from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yt_dlp

from .config import DownloadConfig
from .subtitles import choose_automatic_subtitle, choose_manual_subtitle, download_text
from .utils import safe_id, write_json


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def _ydl_options(
    out_dir: Path,
    cfg: DownloadConfig,
    *,
    playlist: bool,
    format_selector: str | None = None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "format": format_selector or cfg.format,
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s" / "download.%(ext)s"),
        "writeinfojson": True,
        "noplaylist": not playlist,
        "quiet": False,
        "ignoreerrors": False,
        "retries": cfg.retries,
        "fragment_retries": cfg.fragment_retries,
        "socket_timeout": cfg.socket_timeout,
        # Official yt-dlp API option corresponding to:
        # --remote-components ejs:github
        "remote_components": set(cfg.remote_components),
        # Explicitly allow the runtime already detected on the user's Mac.
        "js_runtimes": {cfg.js_runtime: {}},
        # Produce a conventional MP4 when yt-dlp initially merges to WebM/MKV.
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
        ],
    }
    if playlist and cfg.playlist_end:
        opts["playlistend"] = cfg.playlist_end
    return opts


def _extract_with_fallback(
    url: str,
    raw_root: Path,
    cfg: DownloadConfig,
    *,
    playlist: bool,
) -> dict[str, Any]:
    selectors = [
        cfg.format,
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "bv*+ba/b",
    ]
    last_error: Exception | None = None
    used: set[str] = set()

    for selector in selectors:
        if selector in used:
            continue
        used.add(selector)
        try:
            with yt_dlp.YoutubeDL(
                _ydl_options(
                    raw_root,
                    cfg,
                    playlist=playlist,
                    format_selector=selector,
                )
            ) as ydl:
                return ydl.extract_info(url, download=True)
        except Exception as exc:
            last_error = exc
            print(f"[yt-dlp retry] Format selector failed: {selector}")
            print(f"[yt-dlp retry] Reason: {exc}")

    assert last_error is not None
    raise RuntimeError(
        "yt-dlp failed after all automatic EJS/format fallbacks"
    ) from last_error


def _find_downloaded_video(item_dir: Path) -> Path:
    candidates = [
        p for p in item_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.name.endswith(".part")
    ]
    # Prefer the final remuxed file, then the largest valid media file.
    candidates.sort(
        key=lambda p: (
            p.suffix.lower() == ".mp4",
            p.stat().st_size,
        ),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed video found under {item_dir}")
    return candidates[0]


def download(
    url: str,
    raw_root: Path,
    cfg: DownloadConfig,
    *,
    playlist: bool = False,
) -> list[dict]:
    raw_root.mkdir(parents=True, exist_ok=True)
    info = _extract_with_fallback(
        url, raw_root, cfg, playlist=playlist
    )

    entries = info.get("entries") if isinstance(info, dict) else None
    items = [entry for entry in entries if entry] if entries else [info]
    results: list[dict] = []

    for item in items:
        video_id = safe_id(str(item["id"]))
        item_dir = raw_root / video_id
        downloaded = _find_downloaded_video(item_dir)

        # Keep the real extension. FFmpeg normalization reads any supported container.
        source = item_dir / f"source{downloaded.suffix.lower()}"
        if downloaded != source:
            if source.exists():
                source.unlink()
            shutil.move(str(downloaded), source)

        subtitle = choose_manual_subtitle(item, cfg.subtitle_languages)
        subtitle_kind = "manual"
        if subtitle is None and cfg.use_automatic_youtube_captions:
            subtitle = choose_automatic_subtitle(item, cfg.subtitle_languages)
            subtitle_kind = "automatic"

        subtitle_path = None
        subtitle_language = None
        if subtitle:
            subtitle_language, subtitle_url, subtitle_ext = subtitle
            subtitle_path = item_dir / f"{subtitle_kind}_subtitles.{subtitle_ext}"
            download_text(subtitle_url, subtitle_path)

        metadata = {
            "id": video_id,
            "title": item.get("title"),
            "source_url": item.get("webpage_url") or url,
            "channel": item.get("channel"),
            "channel_id": item.get("channel_id"),
            "upload_date": item.get("upload_date"),
            "duration": item.get("duration"),
            "license": item.get("license"),
            "subtitle_source": subtitle_kind if subtitle_path else None,
            "subtitle_language": subtitle_language,
            "download_container": source.suffix.lower(),
        }
        write_json(item_dir / "metadata.json", metadata)
        results.append(
            {
                "id": video_id,
                "source": source,
                "metadata": metadata,
                "subtitle_path": subtitle_path,
                "subtitle_automatic": subtitle_kind == "automatic" if subtitle_path else False,
            }
        )
    return results


def register_local(path: Path, raw_root: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    video_id = safe_id(path.stem)
    item_dir = raw_root / video_id
    item_dir.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() if path.suffix else ".mp4"
    target = item_dir / f"source{suffix}"
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    metadata = {
        "id": video_id,
        "title": path.stem,
        "source_url": f"file://{path.resolve()}",
        "channel": None,
        "license": None,
        "subtitle_source": None,
        "download_container": suffix,
    }
    write_json(item_dir / "metadata.json", metadata)
    return {
        "id": video_id,
        "source": target,
        "metadata": metadata,
        "subtitle_path": None,
        "subtitle_automatic": False,
    }
