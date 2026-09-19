from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

# Standard YouTube 11-character video ID regex
YOUTUBE_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str | None:
    """Extract YouTube video ID from a URL, command line argument, or plain ID."""
    raw = url_or_id.strip()
    if not raw or raw.startswith("#"):
        return None

    # Handle optional prefix: "video https://..."
    parts = raw.split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() in {"video", "playlist"}:
        raw = parts[1].strip()

    # Plain 11-char ID
    if YOUTUBE_ID_REGEX.match(raw):
        return raw

    try:
        parsed = urlparse(raw)
    except Exception:
        return None

    hostname = (parsed.hostname or "").lower()

    # youtu.be/<video_id>
    if "youtu.be" in hostname:
        path = parsed.path.strip("/")
        vid = path.split("/")[0] if path else ""
        if YOUTUBE_ID_REGEX.match(vid):
            return vid

    # youtube.com/watch?v=<video_id>
    if "youtube.com" in hostname:
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            vid = qs.get("v", [""])[0]
            if YOUTUBE_ID_REGEX.match(vid):
                return vid

        # youtube.com/embed/<video_id>, /v/<video_id>, /shorts/<video_id>, /live/<video_id>
        for prefix in ("/embed/", "/v/", "/shorts/", "/live/"):
            if parsed.path.startswith(prefix):
                candidate = parsed.path[len(prefix):].split("/")[0]
                if YOUTUBE_ID_REGEX.match(candidate):
                    return candidate

    return None


def extract_playlist_id(url_or_id: str) -> str | None:
    """Extract YouTube playlist ID from a URL or plain ID."""
    raw = url_or_id.strip()
    if not raw or raw.startswith("#"):
        return None

    parts = raw.split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() in {"video", "playlist"}:
        raw = parts[1].strip()

    try:
        parsed = urlparse(raw)
        if "youtube.com" in (parsed.hostname or "").lower():
            qs = parse_qs(parsed.query)
            if "list" in qs and qs["list"]:
                return qs["list"][0]
    except Exception:
        pass

    if raw.startswith(("PL", "UU", "FL", "RD", "OL")):
        return raw
    return None


def canonicalize_source(url_or_id: str) -> str:
    """Return standard https://www.youtube.com/watch?v=ID if video ID found, else stripped URL."""
    vid = extract_video_id(url_or_id)
    if vid:
        return f"https://www.youtube.com/watch?v={vid}"
    return url_or_id.strip()


def load_processed_ids(path: Path = Path("processed_sources.txt")) -> set[str]:
    """Load all processed video IDs and URLs from a tracking file."""
    if not path.exists():
        return set()

    processed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        vid = extract_video_id(line)
        if vid:
            processed.add(vid)
        # Also store raw normalized string for direct matching
        processed.add(line)
        processed.add(canonicalize_source(line))
    return processed


def is_source_processed(url_or_id: str, processed_ids: set[str]) -> bool:
    """Check if a video URL or ID is already in processed_ids."""
    vid = extract_video_id(url_or_id)
    if vid and vid in processed_ids:
        return True

    raw = url_or_id.strip()
    if raw in processed_ids or canonicalize_source(raw) in processed_ids:
        return True
    return False


def append_processed_sources(
    records: Iterable[dict[str, Any] | str],
    path: Path = Path("processed_sources.txt"),
) -> int:
    """Append new processed video records or URLs to processed_sources.txt.

    Returns the number of newly added entries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_processed_ids(path)

    new_lines: list[str] = []
    added_count = 0

    for item in records:
        if isinstance(item, str):
            raw = item.strip()
            vid = extract_video_id(raw)
            title = None
            url = f"https://www.youtube.com/watch?v={vid}" if vid else raw
        else:
            vid = item.get("id") or item.get("item_id")
            title = item.get("title")
            url = item.get("source_url") or (
                f"https://www.youtube.com/watch?v={vid}" if vid else ""
            )

        key = vid or url
        if not key:
            continue

        if is_source_processed(key, existing_ids):
            continue

        if title:
            new_lines.append(f"# {title}")
        new_lines.append(canonicalize_source(url or key))
        if vid:
            existing_ids.add(vid)
        existing_ids.add(canonicalize_source(url or key))
        added_count += 1

    if new_lines:
        current_content = path.read_text(encoding="utf-8") if path.exists() else ""
        separator = "\n" if current_content and not current_content.endswith("\n") else ""
        text_to_append = separator + "\n".join(new_lines) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(text_to_append)

    return added_count


def sync_processed_from_hf(
    repo_id: str,
    token: str | None = None,
    path: Path = Path("processed_sources.txt"),
) -> list[dict[str, Any]]:
    """Scan Hugging Face dataset repository for manifests and clips uploaded by all contributors.

    Returns a list of all unique video records discovered.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from .cloud import _get_token

    token = _get_token(token)
    api = HfApi(token=token)

    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(f"Failed to access Hugging Face dataset '{repo_id}': {exc}") from exc

    manifest_files = [
        f for f in repo_files
        if f.endswith(("all.csv", "accepted.csv", "manifests/all.jsonl"))
    ]

    discovered_records: dict[str, dict[str, Any]] = {}

    for manifest_path in manifest_files:
        try:
            local_manifest = hf_hub_download(
                repo_id=repo_id,
                filename=manifest_path,
                repo_type="dataset",
                token=token,
            )
            manifest_content = Path(local_manifest).read_text(encoding="utf-8", errors="replace")

            if manifest_path.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(manifest_content))
                for row in reader:
                    item_id = row.get("item_id")
                    if not item_id:
                        continue
                    if item_id not in discovered_records:
                        discovered_records[item_id] = {
                            "id": item_id,
                            "item_id": item_id,
                            "source_url": row.get("source_url") or f"https://www.youtube.com/watch?v={item_id}",
                            "title": row.get("title"),
                            "channel": row.get("channel"),
                        }
            elif manifest_path.endswith(".jsonl"):
                import json
                for line in manifest_content.splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    item_id = row.get("item_id")
                    if not item_id:
                        continue
                    if item_id not in discovered_records:
                        discovered_records[item_id] = {
                            "id": item_id,
                            "item_id": item_id,
                            "source_url": row.get("source_url") or f"https://www.youtube.com/watch?v={item_id}",
                            "title": row.get("title"),
                            "channel": row.get("channel"),
                        }
        except Exception as exc:
            print(f"[warning] Could not read manifest '{manifest_path}' from HF: {exc}")

    # Fallback / augment: scan clips paths if manifests weren't found or complete
    # e.g. data/<contributor>/clips/<item_id>/...
    clip_pattern = re.compile(r"data/[^/]+/clips/([^/]+)/")
    for f in repo_files:
        match = clip_pattern.search(f)
        if match:
            item_id = match.group(1)
            if item_id not in discovered_records:
                discovered_records[item_id] = {
                    "id": item_id,
                    "item_id": item_id,
                    "source_url": f"https://www.youtube.com/watch?v={item_id}",
                    "title": None,
                }

    records = list(discovered_records.values())
    if records:
        append_processed_sources(records, path)

    return records
