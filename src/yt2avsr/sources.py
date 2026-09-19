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


def get_source_key(url_or_id: str) -> str | None:
    """Return a normalized unique key for deduplication.

    - YouTube videos: 'video:<11_char_id>'
    - YouTube playlists: 'playlist:<playlist_id>'
    - Other URLs: canonicalized URL or stripped string
    Returns None for comments, blank lines, or invalid inputs.
    """
    raw = url_or_id.strip()
    if not raw or raw.startswith("#"):
        return None

    parts = raw.split(maxsplit=1)
    is_explicit_playlist = len(parts) == 2 and parts[0].lower() == "playlist"
    is_explicit_video = len(parts) == 2 and parts[0].lower() == "video"
    target = parts[1].strip() if (is_explicit_playlist or is_explicit_video) else raw

    if not is_explicit_playlist:
        vid = extract_video_id(target)
        if vid:
            return f"video:{vid}"

    pid = extract_playlist_id(target)
    if pid:
        return f"playlist:{pid}"

    return canonicalize_source(target)


def is_playlist_source(mode: str, url: str) -> bool:
    """Determine whether a source line represents a playlist."""
    if mode == "playlist":
        return True
    if mode == "video":
        return False
    # mode == "auto"
    if extract_video_id(url) is not None:
        return False
    return bool(extract_playlist_id(url) or "/playlist" in url or "list=" in url)


def canonicalize_source(url_or_id: str) -> str:
    """Return standard https://www.youtube.com/watch?v=ID if video ID found,
    or https://www.youtube.com/playlist?list=ID if playlist ID found,
    else stripped URL."""
    raw = url_or_id.strip()
    if not raw or raw.startswith("#"):
        return raw

    parts = raw.split(maxsplit=1)
    prefix = ""
    if len(parts) == 2 and parts[0].lower() in {"video", "playlist"}:
        if parts[0].lower() == "playlist":
            prefix = "playlist "
        clean_target = parts[1].strip()
    else:
        clean_target = raw

    if not prefix:
        vid = extract_video_id(clean_target)
        if vid:
            return f"https://www.youtube.com/watch?v={vid}"

    pid = extract_playlist_id(clean_target)
    if pid:
        return f"{prefix}https://www.youtube.com/playlist?list={pid}".strip()

    return raw


def deduplicate_source_lines(
    lines: Iterable[str],
    *,
    seen_keys: set[str] | None = None,
    canonicalize: bool = False,
) -> tuple[list[str], list[str]]:
    """Deduplicate lines from a source file, preserving comments and empty lines.

    Args:
        lines: Sequence of raw lines from a source file.
        seen_keys: Optional set of already seen keys (mutated in place).
        canonicalize: If True, standardizes video and playlist URLs.

    Returns:
        (deduped_lines, duplicate_lines)
    """
    if seen_keys is None:
        seen_keys = set()

    deduped: list[str] = []
    duplicates: list[str] = []

    for raw in lines:
        line = raw.rstrip("\r\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            deduped.append(line)
            continue

        key = get_source_key(stripped)
        if not key:
            deduped.append(line)
            continue

        if key in seen_keys:
            duplicates.append(stripped)
            continue

        seen_keys.add(key)
        if canonicalize:
            deduped.append(canonicalize_source(stripped))
        else:
            deduped.append(line)

    return deduped, duplicates


def deduplicate_source_file(
    path: Path,
    *,
    in_place: bool = True,
    canonicalize: bool = False,
    seen_keys: set[str] | None = None,
) -> tuple[int, list[str]]:
    """Inspect and deduplicate a sources file.

    Returns:
        (count_removed, list_of_duplicate_urls)
    """
    if not path.exists():
        return 0, []

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    deduped_lines, duplicates = deduplicate_source_lines(
        lines, seen_keys=seen_keys, canonicalize=canonicalize
    )

    if duplicates and in_place:
        new_content = "\n".join(deduped_lines)
        if content.endswith("\n") or not new_content.endswith("\n"):
            new_content += "\n"
        path.write_text(new_content, encoding="utf-8")

    return len(duplicates), duplicates


def resolve_source_inputs(items: Iterable[str]) -> list[str]:
    """Expand a list of items which may contain URLs, IDs, or paths to files."""
    resolved: list[str] = []
    for item in items:
        raw = item.strip()
        if not raw:
            continue
        path_candidate = Path(raw)
        if path_candidate.exists() and path_candidate.is_file():
            for line in path_candidate.read_text(encoding="utf-8").splitlines():
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith("#"):
                    resolved.append(stripped_line)
        else:
            resolved.append(raw)
    return resolved


def add_sources_to_file(
    items: Iterable[str],
    target_path: Path = Path("sources_no_voiceover.txt"),
    *,
    processed_path: Path = Path("processed_sources.txt"),
    other_source_path: Path | None = None,
    canonicalize: bool = True,
) -> dict[str, list[str]]:
    """Add new source URLs or file paths to a target sources file, filtering duplicates.

    Returns a dictionary with:
      - 'added': list of newly added URLs
      - 'duplicates': list of items already in target file or duplicated in input
      - 'already_processed': list of items already in processed_sources.txt
      - 'cross_file_warnings': list of items also present in other_source_path
      - 'invalid': list of unrecognized/invalid items
    """
    resolved = resolve_source_inputs(items)

    # Load existing target file keys
    target_keys: set[str] = set()
    if target_path.exists():
        for line in target_path.read_text(encoding="utf-8").splitlines():
            k = get_source_key(line)
            if k:
                target_keys.add(k)

    # Load processed keys
    processed_ids = load_processed_ids(processed_path)

    # Load other source file keys if provided
    other_keys: set[str] = set()
    if other_source_path and other_source_path.exists():
        for line in other_source_path.read_text(encoding="utf-8").splitlines():
            k = get_source_key(line)
            if k:
                other_keys.add(k)

    added: list[str] = []
    duplicates: list[str] = []
    already_processed: list[str] = []
    cross_warnings: list[str] = []
    invalid: list[str] = []

    lines_to_append: list[str] = []

    for item in resolved:
        k = get_source_key(item)
        if not k:
            invalid.append(item)
            continue

        if is_source_processed(item, processed_ids):
            already_processed.append(item)
            continue

        if k in target_keys:
            duplicates.append(item)
            continue

        if k in other_keys:
            cross_warnings.append(item)

        target_keys.add(k)
        cleaned = canonicalize_source(item) if canonicalize else item
        added.append(cleaned)
        lines_to_append.append(cleaned)

    if lines_to_append:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        current_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        separator = "\n" if current_content and not current_content.endswith("\n") else ""
        text_to_append = separator + "\n".join(lines_to_append) + "\n"
        with target_path.open("a", encoding="utf-8") as f:
            f.write(text_to_append)

    return {
        "added": added,
        "duplicates": duplicates,
        "already_processed": already_processed,
        "cross_file_warnings": cross_warnings,
        "invalid": invalid,
    }


def load_processed_ids(path: Path = Path("processed_sources.txt")) -> set[str]:
    """Load all processed video IDs and URLs from a tracking file."""
    if not path.exists():
        return set()

    processed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        key = get_source_key(line)
        if key:
            processed.add(key)

        vid = extract_video_id(line)
        if vid:
            processed.add(vid)
            processed.add(f"video:{vid}")
        # Also store raw normalized string for direct matching
        processed.add(line)
        processed.add(canonicalize_source(line))
    return processed


def is_source_processed(url_or_id: str, processed_ids: set[str]) -> bool:
    """Check if a video URL or ID is already in processed_ids."""
    key = get_source_key(url_or_id)
    if key and key in processed_ids:
        return True

    vid = extract_video_id(url_or_id)
    if vid and (vid in processed_ids or f"video:{vid}" in processed_ids):
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


def parse_shard(shard_str: str | None) -> tuple[int, int] | None:
    """Parse and validate a shard specification like '0/3', '1/3', '2/3'.

    Returns (index, total) as 0-indexed integers, or None if shard_str is None.
    Raises ValueError on invalid format.
    """
    if not shard_str:
        return None
    raw = shard_str.strip()
    if "/" not in raw:
        raise ValueError(
            f"Invalid shard format {shard_str!r}. Expected format: <index>/<total> (e.g. 0/3, 1/3, 2/3)"
        )
    parts = raw.split("/", 1)
    try:
        index, total = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"Invalid shard format {shard_str!r}. Index and total must be integers (e.g. 0/3, 1/3, 2/3)"
        )
    if total < 1:
        raise ValueError(f"Shard total must be at least 1, got {total}")
    if not (0 <= index < total):
        raise ValueError(
            f"Shard index {index} is out of bounds for total {total}. Must be between 0 and {total - 1}."
        )
    return index, total


def filter_by_shard(items: list[Any], shard: tuple[int, int] | None) -> list[Any]:
    """Return only the items belonging to the given shard (0-indexed modulo)."""
    if not shard or shard[1] <= 1:
        return items
    index, total = shard
    return [item for i, item in enumerate(items) if (i % total) == index]


def partition_sources(
    sources: list[tuple[str, str]],
    shard: tuple[int, int] | None,
) -> list[tuple[str, str]]:
    """Partition a list of (mode, url) sources for a given shard.

    Single video URLs are distributed across shards (i % total == index).
    Playlist URLs are returned for all shards because their individual items
    are partitioned internally during download.
    """
    if not shard or shard[1] <= 1:
        return sources
    index, total = shard
    playlist_sources: list[tuple[str, str]] = []
    single_sources: list[tuple[str, str]] = []
    for mode, url in sources:
        if is_playlist_source(mode, url):
            playlist_sources.append((mode, url))
        else:
            single_sources.append((mode, url))

    sharded_singles = [
        item for i, item in enumerate(single_sources) if (i % total) == index
    ]
    return sharded_singles + playlist_sources


