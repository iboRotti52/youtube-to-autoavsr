from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

if "huggingface_hub" not in sys.modules:
    mock_hf = MagicMock()
    sys.modules["huggingface_hub"] = mock_hf

from yt2avsr.sources import (
    append_processed_sources,
    canonicalize_source,
    extract_playlist_id,
    extract_video_id,
    is_source_processed,
    load_processed_ids,
    sync_processed_from_hf,
)


def test_extract_video_id():
    test_cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("video https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/playlist?list=PL1234567890", None),
        ("# https://www.youtube.com/watch?v=dQw4w9WgXcQ", None),
        ("", None),
    ]
    for raw, expected in test_cases:
        assert extract_video_id(raw) == expected, f"Failed for {raw}: got {extract_video_id(raw)}"


def test_extract_playlist_id():
    assert extract_playlist_id("https://www.youtube.com/playlist?list=PL12345") == "PL12345"
    assert extract_playlist_id("playlist https://www.youtube.com/playlist?list=PL12345") == "PL12345"
    assert extract_playlist_id("PL1234567890abcdef") == "PL1234567890abcdef"


def test_canonicalize_source():
    assert canonicalize_source("https://youtu.be/dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert canonicalize_source("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_load_and_append_processed_sources():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "processed_sources.txt"

        # Initially empty
        assert len(load_processed_ids(path)) == 0

        # Append one
        added = append_processed_sources(
            [{"id": "dQw4w9WgXcQ", "title": "Never Gonna Give You Up"}],
            path=path,
        )
        assert added == 1

        processed = load_processed_ids(path)
        assert "dQw4w9WgXcQ" in processed
        assert is_source_processed("https://youtu.be/dQw4w9WgXcQ", processed)
        assert is_source_processed("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5", processed)
        assert is_source_processed("dQw4w9WgXcQ", processed)
        assert not is_source_processed("https://youtu.be/otherVideo12", processed)

        # Duplicate append shouldn't add anything
        added_dup = append_processed_sources(
            [{"id": "dQw4w9WgXcQ", "title": "Never Gonna Give You Up"}],
            path=path,
        )
        assert added_dup == 0

        # Add string URL
        added_url = append_processed_sources(["https://youtu.be/abc123xyz99"], path=path)
        assert added_url == 1

        processed_after = load_processed_ids(path)
        assert is_source_processed("https://www.youtube.com/watch?v=abc123xyz99", processed_after)


def test_sync_processed_from_hf_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "processed_sources.txt"

        sample_csv = "item_id,source_url,title\nvid00000001,https://www.youtube.com/watch?v=vid00000001,Sample 1\nvid00000002,https://www.youtube.com/watch?v=vid00000002,Sample 2\n"

        with patch("huggingface_hub.HfApi") as mock_api_cls, patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            mock_api.list_repo_files.return_value = [
                "data/user1/manifests/accepted.csv",
                "data/user2/clips/vid00000003/000000/metadata.json",
            ]

            csv_file = Path(tmpdir) / "accepted.csv"
            csv_file.write_text(sample_csv, encoding="utf-8")
            mock_download.return_value = str(csv_file)

            records = sync_processed_from_hf("mock/repo", path=path)

            assert len(records) == 3
            item_ids = {r["id"] for r in records}
            assert "vid00000001" in item_ids
            assert "vid00000002" in item_ids
            assert "vid00000003" in item_ids

            processed = load_processed_ids(path)
            assert is_source_processed("vid00000001", processed)
            assert is_source_processed("https://www.youtube.com/watch?v=vid00000003", processed)
