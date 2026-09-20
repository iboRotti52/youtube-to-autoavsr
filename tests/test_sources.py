from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

if "huggingface_hub" not in sys.modules:
    mock_hf = MagicMock()
    sys.modules["huggingface_hub"] = mock_hf

from yt2avsr.sources import (
    add_sources_to_file,
    append_processed_sources,
    canonicalize_source,
    deduplicate_source_file,
    deduplicate_source_lines,
    extract_playlist_id,
    extract_video_id,
    filter_by_shard,
    get_shard_owner,
    get_source_key,
    is_playlist_source,
    is_source_processed,
    load_processed_ids,
    parse_shard,
    partition_sources,
    resolve_source_inputs,
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


def test_parse_shard():
    assert parse_shard(None) is None
    assert parse_shard("0/3") == (0, 3)
    assert parse_shard("1/3") == (1, 3)
    assert parse_shard("2/3") == (2, 3)
    assert parse_shard("0/1") == (0, 1)

    # Shorthand numbers
    assert parse_shard("0") == (0, 3)
    assert parse_shard("1") == (1, 3)
    assert parse_shard("2") == (2, 3)

    # Fixed team member names
    assert parse_shard("ibrahim-gozlukaya") == (0, 3)
    assert parse_shard("gozlukaya") == (0, 3)
    assert parse_shard("damla") == (1, 3)
    assert parse_shard("damla-kemal") == (1, 3)
    assert parse_shard("ibrahim-billurcu") == (2, 3)
    assert parse_shard("billurcu") == (2, 3)

    # Shard owners
    assert get_shard_owner((0, 3)) == "İbrahim Gözlükaya"
    assert get_shard_owner((1, 3)) == "Damla Kemal"
    assert get_shard_owner((2, 3)) == "İbrahim Billurcu"
    assert get_shard_owner((0, 1)) is None

    # Ambiguous "ibrahim" should raise ValueError
    try:
        parse_shard("ibrahim")
        assert False, "Expected ValueError for ambiguous 'ibrahim'"
    except ValueError as exc:
        assert "iki İbrahim" in str(exc)

    for invalid in ["3/3", "-1/3", "foo", "0/0", "1/-2", "4"]:
        try:
            parse_shard(invalid)
            assert False, f"Expected ValueError for {invalid}"
        except ValueError:
            pass



def test_filter_by_shard():
    items = list(range(9))
    s0 = filter_by_shard(items, (0, 3))
    s1 = filter_by_shard(items, (1, 3))
    s2 = filter_by_shard(items, (2, 3))

    assert s0 == [0, 3, 6]
    assert s1 == [1, 4, 7]
    assert s2 == [2, 5, 8]
    assert sorted(s0 + s1 + s2) == items


def test_partition_sources():
    sources = [
        ("auto", "https://youtube.com/watch?v=video000001"),
        ("auto", "https://youtube.com/watch?v=video000002"),
        ("auto", "https://youtube.com/watch?v=video000003"),
        ("playlist", "https://youtube.com/playlist?list=PL123"),
    ]

    p0 = partition_sources(sources, (0, 2))
    p1 = partition_sources(sources, (1, 2))

    # Playlist is present in both partitions
    assert ("playlist", "https://youtube.com/playlist?list=PL123") in p0
    assert ("playlist", "https://youtube.com/playlist?list=PL123") in p1

    # Single videos are partitioned
    v0_urls = [url for mode, url in p0 if mode != "playlist"]
    v1_urls = [url for mode, url in p1 if mode != "playlist"]

    assert v0_urls == ["https://youtube.com/watch?v=video000001", "https://youtube.com/watch?v=video000003"]
    assert v1_urls == ["https://youtube.com/watch?v=video000002"]


def test_get_source_key():
    # Various representations of the same video
    urls = [
        "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PLNVE_O4kpEJgr50ASEwQLNBgTEthsBw-W&index=114",
        "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PLNVE_O4kpEJgr50ASEwQLNBgTEthsBw-W&index=115",
        "https://youtu.be/kb95KGOUZ48",
        "https://www.youtube.com/shorts/kb95KGOUZ48",
        "video https://www.youtube.com/watch?v=kb95KGOUZ48",
        "kb95KGOUZ48",
    ]
    for url in urls:
        assert get_source_key(url) == "video:kb95KGOUZ48"

    # Playlists
    assert get_source_key("https://www.youtube.com/playlist?list=PL12345") == "playlist:PL12345"
    assert get_source_key("playlist https://www.youtube.com/watch?v=xyz&list=PL12345") == "playlist:PL12345"

    # Comments and empty lines
    assert get_source_key("# some comment") is None
    assert get_source_key("   ") is None


def test_is_playlist_source():
    assert not is_playlist_source("auto", "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PL123")
    assert not is_playlist_source("video", "https://www.youtube.com/playlist?list=PL123")
    assert is_playlist_source("playlist", "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PL123")
    assert is_playlist_source("auto", "https://www.youtube.com/playlist?list=PL123")


def test_deduplicate_source_lines():
    lines = [
        "# Initial comment",
        "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PL123&index=114",
        "",
        "# Next section",
        "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PL123&index=115",
        "https://youtu.be/uniqueVid01",
        "https://www.youtube.com/watch?v=uniqueVid01",
    ]

    deduped, dups = deduplicate_source_lines(lines)

    assert len(dups) == 2
    assert "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PL123&index=115" in dups
    assert "https://www.youtube.com/watch?v=uniqueVid01" in dups

    # Ensure comments and empty lines are preserved in order
    assert deduped == [
        "# Initial comment",
        "https://www.youtube.com/watch?v=kb95KGOUZ48&list=PL123&index=114",
        "",
        "# Next section",
        "https://youtu.be/uniqueVid01",
    ]


def test_deduplicate_source_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "sources.txt"
        file_path.write_text(
            "# Test file\n"
            "https://www.youtube.com/watch?v=kb95KGOUZ48&index=1\n"
            "https://www.youtube.com/watch?v=kb95KGOUZ48&index=2\n"
            "https://youtu.be/otherVid001\n",
            encoding="utf-8",
        )

        removed_count, dups = deduplicate_source_file(file_path, in_place=True)
        assert removed_count == 1
        assert len(dups) == 1

        updated_content = file_path.read_text(encoding="utf-8").splitlines()
        assert len(updated_content) == 3
        assert updated_content[0] == "# Test file"
        assert "index=1" in updated_content[1]
        assert "otherVid001" in updated_content[2]


def test_resolve_source_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_file = Path(tmpdir) / "batch.txt"
        batch_file.write_text(
            "# A comment line\n"
            "https://youtu.be/fileVid0001\n"
            "\n"
            "https://youtu.be/fileVid0002\n",
            encoding="utf-8",
        )

        inputs = [
            "https://youtu.be/singleVid01",
            str(batch_file),
            "https://youtu.be/singleVid02",
        ]

        resolved = resolve_source_inputs(inputs)
        assert resolved == [
            "https://youtu.be/singleVid01",
            "https://youtu.be/fileVid0001",
            "https://youtu.be/fileVid0002",
            "https://youtu.be/singleVid02",
        ]


def test_add_sources_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "sources_no_voiceover.txt"
        target.write_text(
            "# Header\n"
            "https://www.youtube.com/watch?v=vidAlrdyIn1\n",
            encoding="utf-8",
        )
        proc = Path(tmpdir) / "processed_sources.txt"
        proc.write_text(
            "https://www.youtube.com/watch?v=vidProcDone\n",
            encoding="utf-8",
        )
        other = Path(tmpdir) / "sources_voiceover.txt"
        other.write_text(
            "https://www.youtube.com/watch?v=vidVoiceov1\n",
            encoding="utf-8",
        )

        batch_file = Path(tmpdir) / "incoming.txt"
        batch_file.write_text(
            "https://youtu.be/vidFromFile\n"
            "https://www.youtube.com/watch?v=vidAlrdyIn1\n",
            encoding="utf-8",
        )

        to_add = [
            "https://www.youtube.com/watch?v=vidBrandNew&list=PL123&index=1",  # clean video
            "https://youtu.be/vidProcDone",  # already processed -> skip
            "https://youtu.be/vidVoiceov1",  # cross file warning, but add
            str(batch_file),  # 1 new, 1 duplicate in target
        ]

        result = add_sources_to_file(
            to_add,
            target_path=target,
            processed_path=proc,
            other_source_path=other,
            canonicalize=True,
        )

        assert len(result["added"]) == 3
        # Should be canonicalized:
        assert "https://www.youtube.com/watch?v=vidBrandNew" in result["added"]
        assert "https://www.youtube.com/watch?v=vidVoiceov1" in result["added"]
        assert "https://www.youtube.com/watch?v=vidFromFile" in result["added"]

        assert len(result["already_processed"]) == 1
        assert "https://youtu.be/vidProcDone" in result["already_processed"]

        assert len(result["duplicates"]) == 1
        assert "https://www.youtube.com/watch?v=vidAlrdyIn1" in result["duplicates"]

        assert len(result["cross_file_warnings"]) == 1
        assert "https://youtu.be/vidVoiceov1" in result["cross_file_warnings"]

        # Verify file content on disk
        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        assert lines[0] == "# Header"
        assert lines[1] == "https://www.youtube.com/watch?v=vidAlrdyIn1"
        assert lines[2] == "https://www.youtube.com/watch?v=vidBrandNew"
        assert lines[3] == "https://www.youtube.com/watch?v=vidVoiceov1"
        assert lines[4] == "https://www.youtube.com/watch?v=vidFromFile"


def test_resolve_source_inputs_with_quotes_and_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_file = Path(tmp_dir) / "links.txt"
        sample_file.write_text(
            '# Comment\n"https://www.youtube.com/watch?v=vidSample1",\n\'https://www.youtube.com/watch?v=vidSample2\'\n',
            encoding="utf-8-sig",
        )
        resolved = resolve_source_inputs([str(sample_file), '"https://youtu.be/vidSample3",'])
        assert resolved == [
            "https://www.youtube.com/watch?v=vidSample1",
            "https://www.youtube.com/watch?v=vidSample2",
            "https://youtu.be/vidSample3",
        ]


def test_sync_processed_from_hf_with_completion_ledger(tmp_path):
    """Verify that sync_processed_from_hf extracts 100% rejected videos from completion ledgers."""
    import json
    from yt2avsr.sources import sync_processed_from_hf, load_processed_ids, is_source_processed

    ledger_content = "\n".join([
        json.dumps({
            "run_id": "run_12345",
            "item_id": "rej00000001",
            "source_url": "https://www.youtube.com/watch?v=rej00000001",
            "status": "completed",
            "accepted_clips": 0,
            "review_clips": 0,
            "rejected_clips": 10,
        }),
        json.dumps({
            "run_id": "run_12345",
            "item_id": "acc00000002",
            "source_url": "https://www.youtube.com/watch?v=acc00000002",
            "status": "completed",
            "accepted_clips": 5,
            "review_clips": 1,
            "rejected_clips": 0,
        }),
    ])

    fake_ledger_file = tmp_path / "run_12345.jsonl"
    fake_ledger_file.write_text(ledger_content, encoding="utf-8")

    with patch("huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.list_repo_files.return_value = [
            "data/contributor_a/completions/run_12345.jsonl",
        ]
        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_ledger_file)):
            proc_file = tmp_path / "processed_sources.txt"
            records = sync_processed_from_hf("repo/id", path=proc_file)

            assert len(records) == 2
            item_ids = {r["id"] for r in records}
            assert "rej00000001" in item_ids
            assert "acc00000002" in item_ids

            proc_ids = load_processed_ids(proc_file)
            assert is_source_processed("rej00000001", proc_ids)
            assert is_source_processed("acc00000002", proc_ids)
            assert is_source_processed("https://www.youtube.com/watch?v=rej00000001", proc_ids)


def test_downloader_playlist_sharding_no_double_modulo(tmp_path):
    """Verify that downloader does not apply modulo twice on items with playlist_index."""
    from yt2avsr.config import DownloadConfig
    from yt2avsr.downloader import download

    cfg = DownloadConfig()
    cfg.format = "best"

    # Simulate yt-dlp returning 2 items that matched shard 0 of (0, 2) with playlist_index=1 and 3
    fake_info = {
        "_type": "playlist",
        "entries": [
            {"id": "vid_item_1", "title": "Item 1", "playlist_index": 1},
            {"id": "vid_item_3", "title": "Item 3", "playlist_index": 3},
        ]
    }

    # Setup directories and fake video files so _find_downloaded_video succeeds
    for vid in ["vid_item_1", "vid_item_3"]:
        d = tmp_path / vid
        d.mkdir(parents=True)
        (d / "video.mp4").write_text("content")

    with patch("yt2avsr.downloader._extract_with_fallback", return_value=fake_info):
        res = download(
            "https://youtube.com/playlist?list=PL123",
            tmp_path,
            cfg,
            playlist=True,
            shard=(0, 2),
        )

        # BOTH items must be kept! (Previously, enumerate index 1 % 2 != 0 dropped the second item)
        res_ids = [r["id"] for r in res]
        assert "vid_item_1" in res_ids
        assert "vid_item_3" in res_ids
        assert len(res) == 2





