from unittest.mock import patch

from yt2avsr.config import DownloadConfig
from yt2avsr.downloader import download


def _download_text(url, output):
    output.write_text(f"subtitle from {url}", encoding="utf-8")


def test_download_uses_manual_subtitle_tuple_contract(tmp_path):
    cfg = DownloadConfig(
        subtitle_languages=["tr"],
        use_youtube_subtitles=True,
        use_automatic_youtube_captions=True,
    )
    item_dir = tmp_path / "manual_video"
    item_dir.mkdir()
    (item_dir / "video.mp4").write_bytes(b"video")
    fake_info = {
        "id": "manual_video",
        "subtitles": {"tr": [{"url": "https://example.test/manual.srt", "ext": "srt"}]},
        "automatic_captions": {},
    }

    with patch("yt2avsr.downloader._extract_with_fallback", return_value=fake_info), \
         patch("yt2avsr.downloader.download_text", side_effect=_download_text):
        result = download("https://example.test/manual_video", tmp_path, cfg)

    assert result[0]["metadata"]["subtitle_source"] == "manual"
    assert result[0]["metadata"]["subtitle_language"] == "tr"
    assert result[0]["subtitle_path"] == item_dir / "subtitles.srt"
    assert result[0]["subtitle_path"].read_text(encoding="utf-8") == (
        "subtitle from https://example.test/manual.srt"
    )


def test_download_uses_automatic_subtitle_tuple_contract(tmp_path):
    cfg = DownloadConfig(
        subtitle_languages=["tr"],
        use_youtube_subtitles=True,
        use_automatic_youtube_captions=True,
    )
    item_dir = tmp_path / "automatic_video"
    item_dir.mkdir()
    (item_dir / "video.mp4").write_bytes(b"video")
    fake_info = {
        "id": "automatic_video",
        "subtitles": {},
        "automatic_captions": {
            "tr": [{"url": "https://example.test/automatic.vtt", "ext": "vtt"}]
        },
    }

    with patch("yt2avsr.downloader._extract_with_fallback", return_value=fake_info), \
         patch("yt2avsr.downloader.download_text", side_effect=_download_text):
        result = download("https://example.test/automatic_video", tmp_path, cfg)

    assert result[0]["metadata"]["subtitle_source"] == "automatic"
    assert result[0]["metadata"]["subtitle_language"] == "tr"
    assert result[0]["subtitle_automatic"] is True
    assert result[0]["subtitle_path"] == item_dir / "subtitles.vtt"
    assert result[0]["subtitle_path"].read_text(encoding="utf-8") == (
        "subtitle from https://example.test/automatic.vtt"
    )
