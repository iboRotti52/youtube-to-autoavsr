import json
from unittest.mock import patch

from yt2avsr.cloud import push
from yt2avsr.config import AppConfig
from yt2avsr.pipeline import Pipeline


def test_process_both_sources_partitions_before_processed_filter(tmp_path, monkeypatch):
    no_voiceover = [
        "https://www.youtube.com/watch?v=local_shard_a",
        "https://www.youtube.com/watch?v=local_shard_b",
        "https://www.youtube.com/watch?v=local_shard_c",
    ]
    voiceover = [
        "https://www.youtube.com/watch?v=local_shard_d",
        "https://www.youtube.com/watch?v=local_shard_e",
        "https://www.youtube.com/watch?v=local_shard_f",
    ]
    (tmp_path / "sources_no_voiceover.txt").write_text("\n".join(no_voiceover) + "\n")
    (tmp_path / "sources_voiceover.txt").write_text("\n".join(voiceover) + "\n")
    processed_file = tmp_path / "processed_sources.txt"
    processed_file.write_text(no_voiceover[1] + "\n")

    cfg = AppConfig(workspace=tmp_path / "data")
    cfg.sources.processed_file = processed_file
    cfg.sources.auto_sync_hf = False
    cfg.processing.max_workers = 1

    downloaded_urls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("yt2avsr.cli.load_config", lambda _: cfg)
    monkeypatch.setattr(
        "yt2avsr.pipeline.download",
        lambda url, *args, **kwargs: downloaded_urls.append(url) or [],
    )
    monkeypatch.setattr("yt2avsr.pipeline.rebuild", lambda workspace: [])

    from yt2avsr.cli import process_both_sources

    process_both_sources(shard="0/2")

    assert downloaded_urls == [no_voiceover[0], no_voiceover[2], voiceover[1]]


def test_local_zero_clip_completion_is_uploaded_for_hf_sync(tmp_path, monkeypatch):
    cfg = AppConfig(workspace=tmp_path / "data")
    cfg.sources.processed_file = tmp_path / "processed_sources.txt"
    cfg.processing.max_workers = 1
    source_url = "https://www.youtube.com/watch?v=zero_clip_local"
    source_file = tmp_path / "sources.txt"
    source_file.write_text(source_url + "\n")
    item = {
        "id": "zero_clip_local",
        "metadata": {"id": "zero_clip_local", "source_url": source_url},
    }

    monkeypatch.setattr("yt2avsr.pipeline.download", lambda *args, **kwargs: [item])
    monkeypatch.setattr("yt2avsr.pipeline.rebuild", lambda workspace: [])

    pipeline = Pipeline(cfg)
    monkeypatch.setattr(pipeline, "_process_item", lambda processed_item: None)
    pipeline.process_sources_file(source_file)

    ledger_files = list((cfg.workspace / "completions").glob("*.jsonl"))
    assert len(ledger_files) == 1
    ledger_entry = json.loads(ledger_files[0].read_text(encoding="utf-8"))
    assert ledger_entry["status"] == "completed"
    assert ledger_entry["item_id"] == "zero_clip_local"
    assert ledger_entry["item_ids"] == ["zero_clip_local"]
    assert ledger_entry["accepted_clips"] == 0
    assert ledger_entry["review_clips"] == 0
    assert ledger_entry["rejected_clips"] == 0

    (cfg.workspace / "manifests").mkdir(parents=True, exist_ok=True)
    (cfg.workspace / "manifests" / "all.jsonl").write_text("", encoding="utf-8")
    with patch("huggingface_hub.HfApi") as mock_api:
        push(cfg.workspace, "test/repo", contributor="tester", token="fake_token")

    upload_kwargs = mock_api.return_value.upload_folder.call_args.kwargs
    assert "completions/**" in upload_kwargs["allow_patterns"]
