import os
from pathlib import Path
from unittest.mock import patch
import pytest
from yt2avsr.config import load_config
from yt2avsr.cloud import check_hf_login_or_warn, _get_token


def test_retina_1080p_config():
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "retina_1080p.yaml"
    assert cfg_path.exists(), "retina_1080p.yaml must exist"

    cfg = load_config(cfg_path)
    assert cfg.auto_avsr.detector == "retinaface"
    assert cfg.normalization.max_height == 1080
    assert "1080" in cfg.download.format


def test_get_token_precedence():
    # 1. Explicit token
    assert _get_token("explicit_token") == "explicit_token"

    # 2. Env token
    with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
        assert _get_token(None) == "env_token"

    # 3. huggingface_hub token
    with patch.dict(os.environ, {}, clear=True):
        with patch("huggingface_hub.get_token", return_value="cached_hub_token"):
            assert _get_token(None) == "cached_hub_token"


def test_check_hf_login_or_warn_missing(capsys):
    with patch.dict(os.environ, {}, clear=True):
        with patch("huggingface_hub.get_token", return_value=None):
            token, user = check_hf_login_or_warn(required=True)
            assert token is None
            assert user is None
            captured = capsys.readouterr()
            assert "HUGGING FACE GİRİŞİ BULUNAMADI" in captured.out
            assert "huggingface-cli login" in captured.out


def test_check_hf_login_or_warn_present():
    with patch("yt2avsr.cloud._get_token", return_value="fake_token"):
        with patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value.whoami.return_value = {"name": "test-user"}
            token, user = check_hf_login_or_warn(required=True)
            assert token == "fake_token"
            assert user == "test-user"


def test_modal_app_compiles():
    import py_compile
    app_path = Path(__file__).resolve().parent.parent / "modal_app.py"
    assert app_path.exists()
    compiled = py_compile.compile(str(app_path))
    assert compiled is not None


def test_modal_app_ast_structure():
    import ast
    app_path = Path(__file__).resolve().parent.parent / "modal_app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "process_single_video_modal" in func_names, "process_single_video_modal worker must be defined"
    assert "process_sources_on_modal" in func_names, "process_sources_on_modal coordinator must be defined"
    assert "main" in func_names, "main local_entrypoint must be defined"

    # Check main argument defaults
    main_func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main")
    arg_names = [arg.arg for arg in main_func.args.args]
    assert "gpu" in arg_names
    assert "cpu" in arg_names
    assert "max_containers" in arg_names

    # Check volume usage
    code = app_path.read_text(encoding="utf-8")
    assert "modal.Volume.from_name" in code
    assert "volume.commit()" in code
    assert "volume.reload()" in code
    assert "uuid.uuid4()" in code
    assert "duplicate item_id" in code


def test_zero_clip_manifest_push(tmp_path):
    from yt2avsr.cloud import push
    workspace = tmp_path / "ws"
    (workspace / "clips").mkdir(parents=True)
    manifests = workspace / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "all.csv").write_text("item_id,quality_status\nvid1,rejected\n", encoding="utf-8")

    with patch("huggingface_hub.HfApi") as mock_api:
        res = push(workspace, repo_id="test/repo", token="fake_token")
        assert "0 clips" in res
        mock_api.return_value.upload_folder.assert_called_once()
        kwargs = mock_api.return_value.upload_folder.call_args.kwargs
        assert "manifests/**" in kwargs["allow_patterns"]
        assert "completions/**" in kwargs["allow_patterns"]
        assert "0 usable clips" in kwargs["commit_message"]


def test_pipeline_preserves_whisper_num_workers():
    from yt2avsr.config import AppConfig, TranscriptionConfig, ProcessingConfig
    from yt2avsr.pipeline import Pipeline

    cfg = AppConfig()
    cfg.transcription.num_workers = 1
    cfg.processing.max_workers = 4

    p = Pipeline(cfg)
    assert p.cfg.transcription.num_workers == 1


def test_shard_stability_over_time():
    """Verify that deduplicate -> partition -> filter preserves stable shard ownership over time."""
    from yt2avsr.sources import (
        deduplicate_source_lines,
        partition_sources,
        get_source_key,
        is_source_processed,
    )

    all_urls = [
        f"https://www.youtube.com/watch?v=vid000000{i:02d}" for i in range(12)
    ]
    voiceover_lines = all_urls[:6]
    no_voiceover_lines = all_urls[6:]

    def get_pending_for_shard(shard_tuple, processed_set):
        dedup_vo, _ = deduplicate_source_lines(voiceover_lines)
        seen_vo_keys = {get_source_key(l) for l in dedup_vo if get_source_key(l)}
        dedup_nvo, _ = deduplicate_source_lines(no_voiceover_lines, seen_keys=seen_vo_keys)

        sources_pairs = []
        for line in dedup_nvo:
            sources_pairs.append(("no_voiceover", line))
        for line in dedup_vo:
            sources_pairs.append(("voiceover", line))

        # Partition full list
        sources_pairs = partition_sources(sources_pairs, shard_tuple)

        # Filter within shard using is_source_processed
        return [p for p in sources_pairs if not is_source_processed(p[1], processed_set)]

    # Initial assignment for shard (0, 3)
    p0_initial = get_pending_for_shard((0, 3), set())
    p1_initial = get_pending_for_shard((1, 3), set())
    p2_initial = get_pending_for_shard((2, 3), set())

    assert len(p0_initial) == 4
    assert len(p1_initial) == 4
    assert len(p2_initial) == 4

    # Now assume 2 videos assigned to shard 0 were processed
    processed = {p0_initial[0][1], p0_initial[1][1]}
    p0_after = get_pending_for_shard((0, 3), processed)
    p1_after = get_pending_for_shard((1, 3), processed)
    p2_after = get_pending_for_shard((2, 3), processed)

    # Shards 1 and 2 must have EXACTLY the same items as before!
    assert p1_after == p1_initial
    assert p2_after == p2_initial
    # Shard 0 has only the remaining 2 items from its original partition
    assert len(p0_after) == 2
    assert all(item in p0_initial for item in p0_after)


def test_aggregation_duplicate_same_source_vs_collision(tmp_path):
    """Verify safe dedupe for same source and fatal RuntimeError for different source collision."""
    import json
    import shutil

    run_root = tmp_path / "run_test"

    # Setup worker 1 with vid_00000001 from URL A
    w1_dir = tmp_path / "w1" / "clips" / "vid_00000001" / "seg_01"
    w1_dir.mkdir(parents=True)
    (w1_dir / "metadata.json").write_text(json.dumps({
        "item_id": "vid_00000001",
        "source_url": "https://youtube.com/watch?v=vid00000001",
    }))

    # Setup worker 2 with vid_00000001 from URL A (same source)
    w2_dir = tmp_path / "w2" / "clips" / "vid_00000001" / "seg_01"
    w2_dir.mkdir(parents=True)
    (w2_dir / "metadata.json").write_text(json.dumps({
        "item_id": "vid_00000001",
        "source_url": "https://youtube.com/watch?v=vid00000001",
    }))

    def aggregate(successful_results, target_clips):
        from yt2avsr.sources import get_source_key
        seen_items = {}
        for res in successful_results:
            src_url = res.get("url", "unknown")
            worker_clips = Path(res["workspace"]) / "clips"
            if worker_clips.exists():
                for item_dir in worker_clips.iterdir():
                    if item_dir.is_dir():
                        item_id = item_dir.name
                        dst_clips = target_clips / item_id
                        first_meta = next(item_dir.glob("*/metadata.json"), None)
                        item_source = src_url
                        if first_meta and first_meta.exists():
                            m_data = json.loads(first_meta.read_text())
                            item_source = m_data.get("source_url") or src_url

                        if item_id in seen_items:
                            existing_source = seen_items[item_id]
                            k_exist = get_source_key(existing_source) or existing_source
                            k_new = get_source_key(item_source) or item_source
                            if k_exist == k_new:
                                continue
                            else:
                                raise RuntimeError(f"Duplicate item collision: {item_id}")
                        seen_items[item_id] = item_source
                        shutil.copytree(item_dir, dst_clips)

    # 1. Same source: succeeds via safe dedupe
    agg_clips_1 = run_root / "agg_1" / "clips"
    agg_clips_1.mkdir(parents=True)
    res_same = [
        {"url": "https://youtube.com/watch?v=vid00000001", "workspace": str(tmp_path / "w1")},
        {"url": "https://youtube.com/watch?v=vid00000001", "workspace": str(tmp_path / "w2")},
    ]
    aggregate(res_same, agg_clips_1)
    assert (agg_clips_1 / "vid_00000001").exists()

    # 2. Collision: different source produces same item_id -> raises RuntimeError
    agg_clips_2 = run_root / "agg_2" / "clips"
    agg_clips_2.mkdir(parents=True)
    w3_dir = tmp_path / "w3" / "clips" / "vid_00000001" / "seg_01"
    w3_dir.mkdir(parents=True)
    (w3_dir / "metadata.json").write_text(json.dumps({
        "item_id": "vid_00000001",
        "source_url": "https://youtube.com/watch?v=COLLISION_12",
    }))

    res_collision = [
        {"url": "https://youtube.com/watch?v=vid00000001", "workspace": str(tmp_path / "w1")},
        {"url": "https://youtube.com/watch?v=COLLISION_12", "workspace": str(tmp_path / "w3")},
    ]
    with pytest.raises(RuntimeError, match="Duplicate item collision"):
        aggregate(res_collision, agg_clips_2)


def test_modal_main_signature_and_cleanup_volume():
    import ast
    app_path = Path(__file__).resolve().parent.parent / "modal_app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    main_func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main")
    arg_names = [arg.arg for arg in main_func.args.args]
    assert "cleanup_volume" in arg_names
    assert "gpu" in arg_names
    assert "cpu" in arg_names
    assert "max_containers" in arg_names

    coord_func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "process_sources_on_modal")
    coord_args = [arg.arg for arg in coord_func.args.args]
    assert "cleanup_volume" in coord_args

    code = app_path.read_text(encoding="utf-8")
    assert "shutil.rmtree(run_root" in code


def test_worker_zero_clip_item_completion(tmp_path):
    """Verify that worker derives item_ids from Pipeline return items even with 0 clips, and coordinator records it."""
    from yt2avsr.sources import load_processed_ids, is_source_processed, append_processed_sources

    # Mock return from Pipeline: video was processed, but generated 0 clips
    pipeline_items = [{"id": "zero_clip_item_999", "metadata": {"id": "zero_clip_item_999"}}]
    # Worker derivation logic:
    item_ids = [str(it["id"]) for it in pipeline_items if isinstance(it, dict) and it.get("id")]
    assert item_ids == ["zero_clip_item_999"]

    # Coordinator processing: write completion ledger and append to processed_sources
    proc_file = tmp_path / "processed_sources.txt"
    successful_results = [{
        "url": "https://youtube.com/watch?v=zero_clip_item_999",
        "video_id": "zero_clip_item_999",
        "item_ids": item_ids,
        "is_playlist": False,
        "accepted_clips": 0,
        "review_clips": 0,
        "rejected_clips": 0,
        "workspace": str(tmp_path / "worker_ws"),
    }]

    newly_processed = []
    for r in successful_results:
        if r.get("url") and not r.get("is_playlist"):
            newly_processed.append(r["url"])
        for i_id in r.get("item_ids", []):
            newly_processed.append(f"video:{i_id}")

    append_processed_sources(newly_processed, proc_file)

    proc_ids = load_processed_ids(proc_file)
    assert is_source_processed("zero_clip_item_999", proc_ids)
    assert is_source_processed("https://youtube.com/watch?v=zero_clip_item_999", proc_ids)


def test_playlist_all_already_processed_noop(tmp_path):
    """Verify that when all playlist items for this shard are already processed, worker marks it as idempotent noop."""
    from yt2avsr.downloader import DownloadResult, download
    from yt2avsr.config import DownloadConfig
    from unittest.mock import patch

    # 1. Downloader returns DownloadResult with all_already_processed=True
    cfg = DownloadConfig()
    fake_info = {
        "_type": "playlist",
        "entries": [
            {"id": "already_proc_1", "playlist_index": 1},
            {"id": "already_proc_2", "playlist_index": 2},
        ]
    }
    with patch("yt2avsr.downloader._extract_with_fallback", return_value=fake_info):
        res = download(
            "https://youtube.com/playlist?list=PL_DONE",
            tmp_path,
            cfg,
            playlist=True,
            processed_ids={"already_proc_1", "already_proc_2"},
            shard=None,
        )
        assert isinstance(res, DownloadResult)
        assert len(res) == 0
        assert res.all_already_processed is True

    # 2. Downloader returns all_already_processed=False when an item in this shard is NOT processed but fails
    with patch("yt2avsr.downloader._extract_with_fallback", return_value=fake_info):
        res_fail = download(
            "https://youtube.com/playlist?list=PL_DONE",
            tmp_path,
            cfg,
            playlist=True,
            processed_ids={"already_proc_1"},  # already_proc_2 is NOT processed, but has no files on disk
            shard=None,
        )
        assert isinstance(res_fail, DownloadResult)
        assert len(res_fail) == 0
        assert res_fail.all_already_processed is False



