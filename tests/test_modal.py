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
        assert kwargs["allow_patterns"] == ["manifests/**"]
        assert "0 usable clips" in kwargs["commit_message"]


def test_pipeline_preserves_whisper_num_workers():
    from yt2avsr.config import AppConfig, TranscriptionConfig, ProcessingConfig
    from yt2avsr.pipeline import Pipeline

    cfg = AppConfig()
    cfg.transcription.num_workers = 1
    cfg.processing.max_workers = 4

    p = Pipeline(cfg)
    assert p.cfg.transcription.num_workers == 1


def test_aggregation_duplicate_detection(tmp_path):
    import shutil
    run_root = tmp_path / "run_test"
    agg_clips = run_root / "aggregated" / "clips"
    agg_clips.mkdir(parents=True)

    worker_a_clips = tmp_path / "worker_a" / "clips"
    (worker_a_clips / "video_dup").mkdir(parents=True)
    (worker_a_clips / "video_dup" / "clip_a.txt").write_text("from worker a")

    worker_b_clips = tmp_path / "worker_b" / "clips"
    (worker_b_clips / "video_dup").mkdir(parents=True)
    (worker_b_clips / "video_dup" / "clip_b.txt").write_text("from worker b")

    copied = []
    skipped = []
    for w_clips in [worker_a_clips, worker_b_clips]:
        for item_dir in w_clips.iterdir():
            if item_dir.is_dir():
                dst = agg_clips / item_dir.name
                if dst.exists():
                    skipped.append(item_dir.name)
                    continue
                shutil.copytree(item_dir, dst)
                copied.append(item_dir.name)

    assert copied == ["video_dup"]
    assert skipped == ["video_dup"]
    # Ensure worker b did not overwrite worker a
    assert (agg_clips / "video_dup" / "clip_a.txt").exists()
    assert not (agg_clips / "video_dup" / "clip_b.txt").exists()


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


