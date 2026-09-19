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
