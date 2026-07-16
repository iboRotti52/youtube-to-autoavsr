from pathlib import Path

from yt2avsr.config import load_config


def test_1080p_config_extends_default() -> None:
    cfg = load_config(Path("configs/1080p.yaml"))

    assert cfg.download.format == "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    assert cfg.normalization.max_height == 1080
    assert cfg.language == "tr"
    assert cfg.transcription.model == "large-v3-turbo"
