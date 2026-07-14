from yt2avsr.config import SegmentationConfig
from yt2avsr.segment import make_segments


def test_make_segments_splits_long_sequence() -> None:
    words = [
        {"word": f"kelime{i}", "start": i * 0.5, "end": i * 0.5 + 0.4, "probability": 0.9}
        for i in range(40)
    ]
    cfg = SegmentationConfig(min_duration=1.0, max_duration=5.0, min_words=2)
    result = make_segments(words, cfg)
    assert len(result) >= 3
    assert all(s["duration"] <= 5.3 for s in result)
