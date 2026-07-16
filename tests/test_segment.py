from yt2avsr.config import SegmentationConfig
from yt2avsr.segment import make_segments, remove_adjacent_repeated_sentences
from yt2avsr.subtitles import deduplicate


def test_make_segments_splits_long_sequence() -> None:
    words = [
        {"word": f"kelime{i}", "start": i * 0.5, "end": i * 0.5 + 0.4, "probability": 0.9}
        for i in range(40)
    ]
    cfg = SegmentationConfig(min_duration=1.0, max_duration=5.0, min_words=2)
    result = make_segments(words, cfg)
    assert len(result) >= 3
    assert all(s["duration"] <= 5.3 for s in result)


def _uniform_words(n: int) -> list[dict]:
    return [
        {"word": f"k{i}", "start": i * 0.5, "end": i * 0.5 + 0.4, "probability": 0.9}
        for i in range(n)
    ]


def test_scene_cut_splits_segment() -> None:
    words = _uniform_words(20)
    cfg = SegmentationConfig(
        min_duration=1.0, max_duration=30.0, min_words=2, split_on_scene_cut=True
    )
    cut = 5.0
    result = make_segments(words, cfg, cut_times=[cut])
    assert len(result) >= 2
    # Hiçbir klip (padding dahil) bir sahne kesmesini kapsamamalı.
    for seg in result:
        assert not (seg["start"] < cut < seg["end"])


def test_scene_cut_disabled_keeps_single_segment() -> None:
    words = _uniform_words(20)
    cfg = SegmentationConfig(
        min_duration=1.0, max_duration=30.0, min_words=2, split_on_scene_cut=False
    )
    result = make_segments(words, cfg, cut_times=[5.0])
    assert len(result) == 1


def test_subtitle_timed_words_get_extra_end_padding() -> None:
    words = [
        {
            "word": token,
            "start": 4.06 + idx * (2.68 / 7),
            "end": 4.06 + (idx + 1) * (2.68 / 7),
            "probability": 1.0,
            "timing_source": "subtitle",
        }
        for idx, token in enumerate(
            ["Herkese", "merhaba.", "Güncel", "Türkçe'ye", "tekrar", "hoş", "geldin."]
        )
    ]
    cfg = SegmentationConfig(min_duration=2.0, max_duration=16.0, min_words=2)
    result = make_segments(words, cfg)
    assert result[0]["text"] == "Herkese merhaba. Güncel Türkçe'ye tekrar hoş geldin."
    assert result[0]["end"] == 7.34


def _cues(texts: list[str]) -> list[dict]:
    return [{"start": i, "end": i + 1, "text": t} for i, t in enumerate(texts)]


def test_deduplicate_removes_exact_and_rolling_repeats() -> None:
    cues = _cues(
        [
            "Bağımlılık normaldir dersek",
            "Bağımlılık normaldir dersek",  # tam tekrar
            "Bağımlılık normaldir dersek başımız derde",  # rolling
            "başımız derde girer mi?",  # örtüşen kuyruk
        ]
    )
    joined = " ".join(c["text"] for c in deduplicate(cues))
    assert joined.count("Bağımlılık normaldir dersek") == 1
    assert joined.count("başımız derde") == 1


def test_remove_adjacent_repeated_sentences() -> None:
    text = "Başımız derde girer mi? Başımız derde girer mi? Delirmek normaldir."
    assert remove_adjacent_repeated_sentences(text) == (
        "Başımız derde girer mi? Delirmek normaldir."
    )


def test_strict_sentence_boundaries_do_not_merge_mixed_turns() -> None:
    words = [
        {"word": "Alper", "start": 0.0, "end": 0.3, "probability": 0.9},
        {"word": "merhaba.", "start": 0.3, "end": 0.7, "probability": 0.9},
        {"word": "Merhaba", "start": 0.8, "end": 1.1, "probability": 0.9},
        {"word": "İlker.", "start": 1.1, "end": 1.5, "probability": 0.9},
        {"word": "Bugünkü", "start": 1.6, "end": 2.0, "probability": 0.9},
        {"word": "konumuz", "start": 2.0, "end": 2.4, "probability": 0.9},
        {"word": "bağımlılık", "start": 2.4, "end": 3.0, "probability": 0.9},
        {"word": "normal", "start": 3.0, "end": 3.4, "probability": 0.9},
        {"word": "midir?", "start": 3.4, "end": 3.8, "probability": 0.9},
    ]
    cfg = SegmentationConfig(min_duration=2.0, min_words=2)
    loose = make_segments(words, cfg)
    strict = make_segments(words, cfg, strict_sentence_boundaries=True)
    assert len(loose) == 1
    assert [row["text"] for row in strict] == ["Bugünkü konumuz bağımlılık normal midir?"]
