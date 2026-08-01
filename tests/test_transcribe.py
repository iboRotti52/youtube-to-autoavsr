from yt2avsr.transcribe import (
    transcript_similarity,
    words_confidence,
    words_to_text,
)


def test_words_to_text_normalizes_spacing() -> None:
    words = [{"word": "Merhaba"}, {"word": " dünya"}, {"word": " !"}]
    assert words_to_text(words) == "Merhaba dünya!"


def test_words_confidence_uses_lower_word_or_segment_score() -> None:
    words = [
        {"probability": 0.9, "segment_confidence": 0.8},
        {"probability": 0.6, "segment_confidence": 0.9},
    ]
    assert words_confidence(words) == 0.7
    assert words_confidence([]) == 0.0


def test_transcript_similarity_ignores_case_and_punctuation() -> None:
    assert transcript_similarity("Merhaba, Dünya!", "merhaba dünya") == 1.0
    assert transcript_similarity("Bugün hava güzel", "Tamamen farklı sözler") < 0.60
