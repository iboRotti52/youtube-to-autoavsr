import numpy as np
import pytest

from yt2avsr.auto_avsr_crop import _detect_landmarks


class _EmptyOfficialDetector:
    def __call__(self, frames):
        raise AssertionError("Cannot detect any frames in the video")


class _BrokenDetector:
    def __call__(self, frames):
        raise AssertionError("unexpected detector failure")


def test_empty_detector_batch_is_preserved_as_missing_landmarks() -> None:
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)

    assert _detect_landmarks(_EmptyOfficialDetector(), frames) == [None, None, None]


def test_unrelated_detector_assertion_is_not_hidden() -> None:
    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)

    with pytest.raises(AssertionError, match="unexpected detector failure"):
        _detect_landmarks(_BrokenDetector(), frames)
