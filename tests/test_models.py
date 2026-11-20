"""models のテスト。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from musicchar.models import (
    AudioData,
    CharacterParams,
    Chords,
    FeatureSet,
    ImpressionSet,
    KeyMode,
    Melody,
    PartParam,
    PitchRange,
    Tempo,
)


def test_audio_data_duration() -> None:
    a = AudioData(y=np.zeros(22050), sr=22050)
    assert a.duration == pytest.approx(1.0)


@pytest.mark.parametrize(
    "y, sr",
    [(np.zeros((2, 10)), 22050), (np.zeros(10), 0), (np.array([0.0, np.nan]), 22050)],
)
def test_audio_data_rejects_invalid(y: np.ndarray, sr: int) -> None:
    with pytest.raises(ValueError):
        AudioData(y=y, sr=sr)


def test_featureset_defaults_to_none_not_zero() -> None:
    fs = FeatureSet()
    for f in dataclasses.fields(fs):
        if f.name != "errors":
            assert getattr(fs, f.name) is None
    assert fs.errors == {}


def test_featureset_errors_not_shared_between_instances() -> None:
    a, b = FeatureSet(), FeatureSet()
    a.errors["x"] = "y"
    assert b.errors == {}


def test_key_mode_major_score() -> None:
    assert KeyMode("C", "major", 0.5).major_score == 1.0
    assert KeyMode("A", "minor", 0.5).major_score == 0.0
    assert KeyMode(None, None, 0.0).major_score is None


def test_key_mode_validation() -> None:
    with pytest.raises(ValueError):
        KeyMode("H", "major")
    with pytest.raises(ValueError):
        KeyMode("C", "dorian")
    with pytest.raises(ValueError):
        KeyMode("C", "major", confidence=1.5)


def test_feature_validation() -> None:
    with pytest.raises(ValueError):
        Tempo(bpm=0)
    with pytest.raises(ValueError):
        PitchRange(max_midi=50, min_midi=60, range_semitones=0)
    with pytest.raises(ValueError):
        Melody(1, 1, leap_ratio=1.2, repetition_score=0, smoothness=0)
    with pytest.raises(ValueError):
        Chords({}, (), change_rate_per_sec=-1, complexity=0, minor_ratio=0)


def test_feature_models_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Tempo(bpm=100).bpm = 90  # type: ignore[misc]


def test_impression_set_validation() -> None:
    ok = {i: 0.5 for i in range(1, 9)}
    assert ImpressionSet(values=ok).get(3) == 0.5
    with pytest.raises(ValueError):
        ImpressionSet(values={**ok, 1: 1.2})
    with pytest.raises(ValueError):
        ImpressionSet(values={i: 0.5 for i in range(1, 8)})


def test_impression_set_allows_none() -> None:
    values = {i: 0.5 for i in range(1, 9)}
    values[2] = None
    assert ImpressionSet(values=values).get(2) is None


def test_part_param_and_character_params() -> None:
    p = PartParam(name="eye_angle", value=3.0, t=0.6, sources=(1,))
    assert CharacterParams(parts={"eye_angle": p}).value("eye_angle") == 3.0
    with pytest.raises(ValueError):
        PartParam(name="x", value=0, t=1.1, sources=())
