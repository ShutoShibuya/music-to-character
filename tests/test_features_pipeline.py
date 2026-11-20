"""extract_all(特徴量抽出の統括)のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
from synth import SR, click_track, melody

from musicchar.config_loader import load_config
from musicchar.features import FeatureExtractor, extract_all
from musicchar.models import AudioData, FeatureExtractionError, Tempo


@pytest.fixture(scope="module")
def audio_cfg():
    return load_config().audio


@pytest.fixture(scope="module")
def feat_cfg():
    return load_config().features


class _Failing(FeatureExtractor[Tempo]):
    name = "b_tempo"

    def extract(self, audio: AudioData) -> Tempo:
        raise FeatureExtractionError("わざと失敗")


class _Crashing(FeatureExtractor[Tempo]):
    name = "b_tempo"

    def extract(self, audio: AudioData) -> Tempo:
        raise ZeroDivisionError("boom")


def test_extract_all_populates_implemented_features(audio_cfg, feat_cfg) -> None:
    notes = [60, 62, 64, 65, 67, 65, 64, 62] * 3  # 0.5 秒刻み = 120 BPM の旋律
    fs = extract_all(AudioData(melody(notes, 0.5), SR), audio_cfg, feat_cfg)
    for name in ("b_tempo", "c_melody", "d_pitch_center", "e_pitch_range", "f_dynamics"):
        assert getattr(fs, name) is not None, fs.errors
        assert name not in fs.errors


def test_features_without_extractor_are_none_with_reason(audio_cfg, feat_cfg) -> None:
    from musicchar.features import TempoExtractor

    fs = extract_all(AudioData(click_track(120), SR), audio_cfg, feat_cfg, [TempoExtractor(audio_cfg)])
    assert fs.b_tempo is not None
    for name in ("a_key", "c_melody", "d_pitch_center", "e_pitch_range", "f_dynamics", "g_chords", "h_articulation"):
        assert getattr(fs, name) is None
        assert "登録されていません" in fs.errors[name]


def test_default_extractors_cover_all_features(audio_cfg, feat_cfg) -> None:
    from musicchar.features import default_extractors
    from musicchar.models import FeatureSet
    import dataclasses

    expected = {f.name for f in dataclasses.fields(FeatureSet)} - {"errors"}
    assert {e.name for e in default_extractors(audio_cfg, feat_cfg)} == expected


def test_failure_becomes_none_and_is_recorded_without_stopping_others(audio_cfg, feat_cfg) -> None:
    from musicchar.features import DynamicsExtractor

    fs = extract_all(
        AudioData(click_track(120), SR),
        audio_cfg,
        feat_cfg,
        [_Failing(audio_cfg), DynamicsExtractor(audio_cfg)],
    )
    assert fs.b_tempo is None
    assert fs.errors["b_tempo"] == "わざと失敗"
    assert fs.f_dynamics is not None


def test_unexpected_exception_is_contained(audio_cfg, feat_cfg) -> None:
    fs = extract_all(AudioData(click_track(120), SR), audio_cfg, feat_cfg, [_Crashing(audio_cfg)])
    assert fs.b_tempo is None
    assert "ZeroDivisionError" in fs.errors["b_tempo"]


def test_silence_gives_none_tempo_but_measured_dynamics(audio_cfg, feat_cfg) -> None:
    fs = extract_all(AudioData(np.zeros(SR * 5, dtype=np.float32), SR), audio_cfg, feat_cfg)
    assert fs.b_tempo is None
    assert "b_tempo" in fs.errors
    assert fs.f_dynamics is not None and fs.f_dynamics.rms_mean == 0.0


def test_rejects_duplicate_or_unknown_extractors(audio_cfg, feat_cfg) -> None:
    audio = AudioData(click_track(120), SR)
    with pytest.raises(ValueError, match="重複"):
        extract_all(audio, audio_cfg, feat_cfg, [_Failing(audio_cfg), _Crashing(audio_cfg)])

    class _Bogus(_Failing):
        name = "z_nothing"

    with pytest.raises(ValueError, match="存在しない"):
        extract_all(audio, audio_cfg, feat_cfg, [_Bogus(audio_cfg)])
