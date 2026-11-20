"""D(音域の中心)・E(音域の幅)と PitchTracker のテスト。

pyin は重い(初回は numba のコンパイルも走る)ので、トラックは module スコープで共有する。
"""

from __future__ import annotations

import dataclasses

import librosa
import numpy as np
import pytest
from synth import SR, melody, sine

from musicchar.config_loader import load_config
from musicchar.features import PitchCenterExtractor, PitchRangeExtractor, PitchTracker
from musicchar.models import AudioData, FeatureExtractionError

MELODY = [60, 64, 67, 72]  # C4 E4 G4 C5


@pytest.fixture(scope="module")
def cfg():
    return load_config().audio


@pytest.fixture(scope="module")
def melody_audio() -> AudioData:
    return AudioData(melody(MELODY, 1.0), SR)


@pytest.fixture(scope="module")
def tracker(cfg) -> PitchTracker:
    return PitchTracker(cfg)


# ---------- D ----------

def test_center_of_single_tone(cfg) -> None:
    out = PitchCenterExtractor(cfg).extract(AudioData(sine(440.0, 2.0), SR))  # A4 = MIDI 69
    assert out.mean_midi == pytest.approx(69.0, abs=0.3)
    assert out.median_midi == pytest.approx(69.0, abs=0.3)


def test_center_of_melody(cfg, melody_audio, tracker) -> None:
    out = PitchCenterExtractor(cfg, tracker).extract(melody_audio)
    assert out.mean_midi == pytest.approx(np.mean(MELODY), abs=1.0)
    assert 60 <= out.median_midi <= 72


def test_higher_melody_has_higher_center(cfg) -> None:
    ex = PitchCenterExtractor(cfg)
    low = ex.extract(AudioData(melody([48, 52, 55], 1.0), SR))
    high = ex.extract(AudioData(melody([72, 76, 79], 1.0), SR))
    assert high.mean_midi > low.mean_midi + 20


def test_rests_are_excluded_not_counted_as_zero(cfg) -> None:
    """休符を 0 として平均に混ぜていたら、平均が大きく下がる。"""
    audio = AudioData(melody([69, None, 69, None], 1.0), SR)
    out = PitchCenterExtractor(cfg).extract(audio)
    assert out.mean_midi == pytest.approx(69.0, abs=0.3)
    assert out.median_midi == pytest.approx(69.0, abs=0.3)


# ---------- E ----------

def test_range_of_melody(cfg, melody_audio, tracker) -> None:
    out = PitchRangeExtractor(cfg, tracker).extract(melody_audio)
    assert out.max_midi == pytest.approx(72, abs=0.7)
    assert out.min_midi == pytest.approx(60, abs=0.7)
    assert out.range_semitones == pytest.approx(12, abs=1.2)
    assert out.range_semitones == pytest.approx(out.max_midi - out.min_midi)


def test_range_of_single_tone_is_near_zero(cfg) -> None:
    out = PitchRangeExtractor(cfg).extract(AudioData(sine(440.0, 2.0), SR))
    assert out.range_semitones < 0.5


def test_wider_melody_has_wider_range(cfg) -> None:
    ex = PitchRangeExtractor(cfg)
    narrow = ex.extract(AudioData(melody([60, 62, 64], 1.0), SR))
    wide = ex.extract(AudioData(melody([48, 60, 84], 1.0), SR))
    assert wide.range_semitones > narrow.range_semitones + 20


def test_percentile_zero_gives_true_extremes(cfg, melody_audio, tracker) -> None:
    robust = PitchRangeExtractor(cfg, tracker).extract(melody_audio)
    strict_cfg = dataclasses.replace(cfg, pitch_range_percentile=0.0)
    strict = PitchRangeExtractor(strict_cfg, tracker).extract(melody_audio)
    midi = tracker.track(melody_audio).voiced_midi
    assert strict.max_midi == pytest.approx(midi.max())
    assert strict.min_midi == pytest.approx(midi.min())
    assert strict.range_semitones >= robust.range_semitones


# ---------- 失敗時の扱い ----------

def test_silence_fails_instead_of_returning_zero(cfg) -> None:
    silent = AudioData(np.zeros(SR * 3, dtype=np.float32), SR)
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        PitchCenterExtractor(cfg).extract(silent)
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        PitchRangeExtractor(cfg).extract(silent)


def test_too_short_audio_fails(cfg) -> None:
    with pytest.raises(FeatureExtractionError, match="短すぎ"):
        PitchCenterExtractor(cfg).extract(AudioData(np.ones(100, dtype=np.float32), SR))


def test_min_voiced_ratio_is_enforced(cfg) -> None:
    """有声が全体の 1/20 弱しかない音声は、閾値 0.5 では失敗する。"""
    audio = AudioData(melody([69, None, None, None, None, None, None, None, None, None], 1.0), SR)
    strict_cfg = dataclasses.replace(cfg, pitch_min_voiced_ratio=0.5)
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        PitchCenterExtractor(strict_cfg).extract(audio)
    assert PitchCenterExtractor(cfg).extract(audio).mean_midi == pytest.approx(69.0, abs=0.3)


# ---------- PitchTracker ----------

def test_track_marks_unvoiced_frames_as_nan(cfg) -> None:
    track = PitchTracker(cfg).track(AudioData(melody([69, None], 2.0), SR))
    assert np.isnan(track.midi).any()
    assert track.voiced_midi.size == int(track.voiced_mask.sum())
    assert 0.2 < track.voiced_ratio < 0.8
    assert track.hop_sec == pytest.approx(cfg.hop_length / SR)


def test_tracker_runs_pyin_once_per_audio(cfg, monkeypatch) -> None:
    calls = {"n": 0}
    real = librosa.pyin

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(librosa, "pyin", counting)
    audio = AudioData(sine(440.0, 1.5), SR)
    shared = PitchTracker(cfg)
    PitchCenterExtractor(cfg, shared).extract(audio)
    PitchRangeExtractor(cfg, shared).extract(audio)
    assert calls["n"] == 1
    PitchCenterExtractor(cfg, shared).extract(AudioData(sine(330.0, 1.5), SR))  # 別の音声
    assert calls["n"] == 2


def test_tracker_caches_failures_too(cfg, monkeypatch) -> None:
    calls = {"n": 0}
    real = librosa.pyin

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(librosa, "pyin", counting)
    silent = AudioData(np.zeros(SR * 2, dtype=np.float32), SR)
    shared = PitchTracker(cfg)
    for _ in range(2):
        with pytest.raises(FeatureExtractionError):
            shared.track(silent)
    assert calls["n"] == 1
