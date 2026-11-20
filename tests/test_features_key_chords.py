"""A(調)と G(コード)のテスト。和音は純音を重ねて合成する。"""

from __future__ import annotations

import dataclasses

import librosa
import numpy as np
import pytest
from synth import SR, midi_to_hz, sine

from musicchar.config_loader import load_config
from musicchar.features import ChordExtractor, ChromaProvider, KeyModeExtractor
from musicchar.features.chords import (
    ChordSegment,
    build_templates,
    chord_label,
    compute_chords,
    smooth_path,
)
from musicchar.features.key_mode import KS_MAJOR, KS_MINOR, estimate_key
from musicchar.models import AudioData, FeatureExtractionError


@pytest.fixture(scope="module")
def C():
    return load_config()


def chord(midis: list[int], sec: float = 1.0, amp: float = 0.15) -> np.ndarray:
    return sum(sine(midi_to_hz(m), sec, amp) for m in midis)


def progression(chords: list[list[int]], sec: float = 1.0, reps: int = 4) -> AudioData:
    return AudioData(np.concatenate([chord(c, sec) for c in chords] * reps), SR)


def key_extractor(C, provider: ChromaProvider | None = None) -> KeyModeExtractor:
    p = provider or ChromaProvider(C.audio, C.features.chroma)
    return KeyModeExtractor(C.audio, C.features.key, C.features.chroma, p)


def chord_extractor(C) -> ChordExtractor:
    p = ChromaProvider(C.audio, C.features.chroma)
    return ChordExtractor(C.audio, C.features.chord, C.features.chroma, p)


C_AM_F_G = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]]
AM_DM_E_AM = [[57, 60, 64], [50, 53, 57], [52, 56, 59], [57, 60, 64]]
SEVENTHS = [[60, 64, 67, 71], [57, 60, 64, 67], [50, 53, 57, 60], [55, 59, 62, 65]]


def white_noise(sec: int = 8) -> AudioData:
    rng = np.random.default_rng(0)
    return AudioData((rng.standard_normal(SR * sec) * 0.1).astype(np.float32), SR)


# ---------- A 調 ----------

def test_estimate_key_from_ideal_profiles() -> None:
    key, mode, r, _ = estimate_key(np.roll(KS_MAJOR, 2))  # D major
    assert (key, mode) == ("D", "major") and r == pytest.approx(1.0)
    key, mode, r, _ = estimate_key(np.roll(KS_MINOR, 9))  # A minor
    assert (key, mode) == ("A", "minor") and r == pytest.approx(1.0)


def test_estimate_key_rejects_flat_profile() -> None:
    with pytest.raises(FeatureExtractionError, match="偏り"):
        estimate_key(np.ones(12))


def test_key_c_major(C) -> None:
    k = key_extractor(C).extract(progression(C_AM_F_G))
    assert (k.key, k.mode) == ("C", "major")
    assert k.confidence > 0.8


def test_key_a_minor(C) -> None:
    k = key_extractor(C).extract(progression(AM_DM_E_AM))
    assert (k.key, k.mode) == ("A", "minor")


def test_key_transposed_progression(C) -> None:
    up2 = [[n + 2 for n in c] for c in C_AM_F_G]  # D major
    k = key_extractor(C).extract(progression(up2))
    assert (k.key, k.mode) == ("D", "major")


def test_key_silence_fails(C) -> None:
    with pytest.raises(FeatureExtractionError):
        key_extractor(C).extract(AudioData(np.zeros(SR * 3, dtype=np.float32), SR))


def test_key_white_noise_is_not_assigned_a_key(C) -> None:
    with pytest.raises(FeatureExtractionError, match="確信度"):
        key_extractor(C).extract(white_noise())


def test_key_min_correlation_is_configurable(C) -> None:
    strict = dataclasses.replace(C.features.key, min_correlation=0.99)
    ex = KeyModeExtractor(C.audio, strict, C.features.chroma)
    with pytest.raises(FeatureExtractionError, match="確信度"):
        ex.extract(progression(C_AM_F_G))


# ---------- G コード: 補助関数 ----------

def test_templates_are_unit_norm_and_complete() -> None:
    t, labels = build_templates()
    assert t.shape == (12 * 9, 12) and len(labels) == 12 * 9
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0)


def test_chord_label() -> None:
    assert chord_label(0, "maj") == "C"
    assert chord_label(9, "min") == "Am"
    assert chord_label(7, "7") == "G7"
    assert chord_label(1, "min7") == "C#m7"


def test_smooth_path_suppresses_brief_blips() -> None:
    scores = np.array([[1.0, 1.0, 0.9, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]])
    assert list(smooth_path(scores, 0.0)) == [0, 0, 1, 0, 0]  # ペナルティなしでは追従
    assert list(smooth_path(scores, 0.5)) == [0, 0, 0, 0, 0]  # あれば無視
    sustained = np.array([[1.0] * 3 + [0.0] * 3, [0.0] * 3 + [1.0] * 3])
    assert list(smooth_path(sustained, 0.5)) == [0, 0, 0, 1, 1, 1]  # 本当の切り替わりは残る


def test_compute_chords_by_hand(C) -> None:
    cfg = C.features.chord
    segs = [ChordSegment(0, 4, 0, "maj"), ChordSegment(4, 6, 9, "min"), ChordSegment(6, 10, 7, "7")]
    out = compute_chords(segs, n_windows=10, duration_sec=5.0, cfg=cfg)
    assert out.progression == ("C", "Am", "G7")
    assert out.chord_types == {"maj": 1, "min": 1, "7": 1}
    assert out.change_rate_per_sec == pytest.approx(2 / 5.0)
    assert out.minor_ratio == pytest.approx(2 / 10)
    ext, tv, cv = 4 / 10, (3 - 1) / (cfg.type_ref - 1), (3 - 1) / (cfg.chord_ref - 1)
    w = (cfg.weight_extension, cfg.weight_type_variety, cfg.weight_chord_variety)
    assert out.complexity == pytest.approx((w[0] * ext + w[1] * tv + w[2] * cv) / sum(w))


def test_compute_chords_fails_when_too_few_chord_windows(C) -> None:
    cfg = C.features.chord
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        compute_chords([ChordSegment(0, 1, 0, "maj")], n_windows=20, duration_sec=5.0, cfg=cfg)
    with pytest.raises(FeatureExtractionError):
        compute_chords([], n_windows=10, duration_sec=5.0, cfg=cfg)


# ---------- G コード: 音声 ----------

def test_chords_triad_progression(C) -> None:
    out = chord_extractor(C).extract(progression(C_AM_F_G))
    assert out.progression[:8] == ("C", "Am", "F", "G") * 2
    assert set(out.chord_types) == {"maj", "min"}
    assert out.change_rate_per_sec == pytest.approx(15 / 16, abs=0.15)
    assert out.minor_ratio == pytest.approx(0.25, abs=0.05)
    assert out.complexity < 0.3


def test_chords_minor_progression_has_high_minor_ratio(C) -> None:
    out = chord_extractor(C).extract(progression(AM_DM_E_AM))
    assert out.minor_ratio > 0.6
    assert "E" in out.progression


def test_chords_seventh_progression_is_more_complex(C) -> None:
    ex = chord_extractor(C)
    simple = ex.extract(progression(C_AM_F_G))
    rich = ex.extract(progression(SEVENTHS))
    assert rich.progression[:4] == ("Cmaj7", "Am7", "Dm7", "G7")
    assert {"maj7", "min7", "7"} <= set(rich.chord_types)
    assert rich.complexity > simple.complexity + 0.4


def test_chords_faster_changes_give_higher_rate(C) -> None:
    ex = chord_extractor(C)
    slow = ex.extract(progression(C_AM_F_G, sec=2.0, reps=2))
    fast = ex.extract(progression(C_AM_F_G, sec=0.5, reps=8))
    assert fast.change_rate_per_sec > 2 * slow.change_rate_per_sec


def test_chords_single_sustained_chord(C) -> None:
    out = chord_extractor(C).extract(AudioData(chord([60, 64, 67], 5.0), SR))
    assert out.progression == ("C",)
    assert out.change_rate_per_sec == 0.0


def test_chords_silence_between_chords_is_not_a_chord(C) -> None:
    silence = np.zeros(SR * 3, dtype=np.float32)
    y = np.concatenate([chord([60, 64, 67], 3.0), silence, chord([60, 64, 67], 3.0)])
    out = chord_extractor(C).extract(AudioData(y, SR))
    assert out.progression == ("C", "C")  # 無音を挟んだ 2 区間


def test_chords_silence_fails(C) -> None:
    with pytest.raises(FeatureExtractionError):
        chord_extractor(C).extract(AudioData(np.zeros(SR * 3, dtype=np.float32), SR))


def test_chords_noise_fails_instead_of_inventing_chords(C) -> None:
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        chord_extractor(C).extract(white_noise())


def test_chords_too_short_fails(C) -> None:
    with pytest.raises(FeatureExtractionError):
        chord_extractor(C).extract(AudioData(chord([60, 64, 67], 0.3), SR))


def test_chroma_is_computed_once_for_key_and_chords(C, monkeypatch) -> None:
    calls = {"n": 0}
    real = librosa.feature.chroma_cqt

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(librosa.feature, "chroma_cqt", counting)
    audio = progression(C_AM_F_G, reps=1)
    provider = ChromaProvider(C.audio, C.features.chroma)
    key_extractor(C, provider).extract(audio)
    ChordExtractor(C.audio, C.features.chord, C.features.chroma, provider).extract(audio)
    assert calls["n"] == 1
