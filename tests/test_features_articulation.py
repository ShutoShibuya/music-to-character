"""H(アーティキュレーション)のテスト。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from synth import SR, click_track, melody, midi_to_hz, sine

from musicchar.config_loader import load_config
from musicchar.features import ArticulationExtractor
from musicchar.features.articulation import analyze_articulation, drop_decay_onsets
from musicchar.models import AudioData, FeatureExtractionError

NOTES = [midi_to_hz(m) for m in [60, 64, 67, 72] * 6]


@pytest.fixture(scope="module")
def C():
    return load_config()


@pytest.fixture(scope="module")
def ex(C) -> ArticulationExtractor:
    return ArticulationExtractor(C.audio, C.features.articulation)


def notes(
    freqs: list[float], spacing: float, length: float | None = None, tau: float | None = None,
    attack: float = 0.005, amp: float = 0.5, release: float = 0.01,
) -> np.ndarray:
    """spacing 秒ごとに鳴る音。length で切る(release 秒のフェードアウト付き)、tau で指数減衰。"""
    y = np.zeros(int(spacing * len(freqs) * SR), dtype=np.float32)
    for i, f in enumerate(freqs):
        n = int((length or spacing) * SR)
        t = np.arange(n) / SR
        env = np.exp(-t / tau) if tau else np.ones(n)
        env = env * np.minimum(1.0, t / attack)
        r = int(release * SR)
        env[-r:] *= np.linspace(1.0, 0.0, r)
        s = (amp * np.sin(2 * np.pi * f * t) * env).astype(np.float32)
        a = int(i * spacing * SR)
        y[a : a + n] += s[: len(y) - a]
    # 実際の録音には雑音の床がある。完全な無音のままだと、音が切れる瞬間のごく小さな
    # スペクトルの広がりが対数スペクトル上で目立ち、立ち上がりと誤検出されるため加える(約 -66 dBFS)。
    y += (np.random.default_rng(0).standard_normal(len(y)) * 0.0005).astype(np.float32)
    return y


# ---------- analyze_articulation (手計算) ----------

def test_analyze_articulation_by_hand(C) -> None:
    hop = 0.02
    rms = np.zeros(80)
    rms[0:5] = 1.0       # 音0: 5フレームで鳴り止む(以降は 0.01)
    rms[5:20] = 0.01
    rms[20:40] = 1.0     # 音1: 次の発音まで鳴り続ける
    rms[40:50] = 1.0     # 音2: 10フレーム後に 0.05(-26dB)へ減衰
    rms[50:60] = 0.05
    rms[60:62] = 1.0     # 音3: 2フレームで無音
    onset_env = np.full(80, 5.0)
    onsets = np.array([0, 20, 40, 60])
    out = analyze_articulation(rms, onset_env, onsets, hop, C.features.articulation)

    assert out.onset_strength_mean == 5.0
    assert out.onset_density == pytest.approx(4 / 1.6)
    assert out.mean_note_duration == pytest.approx((0.10 + 0.40 + 0.20 + 0.04) / 4)
    assert out.overlap_ratio == pytest.approx(1 / 3)  # 音1 だけが次の発音まで残る
    assert out.gap_ratio == pytest.approx((15 + 18) / 80)  # 0.01 が15フレーム、0 が18フレーム


def test_analyze_articulation_needs_enough_onsets(C) -> None:
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        analyze_articulation(np.ones(50), np.ones(50), np.array([0, 10]), 0.02, C.features.articulation)


def test_analyze_articulation_silent_rms_fails(C) -> None:
    with pytest.raises(FeatureExtractionError, match="小さすぎ"):
        analyze_articulation(np.zeros(80), np.ones(80), np.array([0, 20, 40, 60]), 0.02, C.features.articulation)


def test_drop_decay_onsets() -> None:
    rms = np.array([0, 0, 0, 0.5, 1, 1, 1, 0.9, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    #                          アタック=3            減衰=8           横ばい(レガート)=12
    kept = drop_decay_onsets(np.array([3, 8, 12]), rms, context_frames=2, decay_ratio=0.7)
    assert list(kept) == [3, 12]


def test_drop_decay_onsets_edges_and_empty() -> None:
    rms = np.array([1.0, 0.5, 0.2])
    assert list(drop_decay_onsets(np.array([0, 2]), rms, 2, 0.7)) == [0]  # 端でも添字が壊れない
    assert drop_decay_onsets(np.array([], dtype=int), rms, 2, 0.7).size == 0


def test_drop_decay_onsets_keeps_peak_onset() -> None:
    """クリックのように、発音点のフレームが RMS のピーク上にある場合は残す。"""
    rms = np.array([0.0, 0.17, 0.2, 0.21, 0.12, 0.05])
    assert list(drop_decay_onsets(np.array([3]), rms, 2, 0.7)) == [3]


# ---------- 音声 ----------

def test_staccato_has_short_notes_and_many_gaps(ex) -> None:
    a = ex.extract(AudioData(notes(NOTES, 0.5, length=0.1), SR))
    assert a.onset_density == pytest.approx(2.0, abs=0.4)  # 音の終わりを発音と数えない
    assert a.mean_note_duration < 0.25
    assert a.gap_ratio > 0.4
    assert a.overlap_ratio < 0.2


def test_legato_has_long_notes_overlap_and_no_gaps(ex) -> None:
    a = ex.extract(AudioData(melody([60, 64, 67, 72] * 6, 0.5), SR))
    assert a.mean_note_duration == pytest.approx(0.5, abs=0.1)
    assert a.overlap_ratio > 0.9
    assert a.gap_ratio < 0.05


def test_staccato_vs_legato_ordering(ex) -> None:
    stac = ex.extract(AudioData(notes(NOTES, 0.5, length=0.1), SR))
    leg = ex.extract(AudioData(melody([60, 64, 67, 72] * 6, 0.5), SR))
    assert leg.mean_note_duration > 2 * stac.mean_note_duration
    assert stac.gap_ratio > leg.gap_ratio + 0.4
    assert leg.overlap_ratio > stac.overlap_ratio + 0.5


@pytest.mark.parametrize("release", [0.01, 0.03, 0.08])
def test_staccato_density_is_stable_against_release_time(ex, release: float) -> None:
    a = ex.extract(AudioData(notes(NOTES, 0.5, length=0.15, release=release), SR))
    assert a.onset_density == pytest.approx(2.0, abs=0.4)
    assert a.overlap_ratio < 0.2


def test_decay_speed_controls_overlap(ex) -> None:
    fast = ex.extract(AudioData(notes(NOTES, 0.5, tau=0.05), SR))
    slow = ex.extract(AudioData(notes(NOTES, 0.5, tau=1.0, length=1.5), SR))
    assert fast.overlap_ratio < 0.2 < 0.8 < slow.overlap_ratio
    assert fast.gap_ratio > slow.gap_ratio + 0.3
    assert fast.mean_note_duration < slow.mean_note_duration


def test_onset_density_follows_note_rate(ex) -> None:
    slow = ex.extract(AudioData(melody([60, 64, 67, 72] * 6, 0.5), SR))
    fast = ex.extract(AudioData(melody([60, 64, 67, 72] * 12, 0.25), SR))
    assert slow.onset_density == pytest.approx(2.0, abs=0.4)
    assert fast.onset_density == pytest.approx(4.0, abs=0.6)


def test_sharp_attack_has_larger_onset_strength(ex) -> None:
    sharp = ex.extract(AudioData(notes(NOTES, 0.5, attack=0.003, tau=0.3), SR))
    soft = ex.extract(AudioData(notes(NOTES, 0.5, attack=0.2, tau=0.3), SR))
    assert sharp.onset_strength_mean > soft.onset_strength_mean


def test_click_track(ex) -> None:
    a = ex.extract(AudioData(click_track(120), SR))
    assert a.onset_density == pytest.approx(2.0, abs=0.3)
    assert a.mean_note_duration < 0.15
    assert a.gap_ratio > 0.5


def test_steady_tone_has_no_articulation(ex) -> None:
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        ex.extract(AudioData(sine(440, 8.0), SR))


def test_silence_fails(ex) -> None:
    with pytest.raises(FeatureExtractionError):
        ex.extract(AudioData(np.zeros(SR * 5, dtype=np.float32), SR))


def test_too_short_fails(ex) -> None:
    with pytest.raises(FeatureExtractionError, match="短すぎ"):
        ex.extract(AudioData(np.ones(100, dtype=np.float32), SR))


def test_thresholds_are_configurable(C) -> None:
    audio = AudioData(notes(NOTES, 0.5, tau=0.3), SR)
    loose = ArticulationExtractor(
        C.audio, dataclasses.replace(C.features.articulation, decay_threshold_db=-6.0)
    ).extract(audio)
    strict = ArticulationExtractor(
        C.audio, dataclasses.replace(C.features.articulation, decay_threshold_db=-40.0)
    ).extract(audio)
    assert loose.mean_note_duration < strict.mean_note_duration  # -6dB の方が早く「減衰した」と判定される
