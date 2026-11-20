"""B(テンポ)と F(音量)の抽出テスト。"""

from __future__ import annotations

import numpy as np
import pytest
from synth import SR, click_track, sine

from musicchar.config_loader import load_config
from musicchar.features import DynamicsExtractor, TempoExtractor
from musicchar.models import AudioData, FeatureExtractionError


@pytest.fixture(scope="module")
def audio_cfg():
    return load_config().audio


# ---------- B テンポ ----------

@pytest.mark.parametrize("bpm", [60, 90, 120, 140, 174])
def test_tempo_of_click_track(audio_cfg, bpm: int) -> None:
    out = TempoExtractor(audio_cfg).extract(AudioData(click_track(bpm), SR))
    assert out.bpm == pytest.approx(bpm, abs=4)


def test_tempo_faster_track_gives_larger_bpm(audio_cfg) -> None:
    ex = TempoExtractor(audio_cfg)
    slow = ex.extract(AudioData(click_track(80), SR)).bpm
    fast = ex.extract(AudioData(click_track(150), SR)).bpm
    assert fast > slow + 40


def test_tempo_fails_on_silence_instead_of_returning_a_prior(audio_cfg) -> None:
    """librosa は無音にも 117 BPM 前後を返すので、値を捏造せず失敗させる。"""
    silent = AudioData(np.zeros(SR * 10, dtype=np.float32), SR)
    with pytest.raises(FeatureExtractionError, match="手がかり"):
        TempoExtractor(audio_cfg).extract(silent)


def test_tempo_fails_on_empty_audio(audio_cfg) -> None:
    with pytest.raises(FeatureExtractionError):
        TempoExtractor(audio_cfg).extract(AudioData(np.zeros(0, dtype=np.float32), SR))


# ---------- F 音量 ----------

def test_dynamics_constant_sine(audio_cfg) -> None:
    out = DynamicsExtractor(audio_cfg).extract(AudioData(sine(440, 3.0, amp=0.5), SR))
    expected = 0.5 / np.sqrt(2)
    assert out.rms_mean == pytest.approx(expected, rel=0.02)
    assert out.rms_max == pytest.approx(expected, rel=0.02)
    assert out.rms_std < 0.01


def test_dynamics_louder_signal_has_larger_rms(audio_cfg) -> None:
    ex = DynamicsExtractor(audio_cfg)
    quiet = ex.extract(AudioData(sine(440, 2.0, amp=0.1), SR))
    loud = ex.extract(AudioData(sine(440, 2.0, amp=0.8), SR))
    assert loud.rms_mean == pytest.approx(8 * quiet.rms_mean, rel=0.05)


def test_dynamics_detects_volume_change_over_time(audio_cfg) -> None:
    y = sine(440, 4.0, amp=1.0) * np.linspace(0.1, 1.0, 4 * SR, dtype=np.float32)
    out = DynamicsExtractor(audio_cfg).extract(AudioData(y, SR))
    assert out.rms_std > 0.1
    curve = np.array(out.rms_curve)
    assert curve[-1] > 5 * curve[0]
    assert np.all(np.diff(curve) > -0.02)  # ほぼ単調増加
    assert out.rms_max == pytest.approx(curve.max())


def test_dynamics_curve_hop_matches_config(audio_cfg) -> None:
    audio = AudioData(sine(440, 2.0), SR)
    out = DynamicsExtractor(audio_cfg).extract(audio)
    assert out.rms_curve_hop_sec == pytest.approx(audio_cfg.hop_length / SR)
    n_frames = 1 + (audio.y.size - audio_cfg.frame_length) // audio_cfg.hop_length
    assert len(out.rms_curve) == n_frames


def test_dynamics_silence_is_a_valid_measurement(audio_cfg) -> None:
    """無音の RMS 0 は欠損ではなく実測値。"""
    out = DynamicsExtractor(audio_cfg).extract(AudioData(np.zeros(SR, dtype=np.float32), SR))
    assert out.rms_mean == 0.0 and out.rms_max == 0.0


def test_dynamics_too_short_fails(audio_cfg) -> None:
    with pytest.raises(FeatureExtractionError, match="短すぎ"):
        DynamicsExtractor(audio_cfg).extract(AudioData(np.ones(100, dtype=np.float32), SR))
