"""audio (loader / preprocess) のテスト。音声はすべて合成する。"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicchar.audio import load_audio, preprocess
from musicchar.config_loader import load_config
from musicchar.models import AudioData, AudioLoadError, AudioTooShortError
from musicchar.models.config import AudioConfig

SR = 22050


def make_cfg(**overrides: object) -> AudioConfig:
    values = {"min_duration_sec": 1.0, "max_duration_sec": 10.0, **overrides}
    return dataclasses.replace(load_config().audio, **values)  # type: ignore[arg-type]


def sine(freq: float, sec: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------- loader ----------

def test_load_wav_mono(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    sf.write(p, sine(440, 2.0), SR)
    a = load_audio(p, SR)
    assert a.sr == SR
    assert a.y.ndim == 1
    assert a.duration == pytest.approx(2.0, abs=0.01)
    assert a.source == "a.wav"


def test_load_resamples(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    sf.write(p, sine(440, 2.0, sr=44100), 44100)
    a = load_audio(str(p), SR)
    assert a.sr == SR
    assert a.duration == pytest.approx(2.0, abs=0.01)


def test_load_stereo_is_mixed_to_mono(tmp_path: Path) -> None:
    p = tmp_path / "st.wav"
    stereo = np.stack([sine(440, 1.0), sine(660, 1.0)], axis=1)
    sf.write(p, stereo, SR)
    a = load_audio(p, SR)
    assert a.y.ndim == 1
    assert a.y.size == SR


@pytest.mark.skipif("MP3" not in sf.available_formats(), reason="libsndfile without MP3 support")
def test_load_mp3(tmp_path: Path) -> None:
    p = tmp_path / "a.mp3"
    sf.write(p, sine(440, 2.0, sr=44100), 44100, format="MP3")
    a = load_audio(p, SR)
    assert a.sr == SR
    assert a.duration == pytest.approx(2.0, abs=0.1)


def test_load_from_file_object() -> None:
    buf = io.BytesIO()
    sf.write(buf, sine(440, 1.0), SR, format="WAV")
    buf.seek(0)
    a = load_audio(buf, SR, source="upload.wav")
    assert a.source == "upload.wav"
    assert a.y.size == SR


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError, match="見つかりません"):
        load_audio(tmp_path / "nope.wav", SR)


def test_load_directory(tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path, SR)


def test_load_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.wav"
    p.write_bytes(b"this is not audio" * 10)
    with pytest.raises(AudioLoadError):
        load_audio(p, SR)


def test_load_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioLoadError):
        load_audio(p, SR)


def test_load_zero_length_audio(tmp_path: Path) -> None:
    p = tmp_path / "zero.wav"
    sf.write(p, np.zeros(0, dtype=np.float32), SR)
    with pytest.raises(AudioLoadError):
        load_audio(p, SR)


# ---------- preprocess ----------

def test_peak_normalization() -> None:
    a = AudioData(sine(440, 3.0, amp=0.2), SR, "x")
    out = preprocess(a, make_cfg())
    assert np.max(np.abs(out.y)) == pytest.approx(1.0)
    assert out.source == "x"


def test_peak_normalization_can_be_disabled() -> None:
    a = AudioData(sine(440, 3.0, amp=0.2), SR)
    out = preprocess(a, make_cfg(peak_normalize=False))
    assert np.max(np.abs(out.y)) == pytest.approx(0.2, abs=0.01)


def test_trims_leading_and_trailing_silence() -> None:
    y = np.concatenate([np.zeros(SR * 2), sine(440, 3.0), np.zeros(SR * 2)]).astype(np.float32)
    out = preprocess(AudioData(y, SR), make_cfg())
    assert out.duration == pytest.approx(3.0, abs=0.15)


def test_does_not_mutate_input() -> None:
    y = sine(440, 3.0, amp=0.2)
    before = y.copy()
    preprocess(AudioData(y, SR), make_cfg())
    np.testing.assert_array_equal(y, before)


def test_rejects_all_zero_audio() -> None:
    with pytest.raises(AudioTooShortError, match="無音"):
        preprocess(AudioData(np.zeros(SR * 5, dtype=np.float32), SR), make_cfg())


def test_rejects_near_silent_audio() -> None:
    a = AudioData(sine(440, 5.0, amp=1e-6), SR)
    with pytest.raises(AudioTooShortError):
        preprocess(a, make_cfg())


def test_rejects_too_short_after_trim() -> None:
    y = np.concatenate([np.zeros(SR * 5), sine(440, 0.3), np.zeros(SR * 5)]).astype(np.float32)
    with pytest.raises(AudioTooShortError, match="短すぎ"):
        preprocess(AudioData(y, SR), make_cfg())


def test_crops_long_audio_to_the_center() -> None:
    # 20秒: 前半 220Hz / 後半 880Hz。中央 10 秒を切り出すと両者が半々で含まれる。
    y = np.concatenate([sine(220, 10.0), sine(880, 10.0)])
    out = preprocess(AudioData(y, SR), make_cfg(max_duration_sec=10.0, peak_normalize=False))
    assert out.y.size == 10 * SR
    spec = np.abs(np.fft.rfft(out.y[: 5 * SR]))
    assert np.argmax(spec) * SR / (5 * SR) == pytest.approx(220, abs=2)
    spec = np.abs(np.fft.rfft(out.y[5 * SR :]))
    assert np.argmax(spec) * SR / (5 * SR) == pytest.approx(880, abs=2)


def test_short_enough_audio_is_not_cropped() -> None:
    out = preprocess(AudioData(sine(440, 4.0), SR), make_cfg(max_duration_sec=10.0))
    assert out.duration == pytest.approx(4.0, abs=0.15)
