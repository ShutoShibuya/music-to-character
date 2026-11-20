"""pipeline (音声 -> キャラクター) のテスト。"""

from __future__ import annotations

import dataclasses
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from synth import SR, sine

from musicchar import pipeline
from musicchar.models import AudioLoadError, AudioTooShortError, FeatureSet
from musicchar.pipeline import analyze


def test_analysis_produces_every_stage(sample_result, app_config) -> None:
    r = sample_result
    assert r.source == "sample.wav"
    assert r.analyzed_duration == pytest.approx(8.0, abs=0.3)
    assert r.original_duration == pytest.approx(8.0, abs=0.1)
    assert r.features.b_tempo is not None and r.features.c_melody is not None
    assert r.features.d_pitch_center is not None and r.features.f_dynamics is not None
    assert set(r.impressions.values) == set(range(1, 9))
    assert set(r.character.parts) == set(app_config.parts)
    assert ET.fromstring(r.svg).tag.endswith("svg")


def test_failed_features_are_reported_not_hidden(sample_result) -> None:
    """単音の旋律にはコードがないので G は None で、理由が記録される。"""
    assert sample_result.features.g_chords is None
    assert "g_chords" in sample_result.features.errors


def test_impressions_depending_on_failed_features_still_computed_from_the_rest(sample_result) -> None:
    imp = sample_result.impressions
    assert imp.get(8) is not None  # G が欠けても H だけで計算される
    assert "chord_complexity" in imp.missing_inputs[8]


def test_analysis_is_deterministic(sample_wav_bytes, app_config, sample_result) -> None:
    again = analyze(io.BytesIO(sample_wav_bytes), app_config, source="sample.wav")
    assert again.svg == sample_result.svg
    assert again.impressions == sample_result.impressions


def test_analyze_accepts_a_path(tmp_path: Path, sample_wav_bytes, app_config, sample_result) -> None:
    p = tmp_path / "song.wav"
    p.write_bytes(sample_wav_bytes)
    r = analyze(p, app_config)
    assert r.source == "song.wav"
    assert r.svg == sample_result.svg


def test_result_is_immutable(sample_result) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_result.svg = ""  # type: ignore[misc]


# ---------- 入力の異常 ----------

def test_missing_file(app_config, tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError):
        analyze(tmp_path / "nothing.wav", app_config)


def test_corrupt_file(app_config) -> None:
    with pytest.raises(AudioLoadError):
        analyze(io.BytesIO(b"not an audio file" * 20), app_config, source="bad.wav")


def test_silent_audio_is_rejected(app_config) -> None:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(SR * 6, dtype=np.float32), SR, format="WAV")
    buf.seek(0)
    with pytest.raises(AudioTooShortError, match="無音"):
        analyze(buf, app_config)


def test_too_short_audio_is_rejected(app_config) -> None:
    buf = io.BytesIO()
    sf.write(buf, sine(440, 1.0), SR, format="WAV")
    buf.seek(0)
    with pytest.raises(AudioTooShortError, match="短すぎ"):
        analyze(buf, app_config)


def test_all_features_failing_still_gives_a_neutral_character(sample_wav_bytes, app_config, monkeypatch) -> None:
    """すべての特徴量が抽出できなくても落ちず、中立値のキャラクターが返る (欠損を 0 扱いしない)。"""
    all_failed = FeatureSet(errors={"a_key": "x", "b_tempo": "x"})
    monkeypatch.setattr(pipeline, "extract_all", lambda *a, **k: all_failed)
    r = analyze(io.BytesIO(sample_wav_bytes), app_config, source="x.wav")
    assert all(v is None for v in r.impressions.values.values())
    for name, part in r.character.parts.items():
        assert part.t == 0.5 and part.estimated and part.sources == (), name
    ET.fromstring(r.svg)
