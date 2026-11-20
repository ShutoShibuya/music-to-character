"""テスト共通のフィクスチャ。"""

from __future__ import annotations

import pytest

from musicchar.models import (
    Articulation,
    Chords,
    Dynamics,
    FeatureSet,
    KeyMode,
    Melody,
    PitchCenter,
    PitchRange,
    Tempo,
)


@pytest.fixture
def full_featureset() -> FeatureSet:
    """A〜H がすべて揃った FeatureSet。"""
    return FeatureSet(
        a_key=KeyMode(key="C", mode="major", confidence=0.8),
        b_tempo=Tempo(bpm=120.0),
        c_melody=Melody(
            pitch_change_rate=3.0,
            mean_interval_semitones=2.5,
            leap_ratio=0.2,
            repetition_score=0.5,
            smoothness=0.6,
        ),
        d_pitch_center=PitchCenter(mean_midi=64.0, median_midi=62.0),
        e_pitch_range=PitchRange(max_midi=79.0, min_midi=48.0, range_semitones=31.0),
        f_dynamics=Dynamics(rms_mean=0.1, rms_std=0.05, rms_max=0.4, rms_curve=(0.1, 0.2)),
        g_chords=Chords(
            chord_types={"maj": 8, "min": 4},
            progression=("C", "Am", "F", "G"),
            change_rate_per_sec=0.5,
            complexity=0.3,
            minor_ratio=0.33,
        ),
        h_articulation=Articulation(
            onset_strength_mean=2.0,
            onset_density=3.0,
            mean_note_duration=0.4,
            overlap_ratio=0.3,
            gap_ratio=0.2,
        ),
    )


# ---------- 分析結果の共有 (1 回の分析に約 10 秒かかるのでセッションで 1 度だけ実行) ----------

import io  # noqa: E402

import soundfile as sf  # noqa: E402


@pytest.fixture(scope="session")
def sample_wav_bytes() -> bytes:
    """16 音 (0.5 秒ずつ) のドレミの旋律を WAV にしたもの (約 8 秒)。"""
    from synth import SR, melody

    buf = io.BytesIO()
    sf.write(buf, melody([60, 62, 64, 65, 67, 65, 64, 62] * 2, 0.5), SR, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="session")
def app_config():
    from musicchar.config_loader import load_config

    return load_config()


@pytest.fixture(scope="session")
def sample_result(sample_wav_bytes, app_config):
    """サンプル音声の分析結果 (単音の旋律なので、コード (G) は抽出できない)。"""
    from musicchar.pipeline import analyze

    return analyze(io.BytesIO(sample_wav_bytes), app_config, source="sample.wav")
