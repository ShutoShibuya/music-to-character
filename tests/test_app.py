"""Streamlit 画面 (app.py) のテスト。Streamlit の AppTest でアップロードまで通す。"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest
from synth import SR

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 240  # 初回は numba のコンパイルがあるので長めにとる


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def upload(at: AppTest, name: str, data: bytes) -> AppTest:
    at.file_uploader[0].upload(name, data, "audio/wav")
    return at.run()


def wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, SR, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def uploaded_app(sample_wav_bytes) -> AppTest:
    return upload(new_app(), "demo.wav", sample_wav_bytes)


def test_initial_screen_asks_for_a_file() -> None:
    at = new_app()
    assert not at.exception
    assert at.title[0].value == "Music Character Generator"
    assert len(at.file_uploader) == 1 and "アップロード" in at.info[0].value
    assert len(at.get("image")) == 0 and len(at.get("download_button")) == 0


def test_upload_shows_the_character_and_all_tabs(uploaded_app: AppTest) -> None:
    at = uploaded_app
    assert not at.exception and not at.error
    assert "demo.wav" in at.subheader[0].value
    assert len(at.get("image")) == 1  # キャラクター
    assert [t.label for t in at.tabs] == ["印象", "特徴量", "キャラクターのパーツ"]
    assert len(at.get("download_button")) == 2  # SVG と JSON


def test_all_eight_impressions_are_shown_as_bars(uploaded_app: AppTest) -> None:
    assert len(uploaded_app.get("progress")) == 8  # コード (G) が欠けても、全印象が残りの入力から計算される
    labels = " ".join(m.value for m in uploaded_app.markdown)
    for left, right in (("激しい", "穏やか"), ("明るい", "暗い"), ("おしゃれ", "素朴")):
        assert left in labels and right in labels


def test_failed_features_are_announced_with_a_warning(uploaded_app: AppTest) -> None:
    assert len(uploaded_app.warning) == 1
    assert "g_chords" in uploaded_app.warning[0].value


def test_feature_and_part_tables_are_shown(uploaded_app: AppTest) -> None:
    assert len(uploaded_app.dataframe) == 2  # 特徴量表とパーツ表


def test_corrupt_file_shows_an_error_without_crashing() -> None:
    at = upload(new_app(), "broken.wav", b"this is not audio" * 30)
    assert not at.exception
    assert len(at.error) == 1 and "読み込めませんでした" in at.error[0].value
    assert len(at.get("image")) == 0


def test_silent_file_shows_an_error() -> None:
    at = upload(new_app(), "silence.wav", wav_bytes(np.zeros(SR * 6, dtype=np.float32)))
    assert not at.exception
    assert len(at.error) == 1 and "無音" in at.error[0].value


def test_too_short_file_shows_an_error() -> None:
    t = np.arange(SR) / SR
    at = upload(new_app(), "short.wav", wav_bytes((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)))
    assert not at.exception
    assert len(at.error) == 1 and "短すぎ" in at.error[0].value
