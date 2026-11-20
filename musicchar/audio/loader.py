"""音声ファイルの読み込み。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np

from musicchar.models.audio import AudioData
from musicchar.models.errors import AudioLoadError


def load_audio(
    file: str | os.PathLike[str] | BinaryIO,
    sample_rate: int,
    source: str | None = None,
) -> AudioData:
    """音声ファイルを読み込み、モノラル・指定サンプリングレートの波形にする。

    WAV / MP3 / FLAC など、librosa (soundfile) が読める形式に対応する。
    ステレオはモノラルに混合され、必要ならリサンプルされる。

    Args:
        file: ファイルパス、またはファイルオブジェクト (Streamlit のアップロード等)。
        sample_rate: 読み込み後のサンプリングレート (Hz)。
        source: ``AudioData.source`` に入れる名前。省略時はパス名、
            ファイルオブジェクトでは ``name`` 属性 (なければ空文字)。

    Returns:
        読み込んだ ``AudioData``。

    Raises:
        AudioLoadError: ファイルが存在しない・読めない・形式が不正・中身が空の場合。
    """
    if isinstance(file, (str, os.PathLike)):
        path = Path(file)
        if not path.is_file():
            raise AudioLoadError(f"音声ファイルが見つかりません: {path}")
        name = source if source is not None else path.name
    else:
        name = source if source is not None else str(getattr(file, "name", ""))

    try:
        y, sr = librosa.load(file, sr=sample_rate, mono=True)
    except Exception as e:  # librosa/soundfile は形式不正を RuntimeError 系で返す
        raise AudioLoadError(f"音声ファイルを読み込めません ({name or 'stream'}): {e}") from e

    if y.size == 0:
        raise AudioLoadError(f"音声データが空です: {name or 'stream'}")
    if not np.all(np.isfinite(y)):
        raise AudioLoadError(f"音声データに NaN/inf が含まれています: {name or 'stream'}")
    return AudioData(y=y, sr=int(sr), source=name)
