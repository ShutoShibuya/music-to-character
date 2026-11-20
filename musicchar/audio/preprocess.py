"""音声の前処理(無音トリム・長さ制限・ピーク正規化)。"""

from __future__ import annotations

import librosa
import numpy as np

from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig
from musicchar.models.errors import AudioTooShortError

# ピークがこれ未満の波形は無音として扱う(float 波形で約 -80 dBFS)。
SILENCE_PEAK = 1e-4


def preprocess(audio: AudioData, cfg: AudioConfig) -> AudioData:
    """前後の無音を除去し、長すぎる場合は中央部分を切り出し、ピーク正規化する。

    処理順: 無音判定 → 前後トリム → 最小長チェック → 中央切り出し → ピーク正規化。
    入力の ``audio`` は変更しない。

    Args:
        audio: 読み込み済みの音声。
        cfg: ``AudioConfig`` (trim_top_db / min・max_duration_sec / peak_normalize)。

    Returns:
        前処理後の新しい ``AudioData``。

    Raises:
        AudioTooShortError: 全体が無音、またはトリム後の長さが ``min_duration_sec`` 未満。
    """
    y = audio.y
    if y.size == 0 or float(np.max(np.abs(y))) < SILENCE_PEAK:
        raise AudioTooShortError("音声が無音のため分析できません")

    y, _ = librosa.effects.trim(y, top_db=cfg.trim_top_db)
    duration = y.size / audio.sr
    if duration < cfg.min_duration_sec:
        raise AudioTooShortError(
            f"音声が短すぎます: トリム後 {duration:.2f} 秒 (最小 {cfg.min_duration_sec} 秒)"
        )

    max_len = int(cfg.max_duration_sec * audio.sr)
    if y.size > max_len:
        start = (y.size - max_len) // 2
        y = y[start : start + max_len]

    y = np.array(y, copy=True)  # 元の波形とメモリを共有しない
    if cfg.peak_normalize:
        peak = float(np.max(np.abs(y)))
        if peak < SILENCE_PEAK:  # 切り出した区間が無音だった場合
            raise AudioTooShortError("分析対象区間が無音です")
        y = y / peak

    return AudioData(y=y, sr=audio.sr, source=audio.source)
