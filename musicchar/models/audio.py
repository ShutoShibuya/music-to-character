"""音声データのデータ構造。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class AudioData:
    """モノラル音声波形。

    Attributes:
        y: 1次元の float 波形。
        sr: サンプリングレート(Hz)。
        source: 読み込み元の識別子(ファイル名など)。
    """

    y: np.ndarray
    sr: int
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.y, np.ndarray) or self.y.ndim != 1:
            raise ValueError("AudioData.y must be a 1-D numpy array")
        if self.sr <= 0:
            raise ValueError("AudioData.sr must be positive")
        if self.y.size and not np.all(np.isfinite(self.y)):
            raise ValueError("AudioData.y contains NaN or inf")

    @property
    def duration(self) -> float:
        """再生時間(秒)。"""
        return float(self.y.size) / float(self.sr)
