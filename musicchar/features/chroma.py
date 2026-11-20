"""クロマ(12音名ごとのエネルギー)の計算。A(調)と G(コード)で共有する。"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig, ChromaConfig
from musicchar.models.errors import FeatureExtractionError


@dataclass(frozen=True, eq=False)
class Chroma:
    """フレームごとのクロマ。

    Attributes:
        matrix: 形状 (12, フレーム数)。行は C, C#, ..., B。エネルギーを保持した
            値(フレームごとの正規化はしない)なので、無音フレームはほぼ 0。
        hop_sec: 隣接フレームの時間間隔(秒)。
    """

    matrix: np.ndarray
    hop_sec: float


class ChromaProvider:
    """``librosa.feature.chroma_cqt`` によるクロマ計算。同じ音声の結果をキャッシュする。

    ``chroma.hpss`` が真なら、先に ``librosa.effects.harmonic`` で打楽器成分を除く。
    """

    def __init__(self, cfg: AudioConfig, chroma_cfg: ChromaConfig) -> None:
        self.cfg = cfg
        self.chroma_cfg = chroma_cfg
        self._audio: AudioData | None = None
        self._result: Chroma | None = None
        self._error: FeatureExtractionError | None = None

    def chroma(self, audio: AudioData) -> Chroma:
        """クロマを返す(キャッシュ済みならそれを返す)。

        Raises:
            FeatureExtractionError: 波形が短すぎる、または計算に失敗した場合。
        """
        if self._audio is not audio:
            self._audio, self._result, self._error = audio, None, None
            try:
                self._result = self._compute(audio)
            except FeatureExtractionError as e:
                self._error = e
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def _compute(self, audio: AudioData) -> Chroma:
        hop = self.cfg.hop_length
        if audio.y.size < self.cfg.frame_length:
            raise FeatureExtractionError("音声が短すぎてクロマを計算できません")
        try:
            y = librosa.effects.harmonic(audio.y, margin=3.0) if self.chroma_cfg.hpss else audio.y
            matrix = librosa.feature.chroma_cqt(y=y, sr=audio.sr, hop_length=hop, norm=None)
        except Exception as e:
            raise FeatureExtractionError(f"クロマの計算に失敗: {e}") from e
        if matrix.size == 0 or not np.all(np.isfinite(matrix)):
            raise FeatureExtractionError("クロマが計算できませんでした")
        return Chroma(matrix=matrix, hop_sec=hop / audio.sr)
