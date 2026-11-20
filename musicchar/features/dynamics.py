"""F. 音量および音量変化(RMS)の抽出。"""

from __future__ import annotations

import librosa
import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.models.audio import AudioData
from musicchar.models.errors import FeatureExtractionError
from musicchar.models.features import Dynamics


class DynamicsExtractor(FeatureExtractor[Dynamics]):
    """フレームごとの RMS から平均・標準偏差・最大値・時間変化を求める。

    ``librosa.feature.rms`` を ``center=False`` で使う(端をゼロ詰めすると
    先頭・末尾のフレームが過小評価されるため)。値は入力波形の振幅スケール依存で、
    前処理でピーク正規化されていれば曲間で比較できる。
    """

    name = "f_dynamics"

    def extract(self, audio: AudioData) -> Dynamics:
        """RMS 統計と RMS 曲線を返す。波形がフレーム長より短ければ ``FeatureExtractionError``。"""
        frame, hop = self.cfg.frame_length, self.cfg.hop_length
        if audio.y.size < frame:
            raise FeatureExtractionError(
                f"音声が短すぎて RMS を計算できません ({audio.y.size} < {frame} サンプル)"
            )
        try:
            rms = librosa.feature.rms(y=audio.y, frame_length=frame, hop_length=hop, center=False)[0]
        except Exception as e:
            raise FeatureExtractionError(f"RMS の計算に失敗: {e}") from e
        if rms.size == 0 or not np.all(np.isfinite(rms)):
            raise FeatureExtractionError("RMS が計算できませんでした")
        return Dynamics(
            rms_mean=float(np.mean(rms)),
            rms_std=float(np.std(rms)),
            rms_max=float(np.max(rms)),
            rms_curve=tuple(float(v) for v in rms),
            rms_curve_hop_sec=hop / audio.sr,
        )
