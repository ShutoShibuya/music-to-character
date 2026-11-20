"""B. テンポ(BPM)の抽出。"""

from __future__ import annotations

import librosa
import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.models.audio import AudioData
from musicchar.models.errors import FeatureExtractionError
from musicchar.models.features import Tempo

# オンセット強度の変動がこれ以下なら「拍の手がかりなし」とみなす(数値誤差の閾値)。
_FLAT_ENVELOPE_EPS = 1e-8


class TempoExtractor(FeatureExtractor[Tempo]):
    """オンセット強度の自己相関から全体のテンポを推定する。

    ``librosa.feature.tempo`` を使用する。librosa は拍の手がかりがない入力
    (完全な無音など)にも事前分布由来の値を返してしまうため、
    オンセット強度が平坦な場合はここで失敗として扱う。
    """

    name = "b_tempo"

    def extract(self, audio: AudioData) -> Tempo:
        """BPM を推定する。拍の手がかりがない場合は ``FeatureExtractionError``。"""
        try:
            onset_env = librosa.onset.onset_strength(
                y=audio.y, sr=audio.sr, hop_length=self.cfg.hop_length
            )
            if onset_env.size == 0 or float(np.ptp(onset_env)) <= _FLAT_ENVELOPE_EPS:
                raise FeatureExtractionError("拍の手がかりがありません(オンセット強度が平坦)")
            bpm = float(
                np.ravel(
                    librosa.feature.tempo(
                        onset_envelope=onset_env, sr=audio.sr, hop_length=self.cfg.hop_length
                    )
                )[0]
            )
        except FeatureExtractionError:
            raise
        except Exception as e:
            raise FeatureExtractionError(f"テンポの推定に失敗: {e}") from e
        if not np.isfinite(bpm) or bpm <= 0:
            raise FeatureExtractionError(f"テンポが不正な値でした: {bpm}")
        return Tempo(bpm=bpm)
