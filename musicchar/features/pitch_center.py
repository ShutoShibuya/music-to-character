"""D. 音域(音の大体の高さ)の抽出。"""

from __future__ import annotations

import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.features.pitch_track import PitchTracker
from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig
from musicchar.models.features import PitchCenter


class PitchCenterExtractor(FeatureExtractor[PitchCenter]):
    """有声フレームの音高(MIDI)の平均値と中央値を求める。

    休符・無声フレームは統計から除外する(0 として平均に混ぜない)。
    """

    name = "d_pitch_center"

    def __init__(self, cfg: AudioConfig, tracker: PitchTracker | None = None) -> None:
        super().__init__(cfg)
        self.tracker = tracker if tracker is not None else PitchTracker(cfg)

    def extract(self, audio: AudioData) -> PitchCenter:
        """音高の平均・中央値を返す。音高が推定できなければ ``FeatureExtractionError``。"""
        midi = self.tracker.track(audio).voiced_midi
        return PitchCenter(mean_midi=float(np.mean(midi)), median_midi=float(np.median(midi)))
