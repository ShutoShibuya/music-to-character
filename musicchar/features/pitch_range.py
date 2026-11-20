"""E. 音域(音高の幅)の抽出。"""

from __future__ import annotations

import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.features.pitch_track import PitchTracker
from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig
from musicchar.models.features import PitchRange


class PitchRangeExtractor(FeatureExtractor[PitchRange]):
    """有声フレームの音高の最大値・最小値・範囲を求める。

    pyin は音の切り替わり部分や誤検出で一時的に外れた値を出すことがあるため、
    ``cfg.pitch_range_percentile`` (%) を上下から除いたパーセンタイルを最大/最小とする。
    0 を指定すると真の最大/最小になる。
    """

    name = "e_pitch_range"

    def __init__(self, cfg: AudioConfig, tracker: PitchTracker | None = None) -> None:
        super().__init__(cfg)
        self.tracker = tracker if tracker is not None else PitchTracker(cfg)

    def extract(self, audio: AudioData) -> PitchRange:
        """音高の最大・最小・範囲(半音)を返す。音高が推定できなければ ``FeatureExtractionError``。"""
        midi = self.tracker.track(audio).voiced_midi
        q = self.cfg.pitch_range_percentile
        lo, hi = (float(np.percentile(midi, q)), float(np.percentile(midi, 100.0 - q)))
        return PitchRange(max_midi=hi, min_midi=lo, range_semitones=hi - lo)
