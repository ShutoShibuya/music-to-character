"""音楽的特徴量 A〜H のデータ構造。

抽出できなかった特徴量は ``None`` で表し、0 などの値で代用しない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MODES = ("major", "minor")


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")


@dataclass(frozen=True)
class KeyMode:
    """A. 調。

    Attributes:
        key: 主音名 (PITCH_CLASSES のいずれか)。判定できなければ None。
        mode: "major" / "minor"。判定できなければ None。
        confidence: 判定の確信度 (0〜1)。
    """

    key: Optional[str]
    mode: Optional[str]
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.key is not None and self.key not in PITCH_CLASSES:
            raise ValueError(f"invalid key: {self.key}")
        if self.mode is not None and self.mode not in MODES:
            raise ValueError(f"invalid mode: {self.mode}")
        _check_unit("confidence", self.confidence)

    @property
    def major_score(self) -> Optional[float]:
        """長調なら 1.0、短調なら 0.0、不明なら None。"""
        if self.mode is None:
            return None
        return 1.0 if self.mode == "major" else 0.0


@dataclass(frozen=True)
class Tempo:
    """B. テンポ。bpm: 拍/分。"""

    bpm: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.bpm) and self.bpm > 0):
            raise ValueError("bpm must be a positive finite number")


@dataclass(frozen=True)
class Melody:
    """C. メロディ。

    Attributes:
        pitch_change_rate: 1秒あたりの音高変化回数。
        mean_interval_semitones: 連続する音の平均音程差(半音, 絶対値)。
        leap_ratio: 跳躍(閾値以上の音程差)の割合 (0〜1)。
        repetition_score: 旋律の繰り返しの強さ (0〜1)。
        smoothness: 旋律の滑らかさ (0〜1, 大きいほど滑らか)。
    """

    pitch_change_rate: float
    mean_interval_semitones: float
    leap_ratio: float
    repetition_score: float
    smoothness: float

    def __post_init__(self) -> None:
        if self.pitch_change_rate < 0 or self.mean_interval_semitones < 0:
            raise ValueError("pitch_change_rate / mean_interval must be >= 0")
        _check_unit("leap_ratio", self.leap_ratio)
        _check_unit("repetition_score", self.repetition_score)
        _check_unit("smoothness", self.smoothness)


@dataclass(frozen=True)
class PitchCenter:
    """D. 音域(大体の高さ)。MIDIノート番号。"""

    mean_midi: float
    median_midi: float


@dataclass(frozen=True)
class PitchRange:
    """E. 音域(幅)。MIDIノート番号と半音数。"""

    max_midi: float
    min_midi: float
    range_semitones: float

    def __post_init__(self) -> None:
        if self.max_midi < self.min_midi:
            raise ValueError("max_midi must be >= min_midi")
        if self.range_semitones < 0:
            raise ValueError("range_semitones must be >= 0")


@dataclass(frozen=True)
class Dynamics:
    """F. 音量と音量変化。

    Attributes:
        rms_mean / rms_std / rms_max: RMS の統計量。
        rms_curve: 時間軸上の RMS(固定ホップ間隔)。
        rms_curve_hop_sec: rms_curve の隣接要素の時間間隔(秒)。curve が空なら 0.0。
    """

    rms_mean: float
    rms_std: float
    rms_max: float
    rms_curve: tuple[float, ...] = ()
    rms_curve_hop_sec: float = 0.0

    def __post_init__(self) -> None:
        stats = (self.rms_mean, self.rms_std, self.rms_max)
        if not all(math.isfinite(v) and v >= 0 for v in stats):
            raise ValueError("RMS statistics must be finite and >= 0")
        if self.rms_curve_hop_sec < 0:
            raise ValueError("rms_curve_hop_sec must be >= 0")


@dataclass(frozen=True)
class Chords:
    """G. コード。

    Attributes:
        chord_types: コード種別ごとの出現回数 (例 {"maj": 12, "min": 4})。
        progression: 検出されたコード列 (例 ("C", "Am", "F", "G"))。
        change_rate_per_sec: 1秒あたりのコード変化回数。
        complexity: コードの複雑さ (0〜1)。
        minor_ratio: 短調系コードの割合 (0〜1)。
    """

    chord_types: dict[str, int]
    progression: tuple[str, ...]
    change_rate_per_sec: float
    complexity: float
    minor_ratio: float

    def __post_init__(self) -> None:
        if self.change_rate_per_sec < 0:
            raise ValueError("change_rate_per_sec must be >= 0")
        _check_unit("complexity", self.complexity)
        _check_unit("minor_ratio", self.minor_ratio)


@dataclass(frozen=True)
class Articulation:
    """H. アーティキュレーション。

    Attributes:
        onset_strength_mean: 音の立ち上がりの強さの平均。
        onset_density: 1秒あたりの発音数。
        mean_note_duration: 音の平均持続時間(秒)。
        overlap_ratio: 音の重なりの割合 (0〜1)。
        gap_ratio: 音の切れ目(無音区間)の割合 (0〜1)。
    """

    onset_strength_mean: float
    onset_density: float
    mean_note_duration: float
    overlap_ratio: float
    gap_ratio: float

    def __post_init__(self) -> None:
        if min(self.onset_strength_mean, self.onset_density, self.mean_note_duration) < 0:
            raise ValueError("articulation values must be >= 0")
        _check_unit("overlap_ratio", self.overlap_ratio)
        _check_unit("gap_ratio", self.gap_ratio)


@dataclass(frozen=True)
class FeatureSet:
    """A〜H の特徴量一式。抽出失敗は None、理由は errors に記録する。

    Attributes:
        errors: 特徴量名 -> 失敗理由。
    """

    a_key: Optional[KeyMode] = None
    b_tempo: Optional[Tempo] = None
    c_melody: Optional[Melody] = None
    d_pitch_center: Optional[PitchCenter] = None
    e_pitch_range: Optional[PitchRange] = None
    f_dynamics: Optional[Dynamics] = None
    g_chords: Optional[Chords] = None
    h_articulation: Optional[Articulation] = None
    errors: dict[str, str] = field(default_factory=dict)
