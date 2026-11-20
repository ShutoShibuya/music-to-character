"""設定ファイルを読み込んだ結果のデータ構造。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from musicchar.models.render_config import RenderConfig


@dataclass(frozen=True)
class AudioConfig:
    """音声読込・前処理・音高解析の設定。"""

    sample_rate: int
    min_duration_sec: float
    max_duration_sec: float
    trim_top_db: float
    peak_normalize: bool
    fmin_hz: float
    fmax_hz: float
    frame_length: int
    hop_length: int
    pitch_min_voiced_ratio: float
    pitch_range_percentile: float
    pitch_max_transition_rate: float
    pitch_min_note_frames: int


@dataclass(frozen=True)
class NormRule:
    """特徴量1つの正規化ルール。

    Attributes:
        source: FeatureSet からの属性パス (例 "b_tempo.bpm")。
        method: 正規化方式。現在は "minmax" のみ。
        min / max: 0.0 / 1.0 に対応する特徴量の値。範囲外はクリップする。
    """

    source: str
    method: str
    min: float
    max: float


@dataclass(frozen=True)
class ImpressionInput:
    """印象計算における入力特徴量1つの設定。

    direction: "direct" = 特徴量が大きいほど印象値が大きい / "inverse" = 逆。
    """

    weight: float
    direction: str


@dataclass(frozen=True)
class ImpressionRule:
    """印象1つの設定。label_low が 0.0、label_high が 1.0 側の印象。"""

    id: int
    label_low: str
    label_high: str
    inputs: dict[str, ImpressionInput]


@dataclass(frozen=True)
class PartSource:
    """パーツに影響する印象1つ。direction: "positive" / "negative"(反転)。"""

    impression: int
    direction: str


@dataclass(frozen=True)
class PartRule:
    """キャラクターのパーツ1つの設定。

    Attributes:
        group: "face" / "body" / "other"。
        unit: 値の単位の説明 (例 "deg", "ratio")。
        sources: 影響する印象と変化方向。
        combine: 合成方式 ("mean" / "weighted_mean")。
        weights: weighted_mean のときの sources と同順の重み。
        min / max: t=0 / t=1 のときの実値。
        integer: True なら描画時に整数へ丸める値。
        label: 画面表示用の日本語名。
    """

    name: str
    group: str
    unit: str
    sources: tuple[PartSource, ...]
    combine: str
    weights: Optional[tuple[float, ...]]
    min: float
    max: float
    integer: bool = False
    label: str = ""


@dataclass(frozen=True)
class MelodyConfig:
    """C. メロディ抽出のパラメータ (各項目の意味は features.yaml を参照)。"""

    max_gap_sec: float
    leap_threshold_semitones: float
    ngram_length: int
    smoothness_ref_semitones: float
    min_notes: int


@dataclass(frozen=True)
class ChromaConfig:
    """クロマ計算の設定。hpss: 打楽器成分を除いてから計算するか。"""

    hpss: bool


@dataclass(frozen=True)
class KeyConfig:
    """A. 調の判定設定。min_correlation: 判定に必要な最小の相関。"""

    min_correlation: float


@dataclass(frozen=True)
class ChordConfig:
    """G. コード推定の設定 (各項目の意味は features.yaml を参照)。"""

    window_sec: float
    switch_penalty: float
    min_similarity: float
    silence_ratio: float
    min_chord_ratio: float
    extended_penalty: float
    weight_extension: float
    weight_type_variety: float
    weight_chord_variety: float
    type_ref: int
    chord_ref: int


@dataclass(frozen=True)
class ArticulationConfig:
    """H. アーティキュレーション抽出の設定 (各項目の意味は features.yaml を参照)。"""

    decay_threshold_db: float
    gap_threshold_db: float
    onset_context_frames: int
    onset_decay_ratio: float
    min_onsets: int


@dataclass(frozen=True)
class FeaturesConfig:
    """特徴量抽出のパラメータ全体。"""

    melody: MelodyConfig
    chroma: ChromaConfig
    key: KeyConfig
    chord: ChordConfig
    articulation: ArticulationConfig


@dataclass(frozen=True)
class AppConfig:
    """全設定をまとめたもの。"""

    audio: AudioConfig
    features: FeaturesConfig
    normalization: dict[str, NormRule]
    impressions: dict[int, ImpressionRule]
    parts: dict[str, PartRule]
    render: RenderConfig
