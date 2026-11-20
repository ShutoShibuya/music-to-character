"""特徴量抽出の統括。"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from musicchar.features.base import FeatureExtractor
from musicchar.features.dynamics import DynamicsExtractor
from musicchar.features.articulation import ArticulationExtractor
from musicchar.features.chords import ChordExtractor
from musicchar.features.chroma import ChromaProvider
from musicchar.features.key_mode import KeyModeExtractor
from musicchar.features.melody import MelodyExtractor
from musicchar.features.pitch_center import PitchCenterExtractor
from musicchar.features.pitch_range import PitchRangeExtractor
from musicchar.features.pitch_track import PitchTracker
from musicchar.features.tempo import TempoExtractor
from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig, FeaturesConfig
from musicchar.models.errors import FeatureExtractionError
from musicchar.models.features import FeatureSet

_FEATURE_FIELDS = tuple(f.name for f in dataclasses.fields(FeatureSet) if f.name != "errors")


def default_extractors(
    cfg: AudioConfig, features_cfg: FeaturesConfig
) -> list[FeatureExtractor[Any]]:
    """実装済みの全抽出器を返す。

    音高系 (C・D・E) は 1 つの ``PitchTracker``、クロマ系 (A・G) は 1 つの
    ``ChromaProvider`` を共有し、重い計算を 1 回で済ませる。
    """
    tracker = PitchTracker(cfg)
    provider = ChromaProvider(cfg, features_cfg.chroma)
    return [
        KeyModeExtractor(cfg, features_cfg.key, features_cfg.chroma, provider),
        TempoExtractor(cfg),
        MelodyExtractor(cfg, features_cfg.melody, tracker),
        PitchCenterExtractor(cfg, tracker),
        PitchRangeExtractor(cfg, tracker),
        ChordExtractor(cfg, features_cfg.chord, features_cfg.chroma, provider),
        DynamicsExtractor(cfg),
        ArticulationExtractor(cfg, features_cfg.articulation),
    ]


def extract_all(
    audio: AudioData,
    cfg: AudioConfig,
    features_cfg: FeaturesConfig,
    extractors: Sequence[FeatureExtractor[Any]] | None = None,
) -> FeatureSet:
    """各抽出器を独立に実行して ``FeatureSet`` にまとめる。

    ある特徴量の失敗が他の特徴量に影響しないよう、例外は抽出器ごとに捕捉する。
    失敗した特徴量、および抽出器が登録されていない特徴量は ``None`` とし、
    理由を ``FeatureSet.errors`` に記録する(0 などで代用しない)。

    Args:
        audio: 前処理済みの音声。
        cfg: 音声設定。
        features_cfg: 特徴量抽出の設定。
        extractors: 使用する抽出器。None なら ``default_extractors(cfg, features_cfg)``。

    Returns:
        抽出結果の ``FeatureSet``。
    """
    extractors = default_extractors(cfg, features_cfg) if extractors is None else list(extractors)
    names = [e.name for e in extractors]
    unknown = set(names) - set(_FEATURE_FIELDS)
    if unknown:
        raise ValueError(f"FeatureSet に存在しないフィールド名: {sorted(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("同じフィールドを担当する抽出器が重複しています")

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for ex in extractors:
        try:
            results[ex.name] = ex.extract(audio)
        except FeatureExtractionError as e:
            errors[ex.name] = str(e)
        except Exception as e:  # 想定外の例外でも他の特徴量の抽出は続ける
            errors[ex.name] = f"想定外のエラー: {type(e).__name__}: {e}"

    for field in _FEATURE_FIELDS:
        if field not in results and field not in errors:
            errors[field] = "抽出器が登録されていません"
    return FeatureSet(**results, errors=errors)
