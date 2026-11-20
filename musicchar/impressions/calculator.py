"""印象 1〜8 の計算(正規化と計算式の統括)。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from musicchar.impressions.formulas import Formula, weighted_mean_formula
from musicchar.impressions.normalizer import normalize_features
from musicchar.models.config import ImpressionRule, NormRule
from musicchar.models.features import FeatureSet
from musicchar.models.impressions import IMPRESSION_IDS, ImpressionSet


def compute_impressions(
    normalized: Mapping[str, Optional[float]],
    rules: Mapping[int, ImpressionRule],
    formula: Formula = weighted_mean_formula,
) -> ImpressionSet:
    """正規化済み特徴量から印象 1〜8 を計算する。

    Args:
        normalized: 正規化指標名 -> 0〜1 の値 または None。
        rules: 印象ID -> ルール (``impressions.yaml``)。1〜8 がすべて必要。
        formula: 計算式。省略時は ``weighted_mean_formula``。

    Returns:
        ``ImpressionSet``。計算できなかった印象は ``None``。
        一部の入力が欠けていた印象は、``missing_inputs`` に欠けた入力名が記録される。

    Raises:
        ValueError: ``rules`` に印象 1〜8 のいずれかがない場合。
    """
    if set(rules) != set(IMPRESSION_IDS):
        raise ValueError(f"rules には印象ID {list(IMPRESSION_IDS)} がちょうど必要です")
    values: dict[int, Optional[float]] = {}
    missing: dict[int, tuple[str, ...]] = {}
    for iid in IMPRESSION_IDS:
        value, absent = formula(normalized, rules[iid])
        values[iid] = value
        if absent:
            missing[iid] = absent
    return ImpressionSet(values=values, missing_inputs=missing)


def impressions_from_features(
    features: FeatureSet,
    norm_rules: Mapping[str, NormRule],
    impression_rules: Mapping[int, ImpressionRule],
    formula: Formula = weighted_mean_formula,
) -> ImpressionSet:
    """特徴量の正規化から印象の計算までを一括で行う。"""
    return compute_impressions(normalize_features(features, norm_rules), impression_rules, formula)
