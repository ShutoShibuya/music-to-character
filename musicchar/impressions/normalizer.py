"""特徴量の正規化(特徴量 -> 0〜1 の指標)。

正規化の方式は ``NORMALIZERS`` に登録された関数で、設定ファイルの ``method`` で選ぶ。
新しい方式を足しても、印象の計算式 (formulas.py) には影響しない。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Optional

from musicchar.models.config import NormRule
from musicchar.models.errors import ImpressionError
from musicchar.models.features import FeatureSet


def minmax(value: float, rule: NormRule) -> float:
    """``rule.min`` を 0、``rule.max`` を 1 とする線形変換。範囲外は 0〜1 にクリップする。"""
    return min(1.0, max(0.0, (value - rule.min) / (rule.max - rule.min)))


NORMALIZERS: dict[str, Callable[[float, NormRule], float]] = {"minmax": minmax}


def resolve_source(features: FeatureSet, source: str) -> Optional[float]:
    """``"b_tempo.bpm"`` のような属性パスから特徴量の値を取り出す。

    途中の特徴量が ``None``(抽出失敗)なら ``None`` を返す。

    Raises:
        ImpressionError: パスが存在しない、または値が数値でない(設定ミス)。
    """
    obj: Any = features
    for attr in source.split("."):
        if obj is None:
            return None
        if not hasattr(obj, attr):
            raise ImpressionError(f"特徴量のパスが存在しません: {source!r} ({attr!r})")
        obj = getattr(obj, attr)
    if obj is None:
        return None
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        raise ImpressionError(f"特徴量が数値ではありません: {source!r} = {obj!r}")
    return float(obj)


def normalize_features(
    features: FeatureSet, rules: Mapping[str, NormRule]
) -> dict[str, Optional[float]]:
    """全特徴量を 0〜1 に正規化する。

    欠損(``None``)や非有限値(NaN/inf)は ``None`` のまま返し、0 などで代用しない。

    Args:
        features: 抽出済みの特徴量。
        rules: 正規化指標名 -> ルール (``normalization.yaml``)。

    Returns:
        正規化指標名 -> 0〜1 の値 または None。

    Raises:
        ImpressionError: 未登録の正規化方式、または特徴量パスの設定ミス。
    """
    out: dict[str, Optional[float]] = {}
    for name, rule in rules.items():
        if rule.method not in NORMALIZERS:
            raise ImpressionError(f"未対応の正規化方式です: {rule.method!r} ({name})")
        raw = resolve_source(features, rule.source)
        out[name] = None if raw is None or not math.isfinite(raw) else NORMALIZERS[rule.method](raw, rule)
    return out
