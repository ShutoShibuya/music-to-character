"""印象 -> キャラクターの各パーツの値への変換。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Optional

from musicchar.character.combiner import COMBINERS, Combiner
from musicchar.models.character import CharacterParams, PartParam
from musicchar.models.config import PartRule
from musicchar.models.errors import ConfigError
from musicchar.models.impressions import ImpressionSet

NEUTRAL_T = 0.5


def map_part(
    impressions: ImpressionSet, rule: PartRule, combiner: Combiner
) -> PartParam:
    """1 つのパーツの値を求める。

    1. 各 source の印象 ``v`` を、``direction`` が ``negative`` なら ``1 - v`` に反転する。
    2. 反転後の値を ``combiner`` で合成して ``t`` (0〜1) とする。
    3. 値 = ``min + t * (max - min)``。``integer`` なら四捨五入する。

    推定不能 (``None``) の印象は合成から除き、重みも残りで再正規化する。
    すべての印象が推定不能なら ``t`` は中立値 0.5 とする。いずれの場合も
    ``estimated=True`` にする。

    Args:
        impressions: 印象 1〜8。
        rule: パーツの設定。
        combiner: 合成戦略。

    Returns:
        パーツのパラメータ。
    """
    ts: list[float] = []
    weights: list[float] = []
    used: list[int] = []
    for i, src in enumerate(rule.sources):
        v: Optional[float] = impressions.get(src.impression)
        if v is None:
            continue
        ts.append(v if src.direction == "positive" else 1.0 - v)
        weights.append(rule.weights[i] if rule.weights is not None else 1.0)
        used.append(src.impression)

    estimated = len(used) < len(rule.sources)
    t = combiner.combine(ts, weights) if ts else NEUTRAL_T
    t = min(1.0, max(0.0, t))

    value = rule.min + t * (rule.max - rule.min)
    if rule.integer:
        value = float(math.floor(value + 0.5))
    return PartParam(name=rule.name, value=value, t=t, sources=tuple(used), estimated=estimated)


def map_parts(
    impressions: ImpressionSet,
    rules: Mapping[str, PartRule],
    combiners: Optional[Mapping[str, Combiner]] = None,
) -> CharacterParams:
    """全パーツの値を求める。

    Args:
        impressions: 印象 1〜8。
        rules: パーツ名 -> 設定 (``character_parts.yaml``)。
        combiners: 合成方式名 -> 戦略。省略時は ``COMBINERS`` (mean / weighted_mean)。

    Returns:
        全パーツの ``CharacterParams``。同じ入力に対して常に同じ結果になる。

    Raises:
        ConfigError: パーツの ``combine`` が ``combiners`` に登録されていない場合。
    """
    table = COMBINERS if combiners is None else combiners
    parts: dict[str, PartParam] = {}
    for name, rule in rules.items():
        if rule.combine not in table:
            raise ConfigError(f"パーツ {name!r}: 未対応の合成方式です: {rule.combine!r}")
        parts[name] = map_part(impressions, rule, table[rule.combine])
    return CharacterParams(parts=parts)
