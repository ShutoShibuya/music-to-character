"""複数の印象を 1 つの値にまとめる合成戦略。

初期実装は単純平均 (``MeanCombiner``)。重み付き平均 (``WeightedMeanCombiner``) も用意してあり、
パーツごとに ``character_parts.yaml`` の ``combine`` で切り替える。
新しい戦略は ``Combiner`` を実装して ``COMBINERS`` に登録する。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Protocol


class Combiner(Protocol):
    """0〜1 の値の列を 0〜1 の 1 つの値にまとめる。"""

    def combine(self, values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
        """合成した値を返す。

        Args:
            values: 合成する値 (1 個以上、各 0〜1)。
            weights: ``values`` と同順の重み。戦略によっては無視される。
        """
        ...


class MeanCombiner:
    """単純平均。重みは無視する。"""

    def combine(self, values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
        """値の算術平均を返す。

        Raises:
            ValueError: ``values`` が空の場合。
        """
        if not values:
            raise ValueError("values must not be empty")
        return sum(values) / len(values)


class WeightedMeanCombiner:
    """重み付き平均。"""

    def combine(self, values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
        """``sum(w * v) / sum(w)`` を返す。

        Raises:
            ValueError: ``values`` が空、``weights`` が未指定・個数不一致、または重みが正でない場合。
        """
        if not values:
            raise ValueError("values must not be empty")
        if weights is None or len(weights) != len(values):
            raise ValueError("weights must be given with the same length as values")
        if any(w <= 0 for w in weights):
            raise ValueError("weights must be positive")
        return sum(w * v for w, v in zip(weights, values)) / sum(weights)


COMBINERS: dict[str, Combiner] = {
    "mean": MeanCombiner(),
    "weighted_mean": WeightedMeanCombiner(),
}
