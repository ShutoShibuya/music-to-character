"""印象の計算式。

計算式は「正規化済み特徴量と印象ルールから (印象値, 欠損した入力) を返す関数」として
差し替えられる。式を変えたいときは、このファイルに関数を追加し、
``compute_impressions(..., formula=...)`` に渡す。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional

from musicchar.models.config import ImpressionRule
from musicchar.models.errors import ImpressionError

# (正規化済み特徴量, 印象ルール) -> (印象値 or None, 欠損していた入力名)
Formula = Callable[[Mapping[str, Optional[float]], ImpressionRule], tuple[Optional[float], tuple[str, ...]]]


def weighted_mean_formula(
    normalized: Mapping[str, Optional[float]], rule: ImpressionRule
) -> tuple[Optional[float], tuple[str, ...]]:
    """向きを考慮した重み付き平均。初期実装の標準式。

    各入力 ``x`` (0〜1) を、``direct`` ならそのまま、``inverse`` なら ``1 - x`` に直し、
    ``weight`` で重み付き平均する。欠損した入力は除き、残りの入力で重みを再正規化する
    (欠損を 0 として扱わない)。利用できる入力が 1 つもなければ印象値は ``None``。

    Args:
        normalized: 正規化指標名 -> 0〜1 の値 または None。
        rule: 印象のルール (入力・重み・向き)。

    Returns:
        (印象値 または None, 欠損していた入力名のタプル)。

    Raises:
        ImpressionError: ルールが参照する指標が ``normalized`` に存在しない場合。
    """
    total = weight_sum = 0.0
    missing: list[str] = []
    for name, spec in rule.inputs.items():
        if name not in normalized:
            raise ImpressionError(f"印象{rule.id}: 未定義の正規化指標を参照しています: {name!r}")
        x = normalized[name]
        if x is None:
            missing.append(name)
            continue
        total += spec.weight * (x if spec.direction == "direct" else 1.0 - x)
        weight_sum += spec.weight
    if weight_sum == 0.0:
        return None, tuple(missing)
    return min(1.0, max(0.0, total / weight_sum)), tuple(missing)
