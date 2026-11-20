"""印象値のデータ構造。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

IMPRESSION_IDS = tuple(range(1, 9))


@dataclass(frozen=True)
class ImpressionSet:
    """印象1〜8の値。

    値は 0.0(左端の印象)〜1.0(右端の印象)。計算不能な印象は None。

    Attributes:
        values: 印象ID(1〜8) -> 値 または None。
        missing_inputs: 印象ID -> 欠損していた入力特徴量名。
            欠損があっても利用可能な入力で再計算した印象は、ここに記録される。
    """

    values: dict[int, Optional[float]]
    missing_inputs: dict[int, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.values) != set(IMPRESSION_IDS):
            raise ValueError("values must contain exactly impression IDs 1..8")
        for i, v in self.values.items():
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"impression {i} must be within [0, 1], got {v}")

    def get(self, impression_id: int) -> Optional[float]:
        """指定した印象の値を返す。計算不能なら None。"""
        return self.values[impression_id]
