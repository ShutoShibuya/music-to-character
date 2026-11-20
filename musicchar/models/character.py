"""キャラクターのパラメータのデータ構造。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartParam:
    """1つのパーツの数値パラメータ。

    Attributes:
        name: パーツ名 (character_parts.yaml のキー)。
        value: 描画に使う実値 (単位は設定ファイルの unit)。
        t: min〜max 内の位置 (0〜1)。
        sources: 値の計算に実際に使われた印象ID。すべて推定不能で中立値にした場合は空。
        estimated: 一部または全部の印象が推定不能 (None) で、利用できる印象だけ
            (または中立値 0.5) で計算した場合 True。
    """

    name: str
    value: float
    t: float
    sources: tuple[int, ...]
    estimated: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.t <= 1.0:
            raise ValueError(f"t must be within [0, 1], got {self.t}")


@dataclass(frozen=True)
class CharacterParams:
    """キャラクター全パーツのパラメータ。parts: パーツ名 -> PartParam。"""

    parts: dict[str, PartParam]

    def value(self, name: str) -> float:
        """パーツ名から実値を返す。未定義なら KeyError。"""
        return self.parts[name].value
