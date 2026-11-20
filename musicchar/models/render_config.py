"""描画レイアウト設定 (render.yaml) のデータ構造。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _positive(obj: object, *names: str) -> None:
    for n in names:
        v = getattr(obj, n)
        _require(math.isfinite(v) and v > 0, f"{type(obj).__name__}.{n} must be positive")


@dataclass(frozen=True)
class CanvasConfig:
    """キャンバス。background は #RRGGBB。"""

    width: float
    height: float
    background: str

    def __post_init__(self) -> None:
        _positive(self, "width", "height")
        _require(bool(_HEX_COLOR.match(self.background)), "CanvasConfig.background must be #RRGGBB")


@dataclass(frozen=True)
class HeadConfig:
    """頭(輪郭)の基準配置。"""

    cx: float
    cy: float
    width: float
    height: float
    min_corner: float
    stroke_width: float
    stroke_lightness_delta: float

    def __post_init__(self) -> None:
        _positive(self, "width", "height", "stroke_width")
        _require(self.min_corner >= 0, "HeadConfig.min_corner must be >= 0")
        _require(0.0 <= self.stroke_lightness_delta <= 1.0, "stroke_lightness_delta must be within [0, 1]")


@dataclass(frozen=True)
class ColorsConfig:
    """色。hue_deg 以外は #RRGGBB。"""

    hue_deg: float
    eye: str
    highlight: str
    brow: str
    mouth_line: str
    mouth_inside: str
    cheek: str

    def __post_init__(self) -> None:
        _require(0.0 <= self.hue_deg < 360.0, "ColorsConfig.hue_deg must be within [0, 360)")
        for f in fields(self):
            if f.name != "hue_deg":
                _require(bool(_HEX_COLOR.match(getattr(self, f.name))), f"ColorsConfig.{f.name} must be #RRGGBB")


@dataclass(frozen=True)
class EyesConfig:
    """目の基準配置・大きさ。"""

    y_offset: float
    spacing: float
    rx: float
    ry: float
    highlight_max_r: float
    highlight_dx: float
    highlight_dy: float

    def __post_init__(self) -> None:
        _positive(self, "spacing", "rx", "ry", "highlight_max_r")


@dataclass(frozen=True)
class BrowsConfig:
    """眉の基準配置・大きさ。"""

    gap: float
    length: float
    thickness: float
    max_shortening: float

    def __post_init__(self) -> None:
        _positive(self, "length", "thickness")
        _require(self.gap >= 0, "BrowsConfig.gap must be >= 0")
        _require(0.0 <= self.max_shortening < 1.0, "BrowsConfig.max_shortening must be within [0, 1)")


@dataclass(frozen=True)
class MouthConfig:
    """口の基準配置・大きさ。"""

    y_offset: float
    half_width: float
    curve_depth: float
    max_open: float
    line_width: float

    def __post_init__(self) -> None:
        _positive(self, "half_width", "line_width")
        _require(self.curve_depth >= 0 and self.max_open >= 0, "MouthConfig depth/open must be >= 0")


@dataclass(frozen=True)
class CheeksConfig:
    """ほっぺの基準配置・大きさ。"""

    y_offset: float
    x_offset: float
    rx: float
    ry: float
    max_opacity: float

    def __post_init__(self) -> None:
        _positive(self, "rx", "ry")
        _require(0.0 <= self.max_opacity <= 1.0, "CheeksConfig.max_opacity must be within [0, 1]")


@dataclass(frozen=True)
class BodyConfig:
    """体の基準配置・大きさ・質感。"""

    top_overlap: float
    width: float
    height: float
    corner_ratio: float
    lightness_shift: float
    stroke_width: float
    texture_max_opacity: float
    texture_contrast: float
    texture_freq_min: float
    texture_freq_max: float
    texture_octaves: float
    texture_seed: float

    def __post_init__(self) -> None:
        _positive(self, "width", "height", "stroke_width", "texture_freq_min", "texture_octaves")
        _require(self.top_overlap >= 0, "BodyConfig.top_overlap must be >= 0")
        _require(0.0 < self.corner_ratio <= 0.5, "BodyConfig.corner_ratio must be within (0, 0.5]")
        _require(-1.0 <= self.lightness_shift <= 1.0, "BodyConfig.lightness_shift must be within [-1, 1]")
        _require(0.0 <= self.texture_max_opacity <= 1.0, "BodyConfig.texture_max_opacity must be within [0, 1]")
        _require(0.0 < self.texture_contrast <= 1.0, "BodyConfig.texture_contrast must be within (0, 1]")
        _require(self.texture_freq_min < self.texture_freq_max, "BodyConfig: texture_freq_min < texture_freq_max")
        _require(float(self.texture_octaves).is_integer() and self.texture_octaves <= 8, "texture_octaves must be an integer <= 8")
        _require(float(self.texture_seed).is_integer(), "BodyConfig.texture_seed must be an integer")


EFFECT_KINDS = ("sparkle", "dot")
DECORATION_KINDS = ("ribbon", "button", "dot", "star", "heart")
DECORATION_ANCHORS = ("head", "body")


@dataclass(frozen=True)
class EffectSlot:
    """エフェクト 1 個の固定位置。x, y はキャンバス座標、rot は度。"""

    kind: str
    x: float
    y: float
    size: float
    rot: float

    def __post_init__(self) -> None:
        _require(self.kind in EFFECT_KINDS, f"EffectSlot.kind must be one of {EFFECT_KINDS}")
        _positive(self, "size")


@dataclass(frozen=True)
class EffectsConfig:
    """エフェクトの設定。slots の先頭から ``effect_count`` 個を使う。"""

    color: str
    slots: tuple[EffectSlot, ...]

    def __post_init__(self) -> None:
        _require(bool(_HEX_COLOR.match(self.color)), "EffectsConfig.color must be #RRGGBB")
        _require(len(self.slots) > 0, "EffectsConfig.slots must not be empty")


@dataclass(frozen=True)
class DecorationSlot:
    """装飾 1 個の固定位置。fx, fy は anchor (頭または体) の大きさに対する相対座標。"""

    kind: str
    anchor: str
    fx: float
    fy: float
    size: float
    rot: float
    accent: int

    def __post_init__(self) -> None:
        _require(self.kind in DECORATION_KINDS, f"DecorationSlot.kind must be one of {DECORATION_KINDS}")
        _require(self.anchor in DECORATION_ANCHORS, f"DecorationSlot.anchor must be one of {DECORATION_ANCHORS}")
        _require(-0.5 <= self.fx <= 0.5, "DecorationSlot.fx must be within [-0.5, 0.5]")
        _require(0.0 <= self.fy <= 1.0, "DecorationSlot.fy must be within [0, 1]")
        _require(self.accent in (1, 2), "DecorationSlot.accent must be 1 or 2")
        _positive(self, "size")


@dataclass(frozen=True)
class DecorationsConfig:
    """装飾の設定。slots の先頭から ``decoration_amount`` 個を使う。"""

    accent_1: str
    accent_2: str
    stroke: str
    stroke_width: float
    slots: tuple[DecorationSlot, ...]

    def __post_init__(self) -> None:
        for name in ("accent_1", "accent_2", "stroke"):
            _require(bool(_HEX_COLOR.match(getattr(self, name))), f"DecorationsConfig.{name} must be #RRGGBB")
        _positive(self, "stroke_width")
        _require(len(self.slots) > 0, "DecorationsConfig.slots must not be empty")


@dataclass(frozen=True)
class RenderConfig:
    """描画レイアウト設定の全体。"""

    canvas: CanvasConfig
    head: HeadConfig
    colors: ColorsConfig
    eyes: EyesConfig
    brows: BrowsConfig
    mouth: MouthConfig
    cheeks: CheeksConfig
    body: BodyConfig
    effects: EffectsConfig
    decorations: DecorationsConfig
