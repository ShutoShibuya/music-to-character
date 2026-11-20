"""エフェクトと装飾の SVG 断片。"""

from __future__ import annotations

import math

from musicchar.character.renderer.layout import DecorationGeom, EffectGeom, ExtrasLayout
from musicchar.character.renderer.parts.face import n


def sparkle_path(x: float, y: float, s: float) -> str:
    """4 点のきらきら (中心に向かって凹む星)。s は中心から先端までの距離。"""
    return (
        f"M {n(x)} {n(y - s)} Q {n(x)} {n(y)} {n(x + s)} {n(y)} "
        f"Q {n(x)} {n(y)} {n(x)} {n(y + s)} Q {n(x)} {n(y)} {n(x - s)} {n(y)} "
        f"Q {n(x)} {n(y)} {n(x)} {n(y - s)} Z"
    )


def star_points(x: float, y: float, s: float, points: int = 5, inner: float = 0.45) -> str:
    """多角星の頂点列 (``points`` で使う文字列)。先頭の頂点は真上。s は外側の半径。"""
    coords = []
    for i in range(points * 2):
        r = s if i % 2 == 0 else s * inner
        a = -math.pi / 2 + i * math.pi / points
        coords.append(f"{n(x + r * math.cos(a))},{n(y + r * math.sin(a))}")
    return " ".join(coords)


def heart_path(x: float, y: float, s: float) -> str:
    """ハート。(x, y) が中心、s がおおよその半幅。"""
    return (
        f"M {n(x)} {n(y + s * 0.9)} "
        f"C {n(x - s * 1.4)} {n(y + s * 0.1)} {n(x - s * 0.7)} {n(y - s * 0.9)} {n(x)} {n(y - s * 0.35)} "
        f"C {n(x + s * 0.7)} {n(y - s * 0.9)} {n(x + s * 1.4)} {n(y + s * 0.1)} {n(x)} {n(y + s * 0.9)} Z"
    )


def _effect(e: EffectGeom, index: int, color: str) -> str:
    rot = f' transform="rotate({n(e.rot)} {n(e.x)} {n(e.y)})"' if e.rot else ""
    if e.kind == "sparkle":
        shape = f'<path d="{sparkle_path(e.x, e.y, e.size)}" fill="{color}"/>'
    else:  # dot
        shape = f'<circle cx="{n(e.x)}" cy="{n(e.y)}" r="{n(e.size)}" fill="{color}" fill-opacity="0.85"/>'
    return f'<g id="effect-{index}" class="{e.kind}"{rot}>{shape}</g>'


def render_effects(extras: ExtrasLayout) -> str:
    """エフェクト (キャラクターの周りに浮かぶ飾り)。0 個でも空の ``<g>`` を出力する。"""
    inner = "".join(_effect(e, i, extras.effect_color) for i, e in enumerate(extras.effects, 1))
    return f'<g id="effects">{inner}</g>'


def _decoration(d: DecorationGeom, index: int, stroke: str, sw: float) -> str:
    st = f'stroke="{stroke}" stroke-width="{n(sw)}" stroke-linejoin="round"'
    x, y, s = d.x, d.y, d.size
    if d.kind == "ribbon":
        shape = (
            f'<path d="M {n(x)} {n(y)} L {n(x - s)} {n(y - s * 0.6)} L {n(x - s)} {n(y + s * 0.6)} Z '
            f'M {n(x)} {n(y)} L {n(x + s)} {n(y - s * 0.6)} L {n(x + s)} {n(y + s * 0.6)} Z" '
            f'fill="{d.fill}" {st}/>'
            f'<circle cx="{n(x)}" cy="{n(y)}" r="{n(s * 0.28)}" fill="{d.fill}" {st}/>'
        )
    elif d.kind == "button":
        shape = (
            f'<circle cx="{n(x)}" cy="{n(y)}" r="{n(s)}" fill="{d.fill}" {st}/>'
            f'<circle cx="{n(x - s * 0.3)}" cy="{n(y)}" r="{n(s * 0.14)}" fill="{stroke}"/>'
            f'<circle cx="{n(x + s * 0.3)}" cy="{n(y)}" r="{n(s * 0.14)}" fill="{stroke}"/>'
        )
    elif d.kind == "star":
        shape = f'<polygon points="{star_points(x, y, s)}" fill="{d.fill}" {st}/>'
    elif d.kind == "heart":
        shape = f'<path d="{heart_path(x, y, s)}" fill="{d.fill}" {st}/>'
    else:  # dot
        shape = f'<circle cx="{n(x)}" cy="{n(y)}" r="{n(s)}" fill="{d.fill}" {st}/>'
    rot = f' transform="rotate({n(d.rot)} {n(x)} {n(y)})"' if d.rot else ""
    return f'<g id="decoration-{index}" class="{d.kind}"{rot}>{shape}</g>'


def render_decorations(extras: ExtrasLayout) -> str:
    """装飾 (頭や体に付ける飾り)。顔・体より前に描く。0 個でも空の ``<g>`` を出力する。"""
    inner = "".join(
        _decoration(d, i, extras.stroke, extras.stroke_width) for i, d in enumerate(extras.decorations, 1)
    )
    return f'<g id="decorations">{inner}</g>'
