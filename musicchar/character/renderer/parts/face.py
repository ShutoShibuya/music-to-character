"""顔の各パーツの SVG 断片。各関数は 1 つのパーツだけを描き、互いに独立している。"""

from __future__ import annotations

from musicchar.character.renderer.layout import (
    BrowGeom,
    CheekGeom,
    EyeGeom,
    FaceLayout,
    MouthGeom,
)
from musicchar.models.render_config import ColorsConfig, HeadConfig


def n(x: float) -> str:
    """SVG 用の数値表記 (小数 2 桁、末尾の 0 と ``-0`` を除く)。"""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def render_head(layout: FaceLayout, head_cfg: HeadConfig) -> str:
    """頭(輪郭)。"""
    h = layout.head
    return (
        f'<g id="head"><rect x="{n(h.x)}" y="{n(h.y)}" width="{n(h.width)}" height="{n(h.height)}" '
        f'rx="{n(h.corner)}" ry="{n(h.corner)}" fill="{layout.skin_fill}" '
        f'stroke="{layout.skin_stroke}" stroke-width="{n(head_cfg.stroke_width)}"/></g>'
    )


def _cheek(c: CheekGeom, side: str, color: str) -> str:
    return (
        f'<ellipse id="cheek-{side}" cx="{n(c.cx)}" cy="{n(c.cy)}" rx="{n(c.rx)}" ry="{n(c.ry)}" '
        f'fill="{color}" fill-opacity="{n(c.opacity)}"/>'
    )


def render_cheeks(layout: FaceLayout, colors: ColorsConfig) -> str:
    """ほっぺ(左右)。"""
    return (
        '<g id="cheeks">'
        + _cheek(layout.cheek_left, "left", colors.cheek)
        + _cheek(layout.cheek_right, "right", colors.cheek)
        + "</g>"
    )


def _eye(e: EyeGeom, side: str, colors: ColorsConfig) -> str:
    out = (
        f'<g id="eye-{side}" transform="rotate({n(e.rot)} {n(e.cx)} {n(e.cy)})">'
        f'<ellipse cx="{n(e.cx)}" cy="{n(e.cy)}" rx="{n(e.rx)}" ry="{n(e.ry)}" fill="{colors.eye}"/>'
    )
    if e.highlight_r >= 0.05:
        out += (
            f'<circle class="highlight" cx="{n(e.highlight_cx)}" cy="{n(e.highlight_cy)}" '
            f'r="{n(e.highlight_r)}" fill="{colors.highlight}"/>'
        )
    return out + "</g>"


def render_eyes(layout: FaceLayout, colors: ColorsConfig) -> str:
    """目(左右)。ハイライトは目と一緒に回転する。"""
    return (
        '<g id="eyes">'
        + _eye(layout.eye_left, "left", colors)
        + _eye(layout.eye_right, "right", colors)
        + "</g>"
    )


def _brow(b: BrowGeom, side: str, colors: ColorsConfig) -> str:
    x1, x2 = b.cx - b.length / 2, b.cx + b.length / 2
    return (
        f'<line id="brow-{side}" x1="{n(x1)}" y1="{n(b.cy)}" x2="{n(x2)}" y2="{n(b.cy)}" '
        f'stroke="{colors.brow}" stroke-width="{n(b.thickness)}" stroke-linecap="round" '
        f'transform="rotate({n(b.rot)} {n(b.cx)} {n(b.cy)})"/>'
    )


def render_brows(layout: FaceLayout, colors: ColorsConfig) -> str:
    """眉(左右)。"""
    return (
        '<g id="brows">'
        + _brow(layout.brow_left, "left", colors)
        + _brow(layout.brow_right, "right", colors)
        + "</g>"
    )


def mouth_path(m: MouthGeom) -> str:
    """口の輪郭パス。

    上唇は左右の口角を結ぶ 2 次ベジェ、下唇は同じ口角を結んで中央が ``open_h`` だけ
    下に開く 2 次ベジェ。``curve`` > 0 で口角が中央より上がる(笑顔)。``open_h`` = 0 なら
    上唇と下唇が重なり、ただの線になる。
    """
    left, right = m.cx - m.half_width, m.cx + m.half_width
    upper_ctrl = m.cy + m.curve
    lower_ctrl = m.cy + m.curve + 2.0 * m.open_h  # 2 次ベジェの中点は制御点の半分だけ動く
    return (
        f"M {n(left)} {n(m.cy)} Q {n(m.cx)} {n(upper_ctrl)} {n(right)} {n(m.cy)} "
        f"Q {n(m.cx)} {n(lower_ctrl)} {n(left)} {n(m.cy)} Z"
    )


def render_mouth(layout: FaceLayout, colors: ColorsConfig, line_width: float) -> str:
    """口(口角の上下と開き具合)。"""
    return (
        f'<g id="mouth"><path d="{mouth_path(layout.mouth)}" fill="{colors.mouth_inside}" '
        f'stroke="{colors.mouth_line}" stroke-width="{n(line_width)}" stroke-linejoin="round" '
        f'stroke-linecap="round"/></g>'
    )
