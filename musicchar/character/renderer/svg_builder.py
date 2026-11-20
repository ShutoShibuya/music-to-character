"""キャラクター全体の SVG の組み立て。"""

from __future__ import annotations

from musicchar.character.renderer.layout import (
    compute_body_layout,
    compute_extras_layout,
    compute_face_layout,
)
from musicchar.character.renderer.parts.body import render_body
from musicchar.character.renderer.parts.extras import render_decorations, render_effects
from musicchar.character.renderer.parts.face import (
    n,
    render_brows,
    render_cheeks,
    render_eyes,
    render_head,
    render_mouth,
)
from musicchar.models.character import CharacterParams
from musicchar.models.render_config import RenderConfig


def render_svg(params: CharacterParams, cfg: RenderConfig) -> str:
    """キャラクターの SVG 文字列を生成する。

    各パーツは ``id`` 付きの ``<g>`` として独立しており、描画順は
    エフェクト -> 体 -> 頭 -> ほっぺ -> 目 -> 眉 -> 口 -> 装飾。同じ ``params`` と ``cfg`` からは常に同じ文字列になる。

    Args:
        params: 全パーツの値。
        cfg: 描画レイアウト設定。

    Returns:
        SVG の文字列。

    Raises:
        RenderError: 描画に必要なパーツが ``params`` にない場合。
    """
    layout = compute_face_layout(params, cfg)
    body_layout = compute_body_layout(params, cfg)
    extras = compute_extras_layout(params, layout, body_layout, cfg)
    c = cfg.canvas
    body = (
        render_effects(extras)
        + render_body(body_layout)
        + render_head(layout, cfg.head)
        + render_cheeks(layout, cfg.colors)
        + render_eyes(layout, cfg.colors)
        + render_brows(layout, cfg.colors)
        + render_mouth(layout, cfg.colors, cfg.mouth.line_width)
        + render_decorations(extras)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{n(c.width)}" height="{n(c.height)}" '
        f'viewBox="0 0 {n(c.width)} {n(c.height)}">'
        f'<rect id="background" width="{n(c.width)}" height="{n(c.height)}" fill="{c.background}"/>'
        f"{body}</svg>"
    )
