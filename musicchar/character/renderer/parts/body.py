"""体の SVG 断片。"""

from __future__ import annotations

from musicchar.character.renderer.layout import BodyLayout
from musicchar.character.renderer.parts.face import n

# ざらつきの最小の濃さ。これ未満なら見えないので重ね塗りを出力しない。
_MIN_TEXTURE_OPACITY = 0.005



def _grain_matrix(color: str) -> str:
    """fractalNoise の R チャンネルを、指定色の粒のアルファに変換する feColorMatrix の値。

    alpha = 4 * R - 1.6 (R の平均 0.5 で約 0.4、明るい部分ほど濃くなり、コントラストの高い粒状になる)。
    """
    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return f"0 0 0 0 {r:.3f}  0 0 0 0 {g:.3f}  0 0 0 0 {b:.3f}  4 0 0 0 -1.6"


def render_body(layout: BodyLayout) -> str:
    """体。形・色に加えて、``texture.opacity`` > 0 ならノイズによるざらつきを重ねる。

    ざらつきは ``feTurbulence`` (シード固定) のノイズを体の形で切り抜いたもので、
    同じパラメータなら常に同じ見た目になる。
    """
    b, t = layout.body, layout.texture
    geometry = (
        f'x="{n(b.x)}" y="{n(b.y)}" width="{n(b.width)}" height="{n(b.height)}" '
        f'rx="{n(b.corner)}" ry="{n(b.corner)}"'
    )
    out = (
        f'<g id="body"><rect {geometry} fill="{layout.fill}" stroke="{layout.stroke}" '
        f'stroke-width="{n(layout.stroke_width)}"/>'
    )
    if t.opacity >= _MIN_TEXTURE_OPACITY:
        out += (
            '<defs>'
            f'<clipPath id="body-clip"><rect {geometry}/></clipPath>'
            f'<filter id="body-grain" filterUnits="userSpaceOnUse" x="{n(b.x)}" y="{n(b.y)}" '
            f'width="{n(b.width)}" height="{n(b.height)}" color-interpolation-filters="sRGB">'
            f'<feTurbulence type="fractalNoise" baseFrequency="{t.frequency:.3f}" '
            f'numOctaves="{t.octaves}" seed="{t.seed}" result="noise"/>'
            f'<feColorMatrix in="noise" type="matrix" values="{_grain_matrix(t.color)}"/>'
            '</filter></defs>'
            f'<rect id="body-texture" x="{n(b.x)}" y="{n(b.y)}" width="{n(b.width)}" height="{n(b.height)}" '
            f'filter="url(#body-grain)" clip-path="url(#body-clip)" opacity="{n(t.opacity)}"/>'
        )
    return out + "</g>"
