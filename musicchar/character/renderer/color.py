"""色の変換。"""

from __future__ import annotations

import colorsys


def hsl_to_hex(hue_deg: float, saturation: float, lightness: float) -> str:
    """HSL (色相は度, 彩度・明度は 0〜1) を ``#rrggbb`` に変換する。範囲外は 0〜1 にクリップする。"""
    s = min(1.0, max(0.0, saturation))
    l = min(1.0, max(0.0, lightness))
    r, g, b = colorsys.hls_to_rgb((hue_deg % 360.0) / 360.0, l, s)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))
