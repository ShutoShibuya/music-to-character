"""体の描画 (layout / SVG) のテスト。"""

from __future__ import annotations

import colorsys
import re
import xml.etree.ElementTree as ET

import pytest

from musicchar.character import render_svg
from musicchar.character.renderer.layout import compute_body_layout, compute_face_layout
from musicchar.config_loader import load_config
from musicchar.models import CharacterParams, PartParam, RenderError

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_params(cfg, **values: float) -> CharacterParams:
    parts = {}
    for name, rule in cfg.parts.items():
        v = values.get(name, (rule.min + rule.max) / 2)
        t = min(1.0, max(0.0, (v - rule.min) / (rule.max - rule.min)))
        parts[name] = PartParam(name, v, t, ())
    return CharacterParams(parts)


def lightness(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[1]


def body(cfg, **values: float):
    return compute_body_layout(make_params(cfg, **values), cfg.render)


# ---------- layout ----------

def test_width_and_height_scale_with_params(cfg) -> None:
    b = cfg.render.body
    for w in (0.8, 1.0, 1.3):
        assert body(cfg, body_width=w).body.width == pytest.approx(b.width * w)
    for h in (0.7, 1.0, 1.3):
        assert body(cfg, body_height=h).body.height == pytest.approx(b.height * h)


def test_body_is_centered_under_the_head_and_overlaps_it(cfg) -> None:
    g = body(cfg).body
    h = cfg.render.head
    assert g.x + g.width / 2 == pytest.approx(h.cx)
    assert g.y == pytest.approx(h.cy + h.height / 2 - cfg.render.body.top_overlap)
    assert g.y < h.cy + h.height / 2  # 頭の下端より上から始まる(重なる)


def test_body_height_changes_only_the_bottom_edge(cfg) -> None:
    a, b = body(cfg, body_height=0.7).body, body(cfg, body_height=1.3).body
    assert a.y == b.y and b.y + b.height > a.y + a.height


def test_extreme_bodies_fit_in_the_canvas(cfg) -> None:
    """どんなパーツ値でもキャラクターがキャンバスからはみ出さない(設定の整合性)。"""
    c = cfg.render.canvas
    for rule_name in ("body_width", "body_height"):
        rule = cfg.parts[rule_name]
        for v in (rule.min, rule.max):
            g = body(cfg, **{rule_name: v}).body
            assert g.x >= 0 and g.x + g.width <= c.width
            assert g.y + g.height <= c.height
    head = compute_face_layout(make_params(cfg), cfg.render).head
    assert head.y >= 0 and head.x >= 0 and head.x + head.width <= c.width


def test_body_color_matches_head_hue_and_is_slightly_darker(cfg) -> None:
    p = make_params(cfg, body_lightness=0.7, body_saturation=0.6)
    face = compute_face_layout(p, cfg.render)
    b = compute_body_layout(p, cfg.render)
    hue = lambda c: colorsys.rgb_to_hls(*(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)))[0]  # noqa: E731
    assert hue(b.fill) == pytest.approx(hue(face.skin_fill), abs=0.01)
    assert lightness(b.fill) < lightness(face.skin_fill)
    assert lightness(b.stroke) < lightness(b.fill)


def test_body_color_follows_lightness_and_saturation(cfg) -> None:
    assert lightness(body(cfg, body_lightness=0.9).fill) > lightness(body(cfg, body_lightness=0.25).fill) + 0.4
    gray = body(cfg, body_saturation=0.0).fill
    assert gray[1:3] == gray[3:5] == gray[5:7]


def test_texture_roughness_follows_param(cfg) -> None:
    b = cfg.render.body
    smooth, rough = body(cfg, body_texture=0.0).texture, body(cfg, body_texture=1.0).texture
    assert smooth.opacity == 0.0 and rough.opacity == pytest.approx(b.texture_max_opacity)
    assert smooth.frequency == pytest.approx(b.texture_freq_min) and rough.frequency == pytest.approx(b.texture_freq_max)
    assert rough.seed == smooth.seed == int(b.texture_seed)  # シードは固定
    mid = body(cfg, body_texture=0.5).texture
    assert smooth.opacity < mid.opacity < rough.opacity and smooth.frequency < mid.frequency < rough.frequency


def test_grain_contrasts_with_body_lightness(cfg) -> None:
    dark = body(cfg, body_lightness=0.25, body_texture=1.0)
    light = body(cfg, body_lightness=0.9, body_texture=1.0)
    assert lightness(dark.texture.color) > lightness(dark.fill)   # 暗い体には明るい粒
    assert lightness(light.texture.color) < lightness(light.fill)  # 明るい体には暗い粒


def test_body_layout_requires_its_parts(cfg) -> None:
    params = make_params(cfg)
    for name in ("body_width", "body_height", "body_texture"):
        partial = CharacterParams({k: v for k, v in params.parts.items() if k != name})
        with pytest.raises(RenderError, match=name):
            compute_body_layout(partial, cfg.render)


# ---------- SVG ----------

def body_group(cfg, **values: float) -> ET.Element:
    root = ET.fromstring(render_svg(make_params(cfg, **values), cfg.render))
    return next(el for el in root if el.get("id") == "body")


def test_smooth_body_has_no_texture_elements(cfg) -> None:
    g = body_group(cfg, body_texture=0.0)
    assert [el.tag for el in g] == [SVG + "rect"]
    assert g.find(f".//{SVG}filter") is None


def test_rough_body_has_clipped_noise_overlay(cfg) -> None:
    g = body_group(cfg, body_texture=1.0)
    noise = g.find(f".//{SVG}feTurbulence")
    assert noise is not None
    assert noise.get("type") == "fractalNoise"
    assert int(noise.get("seed")) == int(cfg.render.body.texture_seed)
    assert float(noise.get("baseFrequency")) == pytest.approx(cfg.render.body.texture_freq_max, abs=1e-3)
    assert g.find(f".//{SVG}clipPath[@id='body-clip']") is not None
    overlay = g.find(f".//{SVG}rect[@id='body-texture']")
    assert overlay.get("filter") == "url(#body-grain)" and overlay.get("clip-path") == "url(#body-clip)"
    assert float(overlay.get("opacity")) == pytest.approx(cfg.render.body.texture_max_opacity)


def test_texture_definitions_stay_inside_the_body_group(cfg) -> None:
    root = ET.fromstring(render_svg(make_params(cfg, body_texture=1.0), cfg.render))
    assert [el.get("id") for el in root] == [
        "background", "effects", "body", "head", "cheeks", "eyes", "brows", "mouth", "decorations",
    ]
    for el in root:
        if el.get("id") != "body":
            assert el.find(f".//{SVG}filter") is None


def test_body_is_drawn_behind_the_head(cfg) -> None:
    ids = [el.get("id") for el in ET.fromstring(render_svg(make_params(cfg), cfg.render))]
    assert ids.index("body") < ids.index("head")


def test_texture_is_deterministic(cfg) -> None:
    p = make_params(cfg, body_texture=0.73, body_width=1.1)
    assert render_svg(p, cfg.render) == render_svg(p, cfg.render)


def test_grain_color_is_written_into_the_filter(cfg) -> None:
    g = body_group(cfg, body_texture=1.0, body_lightness=0.25)
    values = g.find(f".//{SVG}feColorMatrix").get("values")
    r, gg, b = (float(x) for x in re.findall(r"0 0 0 0 ([\d.]+)", values))
    expected = compute_body_layout(make_params(cfg, body_texture=1.0, body_lightness=0.25), cfg.render).texture.color
    assert (round(r * 255), round(gg * 255), round(b * 255)) == tuple(int(expected[i : i + 2], 16) for i in (1, 3, 5))


def test_svg_has_no_nan_inf_for_extreme_bodies(cfg) -> None:
    for w in (0.8, 1.3):
        for h in (0.7, 1.3):
            svg = render_svg(make_params(cfg, body_width=w, body_height=h, body_texture=1.0), cfg.render)
            ET.fromstring(svg)
            assert not re.search(r"nan|inf", svg, re.IGNORECASE)
