"""顔の描画 (layout / SVG) のテスト。"""

from __future__ import annotations

import colorsys
import math
import re
import xml.etree.ElementTree as ET

import pytest

from musicchar.character.renderer.color import hsl_to_hex
from musicchar.character.renderer.layout import BODY_PARTS, EXTRA_PARTS, FACE_PARTS, compute_face_layout
from musicchar.character.renderer.parts.face import mouth_path, n
from musicchar.character import render_svg
from musicchar.config_loader import load_config
from musicchar.models import CharacterParams, PartParam, RenderError

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_params(cfg, **values: float) -> CharacterParams:
    """全パーツを中央値で用意し、指定したパーツだけ実値で上書きする。"""
    parts = {}
    for name, rule in cfg.parts.items():
        v = values.get(name, (rule.min + rule.max) / 2)
        t = min(1.0, max(0.0, (v - rule.min) / (rule.max - rule.min)))  # 範囲外の実値も試せるようにクリップ
        parts[name] = PartParam(name, v, t, ())
    return CharacterParams(parts)


def rotate(point: tuple[float, float], center: tuple[float, float], deg: float) -> tuple[float, float]:
    """SVG の rotate(deg cx cy) を点に適用する。"""
    a = math.radians(deg)
    dx, dy = point[0] - center[0], point[1] - center[1]
    return (center[0] + dx * math.cos(a) - dy * math.sin(a), center[1] + dx * math.sin(a) + dy * math.cos(a))


def lightness(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[1]


# ---------- 色 ----------

def test_hsl_to_hex_known_values() -> None:
    assert hsl_to_hex(0, 1, 0.5) == "#ff0000"
    assert hsl_to_hex(120, 1, 0.5) == "#00ff00"
    assert hsl_to_hex(240, 1, 0.5) == "#0000ff"
    assert hsl_to_hex(77, 0, 0.5) == "#808080"
    assert hsl_to_hex(0, 1, 2.0) == "#ffffff" and hsl_to_hex(0, 1, -1.0) == "#000000"  # クリップ
    assert hsl_to_hex(360, 1, 0.5) == hsl_to_hex(0, 1, 0.5)


def test_number_formatting() -> None:
    assert n(1.5) == "1.5" and n(2.0) == "2" and n(-0.001) == "0" and n(0.0) == "0" and n(3.14159) == "3.14"


# ---------- layout: 数値から位置・大きさ・角度・色を制御できる ----------

def test_eyes_are_symmetric_and_spacing_follows_param(cfg) -> None:
    for ratio in (0.8, 1.0, 1.3):
        lay = compute_face_layout(make_params(cfg, eye_spacing=ratio), cfg.render)
        assert lay.eye_right.cx - lay.eye_left.cx == pytest.approx(cfg.render.eyes.spacing * ratio)
        assert (lay.eye_left.cx + lay.eye_right.cx) / 2 == pytest.approx(cfg.render.head.cx)
        assert lay.eye_left.cy == lay.eye_right.cy


def test_eye_height_scales_vertical_radius_only(cfg) -> None:
    small = compute_face_layout(make_params(cfg, eye_height=0.6), cfg.render).eye_left
    big = compute_face_layout(make_params(cfg, eye_height=1.4), cfg.render).eye_left
    assert big.ry == pytest.approx(cfg.render.eyes.ry * 1.4) and small.ry == pytest.approx(cfg.render.eyes.ry * 0.6)
    assert big.rx == small.rx == cfg.render.eyes.rx


def test_positive_eye_angle_raises_the_outer_corners(cfg) -> None:
    """eye_angle > 0 = つり目: 左目の左端も右目の右端も、回転後に中心より上 (y が小さい) になる。"""
    lay = compute_face_layout(make_params(cfg, eye_angle=15), cfg.render)
    el, er = lay.eye_left, lay.eye_right
    _, y_left_outer = rotate((el.cx - el.rx, el.cy), (el.cx, el.cy), el.rot)
    _, y_right_outer = rotate((er.cx + er.rx, er.cy), (er.cx, er.cy), er.rot)
    assert y_left_outer < el.cy and y_right_outer < er.cy
    assert y_left_outer - el.cy == pytest.approx(y_right_outer - er.cy)  # 左右対称

    lay = compute_face_layout(make_params(cfg, eye_angle=-15), cfg.render)  # たれ目
    el = lay.eye_left
    _, y = rotate((el.cx - el.rx, el.cy), (el.cx, el.cy), el.rot)
    assert y > el.cy


def test_positive_brow_angle_lowers_the_inner_ends(cfg) -> None:
    """brow_angle > 0 = 怒り眉: 眉の内側(顔の中心側)の端が、外側より下 (y が大きい) になる。"""
    lay = compute_face_layout(make_params(cfg, brow_angle=25), cfg.render)
    for brow, inner_sign in ((lay.brow_left, +1), (lay.brow_right, -1)):
        c = (brow.cx, brow.cy)
        inner = rotate((brow.cx + inner_sign * brow.length / 2, brow.cy), c, brow.rot)
        outer = rotate((brow.cx - inner_sign * brow.length / 2, brow.cy), c, brow.rot)
        assert inner[1] > outer[1]


def test_brow_distance_shortness_and_thickness(cfg) -> None:
    b = cfg.render.brows
    near = compute_face_layout(make_params(cfg, eye_brow_distance=0.5), cfg.render)
    far = compute_face_layout(make_params(cfg, eye_brow_distance=1.5), cfg.render)
    assert near.brow_left.cy - far.brow_left.cy == pytest.approx(b.gap * 1.0)  # 差は gap × 倍率差
    long_ = compute_face_layout(make_params(cfg, brow_shortness=0.0), cfg.render).brow_left
    short = compute_face_layout(make_params(cfg, brow_shortness=1.0), cfg.render).brow_left
    assert long_.length == pytest.approx(b.length)
    assert short.length == pytest.approx(b.length * (1 - b.max_shortening))
    thick = compute_face_layout(make_params(cfg, brow_thickness=2.0), cfg.render).brow_left
    assert thick.thickness == pytest.approx(b.thickness * 2.0)


def test_brows_stay_above_eyes(cfg) -> None:
    for eh in (0.6, 1.4):
        for d in (0.5, 1.5):
            lay = compute_face_layout(make_params(cfg, eye_height=eh, eye_brow_distance=d), cfg.render)
            assert lay.brow_left.cy < lay.eye_left.cy - lay.eye_left.ry


def test_mouth_corner_and_openness(cfg) -> None:
    m = cfg.render.mouth
    smile = compute_face_layout(make_params(cfg, mouth_corner=1.0), cfg.render).mouth
    frown = compute_face_layout(make_params(cfg, mouth_corner=-1.0), cfg.render).mouth
    assert smile.curve == pytest.approx(m.curve_depth) and frown.curve == pytest.approx(-m.curve_depth)
    wide = compute_face_layout(make_params(cfg, mouth_openness=1.0), cfg.render).mouth
    shut = compute_face_layout(make_params(cfg, mouth_openness=0.0), cfg.render).mouth
    assert wide.open_h == pytest.approx(m.max_open) and shut.open_h == 0.0


def test_cheeks_are_symmetric_with_opacity_from_param(cfg) -> None:
    lay = compute_face_layout(make_params(cfg, cheek_intensity=0.5), cfg.render)
    assert lay.cheek_left.opacity == pytest.approx(0.5 * cfg.render.cheeks.max_opacity)
    assert lay.cheek_left.cx + lay.cheek_right.cx == pytest.approx(2 * cfg.render.head.cx)


def test_head_corner_follows_roundness(cfg) -> None:
    h = cfg.render.head
    square = compute_face_layout(make_params(cfg, outline_roundness=0.0), cfg.render).head
    round_ = compute_face_layout(make_params(cfg, outline_roundness=1.0), cfg.render).head
    assert square.corner == pytest.approx(h.min_corner)
    assert round_.corner == pytest.approx(min(h.width, h.height) / 2)
    assert square.x + square.width / 2 == pytest.approx(h.cx)


def test_skin_color_follows_body_lightness_and_saturation(cfg) -> None:
    dark = compute_face_layout(make_params(cfg, body_lightness=0.25), cfg.render)
    light = compute_face_layout(make_params(cfg, body_lightness=0.9), cfg.render)
    assert lightness(light.skin_fill) > lightness(dark.skin_fill) + 0.4
    gray = compute_face_layout(make_params(cfg, body_saturation=0.0), cfg.render)
    r, g, b = (int(gray.skin_fill[i : i + 2], 16) for i in (1, 3, 5))
    assert r == g == b
    assert lightness(light.skin_stroke) < lightness(light.skin_fill)  # 輪郭線は塗りより暗い


def test_layout_requires_all_face_parts(cfg) -> None:
    params = make_params(cfg)
    for name in FACE_PARTS:
        partial = CharacterParams({k: v for k, v in params.parts.items() if k != name})
        with pytest.raises(RenderError, match=name):
            compute_face_layout(partial, cfg.render)


# ---------- SVG ----------

def parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def groups(svg: str) -> dict[str, str]:
    """トップレベルの要素 id -> その要素の XML 文字列。"""
    return {el.get("id"): ET.tostring(el, encoding="unicode") for el in parse(svg)}


def test_svg_is_well_formed_with_independent_groups(cfg) -> None:
    svg = render_svg(make_params(cfg), cfg.render)
    root = parse(svg)
    assert root.tag == SVG + "svg"
    assert root.get("viewBox") == f"0 0 {n(cfg.render.canvas.width)} {n(cfg.render.canvas.height)}"
    assert list(groups(svg)) == [
        "background", "effects", "body", "head", "cheeks", "eyes", "brows", "mouth", "decorations",
    ]
    ids = {el.get("id") for el in root.iter() if el.get("id")}
    assert {"eye-left", "eye-right", "brow-left", "brow-right", "cheek-left", "cheek-right"} <= ids


def test_same_params_give_identical_svg(cfg) -> None:
    p = make_params(cfg, eye_angle=7.3, mouth_corner=0.4)
    assert render_svg(p, cfg.render) == render_svg(p, cfg.render)


def test_svg_contains_no_nan_or_inf(cfg) -> None:
    svg = render_svg(make_params(cfg), cfg.render)
    assert not re.search(r"nan|inf", svg, re.IGNORECASE)


# 1 つのパーツを変えたときに、見た目が変わってよい要素 (それ以外は完全に同じであること)
AFFECTED = {
    "outline_roundness": {"head"},
    "body_lightness": {"body", "head"},      # 頭と体は同じ色相・彩度で揃える
    "body_saturation": {"body", "head"},
    "body_width": {"body", "decorations"},   # 体に付ける装飾は体の大きさに追従する
    "body_height": {"body", "decorations"},
    "body_texture": {"body"},
    "effect_count": {"effects"},
    "decoration_amount": {"decorations"},
    "cheek_intensity": {"cheeks"},
    "eye_angle": {"eyes"},
    "eye_highlight": {"eyes"},
    "eye_height": {"eyes", "brows"},        # 眉は目の上端を基準に置く
    "eye_spacing": {"eyes", "brows"},       # 眉は目の x に揃える
    "eye_brow_distance": {"brows"},
    "brow_angle": {"brows"},
    "brow_thickness": {"brows"},
    "brow_shortness": {"brows"},
    "mouth_corner": {"mouth"},
    "mouth_openness": {"mouth"},
}


@pytest.mark.parametrize("part", sorted(AFFECTED))
def test_changing_one_part_changes_only_its_elements(cfg, part: str) -> None:
    rule = cfg.parts[part]
    base = groups(render_svg(make_params(cfg), cfg.render))
    changed = groups(render_svg(make_params(cfg, **{part: rule.max}), cfg.render))
    differing = {k for k in base if base[k] != changed[k]}
    assert differing == AFFECTED[part]


def test_every_drawn_part_is_covered_by_the_independence_table() -> None:
    assert set(AFFECTED) == set(FACE_PARTS) | set(BODY_PARTS) | set(EXTRA_PARTS)


def test_highlight_is_omitted_when_zero(cfg) -> None:
    none = render_svg(make_params(cfg, eye_highlight=0.0), cfg.render)
    some = render_svg(make_params(cfg, eye_highlight=1.0), cfg.render)
    assert 'class="highlight"' not in none and some.count('class="highlight"') == 2


def test_rotation_transforms_are_mirrored(cfg) -> None:
    root = parse(render_svg(make_params(cfg, eye_angle=12, brow_angle=20), cfg.render))
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}
    rot = lambda el: float(re.match(r"rotate\((-?[\d.]+) ", el.get("transform")).group(1))  # noqa: E731
    assert rot(by_id["eye-left"]) == -rot(by_id["eye-right"]) == 12
    assert rot(by_id["brow-left"]) == -rot(by_id["brow-right"]) == 20


def test_mouth_path_shape() -> None:
    from musicchar.character.renderer.layout import MouthGeom

    closed = mouth_path(MouthGeom(200, 245, 28, 0.0, 0.0))
    assert closed == "M 172 245 Q 200 245 228 245 Q 200 245 172 245 Z"  # 平らな閉じた口 = 線
    smile_open = mouth_path(MouthGeom(200, 245, 28, 22.0, 10.0))
    assert "Q 200 267 228 245" in smile_open and "Q 200 287 172 245" in smile_open


def test_render_requires_all_parts(cfg) -> None:
    params = make_params(cfg)
    partial = CharacterParams({k: v for k, v in params.parts.items() if k != "mouth_corner"})
    with pytest.raises(RenderError):
        render_svg(partial, cfg.render)
