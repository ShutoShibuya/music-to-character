"""エフェクト・装飾の描画 (layout / SVG) のテスト。"""

from __future__ import annotations

import itertools
import math
import re
import xml.etree.ElementTree as ET

import pytest

from musicchar.character import render_svg
from musicchar.character.renderer.layout import (
    compute_body_layout,
    compute_extras_layout,
    compute_face_layout,
)
from musicchar.character.renderer.parts.extras import heart_path, sparkle_path, star_points
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


def extras(cfg, **values: float):
    p = make_params(cfg, **values)
    face = compute_face_layout(p, cfg.render)
    body = compute_body_layout(p, cfg.render)
    return compute_extras_layout(p, face, body, cfg.render), face, body


# ---------- 個数と順序 ----------

def test_counts_follow_params(cfg) -> None:
    for n_fx in range(0, 9):
        assert len(extras(cfg, effect_count=n_fx)[0].effects) == n_fx
    for n_dec in range(0, 11):
        assert len(extras(cfg, decoration_amount=n_dec)[0].decorations) == n_dec


def test_counts_are_rounded_half_up_and_clamped(cfg) -> None:
    assert len(extras(cfg, effect_count=2.5)[0].effects) == 3
    assert len(extras(cfg, effect_count=2.4)[0].effects) == 2
    assert len(extras(cfg, effect_count=99)[0].effects) == len(cfg.render.effects.slots)
    assert len(extras(cfg, effect_count=-3)[0].effects) == 0
    assert len(extras(cfg, decoration_amount=99)[0].decorations) == len(cfg.render.decorations.slots)


def test_slots_are_used_from_the_front_in_a_fixed_order(cfg) -> None:
    few, many = extras(cfg, effect_count=3)[0], extras(cfg, effect_count=8)[0]
    assert many.effects[:3] == few.effects
    few_d, many_d = extras(cfg, decoration_amount=4)[0], extras(cfg, decoration_amount=10)[0]
    assert [d.kind for d in many_d.decorations[:4]] == [d.kind for d in few_d.decorations]


def test_same_params_give_identical_layout(cfg) -> None:
    assert extras(cfg, effect_count=5, decoration_amount=7)[0] == extras(cfg, effect_count=5, decoration_amount=7)[0]


# ---------- 位置 ----------

def test_decoration_position_is_resolved_relative_to_head_or_body(cfg) -> None:
    ex, face, body = extras(cfg, decoration_amount=10)
    for slot, deco in zip(cfg.render.decorations.slots, ex.decorations):
        box = face.head if slot.anchor == "head" else body.body
        assert deco.x == pytest.approx(box.x + (slot.fx + 0.5) * box.width)
        assert deco.y == pytest.approx(box.y + slot.fy * box.height)


def test_body_decorations_follow_the_body_size(cfg) -> None:
    for w, h in itertools.product((0.8, 1.3), (0.7, 1.3)):
        ex, _, body = extras(cfg, decoration_amount=10, body_width=w, body_height=h)
        b = body.body
        for slot, deco in zip(cfg.render.decorations.slots, ex.decorations):
            if slot.anchor == "body":
                assert b.x <= deco.x <= b.x + b.width and b.y <= deco.y <= b.y + b.height


def test_head_decorations_stay_in_the_forehead_area(cfg) -> None:
    """頭の飾りは上端付近だけ (目・眉の領域に入らない)。"""
    for slot in cfg.render.decorations.slots:
        if slot.anchor == "head":
            assert slot.fy <= 0.2
        else:
            assert slot.fy >= 0.3  # 体の上端は頭と重なっているので、頭の後ろに隠れない位置


def test_decorations_never_overlap_the_brows_or_eyes(cfg) -> None:
    """目・眉の位置が変わる極端な組み合わせ (2^7 通り) でも、頭の飾りが顔に重ならない。"""
    names = ("eye_height", "eye_brow_distance", "eye_spacing", "brow_shortness",
             "brow_thickness", "brow_angle", "eye_angle")
    head_slots = [s for s in cfg.render.decorations.slots if s.anchor == "head"]
    for combo in itertools.product(*[(cfg.parts[n].min, cfg.parts[n].max) for n in names]):
        ex, face, _ = extras(cfg, decoration_amount=10, **dict(zip(names, combo)))
        heads = [d for d, s in zip(ex.decorations, cfg.render.decorations.slots) if s.anchor == "head"]
        assert len(heads) == len(head_slots)
        for d in heads:
            for brow in (face.brow_left, face.brow_right):
                r = brow.length / 2 + brow.thickness / 2  # 回転しても収まる円
                assert math.hypot(d.x - brow.cx, d.y - brow.cy) > r + d.size, (combo, d.kind)
            for eye in (face.eye_left, face.eye_right):
                r = max(eye.rx, eye.ry)
                assert math.hypot(d.x - eye.cx, d.y - eye.cy) > r + d.size, (combo, d.kind)


def test_effects_do_not_overlap_the_character_even_with_the_largest_body(cfg) -> None:
    c = cfg.render.canvas
    ex, face, body = extras(cfg, effect_count=8, body_width=cfg.parts["body_width"].max,
                            body_height=cfg.parts["body_height"].max)
    boxes = [(face.head.x, face.head.y, face.head.width, face.head.height),
             (body.body.x, body.body.y, body.body.width, body.body.height)]
    assert len(ex.effects) == 8
    for e in ex.effects:
        assert e.size <= e.x <= c.width - e.size and e.size <= e.y <= c.height - e.size  # キャンバス内
        for x, y, w, h in boxes:
            overlaps = (e.x + e.size > x and e.x - e.size < x + w and e.y + e.size > y and e.y - e.size < y + h)
            assert not overlaps, (e.kind, e.x, e.y)


def test_extras_layout_requires_its_parts(cfg) -> None:
    p = make_params(cfg)
    face, body = compute_face_layout(p, cfg.render), compute_body_layout(p, cfg.render)
    for name in ("effect_count", "decoration_amount"):
        partial = CharacterParams({k: v for k, v in p.parts.items() if k != name})
        with pytest.raises(RenderError, match=name):
            compute_extras_layout(partial, face, body, cfg.render)


# ---------- 図形 ----------

def test_star_points_are_a_pointing_up_five_pointed_star() -> None:
    pts = [tuple(float(v) for v in p.split(",")) for p in star_points(100, 100, 10).split()]
    assert len(pts) == 10
    assert pts[0] == pytest.approx((100, 90), abs=0.01)  # 先頭の頂点は真上
    radii = [math.hypot(x - 100, y - 100) for x, y in pts]
    assert radii[0::2] == pytest.approx([10.0] * 5, abs=0.02) and radii[1::2] == pytest.approx([4.5] * 5, abs=0.02)


def test_sparkle_and_heart_paths() -> None:
    sp = sparkle_path(50, 50, 10)
    assert sp.count("Q") == 4 and sp.startswith("M 50 40") and sp.endswith("Z")
    assert heart_path(50, 50, 10).startswith("M 50 59") and heart_path(50, 50, 10).count("C") == 2


# ---------- SVG ----------

def children(cfg, **values: float) -> dict[str, ET.Element]:
    root = ET.fromstring(render_svg(make_params(cfg, **values), cfg.render))
    return {el.get("id"): el for el in root}


def test_layer_order(cfg) -> None:
    root = ET.fromstring(render_svg(make_params(cfg), cfg.render))
    assert [el.get("id") for el in root] == [
        "background", "effects", "body", "head", "cheeks", "eyes", "brows", "mouth", "decorations",
    ]


def test_empty_groups_exist_when_count_is_zero(cfg) -> None:
    kids = children(cfg, effect_count=0, decoration_amount=0)
    assert len(kids["effects"]) == 0 and len(kids["decorations"]) == 0


def test_elements_match_counts_and_have_ids(cfg) -> None:
    kids = children(cfg, effect_count=5, decoration_amount=8)
    assert [g.get("id") for g in kids["effects"]] == [f"effect-{i}" for i in range(1, 6)]
    assert [g.get("id") for g in kids["decorations"]] == [f"decoration-{i}" for i in range(1, 9)]
    assert [g.get("class") for g in kids["decorations"]][:4] == ["ribbon", "button", "button", "star"]


def test_every_count_produces_valid_svg(cfg) -> None:
    for n_fx, n_dec in itertools.product((0, 4, 8), (0, 5, 10)):
        svg = render_svg(make_params(cfg, effect_count=n_fx, decoration_amount=n_dec), cfg.render)
        ET.fromstring(svg)
        assert not re.search(r"nan|inf", svg, re.IGNORECASE)


def test_decoration_colors_come_from_accents(cfg) -> None:
    kids = children(cfg, decoration_amount=10)
    fills = {el.get("fill") for el in kids["decorations"].iter() if el.get("fill")}
    allowed = {cfg.render.decorations.accent_1, cfg.render.decorations.accent_2, cfg.render.decorations.stroke}
    assert fills <= allowed


def test_rotation_attribute_only_when_needed(cfg) -> None:
    kids = children(cfg, decoration_amount=10)
    ribbon, button = kids["decorations"][0], kids["decorations"][1]
    assert ribbon.get("transform", "").startswith("rotate(-20 ")  # スロットの rot=-20
    assert button.get("transform") is None  # rot=0 は出力しない


def test_svg_is_deterministic_with_extras(cfg) -> None:
    p = make_params(cfg, effect_count=6, decoration_amount=9, body_width=1.1)
    assert render_svg(p, cfg.render) == render_svg(p, cfg.render)
