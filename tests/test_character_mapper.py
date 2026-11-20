"""character の合成戦略とマッパーのテスト。"""

from __future__ import annotations

import ast
import dataclasses
import random
from pathlib import Path

import pytest

from musicchar.character import COMBINERS, MeanCombiner, WeightedMeanCombiner, map_part, map_parts
from musicchar.config_loader import load_config
from musicchar.models import ConfigError, ImpressionSet, PartRule, PartSource


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def imps(default: float | None = 0.5, **overrides: float | None) -> ImpressionSet:
    """全印象を default にし、i1=..., i2=... で個別に上書きする。"""
    values = {i: default for i in range(1, 9)}
    values.update({int(k[1:]): v for k, v in overrides.items()})
    return ImpressionSet(values=values)


def part_rule(sources, combine="mean", weights=None, lo=0.0, hi=10.0, integer=False) -> PartRule:
    return PartRule("p", "face", "unit", tuple(PartSource(i, d) for i, d in sources),
                    combine, weights, lo, hi, integer)


# ---------- 合成戦略 ----------

def test_mean_combiner() -> None:
    assert MeanCombiner().combine([0.2, 0.6, 1.0]) == pytest.approx(0.6)
    assert MeanCombiner().combine([0.2, 0.6], weights=[100, 1]) == pytest.approx(0.4)  # 重みは無視
    with pytest.raises(ValueError):
        MeanCombiner().combine([])


def test_weighted_mean_combiner() -> None:
    assert WeightedMeanCombiner().combine([0.2, 0.6], [3, 1]) == pytest.approx((0.6 + 0.6) / 4)
    for bad in (None, [1.0], [1.0, 0.0], [1.0, -1.0]):
        with pytest.raises(ValueError):
            WeightedMeanCombiner().combine([0.2, 0.6], bad)
    with pytest.raises(ValueError):
        WeightedMeanCombiner().combine([], [])


def test_registry_has_both_strategies() -> None:
    assert set(COMBINERS) == {"mean", "weighted_mean"}


# ---------- 1 パーツの変換 ----------

def test_positive_and_negative_direction() -> None:
    r_pos = part_rule([(1, "positive")], lo=10, hi=20)
    r_neg = part_rule([(1, "negative")], lo=10, hi=20)
    assert map_part(imps(i1=0.0), r_pos, MeanCombiner()).value == pytest.approx(10)
    assert map_part(imps(i1=1.0), r_pos, MeanCombiner()).value == pytest.approx(20)
    assert map_part(imps(i1=0.0), r_neg, MeanCombiner()).value == pytest.approx(20)
    assert map_part(imps(i1=1.0), r_neg, MeanCombiner()).value == pytest.approx(10)
    assert map_part(imps(i1=0.25), r_neg, MeanCombiner()).value == pytest.approx(17.5)


def test_multiple_sources_use_simple_mean_after_direction() -> None:
    r = part_rule([(2, "negative"), (6, "positive")], lo=0, hi=100)
    p = map_part(imps(i2=0.0, i6=0.5), r, MeanCombiner())  # (1.0 + 0.5) / 2
    assert p.t == pytest.approx(0.75) and p.value == pytest.approx(75)
    assert p.sources == (2, 6) and p.estimated is False


def test_weighted_mean_strategy_uses_rule_weights() -> None:
    r = part_rule([(2, "positive"), (6, "positive")], combine="weighted_mean", weights=(3.0, 1.0), lo=0, hi=1)
    p = map_part(imps(i2=1.0, i6=0.0), r, WeightedMeanCombiner())
    assert p.t == pytest.approx(0.75)


def test_integer_parts_are_rounded_half_up() -> None:
    r = part_rule([(7, "positive")], lo=0, hi=8, integer=True)
    assert map_part(imps(i7=0.0), r, MeanCombiner()).value == 0.0
    assert map_part(imps(i7=0.3), r, MeanCombiner()).value == 2.0   # 2.4
    assert map_part(imps(i7=0.0625), r, MeanCombiner()).value == 1.0  # 0.5 -> 1 (四捨五入)
    assert map_part(imps(i7=1.0), r, MeanCombiner()).value == 8.0
    assert map_part(imps(i7=0.3), r, MeanCombiner()).t == pytest.approx(0.3)  # t は丸めない


# ---------- 推定不能 (None) の印象 ----------

def test_missing_impression_is_dropped_not_treated_as_zero() -> None:
    r = part_rule([(2, "positive"), (6, "positive")], lo=0, hi=100)
    p = map_part(imps(i2=None, i6=0.8), r, MeanCombiner())
    assert p.t == pytest.approx(0.8)  # 0 を混ぜれば 0.4 になる
    assert p.sources == (6,) and p.estimated is True


def test_weights_are_renormalized_over_available_sources() -> None:
    r = part_rule([(2, "positive"), (6, "positive"), (4, "positive")], combine="weighted_mean",
                  weights=(1.0, 3.0, 6.0), lo=0, hi=1)
    p = map_part(imps(i2=0.0, i6=None, i4=1.0), r, WeightedMeanCombiner())
    assert p.t == pytest.approx(6 / 7) and p.estimated is True


def test_all_sources_missing_gives_neutral_value() -> None:
    r = part_rule([(2, "negative")], lo=10, hi=20)
    p = map_part(imps(i2=None), r, MeanCombiner())
    assert p.t == 0.5 and p.value == pytest.approx(15)
    assert p.sources == () and p.estimated is True


# ---------- 全パーツ ----------

def test_all_parts_are_produced(cfg) -> None:
    out = map_parts(imps(), cfg.parts)
    assert list(out.parts) == list(cfg.parts)


def test_neutral_impressions_give_midpoint_for_every_part(cfg) -> None:
    out = map_parts(imps(0.5), cfg.parts)
    for name, rule in cfg.parts.items():
        mid = (rule.min + rule.max) / 2
        expected = float(int(mid + 0.5)) if rule.integer else mid
        assert out.value(name) == pytest.approx(expected), name
        assert out.parts[name].estimated is False


def test_values_stay_within_min_max_for_random_impressions(cfg) -> None:
    rnd = random.Random(0)
    for _ in range(200):
        out = map_parts(ImpressionSet({i: rnd.random() for i in range(1, 9)}), cfg.parts)
        for name, rule in cfg.parts.items():
            p = out.parts[name]
            assert rule.min - 1e-9 <= p.value <= rule.max + 1e-9, name
            assert 0.0 <= p.t <= 1.0


def test_extremes_follow_the_documented_directions(cfg) -> None:
    """character_parts.yaml のコメントに書いた変化方向が、実際の値に反映されること。"""
    p = cfg.parts
    fierce = map_parts(imps(i1=0.0), p)   # 激しい(0)
    calm = map_parts(imps(i1=1.0), p)     # 穏やか(1)
    assert fierce.value("eye_angle") > calm.value("eye_angle")            # 激しいほどつり目
    assert fierce.value("eye_brow_distance") < calm.value("eye_brow_distance")  # 穏やかなほど眉間が広い

    bright = map_parts(imps(i2=0.0), p)   # 明るい(0)
    dark = map_parts(imps(i2=1.0), p)     # 暗い(1)
    assert bright.value("body_lightness") > dark.value("body_lightness")
    assert bright.value("mouth_corner") > dark.value("mouth_corner")      # 明るいほど口角が上がる
    assert bright.value("eye_highlight") > dark.value("eye_highlight")

    grand = map_parts(imps(i5=0.0), p)    # 壮大(0)
    small = map_parts(imps(i5=1.0), p)    # こじんまり(1)
    assert grand.value("body_height") > small.value("body_height")

    simple = map_parts(imps(i7=0.0), p)
    complex_ = map_parts(imps(i7=1.0), p)
    assert complex_.value("body_texture") > simple.value("body_texture")
    assert complex_.value("effect_count") > simple.value("effect_count")

    stylish = map_parts(imps(i8=0.0), p)  # おしゃれ(0)
    plain = map_parts(imps(i8=1.0), p)    # 素朴(1)
    assert stylish.value("decoration_amount") > plain.value("decoration_amount")


def test_only_parts_depending_on_a_missing_impression_are_estimated(cfg) -> None:
    out = map_parts(imps(i2=None), cfg.parts)
    expected = {n for n, r in cfg.parts.items() if any(s.impression == 2 for s in r.sources)}
    assert {n for n, q in out.parts.items() if q.estimated} == expected
    assert {"body_lightness", "mouth_corner", "cheek_intensity"} <= expected


def test_same_input_gives_identical_output(cfg) -> None:
    a = map_parts(imps(0.37, i3=0.9), cfg.parts)
    b = map_parts(imps(0.37, i3=0.9), cfg.parts)
    assert a == b


def test_switching_a_part_to_weighted_mean_via_config(cfg) -> None:
    rule = cfg.parts["cheek_intensity"]  # sources: 印象2(negative), 印象6(negative)
    weighted = dataclasses.replace(rule, combine="weighted_mean", weights=(3.0, 1.0))
    rules = {**cfg.parts, "cheek_intensity": weighted}
    i = imps(i2=0.0, i6=1.0)  # 反転後 t = 1.0, 0.0
    assert map_parts(i, cfg.parts).value("cheek_intensity") == pytest.approx(0.5 * (rule.max - rule.min) + rule.min)
    assert map_parts(i, rules).parts["cheek_intensity"].t == pytest.approx(0.75)


def test_custom_combiner_can_be_injected(cfg) -> None:
    class Max:
        def combine(self, values, weights=None):
            return max(values)

    out = map_parts(imps(i2=0.0, i6=1.0), cfg.parts, {"mean": Max(), "weighted_mean": Max()})
    assert out.parts["cheek_intensity"].t == 1.0  # 反転後 (1.0, 0.0) の最大


def test_unknown_combine_name_is_a_config_error(cfg) -> None:
    bad = {"x": dataclasses.replace(cfg.parts["eye_angle"], combine="median")}
    with pytest.raises(ConfigError, match="未対応"):
        map_parts(imps(), bad)


def test_character_package_depends_only_on_models() -> None:
    pkg = Path(__file__).resolve().parents[1] / "musicchar" / "character"
    for path in pkg.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if name.startswith("musicchar"):
                    assert name.startswith(("musicchar.models", "musicchar.character")), (path.name, name)
