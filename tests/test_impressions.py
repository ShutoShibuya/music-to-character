"""impressions (正規化・計算式・統括) のテスト。"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from musicchar.config_loader import NORM_METHODS, load_config
from musicchar.impressions import (
    NORMALIZERS,
    compute_impressions,
    impressions_from_features,
    normalize_features,
    resolve_source,
    weighted_mean_formula,
)
from musicchar.models import (
    Articulation,
    Chords,
    Dynamics,
    FeatureSet,
    ImpressionError,
    ImpressionInput,
    ImpressionRule,
    KeyMode,
    Melody,
    NormRule,
    PitchCenter,
    PitchRange,
    Tempo,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def rule(inputs: dict[str, tuple[float, str]], iid: int = 1) -> ImpressionRule:
    return ImpressionRule(iid, "左", "右", {k: ImpressionInput(w, d) for k, (w, d) in inputs.items()})


# ---------- 正規化 ----------

def test_minmax_math_and_clipping() -> None:
    r = NormRule("b_tempo.bpm", "minmax", 60, 180)
    fs = lambda bpm: FeatureSet(b_tempo=Tempo(bpm))  # noqa: E731
    assert normalize_features(fs(120), {"bpm": r})["bpm"] == pytest.approx(0.5)
    assert normalize_features(fs(60), {"bpm": r})["bpm"] == 0.0
    assert normalize_features(fs(30), {"bpm": r})["bpm"] == 0.0     # 下限でクリップ
    assert normalize_features(fs(500), {"bpm": r})["bpm"] == 1.0    # 上限でクリップ


def test_missing_features_stay_none_not_zero() -> None:
    r = NormRule("b_tempo.bpm", "minmax", 60, 180)
    assert normalize_features(FeatureSet(), {"bpm": r})["bpm"] is None


def test_nested_none_propagates() -> None:
    """KeyMode はあるが mode が不明 -> major_score が None -> 正規化も None。"""
    r = NormRule("a_key.major_score", "minmax", 0, 1)
    assert normalize_features(FeatureSet(a_key=KeyMode(None, None)), {"m": r})["m"] is None
    assert normalize_features(FeatureSet(a_key=KeyMode("C", "major", 0.9)), {"m": r})["m"] == 1.0
    assert normalize_features(FeatureSet(a_key=KeyMode("A", "minor", 0.9)), {"m": r})["m"] == 0.0


def test_non_finite_value_becomes_none() -> None:
    fs = FeatureSet(b_tempo=SimpleNamespace(bpm=float("nan")))  # type: ignore[arg-type]
    assert normalize_features(fs, {"bpm": NormRule("b_tempo.bpm", "minmax", 60, 180)})["bpm"] is None


def test_bad_source_or_method_is_a_config_error() -> None:
    with pytest.raises(ImpressionError, match="存在しません"):
        resolve_source(FeatureSet(b_tempo=Tempo(100)), "b_tempo.nope")
    with pytest.raises(ImpressionError, match="数値ではありません"):
        resolve_source(FeatureSet(g_chords=Chords({}, ("C",), 0, 0, 0)), "g_chords.progression")
    with pytest.raises(ImpressionError, match="未対応"):
        normalize_features(FeatureSet(b_tempo=Tempo(100)), {"x": NormRule("b_tempo.bpm", "zscore", 0, 1)})


def test_every_config_method_is_implemented() -> None:
    assert set(NORM_METHODS) <= set(NORMALIZERS)


def test_default_config_normalizes_full_featureset(cfg, full_featureset) -> None:
    norm = normalize_features(full_featureset, cfg.normalization)
    assert set(norm) == set(cfg.normalization)
    assert all(v is not None and 0.0 <= v <= 1.0 for v in norm.values())
    assert norm["bpm"] == pytest.approx(0.5)  # 120 BPM は 60〜180 の中央


def test_default_config_on_empty_featureset_is_all_none(cfg) -> None:
    assert all(v is None for v in normalize_features(FeatureSet(), cfg.normalization).values())


# ---------- 計算式 ----------

def test_weighted_mean_with_directions_by_hand() -> None:
    r = rule({"a": (2.0, "direct"), "b": (1.0, "inverse")})
    value, missing = weighted_mean_formula({"a": 0.8, "b": 0.3}, r)
    assert value == pytest.approx((2 * 0.8 + 1 * (1 - 0.3)) / 3)
    assert missing == ()


def test_missing_input_is_dropped_and_weights_renormalized() -> None:
    r = rule({"a": (2.0, "direct"), "b": (1.0, "inverse")})
    value, missing = weighted_mean_formula({"a": 0.8, "b": None}, r)
    assert value == pytest.approx(0.8)  # b を 0 や 0.5 として混ぜない
    assert missing == ("b",)


def test_all_inputs_missing_gives_none() -> None:
    r = rule({"a": (1.0, "direct"), "b": (1.0, "inverse")})
    assert weighted_mean_formula({"a": None, "b": None}, r) == (None, ("a", "b"))


def test_formula_rejects_undefined_metric() -> None:
    with pytest.raises(ImpressionError, match="未定義"):
        weighted_mean_formula({"a": 0.5}, rule({"zzz": (1.0, "direct")}))


def test_formula_result_stays_in_unit_range() -> None:
    r = rule({"a": (1.0, "direct"), "b": (3.0, "inverse")})
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            value, _ = weighted_mean_formula({"a": a, "b": b}, r)
            assert 0.0 <= value <= 1.0


# ---------- 統括 ----------

def test_compute_impressions_full(cfg, full_featureset) -> None:
    imp = impressions_from_features(full_featureset, cfg.normalization, cfg.impressions)
    assert set(imp.values) == set(range(1, 9))
    assert all(v is not None and 0.0 <= v <= 1.0 for v in imp.values.values())
    assert imp.missing_inputs == {}


def test_empty_features_give_all_none_with_missing_recorded(cfg) -> None:
    imp = impressions_from_features(FeatureSet(), cfg.normalization, cfg.impressions)
    assert all(v is None for v in imp.values.values())
    for iid, rule_ in cfg.impressions.items():
        assert set(imp.missing_inputs[iid]) == set(rule_.inputs)


def test_partial_failure_only_affects_dependent_impressions(cfg, full_featureset) -> None:
    """テンポが取れなくても、印象1 は残りの入力で計算され、他の印象は影響を受けない。"""
    full = impressions_from_features(full_featureset, cfg.normalization, cfg.impressions)
    no_tempo = impressions_from_features(
        dataclasses.replace(full_featureset, b_tempo=None), cfg.normalization, cfg.impressions
    )
    assert no_tempo.missing_inputs == {1: ("bpm",)}
    assert no_tempo.get(1) is not None and no_tempo.get(1) != full.get(1)
    for iid in range(2, 9):
        assert no_tempo.get(iid) == pytest.approx(full.get(iid))


def test_impression_becomes_none_only_when_all_its_inputs_fail(cfg, full_featureset) -> None:
    fs = dataclasses.replace(full_featureset, a_key=None, g_chords=None)  # 印象2 は A・G だけを使う
    imp = impressions_from_features(fs, cfg.normalization, cfg.impressions)
    assert imp.get(2) is None
    assert imp.get(8) is not None  # 印象8 は H が残っている
    assert set(imp.missing_inputs[8]) == {"chord_complexity", "chord_change_rate"}


def test_compute_impressions_requires_all_eight_rules(cfg) -> None:
    with pytest.raises(ValueError):
        compute_impressions({}, {1: cfg.impressions[1]})


# ---------- 方向の意味 (仕様の左端=0.0 / 右端=1.0) ----------

def variants(fs: FeatureSet, **changes) -> FeatureSet:
    return dataclasses.replace(fs, **changes)


def imp_of(cfg, fs: FeatureSet):
    return impressions_from_features(fs, cfg.normalization, cfg.impressions)


def test_impression1_intense_vs_calm(cfg, full_featureset) -> None:
    intense = variants(
        full_featureset, b_tempo=Tempo(180), f_dynamics=Dynamics(0.3, 0.1, 1.0),
        h_articulation=Articulation(25, 8, 0.1, 0.0, 0.0),
    )
    calm = variants(
        full_featureset, b_tempo=Tempo(60), f_dynamics=Dynamics(0.01, 0.01, 0.1),
        h_articulation=Articulation(2, 0.5, 1.5, 1.0, 0.0),
    )
    assert imp_of(cfg, intense).get(1) < 0.1 and imp_of(cfg, calm).get(1) > 0.9


def test_impression2_bright_vs_dark(cfg, full_featureset) -> None:
    bright = variants(full_featureset, a_key=KeyMode("C", "major", 0.9),
                      g_chords=Chords({"maj": 4}, ("C",), 0.5, 0.3, 0.0))
    dark = variants(full_featureset, a_key=KeyMode("A", "minor", 0.9),
                    g_chords=Chords({"min": 4}, ("Am",), 0.5, 0.3, 1.0))
    assert imp_of(cfg, bright).get(2) == pytest.approx(0.0) and imp_of(cfg, dark).get(2) == pytest.approx(1.0)


def test_impression4_light_vs_heavy(cfg, full_featureset) -> None:
    light = variants(full_featureset, d_pitch_center=PitchCenter(84, 84))
    heavy = variants(full_featureset, d_pitch_center=PitchCenter(36, 36))
    assert imp_of(cfg, light).get(4) < imp_of(cfg, heavy).get(4) - 0.4


def test_impression5_grand_vs_small(cfg, full_featureset) -> None:
    grand = variants(full_featureset, d_pitch_center=PitchCenter(36, 36), f_dynamics=Dynamics(0.3, 0.05, 1.0))
    small = variants(full_featureset, d_pitch_center=PitchCenter(84, 84), f_dynamics=Dynamics(0.01, 0.05, 0.1))
    assert imp_of(cfg, grand).get(5) == pytest.approx(0.0) and imp_of(cfg, small).get(5) == pytest.approx(1.0)


def test_impression6_emotional_vs_flat(cfg, full_featureset) -> None:
    emotional = variants(full_featureset, f_dynamics=Dynamics(0.1, 0.15, 0.4),
                         c_melody=Melody(8.0, 7.0, 0.3, 0.5, 0.5))
    flat = variants(full_featureset, f_dynamics=Dynamics(0.1, 0.005, 0.4),
                    c_melody=Melody(0.5, 0.0, 0.3, 0.5, 0.5))
    assert imp_of(cfg, emotional).get(6) == pytest.approx(0.0) and imp_of(cfg, flat).get(6) == pytest.approx(1.0)


def test_impression7_simple_vs_complex(cfg, full_featureset) -> None:
    simple = variants(full_featureset, c_melody=Melody(0.5, 1.0, 0.0, 1.0, 0.8),
                      e_pitch_range=PitchRange(60, 54, 6))
    complex_ = variants(full_featureset, c_melody=Melody(8.0, 5.0, 0.5, 0.0, 0.1),
                        e_pitch_range=PitchRange(100, 52, 48))
    assert imp_of(cfg, simple).get(7) < 0.2 and imp_of(cfg, complex_).get(7) > 0.8


def test_impression8_stylish_vs_plain(cfg, full_featureset) -> None:
    stylish = variants(full_featureset, g_chords=Chords({"maj7": 4}, ("Cmaj7",), 2.0, 1.0, 0.0),
                       h_articulation=Articulation(2, 3, 0.4, 1.0, 1.0))
    plain = variants(full_featureset, g_chords=Chords({"maj": 4}, ("C",), 0.1, 0.0, 0.0),
                     h_articulation=Articulation(2, 3, 0.4, 0.0, 0.0))
    assert imp_of(cfg, stylish).get(8) == pytest.approx(0.0) and imp_of(cfg, plain).get(8) == pytest.approx(1.0)


# ---------- 独立性 (式・重み・正規化を他モジュールと無関係に変えられる) ----------

def test_weights_in_config_change_the_result(cfg, full_featureset) -> None:
    base = imp_of(cfg, full_featureset).get(1)
    r1 = cfg.impressions[1]
    reweighted = dataclasses.replace(
        r1, inputs={**r1.inputs, "bpm": dataclasses.replace(r1.inputs["bpm"], weight=100.0)}
    )
    rules = {**cfg.impressions, 1: reweighted}
    changed = impressions_from_features(full_featureset, cfg.normalization, rules).get(1)
    bpm_only = 1 - normalize_features(full_featureset, cfg.normalization)["bpm"]
    assert changed != pytest.approx(base)
    assert changed == pytest.approx(bpm_only, abs=0.02)  # bpm の重みが支配的になる


def test_normalization_range_in_config_changes_the_result(cfg, full_featureset) -> None:
    wider = {**cfg.normalization, "bpm": dataclasses.replace(cfg.normalization["bpm"], min=0, max=240)}
    assert normalize_features(full_featureset, wider)["bpm"] == pytest.approx(0.5)
    assert normalize_features(full_featureset, cfg.normalization)["bpm"] == pytest.approx(0.5)  # 120 は両方の中央
    wider = {**cfg.normalization, "bpm": dataclasses.replace(cfg.normalization["bpm"], min=0, max=480)}
    assert normalize_features(full_featureset, wider)["bpm"] == pytest.approx(0.25)


def test_custom_formula_can_be_injected(cfg, full_featureset) -> None:
    def constant_formula(normalized, rule_):
        return 0.25, ()

    imp = impressions_from_features(full_featureset, cfg.normalization, cfg.impressions, constant_formula)
    assert all(v == 0.25 for v in imp.values.values())


def test_impressions_package_depends_only_on_models() -> None:
    """audio / features / character に依存しないこと(独立して変更できる設計の担保)。"""
    pkg = Path(__file__).resolve().parents[1] / "musicchar" / "impressions"
    for path in pkg.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if name.startswith("musicchar"):
                    assert name.startswith(("musicchar.models", "musicchar.impressions")), (path.name, name)
