"""config_loader のテスト(同梱のデフォルト設定の整合性も検証する)。"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from musicchar.config_loader import (
    DEFAULT_CONFIG_DIR,
    load_config,
    parse_audio,
    parse_features,
    parse_impressions,
    parse_normalization,
    parse_parts,
    parse_decorations,
    parse_effects,
    parse_render,
)
from musicchar.models import AppConfig, ConfigError, FeatureSet

# 仕様書: 印象ごとに使用してよい特徴量(A〜H)
SPEC_IMPRESSION_FEATURES = {
    1: "bfh", 2: "ag", 3: "cg", 4: "dh", 5: "df", 6: "cf", 7: "ce", 8: "gh",
}

# 仕様書: パーツごとに使用する印象
SPEC_PART_IMPRESSIONS = {
    "eye_brow_distance": {1}, "cheek_intensity": {2, 6}, "eye_angle": {1},
    "brow_angle": {1, 3}, "brow_thickness": {1, 4, 8}, "brow_shortness": {8},
    "mouth_corner": {2}, "eye_highlight": {2, 6}, "eye_height": {2, 6},
    "eye_spacing": {4, 6}, "outline_roundness": {1, 3},
    "body_lightness": {2}, "body_saturation": {4}, "body_width": {4, 8},
    "body_height": {5}, "body_texture": {7},
    "mouth_openness": {6}, "effect_count": {7, 8}, "decoration_amount": {8},
}


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config()


def _load(name: str) -> dict[str, Any]:
    with open(DEFAULT_CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- デフォルト設定の整合性 ----------

def test_default_config_loads(cfg: AppConfig) -> None:
    assert cfg.audio.sample_rate > 0
    assert set(cfg.impressions) == set(range(1, 9))


def test_normalization_sources_resolve_on_featureset(
    cfg: AppConfig, full_featureset: FeatureSet
) -> None:
    """source の属性パスが実在し、値が取れること。"""
    for name, rule in cfg.normalization.items():
        obj: Any = full_featureset
        for attr in rule.source.split("."):
            assert hasattr(obj, attr), f"{name}: {rule.source} の {attr} が存在しない"
            obj = getattr(obj, attr)
        assert isinstance(obj, (int, float)), f"{name}: {rule.source} が数値でない"


def test_impressions_use_only_spec_feature_groups(cfg: AppConfig) -> None:
    """各印象が、仕様書で指定された特徴量グループ(A〜H)だけを使うこと。"""
    for iid, rule in cfg.impressions.items():
        for fname in rule.inputs:
            group = cfg.normalization[fname].source[0]  # "a_key.xxx" -> "a"
            assert group in SPEC_IMPRESSION_FEATURES[iid], (
                f"印象{iid} が仕様外の特徴量グループ {group} ({fname}) を使用"
            )


def test_impressions_cover_all_spec_groups(cfg: AppConfig) -> None:
    """各印象が、仕様書で指定されたグループをすべて使っていること。"""
    for iid, rule in cfg.impressions.items():
        used = {cfg.normalization[f].source[0] for f in rule.inputs}
        assert used == set(SPEC_IMPRESSION_FEATURES[iid]), f"印象{iid}: {used}"


def test_parts_match_spec(cfg: AppConfig) -> None:
    assert set(cfg.parts) == set(SPEC_PART_IMPRESSIONS)
    for name, rule in cfg.parts.items():
        assert {s.impression for s in rule.sources} == SPEC_PART_IMPRESSIONS[name], name


def test_initial_parts_use_simple_mean(cfg: AppConfig) -> None:
    assert all(r.combine == "mean" for r in cfg.parts.values())


# ---------- 検証エラー ----------

def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="見つかりません"):
        load_config(tmp_path)


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    for f in DEFAULT_CONFIG_DIR.glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "audio.yaml").write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="構文"):
        load_config(tmp_path)


def test_audio_unknown_key_and_bad_range() -> None:
    data = _load("audio.yaml")
    with pytest.raises(ConfigError, match="未知"):
        parse_audio({**data, "typo": 1})
    with pytest.raises(ConfigError, match="min_duration_sec"):
        parse_audio({**data, "min_duration_sec": 999})
    with pytest.raises(ConfigError, match="数値"):
        parse_audio({**data, "sample_rate": "high"})


def test_audio_pitch_settings_validation() -> None:
    data = _load("audio.yaml")
    for key, bad in [("pitch_max_transition_rate", 0), ("pitch_min_voiced_ratio", 1.5),
                     ("pitch_range_percentile", 50), ("pitch_min_note_frames", 0)]:
        with pytest.raises(ConfigError, match=key):
            parse_audio({**data, key: bad})


def test_normalization_errors() -> None:
    data = _load("normalization.yaml")
    bad = copy.deepcopy(data)
    bad["features"]["bpm"]["min"], bad["features"]["bpm"]["max"] = 180, 60
    with pytest.raises(ConfigError, match="min < max"):
        parse_normalization(bad)
    bad = copy.deepcopy(data)
    bad["features"]["bpm"]["method"] = "zscore"
    with pytest.raises(ConfigError, match="method"):
        parse_normalization(bad)


def test_impression_references_unknown_feature() -> None:
    norm = parse_normalization(_load("normalization.yaml"))
    data = copy.deepcopy(_load("impressions.yaml"))
    data["impressions"][1]["inputs"]["no_such_feature"] = {"weight": 1, "direction": "direct"}
    with pytest.raises(ConfigError, match="存在しない特徴量"):
        parse_impressions(data, norm)


def test_impression_bad_weight_direction_and_ids() -> None:
    norm = parse_normalization(_load("normalization.yaml"))
    base = _load("impressions.yaml")
    data = copy.deepcopy(base)
    data["impressions"][1]["inputs"]["bpm"]["weight"] = 0
    with pytest.raises(ConfigError, match="weight"):
        parse_impressions(data, norm)
    data = copy.deepcopy(base)
    data["impressions"][1]["inputs"]["bpm"]["direction"] = "up"
    with pytest.raises(ConfigError, match="direction"):
        parse_impressions(data, norm)
    data = copy.deepcopy(base)
    del data["impressions"][8]
    with pytest.raises(ConfigError, match="ID"):
        parse_impressions(data, norm)


def test_parts_errors() -> None:
    base = _load("character_parts.yaml")

    data = copy.deepcopy(base)
    data["parts"]["eye_angle"]["min"], data["parts"]["eye_angle"]["max"] = 5, 5
    with pytest.raises(ConfigError, match="min < max"):
        parse_parts(data)

    data = copy.deepcopy(base)
    data["parts"]["eye_angle"]["sources"][0]["impression"] = 9
    with pytest.raises(ConfigError, match="impression"):
        parse_parts(data)

    data = copy.deepcopy(base)
    data["parts"]["eye_angle"]["combine"] = "weighted_mean"
    with pytest.raises(ConfigError, match="weights"):
        parse_parts(data)

    data = copy.deepcopy(base)
    data["parts"]["cheek_intensity"]["weights"] = [1.0]  # sources は2個
    with pytest.raises(ConfigError, match="同じ個数"):
        parse_parts(data)

    data = copy.deepcopy(base)
    data["parts"]["cheek_intensity"]["sources"][1]["impression"] = 2  # 重複
    with pytest.raises(ConfigError, match="重複"):
        parse_parts(data)


def test_weighted_mean_is_accepted_when_weights_given() -> None:
    data = copy.deepcopy(_load("character_parts.yaml"))
    data["parts"]["cheek_intensity"]["combine"] = "weighted_mean"
    data["parts"]["cheek_intensity"]["weights"] = [2.0, 1.0]
    rule = parse_parts(data)["cheek_intensity"]
    assert rule.combine == "weighted_mean"
    assert rule.weights == (2.0, 1.0)


def test_features_config_errors() -> None:
    base = _load("features.yaml")
    assert parse_features(base).melody.ngram_length >= 2
    for key, bad in [("ngram_length", 1), ("max_gap_sec", -1),
                     ("leap_threshold_semitones", 0), ("smoothness_ref_semitones", 0), ("min_notes", 2)]:
        data = copy.deepcopy(base)
        data["melody"][key] = bad
        with pytest.raises(ConfigError, match=key):
            parse_features(data)
    data = copy.deepcopy(base)
    data["melody"]["typo"] = 1
    with pytest.raises(ConfigError, match="未知"):
        parse_features(data)


def test_features_chord_key_chroma_errors() -> None:
    base = _load("features.yaml")
    cases = [
        (("key", "min_correlation"), 1.0, "min_correlation"),
        (("chord", "window_sec"), 0, "window_sec"),
        (("chord", "switch_penalty"), -1, "switch_penalty"),
        (("chord", "min_similarity"), 1.5, "min_similarity"),
        (("chord", "complexity", "type_ref"), 1, "type_ref"),
        (("chord", "complexity", "extension"), -1, "重み"),
        (("chroma", "hpss"), "yes", "hpss"),
    ]
    for path, bad, match in cases:
        data = copy.deepcopy(base)
        node = data
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = bad
        with pytest.raises(ConfigError, match=match):
            parse_features(data)


def test_features_articulation_errors() -> None:
    base = _load("features.yaml")
    for key, bad in [("decay_threshold_db", 0), ("gap_threshold_db", 3), ("min_onsets", 1),
                     ("onset_context_frames", 0), ("onset_decay_ratio", 1.0)]:
        data = copy.deepcopy(base)
        data["articulation"][key] = bad
        with pytest.raises(ConfigError, match=key):
            parse_features(data)


def test_render_config_loads(cfg: AppConfig) -> None:
    assert cfg.render.canvas.width > 0 and 0 <= cfg.render.colors.hue_deg < 360


def test_render_config_errors() -> None:
    base = _load("render.yaml")
    assert parse_render(base).head.width == base["head"]["width"]
    cases = [
        (("colors", "eye"), "red", "#RRGGBB"),
        (("colors", "hue_deg"), 360, "hue_deg"),
        (("canvas", "width"), 0, "width"),
        (("eyes", "rx"), -1, "rx"),
        (("brows", "max_shortening"), 1.0, "max_shortening"),
        (("cheeks", "max_opacity"), 1.5, "max_opacity"),
        (("head", "width"), "wide", "数値"),
        (("body", "corner_ratio"), 0, "corner_ratio"),
        (("body", "texture_octaves"), 2.5, "texture_octaves"),
        (("body", "texture_freq_min"), 5, "texture_freq_min"),
        (("body", "texture_max_opacity"), 2, "texture_max_opacity"),
        (("body", "texture_contrast"), 0, "texture_contrast"),
    ]
    for (section, key), bad, match in cases:
        data = copy.deepcopy(base)
        data[section][key] = bad
        with pytest.raises(ConfigError, match=match):
            parse_render(data)
    data = copy.deepcopy(base)
    del data["mouth"]["half_width"]
    with pytest.raises(ConfigError, match="必須"):
        parse_render(data)
    data = copy.deepcopy(base)
    data["extra"] = {}
    with pytest.raises(ConfigError, match="未知"):
        parse_render(data)


def test_effects_and_decorations_config_errors() -> None:
    base = _load("render.yaml")
    assert len(parse_effects(base["effects"]).slots) == len(base["effects"]["slots"])
    assert parse_decorations(base["decorations"]).slots[0].kind == "ribbon"

    def edit(section, mutate):
        data = copy.deepcopy(base[section])
        mutate(data)
        return data

    with pytest.raises(ConfigError, match="kind"):
        parse_effects(edit("effects", lambda d: d["slots"][0].update(kind="firework")))
    with pytest.raises(ConfigError, match="size"):
        parse_effects(edit("effects", lambda d: d["slots"][0].update(size=0)))
    with pytest.raises(ConfigError, match="空でない"):
        parse_effects(edit("effects", lambda d: d.update(slots=[])))
    with pytest.raises(ConfigError, match="#RRGGBB"):
        parse_effects(edit("effects", lambda d: d.update(color="yellow")))
    with pytest.raises(ConfigError, match="必須"):
        parse_effects(edit("effects", lambda d: d["slots"][0].pop("rot")))
    with pytest.raises(ConfigError, match="fx"):
        parse_decorations(edit("decorations", lambda d: d["slots"][0].update(fx=0.9)))
    with pytest.raises(ConfigError, match="anchor"):
        parse_decorations(edit("decorations", lambda d: d["slots"][0].update(anchor="feet")))
    with pytest.raises(ConfigError, match="accent"):
        parse_decorations(edit("decorations", lambda d: d["slots"][0].update(accent=3)))
    with pytest.raises(ConfigError, match="未知"):
        parse_decorations(edit("decorations", lambda d: d.update(typo=1)))


@pytest.mark.parametrize("section, part", [("effects", "effect_count"), ("decorations", "decoration_amount")])
def test_slot_count_must_cover_the_part_maximum(tmp_path: Path, section: str, part: str) -> None:
    for f in DEFAULT_CONFIG_DIR.glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    render = yaml.safe_load((tmp_path / "render.yaml").read_text(encoding="utf-8"))
    render[section]["slots"] = render[section]["slots"][:-1]  # 最大値より 1 つ足りない
    (tmp_path / "render.yaml").write_text(yaml.safe_dump(render, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match=f"{part}.*足りません|足りません"):
        load_config(tmp_path)
