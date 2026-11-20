"""report (表示用データ・JSON 変換) のテスト。"""

from __future__ import annotations

import dataclasses
import json

import pytest

from musicchar.models import (
    Chords,
    FeatureSet,
    ImpressionSet,
    KeyMode,
    PartParam,
    CharacterParams,
    Tempo,
)
from musicchar.report import (
    FEATURE_LABELS,
    PROGRESSION_PREVIEW,
    feature_rows,
    format_value,
    impression_rows,
    part_rows,
    result_to_dict,
)


def test_feature_labels_cover_every_feature_field(full_featureset) -> None:
    """特徴量の項目を増やしたときに、表示ラベルの付け忘れを検出する。"""
    not_in_table = {"rms_curve", "rms_curve_hop_sec"}  # 時間変化は表ではなくグラフで表示
    for fname, (_, items) in FEATURE_LABELS.items():
        obj = getattr(full_featureset, fname)
        fields = {f.name for f in dataclasses.fields(obj)} - not_in_table
        assert fields == set(items), fname
    assert set(FEATURE_LABELS) == {f.name for f in dataclasses.fields(full_featureset)} - {"errors"}


def test_format_value() -> None:
    assert format_value(123.456789) == "123"
    assert format_value(0.123456) == "0.123"
    assert format_value("C") == "C"
    assert format_value({"maj": 3, "min": 1}) == "maj×3, min×1"
    assert format_value(("C", "G")) == "C → G"
    long = tuple(str(i) for i in range(PROGRESSION_PREVIEW + 4))
    assert format_value(long).endswith(f"… (全 {len(long)} 個)")


def test_feature_rows_for_a_full_featureset(full_featureset) -> None:
    rows = feature_rows(full_featureset)
    groups = {r["特徴量"] for r in rows}
    assert len(groups) == 8
    bpm = next(r for r in rows if r["項目"] == "BPM")
    assert bpm["値"] == "120" and bpm["備考"] == ""
    prog = next(r for r in rows if r["項目"] == "コード進行")
    assert prog["値"] == "C → Am → F → G"
    assert all(r["値"] != "抽出できませんでした" for r in rows)


def test_failed_feature_is_one_row_with_the_reason() -> None:
    fs = FeatureSet(b_tempo=Tempo(100), errors={"a_key": "調の確信度が低い"})
    rows = feature_rows(fs)
    a_rows = [r for r in rows if r["特徴量"].startswith("A")]
    assert a_rows == [{"特徴量": "A 調", "項目": "-", "値": "抽出できませんでした", "備考": "調の確信度が低い"}]
    c_rows = [r for r in rows if r["特徴量"].startswith("C")]
    assert c_rows[0]["備考"] == "不明"  # 理由が記録されていない場合


def test_unknown_mode_is_shown_as_undeterminable() -> None:
    rows = feature_rows(FeatureSet(a_key=KeyMode(None, None, 0.0)))
    mode_row = next(r for r in rows if r["項目"] == "モード")
    assert mode_row["値"] == "判定不能"


def test_impression_rows(app_config) -> None:
    values = {i: 0.5 for i in range(1, 9)}
    values[2] = None
    imp = ImpressionSet(values, missing_inputs={2: ("major_score", "chord_minor_ratio"), 5: ("rms_max",)})
    rows = {r["id"]: r for r in impression_rows(imp, app_config.impressions)}
    assert len(rows) == 8 and rows[1]["left"] == "激しい" and rows[1]["right"] == "穏やか"
    assert rows[1]["value"] == 0.5 and rows[1]["note"] == ""
    assert rows[2]["value"] is None and "推定不能" in rows[2]["note"]
    assert rows[5]["value"] == 0.5 and "rms_max" in rows[5]["note"]


def test_part_rows(app_config) -> None:
    parts = {
        name: PartParam(name, rule.min, 0.0, (1,), estimated=(name == "eye_angle"))
        for name, rule in app_config.parts.items()
    }
    parts["mouth_corner"] = PartParam("mouth_corner", 0.0, 0.5, (), estimated=True)
    rows = part_rows(CharacterParams(parts), app_config.parts)
    assert len(rows) == len(app_config.parts) == 19
    assert all(r["パーツ"] for r in rows) and "eye_angle" not in {r["パーツ"] for r in rows}  # 日本語名で表示
    by_label = {r["パーツ"]: r for r in rows}
    assert by_label["目の角度"]["推定"] == "推定不能の印象あり"
    assert by_label["口角"]["使った印象"] == "なし (中立値)"
    assert by_label["眉の太さ"]["推定"] == ""
    assert {r["グループ"] for r in rows} == {"顔", "体", "その他"}


def test_result_to_dict_is_json_serializable(sample_result) -> None:
    d = result_to_dict(sample_result)
    text = json.dumps(d, ensure_ascii=False)  # NaN や tuple の問題があれば例外になる
    back = json.loads(text)
    assert back["source"] == "sample.wav"
    assert set(back["features"]) == {f.name for f in dataclasses.fields(FeatureSet)} - {"errors"}
    assert back["features"]["g_chords"] is None and "g_chords" in back["feature_errors"]
    assert back["features"]["b_tempo"]["bpm"] == pytest.approx(sample_result.features.b_tempo.bpm)
    assert set(back["impressions"]) == {str(i) for i in range(1, 9)}
    assert len(back["character_parts"]) == 19
    assert "NaN" not in text and "Infinity" not in text


def test_result_to_dict_marks_missing_impressions_as_null(sample_result) -> None:
    from musicchar.pipeline import AnalysisResult

    values = dict(sample_result.impressions.values)
    values[3] = None
    changed = dataclasses.replace(
        sample_result, impressions=ImpressionSet(values, sample_result.impressions.missing_inputs)
    )
    assert isinstance(changed, AnalysisResult)
    assert result_to_dict(changed)["impressions"]["3"] is None
