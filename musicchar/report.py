"""分析結果の表示用データと、JSON 出力用の変換。画面ライブラリには依存しない。"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

from musicchar.models import FeatureSet, ImpressionRule, PartRule
from musicchar.models.character import CharacterParams
from musicchar.models.impressions import ImpressionSet
from musicchar.pipeline import AnalysisResult

# 特徴量フィールド名 -> (グループ名, {属性名: 項目名})。表示専用の文字列。
FEATURE_LABELS: dict[str, tuple[str, dict[str, str]]] = {
    "a_key": ("A 調", {"key": "キー", "mode": "モード", "confidence": "確信度"}),
    "b_tempo": ("B テンポ", {"bpm": "BPM"}),
    "c_melody": ("C メロディ", {
        "pitch_change_rate": "音高の変化 (回/秒)", "mean_interval_semitones": "平均音程 (半音)",
        "leap_ratio": "跳躍の割合", "repetition_score": "旋律の繰り返し", "smoothness": "旋律の滑らかさ",
    }),
    "d_pitch_center": ("D 音域 (高さ)", {"mean_midi": "音高の平均 (MIDI)", "median_midi": "音高の中央値 (MIDI)"}),
    "e_pitch_range": ("E 音域 (幅)", {
        "max_midi": "最大 (MIDI)", "min_midi": "最小 (MIDI)", "range_semitones": "範囲 (半音)",
    }),
    "f_dynamics": ("F 音量", {"rms_mean": "RMS 平均", "rms_std": "RMS 標準偏差", "rms_max": "RMS 最大値"}),
    "g_chords": ("G コード", {
        "chord_types": "コードの種類", "progression": "コード進行",
        "change_rate_per_sec": "変化頻度 (回/秒)", "complexity": "複雑さ", "minor_ratio": "短調系コードの割合",
    }),
    "h_articulation": ("H アーティキュレーション", {
        "onset_strength_mean": "音の立ち上がり", "onset_density": "発音密度 (回/秒)",
        "mean_note_duration": "音の持続時間 (秒)", "overlap_ratio": "音の重なり", "gap_ratio": "音の切れ目",
    }),
}

PROGRESSION_PREVIEW = 16
GROUP_LABELS = {"face": "顔", "body": "体", "other": "その他"}


def format_value(value: Any) -> str:
    """特徴量の値を表示用の文字列にする。"""
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}×{v}" for k, v in value.items())
    if isinstance(value, tuple):
        shown = " → ".join(str(v) for v in value[:PROGRESSION_PREVIEW])
        return shown + (f" … (全 {len(value)} 個)" if len(value) > PROGRESSION_PREVIEW else "")
    return str(value)


def feature_rows(features: FeatureSet) -> list[dict[str, str]]:
    """特徴量 A〜H の表。抽出できなかった特徴量は、理由つきの 1 行にする。"""
    rows: list[dict[str, str]] = []
    for field, (group, items) in FEATURE_LABELS.items():
        obj = getattr(features, field)
        if obj is None:
            reason = features.errors.get(field, "不明")
            rows.append({"特徴量": group, "項目": "-", "値": "抽出できませんでした", "備考": reason})
            continue
        for attr, label in items.items():
            value = getattr(obj, attr)
            rows.append({
                "特徴量": group, "項目": label,
                "値": "判定不能" if value is None else format_value(value), "備考": "",
            })
    return rows


def impression_rows(
    impressions: ImpressionSet, rules: Mapping[int, ImpressionRule]
) -> list[dict[str, Any]]:
    """印象 1〜8 の表示データ。value は 0〜1 または None (推定不能)。"""
    rows = []
    for iid, rule in rules.items():
        value: Optional[float] = impressions.get(iid)
        missing = impressions.missing_inputs.get(iid, ())
        if value is None:
            note = "推定不能 (入力となる特徴量がすべて欠損)"
        elif missing:
            note = f"一部の入力が欠損: {', '.join(missing)}"
        else:
            note = ""
        rows.append({
            "id": iid, "left": rule.label_low, "right": rule.label_high, "value": value, "note": note,
        })
    return rows


def part_rows(character: CharacterParams, rules: Mapping[str, PartRule]) -> list[dict[str, Any]]:
    """キャラクターのパーツ値の表。"""
    rows = []
    for name, rule in rules.items():
        p = character.parts[name]
        rows.append({
            "グループ": GROUP_LABELS.get(rule.group, rule.group),
            "パーツ": rule.label or name,
            "値": round(p.value, 3),
            "単位": rule.unit,
            "位置 (0〜1)": round(p.t, 3),
            "使った印象": ", ".join(str(i) for i in p.sources) or "なし (中立値)",
            "推定": "推定不能の印象あり" if p.estimated else "",
        })
    return rows


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """分析結果を JSON にできる辞書に変換する (特徴量・印象値・パーツのパラメータ)。"""
    features = {
        f.name: (None if getattr(result.features, f.name) is None else dataclasses.asdict(getattr(result.features, f.name)))
        for f in dataclasses.fields(result.features) if f.name != "errors"
    }
    return {
        "source": result.source,
        "original_duration_sec": result.original_duration,
        "analyzed_duration_sec": result.analyzed_duration,
        "features": features,
        "feature_errors": dict(result.features.errors),
        "impressions": {str(i): v for i, v in result.impressions.values.items()},
        "impression_missing_inputs": {str(i): list(m) for i, m in result.impressions.missing_inputs.items()},
        "character_parts": {
            name: {"value": p.value, "t": p.t, "sources": list(p.sources), "estimated": p.estimated}
            for name, p in result.character.parts.items()
        },
    }
