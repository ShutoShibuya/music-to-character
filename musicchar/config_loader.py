"""YAML 設定ファイルの読込と検証。

設定の誤り(キーの typo、範囲の逆転、存在しない特徴量の参照など)は
実行時に黙って誤動作させず、``ConfigError`` として起動時に検出する。
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from musicchar.models.config import (
    AppConfig,
    ArticulationConfig,
    AudioConfig,
    ChordConfig,
    ChromaConfig,
    FeaturesConfig,
    ImpressionInput,
    ImpressionRule,
    KeyConfig,
    MelodyConfig,
    NormRule,
    PartRule,
    PartSource,
)
from musicchar.models.errors import ConfigError
from musicchar.models.impressions import IMPRESSION_IDS
from musicchar.models.render_config import (
    BodyConfig,
    DecorationSlot,
    DecorationsConfig,
    EffectSlot,
    EffectsConfig,
    BrowsConfig,
    CanvasConfig,
    CheeksConfig,
    ColorsConfig,
    EyesConfig,
    HeadConfig,
    MouthConfig,
    RenderConfig,
)

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

NORM_METHODS = ("minmax",)
IMPRESSION_DIRECTIONS = ("direct", "inverse")
PART_DIRECTIONS = ("positive", "negative")
COMBINE_METHODS = ("mean", "weighted_mean")
PART_GROUPS = ("face", "body", "other")


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """YAML ファイルを読み、トップレベルのマッピングを返す。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"設定ファイルが見つかりません: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML の構文エラー: {path}: {e}") from e
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: トップレベルはマッピングである必要があります")
    return data


def _check_keys(
    m: Any, required: Iterable[str], optional: Iterable[str], where: str
) -> Mapping[str, Any]:
    """マッピングのキーを検証する。不足・未知のキーは ConfigError。"""
    if not isinstance(m, Mapping):
        raise ConfigError(f"{where}: マッピングが必要です")
    required, optional = set(required), set(optional)
    missing = required - set(m)
    unknown = set(m) - required - optional
    if missing:
        raise ConfigError(f"{where}: 必須キーがありません: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{where}: 未知のキーがあります: {sorted(unknown)}")
    return m


def _num(m: Mapping[str, Any], key: str, where: str) -> float:
    """有限の数値(bool 不可)を取り出す。"""
    v = m[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ConfigError(f"{where}.{key}: 有限の数値が必要です (got {v!r})")
    return float(v)


def _choice(m: Mapping[str, Any], key: str, choices: tuple[str, ...], where: str) -> str:
    v = m[key]
    if v not in choices:
        raise ConfigError(f"{where}.{key}: {list(choices)} のいずれかが必要です (got {v!r})")
    return v


def _bool(m: Mapping[str, Any], key: str, where: str) -> bool:
    v = m[key]
    if not isinstance(v, bool):
        raise ConfigError(f"{where}.{key}: true/false が必要です (got {v!r})")
    return v


def _str(m: Mapping[str, Any], key: str, where: str) -> str:
    v = m[key]
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{where}.{key}: 空でない文字列が必要です (got {v!r})")
    return v


def parse_audio(data: Mapping[str, Any]) -> AudioConfig:
    """audio.yaml の内容を AudioConfig に変換する。"""
    keys = ("sample_rate", "min_duration_sec", "max_duration_sec", "trim_top_db",
            "peak_normalize", "fmin_hz", "fmax_hz", "frame_length", "hop_length",
            "pitch_min_voiced_ratio", "pitch_range_percentile",
            "pitch_max_transition_rate", "pitch_min_note_frames")
    m = _check_keys(data, keys, (), "audio")
    cfg = AudioConfig(
        sample_rate=int(_num(m, "sample_rate", "audio")),
        min_duration_sec=_num(m, "min_duration_sec", "audio"),
        max_duration_sec=_num(m, "max_duration_sec", "audio"),
        trim_top_db=_num(m, "trim_top_db", "audio"),
        peak_normalize=_bool(m, "peak_normalize", "audio"),
        fmin_hz=_num(m, "fmin_hz", "audio"),
        fmax_hz=_num(m, "fmax_hz", "audio"),
        frame_length=int(_num(m, "frame_length", "audio")),
        hop_length=int(_num(m, "hop_length", "audio")),
        pitch_min_voiced_ratio=_num(m, "pitch_min_voiced_ratio", "audio"),
        pitch_range_percentile=_num(m, "pitch_range_percentile", "audio"),
        pitch_max_transition_rate=_num(m, "pitch_max_transition_rate", "audio"),
        pitch_min_note_frames=int(_num(m, "pitch_min_note_frames", "audio")),
    )
    if not 0 < cfg.hop_length <= cfg.frame_length:
        raise ConfigError("audio: 0 < hop_length <= frame_length が必要です")
    if not 0.0 <= cfg.pitch_min_voiced_ratio <= 1.0:
        raise ConfigError("audio.pitch_min_voiced_ratio: 0〜1 が必要です")
    if cfg.pitch_min_note_frames < 1:
        raise ConfigError("audio.pitch_min_note_frames: 1 以上が必要です")
    if cfg.pitch_max_transition_rate <= 0:
        raise ConfigError("audio.pitch_max_transition_rate: 正の値が必要です")
    if not 0.0 <= cfg.pitch_range_percentile < 50.0:
        raise ConfigError("audio.pitch_range_percentile: 0 以上 50 未満が必要です")
    if cfg.sample_rate <= 0:
        raise ConfigError("audio.sample_rate: 正の値が必要です")
    if not 0 < cfg.min_duration_sec < cfg.max_duration_sec:
        raise ConfigError("audio: 0 < min_duration_sec < max_duration_sec が必要です")
    if cfg.trim_top_db <= 0:
        raise ConfigError("audio.trim_top_db: 正の値が必要です")
    if not 0 < cfg.fmin_hz < cfg.fmax_hz:
        raise ConfigError("audio: 0 < fmin_hz < fmax_hz が必要です")
    return cfg


def parse_features(data: Mapping[str, Any]) -> FeaturesConfig:
    """features.yaml の内容を FeaturesConfig に変換する。"""
    m = _check_keys(data, ("melody", "chroma", "key", "chord", "articulation"), (), "features")
    keys = ("max_gap_sec", "leap_threshold_semitones",
            "ngram_length", "smoothness_ref_semitones", "min_notes")
    mm = _check_keys(m["melody"], keys, (), "features.melody")
    mel = MelodyConfig(
        max_gap_sec=_num(mm, "max_gap_sec", "features.melody"),
        leap_threshold_semitones=_num(mm, "leap_threshold_semitones", "features.melody"),
        ngram_length=int(_num(mm, "ngram_length", "features.melody")),
        smoothness_ref_semitones=_num(mm, "smoothness_ref_semitones", "features.melody"),
        min_notes=int(_num(mm, "min_notes", "features.melody")),
    )
    if mel.max_gap_sec <= 0:
        raise ConfigError("features.melody.max_gap_sec: 正の値が必要です")
    if mel.leap_threshold_semitones <= 0:
        raise ConfigError("features.melody.leap_threshold_semitones: 正の値が必要です")
    if mel.ngram_length < 2:
        raise ConfigError("features.melody.ngram_length: 2 以上が必要です")
    if mel.smoothness_ref_semitones <= 0:
        raise ConfigError("features.melody.smoothness_ref_semitones: 正の値が必要です")
    if mel.min_notes < 3:
        raise ConfigError("features.melody.min_notes: 3 以上が必要です")

    chroma = _check_keys(m["chroma"], ("hpss",), (), "features.chroma")
    chroma_cfg = ChromaConfig(hpss=_bool(chroma, "hpss", "features.chroma"))

    km = _check_keys(m["key"], ("min_correlation",), (), "features.key")
    key_cfg = KeyConfig(min_correlation=_num(km, "min_correlation", "features.key"))
    if not 0.0 <= key_cfg.min_correlation < 1.0:
        raise ConfigError("features.key.min_correlation: 0 以上 1 未満が必要です")

    w = "features.chord"
    cm = _check_keys(
        m["chord"],
        ("window_sec", "switch_penalty", "min_similarity", "silence_ratio",
         "min_chord_ratio", "extended_penalty", "complexity"),
        (), w,
    )
    cx_keys = ("extension", "type_variety", "chord_variety", "type_ref", "chord_ref")
    cx = _check_keys(cm["complexity"], cx_keys, (), f"{w}.complexity")
    chord_cfg = ChordConfig(
        window_sec=_num(cm, "window_sec", w),
        switch_penalty=_num(cm, "switch_penalty", w),
        min_similarity=_num(cm, "min_similarity", w),
        silence_ratio=_num(cm, "silence_ratio", w),
        min_chord_ratio=_num(cm, "min_chord_ratio", w),
        extended_penalty=_num(cm, "extended_penalty", w),
        weight_extension=_num(cx, "extension", f"{w}.complexity"),
        weight_type_variety=_num(cx, "type_variety", f"{w}.complexity"),
        weight_chord_variety=_num(cx, "chord_variety", f"{w}.complexity"),
        type_ref=int(_num(cx, "type_ref", f"{w}.complexity")),
        chord_ref=int(_num(cx, "chord_ref", f"{w}.complexity")),
    )
    if chord_cfg.window_sec <= 0:
        raise ConfigError(f"{w}.window_sec: 正の値が必要です")
    if chord_cfg.switch_penalty < 0 or chord_cfg.extended_penalty < 0:
        raise ConfigError(f"{w}: switch_penalty / extended_penalty は 0 以上が必要です")
    for name in ("min_similarity", "silence_ratio", "min_chord_ratio"):
        if not 0.0 <= getattr(chord_cfg, name) <= 1.0:
            raise ConfigError(f"{w}.{name}: 0〜1 が必要です")
    weights = (chord_cfg.weight_extension, chord_cfg.weight_type_variety, chord_cfg.weight_chord_variety)
    if min(weights) < 0 or sum(weights) <= 0:
        raise ConfigError(f"{w}.complexity: 重みは 0 以上で合計が正である必要があります")
    if chord_cfg.type_ref < 2 or chord_cfg.chord_ref < 2:
        raise ConfigError(f"{w}.complexity: type_ref / chord_ref は 2 以上が必要です")

    aw = "features.articulation"
    am = _check_keys(m["articulation"], ("decay_threshold_db", "gap_threshold_db", "onset_context_frames", "onset_decay_ratio", "min_onsets"), (), aw)
    art_cfg = ArticulationConfig(
        decay_threshold_db=_num(am, "decay_threshold_db", aw),
        gap_threshold_db=_num(am, "gap_threshold_db", aw),
        onset_context_frames=int(_num(am, "onset_context_frames", aw)),
        onset_decay_ratio=_num(am, "onset_decay_ratio", aw),
        min_onsets=int(_num(am, "min_onsets", aw)),
    )
    if art_cfg.decay_threshold_db >= 0 or art_cfg.gap_threshold_db >= 0:
        raise ConfigError(f"{aw}: decay_threshold_db / gap_threshold_db は負の値が必要です")
    if art_cfg.onset_context_frames < 1:
        raise ConfigError(f"{aw}.onset_context_frames: 1 以上が必要です")
    if not 0.0 < art_cfg.onset_decay_ratio < 1.0:
        raise ConfigError(f"{aw}.onset_decay_ratio: 0 より大きく 1 未満が必要です")
    if art_cfg.min_onsets < 2:
        raise ConfigError(f"{aw}.min_onsets: 2 以上が必要です")
    return FeaturesConfig(
        melody=mel, chroma=chroma_cfg, key=key_cfg, chord=chord_cfg, articulation=art_cfg
    )


def parse_normalization(data: Mapping[str, Any]) -> dict[str, NormRule]:
    """normalization.yaml の内容を NormRule の辞書に変換する。"""
    features = _check_keys(data, ("features",), (), "normalization")["features"]
    if not isinstance(features, Mapping) or not features:
        raise ConfigError("normalization.features: 空でないマッピングが必要です")
    rules: dict[str, NormRule] = {}
    for name, raw in features.items():
        where = f"normalization.features.{name}"
        m = _check_keys(raw, ("source", "method", "min", "max"), (), where)
        rule = NormRule(
            source=_str(m, "source", where),
            method=_choice(m, "method", NORM_METHODS, where),
            min=_num(m, "min", where),
            max=_num(m, "max", where),
        )
        if rule.min >= rule.max:
            raise ConfigError(f"{where}: min < max が必要です")
        rules[str(name)] = rule
    return rules


def parse_impressions(
    data: Mapping[str, Any], feature_names: Iterable[str]
) -> dict[int, ImpressionRule]:
    """impressions.yaml を ImpressionRule の辞書に変換する。

    Args:
        data: 読み込んだ YAML。
        feature_names: normalization.yaml で定義された特徴量名(参照先の検証用)。
    """
    known = set(feature_names)
    raw_all = _check_keys(data, ("impressions",), (), "impressions")["impressions"]
    if not isinstance(raw_all, Mapping):
        raise ConfigError("impressions.impressions: マッピングが必要です")
    if set(raw_all) != set(IMPRESSION_IDS):
        raise ConfigError(f"impressions: ID は {list(IMPRESSION_IDS)} がちょうど必要です")
    rules: dict[int, ImpressionRule] = {}
    for iid, raw in raw_all.items():
        where = f"impressions.{iid}"
        m = _check_keys(raw, ("label_low", "label_high", "inputs"), (), where)
        inputs_raw = m["inputs"]
        if not isinstance(inputs_raw, Mapping) or not inputs_raw:
            raise ConfigError(f"{where}.inputs: 空でないマッピングが必要です")
        inputs: dict[str, ImpressionInput] = {}
        for fname, iraw in inputs_raw.items():
            iwhere = f"{where}.inputs.{fname}"
            if fname not in known:
                raise ConfigError(f"{iwhere}: normalization.yaml に存在しない特徴量です")
            im = _check_keys(iraw, ("weight", "direction"), (), iwhere)
            weight = _num(im, "weight", iwhere)
            if weight <= 0:
                raise ConfigError(f"{iwhere}.weight: 正の値が必要です")
            inputs[str(fname)] = ImpressionInput(
                weight=weight, direction=_choice(im, "direction", IMPRESSION_DIRECTIONS, iwhere)
            )
        rules[int(iid)] = ImpressionRule(
            id=int(iid),
            label_low=_str(m, "label_low", where),
            label_high=_str(m, "label_high", where),
            inputs=inputs,
        )
    return rules


def parse_parts(data: Mapping[str, Any]) -> dict[str, PartRule]:
    """character_parts.yaml を PartRule の辞書に変換する。"""
    raw_all = _check_keys(data, ("parts",), (), "character_parts")["parts"]
    if not isinstance(raw_all, Mapping) or not raw_all:
        raise ConfigError("character_parts.parts: 空でないマッピングが必要です")
    rules: dict[str, PartRule] = {}
    for name, raw in raw_all.items():
        where = f"parts.{name}"
        m = _check_keys(
            raw, ("group", "label", "unit", "sources", "combine", "min", "max"), ("weights", "integer"), where
        )
        sources_raw = m["sources"]
        if not isinstance(sources_raw, list) or not sources_raw:
            raise ConfigError(f"{where}.sources: 空でないリストが必要です")
        sources: list[PartSource] = []
        for i, sraw in enumerate(sources_raw):
            swhere = f"{where}.sources[{i}]"
            sm = _check_keys(sraw, ("impression", "direction"), (), swhere)
            iid = sm["impression"]
            if isinstance(iid, bool) or iid not in IMPRESSION_IDS:
                raise ConfigError(f"{swhere}.impression: {list(IMPRESSION_IDS)} のいずれかが必要です")
            sources.append(PartSource(int(iid), _choice(sm, "direction", PART_DIRECTIONS, swhere)))
        if len({s.impression for s in sources}) != len(sources):
            raise ConfigError(f"{where}.sources: 同じ印象を重複して指定できません")

        combine = _choice(m, "combine", COMBINE_METHODS, where)
        weights: tuple[float, ...] | None = None
        if "weights" in m and m["weights"] is not None:
            w = m["weights"]
            if not isinstance(w, list) or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in w
            ):
                raise ConfigError(f"{where}.weights: 正の数のリストが必要です")
            weights = tuple(float(x) for x in w)
            if len(weights) != len(sources):
                raise ConfigError(f"{where}.weights: sources と同じ個数が必要です")
        if combine == "weighted_mean" and weights is None:
            raise ConfigError(f"{where}: weighted_mean には weights が必要です")

        integer = False
        if "integer" in m:
            integer = _bool(m, "integer", where)
        rule = PartRule(
            name=str(name),
            group=_choice(m, "group", PART_GROUPS, where),
            unit=_str(m, "unit", where),
            sources=tuple(sources),
            combine=combine,
            weights=weights,
            min=_num(m, "min", where),
            max=_num(m, "max", where),
            integer=integer,
            label=_str(m, "label", where),
        )
        if rule.min >= rule.max:
            raise ConfigError(f"{where}: min < max が必要です")
        rules[str(name)] = rule
    return rules


def _parse_flat(cls: type, data: Any, where: str) -> Any:
    """フィールドがすべて数値または文字列のデータクラスを、マッピングから生成する。

    キーの過不足と型を検証し、値の範囲はデータクラス側の検証 (ValueError) に任せる。
    """
    flds = dataclasses.fields(cls)
    m = _check_keys(data, [f.name for f in flds], (), where)
    kwargs: dict[str, Any] = {}
    for f in flds:
        kwargs[f.name] = _str(m, f.name, where) if f.type in ("str", str) else _num(m, f.name, where)
    try:
        return cls(**kwargs)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e


def _parse_slots(raw: Any, cls: type, where: str, str_fields: tuple[str, ...], int_fields: tuple[str, ...] = ()) -> tuple:
    """スロットのリスト (マッピングの並び) をデータクラスのタプルに変換する。"""
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{where}: 空でないリストが必要です")
    names = [f.name for f in dataclasses.fields(cls)]
    slots = []
    for i, item in enumerate(raw):
        w = f"{where}[{i}]"
        m = _check_keys(item, names, (), w)
        kwargs: dict[str, Any] = {}
        for name in names:
            if name in str_fields:
                kwargs[name] = _str(m, name, w)
            elif name in int_fields:
                kwargs[name] = int(_num(m, name, w))
            else:
                kwargs[name] = _num(m, name, w)
        try:
            slots.append(cls(**kwargs))
        except ValueError as e:
            raise ConfigError(f"{w}: {e}") from e
    return tuple(slots)


def parse_effects(data: Any) -> EffectsConfig:
    """render.yaml の effects セクションを変換する。"""
    m = _check_keys(data, ("color", "slots"), (), "render.effects")
    slots = _parse_slots(m["slots"], EffectSlot, "render.effects.slots", ("kind",))
    try:
        return EffectsConfig(color=_str(m, "color", "render.effects"), slots=slots)
    except ValueError as e:
        raise ConfigError(f"render.effects: {e}") from e


def parse_decorations(data: Any) -> DecorationsConfig:
    """render.yaml の decorations セクションを変換する。"""
    w = "render.decorations"
    m = _check_keys(data, ("accent_1", "accent_2", "stroke", "stroke_width", "slots"), (), w)
    slots = _parse_slots(m["slots"], DecorationSlot, f"{w}.slots", ("kind", "anchor"), ("accent",))
    try:
        return DecorationsConfig(
            accent_1=_str(m, "accent_1", w), accent_2=_str(m, "accent_2", w), stroke=_str(m, "stroke", w),
            stroke_width=_num(m, "stroke_width", w), slots=slots,
        )
    except ValueError as e:
        raise ConfigError(f"{w}: {e}") from e


def parse_render(data: Mapping[str, Any]) -> RenderConfig:
    """render.yaml の内容を RenderConfig に変換する。"""
    sections = {
        "canvas": CanvasConfig, "head": HeadConfig, "colors": ColorsConfig, "eyes": EyesConfig,
        "brows": BrowsConfig, "mouth": MouthConfig, "cheeks": CheeksConfig, "body": BodyConfig,
    }
    m = _check_keys(data, (*sections, "effects", "decorations"), (), "render")
    flat = {k: _parse_flat(cls, m[k], f"render.{k}") for k, cls in sections.items()}
    return RenderConfig(
        **flat, effects=parse_effects(m["effects"]), decorations=parse_decorations(m["decorations"])
    )


def load_config(config_dir: str | Path | None = None) -> AppConfig:
    """設定ディレクトリから全設定を読み込み、検証して返す。

    Args:
        config_dir: 設定ディレクトリ。None ならリポジトリ直下の ``config/``。

    Returns:
        検証済みの AppConfig。

    Raises:
        ConfigError: ファイルの欠落、構文エラー、検証エラーのいずれか。
    """
    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    audio = parse_audio(_read_yaml(base / "audio.yaml"))
    features = parse_features(_read_yaml(base / "features.yaml"))
    normalization = parse_normalization(_read_yaml(base / "normalization.yaml"))
    impressions = parse_impressions(_read_yaml(base / "impressions.yaml"), normalization)
    parts = parse_parts(_read_yaml(base / "character_parts.yaml"))
    render = parse_render(_read_yaml(base / "render.yaml"))
    for part, slots in (("effect_count", render.effects.slots), ("decoration_amount", render.decorations.slots)):
        if part in parts and len(slots) < parts[part].max:
            raise ConfigError(
                f"render.yaml のスロット数 ({len(slots)}) が {part} の最大値 ({parts[part].max}) に足りません"
            )
    return AppConfig(
        audio=audio,
        features=features,
        normalization=normalization,
        impressions=impressions,
        parts=parts,
        render=render,
    )
