"""音声ファイルからキャラクターまでの一連の処理 (各層の統合点)。

audio -> features -> impressions -> character の各層は互いに独立しており、
このモジュールだけがそれらをつなぐ。画面 (app.py) はここを呼ぶだけにする。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import BinaryIO

from musicchar.audio import load_audio, preprocess
from musicchar.character import map_parts, render_svg
from musicchar.features import extract_all
from musicchar.impressions import impressions_from_features
from musicchar.models import AppConfig, AudioData, CharacterParams, FeatureSet, ImpressionSet


@dataclass(frozen=True)
class AnalysisResult:
    """1 曲の分析結果。

    Attributes:
        source: 入力の名前 (ファイル名など)。
        original_duration: 読み込んだ音声の長さ(秒)。
        analyzed_duration: 前処理 (無音トリム・長さ制限) 後の、実際に分析した長さ(秒)。
        features: 音楽的特徴量 A〜H。抽出できなかったものは None で、理由は ``errors``。
        impressions: 印象 1〜8。計算できなかったものは None。
        character: キャラクターの各パーツの値。
        svg: キャラクターの SVG。
    """

    source: str
    original_duration: float
    analyzed_duration: float
    features: FeatureSet
    impressions: ImpressionSet
    character: CharacterParams
    svg: str


def analyze_audio(audio: AudioData, config: AppConfig) -> AnalysisResult:
    """読み込み済みの音声を分析してキャラクターを生成する。

    Args:
        audio: 読み込み済みの音声。
        config: アプリ全体の設定。

    Returns:
        ``AnalysisResult``。一部の特徴量が抽出できなくても、可能な範囲で結果を返す
        (欠損した印象に依存するパーツは中立値になり、``estimated`` が立つ)。

    Raises:
        AudioTooShortError: 音声が無音、または短すぎる場合。
    """
    processed = preprocess(audio, config.audio)
    features = extract_all(processed, config.audio, config.features)
    impressions = impressions_from_features(features, config.normalization, config.impressions)
    character = map_parts(impressions, config.parts)
    return AnalysisResult(
        source=audio.source,
        original_duration=audio.duration,
        analyzed_duration=processed.duration,
        features=features,
        impressions=impressions,
        character=character,
        svg=render_svg(character, config.render),
    )


def analyze(
    file: str | os.PathLike[str] | BinaryIO, config: AppConfig, source: str | None = None
) -> AnalysisResult:
    """音声ファイルを読み込み、分析してキャラクターを生成する。

    Args:
        file: ファイルパス、またはファイルオブジェクト。
        config: アプリ全体の設定。
        source: 結果に記録する名前。省略時はファイル名。

    Returns:
        ``AnalysisResult``。

    Raises:
        AudioLoadError: 音声ファイルを読み込めない場合。
        AudioTooShortError: 音声が無音、または短すぎる場合。
    """
    return analyze_audio(load_audio(file, config.audio.sample_rate, source), config)
