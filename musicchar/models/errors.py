"""プロジェクト共通の例外クラス。"""

from __future__ import annotations


class MusicCharError(Exception):
    """本プロジェクトの全例外の基底クラス。"""


class ConfigError(MusicCharError):
    """設定ファイルの読込・検証に失敗した。"""


class AudioLoadError(MusicCharError):
    """音声ファイルを読み込めなかった。"""


class AudioTooShortError(MusicCharError):
    """音声が短すぎる、または無音で分析できない。"""


class FeatureExtractionError(MusicCharError):
    """特徴量の抽出に失敗した(呼び出し側で None と理由記録に変換される)。"""


class ImpressionError(MusicCharError):
    """印象値の計算に失敗した。"""


class RenderError(MusicCharError):
    """キャラクターの描画に失敗した。"""
