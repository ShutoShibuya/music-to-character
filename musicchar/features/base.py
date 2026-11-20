"""特徴量抽出器の抽象基底クラス。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig

T = TypeVar("T")


class FeatureExtractor(ABC, Generic[T]):
    """特徴量1種類(A〜H のいずれか)を抽出する抽出器。

    Attributes:
        name: 結果を格納する ``FeatureSet`` のフィールド名 (例 "b_tempo")。
    """

    name: ClassVar[str]

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def extract(self, audio: AudioData) -> T:
        """音声から特徴量を抽出する。

        Args:
            audio: 前処理済みの音声。

        Returns:
            抽出した特徴量。

        Raises:
            FeatureExtractionError: 抽出できない場合。理由はメッセージに書く
                (``extract_all`` が ``FeatureSet.errors`` に記録する)。
                失敗を 0 などの値で代用して返してはならない。
        """
