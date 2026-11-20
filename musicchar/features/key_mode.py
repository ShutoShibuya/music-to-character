"""A. 調(キーとメジャー/マイナー)の抽出。"""

from __future__ import annotations

import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.features.chroma import ChromaProvider
from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig, ChromaConfig, KeyConfig
from musicchar.models.errors import FeatureExtractionError
from musicchar.models.features import PITCH_CLASSES, KeyMode

# Krumhansl-Schmuckler のキープロファイル (Krumhansl & Kessler, 1982)。
# 主音 C を基準にした、各音名の「その調での収まりの良さ」。
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_profile: np.ndarray) -> tuple[str, str, float, float]:
    """12次元のクロマ分布から、最も相関の高い調を求める。

    Args:
        chroma_profile: 長さ 12 の非負ベクトル (C, C#, ..., B)。

    Returns:
        (主音名, "major"/"minor", 最大相関, 2番目に高い相関)。

    Raises:
        FeatureExtractionError: 分布が一様(分散ゼロ)で相関を計算できない場合。
    """
    if chroma_profile.shape != (12,) or np.std(chroma_profile) < 1e-12:
        raise FeatureExtractionError("音名の分布に偏りがなく、調を判定できません")
    candidates: list[tuple[float, str, str]] = []
    for mode, profile in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(chroma_profile, np.roll(profile, tonic))[0, 1])
            candidates.append((r, PITCH_CLASSES[tonic], mode))
    candidates.sort(reverse=True)
    best, second = candidates[0], candidates[1]
    return best[1], best[2], best[0], second[0]


class KeyModeExtractor(FeatureExtractor[KeyMode]):
    """曲全体のクロマ分布と Krumhansl-Schmuckler プロファイルの相関から調を推定する。

    ``confidence`` は最大相関(0 未満は 0)。最大相関が ``key.min_correlation`` 未満なら、
    推測した調を返さず ``FeatureExtractionError`` にする。
    平行調(例: C メジャーと A マイナー)は音名の分布が近く、取り違えることがある。
    また調性のないノイズや打楽器主体の音でも、偶然高い相関が出て調を返すことがある。
    """

    name = "a_key"

    def __init__(
        self,
        cfg: AudioConfig,
        key_cfg: KeyConfig,
        chroma_cfg: ChromaConfig,
        provider: ChromaProvider | None = None,
    ) -> None:
        super().__init__(cfg)
        self.key_cfg = key_cfg
        self.provider = provider if provider is not None else ChromaProvider(cfg, chroma_cfg)

    def extract(self, audio: AudioData) -> KeyMode:
        """調を返す。判定できなければ ``FeatureExtractionError``。"""
        profile = self.provider.chroma(audio).matrix.mean(axis=1)
        key, mode, best, _second = estimate_key(profile)
        if best < self.key_cfg.min_correlation:
            raise FeatureExtractionError(
                f"調の確信度が低く判定できません (相関 {best:.2f} < {self.key_cfg.min_correlation})"
            )
        return KeyMode(key=key, mode=mode, confidence=float(np.clip(best, 0.0, 1.0)))
