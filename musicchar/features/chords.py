"""G. コード(種類・進行・変化頻度・複雑さ)の抽出。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.features.chroma import ChromaProvider
from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig, ChordConfig, ChromaConfig
from musicchar.models.errors import FeatureExtractionError
from musicchar.models.features import PITCH_CLASSES, Chords

# コード種別 -> (表記の接尾辞, 根音からの構成音の半音数)
CHORD_TYPES: dict[str, tuple[str, tuple[int, ...]]] = {
    "maj": ("", (0, 4, 7)),
    "min": ("m", (0, 3, 7)),
    "dim": ("dim", (0, 3, 6)),
    "aug": ("aug", (0, 4, 8)),
    "sus2": ("sus2", (0, 2, 7)),
    "sus4": ("sus4", (0, 5, 7)),
    "7": ("7", (0, 4, 7, 10)),
    "maj7": ("maj7", (0, 4, 7, 11)),
    "min7": ("m7", (0, 3, 7, 10)),
}
BASIC_TYPES = frozenset({"maj", "min"})
MINOR_TYPES = frozenset({"min", "min7", "dim"})
_TYPE_NAMES = tuple(CHORD_TYPES)


def build_templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    """全コード(12根音 × 種別)のテンプレートを作る。

    Returns:
        (L2 正規化したテンプレート行列 (コード数, 12), 各行の (根音, 種別) のリスト)。
    """
    rows, labels = [], []
    for type_name in _TYPE_NAMES:
        for root in range(12):
            v = np.zeros(12)
            for interval in CHORD_TYPES[type_name][1]:
                v[(root + interval) % 12] = 1.0
            rows.append(v / np.linalg.norm(v))
            labels.append((root, type_name))
    return np.array(rows), labels


def chord_label(root: int, type_name: str) -> str:
    """表示用のコード名 (例 "C", "Am", "G7")。"""
    return f"{PITCH_CLASSES[root]}{CHORD_TYPES[type_name][0]}"


def smooth_path(scores: np.ndarray, switch_penalty: float) -> np.ndarray:
    """状態切り替えにペナルティを課した最大得点経路 (Viterbi) を求める。

    各時刻の状態 k の得点 ``scores[k, t]`` の合計から、状態を切り替えるたびに
    ``switch_penalty`` を引いた値が最大になる経路を返す。

    Args:
        scores: 形状 (状態数, 時間)。
        switch_penalty: 切り替え 1 回あたりのペナルティ (0 以上)。

    Returns:
        各時刻の状態番号 (長さ 時間)。
    """
    n_states, n_t = scores.shape
    dp = np.empty((n_t, n_states))
    back = np.zeros((n_t, n_states), dtype=np.int64)
    dp[0] = scores[:, 0]
    for t in range(1, n_t):
        best_prev = int(np.argmax(dp[t - 1]))
        switch = dp[t - 1, best_prev] - switch_penalty
        stay = dp[t - 1]
        take_switch = switch > stay
        dp[t] = scores[:, t] + np.where(take_switch, switch, stay)
        back[t] = np.where(take_switch, best_prev, np.arange(n_states))
    path = np.empty(n_t, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_t - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@dataclass(frozen=True)
class ChordSegment:
    """同じコードが続く区間。start/end は窓番号(end は含まない)。"""

    start: int
    end: int
    root: int
    type_name: str

    @property
    def label(self) -> str:
        return chord_label(self.root, self.type_name)

    @property
    def length(self) -> int:
        return self.end - self.start


def estimate_segments(window_chroma: np.ndarray, cfg: ChordConfig) -> tuple[list[ChordSegment], int]:
    """窓ごとのクロマからコード区間を推定する。

    1. 窓のクロマを L2 正規化し、全テンプレートとのコサイン類似度を得る
       (三和音以外は ``extended_penalty`` を引く)。
    2. エネルギーが小さい窓は無音、最良類似度が ``min_similarity`` 未満の窓は
       「コードなし」とする。
    3. ``switch_penalty`` 付きの Viterbi で、細かい切り替わりを抑えた経路を求める。

    Args:
        window_chroma: 形状 (12, 窓数) のクロマ(エネルギー保持)。
        cfg: コード設定。

    Returns:
        (コードあり区間のリスト, 全窓数)。「コードなし」の窓は区間に含まれない。
    """
    templates, labels = build_templates()
    n_win = window_chroma.shape[1]
    energy = window_chroma.sum(axis=0)
    silent = energy < cfg.silence_ratio * max(float(energy.max()), 1e-12)

    norms = np.linalg.norm(window_chroma, axis=0)
    unit = window_chroma / np.maximum(norms, 1e-12)
    sims = templates @ unit  # (コード数, 窓数)
    penalty = np.array([0.0 if t in BASIC_TYPES else cfg.extended_penalty for _, t in labels])
    chord_scores = sims - penalty[:, None]
    chord_scores[:, silent] = 0.0

    none_scores = np.where(silent, 1.0, cfg.min_similarity)[None, :]
    path = smooth_path(np.vstack([chord_scores, none_scores]), cfg.switch_penalty)

    none_state = len(labels)
    segments: list[ChordSegment] = []
    start = 0
    for i in range(1, n_win + 1):
        if i == n_win or path[i] != path[start]:
            if path[start] != none_state:
                root, type_name = labels[path[start]]
                segments.append(ChordSegment(start, i, root, type_name))
            start = i
    return segments, n_win


def compute_chords(
    segments: list[ChordSegment], n_windows: int, duration_sec: float, cfg: ChordConfig
) -> Chords:
    """コード区間から ``Chords`` を計算する。

    - ``chord_types``: 種別ごとの区間数。``progression``: 区間順のコード名。
    - ``change_rate_per_sec``: コード区間の切り替わり回数 ÷ 音声の長さ(秒)。
    - ``minor_ratio``: コードのあった時間のうち、短調系 (min/min7/dim) の割合。
    - ``complexity``: 三和音以外の時間割合・種類数・異なるコード数を
      ``chord.complexity`` の重みで平均した値 (0〜1)。

    Raises:
        FeatureExtractionError: コードを判定できた窓の割合が ``min_chord_ratio`` 未満の場合。
    """
    total = sum(s.length for s in segments)
    if n_windows == 0 or total / n_windows < cfg.min_chord_ratio:
        raise FeatureExtractionError(
            f"コードを判定できた区間が少なすぎます ({total}/{n_windows} 窓)"
        )
    types = Counter(s.type_name for s in segments)
    labels = [s.label for s in segments]

    minor_ratio = sum(s.length for s in segments if s.type_name in MINOR_TYPES) / total
    extension_ratio = sum(s.length for s in segments if s.type_name not in BASIC_TYPES) / total
    type_variety = (len(types) - 1) / (cfg.type_ref - 1)
    chord_variety = (len(set(labels)) - 1) / (cfg.chord_ref - 1)
    weights = (cfg.weight_extension, cfg.weight_type_variety, cfg.weight_chord_variety)
    terms = (extension_ratio, min(type_variety, 1.0), min(chord_variety, 1.0))
    complexity = sum(w * t for w, t in zip(weights, terms)) / sum(weights)

    return Chords(
        chord_types=dict(types),
        progression=tuple(labels),
        change_rate_per_sec=(len(segments) - 1) / duration_sec,
        complexity=float(np.clip(complexity, 0.0, 1.0)),
        minor_ratio=float(minor_ratio),
    )


class ChordExtractor(FeatureExtractor[Chords]):
    """クロマとコードテンプレートの照合からコードを推定する。

    音声は ``chord.window_sec`` ごとの窓に分け、窓ごとに最も近いコードを求め、
    切り替わりペナルティ付きで平滑化する。単純なテンプレート照合なので、
    転回形や省略された構成音、ベース音の扱いは考慮していない。
    """

    name = "g_chords"

    def __init__(
        self,
        cfg: AudioConfig,
        chord_cfg: ChordConfig,
        chroma_cfg: ChromaConfig,
        provider: ChromaProvider | None = None,
    ) -> None:
        super().__init__(cfg)
        self.chord_cfg = chord_cfg
        self.provider = provider if provider is not None else ChromaProvider(cfg, chroma_cfg)

    def extract(self, audio: AudioData) -> Chords:
        """コード特徴を返す。推定できなければ ``FeatureExtractionError``。"""
        chroma = self.provider.chroma(audio)
        frames_per_window = max(1, round(self.chord_cfg.window_sec / chroma.hop_sec))
        n_win = chroma.matrix.shape[1] // frames_per_window
        if n_win < 2:
            raise FeatureExtractionError("音声が短すぎてコードを推定できません")
        trimmed = chroma.matrix[:, : n_win * frames_per_window]
        window_chroma = trimmed.reshape(12, n_win, frames_per_window).mean(axis=2)
        segments, n_windows = estimate_segments(window_chroma, self.chord_cfg)
        return compute_chords(segments, n_windows, audio.duration, self.chord_cfg)
