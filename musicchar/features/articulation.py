"""H. アーティキュレーション(立ち上がり・持続・重なり・切れ目)の抽出。"""

from __future__ import annotations

import librosa
import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.models.audio import AudioData
from musicchar.models.config import ArticulationConfig, AudioConfig
from musicchar.models.errors import FeatureExtractionError
from musicchar.models.features import Articulation

_EPS = 1e-12


def drop_decay_onsets(
    onsets: np.ndarray, rms: np.ndarray, context_frames: int, decay_ratio: float
) -> np.ndarray:
    """音の減衰を拾った発音点を除く。

    スペクトル変化に基づく発音点検出は、音の終わり(減衰や切断)も拾うことがある。
    発音点のフレームの RMS が、直前 ``context_frames`` フレームの最大 RMS の
    ``decay_ratio`` 倍未満なら、すでに音量が落ちている=音の終わりとみなして捨てる。
    アタック(RMS がピーク上かその手前)や、ピッチだけが変わるレガート(RMS がほぼ同じ)は残る。

    Args:
        onsets: 発音点のフレーム番号。
        rms: フレームごとの RMS。
        context_frames: 直前に遡って最大値を取るフレーム数。
        decay_ratio: 減衰とみなす RMS の比率 (0〜1)。

    Returns:
        除外後の発音点。
    """
    keep = [
        k for k in onsets
        if rms[k] >= decay_ratio * np.max(rms[max(k - context_frames, 0) : k + 1])
    ]
    return np.asarray(keep, dtype=int)


def analyze_articulation(
    rms: np.ndarray,
    onset_env: np.ndarray,
    onsets: np.ndarray,
    hop_sec: float,
    cfg: ArticulationConfig,
) -> Articulation:
    """RMS・オンセット強度・発音フレームからアーティキュレーションを計算する。

    各発音点 i から次の発音点(最後は末尾)までの区間で、RMS がその区間のピークから
    ``decay_threshold_db`` 下がる最初のフレームを「減衰した時点」とする。

    - ``onset_strength_mean``: 発音点でのオンセット強度の平均(立ち上がりの鋭さ)。
    - ``onset_density``: 発音点の数 ÷ 長さ(秒)。
    - ``mean_note_duration``: 発音から減衰までの時間の平均。次の発音までに減衰しない音は
      次の発音までの時間とする。
    - ``overlap_ratio``: 次の発音点を持つ音のうち、その発音までに減衰しなかった
      (次の音と重なる)ものの割合。
    - ``gap_ratio``: RMS が基準レベル (95%タイル) より ``gap_threshold_db`` 以上低い
      フレーム(切れ目)の割合。

    Args:
        rms: フレームごとの RMS。
        onset_env: 同じフレーム軸のオンセット強度。
        onsets: 発音点のフレーム番号 (昇順)。
        hop_sec: フレーム間隔(秒)。
        cfg: アーティキュレーション設定。

    Raises:
        FeatureExtractionError: 発音点が ``min_onsets`` 未満、または音量の基準が取れない場合。
    """
    if len(onsets) < cfg.min_onsets:
        raise FeatureExtractionError(
            f"発音点が少なすぎてアーティキュレーションを評価できません ({len(onsets)} < {cfg.min_onsets})"
        )
    reference = float(np.percentile(rms, 95))
    if reference < _EPS:
        raise FeatureExtractionError("音量が小さすぎてアーティキュレーションを評価できません")

    decay_ratio = 10.0 ** (cfg.decay_threshold_db / 20.0)
    bounds = list(onsets) + [len(rms)]
    durations: list[float] = []
    ringing: list[bool] = []  # 次の発音までに減衰しなかったか
    for i, nxt in zip(bounds[:-1], bounds[1:]):
        window = rms[i:nxt]
        peak_at = int(np.argmax(window))
        below = np.nonzero(window[peak_at:] < window[peak_at] * decay_ratio)[0]
        if below.size:
            durations.append((peak_at + int(below[0])) * hop_sec)
            ringing.append(False)
        else:
            durations.append(len(window) * hop_sec)
            ringing.append(True)

    gap_level = reference * 10.0 ** (cfg.gap_threshold_db / 20.0)
    total_sec = len(rms) * hop_sec
    return Articulation(
        onset_strength_mean=float(np.mean(onset_env[onsets])),
        onset_density=len(onsets) / total_sec,
        mean_note_duration=float(np.mean(durations)),
        overlap_ratio=float(np.mean(ringing[:-1])),  # 最後の音は「次の発音」がないので除く
        gap_ratio=float(np.mean(rms < gap_level)),
    )


class ArticulationExtractor(FeatureExtractor[Articulation]):
    """``librosa.onset`` で発音点を検出し、RMS の減衰からアーティキュレーションを求める。

    RMS は ``center=True`` で計算し、オンセット強度とフレーム位置を揃える。
    音の終わりで RMS が直前より大きく下がっている地点の発音点は、減衰とみなして除外する。
    和音や多声でも音量の包絡だけを見るので、個々の音の重なりを分離するわけではなく、
    「次の発音の時点で全体の音量がまだ残っているか」を重なりの目安にしている。
    """

    name = "h_articulation"

    def __init__(self, cfg: AudioConfig, art_cfg: ArticulationConfig) -> None:
        super().__init__(cfg)
        self.art_cfg = art_cfg

    def extract(self, audio: AudioData) -> Articulation:
        """アーティキュレーションを返す。評価できなければ ``FeatureExtractionError``。"""
        frame, hop = self.cfg.frame_length, self.cfg.hop_length
        if audio.y.size < frame:
            raise FeatureExtractionError("音声が短すぎてアーティキュレーションを評価できません")
        try:
            onset_env = librosa.onset.onset_strength(y=audio.y, sr=audio.sr, hop_length=hop)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=audio.sr, hop_length=hop)
            rms = librosa.feature.rms(y=audio.y, frame_length=frame, hop_length=hop, center=True)[0]
        except Exception as e:
            raise FeatureExtractionError(f"オンセットの検出に失敗: {e}") from e
        n = min(len(rms), len(onset_env))
        onsets = np.asarray(onsets[onsets < n], dtype=int)
        onsets = drop_decay_onsets(
            onsets, rms[:n], self.art_cfg.onset_context_frames, self.art_cfg.onset_decay_ratio
        )
        return analyze_articulation(rms[:n], onset_env[:n], onsets, hop / audio.sr, self.art_cfg)
