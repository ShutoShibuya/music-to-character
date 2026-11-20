"""C. メロディ(音高の変化・跳躍・繰り返し・滑らかさ)の抽出。"""

from __future__ import annotations

from collections import Counter

import numpy as np

from musicchar.features.base import FeatureExtractor
from musicchar.features.notes import Note
from musicchar.features.pitch_track import PitchTracker
from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig, MelodyConfig
from musicchar.models.errors import FeatureExtractionError
from musicchar.models.features import Melody


def split_phrases(notes: list[Note], hop_sec: float, max_gap_sec: float) -> list[list[int]]:
    """音符列を、長い無音(``max_gap_sec`` 超)で区切ったフレーズ(音高列)のリストにする。"""
    phrases: list[list[int]] = []
    for i, note in enumerate(notes):
        if i == 0 or (note.start - notes[i - 1].end) * hop_sec > max_gap_sec:
            phrases.append([])
        phrases[-1].append(note.pitch)
    return phrases


def compute_melody(
    notes: list[Note], hop_sec: float, total_frames: int, cfg: MelodyConfig
) -> Melody:
    """音符列からメロディ特徴量を計算する。

    各指標の定義(音程はすべてフレーズ内の隣接音符間で測り、フレーズをまたがない):

    - ``pitch_change_rate``: 音高が変わった音程の数 ÷ 全体の長さ(秒)。
    - ``mean_interval_semitones``: 音程の絶対値の平均(同音反復は 0 として数える)。
    - ``leap_ratio``: 音程のうち ``leap_threshold_semitones`` 以上のものの割合。
    - ``repetition_score``: 音程列(移調に不変)の ``ngram_length`` 連のうち、
      他の位置にも同じものが現れる位置の割合。
    - ``smoothness``: 1 - 平均|音程の変化量| ÷ ``smoothness_ref_semitones`` (0〜1)。
      音階のように一定の歩幅で進めば 1、上下に折れ曲がるほど 0 に近づく。

    Args:
        notes: ``segment_notes`` の結果。
        hop_sec: フレーム間隔(秒)。
        total_frames: トラック全体のフレーム数(分母の長さに使用)。
        cfg: メロディ設定。

    Raises:
        FeatureExtractionError: 音符が少なすぎる、または音程・繰り返し・滑らかさの
            いずれかを評価できるだけのフレーズ長がない場合。
    """
    if len(notes) < cfg.min_notes:
        raise FeatureExtractionError(
            f"音符が少なすぎてメロディを評価できません ({len(notes)} < {cfg.min_notes})"
        )
    phrases = split_phrases(notes, hop_sec, cfg.max_gap_sec)
    intervals = [np.diff(np.asarray(p)) for p in phrases if len(p) >= 2]
    if not intervals:
        raise FeatureExtractionError("連続した音符がなく、音程を評価できません")
    all_intervals = np.concatenate(intervals)

    second_diffs = [np.diff(iv) for iv in intervals if len(iv) >= 2]
    if not second_diffs:
        raise FeatureExtractionError("3音以上続くフレーズがなく、滑らかさを評価できません")
    mean_jerk = float(np.mean(np.abs(np.concatenate(second_diffs))))

    ngrams = [
        tuple(iv[i : i + cfg.ngram_length])
        for iv in intervals
        for i in range(len(iv) - cfg.ngram_length + 1)
    ]
    if not ngrams:
        raise FeatureExtractionError(
            f"{cfg.ngram_length + 1}音以上続くフレーズがなく、繰り返しを評価できません"
        )
    counts = Counter(ngrams)
    repetition = sum(1 for g in ngrams if counts[g] >= 2) / len(ngrams)

    total_sec = total_frames * hop_sec
    abs_iv = np.abs(all_intervals)
    return Melody(
        pitch_change_rate=float(np.count_nonzero(all_intervals)) / total_sec,
        mean_interval_semitones=float(np.mean(abs_iv)),
        leap_ratio=float(np.mean(abs_iv >= cfg.leap_threshold_semitones)),
        repetition_score=float(repetition),
        smoothness=float(np.clip(1.0 - mean_jerk / cfg.smoothness_ref_semitones, 0.0, 1.0)),
    )


class MelodyExtractor(FeatureExtractor[Melody]):
    """音高トラックを音符列にして、メロディ特徴量を計算する。

    単旋律を前提とする。音高トラックは ``PitchTracker`` を D・E と共有する。
    """

    name = "c_melody"

    def __init__(
        self, cfg: AudioConfig, melody_cfg: MelodyConfig, tracker: PitchTracker | None = None
    ) -> None:
        super().__init__(cfg)
        self.melody_cfg = melody_cfg
        self.tracker = tracker if tracker is not None else PitchTracker(cfg)

    def extract(self, audio: AudioData) -> Melody:
        """メロディ特徴量を返す。評価できなければ ``FeatureExtractionError``。"""
        track = self.tracker.track(audio)
        return compute_melody(list(track.notes), track.hop_sec, track.midi.size, self.melody_cfg)
