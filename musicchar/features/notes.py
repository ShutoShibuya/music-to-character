"""フレームごとの音高列から音符列への変換。"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class Note:
    """1つの音符。

    Attributes:
        start: 開始フレーム(含む)。
        end: 終了フレーム(含まない)。
        pitch: 半音単位に量子化した MIDI ノート番号。
    """

    start: int
    end: int
    pitch: int


def segment_notes(midi: np.ndarray, min_note_frames: int) -> list[Note]:
    """フレームごとの音高(MIDI, 無声は NaN)を音符列に変換する。

    1. 有声フレームの音程ずれ(チューニング)を ``librosa.pitch_tuning`` で推定して補正し、
       半音単位に量子化する。
    2. 同じ音高が連続する区間を 1 つの塊とし、``min_note_frames`` 未満の塊
       (pyin が音の切り替わりで出す中間的な音高など)は音符とみなさず捨てる。
    3. 同じ音高の塊の間が、捨てた有声フレームだけで埋まっている場合は 1 つに結合する
       (ビブラートや揺れでの分断を防ぐ)。無声(NaN)フレームを挟む場合は結合しない。

    同じ高さの音の連打は、間に無声フレームがない限り 1 音として扱われる(制約)。

    Args:
        midi: フレームごとの MIDI ノート番号。無声・休符のフレームは NaN。
        min_note_frames: 音符とみなす最小フレーム数。

    Returns:
        開始順の音符のリスト。有声フレームがなければ空リスト。
    """
    voiced = ~np.isnan(midi)
    if not voiced.any():
        return []

    tuning = float(librosa.pitch_tuning(librosa.midi_to_hz(midi[voiced])))
    quantized = np.full(midi.shape, np.nan)
    quantized[voiced] = np.rint(midi[voiced] - tuning)

    runs: list[Note] = []
    start = 0
    for i in range(1, len(quantized) + 1):
        if i == len(quantized) or not _same(quantized[i], quantized[start]):
            if voiced[start] and i - start >= min_note_frames:
                runs.append(Note(start, i, int(quantized[start])))
            start = i

    merged: list[Note] = []
    for run in runs:
        if (
            merged
            and merged[-1].pitch == run.pitch
            and voiced[merged[-1].end : run.start].all()
        ):
            merged[-1] = Note(merged[-1].start, run.end, run.pitch)
        else:
            merged.append(run)
    return merged


def _same(a: float, b: float) -> bool:
    """量子化済み値の同一判定(NaN 同士は同じ「無声」とみなす)。"""
    return (np.isnan(a) and np.isnan(b)) or a == b
