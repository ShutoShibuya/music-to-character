"""テスト用の合成音生成。"""

from __future__ import annotations

import numpy as np

SR = 22050


def sine(freq: float, sec: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    """一定振幅のサイン波。"""
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def click_track(bpm: float, sec: float = 20.0, sr: int = SR) -> np.ndarray:
    """指定 BPM で鳴る減衰サイン波(1kHz)のクリック列。"""
    y = np.zeros(int(sr * sec), dtype=np.float32)
    n = np.arange(2000) / sr
    click = (np.sin(2 * np.pi * 1000 * n) * np.exp(-n * 60)).astype(np.float32)
    for t in np.arange(0.2, sec - 0.1, 60.0 / bpm):
        i = int(t * sr)
        seg = click[: len(y) - i]
        y[i : i + len(seg)] += seg
    return y


def midi_to_hz(midi: float) -> float:
    """MIDI ノート番号を周波数(Hz)に変換する。"""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def melody(midi_notes: list[float | None], sec_each: float = 1.0, amp: float = 0.5) -> np.ndarray:
    """MIDI ノート列を順に鳴らす。None は休符(無音)。"""
    parts = [
        np.zeros(int(sec_each * SR), dtype=np.float32) if n is None
        else sine(midi_to_hz(n), sec_each, amp)
        for n in midi_notes
    ]
    return np.concatenate(parts)
