"""C(メロディ)のテスト。

純粋関数 (segment_notes / compute_melody) は配列を直接与えて期待値を手計算で検証し、
最後に pyin を通した結合テストを行う。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from synth import SR, melody

from musicchar.config_loader import load_config
from musicchar.features import MelodyExtractor
from musicchar.features.melody import compute_melody, split_phrases
from musicchar.features.notes import Note, segment_notes
from musicchar.models import AudioData, FeatureExtractionError

HOP = 0.02
NF = 10  # 1音のフレーム数
NAN = float("nan")


@pytest.fixture(scope="module")
def mcfg():
    return load_config().features.melody


def make_notes(pitches: list[int], gaps: dict[int, int] | None = None) -> tuple[list[Note], int]:
    """各音 NF フレームの音符列を作る。gaps[i] = i 番目の音の直前に入れる無音フレーム数。"""
    gaps, notes, t = gaps or {}, [], 0
    for i, p in enumerate(pitches):
        t += gaps.get(i, 0)
        notes.append(Note(t, t + NF, p))
        t += NF
    return notes, t


def track(*segments: tuple[float, int]) -> np.ndarray:
    """(MIDI 値 or NaN, フレーム数) の並びからフレームごとの音高配列を作る。"""
    return np.concatenate([np.full(n, v) for v, n in segments])


# ---------- segment_notes ----------

def test_segment_constant_pitch_is_one_note() -> None:
    assert segment_notes(track((60.0, 20)), 4) == [Note(0, 20, 60)]


def test_segment_drops_transition_frames_between_notes() -> None:
    notes = segment_notes(track((60.0, 10), (61.0, 1), (62.0, 1), (63.0, 1), (64.0, 10)), 4)
    assert [n.pitch for n in notes] == [60, 64]


def test_segment_unvoiced_gap_separates_same_pitch() -> None:
    notes = segment_notes(track((60.0, 10), (NAN, 5), (60.0, 10)), 4)
    assert [(n.start, n.end, n.pitch) for n in notes] == [(0, 10, 60), (15, 25, 60)]


def test_segment_merges_flicker_without_unvoiced_gap() -> None:
    notes = segment_notes(track((60.0, 10), (61.0, 2), (60.0, 10)), 4)
    assert notes == [Note(0, 22, 60)]


def test_segment_compensates_tuning_offset() -> None:
    """半音の境界付近 (60.4 / 60.52) で揺れても、チューニング補正で 1 音になる。"""
    midi = np.tile([60.40, 60.52], 12)
    notes = segment_notes(midi, 4)
    assert len(notes) == 1 and notes[0].pitch in (60, 61)


def test_segment_min_note_frames_boundary() -> None:
    t = track((60.0, 3), (NAN, 2), (64.0, 4))
    assert [n.pitch for n in segment_notes(t, 4)] == [64]
    assert [n.pitch for n in segment_notes(t, 3)] == [60, 64]


def test_segment_without_voiced_frames_is_empty() -> None:
    assert segment_notes(track((NAN, 30)), 4) == []


# ---------- split_phrases ----------

def test_split_phrases_on_long_gap() -> None:
    notes, _ = make_notes([60, 62, 64, 65], gaps={2: 100})  # 100 * 0.02 = 2 秒
    assert split_phrases(notes, HOP, max_gap_sec=1.0) == [[60, 62], [64, 65]]
    assert split_phrases(notes, HOP, max_gap_sec=5.0) == [[60, 62, 64, 65]]


# ---------- compute_melody (手計算) ----------

def test_ascending_scale(mcfg) -> None:
    notes, n = make_notes([60, 62, 64, 65, 67, 69, 71, 72])  # 音程 [2,2,1,2,2,2,1]
    m = compute_melody(notes, HOP, n, mcfg)
    assert m.pitch_change_rate == pytest.approx(7 / (n * HOP))
    assert m.mean_interval_semitones == pytest.approx(12 / 7)
    assert m.leap_ratio == 0.0
    assert m.smoothness == pytest.approx(1 - 0.5 / mcfg.smoothness_ref_semitones)  # |Δ²| 平均 3/6
    assert m.repetition_score == 0.0  # 4連の音程列がすべて異なる


def test_repeated_motif_has_full_repetition(mcfg) -> None:
    notes, n = make_notes([60, 64, 62, 65] * 3)
    assert compute_melody(notes, HOP, n, mcfg).repetition_score == 1.0


def test_transposed_motif_counts_as_repetition(mcfg) -> None:
    # 5音のモチーフとその +7 移調。音程列 [4,-2,3,-2, 4, 4,-2,3,-2] の4連 6個のうち、先頭と6番目が同じ
    notes, n = make_notes([60, 64, 62, 65, 63, 67, 71, 69, 72, 70])
    assert compute_melody(notes, HOP, n, mcfg).repetition_score == pytest.approx(2 / 6)


def test_large_leaps_zigzag(mcfg) -> None:
    notes, n = make_notes([60, 72] * 6)
    m = compute_melody(notes, HOP, n, mcfg)
    assert m.leap_ratio == 1.0
    assert m.mean_interval_semitones == 12.0
    assert m.smoothness == 0.0  # |Δ²| = 24 > ref でクリップ


def test_phrases_do_not_create_intervals_across_long_gaps(mcfg) -> None:
    notes, n = make_notes([60, 62, 64, 65, 67, 48, 50, 52, 53, 55], gaps={5: 100})
    m = compute_melody(notes, HOP, n, mcfg)
    assert m.leap_ratio == 0.0  # 67→48 の跳躍は数えない
    assert m.mean_interval_semitones == pytest.approx(1.75)
    assert m.repetition_score == 1.0  # 2フレーズが同じ音程列


def test_repeated_notes_are_zero_intervals(mcfg) -> None:
    notes, n = make_notes([60, 60, 60, 60, 62, 62, 62, 62])
    m = compute_melody(notes, HOP, n, mcfg)
    assert m.pitch_change_rate == pytest.approx(1 / (n * HOP))
    assert m.mean_interval_semitones == pytest.approx(2 / 7)


def test_values_stay_in_documented_ranges(mcfg) -> None:
    rng = np.random.default_rng(1)
    notes, n = make_notes([int(p) for p in rng.integers(40, 90, size=60)])
    m = compute_melody(notes, HOP, n, mcfg)
    for v in (m.leap_ratio, m.repetition_score, m.smoothness):
        assert 0.0 <= v <= 1.0


def test_too_few_notes_fails(mcfg) -> None:
    notes, n = make_notes([60, 62, 64])
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        compute_melody(notes, HOP, n, mcfg)


def test_isolated_pairs_cannot_evaluate_smoothness(mcfg) -> None:
    notes, n = make_notes([60, 62] * 4, gaps={2: 100, 4: 100, 6: 100})
    with pytest.raises(FeatureExtractionError, match="3音以上"):
        compute_melody(notes, HOP, n, mcfg)


def test_short_phrases_cannot_evaluate_repetition(mcfg) -> None:
    notes, n = make_notes([60, 62, 64, 65, 67, 69, 71, 72], gaps={4: 100})  # 4音 + 4音
    with pytest.raises(FeatureExtractionError, match="繰り返し"):
        compute_melody(notes, HOP, n, mcfg)


# ---------- 結合 (pyin) ----------

@pytest.fixture(scope="module")
def full_cfg():
    return load_config()


def test_extractor_scale_melody(full_cfg) -> None:
    scale = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 62]
    audio = AudioData(melody(scale + scale, 0.5), SR)
    m = MelodyExtractor(full_cfg.audio, full_cfg.features.melody).extract(audio)
    assert m.leap_ratio < 0.1
    assert m.mean_interval_semitones == pytest.approx(1.7, abs=0.5)
    assert m.pitch_change_rate == pytest.approx(2.0, abs=0.5)  # 0.5 秒ごとに音が変わる
    assert m.repetition_score > 0.5  # 同じ旋律を 2 回
    assert m.smoothness > 0.7


def test_extractor_leaping_melody_differs_from_scale(full_cfg) -> None:
    jumpy = [60, 72, 57, 69, 62, 74, 55, 67, 64, 76, 59, 71] * 2
    audio = AudioData(melody(jumpy, 0.5), SR)
    m = MelodyExtractor(full_cfg.audio, full_cfg.features.melody).extract(audio)
    assert m.leap_ratio > 0.8
    assert m.mean_interval_semitones > 8
    assert m.smoothness < 0.3


def test_extractor_single_sustained_tone_fails(full_cfg) -> None:
    """動きのない 1 音から、旋律の特徴を捏造しない。"""
    audio = AudioData(melody([69], 4.0), SR)
    with pytest.raises(FeatureExtractionError):
        MelodyExtractor(full_cfg.audio, full_cfg.features.melody).extract(audio)


def test_extractor_silence_fails(full_cfg) -> None:
    audio = AudioData(np.zeros(SR * 3, dtype=np.float32), SR)
    with pytest.raises(FeatureExtractionError):
        MelodyExtractor(full_cfg.audio, full_cfg.features.melody).extract(audio)


def test_min_notes_is_configurable(full_cfg) -> None:
    strict = dataclasses.replace(full_cfg.features.melody, min_notes=40)
    audio = AudioData(melody([60, 62, 64, 65, 67, 69, 71, 72], 0.5), SR)
    with pytest.raises(FeatureExtractionError, match="少なすぎ"):
        MelodyExtractor(full_cfg.audio, strict).extract(audio)
