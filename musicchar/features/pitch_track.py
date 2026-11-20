"""音高(基本周波数)の時系列推定。D・E・C の抽出器で共有する。"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from musicchar.features.notes import Note, segment_notes
from musicchar.models.audio import AudioData
from musicchar.models.config import AudioConfig
from musicchar.models.errors import FeatureExtractionError

# 有声フレームがこれ未満だと統計量を意味のある値として扱えない。
_MIN_VOICED_FRAMES = 2


@dataclass(frozen=True, eq=False)
class PitchTrack:
    """フレームごとの音高。

    Attributes:
        midi: フレームごとの MIDI ノート番号 (float)。音高が取れなかった
            (無声・休符)フレームと、音符に属さない過渡的なフレームは NaN。
            0 などでは埋めない。
        hop_sec: 隣接フレームの時間間隔(秒)。
        notes: ``midi`` から切り出した音符列 (C の入力)。
    """

    midi: np.ndarray
    hop_sec: float
    notes: tuple[Note, ...] = ()

    @property
    def voiced_mask(self) -> np.ndarray:
        """音高が取れたフレームの真偽配列。"""
        return ~np.isnan(self.midi)

    @property
    def voiced_midi(self) -> np.ndarray:
        """音高が取れたフレームの MIDI 値のみ。"""
        return self.midi[self.voiced_mask]

    @property
    def voiced_ratio(self) -> float:
        """全フレームに対する有声フレームの割合。"""
        return float(np.mean(self.voiced_mask)) if self.midi.size else 0.0


class PitchTracker:
    """``librosa.pyin`` による音高推定。同じ音声に対する結果をキャッシュする。

    推定後、音符に属さないフレームは無声(NaN)にして取り除く。
    pyin は処理が重いため、D・E・C の抽出器が同じインスタンスを共有し、
    同一の ``AudioData`` に対しては 1 回だけ推定する。失敗も同様にキャッシュする。
    単旋律を前提とする(多声・歌声は対象外)。
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self._audio: AudioData | None = None
        self._result: PitchTrack | None = None
        self._error: FeatureExtractionError | None = None

    def track(self, audio: AudioData) -> PitchTrack:
        """音高トラックを返す(キャッシュ済みならそれを返す)。

        Raises:
            FeatureExtractionError: 波形が短すぎる、音高が推定できない、
                または有声フレームが少なすぎる場合。
        """
        if self._audio is not audio:
            self._audio, self._result, self._error = audio, None, None
            try:
                self._result = self._estimate(audio)
            except FeatureExtractionError as e:
                self._error = e
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def _estimate(self, audio: AudioData) -> PitchTrack:
        cfg = self.cfg
        if audio.y.size < cfg.frame_length:
            raise FeatureExtractionError(
                f"音声が短すぎて音高を推定できません ({audio.y.size} < {cfg.frame_length} サンプル)"
            )
        try:
            f0, _voiced_flag, _voiced_prob = librosa.pyin(
                audio.y,
                fmin=cfg.fmin_hz,
                fmax=cfg.fmax_hz,
                sr=audio.sr,
                frame_length=cfg.frame_length,
                hop_length=cfg.hop_length,
                max_transition_rate=cfg.pitch_max_transition_rate,
            )
        except Exception as e:
            raise FeatureExtractionError(f"音高の推定に失敗: {e}") from e

        # pyin は無声フレームの f0 を NaN で返す(fill_na の既定値)。
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.where(np.isfinite(f0) & (f0 > 0), librosa.hz_to_midi(f0), np.nan)

        # 音符に属さないフレーム(音の切り替わりや、音の終端で出る倍音の誤検出など)は
        # 音高として信用できないため、D・E・C のいずれの統計にも入れない。
        notes = segment_notes(raw, cfg.pitch_min_note_frames)
        midi = np.full(raw.shape, np.nan)
        for n in notes:
            midi[n.start : n.end] = raw[n.start : n.end]
        track = PitchTrack(midi=midi, hop_sec=cfg.hop_length / audio.sr, notes=tuple(notes))

        n_voiced = int(track.voiced_mask.sum())
        if n_voiced < _MIN_VOICED_FRAMES or track.voiced_ratio < cfg.pitch_min_voiced_ratio:
            raise FeatureExtractionError(
                f"音高を検出できたフレームが少なすぎます "
                f"({n_voiced} フレーム, {track.voiced_ratio:.1%} < {cfg.pitch_min_voiced_ratio:.1%})"
            )
        return track
