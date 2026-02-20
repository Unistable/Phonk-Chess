"""
Phonk Chess Engine — Audio Analyzer
=====================================
Multi-band frequency analysis, onset / beat detection, RMS energy —
all powered by *librosa*.  The analyzer pre-computes every feature once
and exposes a fast per-frame query ``get_features_at_time(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import librosa

from config import Config


@dataclass
class AudioFeatures:
    """Snapshot of audio characteristics at a single moment in time."""
    bass_energy: float       # 0-1  (< 250 Hz)
    mid_energy: float        # 0-1  (250 – 4 000 Hz)
    high_energy: float       # 0-1  (> 4 000 Hz)
    rms_energy: float        # 0-1  overall loudness
    onset_strength: float    # 0-1  beat "hit-ness"
    is_onset: bool           # True within ±50 ms of a detected onset


class AudioAnalyzer:
    """Loads an audio file and extracts time-indexed features for the renderer."""

    def __init__(self, audio_path: str, config: Optional[Config] = None):
        self.audio_path = audio_path
        self.config = config or Config()
        self.y: np.ndarray
        self.sr: int
        self.duration: float
        self.tempo: float
        self.onset_times: np.ndarray
        self.beat_times: np.ndarray
        self._load_and_analyze()

    # ── Public API ───────────────────────────────────────────────────────

    def get_features_at_time(self, t: float) -> AudioFeatures:
        """Return audio features for timestamp *t* (seconds)."""
        frame = self._time_to_frame(t)
        stft_frame = min(frame, self._n_stft - 1)
        onset_frame = min(frame, self._n_onset - 1)
        rms_frame = min(frame, self._n_rms - 1)

        is_onset = False
        if self.onset_times.size > 0:
            is_onset = bool(np.min(np.abs(self.onset_times - t)) < 0.05)

        return AudioFeatures(
            bass_energy=float(self._bass_n[stft_frame]),
            mid_energy=float(self._mids_n[stft_frame]),
            high_energy=float(self._highs_n[stft_frame]),
            rms_energy=float(self._rms_n[rms_frame]),
            onset_strength=float(self._onset_n[onset_frame]),
            is_onset=is_onset,
        )

    def get_onset_energies(self) -> List[Tuple[float, float]]:
        """Return ``[(time, rms_energy), ...]`` for every detected onset."""
        out: List[Tuple[float, float]] = []
        for t in self.onset_times:
            f = min(self._time_to_frame(t), self._n_rms - 1)
            out.append((float(t), float(self._rms_n[f])))
        return out

    # ── Internal ─────────────────────────────────────────────────────────

    def _time_to_frame(self, t: float) -> int:
        return int(np.clip(
            librosa.time_to_frames(t, sr=self.sr, hop_length=self.config.hop_length),
            0, max(self._n_rms - 1, 0),
        ))

    def _load_and_analyze(self) -> None:
        print(f"[AudioAnalyzer] Loading {self.audio_path} …")
        hop = self.config.hop_length
        self.y, self.sr = librosa.load(self.audio_path,
                                       sr=self.config.sample_rate, mono=True)
        self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))
        print(f"[AudioAnalyzer] Duration: {self.duration:.1f}s  SR: {self.sr}")

        # ── Onset detection ──────────────────────────────────────────────
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop)
        onset_frames = librosa.onset.onset_detect(
            y=self.y, sr=self.sr, hop_length=hop, onset_envelope=onset_env,
        )
        self.onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=hop)

        # ── Beat tracking ────────────────────────────────────────────────
        _tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr, hop_length=hop)
        # librosa may return ndarray or scalar depending on version
        if hasattr(_tempo, "__len__"):
            self.tempo = float(_tempo[0]) if len(_tempo) > 0 else 120.0
        else:
            self.tempo = float(_tempo)
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=hop)
        print(f"[AudioAnalyzer] Tempo ≈ {self.tempo:.0f} BPM, "
              f"{len(self.onset_times)} onsets, {len(self.beat_times)} beats")

        # ── RMS energy ───────────────────────────────────────────────────
        rms = librosa.feature.rms(y=self.y, hop_length=hop)[0]
        self._rms_n = rms / (np.max(rms) + 1e-8)
        self._n_rms = len(self._rms_n)

        # ── Onset envelope (normalised) ──────────────────────────────────
        self._onset_n = onset_env / (np.max(onset_env) + 1e-8)
        self._n_onset = len(self._onset_n)

        # ── Multi-band energy (STFT) ────────────────────────────────────
        S = np.abs(librosa.stft(self.y, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=self.sr)

        low_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = freqs >= 4000

        def _band(mask: np.ndarray) -> np.ndarray:
            if not np.any(mask):
                return np.zeros(S.shape[1])
            band = np.mean(S[mask, :], axis=0)
            return band / (np.max(band) + 1e-8)

        self._bass_n = _band(low_mask)
        self._mids_n = _band(mid_mask)
        self._highs_n = _band(high_mask)
        self._n_stft = S.shape[1]
