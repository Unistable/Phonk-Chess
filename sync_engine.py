"""
Phonk Chess Engine — Synchronisation Engine
=============================================
Maps chess moves onto audio onsets/beats so every piece landing is in perfect
sync with the music.  Supports *Turbo Mode* when BPM > threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import bisect
import numpy as np

from chess_core import ChessCore, MoveData
from audio_analyzer import AudioAnalyzer
from config import Config


@dataclass
class SyncEvent:
    """A single move with precise audio-aligned timing."""
    move_index: int
    start_time: float       # animation flight begins
    hit_time: float         # piece "lands" — synced to onset
    end_time: float         # post-impact window ends
    drama_score: float      # combined chess + audio drama  [0, 1]
    audio_energy: float     # RMS energy at hit_time        [0, 1]
    is_turbo: bool          # turbo / rapid-fire mode


class SyncEngine:
    """Beat-to-Move mapper with dynamic importance assignment."""

    def __init__(self, chess_core: ChessCore, audio: AudioAnalyzer,
                 config: Optional[Config] = None):
        self.chess = chess_core
        self.audio = audio
        self.config = config or Config()
        self.events: List[SyncEvent] = []
        self._start_times: List[float] = []   # sorted for bisect
        self._hit_times: List[float] = []     # sorted for bisect
        self._build()

    # ── Queries (binary search) ──────────────────────────────────────

    def get_active_event(self, t: float) -> Optional[SyncEvent]:
        """Return the event whose animation window contains *t*, if any."""
        if not self.events:
            return None
        # Find rightmost event whose start_time <= t
        idx = bisect.bisect_right(self._start_times, t) - 1
        if idx < 0:
            return None
        ev = self.events[idx]
        if ev.start_time <= t <= ev.end_time:
            return ev
        return None

    def get_next_event(self, t: float) -> Optional[SyncEvent]:
        """Return the first event whose start_time is > t."""
        if not self.events:
            return None
        idx = bisect.bisect_right(self._start_times, t)
        if idx < len(self.events):
            return self.events[idx]
        return None

    def get_board_index_at(self, t: float) -> int:
        """Return index of the last fully completed move, or -1."""
        if not self.events:
            return -1
        idx = bisect.bisect_right(self._hit_times, t) - 1
        if idx < 0:
            return -1
        return self.events[idx].move_index

    # ── Build ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        moves = self.chess.moves
        onsets = self.audio.get_onset_energies()

        if not moves:
            return

        is_turbo = self.audio.tempo > self.config.turbo_bpm_threshold

        if not onsets:
            self._distribute_evenly(moves, is_turbo)
            return

        onset_t = np.array([o[0] for o in onsets])
        onset_e = np.array([o[1] for o in onsets])

        if len(onset_t) >= len(moves):
            self._map_surplus_onsets(moves, onset_t, onset_e, is_turbo)
        else:
            self._map_surplus_moves(moves, onsets, is_turbo)

        # ── Post-build: ensure temporal order & prevent overlap ────────
        self.events.sort(key=lambda e: e.hit_time)
        for i in range(len(self.events) - 1):
            curr = self.events[i]
            nxt = self.events[i + 1]
            if curr.end_time > nxt.start_time:
                mid = (curr.hit_time + nxt.hit_time) / 2.0
                curr.end_time = min(curr.end_time, max(curr.hit_time + 0.02, mid))
                nxt.start_time = max(nxt.start_time, mid)

        print(f"[SyncEngine] {len(self.events)} sync events, turbo={is_turbo}")
        # Build sorted index arrays for binary search
        self._start_times = [ev.start_time for ev in self.events]
        self._hit_times = [ev.hit_time for ev in self.events]

    # ── Strategies ───────────────────────────────────────────────────────

    def _distribute_evenly(self, moves: List[MoveData], turbo: bool) -> None:
        dur = self.audio.duration
        interval = dur / len(moves)
        for i, mv in enumerate(moves):
            hit = interval * (i + 0.5)
            ad = self._anim_dur(mv.drama_score, turbo)
            self.events.append(SyncEvent(
                move_index=i,
                start_time=max(0.0, hit - ad),
                hit_time=hit,
                end_time=hit + ad * 0.4,
                drama_score=mv.drama_score,
                audio_energy=0.5,
                is_turbo=turbo,
            ))

    def _map_surplus_onsets(self, moves: List[MoveData],
                           onset_t: np.ndarray, onset_e: np.ndarray,
                           turbo: bool) -> None:
        """More onsets than moves — pick one onset per move, spread evenly."""
        indices = np.round(np.linspace(0, len(onset_t) - 1, len(moves))).astype(int)

        # Ensure monotonically non-decreasing (critical for sorted event times)
        for k in range(1, len(indices)):
            if indices[k] < indices[k - 1]:
                indices[k] = indices[k - 1]

        for i, mv in enumerate(moves):
            oi = int(indices[i])
            hit = float(onset_t[oi])
            energy = float(onset_e[oi])
            ad = self._anim_dur(mv.drama_score, turbo)
            self.events.append(SyncEvent(
                move_index=i,
                start_time=max(0.0, hit - ad),
                hit_time=hit,
                end_time=hit + ad * 0.4,
                drama_score=min(mv.drama_score * (0.5 + 0.5 * energy), 1.0),
                audio_energy=energy,
                is_turbo=turbo,
            ))

    def _map_surplus_moves(self, moves: List[MoveData],
                           onsets: List, turbo: bool) -> None:
        """More moves than onsets — distribute moves evenly between beats."""
        n_onsets = len(onsets)
        mpo = len(moves) / n_onsets  # moves per onset (float)
        mi = 0
        for oi, (t, energy) in enumerate(onsets):
            count = round(mpo)
            if oi == n_onsets - 1:
                count = len(moves) - mi
            count = max(1, min(count, len(moves) - mi))
            # Distribute moves evenly within gap to next onset
            if oi < n_onsets - 1:
                gap = onsets[oi + 1][0] - t
            else:
                gap = max(self.audio.duration - t, 0.5)
            interval = gap / max(count, 1)
            for j in range(count):
                if mi >= len(moves):
                    break
                mv = moves[mi]
                hit = t + j * interval
                ad = self._anim_dur(mv.drama_score, True)
                self.events.append(SyncEvent(
                    move_index=mi,
                    start_time=max(0.0, hit - ad),
                    hit_time=hit,
                    end_time=hit + ad * 0.4,
                    drama_score=min(mv.drama_score * (0.5 + 0.5 * energy), 1.0),
                    audio_energy=energy,
                    is_turbo=True,
                ))
                mi += 1

    # ── Helpers ──────────────────────────────────────────────────────────

    def _anim_dur(self, drama: float, turbo: bool) -> float:
        base = self.config.move_duration_base
        if turbo:
            base *= self.config.turbo_speed_mult
        dur = base * (0.7 + 0.6 * drama)
        # Guarantee minimum animation (prevent instant-looking moves)
        min_dur = self.config.min_anim_frames / max(self.config.fps, 1)
        return max(dur, min_dur)

    @staticmethod
    def _optimize_alignment(indices: np.ndarray, moves: List[MoveData],
                            onset_e: np.ndarray) -> None:
        """Greedy adjacent-swap optimisation to align high drama ↔ high energy."""
        drama = np.array([m.drama_score for m in moves])
        n = len(indices)
        for _ in range(min(n, 30)):
            improved = False
            for i in range(n - 1):
                a, b = int(indices[i]), int(indices[i + 1])
                cur = (drama[i] - onset_e[a]) ** 2 + (drama[i + 1] - onset_e[b]) ** 2
                swp = (drama[i] - onset_e[b]) ** 2 + (drama[i + 1] - onset_e[a]) ** 2
                if swp < cur * 0.8:
                    indices[i], indices[i + 1] = indices[i + 1], indices[i]
                    improved = True
            if not improved:
                break
