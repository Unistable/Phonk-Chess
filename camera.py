"""
Phonk Chess Engine — 6-DOF Cinematic Camera
=============================================
Smooth orbit, predictive focus, action zoom, bass-driven shake.
All state is expressed as 2-D UV-space transforms consumed by the
post-processing shader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from audio_analyzer import AudioFeatures
from config import Config


@dataclass
class CameraState:
    """Snapshot of camera uniforms for the current frame."""
    zoom: float
    pan_x: float
    pan_y: float
    rotation: float
    shake_x: float
    shake_y: float


class Camera:
    """6-DOF dynamic camera (orbit · zoom · shake · predictive focus)."""

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()

        # Smooth values
        self._zoom = self.cfg.camera_zoom_default
        self._target_zoom = self.cfg.camera_zoom_default
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._target_pan_x = 0.0
        self._target_pan_y = 0.0
        self._rotation = 0.0
        self._shake_int = 0.0
        self._shake_x = 0.0
        self._shake_y = 0.0
        self._orbit_angle = 0.0

    # ── Per-frame update ─────────────────────────────────────────────────

    def update(self, dt: float, t: float, audio: AudioFeatures,
               active_sq: Optional[int] = None,
               next_sq: Optional[int] = None,
               drama: float = 0.0) -> CameraState:
        """
        Parameters
        ----------
        dt          : frame delta-time (seconds)
        t           : absolute time (seconds)
        audio       : audio features for this frame
        active_sq   : target square of the active move (0-63), or None
        next_sq     : target square of the upcoming move (predictive), or None
        drama       : drama score of the active move [0, 1]
        """
        # ── Orbit drift ──────────────────────────────────────────────────
        self._orbit_angle += self.cfg.camera_orbit_speed * dt
        orb_r = 0.018
        orb_x = math.cos(self._orbit_angle) * orb_r
        orb_y = math.sin(self._orbit_angle * 0.7) * orb_r * 0.6

        # ── Predictive focus ─────────────────────────────────────────────
        focus_x, focus_y = 0.0, 0.0
        if next_sq is not None:
            fx, fy = self._sq_to_norm(next_sq)
            focus_x = fx * 0.12
            focus_y = fy * 0.12

        # ── Action zoom ──────────────────────────────────────────────────
        if active_sq is not None and drama > 0.50:
            self._target_zoom = self.cfg.camera_zoom_default + \
                (self.cfg.camera_zoom_action - self.cfg.camera_zoom_default) * drama
            ax, ay = self._sq_to_norm(active_sq)
            self._target_pan_x = ax * 0.18 * drama
            self._target_pan_y = ay * 0.18 * drama
        else:
            self._target_zoom = self.cfg.camera_zoom_default
            self._target_pan_x = 0.0
            self._target_pan_y = 0.0

        # ── Lerp smoothing ───────────────────────────────────────────────
        lf = min(3.5 * dt, 1.0)
        self._zoom += (self._target_zoom - self._zoom) * lf
        self._pan_x += (self._target_pan_x + focus_x + orb_x - self._pan_x) * lf
        self._pan_y += (self._target_pan_y + focus_y + orb_y - self._pan_y) * lf

        # ── Bass-driven shake ────────────────────────────────────────────
        if audio.is_onset:
            self._shake_int = max(self._shake_int, audio.bass_energy * 0.014)
        else:
            self._shake_int *= math.exp(-self.cfg.camera_shake_decay * dt)
        self._shake_x = (np.random.random() * 2 - 1) * self._shake_int
        self._shake_y = (np.random.random() * 2 - 1) * self._shake_int

        # ── Subtle rotation ──────────────────────────────────────────────
        self._rotation = math.sin(t * 0.45) * 0.008 + orb_x * 0.4

        return CameraState(
            zoom=self._zoom,
            pan_x=self._pan_x,
            pan_y=self._pan_y,
            rotation=self._rotation,
            shake_x=self._shake_x,
            shake_y=self._shake_y,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _sq_to_norm(sq: int) -> Tuple[float, float]:
        """Convert chess square index (0-63) to normalised offset from centre."""
        col = sq % 8
        row = sq // 8
        return ((col - 3.5) / 4.0, (row - 3.5) / 4.0)
