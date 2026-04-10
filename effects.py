"""
Phonk Chess Engine — Effects Manager
======================================
Particle explosions, neon motion-trails, heatmap glow, shockwave /
flash / invert state tracking.  All CPU-side state that feeds into
the GPU post-processing shader or is composited in the scene pass.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess
import chess.polyglot

from config import Config


# ═════════════════════════════════════════════════════════════════════════
# Particle System
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float
    max_life: float
    size: float


class ParticleSystem:
    """Neon particle explosion with gravity & glow — vectorised via NumPy."""

    def __init__(self, config: Config):
        self.cfg = config
        # Pre-compute glow kernel once
        r = config.particle_glow_radius
        yg, xg = np.ogrid[-r:r + 1, -r:r + 1]
        dist = np.sqrt(xg ** 2 + yg ** 2) / r
        self._glow = np.clip(1.0 - dist, 0.0, 1.0) ** 2  # (2r+1, 2r+1)
        self._r = r
        # Particle arrays (SoA — Structure of Arrays for vectorised update)
        self._max_particles = 2000
        self._n = 0  # current alive count
        self._x  = np.zeros(self._max_particles, dtype=np.float32)
        self._y  = np.zeros(self._max_particles, dtype=np.float32)
        self._vx = np.zeros(self._max_particles, dtype=np.float32)
        self._vy = np.zeros(self._max_particles, dtype=np.float32)
        self._life = np.zeros(self._max_particles, dtype=np.float32)
        self._max_life = np.zeros(self._max_particles, dtype=np.float32)
        self._cr = np.zeros(self._max_particles, dtype=np.float32)
        self._cg = np.zeros(self._max_particles, dtype=np.float32)
        self._cb = np.zeros(self._max_particles, dtype=np.float32)

    def emit(self, cx: float, cy: float, count: int,
             color: Tuple[int, int, int]) -> None:
        """Burst *count* particles from (cx, cy)."""
        space = self._max_particles - self._n
        count = min(count, space)
        if count <= 0:
            return
        s = self._n
        e = s + count
        angles = np.random.random(count).astype(np.float32) * (2.0 * math.pi)
        speeds = self.cfg.particle_speed * (0.3 + np.random.random(count).astype(np.float32) * 0.7)
        self._x[s:e] = cx
        self._y[s:e] = cy
        self._vx[s:e] = np.cos(angles) * speeds
        self._vy[s:e] = np.sin(angles) * speeds - self.cfg.particle_speed * 0.5
        lt = self.cfg.particle_lifetime * (0.6 + np.random.random(count).astype(np.float32) * 0.4)
        self._life[s:e] = lt
        self._max_life[s:e] = lt
        self._cr[s:e] = float(color[0])
        self._cg[s:e] = float(color[1])
        self._cb[s:e] = float(color[2])
        self._n = e

    def update(self, dt: float) -> None:
        if self._n == 0:
            return
        n = self._n
        self._life[:n] -= dt
        alive = self._life[:n] > 0
        if not np.any(alive):
            self._n = 0
            return
        # Physics (in-place on alive slice)
        self._vy[:n] += self.cfg.particle_gravity * dt
        self._x[:n] += self._vx[:n] * dt
        self._y[:n] += self._vy[:n] * dt
        self._vx[:n] *= 0.98
        # Compact dead particles
        if not np.all(alive):
            new_n = int(np.sum(alive))
            idx = np.where(alive)[0]
            for arr in (self._x, self._y, self._vx, self._vy,
                        self._life, self._max_life, self._cr, self._cg, self._cb):
                arr[:new_n] = arr[idx]
            self._n = new_n

    @property
    def particles(self):
        """Compatibility: list-like truth check."""
        return range(self._n)  # truthy when n > 0

    def composite(self, canvas: np.ndarray) -> None:
        """Additively blend all living particles onto *canvas* (H×W×3, uint8)."""
        if self._n == 0:
            return
        h, w = canvas.shape[:2]
        r = self._r
        glow = self._glow
        n = self._n

        buf = canvas.astype(np.float32)
        alphas = np.power(self._life[:n] / self._max_life[:n], 0.8)
        ixs = self._x[:n].astype(np.int32)
        iys = self._y[:n].astype(np.int32)

        for i in range(n):
            alpha = alphas[i]
            ix, iy = int(ixs[i]), int(iys[i])
            y1 = max(0, iy - r);  y2 = min(h, iy + r + 1)
            x1 = max(0, ix - r);  x2 = min(w, ix + r + 1)
            if y1 >= y2 or x1 >= x2:
                continue
            gy1 = y1 - (iy - r);  gy2 = y2 - (iy - r)
            gx1 = x1 - (ix - r);  gx2 = x2 - (ix - r)
            patch = glow[gy1:gy2, gx1:gx2, np.newaxis]
            col = np.array([self._cr[i], self._cg[i], self._cb[i]], dtype=np.float32)
            buf[y1:y2, x1:x2] += patch * col * alpha

        np.clip(buf, 0, 255, out=buf)
        canvas[:] = buf.astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════
# Motion Trail
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class TrailGhost:
    """A single after-image of a moving piece."""
    x: float
    y: float
    alpha: float              # current opacity [0, 1]
    piece_type: chess.PieceType
    piece_color: chess.Color


class TrailManager:
    """Stores fading after-images for the piece currently in flight."""

    def __init__(self, config: Config):
        self.cfg = config
        self.ghosts: List[TrailGhost] = []

    def record(self, x: float, y: float,
               piece_type: chess.PieceType, piece_color: chess.Color) -> None:
        """Add a new ghost; trim to max trail count."""
        self.ghosts.append(TrailGhost(x, y, 1.0, piece_type, piece_color))
        if len(self.ghosts) > self.cfg.trail_count:
            self.ghosts = self.ghosts[-self.cfg.trail_count:]

    def clear(self) -> None:
        self.ghosts.clear()

    def decayed(self) -> List[TrailGhost]:
        """Return ghosts with progressively decayed alpha (oldest = faintest)."""
        out: List[TrailGhost] = []
        n = len(self.ghosts)
        for i, g in enumerate(self.ghosts):
            factor = self.cfg.trail_decay ** (n - i)
            out.append(TrailGhost(g.x, g.y, factor, g.piece_type, g.piece_color))
        return out


# ═════════════════════════════════════════════════════════════════════════
# Heatmap — attacked-square glow
# ═════════════════════════════════════════════════════════════════════════

# Heatmap cache with LRU eviction (max 200 entries)
_heatmap_cache: OrderedDict[int, np.ndarray] = OrderedDict()
_HEATMAP_CACHE_MAX_SIZE = 200


def compute_heatmap(board: chess.Board) -> np.ndarray:
    """Return an 8×8 float array where each cell = number of attackers (both sides).
    Cached per position (zobrist hash) with LRU eviction to prevent memory leak."""
    key = chess.polyglot.zobrist_hash(board)
    
    # Check cache with LRU update
    if key in _heatmap_cache:
        _heatmap_cache.move_to_end(key)
        return _heatmap_cache[key]
    
    hmap = np.zeros((8, 8), dtype=np.float32)
    for sq in range(64):
        col = sq % 8
        row = sq // 8
        w_att = len(board.attackers(chess.WHITE, sq))
        b_att = len(board.attackers(chess.BLACK, sq))
        hmap[row, col] = float(w_att + b_att)
    
    mx = hmap.max()
    if mx > 0:
        hmap /= mx
    
    # Evict oldest entries if cache is full (LRU policy)
    while len(_heatmap_cache) >= _HEATMAP_CACHE_MAX_SIZE:
        _heatmap_cache.popitem(last=False)
    
    _heatmap_cache[key] = hmap
    return hmap


# ═════════════════════════════════════════════════════════════════════════
# Global Effect State (consumed by shader uniforms)
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class EffectsState:
    """Mutable per-frame state driven by sync events + audio."""
    # Shockwave (radial distortion from landing square)
    shockwave_cx: float = 0.5
    shockwave_cy: float = 0.5
    shockwave_time: float = -1.0          # negative = inactive

    # Royal flash / invert
    flash_amount: float = 0.0             # 0-1
    invert_amount: float = 0.0            # 0-1

    # Dynamic post-FX intensities (added on top of base)
    bloom_extra: float = 0.0
    ca_extra: float = 0.0
    glitch_amount: float = 0.0


class EffectManager:
    """Orchestrates particles, trails, shockwave, flash, and per-frame FX state."""

    def __init__(self, config: Config):
        self.cfg = config
        self.particles = ParticleSystem(config)
        self.trails = TrailManager(config)
        self.state = EffectsState()
        self._flash_timer = 0.0
        self._invert_timer = 0.0
        self._shockwave_timer = 0.0

    # ── Triggers ─────────────────────────────────────────────────────────

    def trigger_capture(self, px: float, py: float,
                        color: Tuple[int, int, int]) -> None:
        """Emit particle explosion at pixel position."""
        self.particles.emit(px, py, self.cfg.particle_count, color)

    def trigger_shockwave(self, norm_x: float, norm_y: float) -> None:
        """Start shockwave at normalised screen coords."""
        self.state.shockwave_cx = norm_x
        self.state.shockwave_cy = norm_y
        self._shockwave_timer = self.cfg.shockwave_duration

    def trigger_check_flash(self) -> None:
        self._flash_timer = self.cfg.flash_duration

    def trigger_checkmate_invert(self) -> None:
        self._invert_timer = self.cfg.invert_duration

    # ── Per-frame update ─────────────────────────────────────────────────

    def update(self, dt: float, audio_bass: float, audio_high: float,
               onset_strength: float) -> None:
        """Advance timers, decay effects, update particles."""
        self.particles.update(dt)

        # Shockwave
        if self._shockwave_timer > 0:
            self._shockwave_timer -= dt
            self.state.shockwave_time = 1.0 - (self._shockwave_timer /
                                                 self.cfg.shockwave_duration)
        else:
            self.state.shockwave_time = -1.0

        # Flash
        if self._flash_timer > 0:
            self._flash_timer -= dt
            self.state.flash_amount = max(self._flash_timer / self.cfg.flash_duration, 0.0)
        else:
            self.state.flash_amount = 0.0

        # Invert
        if self._invert_timer > 0:
            self._invert_timer -= dt
            self.state.invert_amount = max(self._invert_timer / self.cfg.invert_duration, 0.0)
        else:
            self.state.invert_amount = 0.0

        # Audio-reactive FX
        self.state.bloom_extra = audio_bass * 0.5
        self.state.ca_extra = audio_bass * 0.4
        self.state.glitch_amount = max(0.0, onset_strength - self.cfg.glitch_threshold) \
                                    / (1.0 - self.cfg.glitch_threshold + 1e-8) * audio_high
