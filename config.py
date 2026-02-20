"""
Phonk Chess Engine — Configuration Module
==========================================
All tuneable constants: colors, physics, shader params, paths.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    """Central configuration for the entire pipeline."""

    # ── Video ────────────────────────────────────────────────────────────
    width: int = 1920
    height: int = 1080
    fps: int = 30
    video_codec: str = "libx264"
    video_crf: int = 18
    video_preset: str = "fast"
    audio_bitrate: str = "192k"

    # ── Board geometry ───────────────────────────────────────────────────
    board_size: int = 800
    square_size: int = 100  # board_size / 8

    @property
    def board_origin_x(self) -> int:
        return (self.width - self.board_size) // 2

    @property
    def board_origin_y(self) -> int:
        return (self.height - self.board_size) // 2

    # ── Phonk / Cyberpunk palette ────────────────────────────────────────
    bg_color: Tuple[int, int, int] = (8, 5, 12)
    dark_square: Tuple[int, int, int] = (30, 10, 18)
    light_square: Tuple[int, int, int] = (55, 25, 38)
    white_piece_color: Tuple[int, int, int] = (0, 255, 255)      # Neon cyan
    black_piece_color: Tuple[int, int, int] = (255, 0, 85)       # Neon magenta-red
    board_border_color: Tuple[int, int, int] = (255, 0, 85)
    board_border_width: int = 3
    neon_glow_color: Tuple[int, int, int] = (255, 0, 85)
    heatmap_color: Tuple[int, int, int] = (255, 40, 40)
    san_text_color: Tuple[int, int, int] = (255, 255, 255)

    # ── Particle system ──────────────────────────────────────────────────
    particle_count: int = 200
    particle_lifetime: float = 1.5        # seconds
    particle_gravity: float = 500.0       # px / s²
    particle_speed: float = 300.0         # px / s (initial burst)
    particle_glow_radius: int = 14        # pixels
    particle_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 0, 85), (0, 255, 255), (255, 0, 255), (255, 200, 0),
    ])

    # ── Motion trails ────────────────────────────────────────────────────
    trail_count: int = 6
    trail_decay: float = 0.65             # alpha multiplier per trail step

    # ── Camera ───────────────────────────────────────────────────────────
    camera_orbit_speed: float = 0.15
    camera_zoom_default: float = 1.0
    camera_zoom_action: float = 1.35
    camera_shake_decay: float = 5.0
    camera_predictive_sec: float = 0.10   # look-ahead seconds

    # ── Audio analysis ───────────────────────────────────────────────────
    sample_rate: int = 22050
    hop_length: int = 512
    onset_strength_threshold: float = 0.5

    # ── Animation & timing ───────────────────────────────────────────────
    move_duration_base: float = 0.40      # seconds per move flight
    turbo_bpm_threshold: float = 140.0
    turbo_speed_mult: float = 0.50
    jump_height_px: float = 60.0          # arc height during flight
    squash_amount: float = 0.30           # max squash on landing

    # ── Post-processing (shader uniforms defaults) ───────────────────────
    bloom_base: float = 0.30
    chromatic_aberration_base: float = 0.20
    vignette_base: float = 1.0
    glitch_threshold: float = 0.70        # onset strength to trigger glitch
    shockwave_duration: float = 0.50      # seconds
    flash_duration: float = 0.25          # royal flash on check
    invert_duration: float = 0.40         # color invert on checkmate

    # ── Check visualisation (neon path from checker to king) ──────────
    check_path_color: Tuple[int, int, int] = (255, 0, 85)   # neon red
    check_king_color: Tuple[int, int, int] = (255, 50, 50)  # brighter for king
    min_anim_frames: int = 8              # minimum flight frames

    # ── Stockfish ────────────────────────────────────────────────────────
    stockfish_depth: int = 15

    # ── Font fallback chain (for chess unicode symbols) ──────────────────
    # On Windows, Pillow needs .ttf filenames (searched in C:\Windows\Fonts)
    font_names: List[str] = field(default_factory=lambda: [
        "seguisym.ttf",         # Segoe UI Symbol (Win 10/11)
        "segmdl2.ttf",          # Segoe MDL2 Assets
        "arialuni.ttf",         # Arial Unicode MS
        "msyh.ttc",             # Microsoft YaHei
        "Segoe UI Symbol",      # display name fallback
        "Arial Unicode MS",
        "DejaVu Sans",
        "DejaVuSans.ttf",
        "NotoSansSymbols2-Regular.ttf",
    ])
    piece_font_size: int = 72
