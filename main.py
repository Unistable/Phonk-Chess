#!/usr/bin/env python3
"""
Phonk Chess Engine — Main Entry Point
=======================================
CLI orchestrator: parses arguments, wires up every sub-system,
runs the per-frame render loop and pipes raw RGB into FFmpeg
for final MP4 export.

Usage
-----
    python main.py --pgn game.pgn --audio track.mp3 --output result.mp4 [--stockfish ./stockfish]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import chess
import numpy as np

from config import Config
from chess_core import ChessCore
from audio_analyzer import AudioAnalyzer
from sync_engine import SyncEngine
from camera import Camera
from effects import EffectManager, compute_heatmap
from renderer import Renderer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Check-path helpers (attacker → king neon line)
# ═════════════════════════════════════════════════════════════════════════

def _attack_ray(from_sq: int, to_sq: int) -> list:
    """Return list of square indices along the attack ray (inclusive)."""
    fr, fc = from_sq // 8, from_sq % 8
    tr, tc = to_sq // 8, to_sq % 8
    dr, dc = tr - fr, tc - fc
    # Knight: direct jump, no intermediate squares
    if (abs(dr), abs(dc)) in ((1, 2), (2, 1)):
        return [from_sq, to_sq]
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    squares = []
    r, c = fr, fc
    for _ in range(8):
        squares.append(r * 8 + c)
        if r == tr and c == tc:
            break
        r += sr
        c += sc
    return squares


def _compute_check_path(board: chess.Board) -> list:
    """Return list of square indices forming the neon check path."""
    if not board.is_check():
        return []
    king_sq = board.king(board.turn)
    if king_sq is None:
        return []
    path: set = set()
    for atk_sq in board.checkers():
        path.update(_attack_ray(atk_sq, king_sq))
    return list(path)


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.webm'}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phonk Chess Edit Generator — turns PGN + audio into an aggressive MP4.",
    )
    ap.add_argument("--pgn", required=True, help="Path to .pgn file")
    ap.add_argument("--audio", required=True, help="Path to audio file (mp3/wav/flac/ogg/m4a)")
    ap.add_argument("--output", required=True, help="Output video path (.mp4/.mkv/.avi/.webm)")
    ap.add_argument("--stockfish", default=None,
                    help="Path to Stockfish binary (optional, for drama scoring)")
    ap.add_argument("--fps", type=int, default=30, help="Frame rate (default 30)")
    ap.add_argument("--width", type=int, default=1920, help="Video width")
    ap.add_argument("--height", type=int, default=1080, help="Video height")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to JSON configuration file (optional)")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return ap.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate all input files and parameters before processing."""
    # Validate PGN file
    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {args.pgn}")
    if not pgn_path.suffix.lower() == '.pgn':
        logger.warning(f"PGN file has unusual extension: {pgn_path.suffix}")
    
    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        logger.warning(
            f"Audio file format '{audio_path.suffix}' may not be supported. "
            f"Supported formats: {SUPPORTED_AUDIO_FORMATS}"
        )
    
    # Validate output path
    output_path = Path(args.output)
    if output_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        logger.warning(
            f"Output format '{output_path.suffix}' may not be supported. "
            f"Supported formats: {SUPPORTED_VIDEO_FORMATS}"
        )
    # Ensure parent directory exists
    if not output_path.parent.exists():
        logger.info(f"Creating output directory: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate Stockfish path if provided
    if args.stockfish:
        stockfish_path = Path(args.stockfish)
        if not stockfish_path.exists():
            raise FileNotFoundError(f"Stockfish binary not found: {args.stockfish}")
        if not os.access(stockfish_path, os.X_OK):
            raise PermissionError(f"Stockfish binary is not executable: {args.stockfish}")
    
    # Validate FPS
    if args.fps < 1 or args.fps > 120:
        raise ValueError(f"FPS must be between 1 and 120, got: {args.fps}")
    
    # Validate dimensions
    if args.width < 320 or args.height < 240:
        raise ValueError(f"Minimum resolution is 320x240, got: {args.width}x{args.height}")
    if args.width > 7680 or args.height > 4320:
        raise ValueError(f"Maximum resolution is 7680x4320, got: {args.width}x{args.height}")
    
    logger.info(f"Input validation passed: PGN={args.pgn}, Audio={args.audio}, Output={args.output}")


# ═════════════════════════════════════════════════════════════════════════
# FFmpeg pipe wrapper
# ═════════════════════════════════════════════════════════════════════════

class FFmpegPipe:
    """Writes raw RGB24 frames into an FFmpeg subprocess that muxes video + audio.
    
    Thread-safe implementation with proper error handling and race condition prevention.
    """

    def __init__(self, config: Config, audio_path: str, output_path: str):
        self._lock = threading.Lock()
        self._closed = False
        self._returncode: Optional[int] = None
        
        # Check if ffmpeg is available
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=5
            )
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg:\n"
                "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  - macOS: brew install ffmpeg\n"
                "  - Windows: Download from https://ffmpeg.org/download.html"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg check timed out. Is FFmpeg installed correctly?")
        
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{config.width}x{config.height}",
            "-pix_fmt", "rgb24",
            "-r", str(config.fps),
            "-i", "pipe:0",
            "-i", audio_path,
            "-c:v", config.video_codec,
            "-preset", config.video_preset,
            "-crf", str(config.video_crf),
            "-c:a", "aac",
            "-b:a", config.audio_bitrate,
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10 * 1024 * 1024,
            )
        except FileNotFoundError:
            raise RuntimeError("FFmpeg executable not found in PATH")
        except PermissionError:
            raise RuntimeError("Permission denied when trying to run FFmpeg")

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single (H, W, 3) uint8 frame. Thread-safe."""
        if self._closed:
            raise RuntimeError("Cannot write frame: FFmpegPipe is closed")
        
        with self._lock:
            try:
                if self._proc.stdin is None:
                    raise RuntimeError("FFmpeg stdin is not available")
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError as e:
                logger.error("FFmpeg pipe broken - encoding may have failed")
                self._returncode = self._proc.poll()
                raise RuntimeError(f"FFmpeg pipe broken (exit code: {self._returncode})") from e
            except OSError as e:
                logger.error(f"OS error writing to FFmpeg: {e}")
                raise

    def close(self) -> int:
        """Close the pipe and wait for FFmpeg to finish. Thread-safe."""
        if self._closed:
            return self._returncode or 0
        
        with self._lock:
            if self._closed:
                return self._returncode or 0
            
            self._closed = True
            
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except BrokenPipeError:
                pass
            
            try:
                self._proc.wait(timeout=60)
                self._returncode = self._proc.returncode
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg did not finish within 60s, terminating...")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                    self._returncode = self._proc.returncode
                except subprocess.TimeoutExpired:
                    logger.error("FFmpeg termination failed, killing process...")
                    self._proc.kill()
                    self._proc.wait()
                    self._returncode = -1
            
            return self._returncode or 0


# ═════════════════════════════════════════════════════════════════════════
# Render loop
# ═════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    """Main render loop orchestrator."""
    try:
        # Validate inputs before starting
        validate_inputs(args)
        
        cfg = Config(width=args.width, height=args.height, fps=args.fps)

        logger.info("Starting Phonk Chess Engine v2.0")
        logger.info(f"Configuration: {args.width}x{args.height}@{args.fps}fps")

        # ── 1. Parse chess game ──────────────────────────────────────────────
        logger.info("Parsing PGN...")
        chess_core = ChessCore(args.pgn, stockfish_path=args.stockfish, config=cfg)

        # ── 2. Analyse audio ─────────────────────────────────────────────────
        logger.info("Analysing audio...")
        audio = AudioAnalyzer(args.audio, config=cfg)

        # ── 3. Build sync map ────────────────────────────────────────────────
        logger.info("Building beat-to-move sync...")
        sync = SyncEngine(chess_core, audio, config=cfg)

        # ── 4. Initialise renderer & effects ─────────────────────────────────
        logger.info("Initialising GPU renderer...")
        renderer = Renderer(cfg)
        camera = Camera(cfg)
        effects = EffectManager(cfg)

        # ── 5. Render loop ───────────────────────────────────────────────────
        logger.info("Rendering frames...")
        total_frames = int(audio.duration * cfg.fps)
        dt = 1.0 / cfg.fps

        pipe = FFmpegPipe(cfg, args.audio, args.output)
        t0_wall = time.time()

        starting_board = chess_core.starting_board()

        # Track which event was "just triggered" to fire one-shot effects
        last_triggered_idx = -1
        trail_interval_counter = 0

        # Progress tracking
        last_progress_frame = 0
        progress_interval = max(1, cfg.fps)  # Update progress every second

        for fi in range(total_frames):
            t = fi * dt

            # ── Audio features ───────────────────────────────────────────────
            af = audio.get_features_at_time(t)

            # ── Sync state ───────────────────────────────────────────────────
            ev = sync.get_active_event(t)
            nxt = sync.get_next_event(t)
            board_idx = sync.get_board_index_at(t)

            # Board to display
            if board_idx >= 0:
                board = chess_core.moves[board_idx].board_after
            else:
                board = starting_board

            # ── Animation state ──────────────────────────────────────────────
            anim_piece = None
            move_data = None
            drama = 0.0

            if ev is not None:
                move_data = chess_core.moves[ev.move_index]
                drama = ev.drama_score

            if t < ev.hit_time:
                # ── In-flight animation ──────────────────────────────────
                # Show board_before (before this move) with piece removed from source
                display_board = move_data.board_before
                board = display_board  # override for rendering

                progress = (t - ev.start_time) / max(ev.hit_time - ev.start_time, 1e-6)
                progress = min(max(progress, 0.0), 1.0)

                fpx, fpy = renderer.square_to_px(move_data.from_square)
                tpx, tpy = renderer.square_to_px(move_data.to_square)

                anim_piece = (
                    move_data.piece_type, move_data.piece_color,
                    fpx, fpy, tpx, tpy, progress,
                )

                # Trail recording (every other frame)
                trail_interval_counter += 1
                if trail_interval_counter % 2 == 0:
                    ep = 1.0 - (1.0 - progress) ** 5
                    cur_x = fpx + (tpx - fpx) * ep
                    cur_y = fpy + (tpy - fpy) * ep - math.sin(progress * math.pi) * cfg.jump_height_px
                    effects.trails.record(cur_x, cur_y,
                                          move_data.piece_type, move_data.piece_color)
            else:
                # Post-hit: board_after is already set above
                effects.trails.clear()
                trail_interval_counter = 0

            # ── One-shot triggers (fire once per event) ──────────────────
            if ev.move_index != last_triggered_idx and t >= ev.hit_time:
                last_triggered_idx = ev.move_index

                # Shockwave at landing square
                sx, sy = renderer.square_to_uv(move_data.to_square)
                effects.trigger_shockwave(sx, sy)

                # Capture → particle explosion
                if move_data.is_capture:
                    cx, cy = renderer.square_center_px(move_data.to_square)
                    exp_color = (cfg.black_piece_color
                                 if move_data.piece_color == chess.WHITE
                                 else cfg.white_piece_color)
                    effects.trigger_capture(cx, cy, exp_color)

                # Check flash
                if move_data.is_check:
                    effects.trigger_check_flash()

                # Checkmate invert
                if move_data.is_checkmate:
                    effects.trigger_checkmate_invert()
        else:
            effects.trails.clear()
            trail_interval_counter = 0

        # ── Update sub-systems ───────────────────────────────────────────
        active_sq = move_data.to_square if move_data else None
        next_sq = None
        if nxt is not None and nxt.move_index < len(chess_core.moves):
            next_sq = chess_core.moves[nxt.move_index].to_square

        cam = camera.update(dt, t, af,
                            active_sq=active_sq,
                            next_sq=next_sq,
                            drama=drama)

        effects.update(dt, af.bass_energy, af.high_energy, af.onset_strength)

        # ── Heatmap ──────────────────────────────────────────────────────
        hmap = compute_heatmap(board)

        # ── Check path (neon highlight from checker to king) ─────────────
        check_path = _compute_check_path(board)

        # ── SAN overlay ──────────────────────────────────────────────────
        san = None
        if ev is not None and t >= ev.hit_time and t < ev.hit_time + 0.6:
            san = chess_core.moves[ev.move_index].san

        # ── Render ───────────────────────────────────────────────────────
        frame = renderer.render_frame(
            board=board,
            anim_piece=anim_piece,
            effect_mgr=effects,
            cam=cam,
            audio_bass=af.bass_energy,
            audio_mid=af.mid_energy,
            audio_high=af.high_energy,
            onset_strength=af.onset_strength,
            time_sec=t,
            san_text=san,
            heatmap=hmap,
            check_path=check_path,
        )

        pipe.write_frame(frame)

        # ── Progress ─────────────────────────────────────────────────────
        if not args.quiet and (fi % progress_interval == 0 or fi == total_frames - 1):
            elapsed = time.time() - t0_wall
            pct = (fi + 1) / total_frames * 100
            fps_real = (fi + 1) / max(elapsed, 0.01)
            eta = (total_frames - fi - 1) / max(fps_real, 0.01)
            sys.stdout.write(
                f"\r  [{pct:5.1f}%]  frame {fi + 1}/{total_frames}  "
                f"@ {fps_real:.1f} fps   ETA {eta:.0f}s   "
            )
            sys.stdout.flush()

        # ── Finalise ─────────────────────────────────────────────────────────
    logger.info("Encoding final MP4...")
    rc = pipe.close()
    renderer.destroy()
    
    if rc == 0:
        sz = os.path.getsize(args.output) / (1024 * 1024)
        logger.info(f"Success! Output: {args.output} ({sz:.1f} MB)")
    else:
        logger.error(f"FFmpeg exited with code {rc}")
        raise RuntimeError(f"Video encoding failed with exit code {rc}")
        
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    sys.exit(1)
except PermissionError as e:
    logger.error(f"Permission denied: {e}")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    sys.exit(1)
except RuntimeError as e:
    logger.error(f"Runtime error: {e}")
    sys.exit(1)
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# Entry
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(parse_args())
