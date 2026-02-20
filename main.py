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
import math
import os
import subprocess
import sys
import time
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

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phonk Chess Edit Generator — turns PGN + audio into an aggressive MP4.",
    )
    ap.add_argument("--pgn", required=True, help="Path to .pgn file")
    ap.add_argument("--audio", required=True, help="Path to audio file (mp3/wav/flac)")
    ap.add_argument("--output", required=True, help="Output .mp4 path")
    ap.add_argument("--stockfish", default=None,
                    help="Path to Stockfish binary (optional, for drama scoring)")
    ap.add_argument("--fps", type=int, default=30, help="Frame rate (default 30)")
    ap.add_argument("--width", type=int, default=1920, help="Video width")
    ap.add_argument("--height", type=int, default=1080, help="Video height")
    return ap.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# FFmpeg pipe wrapper
# ═════════════════════════════════════════════════════════════════════════

class FFmpegPipe:
    """Writes raw RGB24 frames into an FFmpeg subprocess that muxes video + audio."""

    def __init__(self, config: Config, audio_path: str, output_path: str):
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning",
            # Raw video input via pipe
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{config.width}x{config.height}",
            "-pix_fmt", "rgb24",
            "-r", str(config.fps),
            "-i", "pipe:0",
            # Audio input
            "-i", audio_path,
            # Encoding
            "-c:v", config.video_codec,
            "-preset", config.video_preset,
            "-crf", str(config.video_crf),
            "-c:a", "aac",
            "-b:a", config.audio_bitrate,
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        # CRITICAL: stderr=DEVNULL prevents deadlock on Windows
        # (FFmpeg writes lots of output to stderr; if the pipe buffer
        #  fills up, both FFmpeg and our process block forever).
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=10 * 1024 * 1024,  # 10 MB write buffer
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single (H, W, 3) uint8 frame."""
        try:
            self._proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as e:
            print(f"\n[FFmpeg] Pipe broken: {e}", file=sys.stderr)
            raise

    def close(self) -> int:
        try:
            self._proc.stdin.close()  # type: ignore[union-attr]
        except BrokenPipeError:
            pass
        self._proc.wait()
        return self._proc.returncode


# ═════════════════════════════════════════════════════════════════════════
# Render loop
# ═════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    cfg = Config(width=args.width, height=args.height, fps=args.fps)

    print("═" * 60)
    print("  Phonk Chess Engine v2.0")
    print("═" * 60)

    # ── 1. Parse chess game ──────────────────────────────────────────────
    print("\n[1/5] Parsing PGN …")
    chess_core = ChessCore(args.pgn, stockfish_path=args.stockfish, config=cfg)

    # ── 2. Analyse audio ─────────────────────────────────────────────────
    print("[2/5] Analysing audio …")
    audio = AudioAnalyzer(args.audio, config=cfg)

    # ── 3. Build sync map ────────────────────────────────────────────────
    print("[3/5] Building beat-to-move sync …")
    sync = SyncEngine(chess_core, audio, config=cfg)

    # ── 4. Initialise renderer & effects ─────────────────────────────────
    print("[4/5] Initialising GPU renderer …")
    renderer = Renderer(cfg)
    camera = Camera(cfg)
    effects = EffectManager(cfg)

    # ── 5. Render loop ───────────────────────────────────────────────────
    print("[5/5] Rendering frames …\n")
    total_frames = int(audio.duration * cfg.fps)
    dt = 1.0 / cfg.fps

    pipe = FFmpegPipe(cfg, args.audio, args.output)
    t0_wall = time.time()

    starting_board = chess_core.starting_board()

    # Track which event was "just triggered" to fire one-shot effects
    last_triggered_idx = -1
    # Track last completed move index for board state
    # Also keep a snapshot of the trail recording interval
    trail_interval_counter = 0

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
        if fi % cfg.fps == 0 or fi == total_frames - 1:
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
    print("\n\n[Done] Encoding final MP4 …")
    rc = pipe.close()
    renderer.destroy()

    if rc == 0:
        sz = os.path.getsize(args.output) / (1024 * 1024)
        print(f"[Success] {args.output}  ({sz:.1f} MB)")
    else:
        print(f"[Error] FFmpeg exited with code {rc}", file=sys.stderr)
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# Entry
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(parse_args())
