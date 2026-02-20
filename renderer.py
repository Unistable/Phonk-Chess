"""
Phonk Chess Engine — GPU-Accelerated Renderer
===============================================
Two-stage pipeline:
  1. **CPU scene pass** – Pillow / NumPy: board, pieces, heatmap, trails, particles.
  2. **GPU post-process** – ModernGL (OpenGL 3.3+): bloom, chromatic aberration,
     glitch, RGB-shift, vignette, shockwave, flash, invert, camera transform.

Full GLSL code is embedded as Python strings — no external shader files.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import moderngl
except ImportError:
    moderngl = None  # type: ignore[assignment]

from config import Config
from camera import CameraState
from effects import EffectsState, EffectManager, TrailGhost, compute_heatmap

# ═════════════════════════════════════════════════════════════════════════
# GLSL Shaders
# ═════════════════════════════════════════════════════════════════════════

VERT_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

FRAG_SHADER = """
#version 330

uniform sampler2D u_scene;
uniform float u_time;

// Camera
uniform float u_zoom;
uniform vec2  u_pan;
uniform float u_rotation;
uniform vec2  u_shake;

// Audio-reactive
uniform float u_bass;
uniform float u_mid;
uniform float u_high;
uniform float u_onset;

// Post-FX
uniform float u_bloom;
uniform float u_ca;
uniform float u_glitch;
uniform float u_vignette;
uniform float u_flash;
uniform float u_invert;

// Shockwave
uniform vec2  u_sw_center;
uniform float u_sw_time;

in  vec2 v_uv;
out vec4 fragColor;

// ── Utility ─────────────────────────────────────────────────────────
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

// ── Camera transform (2D UV space) ──────────────────────────────────
vec2 cameraUV(vec2 uv) {
    uv -= 0.5;
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    uv = mat2(c, -s, s, c) * uv;
    uv /= max(u_zoom, 0.01);
    uv -= u_pan;
    uv += u_shake;
    uv += 0.5;
    return uv;
}

// ── Shockwave (radial distortion) ───────────────────────────────────
vec2 shockwave(vec2 uv) {
    if (u_sw_time < 0.0 || u_sw_time > 1.0) return uv;
    vec2 diff = uv - u_sw_center;
    float dist = length(diff);
    float radius = u_sw_time * 0.5;
    float width  = 0.12;
    if (abs(dist - radius) < width) {
        float factor = (dist - radius) / width;
        float wave = cos(factor * 3.14159265) * (1.0 - u_sw_time) * 0.035;
        uv += normalize(diff + 1e-6) * wave;
    }
    return uv;
}

// ── Glitch + RGB shift ──────────────────────────────────────────────
vec3 glitch(vec2 uv, float intensity) {
    float line = hash(floor(uv.y * 120.0) + floor(u_time * 19.0)) * 2.0 - 1.0;
    uv.x += line * intensity * 0.06;

    // Block glitch
    float blk = hash(floor(uv.y * 25.0) + floor(u_time * 9.0));
    if (blk > 0.94) {
        uv.x += (hash(floor(u_time * 15.0)) - 0.5) * intensity * 0.12;
    }

    // RGB split
    float off = intensity * 0.012;
    float r = texture(u_scene, uv + vec2( off, 0.0)).r;
    float g = texture(u_scene, uv).g;
    float b = texture(u_scene, uv + vec2(-off, 0.0)).b;
    return vec3(r, g, b);
}

// ── Chromatic aberration ────────────────────────────────────────────
vec3 chromaticAberration(vec2 uv, float intensity) {
    vec2 dir = uv - 0.5;
    float d = length(dir);
    vec2 off = dir * d * intensity * 0.025;
    float r = texture(u_scene, uv + off).r;
    float g = texture(u_scene, uv).g;
    float b = texture(u_scene, uv - off).b;
    return vec3(r, g, b);
}

// ── Bloom (star-tap approximation) ──────────────────────────────────
vec3 bloom(vec2 uv, float intensity) {
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            float w = 1.0 / (1.0 + float(x * x + y * y));
            vec2 off = vec2(float(x), float(y)) * intensity * 0.004;
            sum += texture(u_scene, uv + off).rgb * w;
            total += w;
        }
    }
    return sum / total;
}

// ── Vignette ────────────────────────────────────────────────────────
float vignette(vec2 uv, float strength) {
    vec2 d = uv - 0.5;
    return 1.0 - dot(d, d) * strength * 2.8;
}

// ── Main composite ──────────────────────────────────────────────────
void main() {
    vec2 uv = cameraUV(v_uv);
    uv = shockwave(uv);

    // Base colour
    vec3 col;
    if (u_glitch > 0.01) {
        col = glitch(uv, u_glitch);
    } else {
        col = texture(u_scene, uv).rgb;
    }

    // Chromatic aberration
    if (u_ca > 0.01) {
        vec3 ca = chromaticAberration(uv, u_ca);
        col = mix(col, ca, 0.7);
    }

    // Bloom
    if (u_bloom > 0.01) {
        vec3 bl = bloom(uv, u_bloom);
        col += (bl - col * 0.4) * u_bloom * 0.55;
    }

    // Neon board-edge glow (pulsing with bass)
    float edgeDist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float edgeGlow = smoothstep(0.05, 0.0, edgeDist) * u_bass * 0.3;
    col += vec3(1.0, 0.0, 0.33) * edgeGlow;

    // Royal flash (check)
    col = mix(col, vec3(1.0), u_flash * 0.85);

    // Colour inversion (checkmate)
    col = mix(col, vec3(1.0) - col, u_invert);

    // Vignette
    col *= vignette(v_uv, u_vignette);

    // Subtle colour grading (phonk teal-red push)
    col.r *= 1.04;
    col.b *= 0.96;
    col = pow(col, vec3(0.96));

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# ═════════════════════════════════════════════════════════════════════════
# Unicode piece symbols
# ═════════════════════════════════════════════════════════════════════════

_PIECE_SYM_W = {
    chess.KING: "\u2654", chess.QUEEN: "\u2655", chess.ROOK: "\u2656",
    chess.BISHOP: "\u2657", chess.KNIGHT: "\u2658", chess.PAWN: "\u2659",
}
_PIECE_SYM_B = {
    chess.KING: "\u265A", chess.QUEEN: "\u265B", chess.ROOK: "\u265C",
    chess.BISHOP: "\u265D", chess.KNIGHT: "\u265E", chess.PAWN: "\u265F",
}


# ═════════════════════════════════════════════════════════════════════════
# Renderer
# ═════════════════════════════════════════════════════════════════════════

class Renderer:
    """Hybrid CPU (scene) + GPU (post-FX) renderer."""

    def __init__(self, config: Config):
        self.cfg = config
        self.W = config.width
        self.H = config.height
        self._init_fonts()
        self._init_piece_cache()
        self._init_gl()
        # ── Pre-allocated buffers & caches ────────────────────────────────
        self._bg_np: Optional[np.ndarray] = None        # static board bg (no heatmap)
        self._board_cache_key: Optional[int] = None      # hash for cached board bg
        self._scene_buf = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self._flip_buf = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # piece numpy arrays for fast blitting
        self._piece_np: Dict[Tuple[chess.Color, chess.PieceType], np.ndarray] = {}
        self._piece_alpha: Dict[Tuple[chess.Color, chess.PieceType], np.ndarray] = {}
        for key, pimg in self._piece_imgs.items():
            arr = np.array(pimg, dtype=np.uint8)
            self._piece_np[key] = arr[:, :, :3]
            self._piece_alpha[key] = arr[:, :, 3].astype(np.float32) / 255.0

    # ── Font loading ─────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        self._font: Optional[ImageFont.FreeTypeFont] = None
        self._san_font: Optional[ImageFont.FreeTypeFont] = None
        self._font_has_chess = False

        # Also try full paths in C:\Windows\Fonts\
        import os as _os
        candidates: List[str] = list(self.cfg.font_names)
        winfonts = _os.path.join(_os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        for name in list(self.cfg.font_names):
            fp = _os.path.join(winfonts, name)
            if fp not in candidates:
                candidates.append(fp)

        for name in candidates:
            try:
                fnt = ImageFont.truetype(name, self.cfg.piece_font_size)
                # Test if this font can actually render a chess symbol
                test_img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
                test_draw = ImageDraw.Draw(test_img)
                test_draw.text((10, 10), "\u2654", fill=(255, 255, 255, 255), font=fnt)
                # Check if anything was drawn (not blank)
                import numpy as _np
                arr = _np.array(test_img)
                if arr[:, :, 3].max() > 10:  # something visible was rendered
                    self._font = fnt
                    self._san_font = ImageFont.truetype(name, 36)
                    self._font_has_chess = True
                    print(f"[Renderer] Chess font loaded: {name}")
                    break
            except (OSError, IOError):
                continue

        # SAN font fallback (any readable font)
        if self._san_font is None:
            for name in candidates:
                try:
                    self._san_font = ImageFont.truetype(name, 36)
                    break
                except (OSError, IOError):
                    continue
            if self._san_font is None:
                self._san_font = ImageFont.load_default()

    # ── Piece image cache ────────────────────────────────────────────────

    def _init_piece_cache(self) -> None:
        """Pre-render every (colour, piece_type) as an RGBA PIL image.
        Uses Unicode glyphs if a chess font was found, otherwise draws
        recognisable geometric shapes for each piece type."""
        self._piece_imgs: Dict[Tuple[chess.Color, chess.PieceType], Image.Image] = {}
        sz = self.cfg.square_size
        for color in (chess.WHITE, chess.BLACK):
            syms = _PIECE_SYM_W if color == chess.WHITE else _PIECE_SYM_B
            rgb = self.cfg.white_piece_color if color == chess.WHITE else self.cfg.black_piece_color
            for pt, sym in syms.items():
                if self._font_has_chess and self._font is not None:
                    img = self._render_piece_font(sz, sym, rgb)
                else:
                    img = self._render_piece_shape(sz, pt, color, rgb)
                self._piece_imgs[(color, pt)] = img

    def _render_piece_font(self, sz: int, sym: str,
                           rgb: Tuple[int, int, int]) -> Image.Image:
        """Render a piece via Unicode glyph."""
        img = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), sym, font=self._font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (sz - tw) // 2 - bbox[0]
        ty = (sz - th) // 2 - bbox[1]
        draw.text((tx + 2, ty + 2), sym, fill=(0, 0, 0, 160), font=self._font)
        draw.text((tx, ty), sym, fill=rgb + (255,), font=self._font)
        return img

    @staticmethod
    def _render_piece_shape(sz: int, piece_type: chess.PieceType,
                            piece_color: chess.Color,
                            rgb: Tuple[int, int, int]) -> Image.Image:
        """Draw a recognisable geometric shape for each piece type.
        This is the fallback when no Unicode chess font is available."""
        img = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = sz // 2, sz // 2
        m = sz * 0.40  # margin factor
        fill = rgb + (255,)
        outline = (0, 0, 0, 200)
        shadow = (0, 0, 0, 120)
        base_y = int(sz * 0.82)
        base_w = int(sz * 0.35)

        # Common base pedestal
        def _base():
            draw.rectangle([cx - base_w, base_y - 6, cx + base_w, base_y + 2],
                           fill=fill, outline=outline, width=1)

        if piece_type == chess.PAWN:
            # Circle on a stick
            r = int(m * 0.40)
            head_y = int(sz * 0.30)
            # Stem
            draw.polygon([(cx - 6, head_y + r), (cx + 6, head_y + r),
                          (cx + 10, base_y - 6), (cx - 10, base_y - 6)],
                         fill=fill, outline=outline)
            # Head circle (shadow + main)
            draw.ellipse([cx - r - 1, head_y - r - 1, cx + r + 1, head_y + r + 1],
                         fill=shadow)
            draw.ellipse([cx - r, head_y - r, cx + r, head_y + r],
                         fill=fill, outline=outline)
            _base()

        elif piece_type == chess.ROOK:
            # Castle turret shape
            top = int(sz * 0.15)
            w = int(m * 0.55)
            bw = int(m * 0.65)
            # Main body
            draw.polygon([(cx - w, top + 14), (cx + w, top + 14),
                          (cx + bw, base_y - 6), (cx - bw, base_y - 6)],
                         fill=fill, outline=outline)
            # Battlements
            cren_w = int(w * 0.4)
            for dx in [-w, -w // 3, w // 3 + 1]:
                draw.rectangle([cx + dx, top, cx + dx + cren_w, top + 14],
                               fill=fill, outline=outline, width=1)
            _base()

        elif piece_type == chess.KNIGHT:
            # Horse head silhouette (simplified polygon)
            pts = [
                (cx - int(m * 0.45), base_y - 6),
                (cx - int(m * 0.50), int(sz * 0.40)),
                (cx - int(m * 0.20), int(sz * 0.18)),
                (cx + int(m * 0.10), int(sz * 0.12)),
                (cx + int(m * 0.40), int(sz * 0.25)),
                (cx + int(m * 0.30), int(sz * 0.42)),
                (cx + int(m * 0.50), int(sz * 0.55)),
                (cx + int(m * 0.45), base_y - 6),
            ]
            # Shadow
            draw.polygon([(x + 2, y + 2) for x, y in pts], fill=shadow)
            draw.polygon(pts, fill=fill, outline=outline)
            # Eye
            ex = cx - int(m * 0.15)
            ey = int(sz * 0.30)
            draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=outline)
            _base()

        elif piece_type == chess.BISHOP:
            # Mitre / pointed hat shape
            top = int(sz * 0.10)
            w_base = int(m * 0.50)
            # Body
            draw.polygon([(cx, top), (cx - w_base, base_y - 6),
                          (cx + w_base, base_y - 6)],
                         fill=fill, outline=outline)
            # Ball on top
            draw.ellipse([cx - 5, top - 5, cx + 5, top + 5],
                         fill=fill, outline=outline)
            # Slit
            draw.line([(cx, top + 12), (cx, int(sz * 0.55))],
                      fill=outline, width=2)
            _base()

        elif piece_type == chess.QUEEN:
            # Crown with points
            top = int(sz * 0.10)
            w = int(m * 0.60)
            bw = int(m * 0.55)
            # Body
            draw.polygon([(cx - bw, base_y - 6), (cx + bw, base_y - 6),
                          (cx + w, int(sz * 0.30)), (cx, top + 8),
                          (cx - w, int(sz * 0.30))],
                         fill=fill, outline=outline)
            # Crown points
            for dx in [-w, -w // 2, 0, w // 2, w]:
                px, py = cx + dx, top if dx == 0 else int(sz * 0.18)
                draw.ellipse([px - 4, py - 4, px + 4, py + 4],
                             fill=fill, outline=outline)
            _base()

        elif piece_type == chess.KING:
            # Crown + cross on top
            top = int(sz * 0.08)
            w = int(m * 0.55)
            bw = int(m * 0.55)
            # Body
            draw.polygon([(cx - bw, base_y - 6), (cx + bw, base_y - 6),
                          (cx + w, int(sz * 0.35)), (cx, int(sz * 0.22)),
                          (cx - w, int(sz * 0.35))],
                         fill=fill, outline=outline)
            # Cross
            cw = 4
            draw.rectangle([cx - cw, top, cx + cw, top + 20],
                           fill=fill, outline=outline, width=1)
            draw.rectangle([cx - 10, top + 5, cx + 10, top + 13],
                           fill=fill, outline=outline, width=1)
            _base()

        # Label letter in corner (K/Q/R/B/N/P) for clarity
        _LABELS = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
                   chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: ""}
        lbl = _LABELS.get(piece_type, "")
        if lbl:
            try:
                tiny = ImageFont.truetype("arial.ttf", 14)
            except (OSError, IOError):
                tiny = ImageFont.load_default()
            # White text with black outline for visibility
            tx, ty = 4, 4
            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((tx + ox, ty + oy), lbl, fill=(0, 0, 0, 200), font=tiny)
            draw.text((tx, ty), lbl, fill=(255, 255, 255, 230), font=tiny)

        return img

    # ── ModernGL setup ───────────────────────────────────────────────────

    def _init_gl(self) -> None:
        if moderngl is None:
            raise RuntimeError("moderngl is required. Install via: pip install moderngl")
        try:
            self._ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create OpenGL 3.3 context: {exc}\n"
                "Ensure GPU drivers are installed."
            ) from exc

        self._prog = self._ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        # Fullscreen triangle (3 vertices)
        verts = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        vbo = self._ctx.buffer(verts)
        self._vao = self._ctx.vertex_array(self._prog, [(vbo, "2f", "in_pos")])

        # Scene texture (the CPU-rendered board image is uploaded here each frame)
        self._scene_tex = self._ctx.texture((self.W, self.H), 3)
        self._scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Output FBO
        self._out_tex = self._ctx.texture((self.W, self.H), 3)
        self._fbo = self._ctx.framebuffer(color_attachments=[self._out_tex])

    # ── High-level render call ───────────────────────────────────────────

    def render_frame(
        self,
        board: chess.Board,
        anim_piece: Optional[Tuple[chess.PieceType, chess.Color,
                                    float, float, float, float, float]],
        # (piece_type, color, from_px, from_py, to_px, to_py, progress 0-1)
        effect_mgr: EffectManager,
        cam: CameraState,
        audio_bass: float,
        audio_mid: float,
        audio_high: float,
        onset_strength: float,
        time_sec: float,
        san_text: Optional[str] = None,
        heatmap: Optional[np.ndarray] = None,
        check_path: Optional[list] = None,
    ) -> np.ndarray:
        """Return a (H, W, 3) uint8 RGB frame."""
        # ── Stage 1: CPU scene ───────────────────────────────────────────
        scene = self._render_scene(board, anim_piece, effect_mgr,
                                    san_text, heatmap, check_path, time_sec)
        # ── Stage 2: GPU post-process ────────────────────────────────────
        frame = self._gpu_post(scene, cam, effect_mgr.state,
                                audio_bass, audio_mid, audio_high,
                                onset_strength, time_sec)
        return frame

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1 — CPU scene rendering
    # ══════════════════════════════════════════════════════════════════════

    def _build_board_bg(self, heatmap: Optional[np.ndarray]) -> np.ndarray:
        """Build board background with squares + border. Cached when heatmap is unchanged."""
        cache_key = hash(heatmap.tobytes()) if heatmap is not None else 0
        if self._board_cache_key == cache_key and self._bg_np is not None:
            return self._bg_np

        img = Image.new("RGB", (self.W, self.H), self.cfg.bg_color)
        draw = ImageDraw.Draw(img)
        ox = self.cfg.board_origin_x
        oy = self.cfg.board_origin_y
        sq = self.cfg.square_size
        bsz = self.cfg.board_size
        for r in range(8):
            for c in range(8):
                x0 = ox + c * sq
                y0 = oy + (7 - r) * sq
                is_light = (r + c) % 2 == 0
                base = self.cfg.light_square if is_light else self.cfg.dark_square
                if heatmap is not None:
                    h_val = heatmap[r, c]
                    hr, hg, hb = self.cfg.heatmap_color
                    colour = (
                        min(255, int(base[0] + hr * h_val * 0.35)),
                        min(255, int(base[1] + hg * h_val * 0.15)),
                        min(255, int(base[2] + hb * h_val * 0.15)),
                    )
                else:
                    colour = base
                draw.rectangle([x0, y0, x0 + sq - 1, y0 + sq - 1], fill=colour)
        bw = self.cfg.board_border_width
        bc = self.cfg.board_border_color
        draw.rectangle([ox - bw, oy - bw, ox + bsz + bw - 1, oy + bsz + bw - 1],
                        outline=bc, width=bw)
        self._bg_np = np.array(img, dtype=np.uint8)
        self._board_cache_key = cache_key
        return self._bg_np

    def _draw_check_path(self, scene: np.ndarray, board: chess.Board,
                         check_squares: list, time_sec: float) -> None:
        """Draw pulsing neon highlights on the check path (attacker → king)."""
        if not check_squares:
            return
        king_sq = board.king(board.turn) if board.is_check() else None
        ox = self.cfg.board_origin_x
        oy = self.cfg.board_origin_y
        sq_sz = self.cfg.square_size
        pulse = 0.50 + 0.50 * math.sin(time_sec * 10.0)
        color = np.array(self.cfg.check_path_color, dtype=np.float32)
        king_clr = np.array(self.cfg.check_king_color, dtype=np.float32)

        for sq_idx in check_squares:
            c = sq_idx % 8
            r = sq_idx // 8
            x0 = max(ox + c * sq_sz, 0)
            y0 = max(oy + (7 - r) * sq_sz, 0)
            x1 = min(x0 + sq_sz, self.W)
            y1 = min(y0 + sq_sz, self.H)
            if x0 >= x1 or y0 >= y1:
                continue
            is_king = (sq_idx == king_sq)
            clr = king_clr if is_king else color
            # Semi-transparent fill
            alpha = (0.45 if is_king else 0.28) * pulse
            region = scene[y0:y1, x0:x1].astype(np.float32)
            region[:] = region * (1.0 - alpha) + clr * alpha
            scene[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
            # Bright neon border
            bw = 3
            ba = (0.90 if is_king else 0.65) * pulse
            for ys, ye, xs, xe in [
                (y0, min(y0 + bw, y1), x0, x1),
                (max(y1 - bw, y0), y1, x0, x1),
                (y0, y1, x0, min(x0 + bw, x1)),
                (y0, y1, max(x1 - bw, x0), x1),
            ]:
                if ys < ye and xs < xe:
                    br = scene[ys:ye, xs:xe].astype(np.float32)
                    br[:] = br * (1.0 - ba) + clr * ba
                    scene[ys:ye, xs:xe] = np.clip(br, 0, 255).astype(np.uint8)

    def _render_scene(
        self,
        board: chess.Board,
        anim_piece: Optional[Tuple],
        fx: EffectManager,
        san_text: Optional[str],
        heatmap: Optional[np.ndarray],
        check_path: Optional[list] = None,
        time_sec: float = 0.0,
    ) -> np.ndarray:
        # Start from cached board background (fast copy)
        bg = self._build_board_bg(heatmap)
        scene = self._scene_buf
        np.copyto(scene, bg)

        # ── Check path overlay (pulsing neon squares) ────────────────────
        if check_path:
            self._draw_check_path(scene, board, check_path, time_sec)

        ox = self.cfg.board_origin_x
        oy = self.cfg.board_origin_y
        sq = self.cfg.square_size

        # ── Motion trails (ghosts) — fast numpy alpha blend ────────────
        for ghost in fx.trails.decayed():
            rgb = self._piece_np.get((ghost.piece_color, ghost.piece_type))
            alp = self._piece_alpha.get((ghost.piece_color, ghost.piece_type))
            if rgb is None or alp is None:
                continue
            gx, gy = int(ghost.x), int(ghost.y)
            ph, pw = rgb.shape[:2]
            y1, y2 = max(0, gy), min(self.H, gy + ph)
            x1, x2 = max(0, gx), min(self.W, gx + pw)
            sy, sx = y1 - gy, x1 - gx
            if y1 >= y2 or x1 >= x2:
                continue
            a = alp[sy:sy+(y2-y1), sx:sx+(x2-x1), np.newaxis] * ghost.alpha * 0.5
            dst = scene[y1:y2, x1:x2]
            src = rgb[sy:sy+(y2-y1), sx:sx+(x2-x1)]
            dst[:] = (dst * (1.0 - a) + src * a).astype(np.uint8)

        # ── Pieces — fast numpy alpha composite ────────────────────────
        skip_from = None
        if anim_piece is not None:
            _, _, fp_x, fp_y, _, _, prog = anim_piece
            skip_from = (int(round(fp_x)), int(round(fp_y)))

        pm = board.piece_map()
        for sq_idx, piece in pm.items():
            c = sq_idx % 8
            r = sq_idx // 8
            px = ox + c * sq
            py = oy + (7 - r) * sq
            if skip_from and (px, py) == skip_from:
                continue
            rgb = self._piece_np.get((piece.color, piece.piece_type))
            alp = self._piece_alpha.get((piece.color, piece.piece_type))
            if rgb is None:
                continue
            ph, pw = rgb.shape[:2]
            y1, y2 = max(0, py), min(self.H, py + ph)
            x1, x2 = max(0, px), min(self.W, px + pw)
            sy, sx = y1 - py, x1 - px
            if y1 >= y2 or x1 >= x2:
                continue
            a = alp[sy:sy+(y2-y1), sx:sx+(x2-x1), np.newaxis]
            dst = scene[y1:y2, x1:x2]
            src = rgb[sy:sy+(y2-y1), sx:sx+(x2-x1)]
            dst[:] = (dst * (1.0 - a) + src * a).astype(np.uint8)

        # ── Animated piece (with arc + squash & stretch) ─────────────────
        if anim_piece is not None:
            pt, pc, fpx, fpy, tpx, tpy, prog = anim_piece
            # Ease-out quintic
            ep = 1.0 - (1.0 - prog) ** 5
            cur_x = fpx + (tpx - fpx) * ep
            cur_y = fpy + (tpy - fpy) * ep
            # Arc (parabolic jump)
            arc = math.sin(prog * math.pi) * self.cfg.jump_height_px
            cur_y -= arc
            # Squash & stretch
            if prog > 0.85:
                landing = (prog - 0.85) / 0.15
                sy = 1.0 - self.cfg.squash_amount * math.sin(landing * math.pi)
                sx = 1.0 + self.cfg.squash_amount * 0.5 * math.sin(landing * math.pi)
            elif prog < 0.3:
                # Stretch during launch
                sy = 1.0 + self.cfg.squash_amount * 0.3 * math.sin(prog / 0.3 * math.pi)
                sx = 1.0 - self.cfg.squash_amount * 0.15 * math.sin(prog / 0.3 * math.pi)
            else:
                sx, sy = 1.0, 1.0

            p_img = self._piece_imgs.get((pc, pt))
            if p_img:
                w0, h0 = p_img.size
                nw = max(1, int(w0 * sx))
                nh = max(1, int(h0 * sy))
                resized = p_img.resize((nw, nh), Image.BILINEAR)
                paste_x = int(cur_x + (w0 - nw) // 2)
                paste_y = int(cur_y + (h0 - nh))
                # Paste animated piece onto numpy scene via PIL
                tmp = Image.fromarray(scene)
                tmp.paste(resized, (paste_x, paste_y), resized)
                np.copyto(scene, np.array(tmp, dtype=np.uint8))

        # ── Particles ────────────────────────────────────────────────────
        fx.particles.composite(scene)

        # ── SAN text overlay ─────────────────────────────────────────────
        if san_text:
            txt_img = Image.fromarray(scene)
            td = ImageDraw.Draw(txt_img)
            bbox = td.textbbox((0, 0), san_text, font=self._san_font)
            tw = bbox[2] - bbox[0]
            tx = (self.W - tw) // 2
            ty = self.cfg.board_origin_y - 50
            td.text((tx + 1, ty + 1), san_text, fill=(0, 0, 0), font=self._san_font)
            td.text((tx, ty), san_text, fill=self.cfg.san_text_color, font=self._san_font)
            np.copyto(scene, np.array(txt_img, dtype=np.uint8))

        return scene

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2 — GPU post-processing
    # ══════════════════════════════════════════════════════════════════════

    def _gpu_post(self, scene: np.ndarray, cam: CameraState,
                  fxs: EffectsState,
                  bass: float, mid: float, high: float,
                  onset: float, t: float) -> np.ndarray:
        # Upload scene to texture (flip vertically for OpenGL) — reuse buffer
        np.copyto(self._flip_buf, scene[::-1])
        self._scene_tex.write(self._flip_buf.tobytes())
        self._scene_tex.use(location=0)

        # Set uniforms
        p = self._prog
        _su(p, "u_time", t)
        _su(p, "u_zoom", cam.zoom)
        _su(p, "u_pan", (cam.pan_x, cam.pan_y))
        _su(p, "u_rotation", cam.rotation)
        _su(p, "u_shake", (cam.shake_x, cam.shake_y))
        _su(p, "u_bass", bass)
        _su(p, "u_mid", mid)
        _su(p, "u_high", high)
        _su(p, "u_onset", onset)
        _su(p, "u_bloom", self.cfg.bloom_base + fxs.bloom_extra)
        _su(p, "u_ca", self.cfg.chromatic_aberration_base + fxs.ca_extra)
        _su(p, "u_glitch", fxs.glitch_amount)
        _su(p, "u_vignette", self.cfg.vignette_base)
        _su(p, "u_flash", fxs.flash_amount)
        _su(p, "u_invert", fxs.invert_amount)
        _su(p, "u_sw_center", (fxs.shockwave_cx, fxs.shockwave_cy))
        _su(p, "u_sw_time", fxs.shockwave_time)

        # Render
        self._fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._vao.render(moderngl.TRIANGLES)

        # Read back (flip back) — avoid extra copy
        raw = self._fbo.read(components=3)
        out = np.frombuffer(raw, dtype=np.uint8).reshape(self.H, self.W, 3)
        return np.ascontiguousarray(out[::-1])

    # ── Coordinate helpers ───────────────────────────────────────────────

    def square_to_px(self, sq: int) -> Tuple[float, float]:
        """Return top-left pixel of the given chess square."""
        c = sq % 8
        r = sq // 8
        px = self.cfg.board_origin_x + c * self.cfg.square_size
        py = self.cfg.board_origin_y + (7 - r) * self.cfg.square_size
        return (float(px), float(py))

    def square_center_px(self, sq: int) -> Tuple[float, float]:
        """Return center pixel of the given chess square."""
        px, py = self.square_to_px(sq)
        half = self.cfg.square_size / 2
        return (px + half, py + half)

    def square_to_uv(self, sq: int) -> Tuple[float, float]:
        """Return normalised (0-1) screen position for shader uniforms."""
        cx, cy = self.square_center_px(sq)
        return (cx / self.W, 1.0 - cy / self.H)  # flip Y for GL

    def destroy(self) -> None:
        """Release GPU resources."""
        if hasattr(self, "_ctx") and self._ctx:
            self._ctx.release()


# ═════════════════════════════════════════════════════════════════════════
# Uniform helper (ignores missing uniforms gracefully)
# ═════════════════════════════════════════════════════════════════════════

def _su(prog, name: str, value) -> None:
    """Set a uniform, silently skipping if optimised out by the driver."""
    try:
        prog[name].value = value
    except KeyError:
        pass
