"""
Phonk Chess Engine — Chess Core
================================
PGN parsing, move metadata extraction, Stockfish evaluation.
Uses the Command pattern: each move is a self-contained ``MoveData`` object
carrying all information the renderer and sync engine need.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import chess
import chess.pgn
import chess.engine

from config import Config

# ── Piece material values (for sacrifice detection) ─────────────────────
_PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass
class MoveData:
    """Immutable command object describing a single ply."""

    index: int
    move: chess.Move
    san: str
    from_square: int
    to_square: int
    piece_type: chess.PieceType
    piece_color: chess.Color
    is_capture: bool
    is_check: bool
    is_checkmate: bool
    is_castling: bool
    is_en_passant: bool
    captured_piece_type: Optional[chess.PieceType]
    drama_score: float                # 0.0 – 1.0
    eval_before: Optional[float]      # centipawns (white POV)
    eval_after: Optional[float]
    board_before: chess.Board
    board_after: chess.Board


class ChessCore:
    """Parses a PGN file and optionally evaluates each position via Stockfish."""

    def __init__(self, pgn_path: str, stockfish_path: Optional[str] = None,
                 config: Optional[Config] = None):
        self.pgn_path = pgn_path
        self.stockfish_path = stockfish_path
        self.config = config or Config()
        self.game: Optional[chess.pgn.Game] = None
        self.moves: List[MoveData] = []
        self._parse()

    # ── Public helpers ───────────────────────────────────────────────────

    def starting_board(self) -> chess.Board:
        """Return the starting position of the game."""
        if self.game:
            return self.game.board()
        return chess.Board()

    # ── Internals ────────────────────────────────────────────────────────

    def _parse(self) -> None:
        with open(self.pgn_path, encoding="utf-8") as fh:
            self.game = chess.pgn.read_game(fh)
        if self.game is None:
            raise ValueError(f"No valid game found in {self.pgn_path}")

        engine: Optional[chess.engine.SimpleEngine] = None
        if self.stockfish_path:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                print(f"[ChessCore] Stockfish loaded: {self.stockfish_path}")
            except Exception as exc:
                print(f"[ChessCore] Stockfish unavailable ({exc}); drama heuristic only.")

        board = self.game.board()
        prev_eval: float = 0.0
        idx = 0

        for move in self.game.mainline_moves():
            san = board.san(move)
            piece = board.piece_at(move.from_square)
            captured = board.piece_at(move.to_square)
            is_capture = board.is_capture(move)
            is_en_passant = board.is_en_passant(move)
            is_castling = board.is_castling(move)

            # If en-passant, the captured pawn is not on to_square
            if is_en_passant:
                captured_type = chess.PAWN
            else:
                captured_type = captured.piece_type if captured else None

            board_before = board.copy()
            eval_before = prev_eval

            board.push(move)

            is_check = board.is_check()
            is_checkmate = board.is_checkmate()

            # Stockfish evaluation
            eval_after: Optional[float] = None
            if engine:
                try:
                    info = engine.analyse(board,
                                          chess.engine.Limit(depth=self.config.stockfish_depth))
                    score = info["score"].white()
                    if score.is_mate():
                        eval_after = 10000.0 if score.mate() > 0 else -10000.0
                    else:
                        eval_after = float(score.score())  # type: ignore[arg-type]
                except Exception:
                    eval_after = None

            drama = self._compute_drama(
                is_capture, is_check, is_checkmate,
                piece.piece_type if piece else chess.PAWN,
                captured_type, eval_before, eval_after,
            )

            prev_eval = eval_after if eval_after is not None else prev_eval

            self.moves.append(MoveData(
                index=idx,
                move=move,
                san=san,
                from_square=move.from_square,
                to_square=move.to_square,
                piece_type=piece.piece_type if piece else chess.PAWN,
                piece_color=piece.color if piece else chess.WHITE,
                is_capture=is_capture,
                is_check=is_check,
                is_checkmate=is_checkmate,
                is_castling=is_castling,
                is_en_passant=is_en_passant,
                captured_piece_type=captured_type,
                drama_score=drama,
                eval_before=eval_before,
                eval_after=eval_after,
                board_before=board_before,
                board_after=board.copy(),
            ))
            idx += 1

        if engine:
            engine.quit()
        print(f"[ChessCore] Parsed {len(self.moves)} moves from {self.pgn_path}")

    # ── Drama heuristic ──────────────────────────────────────────────────

    @staticmethod
    def _compute_drama(is_capture: bool, is_check: bool, is_checkmate: bool,
                       piece_type: int, captured_type: Optional[int],
                       eval_before: Optional[float],
                       eval_after: Optional[float]) -> float:
        """Return a drama score in [0, 1] for a move."""
        if is_checkmate:
            return 1.0

        d = 0.20  # baseline

        if is_check:
            d += 0.25
        if is_capture:
            d += 0.20
            if captured_type in (chess.QUEEN, chess.ROOK):
                d += 0.15

        # Eval swing
        if eval_before is not None and eval_after is not None:
            swing = abs(eval_after - eval_before)
            d += min(swing / 500.0, 0.30)

        # Sacrifice heuristic (giving up more material than captured)
        if is_capture and captured_type is not None:
            me = _PIECE_VALUE.get(piece_type, 0)
            them = _PIECE_VALUE.get(captured_type, 0)
            if me > them + 1:
                d += 0.10

        return min(d, 1.0)
