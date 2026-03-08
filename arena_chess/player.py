import chess
import chess.engine
import json
import logging
import re
import shutil
import os
from pathlib import Path

from .llm import LLM
from .board import WHITE, BLACK

logger = logging.getLogger(__name__)

def _find_stockfish():
    """Find Stockfish binary, prioritizing Debian standard location."""
    paths_to_check = [
        "/usr/games/stockfish",     # Debian standard (HF Spaces)
        shutil.which("stockfish"),  # System PATH
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/bin/stockfish",
    ]
    for path in paths_to_check:
        if path and os.path.exists(path):
            return path
    return None

STOCKFISH_PATH = _find_stockfish()


def _game_phase(board: chess.Board) -> str:
    """Detect game phase based on individual player material, not combined.
    
    Endgame triggers when EITHER player has < 3 major pieces, ensuring
    both players get Stockfish help fairly and avoiding token waste.
    """
    white_major = sum(
        len(board.pieces(piece_type, chess.WHITE))
        for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    )
    black_major = sum(
        len(board.pieces(piece_type, chess.BLACK))
        for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    )
    
    # Endgame: either player has < 3 major pieces
    if white_major < 3 or black_major < 3:
        return "endgame"
    
    # Middlegame: 3+ major pieces for both, but not fully stocked opening
    if white_major <= 6 or black_major <= 6:
        return "middlegame"
    
    return "opening"


def _material_balance(board: chess.Board) -> str:
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_material = sum(value * len(board.pieces(piece_type, chess.WHITE)) for piece_type, value in piece_values.items())
    black_material = sum(value * len(board.pieces(piece_type, chess.BLACK)) for piece_type, value in piece_values.items())
    diff = white_material - black_material
    if diff > 0: return f"White is up {diff} points of material"
    if diff < 0: return f"Black is up {abs(diff)} points of material"
    return "Material is equal"


def _threats(board: chess.Board) -> str:
    lines = []
    if board.is_check():
        lines.append("⚠ YOU ARE IN CHECK — you must escape check.")
    current_player, opponent = board.turn, not board.turn
    hanging = [
        f"{board.piece_at(square).symbol().upper()} on {chess.square_name(square)}"
        for square in chess.SQUARES
        if board.piece_at(square)
        and board.piece_at(square).color == current_player
        and board.is_attacked_by(opponent, square)
        and not board.is_attacked_by(current_player, square)
    ]
    if hanging:
        lines.append(f"⚠ Your hanging pieces: {', '.join(hanging)}")
    return "\n".join(lines) if lines else "No immediate threats detected."


def _rank_moves(board: chess.Board, legal_moves: list[str]) -> list[str]:
    piece_capture_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    scored_moves = []
    for move_uci in legal_moves:
        try:
            move = chess.Move.from_uci(move_uci)
            score = 0
            board.push(move)
            if board.is_checkmate(): score += 10000
            elif board.is_check():   score += 50
            board.pop()
            captured_piece = board.piece_at(move.to_square)
            if captured_piece: score += 10 + piece_capture_values.get(captured_piece.piece_type, 0)
            if board.is_castling(move): score += 8
            if move.promotion:          score += 15
            scored_moves.append((score, move_uci))
        except Exception:
            scored_moves.append((0, move_uci))
    scored_moves.sort(key=lambda item: -item[0])
    return [move for _, move in scored_moves]


def _stockfish_move(board: chess.Board, time_limit: float = 1.0) -> str | None:
    """Use Stockfish engine to compute best move in endgame positions.
    
    Args:
        board: Chess position
        time_limit: Analysis time in seconds (default 1.0s for quality endgame moves)
    
    Returns:
        Best move in UCI format, or None if unavailable
    """
    if not STOCKFISH_PATH:
        logger.debug("Stockfish not found in PATH")
        return None
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            engine_result = engine.play(board, chess.engine.Limit(time=time_limit))
            move = engine_result.move.uci() if engine_result.move else None
            if move:
                logger.debug(f"Stockfish move: {move}")
            return move
    except Exception as error:
        logger.warning(f"Stockfish error: {error}")
        return None


class Player:
    def __init__(self, model: str, color: bool):
        self.color = color
        self.model = model
        self.llm = LLM.create(model)
        self.evaluation = ""
        self.tactics = ""
        self.plan = ""
        self.move_reason = ""

    def _side_name(self) -> str:
        return "White" if self.color == WHITE else "Black"

    def system(self) -> str:
        return f"""You are an expert chess engine playing as {self._side_name()}.

Your ONLY goal is to win. You think like a grandmaster.

Core principles you ALWAYS follow:
1. NEVER ignore check — if you are in check, you MUST escape it.
2. NEVER leave pieces hanging (undefended and attacked).
3. ALWAYS look for checkmate first — it wins immediately.
4. If you are winning on material, shift focus to checkmate rather than trading pieces.
5. In the opening: develop pieces, control the center, castle early.
6. In the middlegame: coordinate pieces, identify king weaknesses, build mating nets.
7. In the endgame: relentlessly hunt the king with your remaining pieces.

You MUST output ONLY valid JSON. No markdown, no extra text. The "move_uci" field
MUST be copied EXACTLY from the provided legal moves list — any deviation forfeits the game.

JSON schema:
{{
  "evaluation": "1-2 sentence position assessment with material count",
  "move_reason": "why THIS move best achieves your plan",
  "move_uci": "<exact UCI string from legal moves>"
}}"""

    def user(self, board: chess.Board, legal_moves: list[str], engine_suggestion: str | None = None) -> str:
        ranked_moves = _rank_moves(board, legal_moves)
        top_moves  = ", ".join(ranked_moves[:8])
        remaining_moves = f"\n  (other legal moves: {', '.join(ranked_moves[8:])})" if ranked_moves[8:] else ""
        phase = _game_phase(board)
        
        # Add engine suggestion if provided (endgame with Stockfish help)
        engine_note = ""
        if engine_suggestion:
            engine_note = f"\n🔹 STOCKFISH SUGGESTION: {engine_suggestion}"
        
        return f"""You are playing as: {self._side_name()}
Game phase: {phase} | {_material_balance(board)}

{_threats(board)}

FEN: {board.fen()}{engine_note}

Legal moves (UCI) — top candidates listed first:
  {top_moves}{remaining_moves}

{"Prioritize: center control, piece development, early castling." if phase == "opening" else ""}
{"Prioritize: king attacks, mating nets, forcing threats. Avoid piece trades." if phase == "middlegame" else ""}
{"Prioritize: checkmate hunting, passed pawns, king activation. DO NOT simplify into a draw." if phase == "endgame" else ""}

move_uci MUST be copied exactly from the legal moves list above."""

    def _retry_user(self, legal_moves: list[str], attempt: int, bad_move: str) -> str:
        return f"""INVALID MOVE: "{bad_move}" is not in the legal moves list (attempt {attempt}/4).

You MUST copy one move EXACTLY from this list:
{", ".join(legal_moves)}

Output ONLY this JSON:
{{"evaluation":"","move_reason":"<why you chose this move>","move_uci":"<exact move from list>"}}"""

    @staticmethod
    def _normalize_uci(raw) -> str:
        if not isinstance(raw, str) or not raw:
            return ""
        return raw.strip().lower().strip("\"'`").rstrip(".,;:")

    @staticmethod
    def _extract_json(text: str) -> dict:
        if not text:
            return {}
        open_brace = text.find("{")
        close_brace = text.rfind("}")
        if open_brace != -1 and close_brace > open_brace:
            try:
                return json.loads(text[open_brace:close_brace + 1])
            except Exception:
                pass
        return {}

    @staticmethod
    def _salvage_uci(text: str, legal_moves_set: set) -> str | None:
        for candidate in re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", (text or "").lower()):
            if candidate in legal_moves_set:
                return candidate
        return None

    def _store_thoughts(self, parsed: dict):
        if not isinstance(parsed, dict):
            return
        def ensure_string(value, fallback):
            return str(value) if value and not isinstance(value, (dict, list)) else fallback
        self.evaluation  = ensure_string(parsed.get("evaluation"),  self.evaluation)
        # self.tactics     = ensure_string(parsed.get("tactics"),     self.tactics)
        # self.plan        = ensure_string(parsed.get("plan"),        self.plan)
        self.move_reason = ensure_string(parsed.get("move_reason"), self.move_reason)

    def move(self, board):
        legal_moves = board.legal_moves_uci()
        legal_moves_set = set(legal_moves)
        if not legal_moves:
            return

        game_phase = _game_phase(board.board)
        
        # Use Stockfish only if assigned to this player (locked permanently)
        use_stockfish = board.stockfish_user == self.color
        
        # If assigned Stockfish: get the move, show it to LLM for analysis, then play it
        if use_stockfish:
            engine_move = _stockfish_move(board.board)
            
            # Ask LLM to analyze the Stockfish move
            llm_response = self.llm.send(self.system(), self.user(board.board, legal_moves, engine_suggestion=engine_move), max_tokens=400)
            parsed_response = self._extract_json(llm_response)
            self._store_thoughts(parsed_response)
            
            # Play the Stockfish move (guaranteed legal)
            if engine_move and engine_move in legal_moves_set:
                if board.move_uci(engine_move):
                    logger.info(f"{self._side_name()} ({game_phase}—Stockfish): {engine_move}")
                    return
            
            # Stockfish failed catastrophically, fallback to ranked
            best_move = _rank_moves(board.board, legal_moves)[0]
            if board.move_uci(best_move):
                logger.info(f"{self._side_name()} ({game_phase}—Stockfish fallback): {best_move}")
                return

        # Not assigned Stockfish: use LLM with full fallback chain
        llm_response = self.llm.send(self.system(), self.user(board.board, legal_moves, engine_suggestion=None), max_tokens=400)
        parsed_response = self._extract_json(llm_response)
        self._store_thoughts(parsed_response)

        move_uci = self._normalize_uci(parsed_response.get("move_uci") or "")
        if move_uci and move_uci in legal_moves_set and board.move_uci(move_uci):
            logger.info(f"{self._side_name()} ({game_phase}—LLM): {move_uci}")
            return

        salvaged_move = self._salvage_uci(llm_response, legal_moves_set)
        if salvaged_move and board.move_uci(salvaged_move):
            logger.info(f"{self._side_name()} ({game_phase}—salvaged): {salvaged_move}")
            return

        # All else failed, use ranked heuristic
        best_move = _rank_moves(board.board, legal_moves)[0]
        if board.move_uci(best_move):
            logger.info(f"{self._side_name()} ({game_phase}—ranked heuristic): {best_move}")
            return

    def thoughts(self) -> str:
        return f"""
        <pre style="white-space: pre-wrap;">
Evaluation:
{self.evaluation}

Move reason:
{self.move_reason}
        </pre>
        """

    def switch_model(self, new_model_name: str):
        self.llm = LLM.create(new_model_name)
