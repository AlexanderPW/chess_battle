import chess
import json
import logging
import random
import re

from .llm import LLM
from .board import WHITE, BLACK

logger = logging.getLogger(__name__)


def _game_phase(board: chess.Board) -> str:
    """Rough game phase detection based on material remaining."""
    major_pieces = len(board.pieces(chess.QUEEN, chess.WHITE)) + \
                   len(board.pieces(chess.QUEEN, chess.BLACK)) + \
                   len(board.pieces(chess.ROOK, chess.WHITE)) + \
                   len(board.pieces(chess.ROOK, chess.BLACK))
    minor_pieces = len(board.pieces(chess.BISHOP, chess.WHITE)) + \
                   len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                   len(board.pieces(chess.KNIGHT, chess.WHITE)) + \
                   len(board.pieces(chess.KNIGHT, chess.BLACK))
    total = major_pieces + minor_pieces
    if total >= 10:
        return "opening"
    elif total >= 5:
        return "middlegame"
    else:
        return "endgame"


def _material_balance(board: chess.Board) -> str:
    """Return a human-readable material balance string."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9}
    white_mat = sum(v * len(board.pieces(p, chess.WHITE)) for p, v in values.items())
    black_mat = sum(v * len(board.pieces(p, chess.BLACK)) for p, v in values.items())
    diff = white_mat - black_mat
    if diff > 0:
        return f"White is up {diff} points of material"
    elif diff < 0:
        return f"Black is up {abs(diff)} points of material"
    else:
        return "Material is equal"


def _threats(board: chess.Board) -> str:
    """Describe immediate threats on the board."""
    lines = []
    # Is the side to move in check?
    if board.is_check():
        lines.append("⚠ YOU ARE IN CHECK — you must escape check.")
    # Identify opponent's attacked squares for hanging piece detection
    stm = board.turn
    opp = not stm
    hanging = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == stm:
            if board.is_attacked_by(opp, sq) and not board.is_attacked_by(stm, sq):
                hanging.append(f"{piece.symbol().upper()} on {chess.square_name(sq)}")
    if hanging:
        lines.append(f"⚠ Your hanging (undefended & attacked) pieces: {', '.join(hanging)}")
    return "\n".join(lines) if lines else "No immediate threats detected."


def _move_history_str(board: chess.Board) -> str:
    """Return the last 10 moves in SAN for context."""
    moves = list(board.move_stack)
    if not moves:
        return "Game just started."
    tmp = board.copy()
    # rewind to start
    tmp2 = chess.Board()
    san_moves = []
    for mv in moves:
        san_moves.append(tmp2.san(mv))
        tmp2.push(mv)
    recent = san_moves[-10:]
    return " ".join(recent)


def _rank_moves_by_strength(board: chess.Board, legal_moves: list[str]) -> list[str]:
    """
    Score and sort moves by strength so the LLM sees better moves first:
    checkmate > checks > captures (by value) > castling > others.
    Does NOT filter — just reorders.
    """
    scored = []
    capture_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                      chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    for uci in legal_moves:
        try:
            mv = chess.Move.from_uci(uci)
            score = 0
            board.push(mv)
            if board.is_checkmate():
                score += 1000  # CHECKMATE — immediate win, highest priority
            elif board.is_check():
                score += 20    # Regular check
            board.pop()
            captured = board.piece_at(mv.to_square)
            if captured:
                score += 10 + capture_values.get(captured.piece_type, 0)
            if board.is_castling(mv):
                score += 8
            if mv.promotion:
                score += 15
            scored.append((score, uci))
        except Exception:
            scored.append((0, uci))
    scored.sort(key=lambda x: -x[0])
    return [uci for _, uci in scored]


class Player:
    """
    LLM-backed chess player with strong prompting for better gameplay.
    """

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

    def _opp_name(self) -> str:
        return "Black" if self.color == WHITE else "White"

    def system(self) -> str:
        return f"""You are an expert chess engine playing as {self._side_name()}.

Your ONLY goal is to win. You think like a grandmaster.

Core principles you ALWAYS follow:
1. NEVER ignore check — if you are in check, you MUST escape it.
2. NEVER leave pieces hanging (undefended and attacked) — if you have hanging pieces, you MUST address them immediately unless strategizing or setting up trap.
3. ALWAYS look for checkmate first — it wins immediately. 
4. If you are winning on material, shift focus to finding or creating checkmate rather than trading pieces.
5. In the opening: develop pieces, control the center, castle early.
6. In the middlegame: coordinate pieces, identify king weaknesses, build mating nets, create forcing threats.
7. In the endgame: relentlessly hunt the king, try mating with multiple pieces.

You MUST output ONLY valid JSON. No markdown, no extra text. The "move_uci" field
MUST be copied EXACTLY from the provided legal moves list — any deviation forfeits the game.

JSON schema:
{{
  "evaluation": "1-2 sentence position assessment with material count",
  "tactics": "specific threats you see (checks, captures, forks, pins, skewers)",
  "plan": "your 2-3 move plan",
  "move_reason": "why THIS move best achieves your plan",
  "move_uci": "<exact UCI string from legal moves>"
}}"""

    def user(self, board: chess.Board, legal_moves: list[str]) -> str:
        ranked = _rank_moves_by_strength(board, legal_moves)
        # Show top 8 promising moves first, then the rest
        top = ranked[:8]
        rest = ranked[8:]
        moves_display = ", ".join(top)
        if rest:
            moves_display += f"\n  (other legal moves: {', '.join(rest)})"

        phase = _game_phase(board)
        material = _material_balance(board)
        threats = _threats(board)
        history = _move_history_str(board)

        return f"""=== CHESS POSITION ===
You are playing as: {self._side_name()}
Game phase: {phase}
{material}

Recent moves: {history}

Threats & Warnings:
{threats}

Opponent's King:
- Is the king exposed, trapped, or vulnerable to a mating attack?
- Can you create a forcing sequence (checks/threats) that delivers checkmate?

FEN: {board.fen()}

Legal moves (UCI) — top candidates listed first:
  {moves_display}

Phase-specific guidance ({phase}):
{"- Prioritize: e4/d4/e5/d5 center control, develop knights before bishops, castle within 10 moves." if phase == "opening" else ""}
{"- Prioritize: CHECKMATE tactics (forks, pins, skewers, back-rank), king hunts, forcing attacks." if phase == "middlegame" else ""}
{"- Prioritize: CHECKMATE hunting relentlessly, back-rank mates, passed pawn promotion, king activation toward mate. DO NOT trade pieces away — finish the game." if phase == "endgame" else ""}

Think step by step, analyze the opponent's king vulnerabilities, then output your JSON move. Remember: move_uci MUST be from the legal list above."""

    def _retry_user(self, board: chess.Board, legal_moves: list[str], attempt: int, bad_move: str) -> str:
        legal_csv = ", ".join(legal_moves)
        return f"""INVALID MOVE: "{bad_move}" is not in the legal moves list (attempt {attempt}/4).

You MUST copy one move EXACTLY — character for character — from this list:
{legal_csv}

Common mistakes to avoid:
- Do NOT add quotes around the move inside the JSON value
- Do NOT invent a move that looks plausible — only moves from the list above are legal
- Promotions MUST include piece letter (e.g. e7e8q not e7e8)

Output ONLY this JSON:
{{"evaluation":"","tactics":"","plan":"","move_reason":"<why you chose this move>","move_uci":"<exact move from list>"}}"""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_uci(raw: str) -> str:
        """Strip surrounding quotes, whitespace, punctuation that models often add."""
        if not raw:
            return ""
        # lowercase, strip whitespace
        cleaned = raw.strip().lower()
        # remove wrapping quotes of any kind
        cleaned = cleaned.strip("\"'`")
        # remove trailing punctuation like periods or commas
        cleaned = cleaned.rstrip(".,;:")
        return cleaned.strip()

    @staticmethod
    def _extract_json(text: str) -> dict:
        if not text:
            return {}
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right > left:
            try:
                return json.loads(text[left:right + 1])
            except Exception:
                pass
        return {}

    @staticmethod
    def _salvage_uci(text: str, legal_set: set) -> str | None:
        """Pull any legal UCI token out of raw text."""
        candidates = re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", (text or "").lower())
        for c in candidates:
            if c in legal_set:
                return c
        return None

    def _store_thoughts(self, parsed: dict):
        if not isinstance(parsed, dict):
            return
        self.evaluation = parsed.get("evaluation") or self.evaluation
        self.tactics    = parsed.get("tactics")    or self.tactics
        self.plan       = parsed.get("plan")       or self.plan
        # Only update move_reason if it's a string to avoid type errors
        move_reason = parsed.get("move_reason")
        if isinstance(move_reason, str):
            self.move_reason = move_reason or self.move_reason

    # ── Main move method ──────────────────────────────────────────────────────

    def move(self, board):
        MAX_ATTEMPTS = 4
        last_bad = ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            legal_list = board.legal_moves_uci()
            legal_set  = set(legal_list)

            if not legal_list:
                return  # game over / stalemate

            # Build prompt
            if attempt == 1:
                system_msg = self.system()
                user_msg   = self.user(board.board, legal_list)
            else:
                system_msg = self.system()
                user_msg   = self._retry_user(board.board, legal_list, attempt, last_bad)

            reply = self.llm.send(system_msg, user_msg, max_tokens=400)
            logger.debug("LLM reply (attempt %d): %s", attempt, reply)

            # Parse JSON
            parsed = self._extract_json(reply)
            self._store_thoughts(parsed)

            # Get move from JSON
            move_uci = self._normalize_uci(parsed.get("move_uci") or "")

            # Validate and play
            if move_uci and move_uci in legal_set:
                if board.move_uci(move_uci):
                    return
            else:
                last_bad = move_uci or reply[:60]

            # Salvage: scan raw text for any legal UCI token
            salvaged = self._salvage_uci(reply, legal_set)
            if salvaged:
                self.move_reason += f"\n[SALVAGED from text: {salvaged}]"
                if board.move_uci(salvaged):
                    return

        # Last resort: use strongest move rather than pure random
        legal_list = board.legal_moves_uci()
        if legal_list:
            ranked = _rank_moves_by_strength(board.board, legal_list)
            fallback = ranked[0]  # strongest move, not random
            self.move_reason += f"\n[FALLBACK strongest move: {fallback}]"
            board.move_uci(fallback)

    # ── UI ────────────────────────────────────────────────────────────────────

    def thoughts(self) -> str:
        return f"""
        <pre style="white-space: pre-wrap;">
Evaluation:
{self.evaluation}

Tactics:
{self.tactics}

Plan:
{self.plan}

Move reason:
{self.move_reason}
        </pre>
        """

    def switch_model(self, new_model_name: str):
        self.llm = LLM.create(new_model_name)
