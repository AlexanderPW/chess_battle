import chess
import json
import logging
import re

from .llm import LLM
from .board import WHITE, BLACK

logger = logging.getLogger(__name__)


def _game_phase(board: chess.Board) -> str:
    total = sum(
        len(board.pieces(pt, c))
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        for c in (chess.WHITE, chess.BLACK)
    )
    if total >= 10: return "opening"
    if total >= 5:  return "middlegame"
    return "endgame"


def _material_balance(board: chess.Board) -> str:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    w = sum(v * len(board.pieces(p, chess.WHITE)) for p, v in values.items())
    b = sum(v * len(board.pieces(p, chess.BLACK)) for p, v in values.items())
    diff = w - b
    if diff > 0:   return f"White +{diff}"
    if diff < 0:   return f"Black +{abs(diff)}"
    return "Equal material"


def _context(board: chess.Board) -> str:
    """Key position facts: check, hanging pieces, king exposure."""
    lines = []
    if board.is_check():
        lines.append("YOU ARE IN CHECK — escape is mandatory.")
    stm, opp = board.turn, not board.turn
    hanging = [
        f"{board.piece_at(sq).symbol().upper()}@{chess.square_name(sq)}"
        for sq in chess.SQUARES
        if board.piece_at(sq)
        and board.piece_at(sq).color == stm
        and board.is_attacked_by(opp, sq)
        and not board.is_attacked_by(stm, sq)
    ]
    if hanging:
        lines.append(f"YOUR hanging pieces (save them or trade up): {', '.join(hanging)}")
    opp_king = board.king(opp)
    if opp_king:
        attackers = len(board.attackers(stm, opp_king))
        if attackers:
            lines.append(f"Opponent king on {chess.square_name(opp_king)} is ATTACKED by {attackers} of your pieces — hunt it.")
    return "\n".join(lines) if lines else "No immediate threats."


def _rank_moves(board: chess.Board, legal_moves: list[str]) -> list[str]:
    """Checkmate > checks > captures (by value) > castling > promotions > rest."""
    capture_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    scored = []
    for uci in legal_moves:
        try:
            mv = chess.Move.from_uci(uci)
            score = 0
            board.push(mv)
            if board.is_checkmate(): score += 10000
            elif board.is_check():   score += 50
            board.pop()
            cap = board.piece_at(mv.to_square)
            if cap: score += 10 + capture_values.get(cap.piece_type, 0)
            if board.is_castling(mv): score += 8
            if mv.promotion:          score += 15
            scored.append((score, uci))
        except Exception:
            scored.append((0, uci))
    scored.sort(key=lambda x: -x[0])
    return [uci for _, uci in scored]


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
        return f"""You are a ruthless chess engine playing as {self._side_name()}. Your ONLY goal is CHECKMATE.

PRIORITIES (in order):
1. If you can CHECKMATE, do it immediately — nothing else matters.
2. If you cannot checkmate, create UNSTOPPABLE mating threats that force the opponent to react.
3. NEVER trade pieces just to trade — only capture if it brings you closer to checkmate.
4. If ahead on material, use it to HUNT the king, not to simplify into a draw.
5. AVOID draws at all costs — stalemate, repetition, and piece trading are FAILURES.

Output ONLY valid JSON — no markdown, no explanation outside the JSON:
{{"evaluation":"material + king safety","tactics":"mating threats or forcing moves you see","plan":"2-3 move mating plan","move_reason":"why this move threatens checkmate","move_uci":"<exact UCI from legal list>"}}"""

    def user(self, board: chess.Board, legal_moves: list[str]) -> str:
        ranked = _rank_moves(board, legal_moves)
        top    = ", ".join(ranked[:8])
        rest   = f"\n  (other: {', '.join(ranked[8:])})" if ranked[8:] else ""
        phase  = _game_phase(board)
        return f"""Phase: {phase} | {_material_balance(board)}
{_context(board)}
FEN: {board.fen()}
Best candidate moves: {top}{rest}
move_uci MUST be copied exactly from the list above."""

    def _retry_user(self, legal_moves: list[str], attempt: int, bad_move: str) -> str:
        return f""""{bad_move}" is illegal (attempt {attempt}/4). Copy EXACTLY from:
{", ".join(legal_moves)}
{{"evaluation":"","tactics":"","plan":"","move_reason":"","move_uci":"<exact move>"}}"""

    @staticmethod
    def _normalize_uci(raw) -> str:
        if not isinstance(raw, str) or not raw:
            return ""
        return raw.strip().lower().strip("\"'`").rstrip(".,;:")

    @staticmethod
    def _extract_json(text: str) -> dict:
        if not text:
            return {}
        left, right = text.find("{"), text.rfind("}")
        if left != -1 and right > left:
            try:
                return json.loads(text[left:right + 1])
            except Exception:
                pass
        return {}

    @staticmethod
    def _salvage_uci(text: str, legal_set: set):
        for c in re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", (text or "").lower()):
            if c in legal_set:
                return c
        return None

    def _store_thoughts(self, parsed: dict):
        if not isinstance(parsed, dict):
            return
        def _s(val, fallback):
            return str(val) if val and not isinstance(val, (dict, list)) else fallback
        self.evaluation  = _s(parsed.get("evaluation"),  self.evaluation)
        self.tactics     = _s(parsed.get("tactics"),     self.tactics)
        self.plan        = _s(parsed.get("plan"),        self.plan)
        self.move_reason = _s(parsed.get("move_reason"), self.move_reason)

    def move(self, board):
        last_bad = ""
        for attempt in range(1, 5):
            legal_list = board.legal_moves_uci()
            legal_set  = set(legal_list)
            if not legal_list:
                return

            if attempt == 1:
                system_msg = self.system()
                user_msg   = self.user(board.board, legal_list)
            else:
                system_msg = self.system()
                user_msg   = self._retry_user(legal_list, attempt, last_bad)

            reply  = self.llm.send(system_msg, user_msg, max_tokens=400)
            parsed = self._extract_json(reply)
            self._store_thoughts(parsed)

            move_uci = self._normalize_uci(parsed.get("move_uci") or "")
            if move_uci and move_uci in legal_set and board.move_uci(move_uci):
                return

            last_bad = move_uci or reply[:60]

            salvaged = self._salvage_uci(reply, legal_set)
            if salvaged:
                self.move_reason += f"\n[SALVAGED: {salvaged}]"
                if board.move_uci(salvaged):
                    return

        # Fallback: strongest move by heuristic
        legal_list = board.legal_moves_uci()
        if legal_list:
            fallback = _rank_moves(board.board, legal_list)[0]
            self.move_reason += f"\n[FALLBACK: {fallback}]"
            board.move_uci(fallback)

    def thoughts(self) -> str:
        return f"""<pre style="white-space:pre-wrap;">
Evaluation:
{self.evaluation}

Tactics:
{self.tactics}

Plan:
{self.plan}

Move reason:
{self.move_reason}
</pre>"""

    def switch_model(self, new_model_name: str):
        self.llm = LLM.create(new_model_name)
