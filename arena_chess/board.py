import logging
import chess
import chess.svg

logger = logging.getLogger(__name__)

WHITE = chess.WHITE
BLACK = chess.BLACK


class Board:
    """
    Chess Board wrapper that plays the same role as arena/board.py in Ed's Connect-4 app.
    Keeps the interface simple for Game/UI:
      - fen() for position representation
      - legal_moves_uci() for valid moves
      - move_uci() to apply a move (illegal => forfeit)
      - svg() to render a board for Gradio
      - message() for status display
    """

    def __init__(self):
        self.board = chess.Board()
        self.player = self.board.turn  # True=White, False=Black

        self.winner = None   # True=White, False=Black, None=ongoing
        self.draw = False
        self.forfeit = False

        self.latest_move = None  # chess.Move | None
        self.stockfish_user = None  # Which color gets Stockfish in endgame (locked once assigned)

    def fen(self) -> str:
        return self.board.fen()

    def legal_moves_uci(self) -> list[str]:
        return [m.uci() for m in self.board.legal_moves]

    def is_active(self) -> bool:
        return self.winner is None and not self.draw and not self.forfeit

    def _update_end_state(self):
        if self.board.is_checkmate():
            # Side to move is checkmated, so opponent won
            self.winner = (not self.board.turn)
            return

        # Assign Stockfish to winning side on first endgame detection
        white_major = sum(
            len(self.board.pieces(piece_type, chess.WHITE))
            for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        )
        black_major = sum(
            len(self.board.pieces(piece_type, chess.BLACK))
            for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        )
        
        logger.debug(f"Material: White={white_major} major pieces, Black={black_major} major pieces | Stockfish assigned to: {self.stockfish_user}")
        
        # Assign Stockfish as soon as either player enters endgame territory (< 3 major pieces)
        # Give it to whoever has more material at that moment - locks in permanently
        if (white_major < 3 or black_major < 3) and self.stockfish_user is None:
            self.stockfish_user = chess.WHITE if white_major > black_major else chess.BLACK
            logger.info(f"🔧 STOCKFISH LOCKED: {('White' if self.stockfish_user == chess.WHITE else 'Black')} gets Stockfish (W={white_major} vs B={black_major} major pieces)")

        if self.board.is_stalemate():
            self.draw = True
            logger.info("🤝 DRAW: Stalemate")
            return
        
        if self.board.is_insufficient_material():
            self.draw = True
            logger.info("🤝 DRAW: Insufficient material")
            return
        
        if self.board.can_claim_draw():
            self.draw = True
            # Determine which draw condition
            if self.board.is_repetition():
                logger.info("🤝 DRAW: Threefold repetition")
            elif self.board.halfmove_clock >= 100:
                logger.info(f"🤝 DRAW: 50-move rule (halfmove_clock={self.board.halfmove_clock})")
            else:
                logger.info("🤝 DRAW: Draw claim available")
            return


    def move_uci(self, uci: str) -> bool:
        """
        Apply a UCI move if legal.
        Returns True if applied, False if invalid/illegal.
        Never forfeits the game.
        """
        try:
            uci = (uci or "").strip().lower()
            move = chess.Move.from_uci(uci)
            if move not in self.board.legal_moves:
                return False

            self.board.push(move)
            self.latest_move = move
            self.player = self.board.turn
            self._update_end_state()
            return True
        except Exception:
            return False

    def message(self) -> str:
        if self.winner is not None:
            side = "White" if self.winner == WHITE else "Black"
            return f"{side} wins"
        if self.draw:
            return "Draw"
        side = "White" if self.player == WHITE else "Black"
        check = " (in check)" if self.board.is_check() else ""
        return f"{side} to play{check}"
    

    def svg(self) -> str:
        return chess.svg.board(
            board=self.board,
            lastmove=self.latest_move,
            size=440,
            coordinates=True,
        )
