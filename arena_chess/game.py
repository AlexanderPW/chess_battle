from datetime import datetime
import threading
from .llm import LLM
from .record import Result, record_game, ratings, get_games

from .board import Board, WHITE, BLACK
from .player import Player


class Game:
    def __init__(self, model_white: str, model_black: str):
        self.board = Board()
        self._lock = threading.Lock()
        self.players = {
            WHITE: Player(model_white, WHITE),
            BLACK: Player(model_black, BLACK),
        }

    def reset(self):
        self.board = Board()

    def is_active(self) -> bool:
        return self.board.is_active()

    # def move(self):
    #     self.players[self.board.player].move(self.board)

    def move(self):
        with self._lock:
            if not self.is_active():
                return
            self.players[self.board.player].move(self.board)

    def thoughts(self, side) -> str:
        return self.players[side].thoughts()

    def record(self):
        white_player = self.players[WHITE].llm.model_name
        black_player = self.players[BLACK].llm.model_name

        white_won = (self.board.winner == WHITE) if self.board.winner is not None else False
        black_won = (self.board.winner == BLACK) if self.board.winner is not None else False

        result = Result(white_player, black_player, white_won, black_won, datetime.now())
        record_game(result)

    @staticmethod
    def get_games():
        return get_games()

    @staticmethod
    def get_ratings():
        # Only show ratings for models we support in this running env
        supported = set(LLM.all_supported_model_names())
        return {m: r for m, r in ratings().items() if m in supported}
