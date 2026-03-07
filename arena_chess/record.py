import logging
import os
import math
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


@dataclass
class Result:
    white_player: str
    black_player: str
    white_won: bool
    black_won: bool
    when: datetime


COLLECTION = "games"


def _get_collection():
    """Helper function to get MongoDB collection with error handling"""
    try:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            return None
        # Use very short timeout - fail fast if DB is unavailable
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)
        client.admin.command("ismaster")
        db = client.chess_battle
        return db[COLLECTION]
    except Exception as e:
        logging.debug(f"MongoDB unavailable: {e}")
        return None


def record_game(result: Result) -> bool:
    """
    Store the results in the database, if database is available.
    Returns True if successful, False if database is unavailable.
    """
    collection = _get_collection()
    if collection is None:
        logging.warning("MongoDB unavailable - game not recorded")
        return False

    # Convert Result object to dictionary for MongoDB storage
    game_dict = asdict(result)
    # Convert datetime to ISO format string for JSON serialization
    game_dict["when"] = result.when.isoformat()

    try:
        collection.insert_one(game_dict)
        logging.info("Game recorded in database")
        return True
    except Exception as e:
        logging.error("Failed to record a game in the database")
        logging.exception(e)
        return False


def get_games() -> List[Result]:
    """
    Return all games in the order that they were played.
    Returns empty list if database is unavailable.
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        # Sort by _id to maintain insertion order
        games = collection.find().sort("_id", 1)

        # Convert MongoDB documents back to Result objects
        results = []
        for game in games:
            # Remove MongoDB's _id field
            game.pop("_id", None)
            # Convert ISO string back to datetime
            if isinstance(game.get("when"), str):
                game["when"] = datetime.fromisoformat(game["when"])
            results.append(Result(**game))

        return results
    except Exception as e:
        logging.error("Error getting games")
        logging.exception(e)
        return []


class EloCalculator:
    def __init__(self, k_factor: float = 32, default_rating: int = 1000):
        """
        Initialize the ELO calculator.

        Args:
            k_factor: Determines how much ratings change after each game
            default_rating: Starting rating for new players
        """
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings: Dict[str, float] = {}

    def get_player_rating(self, player: str) -> float:
        """Get a player's current rating, or default if they're new."""
        return self.ratings.get(player, self.default_rating)

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate the expected score (win probability) for player A against player B.
        Uses the ELO formula: 1 / (1 + 10^((ratingB - ratingA)/400))
        """
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update_ratings(
        self, player_a: str, player_b: str, score_a: float, score_b: float
    ) -> None:
        """
        Update ratings for two players based on their game outcome.

        Args:
            player_a: Name of first player
            player_b: Name of second player
            score_a: Actual score for player A (1 for win, 0.5 for draw, 0 for loss)
            score_b: Actual score for player B (1 for win, 0.5 for draw, 0 for loss)
        """
        rating_a = self.get_player_rating(player_a)
        rating_b = self.get_player_rating(player_b)

        expected_a = self.calculate_expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a

        # Update ratings using the ELO formula: R' = R + K * (S - E)
        # where R is the current rating, K is the k-factor,
        # S is the actual score, and E is the expected score
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b


def calculate_elo_ratings(
    results: List[Result], exclude_self_play: bool = True
) -> Dict[str, float]:
    """
    Calculate final ELO ratings for all players based on a list of game results.

    Args:
        results: List of game results, sorted by date
        exclude_self_play: If True, skip games where a player plays against themselves

    Returns:
        Dictionary mapping player names to their final ELO ratings
    """
    calculator = EloCalculator()

    for result in results:
        # Skip self-play games if requested
        if exclude_self_play and result.white_player == result.black_player:
            continue

        # Convert game result to ELO scores (1 for win, 0.5 for draw, 0 for loss)
        if result.white_won and not result.black_won:
            white_score, black_score = 1.0, 0.0
        elif result.black_won and not result.white_won:
            white_score, black_score = 0.0, 1.0
        else:
            # Draw (both won, both lost, or neither)
            white_score, black_score = 0.5, 0.5

        calculator.update_ratings(result.white_player, result.black_player, white_score, black_score)

    return calculator.ratings


def ratings() -> Dict[str, float]:
    """
    Return the ELO ratings from all prior games in the DB
    """
    games = get_games()
    return calculate_elo_ratings(games)
