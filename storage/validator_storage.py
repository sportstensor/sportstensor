from abc import ABC, abstractmethod
from typing import Optional, List
import datetime as dt

from common.data import League, Match, MatchPrediction, PlayerStat, PlayerPrediction
from common.protocol import GetMatchPrediction


class ValidatorStorage(ABC):
    """An abstract class which defines the contract that all implementations of ValidatorStorage must fulfill."""

    @abstractmethod
    def cleanup(self):
        """Cleans up the database."""
        raise NotImplemented

    @abstractmethod
    def insert_leagues(self, leagues: List[League]):
        """Stores leagues associated with sports. Indicates which leagues are active to run predictions on."""
        raise NotImplemented

    @abstractmethod
    def update_leagues(self, leagues: List[League]):
        """Updates leagues."""
        raise NotImplemented

    @abstractmethod
    def insert_matches(self, matches: List[Match]):
        """Stores official matches to score predictions from miners on."""
        raise NotImplemented

    @abstractmethod
    def update_matches(self, matches: List[Match]):
        """Updates matches. Typically only used when updating final score."""
        raise NotImplemented

    @abstractmethod
    def check_match(self, matchId: str) -> Match:
        """Check if a match with the given ID exists in the database."""
        return NotImplemented
    
    @abstractmethod
    def get_match(self, matchId: str) -> Match:
        """Gets a match with the given ID from the database."""
        return NotImplemented

    @abstractmethod
    def get_matches_to_predict(self, batchsize: int) -> List[Match]:
        """Gets batchsize number of matches ready to be predicted."""
        raise NotImplemented

    @abstractmethod
    def insert_match_predictions(self, predictions: List[GetMatchPrediction]):
        """Stores unscored predictions returned from miners."""
        raise NotImplemented

    @abstractmethod
    def get_match_predictions_to_score(
        self, batchsize: int
    ) -> Optional[List[MatchPrediction]]:
        """Gets batchsize number of predictions that need to be scored and are eligible to be scored (the match is complete)"""
        raise NotImplemented

    @abstractmethod
    def update_match_predictions(self, predictions: List[MatchPrediction]):
        """Updates predictions. Typically only used when marking predictions as being scored."""
        raise NotImplemented

    @abstractmethod
    def get_miner_match_predictions(
        self, miner_hotkey: str, scored=False
    ) -> Optional[List[MatchPrediction]]:
        """Gets a list of all predictions made by a miner."""
        raise NotImplemented
    
    @abstractmethod
    def insert_player_stats(self, stats: List[PlayerStat]):
        """Stores player stats to score predictions from miners on."""
        raise NotImplemented

    @abstractmethod
    def update_player_stats(self, stats: List[PlayerStat]):
        """Updates player stats. Typically only used when updating final stats."""
        raise NotImplemented

    @abstractmethod
    def check_player_stat(self, playerStatId: str) -> PlayerStat:
        """Check if a player stat with the given ID exists in the database."""
        return NotImplemented

    @abstractmethod
    def delete_miner(self, miner_hotkey: str):
        """Removes the predictions and miner information for the specified miner."""
        raise NotImplemented

    @abstractmethod
    def read_miner_last_prediction(self, miner_hotkey: str) -> Optional[dt.datetime]:
        """Gets when a specific miner last returned a prediction."""
        raise NotImplemented
