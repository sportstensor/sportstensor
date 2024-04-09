from abc import ABC, abstractmethod
from typing import Optional, List
import datetime as dt

from common.data import Match, Prediction, MatchPrediction


class ValidatorStorage(ABC):
    """An abstract class which defines the contract that all implementations of ValidatorStorage must fulfill."""

    @abstractmethod
    def insert_matches(self, predictions: List[Match]):
        """Stores official matches to score predictions from miners on."""
        raise NotImplemented
    
    @abstractmethod
    def update_matches(self, matches: List[Match]):
        """Updates matches. Typically only used when updating final score."""
        raise NotImplemented

    @abstractmethod
    def insert_match_predictions(self, predictions: List[MatchPrediction]):
        """Stores unscored predictions returned from miners."""
        raise NotImplemented
    
    @abstractmethod
    def get_predictions_to_score(self, batchsize: int) -> Optional[List[MatchPrediction]]:
        """Gets batchsize number of predictions that need to be scored."""
        raise NotImplemented
    
    @abstractmethod
    def update_match_predictions(self, predictions: List[MatchPrediction]):
        """Updates predictions. Typically only used when marking predictions as being scored."""
        raise NotImplemented

    @abstractmethod
    def get_miner_match_predictions(self, miner_hotkey: str, scored = False) -> Optional[List[MatchPrediction]]:
        """Gets a list of all predictions made by a miner."""
        raise NotImplemented

    @abstractmethod
    def delete_miner(self, miner_hotkey: str):
        """Removes the predictions and miner information for the specified miner."""
        raise NotImplemented

    @abstractmethod
    def read_miner_last_prediction(self, miner_hotkey: str) -> Optional[dt.datetime]:
        """Gets when a specific miner last returned a prediction."""
        raise NotImplemented