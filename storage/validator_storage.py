from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
import datetime as dt

from common.data import League, Match, MatchPrediction, MatchPredictionWithMatchData
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
    def insert_match_odds(self, match_odds: List[tuple[str, float, dt.datetime]]):
        """Stores match odds in the database."""
        return NotImplemented
    
    @abstractmethod
    def delete_match_odds(self):
        """Deletes all match odds from the database."""
        return NotImplemented
    
    @abstractmethod
    def check_match_odds(self, matchId: str) -> bool:
        """Check if match odds with the given ID exists in the database."""
        return NotImplemented
    
    @abstractmethod
    def get_match_odds(self, matchId: str):
        """Gets all the match odds for the provided matchId."""
        return NotImplemented
    
    @abstractmethod
    def get_match_odds_by_batch(self, matchIds: List[str]) -> Dict[str, List[Tuple[str, float, float, float, dt.datetime]]]:
        """Gets all the match odds for the provided matchIds in a single query."""
        return NotImplemented

    @abstractmethod
    def get_matches_to_predict(self, batchsize: Optional[int]) -> List[Match]:
        """Gets batchsize number of matches ready to be predicted."""
        raise NotImplemented
    
    @abstractmethod
    def get_recently_completed_matches(self, matchDateSince: dt.datetime, league: Optional[League] = None) -> List[Match]:
        """Gets completed matches since the passed in date."""
        raise NotImplemented
    
    @abstractmethod
    def get_total_prediction_requests_count(self, matchDateSince: dt.datetime, league: Optional[League] = None) -> int:
        """Gets total count of prediction requests sent to miners since the passed in date."""
        raise NotImplemented
    
    @abstractmethod
    def update_match_prediction_request(self, matchId: str, request_time: str):
        """Updates a match prediction request with the status of the request_time."""
        raise NotImplemented
    
    @abstractmethod
    def get_match_prediction_requests(self, matchId: Optional[str] = None) -> Dict[str, Dict[str, bool]]:
        """Gets all match prediction requests or a specific match prediction request."""
        raise NotImplemented
    
    @abstractmethod
    def check_and_fix_match_prediction_requests(self, matchId: Optional[str] = None) -> None:
        """Checks and fixes the match prediction requests in the database."""
        raise NotImplemented
    
    @abstractmethod
    def delete_match_prediction_requests(self, matchId: Optional[str] = None) -> None:
        """Deletes specific match prediction requests, or match prediction requests from matches that are older than 1 day."""
        raise NotImplemented

    @abstractmethod
    def insert_match_predictions(self, predictions: List[GetMatchPrediction]):
        """Stores unscored predictions returned from miners."""
        raise NotImplemented
    
    @abstractmethod
    def delete_unscored_deregistered_match_predictions(self, miner_hotkeys: List[str], miner_uids: List[int]):
        """Deletes unscored predictions returned from miners that are no longer registered."""
        raise NotImplemented

    @abstractmethod
    def get_match_predictions_to_score(self, batchsize: int) -> Optional[List[MatchPrediction]]:
        """Gets batchsize number of predictions that need to be scored and are eligible to be scored (the match is complete)"""
        raise NotImplemented

    @abstractmethod
    def update_match_predictions(self, predictions: List[MatchPrediction]):
        """Updates predictions. Typically only used when marking predictions as being scored."""
        raise NotImplemented
    
    @abstractmethod
    def archive_match_predictions(self, miner_hotkeys: List[str], miner_uids: List[int]):
        """Updates predictions with isArchived 1. Typically only used when marking predictions achived after miner has been deregistered."""
        raise NotImplemented

    @abstractmethod
    def get_total_match_predictions_by_miner(self, miner_hotkey: str, miner_uid: int, matchDateSince: Optional[dt.datetime] = None, league: Optional[League] = None) -> int:
        """Gets the total number of predictions a miner has made since being registered. Must be scored and not archived."""
        raise NotImplemented
    
    @abstractmethod
    def get_miner_match_predictions(self, miner_hotkey: str, miner_uid: int, league: League=None, scored: bool=False, batchSize: int=None) -> Optional[List[MatchPredictionWithMatchData]]:
        """Gets a list of all predictions made by a miner. Include match data."""
        raise NotImplemented
    
    @abstractmethod
    def get_miner_match_predictions_by_batch(self, miner_data: List[tuple[str, int]], league: League=None, scored: bool=True, batch_size: int=None) -> Optional[Dict[int, List[MatchPredictionWithMatchData]]]:
        """Gets a dictionary of miner UIDs to a list of predictions made by that miner. Include match data."""
        raise NotImplemented

    @abstractmethod
    def delete_miner(self, miner_hotkey: str):
        """Removes the predictions and miner information for the specified miner."""
        raise NotImplemented

    @abstractmethod
    def read_miner_last_prediction(self, miner_hotkey: str) -> Optional[dt.datetime]:
        """Gets when a specific miner last returned a prediction."""
        raise NotImplemented
