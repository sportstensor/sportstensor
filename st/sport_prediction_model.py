from abc import ABC, abstractmethod
import bittensor as bt
from common.data import MatchPrediction, Sport, League


class SportPredictionModel(ABC):
    def __init__(self, prediction):
        self.prediction = prediction
        self.huggingface_model = None

    @abstractmethod
    def make_prediction(self):
        pass

    def set_default_scores(self):
        self.prediction.homeTeamScore = 0
        self.prediction.awayTeamScore = 0

def make_match_prediction(prediction: MatchPrediction):
    # Lazy import to avoid circular dependency
    from st.models.soccer import SoccerPredictionModel
    from st.models.football import FootballPredictionModel
    from st.models.baseball import BaseballPredictionModel
    from st.models.basketball import BasketballPredictionModel
    from st.models.cricket import CricketPredictionModel
    # Add new league classes here
    from st.models.soccer_mls import MLSSoccerPredictionModel

    sport_classes = {
        Sport.SOCCER: SoccerPredictionModel,
        Sport.FOOTBALL: FootballPredictionModel,
        Sport.BASEBALL: BaseballPredictionModel,
        Sport.BASKETBALL: BasketballPredictionModel,
        Sport.CRICKET: CricketPredictionModel
    }
    league_classes = {
        League.MLS: MLSSoccerPredictionModel
    }

    league_class = league_classes.get(prediction.league)
    sport_class = sport_classes.get(prediction.sport)

    # Check if we have a league-specific prediction model first
    if league_class:
        league_prediction = league_class(prediction)
        league_prediction.set_default_scores()
        league_prediction.make_prediction()
    # If not, check if we have a sport-specific prediction model
    elif sport_class:
        sport_prediction = sport_class(prediction)
        sport_prediction.set_default_scores()
        sport_prediction.make_prediction()
    # If we don't have a prediction model for the sport, return 0 for both scores
    else:
        bt.logging.info("Unknown sport, returning 0 for both scores")
        prediction.homeTeamScore = 0
        prediction.awayTeamScore = 0

    return prediction