from abc import ABC, abstractmethod
import bittensor as bt
from common.data import MatchPrediction, Sport


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

    sport_classes = {
        Sport.SOCCER: SoccerPredictionModel,
        Sport.FOOTBALL: FootballPredictionModel,
        Sport.BASEBALL: BaseballPredictionModel,
        Sport.BASKETBALL: BasketballPredictionModel,
        Sport.CRICKET: CricketPredictionModel
    }

    sport_class = sport_classes.get(prediction.sport)
    if sport_class:
        sport_prediction = sport_class(prediction)
        sport_prediction.set_default_scores()
        sport_prediction.make_prediction()
    else:
        bt.logging.info("Unknown sport, returning 0 for both scores")
        prediction.homeTeamScore = 0
        prediction.awayTeamScore = 0

    return prediction