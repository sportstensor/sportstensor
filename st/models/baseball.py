import random
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel


class BaseballPredictionModel(SportPredictionModel):
    def make_prediction(self):
        bt.logging.info("Predicting baseball game...")
        self.prediction.homeTeamScore = random.randint(0, 10)
        self.prediction.awayTeamScore = random.randint(0, 10)
