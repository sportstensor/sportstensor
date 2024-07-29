import random
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel


class FootballPredictionModel(SportPredictionModel):
    def make_prediction(self):
        bt.logging.info("Predicting American football game...")
        self.prediction.homeTeamScore = random.randint(0, 40)
        self.prediction.awayTeamScore = random.randint(0, 40)
