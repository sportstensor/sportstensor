import random
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel


class CricketPredictionModel(SportPredictionModel):
    def make_prediction(self):
        bt.logging.info("Handling cricket...")
        self.prediction.homeTeamScore = random.randint(0, 160)
        self.prediction.awayTeamScore = random.randint(0, 160)
