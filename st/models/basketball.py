import random
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel

class BasketballPredictionModel(SportPredictionModel):
    def make_prediction(self):
        bt.logging.info("Handling basketball...")
        self.prediction.homeTeamScore = random.randint(30, 130)
        self.prediction.awayTeamScore = random.randint(30, 130)
