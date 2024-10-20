import random
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel
from common.data import ProbabilityChoice

from st.models.baseball import BasketballPredictionModel


class NBABasketballPredictionModel(BasketballPredictionModel):
    def make_prediction(self):
        bt.logging.info("Predicting NBA basketball game...")
        
        # Set your probability predictions here
        #self.prediction.probabilityChoice = random.choice([ProbabilityChoice.HOMETEAM, ProbabilityChoice.AWAYTEAM])
        #self.prediction.probability = 0.5
        