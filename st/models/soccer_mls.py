import random
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel
from common.data import ProbabilityChoice

from st.models.soccer import SoccerPredictionModel


class MLSSoccerPredictionModel(SoccerPredictionModel):
    async def make_prediction(self):
        bt.logging.info("Predicting MLS soccer match...")

        self.set_default_probability(canTie=True)

        # Set your probability predictions here
        #self.prediction.probabilityChoice = random.choice([ProbabilityChoice.HOMETEAM, ProbabilityChoice.AWAYTEAM, ProbabilityChoice.DRAW])
        #self.prediction.probability = 0.5
