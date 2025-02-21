import random
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel
from common.data import ProbabilityChoice


class FootballPredictionModel(SportPredictionModel):
    async def make_prediction(self):
        bt.logging.info("Predicting American football game...")
        
        # Set your probability predictions here
        #self.prediction.probabilityChoice = random.choice([ProbabilityChoice.HOMETEAM, ProbabilityChoice.AWAYTEAM, ProbabilityChoice.DRAW])
        #self.prediction.probability = 0.5
