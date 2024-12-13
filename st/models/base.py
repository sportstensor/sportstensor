import os
import random
import asyncio
import aiohttp
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel
from common.constants import LEAGUES_ALLOWING_DRAWS
from common.data import ProbabilityChoice
from typing import Optional

from dotenv import load_dotenv
MINER_ENV_PATH = 'neurons/miner.env'
load_dotenv(dotenv_path=MINER_ENV_PATH)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Odds API URL
API_URL = ""

class SportstensorBaseModel(SportPredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = ODDS_API_KEY
        if not self.api_key:
            raise ValueError("ODDS API key not found in miner.env")
        
        self.boost_min_percent = 0.03
        self.boost_max_percent = 0.10
        self.probability_cap = 0.95
            
        self.max_retries = 3
        self.retry_delay = 0.5
        self.timeout = 3

    async def _make_api_request(self, session: aiohttp.ClientSession) -> Optional[dict]:
        """Make a single API request with timeout and authorization."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            bt.logging.info(f"Making {self.prediction.league} API request for {self.prediction.homeTeamName} vs {self.prediction.awayTeamName} on {self.prediction.matchDate} ...")
            
            async with session.get(
                url=API_URL + str(self.prediction.matchId),
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    bt.logging.error("API authentication failed")
                    return None
                else:
                    bt.logging.warning(f"API request failed with status {response.status}")
                    return None
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            bt.logging.warning(f"API request failed: {str(e)}")
            return None
    
    async def _make_prediction_with_retries(self) -> Optional[dict]:
        """Makes API calls with retries and backoff."""
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                if attempt > 0:
                    bt.logging.info(f"Retry attempt {attempt + 1}/{self.max_retries}")
                    await asyncio.sleep(self.retry_delay)

                result = await self._make_api_request(session)
                if result is not None:
                    return result

            bt.logging.warning("Failed to get prediction after maximum retries")
            return None
        
    def get_boosted_prediction(self, home_odds, away_odds, draw_odds):
        # Get market probabilities for all outcomes
        home_prob = 1 / home_odds
        away_prob = 1 / away_odds
        if not draw_odds or draw_odds is None or draw_odds == 0:
            draw_prob = 0
        else:
            draw_prob = 1 / draw_odds
        
        # Find the highest probability outcome
        market_probs = [home_prob, away_prob, draw_prob]
        max_prob = max(market_probs)

        # Get the choice with the highest probability
        prob_choice = None
        if home_prob == max_prob:
            prob_choice = ProbabilityChoice.HOMETEAM
        elif away_prob == max_prob:
            prob_choice = ProbabilityChoice.AWAYTEAM
        else:
            prob_choice = ProbabilityChoice.DRAW
        
        # Generate random probability boost between 3% and 10%
        prob_boost = random.uniform(self.boost_min_percent, self.boost_max_percent)
        
        # Add boost to create new probability (cap at probability_cap to keep realistic)
        boosted_prob = min(max_prob + prob_boost, self.probability_cap)
        
        return boosted_prob, prob_choice

    def make_prediction(self):
        """Synchronous wrapper for async prediction logic."""
        bt.logging.info(f"Predicting {self.prediction.league} game...")
        
        try:
            result = asyncio.run(self._make_prediction_with_retries())
            
            # Only set predictions if we got a valid result
            if result is not None:
                if 'homeTeamOdds' in result and 'awayTeamOdds' in result and 'drawOdds' in result:
                    if self.prediction.league in LEAGUES_ALLOWING_DRAWS:
                        # Get the choice with the highest probability, boosted
                        prob, choice = self.get_boosted_prediction(result['homeTeamOdds'], result['awayTeamOdds'], result['drawOdds'])
                        self.prediction.probabilityChoice = choice
                        self.prediction.probability = prob
                    else:
                        # Get the choice with the highest probability, boosted
                        prob, choice = self.get_boosted_prediction(result['homeTeamOdds'], result['awayTeamOdds'], None)
                        self.prediction.probabilityChoice = choice
                        self.prediction.probability = prob
                else:
                    bt.logging.warning("API response missing required fields")
            
        except Exception as e:
            bt.logging.error(f"Failed to make prediction: {str(e)}")