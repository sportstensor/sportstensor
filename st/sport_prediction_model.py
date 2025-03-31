from abc import ABC, abstractmethod
from typing import List
import bittensor as bt
from common.data import MatchPrediction, Sport, League, get_league_from_string, ProbabilityChoice
import logging
import random


class SportPredictionModel(ABC):
    def __init__(self, prediction):
        self.prediction = prediction
        self.huggingface_model = None

    @abstractmethod
    async def make_prediction(self):
        pass

    def set_default_scores(self):
        self.prediction.homeTeamScore = 0
        self.prediction.awayTeamScore = 0

    def set_default_probability(self, canTie: bool = False):
        if canTie:

            probs = generate_random_probabilities_with_tie()
            # Check which probability is the highest to determine the choice
            max_prob = max(probs)
            if probs[0] == max_prob:
                self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
            elif probs[1] == max_prob:
                self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
            else:
                self.prediction.probabilityChoice = ProbabilityChoice.DRAW

            self.prediction.probability = max_prob

        else:
            prob_a, prob_b = generate_random_probability_no_tie()
            if prob_a > prob_b:
                self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
            else:
                self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM

            self.prediction.probability = max(prob_a, prob_b)


async def make_match_prediction(prediction: MatchPrediction):
    # Lazy import to avoid circular dependency
    from st.models.base import SportstensorBaseModel

    from st.models.soccer import SoccerPredictionModel
    from st.models.football import FootballPredictionModel
    from st.models.baseball import BaseballPredictionModel
    from st.models.basketball import BasketballPredictionModel

    # Add new league classes here
    from st.models.soccer_mls import MLSSoccerPredictionModel
    from st.models.baseball_mlb import MLBBaseballPredictionModel
    from st.models.soccer_epl import EPLSoccerPredictionModel
    from st.models.football_nfl import NFLFootballPredictionModel
    from st.models.basketball_nba import NBABasketballPredictionModel
    
    base_classes = {
        'base': SportstensorBaseModel
    }
    sport_classes = {
        Sport.SOCCER: SoccerPredictionModel,
        Sport.FOOTBALL: FootballPredictionModel,
        Sport.BASEBALL: BaseballPredictionModel,
        Sport.BASKETBALL: BasketballPredictionModel,
    }
    league_classes = {
        League.MLS: MLSSoccerPredictionModel,
        League.MLB: MLBBaseballPredictionModel,
        League.EPL: EPLSoccerPredictionModel,
        League.NFL: NFLFootballPredictionModel,
        League.NBA: NBABasketballPredictionModel,
    }

    # Convert the league string back to the League enum
    league_enum = get_league_from_string(prediction.league)
    if league_enum is None:
        bt.logging.error(f"Unknown league: {prediction.league}. Returning.")
        return prediction

    base_class = base_classes.get('base')
    league_class = league_classes.get(league_enum)
    sport_class = sport_classes.get(prediction.sport)

    # Check if we have a base prediction model first
    if base_class:
        bt.logging.info("Using base prediction model.")
        base_prediction = base_class(prediction)
        await base_prediction.make_prediction()
    # Check if we have a league-specific prediction model first
    elif league_class:
        bt.logging.info(
            f"Using league-specific prediction model: {league_class.__name__}"
        )
        league_prediction = league_class(prediction)
        league_prediction.set_default_probability()
        await league_prediction.make_prediction()
    # If not, check if we have a sport-specific prediction model
    elif sport_class:
        bt.logging.info(
            f"Using sport-specific prediction model: {sport_class.__name__}"
        )
        sport_prediction = sport_class(prediction)
        sport_prediction.set_default_probability()
        await sport_prediction.make_prediction()
    # If we don't have a prediction model for the sport, return 0 for both scores
    else:
        bt.logging.info("Unknown sport, returning default probability.")
        prob_a, prob_b = generate_random_probability_no_tie()
        if prob_a > prob_b:
            prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
        else:
            prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM

        prediction.probability = max(prob_a, prob_b)

    return prediction

def generate_random_probability_no_tie() -> List[float]:
    # Generate a random probability for team A
    prob_a = random.uniform(0.05, 0.95)
    prob_b = 1 - prob_a
    
    return [prob_a, prob_b]

def generate_random_probabilities_with_tie() -> List[float]:
    # Generate random probabilities for win, lose, draw
    total = 0
    probs = [0, 0, 0]
    for i in range(3):
        probs[i] = random.uniform(0.1, 0.8)
        total += probs[i]
    
    # Normalize probabilities so they sum to 1
    probs = [p / total for p in probs]

    return probs
