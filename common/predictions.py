import bittensor as bt
from miner_dashboard import activate
from common.data import MatchPrediction, Sport
import random

def make_match_prediction(prediction: MatchPrediction):

    # Setting default prediction scores to 0
    prediction.homeTeamScore = 0
    prediction.awayTeamScore = 0

    if prediction.sport == Sport.SOCCER:
        bt.logging.info("Predicting soccer match...")
        MatchDate = prediction.matchDate.strftime("%Y-%m-%d")

        homeTeamName = prediction.homeTeamName
        awayTeamName = prediction.awayTeamName
        
        #predictions = predictSoccerMatch(date, prediction.homeTeamName, prediction.awayTeamName)
        predictions = activate(MatchDate, homeTeamName, awayTeamName)

        if predictions is not None and (prediction.homeTeamName, prediction.awayTeamName) in predictions:
            pred_scores = predictions[(prediction.homeTeamName, prediction.awayTeamName)]
            prediction.homeTeamScore = int(pred_scores[0])
            prediction.awayTeamScore = int(pred_scores[1])
        else:
            # Default prediction if no specific prediction exists
            prediction.homeTeamScore = random.randint(0, 10)
            prediction.awayTeamScore = random.randint(0, 10)

    elif prediction.sport == Sport.FOOTBALL:
        bt.logging.info("Handling football...")
        prediction.homeTeamScore = random.randint(0, 40)
        prediction.awayTeamScore = random.randint(0, 40)

    elif prediction.sport == Sport.BASEBALL:
        bt.logging.info("Predicting baseball game...")
        prediction.homeTeamScore = random.randint(0, 10)
        prediction.awayTeamScore = random.randint(0, 10)

    elif prediction.sport == Sport.BASKETBALL:
        bt.logging.info("Handling basketball...")
        prediction.homeTeamScore = random.randint(30, 130)
        prediction.awayTeamScore = random.randint(30, 130)

    elif prediction.sport == Sport.CRICKET:
        bt.logging.info("Handling cricket...")
        prediction.homeTeamScore = random.randint(0, 160)
        prediction.awayTeamScore = random.randint(0, 160)

    else:
        bt.logging.info("Unknown sport, returning 0 for both scores")
    

    return prediction
