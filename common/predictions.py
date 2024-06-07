import bittensor as bt
#from miner_dashboard import activate
from common.data import MatchPrediction, Sport

def make_match_prediction(prediction: MatchPrediction):

    # Setting default prediction scores to 0
    prediction.homeTeamScore = 0
    prediction.awayTeamScore = 0

    if prediction.sport == Sport.SOCCER:
        bt.logging.info("Predicting soccer match...")
        date = prediction.matchDate.strftime("%Y-%m-%d")
        #predictions = predictSoccerMatch(date, prediction.homeTeamName, prediction.awayTeamName)
        predictions = None

        if predictions is not None and (prediction.homeTeamName, prediction.awayTeamName) in predictions:
            pred_scores = predictions[(prediction.homeTeamName, prediction.awayTeamName)]
            prediction.homeTeamScore = int(pred_scores[0])
            prediction.awayTeamScore = int(pred_scores[1])
        else:
            # Default prediction if no specific prediction exists
            prediction.homeTeamScore = 1
            prediction.awayTeamScore = 0

    elif prediction.sport == Sport.FOOTBALL:
        bt.logging.info("Handling football...")
        

    elif prediction.sport == Sport.BASEBALL:
        bt.logging.info("Predicting baseball game...")
        

    elif prediction.sport == Sport.BASKETBALL:
        bt.logging.info("Handling basketball...")
        

    elif prediction.sport == Sport.CRICKET:
        bt.logging.info("Handling cricket...")

    else:
        bt.logging.info("Unknown sport, returning 0 for both scores")
    

    return prediction
