from miner_dashboard import activate
from common.data import MatchPrediction

def make_match_prediction(match_prediction: MatchPrediction):
    date = match_prediction.matchDatetime.strftime("%Y-%m-%d")
    home_team = match_prediction.homeTeamName
    away_team = match_prediction.awayTeamName

    predictions = activate(date, home_team, away_team)

    if (home_team, away_team) in predictions:
        pred_scores = predictions[(home_team, away_team)]
        match_prediction.homeTeamScore = int(pred_scores[0])
        match_prediction.awayTeamScore = int(pred_scores[1])
    else:
        # Default prediction if no specific prediction exists
        match_prediction.homeTeamScore = 1
        match_prediction.awayTeamScore = 0

    return match_prediction
