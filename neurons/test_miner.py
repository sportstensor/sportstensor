import datetime as dt
from common.data import Sport, League, MatchPrediction
from st.sport_prediction_model import make_match_prediction
import bittensor as bt

bt.logging.set_trace(True)
bt.logging.set_debug(True)


def mls():
    matchDate = "2024-08-20"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.MLS,
        homeTeamName="Inter Miami",
        awayTeamName="Miami Fusion",
    )

    match_prediction = make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

    return match_prediction


def mlb():
    matchDate = "2024-08-25"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.BASEBALL,
        league=League.MLB,
        homeTeamName="Los Angeles Dodgers",
        awayTeamName="Oakland Athletics",
    )

    match_prediction = make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

def epl():
    matchDate = "2024-12-20"
    match_prediction = MatchPrediction(
        matchId="0b25bc4bd29ca0cd5d4b8031a3a36480",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.EPL,
        homeTeamName="Aston Villa",
        awayTeamName="Manchester City",
    )

    match_prediction = make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

def nfl():
    matchDate = "2024-12-20"
    match_prediction = MatchPrediction(
        matchId="d308633813328ef6c47859652c6970e2",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.FOOTBALL,
        league=League.NFL,
        homeTeamName="Los Angeles Chargers",
        awayTeamName="Denver Broncos",
    )

    match_prediction = make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

    return match_prediction

def nba():
    matchDate = "2024-12-20"
    match_prediction = MatchPrediction(
        matchId="ad71b94e2848f96ac01305e357df1e8a",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.BASKETBALL,
        league=League.NBA,
        homeTeamName="Detroit Pistons",
        awayTeamName="Utah Jazz",
    )

    match_prediction = make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

    return match_prediction

if __name__ == "__main__":
    #mls()
    #mlb()
    #epl()
    nfl()
    #nba()
