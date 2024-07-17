import datetime as dt
from common.data import Sport, League, MatchPrediction
from st.sport_prediction_model import make_match_prediction
#from sportstensor.predictions import make_match_prediction

def mls():
  matchDate = "2024-07-20"
  match_prediction = MatchPrediction(
    matchId = 1234,
    matchDate = dt.datetime.strptime(matchDate, "%Y-%m-%d"),
    sport = Sport.SOCCER,
    league = League.MLS,
    homeTeamName = "FC Dallas",
    awayTeamName = "Orlando City",
  )
  match_prediction = make_match_prediction(match_prediction)

  print(f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
  {match_prediction.awayTeamName} {match_prediction.awayTeamScore}, {match_prediction.homeTeamName} {match_prediction.homeTeamScore}"
  )

def mlb():
  matchDate = "2024-07-25"
  match_prediction = MatchPrediction(
    matchId = 1234,
    matchDate = dt.datetime.strptime(matchDate, "%Y-%m-%d"),
    sport = Sport.BASEBALL,
    league = League.MLB,
    homeTeamName = "Los Angeles Dodgers",
    awayTeamName = "Oakland Athletics",
  )
  match_prediction = make_match_prediction(match_prediction)

  print(f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
  {match_prediction.awayTeamName} {match_prediction.awayTeamScore}, {match_prediction.homeTeamName} {match_prediction.homeTeamScore}"
  )

if __name__ == "__main__":
  mlb()
  #mls()