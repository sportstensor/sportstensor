import datetime as dt
from common.data import Sport, MatchPrediction
from common.predictions import make_match_prediction

def test():
  match_datetime = "2024-05-02"
  match_prediction = MatchPrediction(
    matchId = 1234,
    matchDatetime = dt.datetime.strptime(match_datetime, "%Y-%m-%d"),
    sport = Sport.SOCCER,
    homeTeamName = "Chelsea",
    awayTeamName = "Tottenham Hotspur",
  )
  match_prediction = make_match_prediction(match_prediction)

  print(f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {match_datetime}: \
  {match_prediction.awayTeamName} {match_prediction.awayTeamScore}, {match_prediction.homeTeamName} {match_prediction.homeTeamScore}"
  )

if __name__ == "__main__":
  test()