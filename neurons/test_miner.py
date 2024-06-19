import datetime as dt
from common.data import Sport, MatchPrediction
from common.predictions import make_match_prediction

def test():
  matchDate = "2024-06-20"
  match_prediction = MatchPrediction(
    matchId = 1234,
    matchDate = dt.datetime.strptime(matchDate, "%Y-%m-%d"),
    sport = Sport.SOCCER,
    homeTeamName = "Toronto FC",
    awayTeamName = "Columbus Crew",
  )
  match_prediction = make_match_prediction(match_prediction)

  print(f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
  {match_prediction.awayTeamName} {match_prediction.awayTeamScore}, {match_prediction.homeTeamName} {match_prediction.homeTeamScore}"
  )

if __name__ == "__main__":
  test()