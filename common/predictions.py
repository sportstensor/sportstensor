from common.data import MatchPrediction

def make_match_prediction(match_prediction: MatchPrediction):
  # TODO: Miners must predict the score of the requested match by setting match_prediction.homeTeamScore and match_prediction.awayTeamScore
  match_prediction.homeTeamScore = 1
  match_prediction.awayTeamScore = 0

  return match_prediction
