import datetime as dt
import os
import asyncio  # Add asyncio import
from common.data import Sport, League, MatchPrediction
from st.sport_prediction_model import make_match_prediction


# from sportstensor.predictions import make_match_prediction


async def mls():
    matchDate = "2024-08-20"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.MLS,
        homeTeamName="Inter Miami",
        awayTeamName="Miami Fusion",
    )

    match_prediction = await make_match_prediction(match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

    return match_prediction


async def mlb():
    matchDate = "2024-08-25"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.BASEBALL,
        league=League.MLB,
        homeTeamName="Los Angeles Dodgers",
        awayTeamName="Oakland Athletics",
    )

    match_prediction = await make_match_prediction(match_prediction)

    print("match_prediction", match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

async def epl():
    matchDate = "2024-09-20"
    match_prediction = MatchPrediction(
        matchId="0b25bc4bd29ca0cd5d4b8031a3a36480",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.EPL,
        homeTeamName="Arsenal",
        awayTeamName="Chelsea",
    )

    match_prediction = await make_match_prediction(match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

async def nfl():
    matchDate = "2024-12-20"
    match_prediction = MatchPrediction(
        matchId="1bbabf31f6f1b885b0bfdbc4950d1c76",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.FOOTBALL,
        league=League.NFL,
        homeTeamName="Los Angeles Chargers",
        awayTeamName="Denver Broncos",
    )

    match_prediction = await make_match_prediction(match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

    return match_prediction

async def nba():
    matchDate = "2024-12-20"
    match_prediction = MatchPrediction(
        matchId="f7e9876fa54793d1ec8e693b27c99375",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.BASKETBALL,
        league=League.NBA,
        homeTeamName="Detroit Pistons",
        awayTeamName="Utah Jazz",
    )

    match_prediction = await make_match_prediction(match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

    return match_prediction

if __name__ == "__main__":
    #mls()
    #mlb()
    #epl()
    #asyncio.run(nfl())
    asyncio.run(nba())
