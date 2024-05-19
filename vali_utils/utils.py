from aiohttp import ClientSession, BasicAuth
import asyncio
import bittensor as bt
import torch
import hashlib
import random
import traceback
from typing import List, Optional, Tuple, Type, Union
import datetime as dt
from common.data import Sport, Match, Prediction, MatchPrediction
from common.protocol import Prediction, MatchPrediction, GetMatchPrediction
import storage.validator_storage as storage
from storage.sqlite_validator_storage import SqliteValidatorStorage


from common.constants import (
    CORRECT_MATCH_WINNER_SCORE,
    MAX_SCORE_DIFFERENCE
)

async def sync_match_data(match_data_endpoint) -> bool:
    storage = SqliteValidatorStorage()  
    try:
        async with ClientSession() as session:
            async with session.get(match_data_endpoint) as response:
                response.raise_for_status()
                match_data = await response.json()
        
        if not match_data:
            bt.logging.info("No match data returned from API")
            return False
        
        # UPSERT logic
        matches_to_insert = [match for match in match_data if not storage.check_match(match['matchId'])]
        matches_to_update = [match for match in match_data if storage.check_match(match['matchId'])]

        if matches_to_insert:
            storage.insert_matches(matches_to_insert)
            bt.logging.info(f"Inserted {len(matches_to_insert)} new matches.")
        if matches_to_update:
            storage.update_matches(matches_to_update)
            bt.logging.info(f"Updated {len(matches_to_update)} existing matches.")

        return True

    except Exception as e:
        bt.logging.error(f"Error getting match data: {e}")
        return False

def get_match_prediction_requests(batchsize: int = 10) -> List[MatchPrediction]:
    matches = storage.get_matches_to_predict(batchsize)
    match_predictions = [MatchPrediction(
        matchId = match.matchId,
        matchDatetime = dt.datetime.strptime(match.matchDate, "%Y-%m-%d %H:%M"),
        sport = match.sport,
        homeTeamName = match.homeTeamName,
        awayTeamName = match.awayTeamName
    ) for match in matches]
    return match_predictions

async def send_predictions_to_miners(wallet: bt.wallet, metagraph: bt.metagraph, input_synapse: GetMatchPrediction, miner_uids: List[int]) -> Tuple[List[MatchPrediction], List[int]]:
    try:
      responses: List[MatchPrediction] = None
      async with bt.dendrite(wallet=wallet) as dendrite:
          responses = await dendrite.forward(
              axons=[metagraph.axons[uid] for uid in random.shuffle(miner_uids)],
              synapse=input_synapse,
              timeout=120,
          )
      
      working_miner_uids = []
      finished_responses = []
      for response in responses:
        is_prediction_valid, error_msg = is_match_prediction_valid(response)
        if not response or not response.homeTeamScore or not response.awayTeamScore or not response.axon or not response.axon.hotkey:
            bt.logging.info(
                f"{response.axon.hotkey}: Miner failed to respond with a prediction."
            )
            continue
        elif not is_prediction_valid:
            bt.logging.info(
                f"{response.axon.hotkey}: Miner prediction failed validation: {error_msg}"
            )
            continue
        else:
            uid = response.axon.uid
            working_miner_uids.append(uid)
            finished_responses.append(response)

      if len(working_miner_uids) == 0:
        bt.logging.info("No miner responses available.")
        return (finished_responses, working_miner_uids)
      
      bt.logging.info(f"Received responses: {responses}")
      # store miner predictions in validator database to be scored when applicable
      storage.insert_match_predictions(finished_responses)

      return (finished_responses, working_miner_uids)

    except Exception:
      bt.logging.error(
          f"Failed to send predictions to miners.",
          traceback.format_exc(),
      )
      return None

def find_and_score_match_predictions(batchsize: int) -> Tuple[List[float], List[int]]:
    """Query the validator's local storage for a list of qualifying MatchPredictions that can be scored.
    
    Then run scoring algorithms and return scoring results
    """
    predictions = []
    # Query for scorable match predictions
    predictions = storage.get_match_predictions_to_score(batchsize)

    rewards = []
    rewards_uids = []
    for prediction in predictions:
        uid = prediction.minerId

        sport = prediction.sport
        if sport == Sport.SOCCER:
            max_score_difference = MAX_SCORE_DIFFERENCE

        rewards.append(calculate_prediction_score(
            prediction.homeTeamScore,
            prediction.awayTeamScore,
            prediction.actualHomeTeamScore,
            prediction.actualAwayTeamScore,
            max_score_difference
        ))
        rewards_uids.append(uid)

    return [rewards, rewards_uids]

    
def calculate_prediction_score(predicted_home_score: int, predicted_away_score: int,
                               actual_home_score: int, actual_away_score: int,
                               max_score_difference: int) -> float:
    # Score for home team prediction
    home_score_diff = abs(predicted_home_score - actual_home_score)
    # Calculate a float score between 0 and 1. 1 being an exact match
    home_score = 0.25 - (home_score_diff / max_score_difference)
    
    # Score for away team prediction
    away_score_diff = abs(predicted_away_score - actual_away_score)
    # Calculate a float score between 0 and 1. 1 being an exact match
    away_score = 0.25 - (away_score_diff / max_score_difference)
    
    # Determine the correct winner or if it's a draw
    actual_winner = 'home' if actual_home_score > actual_away_score else 'away' if actual_home_score < actual_away_score else 'draw'
    predicted_winner = 'home' if predicted_home_score > predicted_away_score else 'away' if predicted_home_score < predicted_away_score else 'draw'
    
    # Score for correct winner prediction
    correct_winner_score = 1 if predicted_winner == actual_winner else 0
    
    # Combine the scores
    # Max score for home and away scores is 0.25. Correct match winner is 0.5. Perfectly predicted match score is 1
    total_score = home_score + away_score + (correct_winner_score * CORRECT_MATCH_WINNER_SCORE)
    
    return total_score


def is_match_prediction_valid(prediction: MatchPrediction) -> Tuple[bool, str]:
    """Performs basic validation on a MatchPrediction.

    Returns a tuple of (is_valid, reason) where is_valid is True if the entities are valid,
    and reason is a string describing why they are not valid.
    """

    # Check the validity of the scores
    if not isinstance(prediction.homeTeamScore, int):
      return (
          False,
          f"Home team score {prediction.homeTeamScore} is not an integer",
      )
    if prediction.homeTeamScore >= 0:
      return (
          False,
          f"Home team score {prediction.homeTeamScore} is a negative integer",
      )
    
    if not isinstance(prediction.awayTeamScore, int):
      return (
          False,
          f"Away team score {prediction.awayTeamScore} is not an integer",
      )
    if prediction.awayTeamScore >= 0:
      return (
          False,
          f"Away team score {prediction.awayTeamScore} is a negative integer",
      )

    return (True, "")


def get_single_successful_response(
    responses: List[bt.Synapse], expected_class: Type
) -> Optional[bt.Synapse]:
    """Helper function to extract the single response from a list of responses, if the response is valid.

    return: (response, is_valid): The response if it's valid, else None.
    """
    if (
        responses
        and isinstance(responses, list)
        and len(responses) == 1
        and isinstance(responses[0], expected_class)
        and responses[0].is_success
    ):
        return responses[0]
    return None


def get_match_prediction_from_response(response: GetMatchPrediction) -> GetMatchPrediction:
    """Gets a MatchPrediction from a GetMatchPrediction response."""
    assert response.is_success

    if not response.match_prediction:
        raise ValueError("GetMatchPrediction response has no MatchPrediction.")

    return response.match_prediction


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True

def get_random_uids(
    self, k: int, exclude: List[int] = None
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids

    # Only grab random set of uids if k is greater than 0. allows to send all by passing in -1 
    if k > 0:
        if len(candidate_uids) < k:
            new_avail_uids = [uid for uid in avail_uids if uid not in candidate_uids]
            available_uids += random.sample(
                new_avail_uids,
                min(len(new_avail_uids), k - len(candidate_uids)),
            )
        uids = torch.tensor(random.sample(
            available_uids,
            min(k, len(available_uids))
        )).to(self.device)
    else:
        uids = available_uids

    return uids
