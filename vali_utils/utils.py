import asyncio
import bittensor as bt
import torch
import hashlib
import random
import traceback
from typing import List, Optional, Tuple, Type, Union
import datetime as dt
from common import constants
from common.data import (
    MatchPrediction
)
from common.protocol import Prediction, MatchPrediction, GetMatchPrediction
import storage.validator_storage as storage


async def send_predictions_to_miners(wallet: bt.wallet, metagraph: bt.metagraph, input_synapse: GetMatchPrediction, miner_uids: List[int]):
    try:
      responses: List[MatchPrediction] = None
      async with bt.dendrite(wallet=wallet) as dendrite:
          responses = await dendrite.forward(
              axons=[metagraph.axons[uid] for uid in miner_uids],
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

def find_match_predictions_to_score(batchsize: int) -> List[MatchPrediction]:
    """Query the validator's local storage for a list of qualifying MatchPredictions that can be scored.
    
    This should only get executed once match stats have been properly pulled into the validator.
    Or that check should happen here.
    """

    predictions = []
    # TODO: query predictions that have not been scored that we have data on.
    storage.get_match_predictions_to_score(batchsize)

    return predictions


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
