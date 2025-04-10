import os
from aiohttp import ClientSession, BasicAuth
import asyncio
import requests
from requests.auth import HTTPBasicAuth
import bittensor as bt
import random
import traceback
from typing import List, Optional, Tuple, Dict
import datetime as dt
from datetime import timedelta, timezone
import time
import copy

from common.data import League, Match, MatchPrediction, ProbabilityChoice, get_probablity_choice_from_string
from common.protocol import GetLeagueCommitments, GetMatchPrediction
import storage.validator_storage as storage
from storage.sqlite_validator_storage import SqliteValidatorStorage

from common.constants import (
    IS_DEV,
    VALIDATOR_TIMEOUT,
    MATCH_ODDS_CUTOFF_IN_DAYS,
    LEAGUES_ALLOWING_DRAWS,
    ROI_BET_AMOUNT,
)

from neurons.validator import Validator
from vali_utils import scoring_utils

# initialize our validator storage class
storage = SqliteValidatorStorage.get_instance()
storage.initialize()


async def sync_match_data(match_data_endpoint) -> bool:
    try:
        async with ClientSession() as session:
            # TODO: add in authentication
            async with session.get(match_data_endpoint) as response:
                response.raise_for_status()
                match_data = await response.json()

        if not match_data or "matches" not in match_data:
            bt.logging.info("No match data returned from API")
            return False

        match_data = match_data["matches"]

        # UPSERT logic
        matches_to_insert = []
        matches_to_update = []
        for item in match_data:
            if "matchId" not in item:
                bt.logging.error(f"Skipping match data missing matchId: {item}")
                continue

            match = Match(
                matchId=item["matchId"],
                matchDate=item["matchDate"],
                sport=item["sport"],
                league=item["matchLeague"],
                homeTeamName=item["homeTeamName"],
                awayTeamName=item["awayTeamName"],
                homeTeamScore=item["homeTeamScore"],
                awayTeamScore=item["awayTeamScore"],
                homeTeamOdds=item["homeTeamOdds"],
                awayTeamOdds=item["awayTeamOdds"],
                drawOdds=item["drawOdds"],
                isComplete=item["isComplete"],
            )
            if storage.check_match(item["matchId"]):
                matches_to_update.append(match)
            else:
                matches_to_insert.append(match)

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
    
def fetch_match_odds(vali, match_odds_data_endpoint: str, match_id: str) -> Dict:
    url = f"{match_odds_data_endpoint}?matchId={match_id}"
    
    keypair = vali.dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"

    response = requests.get(
        url, 
        auth=HTTPBasicAuth(hotkey, signature),
        timeout=10
    )
    response.raise_for_status()
    return response.json()

def sync_match_odds_data(vali, match_odds_data_endpoint: str) -> bool:
    try:
        # Get matches from the last MATCH_ODDS_CUTOFF_IN_DAYS days
        x_days_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=MATCH_ODDS_CUTOFF_IN_DAYS)
        recent_matches = storage.get_recently_completed_matches(x_days_ago)

        if not recent_matches:
            bt.logging.debug("No recent matches found in the database")
            return False

        odds_to_insert = []

        for match in recent_matches:
            try:
                odds_data = fetch_match_odds(vali, match_odds_data_endpoint, match.matchId)
            except Exception as e:
                bt.logging.error(f"Error fetching odds for match {match.matchId}: {e}")
                continue

            if not odds_data or "match_odds" not in odds_data:
                bt.logging.debug(f"No odds data returned from API for match {match.matchId}")
                continue

            match_odds = odds_data["match_odds"]
            if not match_odds:
                bt.logging.debug(f"Empty odds data returned from API for match {match.matchId}")
                continue

            for item in match_odds:
                if "matchId" not in item:
                    bt.logging.error(f"Skipping odds data missing matchId: {item}")
                    continue

                lastUpdated = dt.datetime.strptime(item["lastUpdated"], "%Y-%m-%dT%H:%M:%S")
                if not storage.check_match_odds(item["matchId"], lastUpdated):
                    odds_to_insert.append((
                        item["matchId"],
                        float(item.get("homeTeamOdds", 0) or 0),
                        float(item.get("awayTeamOdds", 0) or 0),
                        float(item.get("drawOdds", 0) or 0),
                        lastUpdated
                    ))

        if odds_to_insert:
            storage.insert_match_odds(odds_to_insert)
            bt.logging.info(f"Inserted {len(odds_to_insert)} odds for matches.")
            return True
        else:
            bt.logging.info("No new odds data collected from API.")
            return True

    except Exception as e:
        bt.logging.error(f"Unexpected error in sync_match_odds_data: {e}")
        return False

async def process_app_prediction_requests(
    vali: Validator,
    app_prediction_requests_endpoint: str,
    app_prediction_responses_endpoint: str,
) -> bool:
    keypair = vali.dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"
    try:
        async with ClientSession() as session:
            async with session.get(
                app_prediction_requests_endpoint, auth=BasicAuth(hotkey, signature)
            ) as response:
                response.raise_for_status()
                prediction_requests = await response.json()

        if not prediction_requests or "requests" not in prediction_requests:
            bt.logging.info("No app prediction requests returned from API")
            return False

        prediction_requests = prediction_requests["requests"]

        prediction_responses = []
        bt.logging.info(
            f"Sending {len(prediction_requests)} app requests to miners for predictions."
        )
        for pr in prediction_requests:
            match_prediction = MatchPrediction(
                matchId=pr["matchId"],
                matchDate=pr["matchDate"],
                sport=pr["sport"],
                league=pr["league"],
                homeTeamName=pr["homeTeamName"],
                awayTeamName=pr["awayTeamName"],
            )
            miner_hotkey = pr["miner_hotkey"]
            if IS_DEV:
                miner_uids = [9999]
            else:
                if miner_hotkey in vali.metagraph.hotkeys:
                    miner_uids = [
                        vali.metagraph.hotkeys.index(miner_hotkey) 
                    ]

            if len(miner_uids) > 0:
                bt.logging.info(
                    f"-- Sending match to miners {miner_uids} for predictions."
                )
                input_synapse = GetMatchPrediction(match_prediction=match_prediction)
                # Send prediction requests to miners and store their responses. TODO: do we need to mark the stored prediction as being an app request prediction? not sure it matters
                finished_responses, working_miner_uids = await send_predictions_to_miners(
                    vali, input_synapse, miner_uids
                )
                # Add the responses to the list of responses
                for response in finished_responses:
                    # Extract match_prediction and add app_request_id
                    match_prediction = response.match_prediction
                    match_prediction_dict = match_prediction.__dict__
                    match_prediction_dict["app_request_id"] = pr["app_request_id"]
                    match_prediction_dict["matchDate"] = str(match_prediction_dict["matchDate"]), # convert matchDate to string for serialization
                    prediction_responses.append(match_prediction_dict)

                # Loop through miners not responding and add them to the response list with a flag
                for uid in miner_uids:
                    if uid not in working_miner_uids:
                        match_prediction_not_working = pr
                        match_prediction_not_working["minerHasIssue"] = True
                        match_prediction_not_working["minerIssueMessage"] = "Miner did not respond with a prediction."
                        prediction_responses.append(match_prediction_not_working)

            else:
                bt.logging.info(
                    f"-- No miner uid found for {miner_hotkey}, skipping."
                )

        if len(prediction_responses) > 0:
            max_retries = 3
            # Post the prediction responses back to API
            for attempt in range(max_retries):
                try:
                    # Attempt to post the prediction responses
                    post_result = await post_app_prediction_responses(
                        vali, app_prediction_responses_endpoint, prediction_responses
                    )
                    return post_result
                except Exception as e:
                    bt.logging.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        # Wait before retrying
                        await asyncio.sleep(2)
                    else:
                        # Raise the exception if the maximum number of retries is reached
                        bt.logging.error(
                            f"Failed to post app prediction responses after {max_retries} attempts."
                        )
                        # Return False to indicate that the post failed
                        return False

        return True

    except Exception as e:
        bt.logging.error(f"Error processing app prediction requests: {e}")
        return False


async def post_app_prediction_responses(
    vali, prediction_responses_endpoint, prediction_responses
):
    keypair = vali.dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"
    try:
        # Post the app prediction request responses back to the api
        """
        prediction_responses = [
            {
                "app_request_id": "frontend-12345",
                "match_prediction":{
                    "matchId": "TeamATeamB202408011530",
                    "matchDate": "2024-08-01 15:30:00",
                    "sport": "Baseball",
                    "league": "MLB",
                    "homeTeamName": "Team A",
                    "awayTeamName": "Team B",
                    "homeTeamScore": 2,
                    "awayTeamScore": 3,
                    "isComplete": 1,
                    "miner_hotkey": "hotkey123"
                }
            }
        ]
        """
        async with ClientSession() as session:
            async with session.post(
                prediction_responses_endpoint,
                auth=BasicAuth(hotkey, signature),
                json=prediction_responses,
            ) as response:
                response.raise_for_status()
                bt.logging.info("Successfully posted app prediction responses to API.")
                return True

    except Exception as e:
        bt.logging.error(f"Error posting app prediction responses to API: {e}")
        return False


async def send_league_commitments_to_miners(
    vali: Validator, input_synapse: GetLeagueCommitments, miner_uids: List[int]
):
    try:
        random.shuffle(miner_uids)
        
        async def process_batch(batch_uids: List[int]):
            axons = [vali.metagraph.axons[uid] for uid in batch_uids]
            
            responses = await vali.dendrite(
                axons=axons,
                synapse=input_synapse,
                deserialize=True,
                timeout=VALIDATOR_TIMEOUT,
            )

            working_miner_uids = []
            finished_responses = []
            uid_league_updates = {}

            for response, uid in zip(responses, batch_uids):
                if (
                    response is None
                    or response.leagues is None
                    or response.axon is None
                    or response.axon.hotkey is None
                ):
                    bt.logging.info(
                        f"UID {uid}: Miner failed to respond to league commitments."
                    )
                    uid_league_updates[uid] = []
                    continue
                else:
                    working_miner_uids.append(uid)
                    finished_responses.append(response)
                    valid_leagues = [league for league in response.leagues if league in vali.ACTIVE_LEAGUES]
                    if len(valid_leagues) != len(response.leagues):
                        bt.logging.info(
                            f"UID {uid}: Some leagues were invalid and have been filtered out."
                        )
                    uid_league_updates[uid] = [valid_leagues[0]] if valid_leagues else [] # Only one league commitment per miner

            return finished_responses, working_miner_uids, uid_league_updates

        all_finished_responses = []
        all_working_miner_uids = []
        all_uid_league_updates: Dict[int, List[League]] = {}

        for i in range(0, len(miner_uids), vali.config.neuron.batch_size):
            batch = miner_uids[i:i+vali.config.neuron.batch_size]
            bt.logging.info(f"Sending league commitment requests to miners batch {batch}")
            finished_responses, working_miner_uids, uid_league_updates = await process_batch(batch)
            
            all_finished_responses.extend(finished_responses)
            all_working_miner_uids.extend(working_miner_uids)
            all_uid_league_updates.update(uid_league_updates)

        if len(all_working_miner_uids) == 0:
            bt.logging.info("No miner responses available.")
            return (all_finished_responses, all_working_miner_uids)

        bt.logging.info(f"Received responses from {len(all_working_miner_uids)} miners")
        bt.logging.info(f"Storing miner league commitments to validator storage.")
        
        # Bulk update of uids_to_leagues, uids_to_last_leagues, and uids_to_leagues_last_updated
        with vali.uids_to_leagues_lock and vali.uids_to_last_leagues_lock and vali.uids_to_leagues_last_updated_lock:
            for uid, leagues in all_uid_league_updates.items():
                vali.uids_to_leagues[uid] = leagues
                if len(leagues) > 0:
                    vali.uids_to_last_leagues[uid] = leagues
                    vali.uids_to_leagues_last_updated[uid] = dt.datetime.now(dt.timezone.utc)

        return (all_finished_responses, all_working_miner_uids)

    except Exception as e:
        bt.logging.error(
            f"Failed to send predictions to miners and store in validator database: {str(e)}",
            traceback.format_exc(),
        )
        return None
    

def clean_up_unscored_deregistered_match_predictions(active_miner_hotkeys: List[str]):
    """Deletes unscored predictions returned from miners that are no longer registered."""
    try:
        storage.delete_unscored_deregistered_match_predictions(active_miner_hotkeys)
    except Exception as e:
        bt.logging.error(f"Error cleaning up unscored deregistered predictions: {e}")


def get_match_prediction_requests(vali: Validator) -> Tuple[List[MatchPrediction], str]:
    matches = storage.get_matches_to_predict()
    if not matches:
        return [], "No upcoming matches scheduled."
    
    for match in matches:
        # Check if the matchId is in the database already and fix if needed
        storage.check_and_fix_match_prediction_requests(match.matchId)
    
    current_time = dt.datetime.now(dt.timezone.utc)
    match_prediction_requests = storage.get_match_prediction_requests()

    match_predictions = []
    next_predictions = []

    prediction_windows = [
        ('24_hour', timedelta(hours=24), timedelta(hours=23), 'prediction_24_hour'),
        ('12_hour', timedelta(hours=12), timedelta(hours=11), 'prediction_12_hour'),
        ('4_hour', timedelta(hours=4), timedelta(hours=3), 'prediction_4_hour'),
        ('10_min', timedelta(minutes=10), timedelta(minutes=5), 'prediction_10_min')
    ]
    
    for match in matches:
        if match.league not in vali.ACTIVE_LEAGUES:
            continue

        if match.matchId not in match_prediction_requests:
            match_prediction_requests[match.matchId] = {window[0]: False for window in prediction_windows}

        match_date_aware = match.matchDate.replace(tzinfo=timezone.utc)
        time_until_match = match_date_aware - current_time

        for window, upper_bound, lower_bound, update_key in prediction_windows:
            # Check if this match needs a prediction in the current cycle
            if (not match_prediction_requests[match.matchId][window] and
                upper_bound >= time_until_match > lower_bound):
                bt.logging.debug(f"Match found in prediction window {window}: {match.awayTeamName} at {match.homeTeamName} on {match.matchDate}")
                match_predictions.append(
                    MatchPrediction(
                        matchId=match.matchId,
                        matchDate=match.matchDate,
                        sport=match.sport,
                        league=match.league,
                        homeTeamName=match.homeTeamName,
                        awayTeamName=match.awayTeamName
                    )
                )
                storage.update_match_prediction_request(match.matchId, update_key)
                break  # Only one prediction per match per cycle

            elif not match_prediction_requests[match.matchId][window]:
                prediction_time = match_date_aware - upper_bound
                if prediction_time > current_time or time_until_match > upper_bound:
                    next_predictions.append((prediction_time, match, window))

    # Sort and get the top 3 upcoming matches
    next_predictions.sort(key=lambda x: x[0])  # Sort by prediction_time
    next_predictions = next_predictions[:3]
    
    # Prepare next matches info string
    window_display = {
        '24_hour': 'T-24h',
        '12_hour': 'T-12h',
        '4_hour': 'T-4h',
        '10_min': 'T-10m'
    }
    
    if next_predictions:
        next_match_info = "Next match prediction requests: "
        for pred_time, match, window in next_predictions:
            time_until_next = pred_time - current_time
            hours, remainder = divmod(time_until_next.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            
            next_match_info += (
                f"\n\t\t\t\t\t\t{int(hours)}h {int(minutes)}m | "
                f"{window_display.get(window, '?')} | "
                f"{match.homeTeamName} vs {match.awayTeamName} "
                f"({match.matchDate.strftime('%Y-%m-%d %H:%M')} UTC) | "
                f"{match.league}"
            )
    else:
        next_match_info = "No upcoming matches scheduled for prediction requests."

    return match_predictions, next_match_info


async def send_predictions_to_miners(
    vali: Validator, input_synapse: GetMatchPrediction, miner_uids: List[int]
) -> Tuple[List[MatchPrediction], List[int]]:
    try:
        random.shuffle(miner_uids)

        async def process_batch(batch_uids: List[int]):
            axons = [vali.metagraph.axons[uid] for uid in batch_uids]

            responses = await vali.dendrite(
                axons=axons,
                synapse=input_synapse,
                deserialize=True,
                timeout=VALIDATOR_TIMEOUT,
            )

            working_miner_uids = []
            finished_responses = []
            
            for response, uid in zip(responses, batch_uids):
                is_prediction_valid, error_msg = is_match_prediction_valid(
                    response.match_prediction,
                    input_synapse,
                )
                if (
                    response is None
                    or response.match_prediction is None
                    or response.match_prediction.probabilityChoice is None
                    or response.match_prediction.probability is None
                    or response.axon is None
                    or response.axon.hotkey is None
                ):
                    bt.logging.info(
                        f"UID {uid}: Miner failed to respond with a valid prediction."
                    )
                    continue
                elif not is_prediction_valid:
                    bt.logging.info(
                        f"UID {uid}: Miner prediction failed validation: {error_msg}"
                    )
                    continue
                else:
                    working_miner_uids.append(uid)
                    response.match_prediction.minerId = uid
                    response.match_prediction.hotkey = response.axon.hotkey
                    response.match_prediction.predictionDate = dt.datetime.now(dt.timezone.utc)
                    # round the probability to 4 decimal places
                    response.match_prediction.probability = round(response.match_prediction.probability, 4)
                    finished_responses.append(response)

            return finished_responses, working_miner_uids

        all_finished_responses = []
        all_working_miner_uids = []

        for i in range(0, len(miner_uids), vali.config.neuron.batch_size):
            batch = miner_uids[i:i+vali.config.neuron.batch_size]
            bt.logging.info(f"Sending prediction requests to miners batch {batch}")
            finished_responses, working_miner_uids = await process_batch(batch)
            
            all_finished_responses.extend(finished_responses)
            all_working_miner_uids.extend(working_miner_uids)

            # add a random delay between batches to spread out the requests
            await asyncio.sleep(random.uniform(1.0, 30.0)) # 1-30 seconds

        if len(all_working_miner_uids) == 0:
            bt.logging.info("No miner responses available.")
            return (all_finished_responses, all_working_miner_uids)

        bt.logging.info(f"Received responses from {len(all_working_miner_uids)} miners")
        bt.logging.info(f"Responses: {redact_scores(all_finished_responses)}")
        
        # store miner predictions in validator database to be scored when applicable
        bt.logging.info(f"Storing predictions in validator database.")
        storage.insert_match_predictions(all_finished_responses)

        return (all_finished_responses, all_working_miner_uids)

    except Exception as e:
        bt.logging.error(
            f"Failed to send predictions to miners and store in validator database: {str(e)}",
            traceback.format_exc(),
        )
        return None


def clean_up_unscored_deregistered_match_predictions(active_miner_hotkeys: List[str], active_miner_uids: List[int]):
    """Deletes unscored predictions returned from miners that are no longer registered."""
    try:
        storage.delete_unscored_deregistered_match_predictions(active_miner_hotkeys, active_miner_uids)
    except Exception as e:
        bt.logging.error(f"Error cleaning up unscored deregistered predictions: {e}")


def archive_deregistered_match_predictions(active_miner_hotkeys: List[str], active_miner_uids: List[int]):
    """Archives predictions from miners that are no longer registered."""
    try:
        storage.archive_match_predictions(active_miner_hotkeys, active_miner_uids)
    except Exception as e:
        bt.logging.error(f"Error archiving unscored deregistered predictions: {e}")


def find_and_score_edge_match_predictions(batchsize: int) -> Tuple[List[float], List[int], List[int], List[str], List[str]]:
    """Query the validator's local storage for a list of qualifying MatchPredictions that can be scored.

    Then run Closing Edge calculations and return results
    """

    # Query for scorable match predictions with actual match data
    predictions_with_match_data = storage.get_match_predictions_to_score(batchsize)

    edge_scores = []
    correct_winner_results = []
    miner_uids = []
    predictions = []
    sports = []
    leagues = []
    for pwmd in predictions_with_match_data:
        prediction = pwmd.prediction
        uid = prediction.minerId

        # Calculate the Closing Edge for the prediction
        edge, correct_winner_score = scoring_utils.calculate_edge(
            prediction_team=prediction.get_predicted_team(),
            prediction_prob=prediction.probability,
            actual_team=pwmd.get_actual_winner(),
            closing_odds=pwmd.get_closing_odds_for_predicted_outcome(),
        )
        prediction.closingEdge = edge
        
        edge_scores.append(edge)
        correct_winner_results.append(correct_winner_score)
        miner_uids.append(uid)
        predictions.append(prediction)
        sports.append(prediction.sport)
        leagues.append(prediction.league)

    # mark predictions as scored in the local db
    if len(predictions) > 0:
        storage.update_match_predictions(predictions)

    return [
        predictions,
        edge_scores,
        correct_winner_results,
        miner_uids,
        sports,
        leagues,
    ]


def is_match_prediction_valid(
    prediction: MatchPrediction, input_synapse: GetMatchPrediction
) -> Tuple[bool, str]:
    """Performs basic validation on a MatchPrediction.

    Returns a tuple of (is_valid, reason) where is_valid is True if the entities are valid,
    and reason is a string describing why they are not valid.
    """

    # Check if probabilityChoice is None
    if prediction.probabilityChoice is None:
        return (
            False,
            "Probability choice is None",
        )
    # Check the validity of the probability predictions
    if isinstance(prediction.probabilityChoice, str):
        if get_probablity_choice_from_string(prediction.probabilityChoice) is None:
            return (
                False,
                f"Probability choice {prediction.probabilityChoice} is not a valid choice",
            )
    elif isinstance(prediction.probabilityChoice, ProbabilityChoice):
        probability_choice = prediction.probabilityChoice
    else:
        return (
            False,
            f"Probability choice {prediction.probabilityChoice} is not of type ProbabilityChoice or str",
        )
    # Check if probability choice is 'Draw' and if so, that the prediction league is eligible to have Draws
    if (prediction.probabilityChoice == ProbabilityChoice.DRAW or prediction.probabilityChoice == ProbabilityChoice.DRAW.value):
        if input_synapse.match_prediction.league not in LEAGUES_ALLOWING_DRAWS:
            return (
                False,
                f"Probability choice {prediction.probabilityChoice} is not allowed for league: {prediction.league}",
            )
    
    if not isinstance(prediction.probability, float):
        return (
            False,
            f"Probability {prediction.probability} is not a float",
        )
    
    # Check that the probability is within the allowed range
    if round(prediction.probability, 4) <= 0 or round(prediction.probability, 4) >= 1:
        return (
            False,
            f"Probability {prediction.probability} rounded to {round(prediction.probability, 4)} is not between 0 and 1",
        )
    
    # Check that the current time is before the match date
    current_time = dt.datetime.now(dt.timezone.utc)
    # Ensure prediction.matchDate is offset-aware
    if prediction.matchDate.tzinfo is None:
        prediction_match_date = prediction.matchDate.replace(tzinfo=dt.timezone.utc)
    else:
        prediction_match_date = prediction.matchDate
    if current_time >= prediction_match_date:
        return (
            False,
            f"Current time {current_time} is not before start of match date {prediction_match_date}",
        )

    # Check that the prediction response matches the prediction request
    if (
        prediction.matchId != input_synapse.match_prediction.matchId
        or str(prediction.matchDate) != str(input_synapse.match_prediction.matchDate)
        or prediction.sport != input_synapse.match_prediction.sport
        or prediction.league != input_synapse.match_prediction.league
        or prediction.homeTeamName != input_synapse.match_prediction.homeTeamName
        or prediction.awayTeamName != input_synapse.match_prediction.awayTeamName
        or prediction.closingEdge != input_synapse.match_prediction.closingEdge # closingEdge is not part of the request and should be ignored/None
    ):
        return (
            False,
            f"Prediction response does not match prediction request",
        )

    return (True, "")


def post_prediction_edge_results(
    vali,
    prediction_edge_results_endpoint: str,
    league_scores: Dict[League, List[float]],
    league_edge_scores: Dict[League, List[float]],
    league_roi_scores: Dict[League, List[float]],
    league_roi_counts: Dict[League, List[int]],
    league_roi_payouts: Dict[League, List[float]],
    league_roi_market_payouts: Dict[League, List[float]],
    league_roi_incr_counts: Dict[League, List[int]],
    league_roi_incr_payouts: Dict[League, List[float]],
    league_roi_incr_market_payouts: Dict[League, List[float]],
    league_pred_counts: Dict[League, List[int]],
    league_pred_win_counts: Dict[League, List[int]],
    all_scores: Dict[str, List[float]],
):
    keypair = vali.dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"
    max_retries = 3
    
    # Get current timestamp for lastUpdated field
    current_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize the miner scores dictionary
    miner_scores = {}
    
    all_uids = vali.metagraph.uids.tolist()

    # if league doesn't exist in league_pred_counts or league_scores, add it with empty list
    for league in League:
        # only include the leagues we are using in our subnet
        if league.name not in (League.MLB.name, League.NFL.name, League.NBA.name, League.MLS.name, League.EPL.name):
            continue
        if league not in league_pred_counts:
            league_pred_counts[league] = [0] * len(all_uids)
        if league not in league_pred_win_counts:
            league_pred_win_counts[league] = [0] * len(all_uids)
        if league not in league_scores:
            league_scores[league] = [0] * len(all_uids)
        if league not in league_edge_scores:
            league_edge_scores[league] = [0] * len(all_uids)
        if league not in league_roi_scores:
            league_roi_scores[league] = [0] * len(all_uids)
    
    # Build complete score data for each UID
    for uid in all_uids:
        total_pred_count = (
            league_pred_counts[League.MLB][uid] +
            league_pred_counts[League.NFL][uid] +
            league_pred_counts[League.NBA][uid] +
            league_pred_counts[League.MLS][uid] +
            league_pred_counts[League.EPL][uid]
        )

        # Initialize ROI values for each league
        roi_values = {
            League.MLB: {"roi": 0.0, "market_roi": 0.0, "incr_roi": 0.0, "incr_market_roi": 0.0},
            League.NFL: {"roi": 0.0, "market_roi": 0.0, "incr_roi": 0.0, "incr_market_roi": 0.0},
            League.NBA: {"roi": 0.0, "market_roi": 0.0, "incr_roi": 0.0, "incr_market_roi": 0.0},
            League.MLS: {"roi": 0.0, "market_roi": 0.0, "incr_roi": 0.0, "incr_market_roi": 0.0},
            League.EPL: {"roi": 0.0, "market_roi": 0.0, "incr_roi": 0.0, "incr_market_roi": 0.0},
        }

        # Calculate ROI values
        for league in roi_values:
            if league not in league_roi_counts:
                continue
            if league_roi_counts[league][uid] > 0:
                roi_values[league]["roi"] = (
                    league_roi_payouts[league][uid] / (league_roi_counts[league][uid] * ROI_BET_AMOUNT)
                )
                roi_values[league]["market_roi"] = (
                    league_roi_market_payouts[league][uid] / (league_roi_counts[league][uid] * ROI_BET_AMOUNT)
                )
            if league_roi_incr_counts[league][uid] > 0:
                roi_values[league]["incr_roi"] = (
                    league_roi_incr_payouts[league][uid] / (league_roi_incr_counts[league][uid] * ROI_BET_AMOUNT)
                )
                roi_values[league]["incr_market_roi"] = (
                    league_roi_incr_market_payouts[league][uid] / (league_roi_incr_counts[league][uid] * ROI_BET_AMOUNT)
                )

        # Construct miner scores dictionary
        miner_scores[uid] = {
            "uid": uid,
            "hotkey": vali.metagraph.axons[uid].hotkey,
            "vali_hotkey": vali.wallet.hotkey.ss58_address,
            "total_score": all_scores[uid],
            "total_pred_count": total_pred_count,
            "lastUpdated": current_time,
        }

        # Dynamically update miner_scores with league-specific data
        for league in roi_values:
            league_name = league.name.lower()  # Convert to lowercase for consistency
            miner_scores[uid].update({
                f"{league_name}_score": league_scores[league][uid],
                f"{league_name}_edge_score": league_edge_scores[league][uid],
                f"{league_name}_roi_score": league_roi_scores[league][uid],
                f"{league_name}_roi": roi_values[league]["roi"],
                f"{league_name}_market_roi": roi_values[league]["market_roi"],
                f"{league_name}_incr_roi": roi_values[league]["incr_roi"],
                f"{league_name}_incr_market_roi": roi_values[league]["incr_market_roi"],
                f"{league_name}_pred_count": league_pred_counts[league][uid],
                f"{league_name}_pred_win_count": league_pred_win_counts[league][uid],
            })

    for attempt in range(max_retries):
        try:
            # Post the scoring results back to the api
            scoring_results = {
                "miner_scores": miner_scores,
            }
            
            response = requests.post(
                prediction_edge_results_endpoint,
                auth=HTTPBasicAuth(hotkey, signature),
                json=scoring_results,
            )
            response.raise_for_status()
            bt.logging.info("Successfully posted prediction edge results to API.")
            return response

        except Exception as e:
            bt.logging.error(
                f"Error posting prediction edge results to API, attempt {attempt + 1}: {e}"
            )
            if attempt < max_retries - 1:
                # Wait before retrying
                time.sleep(2)
            else:
                bt.logging.error(
                    f"Max retries attempted posting prediction edge results to API. Contact a Sportstensor admin."
                )
                raise  # Re-raise the last exception after all retries are exhausted


async def post_scored_predictions(
    vali,
    scored_predictions_endpoint,
    predictions,
):
    keypair = vali.dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"
    max_retries = 3

    # Filter down our scored predictions to only those predicted within 10 minutes of the match start
    filtered_predictions = []
    for prediction in predictions:
        # Ensure our dates are offset-aware
        match_datetime = prediction.matchDate
        if prediction.matchDate.tzinfo is None:
            match_datetime = prediction.matchDate.replace(tzinfo=dt.timezone.utc)
        prediction_datetime = prediction.predictionDate
        if prediction.predictionDate.tzinfo is None:
            prediction_datetime = prediction.predictionDate.replace(tzinfo=dt.timezone.utc)

        # Filter down our predictions to only those predicted within 10 minutes of the match start
        if (match_datetime - prediction_datetime).total_seconds() < 600:
            prediction_dict = prediction.__dict__
            prediction_dict["predictionDate"] = str(prediction_dict["predictionDate"]) # convert predictionDate to string for serialization
            prediction_dict["matchDate"] = str(prediction_dict["matchDate"]) # convert matchDate to string for serialization
            prediction_dict["scoredDate"] = str(prediction_dict["scoredDate"]) # convert scoredDate to string for serialization
            filtered_predictions.append(prediction_dict)

    for attempt in range(max_retries):
        try:
            # Post the scored predictions back to the api
            results = {
                "predictions": filtered_predictions,
            }
            async with ClientSession() as session:
                async with session.post(
                    scored_predictions_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json=results,
                ) as response:
                    response.raise_for_status()
                    bt.logging.info("Successfully posted scored predictions to API.")
                    return response

        except Exception as e:
            bt.logging.error(
                f"Error posting scored predictions to API, attempt {attempt + 1}: {e}"
            )
            if attempt < max_retries - 1:
                # Wait before retrying
                await asyncio.sleep(2)
            else:
                bt.logging.error(
                    f"Max retries attempted posting scored predictions to API. Contact a Sportstensor admin."
                )


def redact_scores(responses):
    redacted_responses = []
    for response in responses:
        # Create a copy of the response to avoid modifying the original
        redacted_response = copy.deepcopy(response)

        # Redact the homeTeamScore, awayTeamScore, probabilityChoice, and probability fields
        if (
            hasattr(redacted_response.match_prediction, "homeTeamScore")
            and redacted_response.match_prediction.homeTeamScore is not None
        ):
            redacted_response.match_prediction.homeTeamScore = "REDACTED"
        if (
            hasattr(redacted_response.match_prediction, "awayTeamScore")
            and redacted_response.match_prediction.awayTeamScore is not None
        ):
            redacted_response.match_prediction.awayTeamScore = "REDACTED"
        if (
            hasattr(redacted_response.match_prediction, "probabilityChoice")
            and redacted_response.match_prediction.probabilityChoice is not None
        ):
            redacted_response.match_prediction.probabilityChoice = "REDACTED"
        if (
            hasattr(redacted_response.match_prediction, "probability")
            and redacted_response.match_prediction.probability is not None
        ):
            redacted_response.match_prediction.probability = "REDACTED"

        redacted_responses.append(redacted_response)
    return redacted_responses


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


def get_random_uids(self, k: int, exclude: List[int] = None) -> List[int]:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (List[int]): Randomly sampled available uids.
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

    # Check if candidate_uids contain enough for querying, if not grab all available uids
    available_uids = candidate_uids

    # Only grab random set of uids if k is greater than 0. allows to send all by passing in -1
    if k > 0:
        if len(candidate_uids) < k:
            new_avail_uids = [uid for uid in avail_uids if uid not in candidate_uids]
            available_uids += random.sample(
                new_avail_uids,
                min(len(new_avail_uids), k - len(candidate_uids)),
            )
        uids = random.sample(available_uids, min(k, len(available_uids)))
    else:
        uids = available_uids

    return uids
