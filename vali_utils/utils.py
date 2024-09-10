from aiohttp import ClientSession, BasicAuth
import asyncio
import bittensor as bt
import random
import traceback
from typing import List, Optional, Tuple, Type
from collections import defaultdict
import copy
import gspread
from common.data import Sport, Match, Stat, Player, MatchPrediction, PlayerStat, PlayerPrediction, Stat, PlayerEligibleStat
from common.protocol import GetMatchPrediction, GetPrediction
import storage.validator_storage as storage
from storage.sqlite_validator_storage import SqliteValidatorStorage

from common.constants import (
    IS_DEV,
    CORRECT_MATCH_WINNER_SCORE,
    TOTAL_SCORE_THRESHOLD,
    MAX_SCORE_DIFFERENCE,
    MAX_SCORE_DIFFERENCE_SOCCER,
    MAX_SCORE_DIFFERENCE_FOOTBALL,
    MAX_SCORE_DIFFERENCE_BASEBALL,
    MAX_SCORE_DIFFERENCE_BASKETBALL,
    MAX_SCORE_DIFFERENCE_CRICKET,
    NUM_PLAYERS_TO_PICK,
    NUM_ELIGIBLE_PLAYER_STATS,
)

from neurons.validator import Validator

# initialize our validator storage class
storage = SqliteValidatorStorage()


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
    

async def sync_player_match__stats_data(match_data_endpoint) -> bool:
    try:
        async with ClientSession() as session:
            async with session.get(match_data_endpoint) as response:
                response.raise_for_status()
                player_stats = await response.json()

        if not player_stats or "stats" not in player_stats:
            bt.logging.info("No player stats returned from API")
            return False

        player_stats = player_stats["stats"]

        # UPSERT logic
        stats_to_insert = []
        stats_to_update = []
        for item in player_stats:
            if "playerStatId" not in item:
                bt.logging.error(f"Skipping player stat missing playerStatId: {item}")
                continue

            match = storage.get_match(item["matchId"])
            # If the match doesn't exist, skip this player stat
            if match is not None:
                player_stat = PlayerStat(
                    playerStatId=item["playerStatId"],
                    match=match,
                    playerName=item["playerName"],
                    playerTeam=item["playerTeam"],
                    playerPosition=item["playerPosition"],
                    statType=item["statType"],
                    statValue=item["statValue"],
                )
                if storage.check_player_stat(item["playerStatId"]):
                    stats_to_update.append(player_stat)
                else:
                    stats_to_insert.append(player_stat)

        if stats_to_insert:
            storage.insert_player_stats(stats_to_insert)
            bt.logging.info(f"Inserted {len(stats_to_insert)} new player stats.")
        if stats_to_update:
            storage.update_player_stats(stats_to_update)
            bt.logging.info(f"Updated {len(stats_to_update)} existing player stats.")

        return True

    except Exception as e:
        bt.logging.error(f"Error getting player data: {e}")
        return False

async def sync_player_data(player_data_endpoint) -> bool:
    try:
        gc = gspread.api_key('AIzaSyAOj3881YK1QGkK07tyJr_bz2o106YcIXg')
        spreadsheet = gc.open_by_url(player_data_endpoint)
        statsSheet = spreadsheet.sheet1
        playersSheet = spreadsheet.get_worksheet(1)
        stats_data = statsSheet.get_all_records()
        players_data = playersSheet.get_all_records()

        # Sync stats data
        stats_to_insert = []
        for item in stats_data:
            if "statId" not in item or not item["statId"]:
                bt.logging.error(f"Skipping stats data missing statId or empty statId: {item}")
                continue
            stat = Stat(
                statId=item["statId"],
                statName=item["statName"],
                statAbbr=item["statAbbr"],
                statDescription=item["statDescription"],
                statType=item["statType"],
                sport=item["sport"],
            )
            if not storage.check_stats(item["statId"]):
                stats_to_insert.append(stat)
        if stats_to_insert:
            storage.insert_stats(stats_to_insert)
            bt.logging.info(f"Inserted {len(stats_to_insert)} new stats.")
        
        # Sync players data
        players_to_insert = []
        players_to_update = []
        players_elgible_stats_update = []
        players_elgible_stats_insert = []
        for item in players_data:
            if "playerId" not in item:
                bt.logging.error(f"Skipping player data missing playerId: {item}")
                continue

            player = Player(
                playerId=item["playerId"],
                playerName=item["playerName"],
                playerTeam=item["playerTeam"],
                playerPosition=item["playerPosition"],
                sport=item["sport"],
                league=item["league"],
                stats=item["stats"],
            )
            if storage.check_player(item['playerId']):
                players_to_update.append(player)
                # Prepare eligible stats based on player sport and stats type
                eligible_stats = [
                    PlayerEligibleStat(
                        playerId=player.playerId,
                        statId=stat["statId"]
                    )
                    for stat in stats_data
                    if "statId" in stat and player.sport == stat["sport"] and player.stats == stat['statType']
                ]
                players_elgible_stats_update.extend(eligible_stats)
            else:
                players_to_insert.append(player)
                # Prepare eligible stats based on player sport and stats type
                eligible_stats = [
                    PlayerEligibleStat(
                        playerId=player.playerId,
                        statId=stat["statId"]
                    )
                    for stat in stats_data
                    if "statId" in stat and player.sport == stat["sport"] and player.stats == stat['statType']
                ]
                players_elgible_stats_insert.extend(eligible_stats)
        if players_to_insert:
            storage.insert_players(players_to_insert)
            bt.logging.info(f"Inserted {len(players_to_insert)} new players.")
        if players_to_update:
            storage.update_players(players_to_update)
            bt.logging.info(f"Updated {len(players_to_update)} existing players.")
        if players_elgible_stats_insert:
            storage.insert_player_eligible_stats(players_elgible_stats_insert)
            bt.logging.info(f"Inserted {len(players_elgible_stats_insert)} new player elgible stats.")
        if players_elgible_stats_update:
            storage.update_player_eligible_stats(players_elgible_stats_update)
            bt.logging.info(f"Updated {len(players_elgible_stats_update)} existing player elgible stats.")
        
        # Sync eligible player stats data

        return True

    except Exception as e:
        bt.logging.error(f"Error getting stats and players data: {e}")
        return False

async def process_app_prediction_requests(
    vali: Validator, app_prediction_requests_endpoint: str
) -> bool:
    try:
        async with ClientSession() as session:
            # TODO: add in authentication
            async with session.get(app_prediction_requests_endpoint) as response:
                response.raise_for_status()
                prediction_requests = await response.json()

        if not prediction_requests or "matches" not in prediction_requests:
            bt.logging.info("No app prediction requests returned from API")
            return False

        prediction_requests = prediction_requests["matches"]

        bt.logging.info(
            f"Sending {len(prediction_requests)} app requests to miners for predictions."
        )
        for pr in prediction_requests:
            match_prediction = MatchPrediction(
                matchId=pr["matchId"],
                matchDate=pr["matchDate"],
                sport=pr["sport"],
                homeTeamName=pr["homeTeamName"],
                awayTeamName=pr["awayTeamName"],
            )
            miner_hotkey = pr["miner_hotkey"]
            if IS_DEV:
                miner_uids = [9999]
            else:
                miner_uids = [
                    ax.uid for ax in vali.metagraph.axons if ax.hotkey == miner_hotkey
                ]

            input_synapse = GetMatchPrediction(match_prediction=match_prediction)
            # Send prediction requests to miners and store their responses. TODO: do we need to mark the stored prediction as being an app request prediction? not sure it matters
            finished_responses, working_miner_uids = await send_predictions_to_miners(
                vali, input_synapse, miner_uids
            )

            # Post the response back per prediction_request or batch? Probably batch.

        return True

    except Exception as e:
        bt.logging.error(f"Error syncing app prediction requests: {e}")
        return False


def get_match_prediction_requests(batchsize: int = 1) -> List[MatchPrediction]:
    matches = storage.get_matches_to_predict(batchsize)
    match_predictions = [
        MatchPrediction(
            matchId=match.matchId,
            matchDate=str(match.matchDate),
            sport=match.sport,
            league=match.league,
            homeTeamName=match.homeTeamName,
            awayTeamName=match.awayTeamName,
        )
        for match in matches
    ]
    return match_predictions

def get_player_prediction_requests(match: MatchPrediction, batchsize: int = NUM_PLAYERS_TO_PICK) -> List[PlayerPrediction]:
    player_prediction_requests: List[PlayerPrediction] = []
    players = storage.get_players_to_predict(match, batchsize)
    for player in players:
        player_eligible_stats: List[Stat] = storage.get_eligible_player_stats(player, NUM_ELIGIBLE_PLAYER_STATS)
        for stat in player_eligible_stats:
            player_prediction_requests.append(
                PlayerPrediction(
                    matchId=match.matchId,
                    matchDate=str(match.matchDate),
                    sport=match.sport,
                    league=match.league,
                    playerName=player.playerName,
                    playerTeam=player.playerTeam,
                    playerPosition=player.playerPosition,
                    statType=stat.statType,
                    statAbbr=stat.statAbbr,
                    statDescription=stat.statDescription,
                    statName=stat.statName,
                )
            )
    return player_prediction_requests

async def send_predictions_to_miners(
    vali: Validator, input_synapse: GetPrediction, miner_uids: List[int]
) -> Tuple[List[MatchPrediction], List[list[PlayerPrediction]], List[int]]:
    mp_prediction = input_synapse.prediction['mp']
    pp_predictions = input_synapse.prediction['ipp']
    try:
        if IS_DEV:
            # For now, just return a list of random MatchPrediction and PlayerPrediction responses
            mp_responses: List[MatchPrediction] = []
            pp_responses: List[list[PlayerPrediction]] = []
            for uid in miner_uids:
                mp_responses.append(MatchPrediction(
                        matchId=mp_prediction.matchId,
                        matchDate=mp_prediction.matchDate,
                        sport=mp_prediction.sport,
                        league=mp_prediction.league,
                        homeTeamName=mp_prediction.homeTeamName,
                        awayTeamName=mp_prediction.awayTeamName,
                        homeTeamScore=random.randint(0, 10),
                        awayTeamScore=random.randint(0, 10),
                    )
                )
                ipp_responses = []
                for playerPrediction in pp_predictions:
                    ipp_responses.append(
                        PlayerPrediction(
                            matchId=playerPrediction.matchId,
                            matchDate=playerPrediction.matchDate,
                            sport=playerPrediction.sport,
                            league=playerPrediction.league,
                            playerName=playerPrediction.playerName,
                            playerTeam=playerPrediction.playerTeam,
                            playerPosition=playerPrediction.playerPosition,
                            statName=playerPrediction.statName,
                            statAbbr=playerPrediction.statAbbr,
                            statDescription=playerPrediction.statDescription,
                            statType=playerPrediction.statType,
                            statValue=random.randint(0, 10),
                        )
                    )
                pp_responses.append(ipp_responses)
        else:
            random.shuffle(miner_uids)
            axons = [vali.metagraph.axons[uid] for uid in miner_uids]

            # convert matchDate to string for serialization
            mp_prediction.matchDate = str(
                mp_prediction.matchDate
            )
            for playerPrediction in pp_predictions:
                playerPrediction.matchDate = str(
                    playerPrediction.matchDate
                )
            responses: List[GetPrediction] = await vali.dendrite(
                # Send the query to selected miner axons in the network.
                axons=axons,
                synapse=input_synapse,
                deserialize=True,
                timeout=120
            )

        working_miner_uids = []
        finished_mp_responses = []
        finished_pp_responses = []
        for response in responses:
            mp_response = response.prediction['mp']
            pp_response = response.prediction['ipp']
            is_mp_valid, error_msg_for_mp = is_match_prediction_valid(
                mp_response,
                mp_prediction,
            )
            is_ipp_valid, error_msg_for_ipp = is_player_prediction_valid(
                pp_response,
                pp_predictions,
            )
            if IS_DEV:
                uid = miner_uids.pop(random.randrange(len(miner_uids)))
                working_miner_uids.append(uid)
                finished_mp_responses.append(mp_response)
                finished_pp_responses.append(pp_response)
            else:
                if (
                    mp_response is None
                    or mp_response.homeTeamScore is None
                    or mp_response.awayTeamScore is None
                    or input_synapse.axon is None
                    or input_synapse.axon.hotkey is None
                ):
                    bt.logging.info(
                        f"{input_synapse.axon.hotkey}: Miner failed to respond with a match prediction."
                    )
                    continue
                elif (pp_response is None) or (pp_response == []) or any(prediction.statValue is None for prediction in pp_response):
                    bt.logging.info(
                        f"{input_synapse.axon.hotkey}: Miner failed to respond with a player prediction."
                    )
                    continue
                elif not is_mp_valid:
                    bt.logging.info(
                        f"{input_synapse.axon.hotkey}: Miner match prediction failed validation: {error_msg_for_mp}"
                    )
                    continue
                elif not is_ipp_valid:
                    bt.logging.info(
                        f"{input_synapse.axon.hotkey}: Miner player prediction failed validation: {error_msg_for_ipp}"
                    )
                    continue
                else:
                    uid = [
                        uid
                        for uid, axon in zip(miner_uids, axons)
                        if axon.hotkey == input_synapse.axon.hotkey
                    ][0]
                    working_miner_uids.append(uid)
                    mp_response.minerId = uid
                    mp_response.hotkey = input_synapse.axon.hotkey
                    for pp in pp_response:
                        pp.minerId = uid
                        pp.hotkey = input_synapse.axon.hotkey
                    finished_mp_responses.append(mp_response)
                    finished_pp_responses.append(pp_response)

        if len(working_miner_uids) == 0:
            bt.logging.info("No miner responses available.")
            return (finished_mp_responses, finished_pp_responses, working_miner_uids)

        bt.logging.info(f"Received responses: {redact_scores(responses)}")
        # store miner predictions in validator database to be scored when applicable
        bt.logging.info(f"Storing predictions in validator database.")
        storage.insert_match_predictions(finished_mp_responses)
        storage.insert_player_predictions(finished_pp_responses)

        return (finished_mp_responses, finished_pp_responses, working_miner_uids)

    except Exception:
        bt.logging.error(
            f"Failed to send predictions to miners and store in validator database.",
            traceback.format_exc(),
        )
        return None


def find_and_score_match_predictions(batchsize: int) -> Tuple[List[float], List[int]]:
    """Query the validator's local storage for a list of qualifying MatchPredictions that can be scored.

    Then run scoring algorithms and return scoring results
    """

    # Query for scorable match predictions with actual match data
    predictions_with_match_data = storage.get_match_predictions_to_score(batchsize)

    rewards = []
    correct_winner_results = []
    rewards_uids = []
    predictions = []
    sports = []
    leagues = []
    for pwmd in predictions_with_match_data:
        prediction = pwmd.prediction
        uid = prediction.minerId

        sport = prediction.sport
        max_score_difference = MAX_SCORE_DIFFERENCE
        if sport == Sport.SOCCER:
            max_score_difference = MAX_SCORE_DIFFERENCE_SOCCER
        elif sport == Sport.FOOTBALL:
            max_score_difference = MAX_SCORE_DIFFERENCE_FOOTBALL
        elif sport == Sport.BASEBALL:
            max_score_difference = MAX_SCORE_DIFFERENCE_BASEBALL
        elif sport == Sport.BASKETBALL:
            max_score_difference = MAX_SCORE_DIFFERENCE_BASKETBALL
        elif sport == Sport.CRICKET:
            max_score_difference = MAX_SCORE_DIFFERENCE_CRICKET

        total_score, correct_winner_score = calculate_prediction_score(
            prediction.homeTeamScore,
            prediction.awayTeamScore,
            pwmd.actualHomeTeamScore,
            pwmd.actualAwayTeamScore,
            max_score_difference,
        )
        rewards.append(total_score)
        correct_winner_results.append(correct_winner_score)
        rewards_uids.append(uid)
        predictions.append(prediction)
        sports.append(sport)
        leagues.append(prediction.league)

    # mark predictions as scored in the local db
    if len(predictions) > 0:
        storage.update_match_predictions(predictions)

    # Aggregate rewards for each miner
    aggregated_rewards = defaultdict(float)
    num_predictions_below_threshold = 0
    for uid, reward in zip(rewards_uids, rewards):
        # Adjust the reward to 0 if it doesn't meat our threshold
        if reward < TOTAL_SCORE_THRESHOLD:
            num_predictions_below_threshold += 1
            reward = 0
        aggregated_rewards[uid] += reward

    bt.logging.debug(f"Total prediction scores below threshold of {TOTAL_SCORE_THRESHOLD}: {num_predictions_below_threshold} of {len(rewards)}")

    # Convert the aggregated rewards to a list of tuples (uid, aggregated_reward)
    aggregated_rewards_list = list(aggregated_rewards.items())

    # Normalize the aggregated rewards so that they sum up to 1.0
    total_rewards = sum(reward for uid, reward in aggregated_rewards_list)
    if total_rewards > 0:
        normalized_rewards = [
            reward / total_rewards for uid, reward in aggregated_rewards_list
        ]
    else:
        # Handle the case where total_rewards is 0 to avoid division by zero
        normalized_rewards = [0 for uid, reward in aggregated_rewards_list]

    # Extract the corresponding UIDs for the normalized rewards
    normalized_rewards_uids = [uid for uid, reward in aggregated_rewards_list]

    return [
        normalized_rewards,
        normalized_rewards_uids,
        rewards,
        correct_winner_results,
        rewards_uids,
        sports,
        leagues,
    ]


def calculate_prediction_score(
    predicted_home_score: int,
    predicted_away_score: int,
    actual_home_score: int,
    actual_away_score: int,
    max_score_difference: int,
) -> float:

    # Score for home team prediction
    home_score_diff = abs(predicted_home_score - actual_home_score)
    # Calculate a float score between 0 and 1. 1 being an exact match
    home_score = max(0, 0.25 - (home_score_diff / max_score_difference))

    # Score for away team prediction
    away_score_diff = abs(predicted_away_score - actual_away_score)
    # Calculate a float score between 0 and 1. 1 being an exact match
    away_score = max(0, 0.25 - (away_score_diff / max_score_difference))

    # Determine the correct winner or if it's a draw
    actual_winner = (
        "home"
        if actual_home_score > actual_away_score
        else "away" if actual_home_score < actual_away_score else "draw"
    )
    predicted_winner = (
        "home"
        if predicted_home_score > predicted_away_score
        else "away" if predicted_home_score < predicted_away_score else "draw"
    )

    # Score for correct winner prediction
    correct_winner_score = 1 if predicted_winner == actual_winner else 0

    # Combine the scores
    # Max score for home and away scores is 0.25. Correct match winner is 0.5. Perfectly predicted match score is 1
    total_score = (
        home_score + away_score + (correct_winner_score * CORRECT_MATCH_WINNER_SCORE)
    )

    return total_score, correct_winner_score

def is_player_prediction_valid(prediction: List[PlayerPrediction], input_synapse: List[PlayerPrediction]) -> Tuple[bool, str]:
    """Performs basic validation on a PlayerPrediction.

    Returns a tuple of (is_valid, reason) where is_valid is True if the entities are valid,
    and reason is a string describing why they are not valid.
    """

    # Check if the length of predictions matches
    if len(prediction) != len(input_synapse):
        return (False, "The number of predictions does not match the number of input synapses.")

    # Validate each prediction against the input synapse
    for pred, input_pred in zip(prediction, input_synapse):
        # Validate stat value
        if not isinstance(pred.statValue, int):
            return (
                False,
                f"Player stat value {pred.statValue} is not an integer",
            )
        if pred.statValue < 0:
            return (
                False,
                f"Player stat value {pred.statValue} is a negative integer",
            )

        # Check that the prediction response matches the prediction request
        if (
            str(pred.matchDate) != str(input_pred.matchDate)
            or pred.sport != input_pred.sport
            or pred.league != input_pred.league
            or pred.playerName != input_pred.playerName
            or pred.playerTeam != input_pred.playerTeam
        ):
            return (
                False,
                "Prediction response does not match prediction request",
            )

    return (True, "")

def is_match_prediction_valid(
    prediction: MatchPrediction, input_synapse: MatchPrediction
) -> Tuple[bool, str]:
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
    if prediction.homeTeamScore < 0:
        return (
            False,
            f"Home team score {prediction.homeTeamScore} is a negative integer",
        )

    if not isinstance(prediction.awayTeamScore, int):
        return (
            False,
            f"Away team score {prediction.awayTeamScore} is not an integer",
        )
    if prediction.awayTeamScore < 0:
        return (
            False,
            f"Away team score {prediction.awayTeamScore} is a negative integer",
        )

    # Check that the prediction response matches the prediction request
    if (
        str(prediction.matchDate) != str(input_synapse.matchDate)
        or prediction.sport != input_synapse.sport
        or prediction.league != input_synapse.league
        or prediction.homeTeamName != input_synapse.homeTeamName
        or prediction.awayTeamName != input_synapse.awayTeamName
    ):
        return (
            False,
            f"Prediction response does not match prediction request",
        )

    return (True, "")


async def post_prediction_results(
    vali,
    prediction_results_endpoint,
    prediction_scores,
    correct_winner_results,
    prediction_rewards_uids,
    prediction_results_hotkeys,
    prediction_sports,
    prediction_leagues,
):
    keypair = vali.dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Post the scoring results back to the api
            scoring_results = {
                "scores": prediction_scores,
                "correct_winner_results": correct_winner_results,
                "uids": prediction_rewards_uids,
                "hotkeys": prediction_results_hotkeys,
                "sports": prediction_sports,
                "leagues": prediction_leagues,
            }
            async with ClientSession() as session:
                async with session.post(
                    prediction_results_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json=scoring_results,
                ) as response:
                    response.raise_for_status()
                    bt.logging.info("Successfully posted prediction results to API.")
                    return response

        except Exception as e:
            bt.logging.error(
                f"Error posting prediction results to API, attempt {attempt + 1}: {e}"
            )
            if attempt < max_retries - 1:
                # Wait before retrying
                await asyncio.sleep(2)
            else:
                bt.logging.error(
                    f"Max retries attempted posting prediction results to API. Contact a Sportstensor admin."
                )


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


def get_match_prediction_from_response(
    response: GetMatchPrediction,
) -> GetMatchPrediction:
    """Gets a MatchPrediction from a GetMatchPrediction response."""
    assert response.is_success

    if not response.match_prediction:
        raise ValueError("GetMatchPrediction response has no MatchPrediction.")

    return response.match_prediction


def redact_scores(responses: List[GetPrediction]):
    redacted_responses = []
    for response in responses:
        # Create a copy of the response to avoid modifying the original
        mp_response = response.prediction["mp"]
        redacted_response = copy.deepcopy(mp_response)

        # Redact the homeTeamScore and awayTeamScore
        if (
            hasattr(redacted_response, "homeTeamScore")
            and redacted_response.homeTeamScore is not None
        ):
            redacted_response.homeTeamScore = "REDACTED"
        if (
            hasattr(redacted_response, "awayTeamScore")
            and redacted_response.awayTeamScore is not None
        ):
            redacted_response.awayTeamScore = "REDACTED"

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
