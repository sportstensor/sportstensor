import math
import numpy as np
from scipy.stats import pareto
import torch
import datetime as dt
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Tuple, Optional
from tabulate import tabulate

import bittensor as bt
from storage.sqlite_validator_storage import get_storage
from vali_utils import utils
from common.data import League, MatchPredictionWithMatchData, ProbabilityChoice
from common.constants import (
    MAX_PREDICTION_DAYS_THRESHOLD,
    ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE,
    NO_LEAGUE_COMMITMENT_PENALTY
)

def calculate_edge(prediction_team: str, prediction_prob: float, actual_team: str, winning_closing_odds: float, losing_closing_odds: float) -> Tuple[float, int]:
    """
    Calculate the edge for a prediction on a two-sided market.
    
    :param prediction_team: str, either 'A' or 'B', representing the team chosen
    :param prediction_prob: float, weak learner's probability of winning for the chosen team at prediction time
    :param actual_team: str, either 'A' or 'B', representing the team that actually won
    :param winning_closing_odds: float, consensus probability of winning team at match start time
    :param losing_closing_odds: float, consensus probability of losing team at match start time
    :return: float, the calculated edge
    """
    model_prediction_correct = (prediction_team == actual_team)
    # consensus_closing_odds: float, consensus probability of winning for the chosen team at match start time
    if model_prediction_correct:
        reward_punishment = 1
        consensus_closing_odds = winning_closing_odds
    else:
        reward_punishment = -1
        consensus_closing_odds = losing_closing_odds

    # draws have no edge. temporary
    if prediction_team == "Draw" or winning_closing_odds == losing_closing_odds:
        reward_punishment = 0
    
    edge = consensus_closing_odds - (1 / prediction_prob)
    return reward_punishment * edge, 1 if reward_punishment == 1 else 0

def compute_significance_score(num_miner_predictions: int, num_threshold_predictions: int, alpha: float) -> float:
    """
    Based on the number of predictions, calculate the statistical signifigance score.

    :param num_miner_predictions: int, the number of predictions made by the miner
    :param num_threshold_predictions: int, the number of predictions made by the threshold
    :param alpha: float, the sensitivity alpha
    :return: float, the calculated significance score
    """
    exponent = -alpha * (num_miner_predictions - num_threshold_predictions)
    denominator = 1 + math.exp(exponent)
    return 1 / denominator

def calculate_incentive_score(delta_t: int, clv: float, gamma: float, kappa: float, beta: float) -> float:
    """
    Calculate the incentive score considering time differential and closing line value.

    :param delta_t: int, the time differential in minutes
    :param clv: float, the miner's closing line value
    :param gamma: float, the time decay gamma
    :param kappa: float, the transition kappa
    :param beta: float, the extremis beta
    :return: float, the calculated incentive score
    """
    time_component = math.exp(-gamma * delta_t)
    clv_component = (1 - (2 * beta)) / (1 + math.exp(kappa * clv)) + beta
    return time_component + (1 - time_component) * clv_component

def calculate_sigma(pwmd: MatchPredictionWithMatchData) -> float:
    """
    Calculate the incentive sigma as a function of skill.

    np.sign(score of team chose - score of team not chosen) * (odds at close - 1 / probability of team chosen)
    
    :return: float, the calculated incentive score (sigma)
    """
    model_prediction_correct = 1 if (pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner()) else -1
    closing_odds = pwmd.get_actual_winner_odds()
    return model_prediction_correct * (closing_odds - (1 / pwmd.prediction.probability))

def calculate_clv(match_odds: List[Tuple[str, float, datetime]], pwmd: MatchPredictionWithMatchData):
    """
    Calculate the closing line value for this prediction.

    :param match_odds: List of tuples (matchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated)
    :param pwmd: MatchPredictionWithMatchData
    :return: float, closing line value, or None if unable to calculate
    """
    
    # Find the odds for the match at the time of the prediction
    prediction_odds = find_closest_odds(match_odds, pwmd.prediction.predictionDate, pwmd.prediction.probabilityChoice)
    
    if prediction_odds is None:
        bt.logging.debug(f"Unable to find suitable odds for matchId {pwmd.prediction.matchId} at prediction time. Skipping calculation of this prediction.")
        return None

    bt.logging.debug(f"  • Implied odds: {(1/pwmd.prediction.probability):.4f}")

    # clv is the distance between the odds at the time of prediction to the closing odds. Anything above 0 is derived value based on temporal drift of information
    bt.logging.debug(f"  • Closing Odds: {pwmd.get_actual_winner_odds()}")
    return prediction_odds - pwmd.get_actual_winner_odds()

def find_closest_odds(match_odds: List[Tuple[str, float, float, float, datetime]], prediction_time: datetime, probability_choice: str) -> Optional[float]:
    """
    Find the closest odds to the prediction time, ensuring the odds are before or at the prediction time.

    :param match_odds: List of tuples (matchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated)
    :param prediction_time: DateTime of the prediction
    :param probability_choice: ProbabilityChoice selection of the prediction
    :return: The closest odds value before or at the prediction time, or None if no suitable odds are found
    """
    closest_odds = None
    closest_odds_time = None

    # Ensure prediction_time is offset-aware
    if prediction_time.tzinfo is None:
        prediction_time = prediction_time.replace(tzinfo=pytz.UTC)

    for _, homeTeamOdds, awayTeamOdds, drawOdds, odds_datetime in match_odds:
        # Ensure odds_datetime is offset-aware
        if odds_datetime.tzinfo is None:
            odds_datetime = odds_datetime.replace(tzinfo=pytz.UTC)

        # Skip odds that are after the prediction time
        if odds_datetime > prediction_time:
            continue
        
        if probability_choice in [ProbabilityChoice.HOMETEAM, ProbabilityChoice.HOMETEAM.value]:
            odds = homeTeamOdds
        elif probability_choice in [ProbabilityChoice.AWAYTEAM, ProbabilityChoice.AWAYTEAM.value]:
            odds = awayTeamOdds
        else:
            odds = drawOdds

        if odds is None:
            continue

        # Update closest odds if this is the first valid odds or if it's closer to the prediction time
        if closest_odds_time is None or odds_datetime > closest_odds_time:
            closest_odds = odds
            closest_odds_time = odds_datetime

    if closest_odds is not None:
        time_diff = prediction_time - closest_odds_time
        time_diff_readable = str(time_diff)
        bt.logging.debug(f"  • Prediction Time: {prediction_time}")
        bt.logging.debug(f"  • Closest Odds Time: {closest_odds_time}")
        bt.logging.debug(f"  • Time Difference: {time_diff_readable}")
        bt.logging.debug(f"  • Prediction Time Odds: {closest_odds}")
        bt.logging.debug(f"  • Probability Choice: {probability_choice}")
    else:
        bt.logging.debug(f"No suitable odds found before or at the prediction time: {prediction_time}")

    return closest_odds

def apply_pareto(all_scores: List[float], all_uids: List[int], mu: float, alpha: int) -> List[float]:
    """
    Apply a Pareto distribution to the scores.

    :param all_scores: List of scores to apply the Pareto distribution to
    :param all_uids: List of UIDs corresponding to the scores
    :param mu: Minimum value for the Pareto distribution
    :param alpha: Shape parameter for the Pareto distribution
    :return: List of scores after applying the Pareto distribution
    """
    scores_array = np.array(all_scores)
    
    # Treat all non-positive scores as zero
    positive_mask = scores_array > 0
    positive_scores = scores_array[positive_mask]
    
    transformed_scores = np.zeros_like(scores_array, dtype=float)
    
    if len(positive_scores) > 0:
        # Transform positive scores
        range_transformed = (positive_scores - np.min(positive_scores)) + 1
        transformed_positive = mu * np.power(range_transformed, alpha)
        transformed_scores[positive_mask] = transformed_positive
    
    return transformed_scores
    
def check_and_apply_league_commitment_penalties(vali, all_scores: List[float], all_uids: List[int]) -> List[float]:
    """
    Check if all miners have at least one league commitment. If not, penalize to ensure that miners are committed to at least one league.

    :param vali: Validator, the validator object
    :param all_scores: List of scores to check and penalize
    :param all_uids: List of UIDs corresponding to the scores
    :return: List of scores after penalizing miners not committed to any active leagues
    """
    no_commitment_miner_uids = []
    
    for i, uid in enumerate(all_uids):
        with vali.accumulated_league_commitment_penalties_lock:
            # Initialize penalty for this UID if it doesn't exist
            if uid not in vali.accumulated_league_commitment_penalties:
                vali.accumulated_league_commitment_penalties[uid] = 0.0

            if uid not in vali.uids_to_leagues:
                vali.accumulated_league_commitment_penalties[uid] += NO_LEAGUE_COMMITMENT_PENALTY
                no_commitment_miner_uids.append(uid)
            else:
                miner_leagues = vali.uids_to_leagues[uid]
                # Remove any leagues that are not active
                active_leagues = [league for league in miner_leagues if league in vali.ACTIVE_LEAGUES]
                if len(active_leagues) == 0:
                    vali.accumulated_league_commitment_penalties[uid] += NO_LEAGUE_COMMITMENT_PENALTY
                    no_commitment_miner_uids.append(uid)
                else:
                    # Reset penalty if miner is now compliant
                    vali.accumulated_league_commitment_penalties[uid] = 0.0

            # Apply the accumulated penalty to the score
            all_scores[i] += vali.accumulated_league_commitment_penalties[uid]

    if no_commitment_miner_uids:
        bt.logging.info(
            f"Penalizing miners {no_commitment_miner_uids} that are not committed to any active leagues.\n"
            f"Accumulated penalties: {[vali.accumulated_league_commitment_penalties[uid] for uid in no_commitment_miner_uids]}"
        )
    
    return all_scores

def apply_no_prediction_response_penalties(vali, all_scores: List[float], all_uids: List[int]) -> List[float]:
    """
    Apply any penalties for miners that have not responded to prediction requests.

    :param vali: Validator, the validator object
    :param all_scores: List of scores to check and penalize
    :param all_uids: List of UIDs corresponding to the scores
    :return: List of scores after penalizing miners that have not responded to prediction requests
    """
    no_response_miner_uids = []
    no_response_miner_penalties = []
    
    for i, uid in enumerate(all_uids):
        with vali.accumulated_no_response_penalties_lock:
            # Initialize penalty for this UID if it doesn't exist
            if uid not in vali.accumulated_no_response_penalties:
                vali.accumulated_no_response_penalties[uid] = 0.0
            elif vali.accumulated_no_response_penalties[uid] < 0.0:
                # Apply the penalty to the score
                all_scores[i] += vali.accumulated_no_response_penalties[uid]
                no_response_miner_uids.append(uid)
                no_response_miner_penalties.append(vali.accumulated_no_response_penalties[uid])
                # Clear the penalty for this UID because it has been applied
                vali.accumulated_no_response_penalties[uid] = 0.0

    if no_response_miner_uids:
        bt.logging.info(
            f"Penalizing miners {no_response_miner_uids} that did not respond to prediction requests this run step.\n"
            f"Accumulated penalties: {no_response_miner_penalties}"
        )
    
    return all_scores

def calculate_incentives_and_update_scores(vali):
    """
    Calculate the incentives and update the scores for all miners with predictions.

    This function:
    1. Loops through every league
    2. For each league, loops through every miner
    4. Calculates incentives for miners committed to the league
    5. Updates scores for each miner for each league
    6. Updates the validator scores for each miner to set their weights
    7. Logs detailed results for each league and final scores

    :param vali: Validator, the validator object
    """
    storage = get_storage()
    all_uids = vali.metagraph.uids.tolist()
    
    # Initialize league_scores dictionary
    league_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}

    for league in vali.ACTIVE_LEAGUES:
        bt.logging.info(f"Processing league: {league.name}")
        league_table_data = []

        for index, uid in enumerate(all_uids):
            hotkey = vali.metagraph.hotkeys[uid]

            predictions_with_match_data = storage.get_miner_match_predictions(
                miner_hotkey=hotkey,
                miner_uid=uid,
                league=league,
                scored=True,
                batchSize=(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * 2)
            )

            if not predictions_with_match_data:
                continue  # No predictions for this league, keep score as 0

            # Calculate rho
            rho = compute_significance_score(
                num_miner_predictions=len(predictions_with_match_data),
                num_threshold_predictions=ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league],
                alpha=vali.SENSITIVITY_ALPHA
            )

            bt.logging.debug(f"Scoring predictions for miner {uid} in league {league.name}:")
            bt.logging.debug(f"  • Number of predictions: {len(predictions_with_match_data)}")
            bt.logging.debug(f"  • League rolling threshold count: {ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}")
            bt.logging.debug(f"  • Rho: {rho:.4f}")
            total_score = 0
            for pwmd in predictions_with_match_data:
                # Grab the match odds from local db
                match_odds = storage.get_match_odds(matchId=pwmd.prediction.matchId)
                if match_odds is None or len(match_odds) == 0:
                    bt.logging.debug(f"Odds were not found for matchId {pwmd.prediction.matchId}. Skipping calculation of this prediction.")
                    continue
                
                # Calculate our time delta expressed in minutes
                # Ensure prediction.matchDate is offset-aware
                if pwmd.prediction.matchDate.tzinfo is None:
                    match_date = pwmd.prediction.matchDate.replace(tzinfo=dt.timezone.utc)
                else:
                    match_date = pwmd.prediction.matchDate
                # Ensure prediction.predictionDate is offset-aware
                if pwmd.prediction.predictionDate.tzinfo is None:
                    prediction_date = pwmd.prediction.predictionDate.replace(tzinfo=dt.timezone.utc)
                else:
                    prediction_date = pwmd.prediction.predictionDate

                # Calculate time delta in minutes    
                delta_t = (MAX_PREDICTION_DAYS_THRESHOLD * 24 * 60) - ((match_date - prediction_date).total_seconds() / 60)
                bt.logging.debug(f"  • Time delta: {delta_t:.4f}")
                
                # Calculate closing line value
                clv = calculate_clv(match_odds, pwmd)
                if clv is None:
                    continue
                else:
                    bt.logging.debug(f"  • Closing line value: {clv:.4f}")

                v = calculate_incentive_score(
                    delta_t=delta_t,
                    clv=clv,
                    gamma=vali.GAMMA,
                    kappa=vali.TRANSITION_KAPPA, 
                    beta=vali.EXTREMIS_BETA,
                )
                bt.logging.debug(f"  • Incentive score (v): {v:.4f}")

                # Calculate sigma
                sigma = calculate_sigma(pwmd)
                bt.logging.debug(f"  • Sigma: {sigma:.4f}")

                # Apply sigma to v
                total_score += v * sigma
                bt.logging.debug(f"  • Total score: {total_score:.4f}")
                bt.logging.debug("-" * 50)

            final_score = rho * total_score
            league_scores[league][index] = final_score
            bt.logging.debug(f"  • Final score: {final_score:.4f}")
            bt.logging.debug("-" * 50)

            league_table_data.append([uid, final_score, len(predictions_with_match_data)])

        # Log league scores
        if league_table_data:
            bt.logging.info(f"\nScores for {league.name}:")
            bt.logging.info("\n" + tabulate(league_table_data, headers=['UID', 'Score', '# Predictions'], tablefmt='grid'))
        else:
            bt.logging.info(f"No non-zero scores for {league.name}")

    # Update all_scores with weighted sum of league scores for each miner
    bt.logging.info("************ Applying leagues scoring percentages to scores ************")
    for league, percentage in vali.LEAGUE_SCORING_PERCENTAGES.items():
        bt.logging.info(f"  • {league}: {percentage*100}%")
    bt.logging.info("*************************************************************")
    all_scores = [0.0] * len(all_uids)
    for i in range(len(all_uids)):
        all_scores[i] = sum(league_scores[league][i] * vali.LEAGUE_SCORING_PERCENTAGES[league] for league in vali.ACTIVE_LEAGUES)

    # Check and penalize miners that are not committed to any active leagues
    all_scores = check_and_apply_league_commitment_penalties(vali, all_scores, all_uids)
    # Apply penalties for miners that have not responded to prediction requests
    all_scores = apply_no_prediction_response_penalties(vali, all_scores, all_uids)
    
    # Apply Pareto to all scores
    bt.logging.info(f"Applying Pareto distribution (mu: {vali.PARETO_MU}, alpha: {vali.PARETO_ALPHA}) to scores...")
    final_scores = apply_pareto(all_scores, all_uids, vali.PARETO_MU, vali.PARETO_ALPHA)
    
    # Update our main self.scores, which scatters the scores
    update_miner_scores(
        vali, 
        torch.FloatTensor(final_scores).to(vali.device),
        all_uids
    )

    # Prepare final scores table
    final_scores_table = []
    for i, uid in enumerate(all_uids):
        final_scores_table.append([uid, all_scores[i], vali.scores[i]])

    # Log final scores
    bt.logging.info("\nFinal Weighted Scores:")
    bt.logging.info("\n" + tabulate(final_scores_table, headers=['UID', 'Pre-Pareto Score', 'Final Score'], tablefmt='grid'))

    # Log summary statistics
    non_zero_scores = [score for score in vali.scores if score > 0]
    if non_zero_scores:
        bt.logging.info(f"\nScore Summary:")
        bt.logging.info(f"Number of miners with non-zero scores: {len(non_zero_scores)}")
        bt.logging.info(f"Average non-zero score: {sum(non_zero_scores) / len(non_zero_scores):.6f}")
        bt.logging.info(f"Highest score: {max(vali.scores):.6f}")
        bt.logging.info(f"Lowest non-zero score: {min(non_zero_scores):.6f}")
    else:
        bt.logging.info("\nNo non-zero scores recorded.")

def update_miner_scores(vali, rewards: torch.FloatTensor, uids: List[int]):
    """Performs exponential moving average on the scores based on the rewards received from the miners."""

    # Check if rewards contains NaN values.
    if torch.isnan(rewards).any():
        bt.logging.warning(f"NaN values detected in rewards: {rewards}")
        # Replace any NaN values in rewards with 0.
        rewards = torch.nan_to_num(rewards, 0)

    # Check if `uids` is already a tensor and clone it to avoid the warning.
    if isinstance(uids, torch.Tensor):
        uids_tensor = uids.clone().detach()
    else:
        uids_tensor = torch.tensor(uids).to(vali.device)

    # Compute forward pass rewards, assumes uids are mutually exclusive.
    # shape: [ metagraph.n ]
    vali.scores: torch.FloatTensor = vali.scores.scatter(
        0, uids_tensor, rewards
    ).to(vali.device)
    bt.logging.debug(f"Scattered rewards. self.scores: {vali.scores}")
    bt.logging.debug(f"UIDs: {uids_tensor}")
