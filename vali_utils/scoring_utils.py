import math
from scipy import stats
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from tabulate import tabulate

import bittensor as bt
from storage.sqlite_validator_storage import get_storage
from vali_utils import utils
from common.data import MatchPrediction, League, MatchPredictionWithMatchData, ProbabilityChoice
from common.constants import (
    MAX_PREDICTION_DAYS_THRESHOLD,
    ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE
)

# ALPHA controls how many predictions are needed to start getting rewards.
SENSITIVITY_ALPHA = 0.025
# GAMMA controls the time decay of CLV.
GAMMA = 0.00125
# KAPPA controls the sharpness of the interchange between CLV and Time.
TRANSITION_KAPPA = 35
# BETA controls the ranges that the CLV component lives within.
EXTREMIS_BETA = 0.25

def calculate_edge(prediction_team: str, prediction_prob: float, actual_team: str, consensus_closing_odds: float) -> Tuple[float, int]:
    """
    Calculate the edge for a prediction on a two-sided market.
    
    :param prediction_team: str, either 'A' or 'B', representing the team chosen
    :param prediction_prob: float, weak learner's probability of winning for the chosen team at prediction time
    :param actual_team: str, either 'A' or 'B', representing the team that actually won
    :param consensus_closing_odds: float, consensus probability of winning for the chosen team at match start time
    :return: float, the calculated edge
    """
    model_prediction_correct = (prediction_team == actual_team)
    reward_punishment = 1 if model_prediction_correct else -1
    
    edge = (1 / prediction_prob) - consensus_closing_odds
    return reward_punishment * edge, 1 if reward_punishment == 1 else 0


def compute_significance_score(num_miner_predictions: int, num_threshold_predictions: int, alpha: float=SENSITIVITY_ALPHA) -> float:
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

def calculate_incentive_score(delta_t: int, clv: float, gamma: float=GAMMA, kappa: float=TRANSITION_KAPPA, beta: float=EXTREMIS_BETA) -> float:
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
    clv_component = (1 - (2 * beta)) / (1 + math.exp(-kappa * clv)) + beta
    incentive_score = time_component + (1 - time_component) * clv_component
    return incentive_score

def calculate_sigma(predictions: List[MatchPredictionWithMatchData]) -> float:
    """
    Calculate the incentive sigma as a function of skill.
    
    This function computes the average closing edge per game and multiplies it
    by the number of games predicted.

    :param predictions: List[MatchPredictionWithMatchData], a list of prediction with match data objects
    :return: float, the calculated incentive score (sigma)
    """
    num_predictions = len(predictions)
    total_edge = sum(p.prediction.closingEdge for p in predictions)
    average_edge = total_edge / num_predictions
    sigma = average_edge * num_predictions
    return sigma

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
        bt.logging.error(f"Unable to find suitable odds for matchId {pwmd.prediction.matchId} at prediction time. Skipping calculation of this prediction.")
        return None

    # clv is the distance between the odds at the time of prediction to the closing odds. Anything above 0 is derived value based on temporal drift of information
    return prediction_odds - pwmd.get_actual_winner_odds()

def find_closest_odds(match_odds: List[Tuple[str, float, float, float, datetime]], prediction_time: datetime, probability_choice: ProbabilityChoice) -> float:
    """
    Find the closest odds to the prediction time.

    :param match_odds: List of tuples (matchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated)
    :param prediction_time: DateTime of the prediction
    :param probability_choice: ProbabilityChoice selection of the prediction
    :return: The closest odds value, or None if no suitable odds are found
    """
    closest_odds = None
    smallest_time_diff = float('inf')
    closest_odds_time = None

    for _, homeTeamOdds, awayTeamOdds, drawOdds, odds_datetime in match_odds:
        if probability_choice == ProbabilityChoice.HOMETEAM:
            odds = homeTeamOdds
        elif probability_choice == ProbabilityChoice.AWAYTEAM:
            odds = awayTeamOdds
        else:
            odds = drawOdds

        if odds is None:
            continue

        time_diff = abs((odds_datetime - prediction_time).total_seconds())

        if time_diff < smallest_time_diff:
            smallest_time_diff = time_diff
            closest_odds = odds
            closest_odds_time = odds_datetime

    if closest_odds is not None:
        time_diff_readable = str(timedelta(seconds=int(smallest_time_diff)))
        bt.logging.info(f"Prediction Time: {prediction_time}")
        bt.logging.info(f"Closest Odds Time: {closest_odds_time}")
        bt.logging.info(f"Time Difference: {time_diff_readable}")
        bt.logging.info(f"Chosen Odds: {closest_odds}")
        bt.logging.info(f"Probability Choice: {probability_choice}")
        bt.logging.info("-" * 50)  # Separator for readability

    return closest_odds

def apply_pareto(all_scores: List[float], all_uids: List[int]) -> List[float]:
    """
    Apply a Pareto distribution to the scores.

    :param all_scores: List of scores to apply the Pareto distribution to
    :param all_uids: List of UIDs corresponding to the scores
    :return: List of scores after applying the Pareto distribution
    """
    # Separate non-zero scores and their indices
    non_zero_scores = []
    non_zero_indices = []
    for i, score in enumerate(all_scores):
        if score > 0:
            non_zero_scores.append(score)
            non_zero_indices.append(i)

    pareto_scores = []
    # Apply Pareto distribution to the non-zero scores
    if non_zero_scores:
        # Fit a Pareto distribution to the non-zero scores
        xmin = min(non_zero_scores)
        alpha = 1 + len(non_zero_scores) / sum(np.log(np.array(non_zero_scores) / xmin))
        pareto = stats.pareto(alpha, scale=xmin)
        
        # Transform non-zero scores using the Pareto CDF
        transformed_scores = pareto.cdf(non_zero_scores)
        
        # Normalize the transformed scores
        total_transformed = sum(transformed_scores)
        normalized_scores = [score / total_transformed for score in transformed_scores]

        # Update pareto_scores with the Pareto-distributed scores
        pareto_scores = [0.0] * len(all_uids)
        for i, score in zip(non_zero_indices, normalized_scores):
            pareto_scores[i] = score
    else:
        bt.logging.warning("No non-zero scores to apply Pareto distribution. All scores remain zero.")
        pareto_scores = [0.0] * len(all_uids)

    return pareto_scores

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
                alpha=SENSITIVITY_ALPHA
            )
            # Calculate sigma
            sigma = calculate_sigma(predictions_with_match_data)

            total_score = 0
            for pwmd in predictions_with_match_data:
                # Grab the match odds from local db
                match_odds = storage.get_match_odds(matchId=pwmd.prediction.matchId)
                if match_odds is None or len(match_odds) == 0:
                    bt.logging.debug(f"Odds were not found for matchId {pwmd.prediction.matchId} in db. Attempting sync with API.")
                    # Try and grab match odds from API
                    match_odds = utils.sync_match_odds_data(vali.match_odds_endpoint, pwmd.prediction.matchId)
                    if match_odds is None:
                        bt.logging.error(f"Odds were not found for matchId {pwmd.prediction.matchId}. Skipping calculationg of this prediction.")
                        continue
                
                # Calculate our time delta expressed in minutes
                delta_t = (MAX_PREDICTION_DAYS_THRESHOLD * 24 * 60) - ((pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60)
                # Calculate closing line value
                clv = calculate_clv(match_odds, pwmd)

                v = calculate_incentive_score(
                    delta_t=delta_t,
                    clv=clv,
                    kappa=TRANSITION_KAPPA, 
                    beta=EXTREMIS_BETA,
                )
                total_score += v * sigma

            final_score = rho * total_score
            league_scores[league][index] = final_score

            if final_score > 0:
                league_table_data.append([uid, final_score])

        # Log league scores
        if league_table_data:
            bt.logging.info(f"\nScores for {league.name}:")
            bt.logging.info("\n" + tabulate(league_table_data, headers=['UID', 'Score'], tablefmt='grid'))
        else:
            bt.logging.info(f"No non-zero scores for {league.name}")

    # Update all_scores with weighted sum of league scores for each miner
    all_scores = [0.0] * len(all_uids)
    for i in range(len(all_uids)):
        all_scores[i] = sum(league_scores[league][i] * vali.LEAGUE_SCORING_PERCENTAGES[league] for league in vali.ACTIVE_LEAGUES)

    # Apply Pareto to all scores
    #final_scores = apply_pareto(all_scores, all_uids)
    final_scores = all_scores
    
    # Update our main self.scores, which scatters the scores
    update_miner_scores(
        vali, 
        torch.FloatTensor(final_scores).to(vali.device),
        all_uids
    )

    # Prepare final scores table
    final_scores_table = []
    for i, uid in enumerate(all_uids):
        final_scores_table.append([uid, vali.scores[i]])

    # Log final scores
    bt.logging.info("\nFinal Weighted Scores:")
    bt.logging.info("\n" + tabulate(final_scores_table, headers=['UID', 'Final Score'], tablefmt='grid'))

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


if __name__ == "__main__":
    # Test cases
    """
    prediction_teams = ['A', 'A', 'A', 'B']
    prediction_probs = [0.6, 0.7, 0.65, 0.57]
    actual_teams = ['A', 'B', 'A', 'B']
    consensus_odds = [0.55, 0.45, 0.6, 0.55]

    for pt, pp, at, odds in zip(prediction_teams, prediction_probs, actual_teams, consensus_odds):
        edge = calculate_edge(prediction_team=pt, prediction_prob=pp, actual_team=at, consensus_closing_odds=odds)
        print(f"Edge: {edge:.4f}")
    """
    
    
    """
    score = calculate_incentive_score(n=50)
    print(f"Incentive score: {score:.4f}")

    score = calculate_advanced_incentive(delta_t=2.0, z=0.7, clv=0.8)
    print(f"Advanced incentive score: {score:.4f}")

    predictions = [
        {'delta_t': 2.0, 'z': 0.7, 'clv': 0.8, 'sigma': 1.2},
        {'delta_t': 1.5, 'z': 0.8, 'clv': 0.7, 'sigma': 1.1},
        {'delta_t': 1.0, 'z': 0.9, 'clv': 0.6, 'sigma': 1.0},
        {'delta_t': 0.5, 'z': 1.0, 'clv': 0.5, 'sigma': 0.9},
        {'delta_t': 0.0, 'z': 1.1, 'clv': 0.4, 'sigma': 0.8},
    ]
    final_score = calculate_combined_incentive_score(n=len(predictions), predictions=predictions)
    print(f"Final combined incentive score: {final_score:.4f}")
    """
