import math
import numpy as np
from scipy.stats import pareto
import datetime as dt
from datetime import datetime, timezone
import pytz
from typing import List, Dict, Tuple, Optional
from tabulate import tabulate
import random

import bittensor as bt
from storage.sqlite_validator_storage import get_storage
from vali_utils.prediction_integrity_controller import PredictionIntegrityController
from common.data import League, MatchPredictionWithMatchData, ProbabilityChoice
from common.constants import (
    MAX_PREDICTION_DAYS_THRESHOLD,
    NO_LEAGUE_COMMITMENT_PENALTY,
    NO_LEAGUE_COMMITMENT_GRACE_PERIOD,
    MINER_RELIABILITY_CUTOFF_IN_DAYS,
    MIN_MINER_RELIABILITY,
    MAX_GFILTER_FOR_WRONG_PREDICTION,
    MIN_GFILTER_FOR_CORRECT_UNDERDOG_PREDICTION,
    MIN_GFILTER_FOR_WRONG_UNDERDOG_PREDICTION,
    LEAGUES_ALLOWING_DRAWS,
    LEAGUE_MINIMUM_RHOS,
    MIN_EDGE_SCORE,
    MAX_MIN_EDGE_SCORE,
    ROI_BET_AMOUNT,
    ROI_INCR_PRED_COUNT_PERCENTAGE,
    MAX_INCR_ROI_DIFF_PERCENTAGE,
    ROI_SCORING_WEIGHT
)

def calculate_edge(prediction_team: str, prediction_prob: float, actual_team: str, closing_odds: float | None) -> Tuple[float, int]:
    """
    Calculate the edge for a prediction on a three-sided market.
    
    :param prediction_team: str, either 'A' or 'B' or 'Draw' representing the team chosen
    :param prediction_prob: float, weak learner's probability of winning for the chosen team at prediction time
    :param actual_team: str, either 'A' or 'B', representing the team that actually won
    :param closing_odds: float, consensus probability of outcome at match start time
    :return: Tuple[float, int], the calculated edge and a correctness indicator (1 if correct, 0 otherwise)
    """
    model_prediction_correct = (prediction_team == actual_team)
    reward_punishment = 1 if model_prediction_correct else -1

    # check if closing_odd is available
    if closing_odds is None or prediction_prob == 0:
        return 0.0, 0

    edge = closing_odds - (1 / prediction_prob)
    return reward_punishment * edge, 1 if model_prediction_correct else 0

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

def apply_gaussian_filter(pwmd: MatchPredictionWithMatchData) -> float:
    """
    Apply a Gaussian filter to the closing odds and prediction probability. 
    This filter is used to suppress the score when the prediction is far from the closing odds, simulating a more realistic prediction.

    :param pwmd: MatchPredictionWithMatchData
    :return: float, the calculated Gaussian filter
    """
    closing_odds = pwmd.get_closing_odds_for_predicted_outcome()

    t = 0.5 # Controls the spread/width of the Gaussian curve outside the plateau region. Larger t means slower decay in the exponential term
    t2 = 0.10 # Controls the spread/width of the Gaussian curve inside the plateau region. t2 is used on lay predictions
    a = 0.25 # Controls the height of the plateau boundary. More negative a means lower plateau boundary
    b = 0.3 # Controls how quickly the plateau boundary changes with odds. Larger b means faster exponential decay in plateau width
    c = 0.25 # Minimum plateau width/boundary
    pwr = 1.1 # Power to raise the difference between odds and 1/prob to in the exponential term

    # Plateau width calculation
    w = c - a * np.exp(-b * (closing_odds - 1))
    diff = abs(closing_odds - 1 / pwmd.prediction.probability)

    # If wp is less than implied wp, or wp odds is greater than implied odds, then use t2
    if closing_odds < 1 / pwmd.prediction.probability:
        t = t2

    # Plateaud curve with with uniform decay
    exp_component = 1.0 if diff <= w else np.exp(-(diff - w) / (t * np.power((closing_odds-1),pwr)))
    
    return exp_component

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

def calculate_clv(match_odds: List[Tuple[str, float, float, float, datetime]], pwmd: MatchPredictionWithMatchData, log_prediction: bool = False) -> Optional[float]:
    """
    Calculate the closing line value for this prediction.

    :param match_odds: List of tuples (matchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated)
    :param pwmd: MatchPredictionWithMatchData
    :return: float, closing line value, or None if unable to calculate
    """
    
    # Find the odds for the match at the time of the prediction
    prediction_odds = find_closest_odds(match_odds, pwmd.prediction.predictionDate, pwmd.prediction.probabilityChoice, log_prediction)
    
    if prediction_odds is None:
        return None

    if log_prediction:
        bt.logging.debug(f"      • Probability: {pwmd.prediction.probability}")
        bt.logging.debug(f"      • Implied odds: {(1/pwmd.prediction.probability):.4f}")

    # Get the closing odds for the predicted outcome
    closing_odds = pwmd.get_closing_odds_for_predicted_outcome()
    if closing_odds is None:
        bt.logging.debug(f"Closing odds were not found for matchId {pwmd.prediction.matchId}. Skipping calculation of this prediction.")
        return None

    if log_prediction:
        bt.logging.debug(f"      • Closing Odds: {closing_odds}")

     # clv is the distance between the odds at the time of prediction to the closing odds. Anything above 0 is derived value based on temporal drift of information
    return prediction_odds - closing_odds

def find_closest_odds(match_odds: List[Tuple[str, float, float, float, datetime]], prediction_time: datetime, probability_choice: str, log_prediction: bool) -> Optional[float]:
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
        elif probability_choice in [ProbabilityChoice.DRAW, ProbabilityChoice.DRAW.value]:
            odds = drawOdds
        else:
            continue  # invalid probability choice

        if odds is None:
            continue

        # Update closest odds if this is the first valid odds or if it's closer to the prediction time
        if closest_odds_time is None or odds_datetime > closest_odds_time:
            closest_odds = odds
            closest_odds_time = odds_datetime

    if closest_odds is not None:
        time_diff = prediction_time - closest_odds_time
        time_diff_readable = str(time_diff)
        if log_prediction:
            bt.logging.debug(f"  ---- Randomly logged prediction ----")
            bt.logging.debug(f"      • Prediction Time: {prediction_time}")
            bt.logging.debug(f"      • Closest Odds Time: {closest_odds_time}")
            bt.logging.debug(f"      • Time Difference: {time_diff_readable}")
            bt.logging.debug(f"      • Prediction Time Odds: {closest_odds}")
            bt.logging.debug(f"      • Probability Choice: {probability_choice}")

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

            # Skip miners with zero or negative scores
            if all_scores[i] <= 0.0:
                continue
            
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

def apply_no_prediction_response_penalties(
        metagraph: bt.metagraph, 
        league: League, 
        uids_to_last_leagues: Dict[int, List[League]], 
        uids_to_leagues_last_updated: Dict[int, dt.datetime],
        league_rhos: Dict[League, List[float]],
        league_scores: List[float], 
        all_uids: List[int]
    ) -> List[float]:
    """
    Apply any penalties for miners that have not responded to prediction requests.

    :param metagraph: The metagraph object
    :param league: The league to check for prediction response
    :param uids_to_last_leagues: Dictionary mapping UIDs to their last league commitments
    :param uids_to_leagues_last_updated: Dictionary mapping UIDs to the last updated time of their leagues
    :param league_rhos: Dictionary mapping leagues to their rho values
    :param league_scores: List of scores to check and penalize
    :param all_uids: List of UIDs corresponding to the scores
    :return: List of scores after penalizing miners that have not responded to prediction requests
    """

    league_miner_uids = []
    for uid in all_uids:
        if uid not in uids_to_last_leagues:
            continue
        if league in uids_to_last_leagues[uid] and uids_to_leagues_last_updated[uid] >= (datetime.now(timezone.utc) - dt.timedelta(seconds=NO_LEAGUE_COMMITMENT_GRACE_PERIOD)):
            league_miner_uids.append(uid)

    storage = get_storage()

    # Query the database for all eligible matches within the last MINER_RELIABILITY_CUTOFF_IN_DAYS days
    match_date_since = dt.datetime.now(timezone.utc) - dt.timedelta(days=MINER_RELIABILITY_CUTOFF_IN_DAYS)
    total_possible_predictions = storage.get_total_prediction_requests_count(matchDateSince=match_date_since, league=league)
    if total_possible_predictions and total_possible_predictions > 0:
        print(f"Checking miner prediction responses for league {league.name} (total possible predictions: {total_possible_predictions})")
        # Check all miners committed to this league
        for uid in league_miner_uids:
            # skip miners that haven't met min rho as they are 0 already anyway
            if league_rhos[league][uid] < LEAGUE_MINIMUM_RHOS[league]:
                print(f"Miner {uid} has rho {league_rhos[league][uid]:.4f} < {LEAGUE_MINIMUM_RHOS[league]}. Skipping.")
                continue
            miner_hotkey = metagraph.hotkeys[uid]
            get_miner_predictions_count = storage.get_total_match_predictions_by_miner(miner_hotkey, uid, match_date_since, league)
            # Check if miner predictions count is within MIN_MINER_RELIABILITY % of total possible predictions
            if get_miner_predictions_count < total_possible_predictions * MIN_MINER_RELIABILITY:
                # Miner has not been responding to prediction requests and will score 0
                league_scores[uid] = 0
                print(f"Miner {uid} has only fulfilled {get_miner_predictions_count} out of {total_possible_predictions} predictions ({round((get_miner_predictions_count/total_possible_predictions)*100)}%) in the last {MINER_RELIABILITY_CUTOFF_IN_DAYS} days. Setting score to 0.")
        print("-" * 75)
    
    return league_scores

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

    # Initialize Prediction Integrity Controller
    prediction_integrity_controller = PredictionIntegrityController()
    final_suspicious_miners = set()
    final_integrity_penalties = {}
    
    # Initialize league_scores dictionary
    league_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_edge_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_pred_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_pred_win_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    # Initialize overall ROI dictionaries
    league_roi_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_roi_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_roi_market_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    # Initialize incremental ROI dictionaries
    league_roi_incr_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_roi_incr_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_roi_incr_market_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    league_roi_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    # Initialize league_rhos dictionary
    league_rhos: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}

    for league in vali.ACTIVE_LEAGUES:
        bt.logging.info(f"Processing league: {league.name} (Rolling Pred Threshold: {vali.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}, Rho Sensitivity Alpha: {vali.LEAGUE_SENSITIVITY_ALPHAS[league]:.4f})")
        league_table_data = []
        predictions_for_integrity_analysis = []
        matches_without_odds = []

        # Get all miners committed to this league within the grace period
        league_miner_uids = []
        league_miner_data = []
        for uid in all_uids:
            if uid not in vali.uids_to_last_leagues:
                continue
            if league in vali.uids_to_last_leagues[uid] and vali.uids_to_leagues_last_updated[uid] >= (datetime.now(timezone.utc) - dt.timedelta(seconds=NO_LEAGUE_COMMITMENT_GRACE_PERIOD)):
                league_miner_uids.append(uid)
                league_miner_data.append((uid, vali.metagraph.hotkeys[uid]))
            elif league in vali.uids_to_leagues[uid] and vali.uids_to_leagues_last_updated[uid] < (datetime.now(timezone.utc) - dt.timedelta(seconds=NO_LEAGUE_COMMITMENT_GRACE_PERIOD)):
                bt.logging.info(f"Miner {uid} has not committed to league {league.name} within the grace period. Last updated: {vali.uids_to_leagues_last_updated[uid]}. Miner's predictions will not be considered.")

        # Single query for all miners in this league
        all_predictions_by_miner = storage.get_miner_match_predictions_by_batch(
            miner_data=league_miner_data,
            league=league,
            scored=True,
            batch_size=(vali.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * 2)
        )

        # Collect all unique match IDs for this league
        all_match_ids = set()
        for uid, predictions in all_predictions_by_miner.items():
            for pwmd in predictions:
                all_match_ids.add(pwmd.prediction.matchId)
        # Bulk load all odds for this league
        all_match_odds = storage.get_match_odds_by_batch(list(all_match_ids))

        for index, uid in enumerate(all_uids):
            total_score, rho = 0, 0
            predictions_with_match_data = []
            # Only process miners that are committed to the league
            if uid in league_miner_uids:
                hotkey = vali.metagraph.hotkeys[uid]
                # Get the predictions for this miner from the preloaded all_predictions_by_miner
                predictions_with_match_data = all_predictions_by_miner.get(uid, [])

                if not predictions_with_match_data:
                    continue  # No predictions for this league, keep score as 0

                # Add eligible predictions to predictions_for_integrity_analysis
                predictions_for_integrity_analysis.extend([p for p in predictions_with_match_data])

                # Calculate rho
                rho = compute_significance_score(
                    num_miner_predictions=len(predictions_with_match_data),
                    num_threshold_predictions=vali.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league],
                    alpha=vali.LEAGUE_SENSITIVITY_ALPHAS[league]
                )
                
                total_score = 0
                for pwmd in predictions_with_match_data:
                    #log_prediction = random.random() < 0.005
                    # turning off randomly logged prediction info for now.
                    log_prediction = False
                    if log_prediction:
                        bt.logging.debug(f"Randomly logged prediction for miner {uid} in league {league.name}:")
                        bt.logging.debug(f"  • Number of predictions: {len(predictions_with_match_data)}")
                        bt.logging.debug(f"  • League rolling threshold count: {vali.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}")
                        bt.logging.debug(f"  • League rho sensitivity alpha: {vali.LEAGUE_SENSITIVITY_ALPHAS[league]:.4f}")
                        bt.logging.debug(f"  • Rho: {rho:.4f}")
                    
                    # Grab the match odds from the preloaded all_match_odds
                    match_odds = all_match_odds.get(pwmd.prediction.matchId, [])
                    if match_odds is None or len(match_odds) == 0:
                        bt.logging.debug(f"Odds were not found for matchId {pwmd.prediction.matchId}. Skipping calculation of this prediction.")
                        continue

                    # Check if there is an odds anomaly and skip if so.
                    if pwmd.get_closing_odds_for_predicted_outcome() == 0:
                        bt.logging.debug(f"Closing odds were found to be 0 for matchId {pwmd.prediction.matchId}. homeTeamOdds: {pwmd.homeTeamOdds}, awayTeamOdds: {pwmd.awayTeamOdds}, drawOdds: {pwmd.drawOdds}")
                        bt.logging.debug(f"Skipping calculation of this prediction.")
                        continue

                    # Add to total prediction wins, if applicable
                    if pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner():
                        league_pred_win_counts[league][index] += 1

                    # Calculate ROI for the prediction
                    league_roi_counts[league][index] += 1
                    if pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner():
                        league_roi_payouts[league][index] += ROI_BET_AMOUNT * (pwmd.get_actual_winner_odds()-1)
                    else:
                        league_roi_payouts[league][index] -= ROI_BET_AMOUNT

                    # Calculate the market ROI for the prediction
                    if pwmd.actualHomeTeamScore > pwmd.actualAwayTeamScore and pwmd.homeTeamOdds < pwmd.awayTeamOdds:
                        league_roi_market_payouts[league][index] += ROI_BET_AMOUNT * (pwmd.get_actual_winner_odds()-1)
                    elif pwmd.actualAwayTeamScore > pwmd.actualHomeTeamScore and pwmd.awayTeamOdds < pwmd.homeTeamOdds:
                        league_roi_market_payouts[league][index] += ROI_BET_AMOUNT * (pwmd.get_actual_winner_odds()-1)
                    elif pwmd.actualHomeTeamScore == pwmd.actualAwayTeamScore and pwmd.drawOdds < pwmd.homeTeamOdds and pwmd.drawOdds < pwmd.awayTeamOdds:
                        league_roi_market_payouts[league][index] += ROI_BET_AMOUNT * (pwmd.get_actual_winner_odds()-1)
                    else:
                        league_roi_market_payouts[league][index] -= ROI_BET_AMOUNT

                    # Store the ROI incremental counts and payouts. Incremental ROI is a subset of the most recent prediction ROI
                    if league_roi_counts[league][index] == round(vali.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * ROI_INCR_PRED_COUNT_PERCENTAGE, 0):
                        league_roi_incr_counts[league][index] = league_roi_counts[league][index]
                        league_roi_incr_payouts[league][index] = league_roi_payouts[league][index]
                        league_roi_incr_market_payouts[league][index] = league_roi_market_payouts[league][index]

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
                    delta_t = min(MAX_PREDICTION_DAYS_THRESHOLD * 24 * 60, (match_date - prediction_date).total_seconds() / 60)
                    if log_prediction:
                        bt.logging.debug(f"      • Time delta: {delta_t:.4f}")
                    
                    # Calculate closing line value
                    clv = calculate_clv(match_odds, pwmd, log_prediction)
                    if clv is None:
                        if (match_date - prediction_date).total_seconds() / 60 <= 10:
                            t_interval = "T-10m"
                        elif (match_date - prediction_date).total_seconds() / 60 <= 240:
                            t_interval = "T-4h"
                        elif (match_date - prediction_date).total_seconds() / 60 <= 720:
                            t_interval = "T-12h"
                        elif (match_date - prediction_date).total_seconds() / 60 <= 1440:
                            t_interval = "T-24h"
                        # only add to matches_without_odds if the match and t-interval are not already in the list
                        if (pwmd.prediction.matchId, t_interval) not in matches_without_odds:
                            matches_without_odds.append((pwmd.prediction.matchId, t_interval))
                        continue
                    elif log_prediction:
                        bt.logging.debug(f"      • Closing line value: {clv:.4f}")

                    v = calculate_incentive_score(
                        delta_t=delta_t,
                        clv=clv,
                        gamma=vali.GAMMA,
                        kappa=vali.TRANSITION_KAPPA, 
                        beta=vali.EXTREMIS_BETA
                    )
                    if log_prediction:
                        bt.logging.debug(f"      • Incentive score (v): {v:.4f}")

                    # Calculate sigma, aka the closing edge
                    sigma, _ = calculate_edge(
                        prediction_team=pwmd.prediction.get_predicted_team(),
                        prediction_prob=pwmd.prediction.probability,
                        actual_team=pwmd.get_actual_winner(),
                        closing_odds=pwmd.get_closing_odds_for_predicted_outcome(),
                    )
                    if log_prediction:
                        bt.logging.debug(f"      • Sigma (aka Closing Edge): {sigma:.4f}")

                    # Calculate the Gaussian filter
                    gfilter = apply_gaussian_filter(pwmd)
                    if log_prediction:
                        bt.logging.debug(f"      • Gaussian filter: {gfilter:.4f}")
                    
                    # Set minimum value for Gaussian filter if the prediction was for the underdog
                    if pwmd.is_prediction_for_underdog(LEAGUES_ALLOWING_DRAWS) and pwmd.get_closing_odds_for_predicted_outcome() > (1 / pwmd.prediction.probability) and round(gfilter, 4) > 0:
                        prev_gfilter = gfilter
                        if pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner():
                            gfilter = max(MIN_GFILTER_FOR_CORRECT_UNDERDOG_PREDICTION, gfilter)
                        else:
                            gfilter =  max(MIN_GFILTER_FOR_WRONG_UNDERDOG_PREDICTION, gfilter)
                        if log_prediction:
                            bt.logging.debug(f"      • Underdog prediction detected. gfilter: {gfilter:.4f} | old_gfilter: {prev_gfilter:.4f}")
                    
                    # Apply a penalty if the prediction was incorrect and the Gaussian filter is less than 1 and greater than 0
                    elif pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner() and round(gfilter, 4) > 0 and round(gfilter, 4) < 1 and sigma < 0:
                        gfilter = max(MAX_GFILTER_FOR_WRONG_PREDICTION, gfilter)
                        if log_prediction:
                            bt.logging.debug(f"      • Penalty applied for wrong prediction. gfilter: {gfilter:.4f}")

                    # Apply sigma and G (gaussian filter) to v
                    total_score += v * sigma * gfilter
                    
                    if log_prediction:
                        bt.logging.debug(f"      • Total prediction score: {(v * sigma * gfilter):.4f}")
                        bt.logging.debug("-" * 50)

            # Store rho for this miner
            league_rhos[league][index] = rho
            # Calculate final edge score
            final_edge_score = rho * total_score
            league_scores[league][index] = final_edge_score
            league_edge_scores[league][index] = final_edge_score
            league_pred_counts[league][index] = len(predictions_with_match_data)
            # Calculate market ROI
            market_roi = league_roi_market_payouts[league][index] / (league_roi_counts[league][index] * ROI_BET_AMOUNT) if league_roi_counts[league][index] > 0 else 0.0
            # Calculate final ROI score
            roi = league_roi_payouts[league][index] / (league_roi_counts[league][index] * ROI_BET_AMOUNT) if league_roi_counts[league][index] > 0 else 0.0
            # Calculate the difference between the miner's ROI and the market ROI
            roi_diff = roi - market_roi
            # Base ROI score requires the miner is beating the market
            final_roi_score = round(rho * ((roi_diff if roi_diff>0 else 0)*100), 4)

            # If ROI is less than 0, but greater than market ROI, penalize the ROI score by distance from 0
            if roi < 0 and roi_diff > 0:
                bt.logging.info(f"Penalizing ROI score for miner {uid} in league {league.name} by {roi:.4f} ({final_roi_score * roi:.4f}): {final_roi_score:.4f} -> {final_roi_score + (final_roi_score * roi):.4f}")
                final_roi_score = final_roi_score + (final_roi_score * roi)
            
            roi_incr = roi
            market_roi_incr = market_roi
            # Calculate incremental ROI score for miner and market. Penalize if too similar.
            if league_roi_incr_counts[league][index] == round(vali.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * ROI_INCR_PRED_COUNT_PERCENTAGE, 0) and final_roi_score > 0:
                market_roi_incr = league_roi_incr_market_payouts[league][index] / (league_roi_incr_counts[league][index] * ROI_BET_AMOUNT) if league_roi_incr_counts[league][index] > 0 else 0.0
                roi_incr = league_roi_incr_payouts[league][index] / (league_roi_incr_counts[league][index] * ROI_BET_AMOUNT) if league_roi_incr_counts[league][index] > 0 else 0.0
                roi_incr_diff = roi_incr - market_roi_incr
                # if incremental ROI and incremental market ROI is within the difference threshold, calculate penalty
                if abs(roi_incr_diff) <= MAX_INCR_ROI_DIFF_PERCENTAGE:
                    # Exponential decay scaling
                    k = 30  # Decay constant; increase for steeper decay
                    # Scale the penalty factor to max at 0.99
                    penalty_factor = 0.99 * np.exp(-k * abs(roi_incr_diff))
                    adjustment_factor = 1 - penalty_factor
                    bt.logging.info(f"Incremental ROI score penalty for miner {uid} in league {league.name}: {roi_incr:.4f} vs {market_roi_incr} ({roi_incr_diff:.4f}), adj. factor {adjustment_factor:.4f}: {final_roi_score:.4f} -> {final_roi_score * adjustment_factor:.4f}")
                    final_roi_score *= adjustment_factor

            league_roi_scores[league][index] = final_roi_score

            # Only log scores for miners committed to the league
            if uid in league_miner_uids:
                league_table_data.append([
                    uid,
                    round(final_edge_score, 2),
                    round(final_roi_score, 2),
                    str(round(roi*100, 2)) + "%",
                    str(round(market_roi*100, 2)) + "%",
                    str(round(roi_diff*100, 2)) + "%", 
                    str(round(roi_incr*100, 2)) + "%", 
                    str(round(market_roi_incr*100, 2)) + "%",
                    len(predictions_with_match_data),
                    str(round(rho, 4)) + "" if rho > LEAGUE_MINIMUM_RHOS[league] else str(round(rho, 4)) + "*",
                ])

        # Log league scores
        if league_table_data:
            bt.logging.info(f"\nScores for {league.name}:")
            bt.logging.info("\n" + tabulate(league_table_data, headers=['UID', 'Edge Score', 'ROI Score', 'ROI', 'Mkt ROI', 'ROI Diff', 'ROI Incr', 'Mkt ROI Incr', '# Predictions', 'Rho'], tablefmt='grid'))
            bt.logging.info("* indicates rho is below minimum threshold and not eligible for rewards yet\n")
        else:
            bt.logging.info(f"No non-zero scores for {league.name}")

        # Normalize league scores and weight for Edge and ROI scores
        min_edge, max_edge = min(league_scores[league]), max(league_scores[league])
        # Avoid division by zero
        if max_edge - MIN_EDGE_SCORE == 0:
            normalized_edge = [0 for _ in league_scores[league]]
        else:
            normalized_edge = []
            for i, score in enumerate(league_scores[league]):
                rho = league_rhos[league][i]
                # Dynamic minimum edge score based on rho
                if rho <= 0.95:
                    min_edge_for_miner = MIN_EDGE_SCORE + (rho / 0.95) * (MAX_MIN_EDGE_SCORE - MIN_EDGE_SCORE)  # Linearly interpolate between min edge and maximum min edge (for older miners)
                else:
                    min_edge_for_miner = MAX_MIN_EDGE_SCORE
                
                # Compute normalized edge score
                if score > min_edge_for_miner:
                    normalized_value = (score - MIN_EDGE_SCORE) / (max_edge - MIN_EDGE_SCORE)
                else:
                    print(f"Miner {i} did not meet minimum edge score of {min_edge_for_miner:.2f} with score {score:.2f} and rho {rho:.2f}. Setting normalized score to 0.")
                    normalized_value = 0

                normalized_edge.append(normalized_value)
        
        # Normalize ROI scores
        min_roi, max_roi = min(league_roi_scores[league]), max(league_roi_scores[league])
        if max_roi - min_roi == 0:
            normalized_roi = [0 for score in league_roi_scores[league]]
        else:
            normalized_roi = [(score - min_roi) / (max_roi - min_roi) if (max_roi - min_roi) > 0 else 0 for score in league_roi_scores[league]]

        # Apply weights and combine and set to final league scores
        league_scores[league] = [
            ((1-ROI_SCORING_WEIGHT) * e + ROI_SCORING_WEIGHT * r) * rho
            if r > 0 and e > 0 and rho >= LEAGUE_MINIMUM_RHOS[league] else 0 # roi and edge must be > 0 and rho must be >= min rho
            for e, r, rho in zip(normalized_edge, normalized_roi, league_rhos[league])
        ]

        # Create top 10 scores table
        top_scores_table = []
        # Sort the final scores in descending order. We need to sort the uids as well so they match
        top_scores, top_uids = zip(*sorted(zip(league_scores[league], all_uids), reverse=True))
        for i in range(10):
            top_scores_table.append([i+1, top_uids[i], top_scores[i]])
        bt.logging.info(f"\nTop 10 Scores for {league.name}:")
        bt.logging.info("\n" + tabulate(top_scores_table, headers=['#', 'UID', 'Final Score'], tablefmt='grid'))

        if len(matches_without_odds) > 0:
            print(f"\n==============================================================================")
            print(f"Odds were not found for the following matches within {league.name}:")
            for mwo in matches_without_odds:
                print(f"{mwo[0]} - {mwo[1]}")
            print(f"==============================================================================")

        # Analyze league for integrity patterns
        suspicious_miners, penalties = prediction_integrity_controller.analyze_league(league, predictions_for_integrity_analysis, vali.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league])
        # Print league results
        print(f"\n==============================================================================")
        print(f"Total suspicious miners in {league.name}: {len(suspicious_miners)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(suspicious_miners))}")
        
        print(f"\nTotal miners to penalize in {league.name}: {len(penalties)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(penalties))}")
        print(f"==============================================================================")
        final_suspicious_miners.update(suspicious_miners)
        final_integrity_penalties.update(penalties)

    print(f"\nTotal miners to penalize across all leagues: {len(final_integrity_penalties)}")
    if len(final_integrity_penalties) > 0:
        # Print a table of miners to penalize
        penalized_table = []
        for uid, penalty_percentage in final_integrity_penalties.items():
            penalized_table.append([uid, f"{penalty_percentage*100:.2f}%"])
        print(tabulate(penalized_table, headers=['UID', 'Penalty %'], tablefmt='grid'))
    print(f"************************************************************************")

    # Update all_scores with weighted sum of league scores for each miner
    bt.logging.info("************ Normalizing and applying penalties and leagues scoring percentages to scores ************")
    for league, percentage in vali.LEAGUE_SCORING_PERCENTAGES.items():
        bt.logging.info(f"  • {league}: {percentage*100}%")
    bt.logging.info("*************************************************************")

    # Apply penalties for integrity miners and no prediction responses
    for league in vali.ACTIVE_LEAGUES:
        # Check and penalize miners that are not committed to any active leagues -- before normalization
        league_scores[league] = check_and_apply_league_commitment_penalties(vali, league_scores[league], all_uids)
        # Apply penalties for miners that have not responded to prediction requests -- before normalization
        league_scores[league] = apply_no_prediction_response_penalties(vali.metagraph, league, vali.uids_to_last_leagues, vali.uids_to_leagues_last_updated, league_rhos, league_scores[league], all_uids)

        # Apply the integrity penalty to the score -- before normalization
        for uid, penalty_percentage in final_integrity_penalties.items():
            score_after_penalty = league_scores[league][uid] * (1 - penalty_percentage)
            league_scores[league][uid] = score_after_penalty
    
    # Initialize total scores array
    all_scores = [0.0] * len(all_uids)

    # Step 1: Calculate total positive scores for each league
    league_totals = {league: 0.0 for league in vali.ACTIVE_LEAGUES}
    for league in vali.ACTIVE_LEAGUES:
        league_totals[league] = sum(score for score in league_scores[league] if score > 0)

    # Step 2: Scale scores within each league to match allocation percentage
    scaled_scores_per_league = {league: [0.0] * len(all_uids) for league in vali.ACTIVE_LEAGUES}
    for league in vali.ACTIVE_LEAGUES:
        total_league_score = league_totals[league]
        allocation = vali.LEAGUE_SCORING_PERCENTAGES[league] * 100  # Convert to percentage

        if total_league_score > 0:
            scaling_factor = allocation / total_league_score  # Factor to scale league scores
            scaled_scores_per_league[league] = [
                (score * scaling_factor if score > 0 else 0) for score in league_scores[league]
            ]

    # Step 3: Aggregate scaled scores across all leagues
    for i in range(len(all_uids)):
        all_scores[i] = sum(scaled_scores_per_league[league][i] for league in vali.ACTIVE_LEAGUES)

    # Step 4: Verify emissions allocation percentages
    league_emissions = {league: sum(scaled_scores_per_league[league]) for league in vali.ACTIVE_LEAGUES}
    total_emissions = sum(league_emissions.values())

    # Print league emissions and verify percentages
    bt.logging.debug("\nLeague Emissions and Allocations:")
    for league, emissions in league_emissions.items():
        percentage = (emissions / total_emissions) * 100 if total_emissions > 0 else 0
        bt.logging.debug(f"League: {league.name}, Total Emissions: {emissions:.4f}, Percentage: {percentage:.2f}%")

        # Cross-check to ensure percentages match expected allocations
        expected_percentage = vali.LEAGUE_SCORING_PERCENTAGES[league] * 100
        if not abs(percentage - expected_percentage) < 0.01:  # Allow a small tolerance
            bt.logging.debug(f"  Warning: Allocation mismatch for {league.name}! Expected: {expected_percentage:.2f}%, Actual: {percentage:.2f}%")
    
    # Apply Pareto to all scores
    bt.logging.info(f"Applying Pareto distribution (mu: {vali.PARETO_MU}, alpha: {vali.PARETO_ALPHA}) to scores...")
    final_scores = apply_pareto(all_scores, all_uids, vali.PARETO_MU, vali.PARETO_ALPHA)
    
    # Update our main self.scores, which scatters the scores
    update_miner_scores(
        vali,
        np.array(final_scores),
        all_uids
    )

    # Prepare final scores table
    final_scores_table = []
    for i, uid in enumerate(all_uids):
        miner_league = ""
        miner_league_last_updated = ""
        if uid in vali.uids_to_last_leagues and len(vali.uids_to_last_leagues[uid]) > 0:
            miner_league = vali.uids_to_last_leagues[uid][0].name
        if uid in vali.uids_to_leagues_last_updated and vali.uids_to_leagues_last_updated[uid] is not None:
            miner_league_last_updated = vali.uids_to_leagues_last_updated[uid].strftime("%Y-%m-%d %H:%M")

        final_scores_table.append([uid, miner_league, miner_league_last_updated, all_scores[i], vali.scores[i]])

    # Log final scores
    bt.logging.info("\nFinal Weighted Scores:")
    bt.logging.info("\n" + tabulate(final_scores_table, headers=['UID', 'League', 'Last Commitment', 'Pre-Pareto Score', 'Final Score'], tablefmt='grid'))

    # Create top 25 scores table
    top_scores_table = []
    # Sort the final scores in descending order. We need to sort the uids as well so they match
    top_scores, top_uids = zip(*sorted(zip(final_scores, all_uids), reverse=True))
    for i in range(200 if len(top_scores) > 200 else len(top_scores)):
        if top_scores[i] is not None and top_scores[i] > 0:
            miner_league = ""
            if top_uids[i] in vali.uids_to_last_leagues and len(vali.uids_to_last_leagues[top_uids[i]]) > 0:
                miner_league = vali.uids_to_last_leagues[top_uids[i]][0].name
            top_scores_table.append([i+1, top_uids[i], top_scores[i], miner_league])
    bt.logging.info("\nTop Miner Scores:")
    bt.logging.info("\n" + tabulate(top_scores_table, headers=['#', 'UID', 'Final Score', 'League'], tablefmt='grid'))

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
    
    return (league_scores, league_edge_scores, league_roi_scores, 
            league_roi_counts, league_roi_payouts, league_roi_market_payouts, 
            league_roi_incr_counts, league_roi_incr_payouts, league_roi_incr_market_payouts,
            league_pred_counts, league_pred_win_counts, all_scores)

def update_miner_scores(vali, rewards: np.ndarray, uids: List[int]):
    """Performs exponential moving average on the scores based on the rewards received from the miners."""

    # Check if rewards contains NaN values.
    if np.isnan(rewards).any():
        bt.logging.warning(f"NaN values detected in rewards: {rewards}")
        # Replace any NaN values in rewards with 0.
        rewards = np.nan_to_num(rewards, nan=0)

    # Ensure rewards is a numpy array.
    rewards = np.asarray(rewards)

    # Check if `uids` is already a numpy array and copy it to avoid the warning.
    if isinstance(uids, np.ndarray):
        uids_array = uids.copy()
    else:
        uids_array = np.array(uids)

    # Handle edge case: If either rewards or uids_array is empty.
    if rewards.size == 0 or uids_array.size == 0:
        bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
        bt.logging.warning(
            "Either rewards or uids_array is empty. No updates will be performed."
        )
        return

    # Check if sizes of rewards and uids_array match.
    if rewards.size != uids_array.size:
        raise ValueError(
            f"Shape mismatch: rewards array of shape {rewards.shape} "
            f"cannot be broadcast to uids array of shape {uids_array.shape}"
        )

    # Compute forward pass rewards, assumes uids are mutually exclusive.
    # shape: [ metagraph.n ]
    scattered_rewards: np.ndarray = np.zeros_like(vali.scores)
    scattered_rewards[uids_array] = rewards
    vali.scores = scattered_rewards
    bt.logging.debug(f"Scattered rewards. self.scores: {vali.scores}")
    bt.logging.debug(f"UIDs: {uids_array}")
