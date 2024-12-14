import os
import matplotlib.pyplot as plt
import numpy as np

import datetime as dt
from datetime import timezone
import random
from typing import Dict, List
from tabulate import tabulate

import bittensor
from storage.sqlite_validator_storage import get_storage

from common.data import Match, League, MatchPrediction, MatchPredictionWithMatchData
from common.constants import (
    ACTIVE_LEAGUES,
    MAX_PREDICTION_DAYS_THRESHOLD,
    ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE,
    LEAGUE_SENSITIVITY_ALPHAS,
    COPYCAT_PUNISHMENT_START_DATE,
    MAX_GFILTER_FOR_WRONG_PREDICTION,
    MIN_PROBABILITY,
    MIN_PROB_FOR_DRAWS,
    LEAGUES_ALLOWING_DRAWS,
    SENSITIVITY_ALPHA,
    GAMMA,
    TRANSITION_KAPPA,
    EXTREMIS_BETA,
    LEAGUE_SCORING_PERCENTAGES,
    COPYCAT_PENALTY_SCORE,
    PARETO_MU,
    PARETO_ALPHA
)

from vali_utils.copycat_controller import CopycatDetectionController

from vali_utils.scoring_utils import (
    calculate_edge,
    compute_significance_score,
    calculate_clv,
    calculate_incentive_score,
    apply_gaussian_filter,
    apply_pareto,
)


def calculate_incentives_and_update_scores():
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
    # Initialize our subtensor and metagraph
    NETWORK = None # "test" or None
    NETUID = 41
    if NETWORK == "test":
        NETUID = 172
    
    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    storage = get_storage()
    all_uids = metagraph.uids.tolist()
    #all_uids = metagraph.uids.tolist()[:10]

    # Initialize Copycat Detection Controller
    copycat_controller = CopycatDetectionController()
    final_suspicious_miners = set()
    final_copycat_penalties = set()
    final_exact_matches = set()
    
    # Initialize league_scores dictionary
    league_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_pred_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in ACTIVE_LEAGUES}
    # Use this to get payouts to calculate ROI
    league_roi_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}

    leagues_to_analyze = ACTIVE_LEAGUES
    #leagues_to_analyze = [League.NBA]

    for league in leagues_to_analyze:
        print(f"Processing league: {league.name} (Rolling Pred Threshold: {ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}, Rho Sensitivity Alpha: {LEAGUE_SENSITIVITY_ALPHAS[league]:.4f})")
        league_table_data = []
        predictions_for_copycat_analysis = []

        # Get all miners committed to this league within the grace period
        league_miner_uids = []
        for uid in all_uids:
            # Randomly select a subset of miners to commit to the league. UIDs 0-90 goto NBA. UIDs 91-180 goto NFL. UIDs 181-240 goto EPL.
            if league == League.NBA and uid < 90:
                league_miner_uids.append(uid)
            elif league == League.NFL and 90 <= uid < 180:
                league_miner_uids.append(uid)
            elif league == League.EPL and 180 <= uid < 240:
                league_miner_uids.append(uid)

        for index, uid in enumerate(league_miner_uids):
            hotkey = metagraph.hotkeys[uid]

            predictions_with_match_data = storage.get_miner_match_predictions(
                miner_hotkey=hotkey,
                miner_uid=uid,
                league=league,
                scored=True,
                batchSize=(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * 2)
            )

            if not predictions_with_match_data:
                continue  # No predictions for this league, keep score as 0

            # Add eligible predictions to predictions_for_copycat_analysis
            predictions_for_copycat_analysis.extend([p for p in predictions_with_match_data if p.prediction.predictionDate.replace(tzinfo=timezone.utc) >= COPYCAT_PUNISHMENT_START_DATE])

            # Calculate rho
            rho = compute_significance_score(
                num_miner_predictions=len(predictions_with_match_data),
                num_threshold_predictions=ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league],
                alpha=LEAGUE_SENSITIVITY_ALPHAS[league]
            )

            total_score = 0
            total_wrong_pred_pos_edge_penalty_preds = 0
            for pwmd in predictions_with_match_data:
                log_prediction = random.random() < 0.1
                log_prediction = False
                #if pwmd.prediction.probability <= MIN_PROBABILITY:
                    #log_prediction = True
                if log_prediction:
                    print(f"Randomly logged prediction for miner {uid} in league {league.name}:")
                    print(f"  • Number of predictions: {len(predictions_with_match_data)}")
                    print(f"  • League rolling threshold count: {ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}")
                    print(f"  • Rho: {rho:.4f}")

                # Grab the match odds from local db
                match_odds = storage.get_match_odds(matchId=pwmd.prediction.matchId)
                if match_odds is None or len(match_odds) == 0:
                    print(f"Odds were not found for matchId {pwmd.prediction.matchId}. Skipping calculation of this prediction.")
                    continue

                # if predictionDate within 10 minutes of matchDate, calculate roi payout
                if (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 10 and pwmd.prediction.predictionDate >= dt.datetime(2024, 12, 3, 0, 0, 0):
                    league_roi_counts[league][index] += 1
                    if pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner():
                        league_roi_payouts[league][index] += 100 * (pwmd.get_actual_winner_odds()-1)
                    else:
                        league_roi_payouts[league][index] -= 100

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
                    print(f"      • Time delta: {delta_t:.4f}")
                
                # Calculate closing line value
                clv = calculate_clv(match_odds, pwmd, log_prediction)
                if clv is None:
                    continue
                elif log_prediction:
                    print(f"      • Closing line value: {clv:.4f}")

                v = calculate_incentive_score(
                    delta_t=delta_t,
                    clv=clv,
                    gamma=GAMMA,
                    kappa=TRANSITION_KAPPA, 
                    beta=EXTREMIS_BETA
                )
                if log_prediction:
                    print(f"      • Incentive score (v): {v:.4f}")

                # Get sigma, aka the closing edge
                sigma = pwmd.prediction.closingEdge
                #sigma, correct_winner_score = calculate_edge(
                #    prediction_team=pwmd.prediction.get_predicted_team(),
                #    prediction_prob=pwmd.prediction.probability,
                #    actual_team=pwmd.get_actual_winner(),
                #    closing_odds=pwmd.get_closing_odds_for_predicted_outcome(),
                #)
                if log_prediction:
                    print(f"      • Sigma (aka Closing Edge): {sigma:.4f}")
                
                # Calculate the Gaussian filter
                gfilter = apply_gaussian_filter(pwmd)
                if log_prediction:
                    print(f"      • Gaussian filter: {gfilter:.4f}")
                
                # Zero out all lay predictions, that is if the prediction probability is less than MIN_PROBABILITY
                #if (pwmd.prediction.league in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.probability <= MIN_PROB_FOR_DRAWS) or pwmd.prediction.probability <= MIN_PROBABILITY:
                    #gfilter = 0
                # Apply a penalty if the prediction was incorrect and the Gaussian filter is less than 1 and greater than 0
                #elif pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner() and round(gfilter, 4) > 0 and gfilter < 1 and sigma < 0:
                
                # Apply a penalty if the prediction was incorrect and the Gaussian filter is less than 1 and greater than 0
                if  (
                        (pwmd.prediction.probability > MIN_PROBABILITY and league not in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner())
                        or 
                        (pwmd.prediction.probability < MIN_PROBABILITY and league not in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner())
                    ) or \
                    (
                        (pwmd.prediction.probability > MIN_PROB_FOR_DRAWS and league in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner())
                        or
                        (pwmd.prediction.probability < MIN_PROB_FOR_DRAWS and league in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner())
                    ) \
                    and round(gfilter, 4) > 0 and gfilter < 1 \
                    and sigma < 0:
                    
                    gfilter = max(MAX_GFILTER_FOR_WRONG_PREDICTION, gfilter)
                    if log_prediction:
                        print(f"      • Penalty applied for wrong prediction. gfilter: {gfilter:.4f}")
            
                # Apply sigma and G (gaussian filter) to v
                total_score += v * sigma * gfilter
            
                if log_prediction:
                    print(f"      • Total prediction score: {(v * sigma * gfilter):.4f}")
                    print("-" * 50)

            final_score = rho * total_score
            league_scores[league][index] = final_score
            league_pred_counts[league][index] = len(predictions_with_match_data)
            if log_prediction:
                print(f"  • Final score: {final_score:.4f}")
                print("-" * 50)

            total_lay_preds = len([
                pwmd for pwmd in predictions_with_match_data if pwmd.get_closing_odds_for_predicted_outcome() < 1 / pwmd.prediction.probability
                #if (pwmd.prediction.league in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.probability <= MIN_PROB_FOR_DRAWS) or pwmd.prediction.probability <= MIN_PROBABILITY
            ])
            roi = league_roi_payouts[league][index] / (league_roi_counts[league][index] * 100) if league_roi_counts[league][index] > 0 else 0.0
            league_table_data.append([uid, round(final_score, 2), len(predictions_with_match_data), round(final_score/len(predictions_with_match_data), 4), round(rho, 2), str(total_lay_preds) + "/" + str(len(predictions_with_match_data)), str(round(roi*100, 2)) + "%"])
            #league_table_data.append([uid, final_score, len(predictions_with_match_data), rho, total_wrong_pred_pos_edge_penalty_preds])

        # Log league scores
        if league_table_data:
            print(f"\nScores for {league.name}:")
            print(tabulate(league_table_data, headers=['UID', 'Score', '# Predictions', 'Avg Pred Score', 'Rho', '# Lay Predictions', 'ROI'], tablefmt='grid'))
            #print(tabulate(league_table_data, headers=['UID', 'Score', '# Predictions', 'Rho', '# Wrong Pred Pos Edge'], tablefmt='grid'))
        else:
            print(f"No non-zero scores for {league.name}")

        # Analyze league for copycat patterns
        earliest_match_date = min([p.prediction.matchDate for p in predictions_for_copycat_analysis], default=None)
        pred_matches = []
        if earliest_match_date is not None:
            pred_matches = storage.get_recently_completed_matches(earliest_match_date, league)
        ordered_matches = [(match.matchId, match.matchDate) for match in pred_matches]
        ordered_matches.sort(key=lambda x: x[1])  # Ensure chronological order
        #suspicious_miners, penalties, exact_matches = copycat_controller.analyze_league(league, predictions_for_copycat_analysis, ordered_matches)
        suspicious_miners, penalties, exact_matches = [], [], []
        # Print league results
        print(f"\n==============================================================================")
        print(f"Total suspicious miners in {league.name}: {len(suspicious_miners)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(suspicious_miners))}")

        print(f"\nTotal miners with exact matches in {league.name}: {len(exact_matches)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(exact_matches))}")
        
        print(f"\nTotal miners to penalize in {league.name}: {len(penalties)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(penalties))}")
        print(f"==============================================================================")
        final_suspicious_miners.update(suspicious_miners)
        final_copycat_penalties.update(penalties)
        final_exact_matches.update(exact_matches)

    # Update all_scores with weighted sum of league scores for each miner
    print("************ Applying leagues scoring percentages to scores ************")
    for league, percentage in LEAGUE_SCORING_PERCENTAGES.items():
        print(f"  • {league}: {percentage*100}%")
    print("*************************************************************")
    all_scores = [0.0] * len(all_uids)
    for i in range(len(all_uids)):
        all_scores[i] = sum(league_scores[league][i] * LEAGUE_SCORING_PERCENTAGES[league] for league in ACTIVE_LEAGUES)

    # Check and penalize miners that are not committed to any active leagues
    #all_scores = check_and_apply_league_commitment_penalties(vali, all_scores, all_uids)
    # Apply penalties for miners that have not responded to prediction requests
    #all_scores = apply_no_prediction_response_penalties(vali, all_scores, all_uids)

    # Log final copycat results
    print(f"********************* Copycat Controller Findings  *********************")
    # Get a unique list of coldkeys from metagraph
    coldkeys = list(set(metagraph.coldkeys))
    for coldkey in coldkeys:
        uids_for_coldkey = []
        for miner_uid in final_suspicious_miners:
            if metagraph.coldkeys[miner_uid] == coldkey:
                if miner_uid in final_copycat_penalties:
                    miner_uid = f"{miner_uid} 💀"
                uids_for_coldkey.append(str(miner_uid))
        if len(uids_for_coldkey) > 0:
            print(f"\nColdkey: {coldkey}")
            print(f"Suspicious Miners: {', '.join(str(m) for m in sorted(uids_for_coldkey))}")

    print(f"Total suspicious miners across all leagues: {len(final_suspicious_miners)}")
    if len(final_suspicious_miners) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_suspicious_miners))}")

    print(f"Total miners with exact matches across all leagues: {len(final_exact_matches)}")
    if len(final_exact_matches) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_exact_matches))}")

    print(f"Total miners to penalize across all leagues: {len(final_copycat_penalties)}")
    if len(final_copycat_penalties) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_copycat_penalties))}")
    print(f"************************************************************************")

    for uid in final_copycat_penalties:
        # Apply the penalty to the score
        all_scores[uid] = COPYCAT_PENALTY_SCORE
    
    # Apply Pareto to all scores
    print(f"Applying Pareto distribution (mu: {PARETO_MU}, alpha: {PARETO_ALPHA}) to scores...")
    final_scores = apply_pareto(all_scores, all_uids, PARETO_MU, PARETO_ALPHA)

    # Prepare final scores table
    final_scores_table = []
    for i, uid in enumerate(all_uids):
        final_scores_table.append([uid, all_scores[i], final_scores[i]])

    # Log final scores
    print("\nFinal Weighted Scores:")
    print(tabulate(final_scores_table, headers=['UID', 'Pre-Pareto Score', 'Final Score'], tablefmt='grid'))

    # Create top 50 scores table
    top_scores_table = []
    # Sort the final scores in descending order. We need to sort the uids as well so they match
    top_scores, top_uids = zip(*sorted(zip(final_scores, all_uids), reverse=True))
    for i in range(50):
        is_cabal = ""
        if metagraph.coldkeys[top_uids[i]] in []:
            is_cabal = "✔"
        top_scores_table.append([top_uids[i], top_scores[i], is_cabal, metagraph.coldkeys[top_uids[i]][:8]])
    print("\nTop 50 Miner Scores:")
    print(tabulate(top_scores_table, headers=['UID', 'Final Score', 'Cabal?', 'Coldkey'], tablefmt='grid'))

    # Log summary statistics
    non_zero_scores = [score for score in final_scores if score > 0]
    if non_zero_scores:
        print(f"\nScore Summary:")
        print(f"Number of miners with non-zero scores: {len(non_zero_scores)}")
        print(f"Average non-zero score: {sum(non_zero_scores) / len(non_zero_scores):.6f}")
        print(f"Highest score: {max(final_scores):.6f}")
        print(f"Lowest non-zero score: {min(non_zero_scores):.6f}")
    else:
        print("\nNo non-zero scores recorded.")

    # Generate graph of Pre-Pareto vs Final Pareto Scores
    graph_results(all_uids, all_scores, final_scores)


def graph_results(all_uids, all_scores, final_scores):
    """
    Graphs the Pre-Pareto and Final Pareto scores with smaller, transparent dots and improved aesthetics.

    :param all_uids: List of unique identifiers for the miners
    :param all_scores: List of Pre-Pareto scores
    :param final_scores: List of Final Pareto scores
    """
    # Sort the miners based on Pre-Pareto Scores
    sorted_indices = np.argsort(all_scores)
    sorted_pre_pareto_scores = np.array(all_scores)[sorted_indices]
    sorted_final_pareto_scores = np.array(final_scores)[sorted_indices]

    # X-axis for the miners (from 0 to number of miners)
    x_axis = np.arange(len(all_uids))

    # Create the output directory if it doesn't exist
    output_dir = "tests/imgs"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the graph with smaller, transparent dots
    plt.figure(figsize=(12, 6))
    #plt.scatter(x_axis, sorted_pre_pareto_scores, label="Pre-Pareto Score", color='blue', s=10, alpha=0.6)
    plt.scatter(x_axis, sorted_final_pareto_scores, label="Final Pareto Score", color='orange', s=10, alpha=0.6)
    plt.xlabel("Miners (sorted by Pre-Pareto Score)")
    plt.ylabel("Scores")
    plt.title("Final Scores")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the graph as an image
    output_path = os.path.join(output_dir, "pareto_scores.png")
    plt.savefig(output_path)
    plt.close()

    return output_path


if "__main__" == __name__:
    calculate_incentives_and_update_scores()