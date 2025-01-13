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
    MIN_GFILTER_FOR_UNDERDOG_PREDICTION,
    ROI_BET_AMOUNT,
    MIN_ROI_THRESHOLD,
    ROI_SCORING_WEIGHT,
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
    league_roi_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}

    leagues_to_analyze = ACTIVE_LEAGUES
    #leagues_to_analyze = [League.NBA]

    uids_to_last_leagues = {}
    uids_to_leagues_last_updated = {}

    # Recreating the dictionary with UID to League mapping from the provided data

    uid_to_best_league = {
        0: 'NBA', 4: 'NBA', 5: 'NBA', 6: 'NBA', 7: 'NBA',
        8: 'NBA', 9: 'NBA', 10: 'NBA', 11: 'NFL', 12: 'NFL',
        13: 'NFL', 14: 'NBA', 15: 'NBA', 16: 'NFL', 17: 'NFL',
        18: 'NBA', 19: 'NBA', 20: 'NFL', 21: 'NFL', 22: 'NBA',
        23: 'NFL', 24: 'NBA', 25: 'NBA', 26: 'NFL', 27: 'NBA',
        28: 'NBA', 29: 'NBA', 30: 'NFL', 31: 'NFL', 32: 'NBA',
        33: 'NBA', 34: 'NFL', 35: 'NBA', 36: 'NFL', 37: 'NBA',
        38: 'NFL', 40: 'NBA', 41: 'NBA', 42: 'NFL', 43: 'NBA',
        44: 'NBA', 45: 'NBA', 46: 'NBA', 47: 'NFL', 48: 'NBA',
        49: 'NBA', 50: 'NBA', 51: 'NFL', 52: 'NBA', 53: 'NBA',
        54: 'NFL', 56: 'NBA', 57: 'EPL', 58: 'NBA', 59: 'NBA',
        60: 'NBA', 61: 'NFL', 62: 'NFL', 63: 'NBA', 64: 'NFL',
        65: 'NBA', 66: 'NBA', 67: 'NFL', 68: 'NFL', 69: 'NBA',
        70: 'NBA', 71: 'NBA', 72: 'NBA', 73: 'NFL', 74: 'NBA',
        75: 'NFL', 76: 'NBA', 77: 'NBA', 78: 'NBA', 79: 'NBA',
        80: 'NFL', 81: 'NBA', 82: 'NBA', 83: 'NBA', 84: 'NBA',
        85: 'NBA', 86: 'NBA', 87: 'NBA', 88: 'NBA', 89: 'NBA',
        90: 'NFL', 91: 'NFL', 92: 'NBA', 93: 'NFL', 94: 'NBA',
        95: 'NBA', 97: 'NFL', 98: 'NBA', 99: 'NFL', 100: 'NFL',
        101: 'NFL', 102: 'NFL', 103: 'NFL', 104: 'NFL', 105: 'NBA',
        107: 'NBA', 108: 'NFL', 109: 'NBA', 110: 'NFL', 111: 'NFL',
        112: 'NFL', 113: 'NFL', 114: 'NFL', 115: 'NFL', 116: 'NBA',
        117: 'NFL', 118: 'NFL', 119: 'NFL', 120: 'NFL', 121: 'NFL',
        122: 'NBA', 123: 'NFL', 124: 'NBA', 125: 'NBA', 126: 'NFL',
        127: 'NBA', 128: 'NFL', 129: 'NFL', 130: 'NFL', 131: 'NFL',
        132: 'NBA', 133: 'NFL', 134: 'NFL', 135: 'NBA', 136: 'NBA',
        137: 'NFL', 138: 'NFL', 139: 'NBA', 140: 'NBA', 141: 'NBA',
        143: 'NBA', 144: 'NFL', 145: 'NBA', 146: 'NBA', 147: 'NBA',
        148: 'NBA', 149: 'NFL', 150: 'NFL', 151: 'NBA', 152: 'NFL',
        153: 'NFL', 154: 'NBA', 155: 'NFL', 156: 'NFL', 157: 'NFL',
        158: 'NFL', 159: 'NFL', 160: 'NFL', 161: 'NBA', 162: 'NBA',
        163: 'NFL', 164: 'NFL', 165: 'NFL', 166: 'NBA', 167: 'NBA',
        168: 'NFL', 169: 'NBA', 170: 'NFL', 171: 'NFL', 172: 'NBA',
        173: 'NFL', 174: 'NBA', 175: 'NFL', 176: 'NBA', 177: 'NBA',
        178: 'NFL', 179: 'NBA', 181: 'NFL', 183: 'NBA', 184: 'NBA',
        186: 'NBA', 187: 'NBA', 188: 'NBA', 189: 'NBA', 191: 'NFL',
        192: 'NFL', 193: 'NFL', 194: 'NFL', 195: 'NFL', 196: 'NBA',
        197: 'NFL', 198: 'NBA', 199: 'NBA', 202: 'NBA', 204: 'NBA',
        205: 'NFL', 206: 'NFL', 207: 'NFL', 208: 'NBA', 210: 'NFL',
        211: 'NFL', 213: 'NFL', 214: 'NBA', 215: 'NFL', 216: 'NBA',
        218: 'NFL', 219: 'NBA', 220: 'NFL', 222: 'NFL', 224: 'NFL',
        225: 'NFL', 227: 'NBA', 229: 'NFL', 230: 'NBA', 232: 'NFL',
        234: 'NBA', 235: 'NBA', 236: 'NBA', 237: 'NFL', 238: 'NBA',
        240: 'NFL', 241: 'NFL', 243: 'NBA', 245: 'NBA', 247: 'NBA',
        249: 'NBA', 250: 'NFL', 251: 'NBA', 252: 'NFL', 253: 'NFL',
        254: 'NBA', 255: 'NFL'
    }

    for league in leagues_to_analyze:
        print(f"Processing league: {league.name} (Rolling Pred Threshold: {ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}, Rho Sensitivity Alpha: {LEAGUE_SENSITIVITY_ALPHAS[league]:.4f})")
        league_table_data = []
        predictions_for_copycat_analysis = []
        matches_without_odds = []

        # Get all miners committed to this league within the grace period
        league_miner_uids = []
        for uid in all_uids:
            if uid not in uid_to_best_league:
                continue
            if uid_to_best_league[uid] == league or uid_to_best_league[uid] == league.name:
                league_miner_uids.append(uid)
                uids_to_last_leagues[uid] = [league]
                uids_to_leagues_last_updated[uid] = dt.datetime.now()

        for index, uid in enumerate(all_uids):
            total_score, rho = 0, 0
            predictions_with_match_data = []
            # Only process miners that are committed to the league
            if uid in league_miner_uids:
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
                    #if pwmd.prediction.minerId == 254 and (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 10:
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

                    if pwmd.get_closing_odds_for_predicted_outcome() == 0:
                        print(f"Closing odds were found to be 0 for matchId {pwmd.prediction.matchId}. homeTeamOdds: {pwmd.homeTeamOdds}, awayTeamOdds: {pwmd.awayTeamOdds}, drawOdds: {pwmd.drawOdds}")
                        print(f"Skipping calculation of this prediction.")
                        continue

                    # Calculate ROI for the prediction
                    league_roi_counts[league][index] += 1
                    if pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner():
                        league_roi_payouts[league][index] += ROI_BET_AMOUNT * (pwmd.get_actual_winner_odds()-1)
                    else:
                        league_roi_payouts[league][index] -= ROI_BET_AMOUNT

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
                        if (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 10:
                            t_interval = "T-10m"
                        elif (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 240:
                            t_interval = "T-4h"
                        elif (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 720:
                            t_interval = "T-12h"
                        elif (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 1440:
                            t_interval = "T-24h"
                        # only add to matches_without_odds if the match and t-interval are not already in the list
                        if (pwmd.prediction.matchId, t_interval) not in matches_without_odds:
                            matches_without_odds.append((pwmd.prediction.matchId, t_interval))
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
                        print(f"      • MatchId: {pwmd.prediction.matchId}")
                        print(f"      • {pwmd.prediction.awayTeamName} ({pwmd.awayTeamOdds:.4f}) at {pwmd.prediction.homeTeamName} ({pwmd.homeTeamOdds:.4f})")
                        print(f"      • Prediction:  {pwmd.prediction.get_predicted_team()} ({pwmd.prediction.probability:.4f} | {1/pwmd.prediction.probability:.4f})")
                        print(f"      • Actual Winner: {pwmd.get_actual_winner()}")
                        if round(pwmd.get_closing_odds_for_predicted_outcome(),4) < round((1 / pwmd.prediction.probability), 4):
                            print(f"      • Lay Prediction")
                        print(f"      • Closing Odds: {pwmd.get_closing_odds_for_predicted_outcome():.4f}")
                        print(f"      • Sigma (aka Closing Edge): {sigma:.4f}")
                    
                    # Calculate the Gaussian filter
                    gfilter = apply_gaussian_filter(pwmd)
                    if log_prediction:
                        print(f"      • Gaussian filter: {gfilter:.4f}")
                    
                    # Set minimum value for Gaussian filter if the prediction was for the underdog
                    if pwmd.is_prediction_for_underdog(LEAGUES_ALLOWING_DRAWS) and pwmd.get_closing_odds_for_predicted_outcome() > (1 / pwmd.prediction.probability):
                        
                        prev_gfilter = gfilter
                        gfilter =  max(MIN_GFILTER_FOR_UNDERDOG_PREDICTION, gfilter)

                        # Only log the T-10m interval
                        """
                        if (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 10:
                            print(f"      • Underdog prediction detected. gfilter: {gfilter:.4f} | old_gfilter: {prev_gfilter:.4f}")
                            print(f"      -- Prediction: {pwmd.prediction.get_predicted_team()} ({pwmd.prediction.probability:.4f} | {1/pwmd.prediction.probability:.4f})")
                            print(f"      -- Actual Winner: {pwmd.get_actual_winner()}")
                            print(f"      -- Odds: {pwmd.prediction.homeTeamName} ({pwmd.homeTeamOdds:.4f}) vs {pwmd.prediction.awayTeamName} ({pwmd.awayTeamOdds:.4f})")
                            if pwmd.drawOdds > 0:
                                print(f"      -- Draw Odds: {pwmd.drawOdds:.4f}")
                            print(f"      -- Closing Odds: {pwmd.get_closing_odds_for_predicted_outcome():.4f}")
                            print(f"      -- Raw Edge: {sigma:.4f}")
                        """
                            
                        if log_prediction:
                            print(f"      • Underdog prediction detected. gfilter: {gfilter:.4f} | old_gfilter: {prev_gfilter:.4f}")
                    
                    # Apply a penalty if the prediction was incorrect and the Gaussian filter is less than 1 and greater than 0
                    elif pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner() and round(gfilter, 4) > 0 and round(gfilter, 4) < 1 and sigma < 0:
                        gfilter = max(MAX_GFILTER_FOR_WRONG_PREDICTION, gfilter)
                        if log_prediction:
                            print(f"      • Penalty applied for wrong prediction. gfilter: {gfilter:.4f}")
                
                    # Apply sigma and G (gaussian filter) to v
                    total_score += v * sigma * gfilter
                
                    if log_prediction:
                        print(f"      • Total prediction score: {(v * sigma * gfilter):.4f}")
                        print("-" * 50)

            final_edge_score = rho * total_score
            league_scores[league][index] = final_edge_score
            league_pred_counts[league][index] = len(predictions_with_match_data)
            total_lay_preds = len([
                pwmd for pwmd in predictions_with_match_data if round(pwmd.get_closing_odds_for_predicted_outcome(),2) < round((1 / pwmd.prediction.probability), 2)
            ])
            total_underdog_preds = len([
                pwmd for pwmd in predictions_with_match_data if pwmd.is_prediction_for_underdog(LEAGUES_ALLOWING_DRAWS)
            ])
            avg_pred_score = final_edge_score / len(predictions_with_match_data) if len(predictions_with_match_data) > 0 else 0.0
            raw_edge = 0
            for pwmd in predictions_with_match_data:
                raw_edge += pwmd.prediction.closingEdge
            roi = league_roi_payouts[league][index] / (league_roi_counts[league][index] * ROI_BET_AMOUNT) if league_roi_counts[league][index] > 0 else 0.0
            raw_roi = roi
            if roi < MIN_ROI_THRESHOLD:
                roi = 0.0
            final_roi_score = round(rho * ((roi if roi>0 else 0)*100), 2)
            league_roi_scores[league][index] = final_roi_score
            # Only log scores for miners committed to the league
            if uid in league_miner_uids:
                league_table_data.append([uid, round(final_edge_score, 2), round(final_roi_score, 2), str(round(raw_roi*100, 2)) + "%", len(predictions_with_match_data), round(avg_pred_score, 4), round(rho, 2), str(total_lay_preds) + "/" + str(len(predictions_with_match_data)), total_underdog_preds, round(raw_edge, 4)])

        # Log league scores
        if league_table_data:
            print(f"\nScores for {league.name}:")
            print(tabulate(league_table_data, headers=['UID', 'Score', 'ROI Score', 'ROI', '# Predictions', 'Avg Pred Score', 'Rho', '# Lay Preds', '# Underdog Preds', 'Raw Edge'], tablefmt='grid'))
        else:
            print(f"No non-zero scores for {league.name}")

        # Normalize league scores and weight for Edge and ROI scores
        # Normalize edge scores
        min_edge, max_edge = min(league_scores[league]), max(league_scores[league])
        normalized_edge = [(score - min_edge) / (max_edge - min_edge) if score > 0 else 0 for score in league_scores[league]]
        # Normalize ROI scores
        min_roi, max_roi = min(league_roi_scores[league]), max(league_roi_scores[league])
        normalized_roi = [(score - min_roi) / (max_roi - min_roi) if (max_roi - min_roi) > 0 else 0 for score in league_roi_scores[league]]
        
        # Apply weights and combine and set to final league scores
        league_scores[league] = [
            (1-ROI_SCORING_WEIGHT) * e + ROI_SCORING_WEIGHT * r
            if r > 0 else 0
            for e, r in zip(normalized_edge, normalized_roi)
        ]

        # Create top 10 scores table
        top_scores_table = []
        # Sort the final scores in descending order. We need to sort the uids as well so they match
        top_scores, top_uids = zip(*sorted(zip(league_scores[league], all_uids), reverse=True))
        for i in range(10):
            top_scores_table.append([i+1, top_uids[i], top_scores[i]])
        print(f"\nTop 10 Scores for {league.name}:")
        print(tabulate(top_scores_table, headers=['#', 'UID', 'Final Score'], tablefmt='grid'))

        if len(matches_without_odds) > 0:
            print(f"\n==============================================================================")
            print(f"Odds were not found for the following matches within {league.name}:")
            for mwo in matches_without_odds:
                print(f"{mwo[0]} - {mwo[1]}")
            print(f"==============================================================================")

        # Analyze league for copycat patterns
        earliest_match_date = min([p.prediction.matchDate for p in predictions_for_copycat_analysis], default=None)
        pred_matches = []
        if earliest_match_date is not None:
            pred_matches = storage.get_recently_completed_matches(earliest_match_date, league)
        ordered_matches = [(match.matchId, match.matchDate) for match in pred_matches]
        ordered_matches.sort(key=lambda x: x[1])  # Ensure chronological order
        suspicious_miners, penalties, exact_matches = copycat_controller.analyze_league(league, predictions_for_copycat_analysis, ordered_matches)
        #suspicious_miners, penalties, exact_matches = [], [], []
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

    print(f"\nTotal suspicious miners across all leagues: {len(final_suspicious_miners)}")
    if len(final_suspicious_miners) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_suspicious_miners))}")

    print(f"\nTotal miners with exact matches across all leagues: {len(final_exact_matches)}")
    if len(final_exact_matches) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_exact_matches))}")

    print(f"\nTotal miners to penalize across all leagues: {len(final_copycat_penalties)}")
    if len(final_copycat_penalties) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_copycat_penalties))}")
    print(f"************************************************************************")

    # Update all_scores with weighted sum of league scores for each miner
    print("************ Normalizing and applying penalties and leagues scoring percentages to scores ************")
    for league, percentage in LEAGUE_SCORING_PERCENTAGES.items():
        print(f"  • {league}: {percentage*100}%")
    print("*************************************************************")

    # Apply penalties for copycat miners and no prediction responses
    for league in ACTIVE_LEAGUES:
        # Check and penalize miners that are not committed to any active leagues -- before normalization
        #league_scores[league] = check_and_apply_league_commitment_penalties(vali, league_scores[league], all_uids)
        # Apply penalties for miners that have not responded to prediction requests -- before normalization
        #league_scores[league] = apply_no_prediction_response_penalties(vali, league_scores[league], all_uids)

        # Apply the copycat penalty to the score -- before normalization
        for uid in final_copycat_penalties:
            league_scores[league][uid] = COPYCAT_PENALTY_SCORE
    
    # Initialize total scores array
    all_scores = [0.0] * len(all_uids)

    # Step 1: Calculate total positive scores for each league
    league_totals = {league: 0.0 for league in ACTIVE_LEAGUES}
    for league in ACTIVE_LEAGUES:
        league_totals[league] = sum(score for score in league_scores[league] if score > 0)

    # Step 2: Scale scores within each league to match allocation percentage
    scaled_scores_per_league = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    for league in ACTIVE_LEAGUES:
        total_league_score = league_totals[league]
        allocation = LEAGUE_SCORING_PERCENTAGES[league] * 100  # Convert to percentage

        if total_league_score > 0:
            scaling_factor = allocation / total_league_score  # Factor to scale league scores
            scaled_scores_per_league[league] = [
                (score * scaling_factor if score > 0 else 0) for score in league_scores[league]
            ]

    # Step 3: Aggregate scaled scores across all leagues
    for i in range(len(all_uids)):
        all_scores[i] = sum(scaled_scores_per_league[league][i] for league in ACTIVE_LEAGUES)

    # Step 4: Verify emissions allocation percentages
    league_emissions = {league: sum(scaled_scores_per_league[league]) for league in ACTIVE_LEAGUES}
    total_emissions = sum(league_emissions.values())

    # Print league emissions and verify percentages
    print("\nLeague Emissions and Allocations:")
    for league, emissions in league_emissions.items():
        percentage = (emissions / total_emissions) * 100 if total_emissions > 0 else 0
        print(f"League: {league.name}, Total Emissions: {emissions:.4f}, Percentage: {percentage:.2f}%")

        # Cross-check to ensure percentages match expected allocations
        expected_percentage = LEAGUE_SCORING_PERCENTAGES[league] * 100
        if not abs(percentage - expected_percentage) < 0.01:  # Allow a small tolerance
            print(f"  Warning: Allocation mismatch for {league.name}! Expected: {expected_percentage:.2f}%, Actual: {percentage:.2f}%")
    
    # Apply Pareto to all scores
    print(f"Applying Pareto distribution (mu: {PARETO_MU}, alpha: {PARETO_ALPHA}) to scores...")
    final_scores = apply_pareto(all_scores, all_uids, PARETO_MU, PARETO_ALPHA)

    # Prepare final scores table
    final_scores_table = []
    for i, uid in enumerate(all_uids):
        miner_league = ""
        miner_league_last_updated = ""
        if uid in uids_to_last_leagues and len(uids_to_last_leagues[uid]) > 0:
            miner_league = uids_to_last_leagues[uid][0].name
        if uid in uids_to_leagues_last_updated and uids_to_leagues_last_updated[uid] is not None:
            miner_league_last_updated = uids_to_leagues_last_updated[uid].strftime("%Y-%m-%d %H:%M")

        final_scores_table.append([uid, miner_league, miner_league_last_updated, all_scores[i], final_scores[i]])

    # Log final scores
    print("\nFinal Weighted Scores:")
    print(tabulate(final_scores_table, headers=['UID', 'League', 'Last Commitment', 'Pre-Pareto Score', 'Final Score'], tablefmt='grid'))

    # Create top 50 scores table
    top_scores_table = []
    # Sort the final scores in descending order. We need to sort the uids as well so they match
    top_scores, top_uids = zip(*sorted(zip(final_scores, all_uids), reverse=True))
    for i in range(200):
        miner_league = ""
        if top_uids[i] in uids_to_last_leagues and len(uids_to_last_leagues[top_uids[i]]) > 0:
            miner_league = uids_to_last_leagues[top_uids[i]][0].name
        is_cabal = ""
        if metagraph.coldkeys[top_uids[i]] in []:
            is_cabal = "✔"
        top_scores_table.append([i+1, top_uids[i], top_scores[i], miner_league, is_cabal, metagraph.coldkeys[top_uids[i]][:8]])
    print("\nTop Miner Scores:")
    print(tabulate(top_scores_table, headers=['#', 'UID', 'Final Score', 'League', 'Cabal?', 'Coldkey'], tablefmt='grid'))

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

    cabal_uids = ""
    for uid in all_uids:
        if metagraph.coldkeys[uid] in []:
            cabal_uids += f"{uid},"
    print(f"\nCabal UIDs: {cabal_uids}")


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