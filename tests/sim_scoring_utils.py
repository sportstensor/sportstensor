import os
import matplotlib.pyplot as plt
import itertools
import numpy as np

import datetime as dt
from datetime import timezone
import time
import random
from typing import Dict, List, Optional, Any
from tabulate import tabulate

import bittensor
from storage.sqlite_validator_storage import get_storage

from common.data import League
from common.constants import (
    ACTIVE_LEAGUES,
    MAX_PREDICTION_DAYS_THRESHOLD,
    ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE,
    LEAGUE_SENSITIVITY_ALPHAS,
    MAX_GFILTER_FOR_WRONG_PREDICTION,
    MIN_GFILTER_FOR_CORRECT_UNDERDOG_PREDICTION,
    MIN_GFILTER_FOR_WRONG_UNDERDOG_PREDICTION,
    ROI_BET_AMOUNT,
    ROI_INCR_PRED_COUNT_PERCENTAGE,
    MAX_INCR_ROI_DIFF_PERCENTAGE,
    LEAGUE_MINIMUM_RHOS,
    MIN_EDGE_SCORE,
    MAX_MIN_EDGE_SCORE,
    ROI_SCORING_WEIGHT,
    LEAGUES_ALLOWING_DRAWS,
    SENSITIVITY_ALPHA,
    GAMMA,
    TRANSITION_KAPPA,
    EXTREMIS_BETA,
    LEAGUE_SCORING_PERCENTAGES,
    PARETO_MU,
    PARETO_ALPHA
)

from vali_utils.prediction_integrity_controller import PredictionIntegrityController

from vali_utils.scoring_utils import (
    calculate_edge,
    compute_significance_score,
    calculate_clv,
    calculate_incentive_score,
    apply_gaussian_filter,
    apply_pareto,
    apply_no_prediction_response_penalties
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
    # Start time tracking
    start_time = dt.datetime.now()
    print(f"Started scoring at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    # Initialize Prediction Integrity Controller
    prediction_integrity_controller = PredictionIntegrityController()
    final_suspicious_miners = set()
    final_integrity_penalties = {}
    
    # Initialize league_scores dictionary
    league_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_pred_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in ACTIVE_LEAGUES}
    # Use this to get payouts to calculate ROI
    league_roi_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_market_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_incr_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_incr_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_incr_market_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_incr_max_possible_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    # Initialize league_rhos dictionary
    league_rhos: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}

    leagues_to_analyze = ACTIVE_LEAGUES
    #leagues_to_analyze = [League.NBA]

    uids_to_last_leagues = {}
    uids_to_leagues_last_updated = {}

    uids_to_league = {}
    if len(uids_to_league) == 0:
        print("No UID to League mapping found. Using fallback hardcoded mapping.")
        # Recreating the dictionary with UID to League mapping from the provided data
        uids_to_league = {
            0: 'MLB',
            3: 'MLB',
            4: 'MLB',
            5: 'MLB',
            6: 'MLS',
            8: 'MLS',
            9: 'MLB',
            10: 'MLB',
            11: 'MLB',
            12: 'MLB',
            13: 'NBA',
            14: 'MLB',
            15: 'MLB',
            16: 'MLB',
            17: 'MLB',
            18: 'MLB',
            19: 'MLB',
            20: 'NBA',
            21: 'MLB',
            22: 'MLB',
            23: 'MLB',
            24: 'MLB',
            25: 'MLB',
            26: 'MLB',
            27: 'MLB',
            28: 'MLB',
            29: 'MLB',
            30: 'MLB',
            31: 'MLB',
            32: 'MLB',
            33: 'MLB',
            34: 'MLB',
            35: 'NBA',
            36: 'MLS',
            37: 'MLB',
            38: 'MLB',
            39: 'MLB',
            40: 'MLB',
            41: 'MLB',
            42: 'MLB',
            43: 'MLB',
            44: 'MLB',
            45: 'MLB',
            46: 'MLB',
            47: 'MLB',
            48: 'MLB',
            49: 'MLB',
            50: 'MLB',
            51: 'MLB',
            52: 'MLB',
            53: 'MLB',
            54: 'MLB',
            56: 'MLB',
            57: 'MLB',
            58: 'MLB',
            59: 'MLS',
            60: 'MLB',
            61: 'MLB',
            62: 'MLB',
            63: 'MLB',
            64: 'MLB',
            65: 'MLB',
            66: 'MLB',
            67: 'MLS',
            68: 'MLB',
            69: 'MLB',
            70: 'MLB',
            71: 'MLS',
            72: 'MLS',
            73: 'MLB',
            74: 'MLB',
            75: 'MLB',
            76: 'MLB',
            77: 'MLS',
            78: 'MLS',
            79: 'MLB',
            80: 'MLB',
            81: 'MLB',
            82: 'MLB',
            83: 'MLB',
            84: 'MLB',
            85: 'MLB',
            86: 'MLB',
            87: 'MLB',
            88: 'MLS',
            89: 'MLB',
            90: 'MLB',
            91: 'MLB',
            92: 'MLB',
            93: 'MLB',
            94: 'MLS',
            95: 'MLB',
            97: 'MLB',
            98: 'MLB',
            99: 'MLB',
            100: 'MLB',
            101: 'MLB',
            102: 'MLB',
            103: 'MLB',
            104: 'MLB',
            105: 'MLB',
            106: 'NBA',
            107: 'MLB',
            108: 'MLB',
            109: 'MLB',
            110: 'MLB',
            111: 'MLB',
            112: 'MLB',
            113: 'MLB',
            114: 'MLB',
            115: 'MLB',
            116: 'MLB',
            117: 'MLB',
            118: 'MLB',
            119: 'MLB',
            120: 'MLS',
            121: 'MLB',
            122: 'MLB',
            123: 'NBA',
            124: 'MLB',
            125: 'MLB',
            126: 'MLB',
            127: 'MLB',
            128: 'MLB',
            129: 'MLB',
            130: 'MLB',
            131: 'MLB',
            132: 'MLB',
            133: 'MLS',
            134: 'MLS',
            135: 'MLB',
            136: 'MLB',
            137: 'MLB',
            138: 'MLB',
            139: 'MLB',
            140: 'NBA',
            141: 'MLS',
            142: 'MLB',
            143: 'MLB',
            144: 'MLS',
            145: 'MLB',
            146: 'MLB',
            147: 'MLB',
            148: 'MLS',
            149: 'MLB',
            150: 'MLB',
            151: 'MLB',
            152: 'MLB',
            153: 'MLB',
            154: 'MLB',
            155: 'MLB',
            156: 'MLB',
            157: 'MLB',
            158: 'MLB',
            159: 'MLB',
            160: 'MLB',
            161: 'MLB',
            162: 'MLB',
            163: 'MLB',
            164: 'MLB',
            165: 'MLS',
            166: 'MLB',
            167: 'MLB',
            168: 'MLB',
            169: 'MLB',
            170: 'MLB',
            171: 'MLB',
            172: 'MLB',
            173: 'MLB',
            174: 'MLS',
            175: 'MLB',
            176: 'MLB',
            177: 'MLB',
            178: 'MLS',
            179: 'MLB',
            180: 'MLS',
            181: 'MLB',
            182: 'MLS',
            183: 'MLB',
            184: 'MLS',
            185: 'MLS',
            186: 'MLB',
            187: 'MLB',
            188: 'MLB',
            189: 'MLB',
            190: 'MLB',
            191: 'MLB',
            192: 'MLB',
            193: 'MLS',
            194: 'MLB',
            195: 'MLB',
            196: 'MLB',
            197: 'MLB',
            198: 'MLB',
            199: 'MLB',
            200: 'MLB',
            202: 'MLB',
            203: 'MLB',
            204: 'MLB',
            205: 'MLB',
            206: 'MLS',
            207: 'MLB',
            208: 'MLB',
            209: 'MLB',
            211: 'MLB',
            212: 'MLB',
            213: 'MLB',
            214: 'MLB',
            215: 'MLS',
            216: 'MLB',
            217: 'MLB',
            218: 'MLB',
            219: 'MLB',
            220: 'MLB',
            221: 'MLB',
            222: 'MLB',
            223: 'MLB',
            224: 'MLB',
            225: 'MLB',
            226: 'MLB',
            227: 'MLS',
            229: 'MLB',
            230: 'MLB',
            231: 'MLB',
            232: 'MLB',
            234: 'MLS',
            236: 'MLB',
            237: 'MLB',
            238: 'MLB',
            240: 'MLB',
            241: 'MLB',
            242: 'MLS',
            243: 'MLB',
            244: 'MLB',
            245: 'MLB',
            246: 'MLB',
            247: 'NBA',
            248: 'MLB',
            249: 'MLB',
            250: 'MLS',
            251: 'NBA',
            252: 'MLB',
            253: 'MLB',
            254: 'MLB',
            255: 'MLB',
        }

    for league in leagues_to_analyze:
        print(f"Processing league: {league.name} (Rolling Pred Threshold: {ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}, Rho Sensitivity Alpha: {LEAGUE_SENSITIVITY_ALPHAS[league]:.4f}, Rho Min: {LEAGUE_MINIMUM_RHOS[league]:.4f})")
        league_table_data = []
        predictions_for_integrity_analysis = []
        matches_without_odds = []

        # Get all miners committed to this league within the grace period
        league_miner_uids = []
        league_miner_data = []
        for uid in all_uids:
            if uid not in uids_to_league:
                continue
            if uids_to_league[uid] == league or uids_to_league[uid] == league.name:
                league_miner_uids.append(uid)
                league_miner_data.append((uid, metagraph.hotkeys[uid]))
                uids_to_last_leagues[uid] = [league]
                uids_to_leagues_last_updated[uid] = dt.datetime.now(timezone.utc)

        # Single query for all miners in this league
        all_predictions_by_miner = storage.get_miner_match_predictions_by_batch(
            miner_data=league_miner_data,
            league=league,
            scored=True,
            batch_size=(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * 2)
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
            edge_mismatch_count, edge_mismatch_oppo_count = 0, 0
            predictions_with_match_data = []
            # Only process miners that are committed to the league
            if uid in league_miner_uids:
                hotkey = metagraph.hotkeys[uid]
                # Get the predictions for this miner from the preloaded all_predictions_by_miner
                predictions_with_match_data = all_predictions_by_miner.get(uid, [])

                if not predictions_with_match_data:
                    continue  # No predictions for this league, keep score as 0

                # Add eligible predictions to predictions_for_integrity_analysis
                predictions_for_integrity_analysis.extend([p for p in predictions_with_match_data])

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
                    
                    # Grab the match odds from the preloaded all_match_odds
                    match_odds = all_match_odds.get(pwmd.prediction.matchId, [])
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
                    if league_roi_counts[league][index] < round(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * ROI_INCR_PRED_COUNT_PERCENTAGE, 0):
                        league_roi_incr_max_possible_payouts[league][index] += ROI_BET_AMOUNT * (pwmd.get_actual_winner_odds()-1)
                    if league_roi_counts[league][index] == round(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * ROI_INCR_PRED_COUNT_PERCENTAGE, 0):
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
                    sigma_preScored = pwmd.prediction.closingEdge
                    sigma, _ = calculate_edge(
                        prediction_team=pwmd.prediction.get_predicted_team(),
                        prediction_prob=pwmd.prediction.probability,
                        actual_team=pwmd.get_actual_winner(),
                        closing_odds=pwmd.get_closing_odds_for_predicted_outcome(),
                    )
                    """
                    if sigma_preScored != sigma:
                        edge_mismatch_count += 1
                        if (sigma_preScored < 0 and sigma > 0) or (sigma_preScored > 0 and sigma < 0):
                            edge_mismatch_oppo_count += 1
                            print(f"prediction_team: {pwmd.prediction.get_predicted_team()}, prediction_prob: {pwmd.prediction.probability}, actual_team: {pwmd.get_actual_winner()}, closing_odds: {pwmd.get_closing_odds_for_predicted_outcome()}, sigma: {sigma}, sigma_preScored: {sigma_preScored}")
                        #print(f"Sigma and sigma2 are not equal for matchId {pwmd.prediction.matchId} ({pwmd.prediction.homeTeamName} vs {pwmd.prediction.awayTeamName}, {pwmd.prediction.matchDate}). prediction.closingEdge: {sigma}, calculatedEdge: {sigma2}")
                    """                   
                    
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
                    if pwmd.is_prediction_for_underdog(LEAGUES_ALLOWING_DRAWS) and pwmd.get_closing_odds_for_predicted_outcome() > (1 / pwmd.prediction.probability) and round(gfilter, 4) > 0:
                        
                        prev_gfilter = gfilter
                        if pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner():
                            gfilter = max(MIN_GFILTER_FOR_CORRECT_UNDERDOG_PREDICTION, gfilter)
                        else:
                            gfilter =  max(MIN_GFILTER_FOR_WRONG_UNDERDOG_PREDICTION, gfilter)

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

            league_rhos[league][index] = rho
            final_edge_score = rho * total_score
            league_scores[league][index] = final_edge_score
            league_pred_counts[league][index] = len(predictions_with_match_data)
            total_lay_preds = len([
                pwmd for pwmd in predictions_with_match_data if round(pwmd.get_closing_odds_for_predicted_outcome(),2) < round((1 / pwmd.prediction.probability), 2)
            ])
            total_underdog_preds = len([
                pwmd for pwmd in predictions_with_match_data if pwmd.is_prediction_for_underdog(LEAGUES_ALLOWING_DRAWS)
            ])
            #avg_pred_score = final_edge_score / len(predictions_with_match_data) if len(predictions_with_match_data) > 0 else 0.0
            raw_edge = 0
            for pwmd in predictions_with_match_data:
                raw_edge += sigma
            market_roi = league_roi_market_payouts[league][index] / (league_roi_counts[league][index] * ROI_BET_AMOUNT) if league_roi_counts[league][index] > 0 else 0.0
            roi = league_roi_payouts[league][index] / (league_roi_counts[league][index] * ROI_BET_AMOUNT) if league_roi_counts[league][index] > 0 else 0.0
            roi_diff = roi - market_roi

            # Base ROI score requires the miner is beating the market
            final_roi_score = round(rho * ((roi_diff if roi_diff>0 else 0)*100), 4)

            # If ROI is less than 0, but greater than market ROI, penalize the ROI score by distance from 0
            if roi < 0 and roi_diff > 0:
                print(f"Penalizing ROI score for miner {uid} in league {league.name} by {roi:.4f} ({final_roi_score * roi:.4f}): {final_roi_score:.4f} -> {final_roi_score + (final_roi_score * roi):.4f}")
                final_roi_score = final_roi_score + (final_roi_score * roi)

            """
            # If roi_diff is less than 0, but rho is greater than our threshold and roi is still above 0, give small score. this is for leniency of long-standing miners
            if roi_diff <= 0 and rho > 0.5 and roi > 0:
                print(f"Giving leniency to miner {uid} in league {league.name} with rho {rho:.4f} and roi_diff {roi_diff:.4f}: {final_roi_score:.4f} -> {round((roi * rho) * 50, 4):.4f}")
                final_roi_score = round((roi * rho) * 50, 4)
            """
            
            roi_incr = roi
            market_roi_incr = market_roi
            roi_incr_diff = 0
            max_possible_roi_incr = 0
            # Calculate incremental ROI score for miner and market. Penalize if too similar.
            if league_roi_incr_counts[league][index] == round(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * ROI_INCR_PRED_COUNT_PERCENTAGE, 0) and final_roi_score > 0:
                max_possible_roi_incr = league_roi_incr_max_possible_payouts[league][index] / (league_roi_incr_counts[league][index] * ROI_BET_AMOUNT) if league_roi_incr_counts[league][index] > 0 else 0.0
                market_roi_incr = league_roi_incr_market_payouts[league][index] / (league_roi_incr_counts[league][index] * ROI_BET_AMOUNT) if league_roi_incr_counts[league][index] > 0 else 0.0
                roi_incr = league_roi_incr_payouts[league][index] / (league_roi_incr_counts[league][index] * ROI_BET_AMOUNT) if league_roi_incr_counts[league][index] > 0 else 0.0
                roi_incr_diff = roi_incr - market_roi_incr
                max_possible_roi_incr_diff = max_possible_roi_incr - market_roi_incr
                
                # if incremental ROI and incremental market ROI is within the difference threshold, calculate penalty
                if abs(roi_incr_diff) <= MAX_INCR_ROI_DIFF_PERCENTAGE:
                    # Exponential decay scaling
                    k = 30  # Decay constant; increase for steeper decay
                    # Scale the penalty factor to max at 0.99
                    penalty_factor = 0.99 * np.exp(-k * abs(roi_incr_diff))
                    adjustment_factor = 1 - penalty_factor
                    print(f"Incremental ROI score penalty for miner {uid} in league {league.name}: {roi_incr:.4f} vs {market_roi_incr:.4f} ({roi_incr_diff:.4f}), adj. factor {adjustment_factor:.4f}: {final_roi_score:.4f} -> {final_roi_score * adjustment_factor:.4f}")
                    if abs(max_possible_roi_incr_diff) <= MAX_INCR_ROI_DIFF_PERCENTAGE:
                        print(f"-- Ope, just kidding. Max possible incremental ROI is close to incremental market ROI. Miner and market killing it. No penalty.")
                    else:
                        final_roi_score *= adjustment_factor

            league_roi_scores[league][index] = final_roi_score
            # Only log scores for miners committed to the league
            if uid in league_miner_uids:
                league_table_data.append([
                    uid,
                    round(final_edge_score, 2), 
                    round(final_roi_score, 4), 
                    str(round(roi*100, 2)) + "%", 
                    str(round(market_roi*100, 2)) + "%", 
                    str(round(roi_diff*100, 2)) + "%", 
                    str(round(roi_incr*100, 2)) + "%",
                    str(round(market_roi_incr*100, 2)) + "%",
                    str(round(abs(roi_incr_diff)*100, 2)) + "%",
                    str(round(max_possible_roi_incr*100, 2)) + "%",
                    len(predictions_with_match_data),
                    str(round(rho, 4)) + "" if rho > LEAGUE_MINIMUM_RHOS[league] else str(round(rho, 4)) + "*",
                    str(total_lay_preds) + "/" + str(len(predictions_with_match_data)), 
                    total_underdog_preds, 
                    round(raw_edge, 4)
                ])

            # Print edge mismatch count
            if edge_mismatch_count > 0:
                print(f"Edge mismatch count for miner {uid} in league {league.name}: {edge_mismatch_count}")
            if edge_mismatch_oppo_count > 0:
                print(f"Edge mismatch opposite count for miner {uid} in league {league.name}: {edge_mismatch_oppo_count}")

        # Log league scores
        if league_table_data:
            print(f"\nScores for {league.name}:")
            print(tabulate(league_table_data, headers=['UID', 'Edge Score', 'ROI Score', 'ROI', 'Mkt ROI', 'ROI Diff', 'ROI Incr', 'Mkt ROI Incr', 'ROI Incr Diff', 'Max ROI Incr', '# Preds', 'Rho', '# Lay Preds', '# Underdog Preds', 'Raw Edge'], tablefmt='grid'))
            print("* indicates rho is below minimum threshold and not eligible for rewards yet\n")
        else:
            print(f"No non-zero scores for {league.name}")

        # Normalize league scores and weight for Edge and ROI scores
        # Normalize edge scores
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

                """ # Debugging output for specific case
                if i == 242 and league == League.NBA:
                    print(f"i: {i}, score: {score}, normalized_edge: {normalized_edge[i]}")
                    print(f"min_edge_for_miner: {min_edge_for_miner}, rho: {rho}")
                    print(f"({score} - {MIN_EDGE_SCORE}) / ({max_edge} - {MIN_EDGE_SCORE}): {(score - MIN_EDGE_SCORE) / (max_edge - MIN_EDGE_SCORE)}")
                """
                
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
        print(f"\nTop 10 Scores for {league.name}:")
        print(tabulate(top_scores_table, headers=['#', 'UID', 'Final Score'], tablefmt='grid'))

        if len(matches_without_odds) > 0:
            print(f"\n==============================================================================")
            print(f"Odds were not found for the following matches within {league.name}:")
            for mwo in matches_without_odds:
                print(f"{mwo[0]} - {mwo[1]}")
            print(f"==============================================================================")

        # Analyze league for prediction integrity
        suspicious_miners, penalties = prediction_integrity_controller.analyze_league(league, predictions_for_integrity_analysis, ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league])
        #suspicious_miners, penalties = [], {}
        # Print league results
        print(f"\n==============================================================================")
        print(f"Total suspicious miners in {league.name}: {len(suspicious_miners)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(suspicious_miners))}")
        
        print(f"\nTotal miners to penalize in {league.name}: {len(penalties)}")
        # penalties is a dict of {miner_uid: penalty_score}
        print(f"Miners: {', '.join(str(m) for m in sorted(penalties))}")
        print(f"==============================================================================")
        final_suspicious_miners.update(suspicious_miners)
        final_integrity_penalties.update(penalties)

    # Log final integrity results
    print(f"********************* Prediction Integrity Controller Findings  *********************")
    # Get a unique list of coldkeys from metagraph
    coldkeys = list(set(metagraph.coldkeys))
    for coldkey in coldkeys:
        uids_for_coldkey = []
        for miner_uid in final_suspicious_miners:
            if metagraph.coldkeys[miner_uid] == coldkey:
                #if miner_uid in final_integrity_penalties:
                if final_integrity_penalties.get(miner_uid) is not None and final_integrity_penalties[miner_uid] == 1.0:
                    miner_uid = f"{miner_uid} 💀"
                uids_for_coldkey.append(str(miner_uid))
        if len(uids_for_coldkey) > 0:
            print(f"\nColdkey: {coldkey}")
            print(f"Suspicious Miners: {', '.join(str(m) for m in sorted(uids_for_coldkey))}")

    print(f"\nTotal suspicious miners across all leagues: {len(final_suspicious_miners)}")
    if len(final_suspicious_miners) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_suspicious_miners))}")

    print(f"\nTotal miners to penalize across all leagues: {len(final_integrity_penalties)}")
    if len(final_integrity_penalties) > 0:
        # Print a table of miners to penalize
        penalized_table = []
        for uid, penalty_percentage in final_integrity_penalties.items():
            penalized_table.append([uid, f"{penalty_percentage*100:.2f}%"])
        print(tabulate(penalized_table, headers=['UID', 'Penalty %'], tablefmt='grid'))
    print(f"************************************************************************")

    # Update all_scores with weighted sum of league scores for each miner
    print("************ Normalizing and applying penalties and leagues scoring percentages to scores ************")
    for league, percentage in LEAGUE_SCORING_PERCENTAGES.items():
        print(f"  • {league}: {percentage*100}%")
    print("*************************************************************")

    # Apply integrity penalties for miners and no prediction responses
    for league in ACTIVE_LEAGUES:
        # Check and penalize miners that are not committed to any active leagues -- before normalization
        #league_scores[league] = check_and_apply_league_commitment_penalties(vali, league_scores[league], all_uids)
        # Apply penalties for miners that have not responded to prediction requests -- before normalization
        league_scores[league] = apply_no_prediction_response_penalties(metagraph, league, uids_to_last_leagues, uids_to_leagues_last_updated, league_rhos, league_scores[league], all_uids)

        # Apply the integrity penalty to the score -- before normalization
        for uid, penalty_percentage in final_integrity_penalties.items():
            score_after_penalty = league_scores[league][uid] * (1 - penalty_percentage)
            league_scores[league][uid] = score_after_penalty
    
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

    """ Weird attempt to apply Pareto per league before scaling 
    final_scores = [0.0] * len(all_uids)
    for i in range(len(all_uids)):
        all_scores[i] = sum(league_scores[league][i] for league in ACTIVE_LEAGUES)

    # Step 1: Calculate total positive scores for each league
    league_totals = {league: 0.0 for league in ACTIVE_LEAGUES}
    for league in ACTIVE_LEAGUES:
        league_totals[league] = sum(score for score in league_scores[league] if score > 0)

    # Step 2: Apply Pareto per league before scaling
    pareto_scores_per_league = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    for league in ACTIVE_LEAGUES:
        total_league_score = league_totals[league]

        # Apply Pareto before scaling
        if total_league_score > 0:
            pareto_scores_per_league[league] = apply_pareto(league_scores[league], all_uids, PARETO_MU, PARETO_ALPHA)
            #pareto_scores_per_league[league] = [score if score > (PARETO_MU + PARETO_MU*0.05) else 0 for score in pareto_scores_per_league[league]]

    # Step 3: Scale scores within each league to match allocation percentage
    scaled_scores_per_league = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    for league in ACTIVE_LEAGUES:
        total_league_score = sum(pareto_scores_per_league[league])
        allocation = LEAGUE_SCORING_PERCENTAGES[league] * 100  # Convert to percentage

        if total_league_score > 0:
            scaling_factor = allocation / total_league_score  # Factor to scale league scores
            scaled_scores_per_league[league] = [
                (score * scaling_factor if score > 0 else 0) for score in pareto_scores_per_league[league]
            ]

    # Step 4: Aggregate scaled scores across all leagues
    for i in range(len(all_uids)):
        final_scores[i] = sum(scaled_scores_per_league[league][i] for league in ACTIVE_LEAGUES)

    # Step 5: Verify emissions allocation percentages
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

    """

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
        if top_scores[i] is not None and top_scores[i] > 0:
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

    # End time tracking
    end_time = dt.datetime.now()
    print(f"\nEnded scoring at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken for scoring: {(end_time - start_time).total_seconds() / 60:.2f} minutes")

    # Generate graph of Pre-Pareto vs Final Pareto Scores
    graph_results(all_uids, all_scores, final_scores, uids_to_league)

    cabal_uids = ""
    for uid in all_uids:
        if metagraph.coldkeys[uid] in []:
            cabal_uids += f"{uid},"
    print(f"\nCabal UIDs: {cabal_uids}")


def graph_results(all_uids, all_scores, final_scores, uids_to_league):
    """
    Graphs the Pre-Pareto and Final Pareto scores with league-based color coding.

    :param all_uids: List of unique identifiers for the miners
    :param all_scores: List of Pre-Pareto scores
    :param final_scores: List of Final Pareto scores
    :param uids_to_league: Dictionary mapping miner UID to league
    """
    # Sort the miners based on Final Pareto Scores
    sorted_indices = np.argsort(final_scores)
    sorted_uids = np.array(all_uids)[sorted_indices]
    sorted_final_pareto_scores = np.array(final_scores)[sorted_indices]

    # X-axis for the miners (from 0 to number of miners)
    x_axis = np.arange(len(all_uids))

    # Filter out zero scores after sorting
    non_zero_indices = sorted_final_pareto_scores > 0.1
    x_axis = x_axis[non_zero_indices]
    sorted_uids = sorted_uids[non_zero_indices]
    sorted_final_pareto_scores = sorted_final_pareto_scores[non_zero_indices]

    # Create the output directory if it doesn't exist
    output_dir = "tests/imgs"
    os.makedirs(output_dir, exist_ok=True)

    # Set up dark mode
    plt.style.use("dark_background")

    # Assign colors to leagues dynamically
    unique_leagues = sorted(set(uids_to_league.values()))
    color_cycle = itertools.cycle(["#FF5733", "#33FF57", "#5733FF", "#FFC300", "#33FFF5"])  # Expandable color list
    league_colors = {league: next(color_cycle) for league in unique_leagues}

    # Plot each league separately
    plt.figure(figsize=(12, 6))
    for league in unique_leagues:
        league_indices = [i for i, uid in enumerate(sorted_uids) if uids_to_league.get(uid) == league]
        plt.scatter(
            np.array(x_axis)[league_indices], 
            np.array(sorted_final_pareto_scores)[league_indices], 
            label=league, 
            color=league_colors[league], 
            s=10, 
            alpha=0.8
        )

    plt.xlabel("Miners (sorted by score)", fontsize=12, color='white')
    plt.ylabel("Scores", fontsize=12, color='white')
    plt.title("Final Pareto Scores by League", fontsize=14, color='white')
    plt.legend(title="Leagues", fontsize=10, facecolor='gray', edgecolor='white')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()

    # Save the graph as an image
    output_path = os.path.join(output_dir, "pareto_scores.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


if "__main__" == __name__:
    calculate_incentives_and_update_scores()
