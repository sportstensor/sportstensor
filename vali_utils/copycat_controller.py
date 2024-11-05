import logging
from pathlib import Path
from typing import List, Set, Dict, Union
import random
from datetime import datetime

import bittensor as bt

from common.data import League, MatchPredictionWithMatchData
from common.constants import (
    EXACT_MATCH_PREDICTIONS_THRESHOLD,
    SUSPICIOUS_MATCH_PREDICTIONS_THRESHOLD,
    SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD
)
from vali_utils.analysis_utils import StatisticalAnalyzer


class CopycatDetectionController:
    def __init__(
        self,
    ):  
        # Initialize analyzers
        self.statistical_analyzer = StatisticalAnalyzer()

    def analyze_league(
        self,
        league: League,
        league_predictions: List[MatchPredictionWithMatchData] = None,
        ordered_matches: List[tuple[str, datetime]] = None
    ) -> tuple[Set[int], Set[int]]:
        """
        Analyze a specific league for duplicate predictions and copycat patterns.
        
        Args:
            league: League to analyze
            league_predictions: list of predictions with match data to analyze.
            
        Returns:
            Tuple of suspicious miner ids and miners to penalize.
        """
        bt.logging.info(f"Analyzing league predictions for copycat patterns: {league.name}")

        if not league_predictions:
            bt.logging.warning(f"No predictions found for {league.name}")
            return set()
        
        # Analyze statistical patterns for miner predictions
        cleared_miners = set()
        suspicious_relationships = self.statistical_analyzer.analyze_prediction_clusters(league_predictions, ordered_matches, excluded_miners=cleared_miners)

        suspicious_miner_ids = set()
        miners_to_penalize = set()
        for key, relationship in suspicious_relationships.items():
            suspicious_miner_ids.update(relationship['miners'])

        miners_to_penalize, miners_with_exact_matches = self.get_miners_to_penalize(suspicious_relationships)

        # Print random sample history
        self.print_sample_history(suspicious_relationships, miners_to_penalize)

        return suspicious_miner_ids, miners_to_penalize, miners_with_exact_matches
    
    def get_miners_to_penalize(self, suspicious_relationships: dict[str, list[tuple]]) -> tuple[Set[int], Set[int]]:
        """Get set of miners that should be penalized based on summary results."""
        if len(suspicious_relationships) == 0:
            return set(), set()

        # Create a dict of miners to penalize
        miners_to_penalize = set()
        miners_with_exact_matches = set()
        for key, relationship in suspicious_relationships.items():
            if relationship['num_matches_with_exact'] >= EXACT_MATCH_PREDICTIONS_THRESHOLD:
                for miner in relationship['miners']:
                    miners_to_penalize.add(miner)
                    miners_with_exact_matches.add(miner)
            #elif relationship['num_matches'] >= SUSPICIOUS_MATCH_PREDICTIONS_THRESHOLD:
                #for miner in relationship['miners']:
                    #miners_to_penalize.add(miner)
            elif 'consecutive_patterns' in relationship and relationship['consecutive_patterns']['max_consecutive'] >= SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD:
                for miner in relationship['miners']:
                    miners_to_penalize.add(miner)
        
        return miners_to_penalize, miners_with_exact_matches
    
    def print_sample_history(self, suspicious_relationships: dict[str, list[tuple]], miners_to_penalize: Set[int]) -> None:
        """Print a sample of the history for suspicious relationships."""
        for key, value in suspicious_relationships.items():
            if value['miners'][0] in miners_to_penalize or value['miners'][1] in miners_to_penalize:
                if random.random() < 0.01 and (value['num_exact_predictions'] >= EXACT_MATCH_PREDICTIONS_THRESHOLD or ('consecutive_patterns' in value and len(value['consecutive_patterns']['streak_details']) > 0)):
                    print(f"Miners: {value['miners']}, Matches: {value['num_matches']}, Predictions: {value['num_predictions']}, Predictions per match: {value['predictions_per_match']}, Exact predictions: {value['num_exact_predictions']}")
                    # print consecutive patterns
                    if 'consecutive_patterns' in value and len(value['consecutive_patterns']['streak_details']) > 0:
                        print(f"Consecutive pattern streaks of {SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD} matches or more: {value['consecutive_patterns']['num_streaks']}, Max streak: {value['consecutive_patterns']['max_consecutive']}")
                        for i, streaks in enumerate(value['consecutive_patterns']['streak_details']):
                            # Streak: [{'match_id': '059e075310b172d74bf65aaaf5ef2951', 'match_date': datetime.datetime(2024, 10, 27, 17, 0), 'difference': 0.0046, 'choice': 'AwayTeam', 'prob1': 0.7637, 'prob2': 0.7591}, {'match_id': '2318d3f1037efd157e548084cfb33e94', 'match_date': datetime.datetime(2024, 10, 27, 17, 0), 'difference': 0.0008, 'choice': 'HomeTeam', 'prob1': 0.8837, 'prob2': 0.8845}, {'match_id': '3da3c1f83c87aa70f58bf811a7b2d56c', 'match_date': datetime.datetime(2024, 10, 27, 17, 0), 'difference': 0.0041, 'choice': 'AwayTeam', 'prob1': 0.6391, 'prob2': 0.635}]
                            print(f"Streak {i+1}:")
                            for streak in streaks:
                                print(f"-- Match: {streak['match_id']}, Match Date: {streak['match_date']}, Difference: {streak['difference']}, 'Pronounced Difference: {streak['pronounced_difference']}, Choice: {streak['choice']}, Prob1: {streak['prob1']}, Prob2: {streak['prob2']}")
                    else:
                        # Get random sample of history
                        for history in value['history'][:3]:
                            print(f"Match: {history['match_id']}, Difference: {history['difference']}, 'Pronounced Difference: {history['pronounced_difference']}, Choice: {history['choice']}, Prob1: {history['prob1']}, Prob2: {history['prob2']}")
                
