import logging
from pathlib import Path
from typing import List, Set, Dict, Union
import random

import bittensor as bt

from common.data import League, MatchPredictionWithMatchData
from common.constants import (
    EXACT_MATCH_PREDICTIONS_THRESHOLD,
    SUSPICIOUS_MATCH_PREDICTIONS_THRESHOLD
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
        suspicious_relationships = self.statistical_analyzer.analyze_prediction_clusters(league_predictions, excluded_miners=cleared_miners)

        suspicious_miner_ids = set()
        miners_to_penalize = set()
        for key, relationship in suspicious_relationships.items():
            suspicious_miner_ids.update(relationship['miners'])

        miners_to_penalize = self.get_miners_to_penalize(suspicious_relationships)

        # Print random sample history
        if random.random() < 0.5:
            self.print_sample_history(suspicious_relationships, miners_to_penalize)

        return suspicious_miner_ids, miners_to_penalize
    
    def get_miners_to_penalize(self, suspicious_relationships: dict[str, list[tuple]]) -> Set[int]:
        """Get set of miners that should be penalized based on summary results."""
        if len(suspicious_relationships) == 0:
            return set()

        # Create a dict of miners to penalize
        miners_to_penalize = set()
        for key, relationship in suspicious_relationships.items():
            if relationship['num_matches_with_exact'] >= EXACT_MATCH_PREDICTIONS_THRESHOLD:
                for miner in relationship['miners']:
                    miners_to_penalize.add(miner)
            elif relationship['num_matches'] >= SUSPICIOUS_MATCH_PREDICTIONS_THRESHOLD:
                for miner in relationship['miners']:
                    miners_to_penalize.add(miner)
        
        return miners_to_penalize
    
    def print_sample_history(self, suspicious_relationships: dict[str, list[tuple]], miners_to_penalize: Set[int]) -> None:
        """Print a sample of the history for suspicious relationships."""
        for key, value in suspicious_relationships.items():
            if value['miners'][0] in miners_to_penalize or value['miners'][1] in miners_to_penalize:
                print(f"Miners: {value['miners']}, Matches: {value['num_matches']}, Predictions: {value['num_predictions']}, Predictions per match: {value['predictions_per_match']}, Exact predictions: {value['num_exact_predictions']}")
                # Get random sample of history
                for history in value['history'][:3]:
                    print(f"Match: {history['match_id']}, Difference: {history['difference']}, Choice: {history['choice']}, Prob1: {history['prob1']}, Prob2: {history['prob2']}")
                
