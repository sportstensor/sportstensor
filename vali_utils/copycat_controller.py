from typing import List, Set, Dict, Union
import random
from datetime import datetime
from tabulate import tabulate
from collections import defaultdict

import bittensor as bt

from common.data import League, MatchPredictionWithMatchData
from common.constants import (
    EXACT_MATCH_PREDICTIONS_THRESHOLD,
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
    ) -> tuple[Set[int], Set[int], Set[int]]:
        """
        Analyze a specific league for duplicate predictions and copycat patterns.
        
        Args:
            league: League to analyze
            league_predictions: list of predictions with match data to analyze.
            
        Returns:
            Tuple of suspicious miner ids and miners to penalize.
        """
        bt.logging.info(f"Analyzing {len(league_predictions)} league predictions and {len(ordered_matches)} ordered matches for copycat patterns: {league.name}")

        if not league_predictions:
            bt.logging.warning(f"No predictions found for {league.name}")
            return set(), set(), set()
        
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
        # Print relationship table
        self.print_relationship_table(suspicious_relationships, miners_to_penalize, league)

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
            elif 'consecutive_patterns' in relationship and relationship['consecutive_patterns']['max_consecutive'] >= SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD:
                for miner in relationship['miners']:
                    miners_to_penalize.add(miner)
        
        return miners_to_penalize, miners_with_exact_matches
    
    def print_sample_history(self, suspicious_relationships: dict[str, list[tuple]], miners_to_penalize: Set[int]) -> None:
        """Print a sample of the history for suspicious relationships."""
        abs_diff = 0
        abs_diff_count = 0
        greatest_abs_diff = 0
        for key, value in suspicious_relationships.items():
            for history in value['history']:
                if history['absolute_difference'] > 0:
                    abs_diff += history['absolute_difference']
                    abs_diff_count += 1
                    if history['absolute_difference'] > greatest_abs_diff:
                        greatest_abs_diff = history['absolute_difference']
            
            if value['num_exact_predictions'] >= EXACT_MATCH_PREDICTIONS_THRESHOLD or ('consecutive_patterns' in value and len(value['consecutive_patterns']['streak_details']) > 0):
                print(f"\nMiners: {value['miners']}, Matches: {value['num_matches']}, Predictions: {value['num_predictions']}, Predictions per match: {value['predictions_per_match']}, Exact predictions: {value['num_exact_predictions']}")
                # print consecutive patterns
                if 'consecutive_patterns' in value and len(value['consecutive_patterns']['streak_details']) > 0:
                    print(f"Consecutive pattern streaks of {SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD} matches or more: {value['consecutive_patterns']['num_streaks']}, Max streak: {value['consecutive_patterns']['max_consecutive']}")
                    for i, streaks in enumerate(value['consecutive_patterns']['streak_details']):
                        print(f"Streak {i+1}:")
                        for streak in streaks:
                            print(f"-- Match: {streak['match_id']}, Match Date: {streak['match_date']}, Difference: {streak['difference']}, Abs Difference: {streak['absolute_difference']}, Choice: {streak['choice']}, Prob1: {streak['prob1']}, Prob2: {streak['prob2']}, Pred1 Date: {streak['pred1_date']}, Pred2 Date: {streak['pred2_date']}")
                else:
                    # Get random sample of history
                    for history in value['history']:
                        print(f"Match: {history['match_id']}, Difference: {history['difference']}, Abs Difference: {history['absolute_difference']}, Choice: {history['choice']}, Prob1: {history['prob1']}, Prob2: {history['prob2']}, Pred1 Date: {history['pred1_date']}, Pred2 Date: {history['pred2_date']}")

        #print(f"\nGreatest absolute difference: {round(greatest_abs_diff * 100, 2)}%")
        #if abs_diff_count > 0:
            #print(f"Average absolute difference: {round((abs_diff / abs_diff_count) * 100, 2)}%")                

    def print_relationship_table(self, suspicious_relationships: dict[str, list[tuple]], miners_to_penalize: Set[int] = None, league: League = None) -> None:
        """
        Generate and print a table showing each suspicious UID and their related UIDs.
        """
        # Initialize relationship mapping
        uid_relationships = defaultdict(set)
        
        # Build relationships
        for key, value in suspicious_relationships.items():
            miner1, miner2 = value['miners']
            
            # Add each relationship only once
            uid_relationships[miner1].add(miner2)
            uid_relationships[miner2].add(miner1)
        
        # Convert to table format with wrapped UIDs
        table_data = []
        for uid in sorted(uid_relationships.keys()):
            # Remove duplicates and sort
            related_uids = sorted(set(uid_relationships[uid]))
            
            # Format as chunks of 10
            chunks = [related_uids[i:i+10] for i in range(0, len(related_uids), 10)]
            related_str = '\n'.join(
                ', '.join(str(r) for r in chunk)
                for chunk in chunks
            )
            
            penalized = ""
            if miners_to_penalize is not None:
                if uid in miners_to_penalize:
                    penalized = "✔"
            
            table_data.append([uid, related_str, penalized])
        
        # Print table
        print(f"\n{league.name} Suspicious UID Relationship Table")
        print(tabulate(
            table_data,
            headers=["UID", "Related UIDs", "Penalized"],
            tablefmt="pretty",
            stralign="left",
            numalign="left"
        ))
