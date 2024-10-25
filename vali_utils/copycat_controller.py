import logging
from pathlib import Path
from typing import List, Set

import bittensor as bt

from common.data import League, MatchPredictionWithMatchData
from vali_utils.analysis_utils import PredictionPatternAnalyzer, StatisticalAnalyzer
from vali_utils.suspicious_utils import (
    ExactMatchSummary,
    StatisticalPatternSummary,
    print_exact_match_report,
    print_statistical_report
)

class CopycatDetectionController:
    def __init__(
        self,
    ):  
        # Initialize analyzers
        self.pattern_analyzer = PredictionPatternAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer(variance_threshold=0.02)

    def analyze_league(
        self,
        league: League,
        league_predictions: List[MatchPredictionWithMatchData] = None
    ) -> tuple[Set[int], Set[int]]:
        """
        Analyze a specific league for duplicates.
        
        Args:
            league: League to analyze
            miner_uids: Optional list of miner UIDs to analyze. If None, analyzes all miners.
            
        Returns:
            Tuple of (all duplicate miners, miners to penalize)
        """
        logging.info(f"Processing league: {league.name}")

        if not league_predictions:
            logging.warning(f"No predictions found for {league.name}")
            return set(), set()

        # Analyze exact matches
        duplicate_groups, pattern_details = self.pattern_analyzer.analyze_prediction_patterns(league_predictions)

        # Analyze and report exact matches
        exact_match_summary = ExactMatchSummary(min_confidence_threshold=40.0)
        exact_match_summary.add_patterns(duplicate_groups, pattern_details)
        print_exact_match_report(exact_match_summary, league.name)

        # Get all miners with duplicates
        miner_ids_with_duplicates = set(m for miners in duplicate_groups.values() for m in miners)
        print(f"\nTotal miners with duplicates in {league.name}: {len(miner_ids_with_duplicates)}")
        print(f"Miners: {', '.join(str(m) for m in miner_ids_with_duplicates)}")

        # Get miners to penalize based on exact matches
        miners_to_penalize = self._get_miners_to_penalize(exact_match_summary)
        
        # Analyze statistical patterns for remaining miners
        cleared_miners = miner_ids_with_duplicates - miners_to_penalize
        suspicious_patterns = self.statistical_analyzer.analyze_prediction_clusters(league_predictions)
        
        statistical_summary = StatisticalPatternSummary(min_confidence_threshold=30.0)
        statistical_summary.analyze_patterns(suspicious_patterns, excluded_miners=cleared_miners)
        print_statistical_report(statistical_summary, league.name)

        return miner_ids_with_duplicates, miners_to_penalize

    def _get_miners_to_penalize(self, summary: ExactMatchSummary) -> Set[int]:
        """Get set of miners that should be penalized based on summary results."""
        if not summary.groups:
            return set()

        miners_to_penalize = set()
        for group in summary.groups:
            stats = group.stats
            confidence = group.calculate_confidence()
            if confidence >= 60 and stats['unique_matches'] >= 10:
                miners_to_penalize.update(group.miners)

        return miners_to_penalize
