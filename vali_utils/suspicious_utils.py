from collections import defaultdict
from typing import List, Tuple

# Base classes
class BaseSuspiciousGroup:
    def __init__(self, miners: set[int]):
        self.miners = miners
        self.match_ids = set()
        self.match_details = []  # Generic storage for match-related data
        self.stats_cache = None  # For caching computed statistics
    
    @property
    def stats(self):
        """Basic statistics shared by all group types"""
        return {
            'miner_count': len(self.miners),
            'unique_matches': len(self.match_ids),
            'total_matches': len(self.match_details)
        }
    
    @property
    def evidence_strength(self) -> str:
        """Convert confidence score to categorical strength"""
        confidence = self.calculate_confidence()
        if confidence >= 80:
            return "VERY STRONG"
        elif confidence >= 60:
            return "STRONG"
        elif confidence >= 40:
            return "MODERATE"
        elif confidence >= 20:
            return "WEAK"
        else:
            return "VERY WEAK"

    def calculate_confidence(self) -> float:
        """Must be implemented by subclasses"""
        raise NotImplementedError


class BaseSuspiciousSummary:
    def __init__(self, min_confidence_threshold: float = 20.0):
        self.groups = []
        self.min_confidence_threshold = min_confidence_threshold
    
    def add_group(self, group):
        if group.calculate_confidence() >= self.min_confidence_threshold:
            self.groups.append(group)
            return True
        return False
    
    def sort_groups(self):
        self.groups.sort(key=lambda g: g.calculate_confidence(), reverse=True)


# Exact match specific implementations
class ExactMatchGroup(BaseSuspiciousGroup):
    def __init__(self, miners: set[int]):
        super().__init__(miners)
        self.exact_match_counts = []
        self.pattern_counts = []
    
    @property
    def stats(self):
        base_stats = super().stats
        base_stats.update({
            'exact_matches': {
                'min': min(self.exact_match_counts) if self.exact_match_counts else 0,
                'max': max(self.exact_match_counts) if self.exact_match_counts else 0,
                'avg': sum(self.exact_match_counts) / len(self.exact_match_counts) if self.exact_match_counts else 0
            },
            'total_predictions': len(self.match_details),  # Add this line
            'predictions_per_match': (
                len(self.match_details) / len(self.match_ids) 
                if self.match_ids else 0
            ),
            'unique_matches': len(self.match_ids)  # This is already in base_stats but included here for clarity
        })
        return base_stats

    def calculate_confidence(self) -> float:
        """Confidence calculation specific to exact matches"""
        stats = self.stats
        
        if not self.match_details or len(self.match_ids) < 5:
            return 0.0
        
        predictions_per_match_score = min(stats['predictions_per_match'] / 2, 1.0) * 50
        
        group_consistency = (
            min(stats['exact_matches']['min'] / stats['exact_matches']['max'], 1.0)
            if stats['exact_matches']['max'] > 0 else 0
        ) * 30
        
        group_size_score = min((stats['miner_count'] - 1) / 4, 1.0) * 20
        
        return round(predictions_per_match_score + group_consistency + group_size_score, 1)


class StatisticalGroup(BaseSuspiciousGroup):
    def __init__(self, miners: set[int]):
        super().__init__(miners)
        self.differences = []
        self.variances = []
        self.formatted_match_details = []
    
    @property
    def stats(self):
        base_stats = super().stats
        base_stats.update({
            'mean_difference': sum(self.differences) / len(self.differences) if self.differences else 0,
            'variance': sum(self.variances) / len(self.variances) if self.variances else 0,
            'consistency': self._calculate_consistency()
        })
        return base_stats

    def _calculate_consistency(self) -> float:
        if not self.differences:
            return 1.0
        mean_diff = sum(self.differences) / len(self.differences)
        variance = sum((d - mean_diff) ** 2 for d in self.differences) / len(self.differences)
        return variance / mean_diff if mean_diff > 0 else 1.0

    def calculate_confidence(self) -> float:
        """Confidence calculation specific to statistical patterns"""
        stats = self.stats
        
        consistency_score = (1 - stats['consistency']) * 40
        match_score = min(len(self.match_ids) / 10, 1.0) * 30
        group_score = min(len(self.miners) / 5, 1.0) * 20
        variance_score = (1 - min(stats['variance'] / 0.02, 1.0)) * 10
        
        return round(
            consistency_score + match_score + group_score + variance_score,
            1
        )


class ExactMatchSummary(BaseSuspiciousSummary):
    def add_patterns(self, duplicate_groups: dict, pattern_details: dict):
        """
        Process exact match patterns and create groups.
        
        Args:
            duplicate_groups: Dictionary mapping pattern hash to list of miner IDs
            pattern_details: Dictionary mapping pattern hash to pattern details
        """
        # First, combine all related miners into groups
        miner_relationships = self._build_relationships(duplicate_groups)
        
        # Create groups from relationships
        grouped_miners = self._build_miner_groups(miner_relationships)
        
        # Create and populate groups
        for group_miners in grouped_miners:
            group = ExactMatchGroup(group_miners)
            self._populate_group_details(group, duplicate_groups, pattern_details)
            self.add_group(group)
        
        self.sort_groups()

    def _build_relationships(self, duplicate_groups: dict) -> dict:
        """
        Build initial relationships between miners based on duplicate patterns.
        
        Args:
            duplicate_groups: Dictionary mapping pattern hash to list of miner IDs
        
        Returns:
            Dictionary mapping miner ID to set of related miner IDs
        """
        miner_relationships = defaultdict(set)
        
        for miners in duplicate_groups.values():
            for m1 in miners:
                for m2 in miners:
                    if m1 != m2:
                        miner_relationships[m1].add(m2)
                        miner_relationships[m2].add(m1)
        
        return miner_relationships

    def _build_miner_groups(self, miner_relationships: dict) -> List[set]:
        """
        Build groups of related miners using relationship graph.
        
        Args:
            miner_relationships: Dictionary mapping miner ID to set of related miner IDs
        
        Returns:
            List of sets, each set containing miner IDs that form a group
        """
        processed_miners = set()
        groups = []
        
        for miner in miner_relationships:
            if miner in processed_miners:
                continue
            
            # Find all connected miners (graph traversal)
            group = {miner}
            to_process = miner_relationships[miner].copy()
            
            while to_process:
                related_miner = to_process.pop()
                if related_miner not in group:
                    group.add(related_miner)
                    to_process.update(
                        m for m in miner_relationships[related_miner] 
                        if m not in group
                    )
            
            processed_miners.update(group)
            groups.append(group)
        
        return groups

    def _populate_group_details(self, group: ExactMatchGroup, duplicate_groups: dict, pattern_details: dict):
        """
        Populate group with match details and statistics.
        
        Args:
            group: ExactMatchGroup to populate
            duplicate_groups: Dictionary mapping pattern hash to list of miner IDs
            pattern_details: Dictionary mapping pattern hash to pattern details
        """
        # Track matches and patterns for this group
        for miner_id in group.miners:
            # Process exact matches
            for pattern_hash, miners in duplicate_groups.items():
                if miner_id in miners:
                    group.exact_match_counts.append(1)
                    pattern = pattern_details[pattern_hash]
                    group.match_details.extend([
                        (match_id, choice, prob)
                        for match_id, choice, prob in pattern
                    ])
                    group.match_ids.update(match_id for match_id, _, _ in pattern)


class StatisticalPatternSummary(BaseSuspiciousSummary):
    def analyze_patterns(self, suspicious_patterns: dict, excluded_miners: set[int] = None):
        """
        Analyze statistical patterns and create groups.
        
        Args:
            suspicious_patterns: Dictionary of suspicious pattern data
            excluded_miners: Set of miner IDs to exclude from analysis
        """
        excluded_miners = excluded_miners or set()
        
        # Build relationships excluding filtered miners
        relationships, pattern_details = self._build_relationships(suspicious_patterns, excluded_miners)
        
        # Create groups from relationships
        grouped_miners = self._build_miner_groups(relationships)
        
        # Create and populate groups
        for group_miners in grouped_miners:
            group = StatisticalGroup(group_miners)
            self._populate_group_details(group, pattern_details)
            self.add_group(group)
        
        self.sort_groups()

    def _build_relationships(
        self,
        suspicious_patterns: dict,
        excluded_miners: set[int]
    ) -> Tuple[dict, dict]:
        """
        Build relationships between miners and collect pattern details.
        
        Args:
            suspicious_patterns: Dictionary of suspicious pattern data
            excluded_miners: Set of miner IDs to exclude
            
        Returns:
            Tuple of (relationships dict, pattern details dict)
        """
        miner_relationships = defaultdict(set)
        pattern_details = defaultdict(list)
        
        for pattern_key, pattern_data in suspicious_patterns.items():
            miners = [m for m in pattern_data['miners'] if m not in excluded_miners]
            if len(miners) < 2:
                continue
            
            for m1 in miners:
                for m2 in miners:
                    if m1 != m2:
                        miner_relationships[m1].add(m2)
                        pattern_details[m1].append({
                            'related_miner': m2,
                            'mean_difference': pattern_data['mean_difference'],
                            'variance': pattern_data['variance'],
                            'matches': pattern_data['history']
                        })
        
        return miner_relationships, pattern_details

    def _build_miner_groups(self, miner_relationships: dict) -> List[set]:
        """
        Build groups of related miners using relationship graph.
        
        Args:
            miner_relationships: Dictionary mapping miner ID to set of related miner IDs
            
        Returns:
            List of sets, each set containing miner IDs that form a group
        """
        processed_miners = set()
        groups = []
        
        for miner in miner_relationships:
            if miner in processed_miners:
                continue
            
            # Find connected group
            group = {miner}
            to_process = miner_relationships[miner].copy()
            
            while to_process:
                related_miner = to_process.pop()
                if related_miner not in group:
                    group.add(related_miner)
                    to_process.update(
                        m for m in miner_relationships[related_miner]
                        if m not in group
                    )
            
            processed_miners.update(group)
            groups.append(group)
        
        return groups

    def _populate_group_details(self, group: StatisticalGroup, pattern_details: dict):
        """
        Populate group with statistical details.
        
        Args:
            group: StatisticalGroup to populate
            pattern_details: Dictionary of pattern details by miner
        """
        for m1 in group.miners:
            for detail in pattern_details[m1]:
                if detail['related_miner'] in group.miners:
                    group.differences.append(detail['mean_difference'])
                    group.variances.append(detail['variance'])
                    
                    # Store match details in a format suitable for printing
                    for match in detail['matches']:
                        group.match_ids.add(match['match_id'])
                        group.match_details.extend(detail['matches'])
                        group.formatted_match_details.append(
                            (match['match_id'], match['choice'], match['prob1'])  # Use prob1 as reference
                        )

def print_exact_match_report(summary: ExactMatchSummary, league_name: str = None):
    """
    Print analysis report for exact match duplicates.

    Args:
        summary: ExactMatchSummary containing the analysis results
        league_name: Optional name of league being analyzed
    """
    header = f"\n{'='*80}\n"
    if league_name:
        header += f"HIGH CONFIDENCE DUPLICATE GROUPS IN {league_name}\n"
    else:
        header += "HIGH CONFIDENCE DUPLICATE GROUPS\n"
    header += f"{'='*80}\n"
    print(header)
    
    if not summary.groups:
        print("No high-confidence duplicate groups detected.")
        return
    
    for i, group in enumerate(summary.groups, 1):
        stats = group.stats
        confidence = group.calculate_confidence()
        
        print(f"\nGroup {i}: {stats['miner_count']} miners")
        print(f"{'─'*40}")
        print(f"Confidence: {confidence}% ({group.evidence_strength})")
        print(f"Miners: {', '.join(str(m) for m in sorted(group.miners))}")
        print(
            f"Exact Duplicates: {stats['total_predictions']} predictions across {stats['unique_matches']} unique matches "
        )
        
        # Show sample of duplicate matches if confidence is high
        if confidence >= 60 and group.match_details:
            print("\nSample Duplicate Matches:")
            # Get unique matches (some might be repeated in exact_match_details)
            unique_matches = list(set(group.match_details))[:3]  # Show up to 3 examples
            for match_id, choice, prob in unique_matches:
                print(f"  Match {match_id} : {choice} @ {prob:.3f}")

def print_statistical_report(summary: StatisticalPatternSummary, league_name: str = None):
    """
    Print analysis report for statistical patterns.
    
    Args:
        summary: StatisticalPatternSummary containing the analysis results
        league_name: Optional name of league being analyzed
    """
    header = f"\n{'='*80}\n"
    if league_name:
        header += f"STATISTICAL PATTERN ANALYSIS FOR {league_name}\n"
    else:
        header += "STATISTICAL PATTERN ANALYSIS\n"
    header += f"{'='*80}\n"
    print(header)
    
    if not summary.groups:
        print("No significant statistical patterns detected.")
        return
    
    for i, group in enumerate(summary.groups, 1):
        stats = group.stats
        confidence = group.calculate_confidence()
        
        print(f"\nPattern Group {i}: {stats['miner_count']} miners")
        print(f"{'─'*40}")
        print(f"Confidence: {confidence}%")
        print(f"Miners: {', '.join(str(m) for m in sorted(group.miners))}")
        print(f"Matches Analyzed: {stats['unique_matches']}")
        print(f"Mean Difference: {stats['mean_difference']:.4f}")
        print(f"Pattern Variance: {stats['variance']:.4f}")

        # Show sample of matches with similar predictions if confidence is high
        if confidence >= 60 and group.formatted_match_details:
            print("\nSample Similar Predictions:")
            # Get unique matches
            unique_matches = list(set(group.formatted_match_details))[:3]
            for match_id, choice, prob in unique_matches:
                print(f"  Match {match_id} : {choice} @ {prob:.3f}")
