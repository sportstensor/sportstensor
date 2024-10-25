import json
import hashlib
from collections import defaultdict
from typing import List, Tuple

from common.data import MatchPrediction, MatchPredictionWithMatchData


### Pattern-Based Detection
class PredictionPatternAnalyzer:
    def analyze_prediction_patterns(
        self,
        predictions: list[MatchPredictionWithMatchData],
        window_size: int = 50
    ) -> tuple[dict[str, list[str]], dict[str, list[tuple]]]:
        """
        Analyze patterns of predictions across multiple matches to find groups
        of miners with identical or very similar predictions.
        
        Returns:
            Tuple containing:
            - Dictionary of pattern groups (hash -> list of miner IDs)
            - Dictionary of pattern details (hash -> list of prediction details)
        """
        miner_patterns = defaultdict(list)
        pattern_groups = defaultdict(list)
        pattern_details = {}
        
        # Build pattern fingerprints for each miner
        for miner_id in set(p.prediction.minerId for p in predictions):
            miner_preds = [p for p in predictions if p.prediction.minerId == miner_id]
            pattern = self._build_miner_pattern(miner_preds, window_size)
            miner_patterns[miner_id] = pattern
            
        # Group similar patterns
        for miner_id, pattern in miner_patterns.items():
            pattern_hash = self._hash_pattern(pattern)
            pattern_groups[pattern_hash].append(miner_id)
            pattern_details[pattern_hash] = pattern
            
        duplicate_groups = {k: v for k, v in pattern_groups.items() if len(v) > 1}
        duplicate_details = {k: v for k, v in pattern_details.items() if k in duplicate_groups}
            
        return duplicate_groups, duplicate_details

    def _build_miner_pattern(
        self,
        predictions: list[MatchPredictionWithMatchData],
        window_size: int
    ) -> list[tuple]:
        """
        Build a pattern fingerprint from recent predictions.
        Returns list of (matchId, choice, probability) tuples with match details.
        """
        recent_preds = sorted(predictions, key=lambda x: x.prediction.predictionDate)[-window_size:]
        return [(
            p.prediction.matchId,
            p.prediction.probabilityChoice,
            round(p.prediction.probability, 3)
        ) for p in recent_preds]
    
    def _hash_pattern(self, pattern: List[Tuple[str, float]]) -> str:
        """
        Create a unique hash for a prediction pattern.
        
        Args:
            pattern: List of (choice, probability) tuples representing predictions
            
        Returns:
            str: A unique hash representing the pattern
            
        The hash is created by:
        1. Converting the pattern to a JSON string to ensure consistent serialization
        2. Creating an MD5 hash of the string (SHA-256 could be used for more security if needed)
        3. Returning the hex digest of the hash
        
        Note: The pattern must be exactly the same (including probability up to 3 decimal places)
        to generate the same hash.
        """
        # Convert pattern to a JSON string for consistent serialization
        pattern_str = json.dumps(pattern, sort_keys=True)
        
        # Create hash
        hash_object = hashlib.md5(pattern_str.encode())
        return hash_object.hexdigest()
    

class StatisticalAnalyzer:
    def __init__(self, variance_threshold: float = 0.02):
        self.variance_threshold = variance_threshold

    def analyze_prediction_clusters(
        self,
        predictions: list[MatchPredictionWithMatchData],
        window_size: int = 50
    ) -> dict[str, list[tuple]]:
        """
        Analyze predictions to find statistically suspicious clusters.
        Returns groups of miners that show unnaturally low variance in their differences.
        """
        # Group predictions by match
        matches = defaultdict(list)
        for pred in predictions:
            matches[pred.prediction.matchId].append(pred.prediction)

        # Analyze variance between miners for each match
        #suspicious_groups = defaultdict(list)
        
        # Track miner relationships over time
        miner_relationships = defaultdict(lambda: defaultdict(list))

        for match_id, match_predictions in matches.items():
            if len(match_predictions) < 2:
                continue

            # Calculate all pairwise differences for this match
            for i in range(len(match_predictions)):
                for j in range(i + 1, len(match_predictions)):
                    pred1 = match_predictions[i]
                    pred2 = match_predictions[j]
                    
                    # Only compare predictions with same choice
                    if pred1.probabilityChoice != pred2.probabilityChoice:
                        continue
                        
                    difference = abs(pred1.probability - pred2.probability)
                    
                    # Track the difference between these miners
                    miner_relationships[pred1.minerId][pred2.minerId].append({
                        'match_id': match_id,
                        'difference': difference,
                        'choice': pred1.probabilityChoice,
                        'prob1': pred1.probability,
                        'prob2': pred2.probability
                    })

        # Analyze the relationships for consistent patterns
        suspicious_patterns = self._analyze_miner_relationships(miner_relationships)
        
        return suspicious_patterns

    def _analyze_miner_relationships(
        self,
        relationships: dict
    ) -> dict[str, list[tuple]]:
        """
        Analyze the history of differences between miners to find suspicious patterns.
        """
        suspicious_patterns = {}
        
        for miner1, related_miners in relationships.items():
            for miner2, history in related_miners.items():
                if len(history) < 5:  # Require minimum number of common predictions
                    continue
                    
                differences = [h['difference'] for h in history]
                mean_diff = sum(differences) / len(differences)
                
                # Calculate variance of differences
                variance = sum((d - mean_diff) ** 2 for d in differences) / len(differences)
                
                # If variance is suspiciously low and mean difference is small
                if variance < self.variance_threshold and mean_diff < 0.05:
                    key = f"cluster_{miner1}_{miner2}"
                    suspicious_patterns[key] = {
                        'miners': [miner1, miner2],
                        'mean_difference': mean_diff,
                        'variance': variance,
                        'num_matches': len(history),
                        'history': history
                    }
                    
        return suspicious_patterns
