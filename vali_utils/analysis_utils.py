import json
import hashlib
from collections import defaultdict
from typing import List, Tuple
import datetime as dt

from common.data import MatchPredictionWithMatchData
from common.constants import COPYCAT_VARIANCE_THRESHOLD, EXACT_MATCH_PREDICTIONS_THRESHOLD
  

class StatisticalAnalyzer:
    def __init__(self, variance_threshold: float = COPYCAT_VARIANCE_THRESHOLD, min_suspicious_matches: int = EXACT_MATCH_PREDICTIONS_THRESHOLD):
        self.variance_threshold = variance_threshold
        self.min_suspicious_matches = min_suspicious_matches

    def analyze_prediction_clusters(
        self,
        predictions: list[MatchPredictionWithMatchData],
        excluded_miners: set[int] = None,
        window_size: int = 50
    ) -> dict[str, list[tuple]]:
        """
        Analyze predictions to find statistically suspicious clusters.
        Returns groups of miners that show unnaturally low variance in their differences.
        """
        # Group predictions by match
        matches = defaultdict(list)
        for pred in predictions:
            # Skip excluded miners
            if pred.prediction.minerId in excluded_miners:
                continue
            matches[pred.prediction.matchId].append(pred.prediction)

        # Track miner relationships over time
        miner_relationships = defaultdict(lambda: defaultdict(list))

        for match_id, match_predictions in matches.items():
            if len(match_predictions) < 2:
                continue

            # Calculate all pairwise differences for this match
            for i in range(len(match_predictions)):
                for j in range(i + 1, len(match_predictions)):
                    # Skip comparisons between the same miner
                    if match_predictions[i].minerId == match_predictions[j].minerId:
                        continue

                    pred1 = match_predictions[i]
                    pred2 = match_predictions[j]
                    
                    # Only compare predictions with same choice
                    if pred1.probabilityChoice != pred2.probabilityChoice:
                        continue
                    
                    difference = abs(pred1.probability - pred2.probability)
                    if difference > self.variance_threshold:
                        continue
                    
                    # Track the difference between these miners
                    miner_relationships[pred1.minerId][pred2.minerId].append({
                        'match_id': match_id,
                        'difference': difference,
                        'choice': pred1.probabilityChoice,
                        'prob1': pred1.probability,
                        'prob2': pred2.probability
                    })

        # Analyze the relationships
        suspicious_relationships = self.analyze_miner_relationships(miner_relationships)
        
        # print suspicious relationships, skipping history
        #for key, value in suspicious_relationships.items():
            #print(f"Miners: {value['miners']}, Matches: {value['num_matches']}, Predictions: {value['num_predictions']}, Predictions per match: {value['predictions_per_match']}, Exact predictions: {value['num_exact_predictions']}")
        
        return suspicious_relationships
    
    def analyze_miner_relationships(
        self,
        relationships: dict
    ) -> dict[str, list[tuple]]:
        """
        Analyze the history of differences between miners to find suspicious patterns.
        
        Args:
            relationships: Dictionary mapping miner pairs to their prediction history
            
        Returns:
            Dictionary of suspicious relationships with details about matching predictions
        """
        suspicious_relationships = {}
        
        for miner1, related_miners in relationships.items():
            for miner2, history in related_miners.items():
                # Skip if we've already analyzed this pair
                relationship_key = '_'.join(sorted([str(miner1), str(miner2)]))
                if relationship_key in suspicious_relationships:
                    continue
                
                # Get unique matches where predictions are similar
                suspicious_matches = set(h['match_id'] for h in history)
                if len(suspicious_matches) < self.min_suspicious_matches:
                    continue
                
                # Calculate predictions per match for additional context
                predictions_per_match = len(history) / len(suspicious_matches)

                # Calculate the number of matches with exact predictions
                num_matches_with_exact = len(set(h['match_id'] for h in history if h['difference'] == 0))
                
                suspicious_relationships[relationship_key] = {
                    'miners': sorted([miner1, miner2]),  # Sort for consistency
                    'num_matches': len(suspicious_matches),
                    'num_predictions': len(history),
                    'predictions_per_match': predictions_per_match,
                    'match_ids': list(suspicious_matches),  # Store match IDs for reference
                    'num_matches_with_exact': num_matches_with_exact,
                    'num_exact_predictions': sum(1 for h in history if h['difference'] == 0),
                    'history': history
                }
                        
        return suspicious_relationships
