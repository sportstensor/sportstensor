import math
from collections import defaultdict
from typing import List
from datetime import datetime, timezone

from common.data import MatchPredictionWithMatchData
from common.constants import COPYCAT_VARIANCE_THRESHOLD, EXACT_MATCH_PREDICTIONS_THRESHOLD, SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD
  

class StatisticalAnalyzer:
    def __init__(self, variance_threshold: float = COPYCAT_VARIANCE_THRESHOLD, min_suspicious_matches: int = EXACT_MATCH_PREDICTIONS_THRESHOLD):
        self.variance_threshold = variance_threshold
        self.min_suspicious_matches = min_suspicious_matches

    def analyze_prediction_clusters(
        self,
        predictions: list[MatchPredictionWithMatchData],
        ordered_matches: List[tuple[str, datetime]],
        excluded_miners: set[int] = None
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

                    # Ensure pred1.predictionDate is offset-aware
                    if pred1.predictionDate.tzinfo is None:
                        pred1_predictionDate = pred1.predictionDate.replace(tzinfo=timezone.utc)
                    else:
                        pred1_predictionDate = pred1.predictionDate

                    # Ensure pred2.predictionDate is offset-aware
                    if pred2.predictionDate.tzinfo is None:
                        pred2_predictionDate = pred2.predictionDate.replace(tzinfo=timezone.utc)
                    else:
                        pred2_predictionDate = pred2.predictionDate

                    # Only compare if predictions are within 1 hour of each other
                    if abs((pred1_predictionDate - pred2_predictionDate).total_seconds()) > 3600:
                        continue
                    
                    absolute_difference = round(abs(pred1.probability - pred2.probability), 4)
                    difference = round(math.exp(-(pred1.probability*100 - pred2.probability*100) ** 2), 2)
                    
                    if difference < self.variance_threshold:
                        continue
                    
                    # Track the difference between these miners
                    miner_relationships[pred1.minerId][pred2.minerId].append({
                        'match_id': match_id,
                        'match_date': pred1.matchDate,
                        'difference': difference,
                        'absolute_difference': absolute_difference,
                        'choice': pred1.probabilityChoice,
                        'prob1': pred1.probability,
                        'prob2': pred2.probability,
                        'pred1_date': pred1_predictionDate.strftime('%Y-%m-%d %H:%M:%S'),
                        'pred2_date': pred2_predictionDate.strftime('%Y-%m-%d %H:%M:%S')
                    })

        # Analyze the relationships
        suspicious_relationships = self.analyze_miner_relationships(miner_relationships)

        # Add consecutive match analysis
        consecutive_patterns = self.analyze_consecutive_matches(suspicious_relationships, ordered_matches)
        
        # Add consecutive pattern information to suspicious relationships
        for key in suspicious_relationships:
            if key in consecutive_patterns:
                suspicious_relationships[key]['consecutive_patterns'] = consecutive_patterns[key]
        
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
                predictions_per_match = round(len(history) / len(suspicious_matches), 2)

                # Calculate the number of matches with exact predictions
                num_matches_with_exact = len(set(h['match_id'] for h in history if h['absolute_difference'] == 0))
                
                suspicious_relationships[relationship_key] = {
                    'miners': sorted([miner1, miner2]),  # Sort for consistency
                    'num_matches': len(suspicious_matches),
                    'num_predictions': len(history),
                    'predictions_per_match': predictions_per_match,
                    'match_ids': list(suspicious_matches),  # Store match IDs for reference
                    'num_matches_with_exact': num_matches_with_exact,
                    'num_exact_predictions': sum(1 for h in history if h['absolute_difference'] == 0),
                    'history': history
                }
                        
        return suspicious_relationships
    
    def analyze_consecutive_matches(
        self,
        relationships: dict,
        ordered_matches: List[tuple[str, datetime]]
    ) -> dict[str, dict]:
        """
        Analyze relationships to find consecutive matches with similar predictions.
        
        Args:
            relationships: Dictionary of miner relationships from analyze_miner_relationships
            ordered_matches: List of (match_id, match_date) tuples in chronological order
            
        Returns:
            Dictionary mapping miner pairs to their consecutive match statistics
        """
        consecutive_patterns = {}
        
        if not ordered_matches or len(ordered_matches) < 2:
            return consecutive_patterns
        
        for key, relationship in relationships.items():
            # Create a mapping of match_id to prediction details
            match_predictions = defaultdict(list)
            for pred in relationship['history']:
                # Skip if we've already analyzed this match
                if pred['match_id'] not in match_predictions:
                    match_predictions[pred['match_id']].append(pred)
            
            current_streak = 0
            max_streak = 0
            streak_details = []
            current_streak_details = []
            last_match_index = None
            
            # Iterate through matches in chronological order
            for match_idx, (match_id, _) in enumerate(ordered_matches):
                if match_id not in match_predictions:
                    # If we hit a match without predictions, check if we need to save the current streak
                    if current_streak >= SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD:
                        streak_details.append(current_streak_details)
                    current_streak = 0
                    current_streak_details = []
                    continue
                
                # Check if this match is consecutive with the last one
                if last_match_index is not None and match_idx != last_match_index + 1:
                    # Break in consecutive matches
                    if current_streak >= SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD:
                        streak_details.append(current_streak_details)
                    current_streak = 0
                    current_streak_details = []
                
                # Add current match to streak
                current_streak += 1
                current_streak_details.extend(match_predictions[match_id])
                max_streak = max(max_streak, current_streak)
                last_match_index = match_idx
            
            # Check final streak
            if current_streak >= SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD:
                streak_details.append(current_streak_details)
            
            if max_streak >= SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD:  # Only store if we found meaningful streaks
                consecutive_patterns[key] = {
                    'miners': relationship['miners'],
                    'max_consecutive': max_streak,
                    'total_matches': len(match_predictions),
                    'num_streaks': len(streak_details),
                    'streak_details': streak_details,
                    'exact_matches_in_streaks': sum(
                        1 for streak in streak_details 
                        for match in streak if match['difference'] == 0
                    ),
                    'average_streak_length': (
                        sum(len(streak) for streak in streak_details) / len(streak_details)
                        if streak_details else 0
                    ),
                    'streak_match_ids': [
                        [match['match_id'] for match in streak]
                        for streak in streak_details
                    ]
                }
                
                """
                print(
                    f"Found consecutive pattern for miners {relationship['miners']}:\n"
                    f"Max consecutive matches: {max_streak}\n"
                    f"Number of streaks: {len(streak_details)}\n"
                    f"Streak matches: {consecutive_patterns[key]['streak_match_ids']}"
                )
                """
        
        return consecutive_patterns
