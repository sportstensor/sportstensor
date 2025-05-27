from typing import List, Set, Dict, Tuple
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import bittensor as bt
from tabulate import tabulate

from common.data import League, MatchPredictionWithMatchData
from common.constants import (
    EXACT_MATCH_PREDICTIONS_THRESHOLD,
    SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD,
    LEAGUES_ALLOWING_DRAWS
)

class CopycatDetectionControllerV2:
    def __init__(
        self,
        min_sample_size: int = 100,
        max_sample_size: int = 500, # Maximum number of most recent shared predictions to analyze.
        prob_correlation_weight: float = 0.4,
        choice_agreement_weight: float = 0.6,
        obviousness_threshold: float = 1.2, # Odds threshold for obvious predictions == ~83.33% implied probability
        min_non_obvious_predictions: int = 30
    ):
        """
        Initialize the copycat detection controller.
        
        Args:
            min_sample_size: Minimum number of shared predictions required for analysis
            prob_correlation_weight: Weight for probability correlation in similarity score
            choice_agreement_weight: Weight for choice agreement in similarity score
            obviousness_threshold: Threshold for considering a prediction obvious
            min_non_obvious_predictions: Minimum number of non-obvious predictions required
        """
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size
        self.prob_correlation_weight = prob_correlation_weight
        self.choice_agreement_weight = choice_agreement_weight
        self.obviousness_threshold = obviousness_threshold
        self.min_non_obvious_predictions = min_non_obvious_predictions

    def analyze_league(
        self,
        league: League,
        league_predictions: List[MatchPredictionWithMatchData] = None
    ) -> tuple[Set[int], Set[int]]:
        """
        Analyze a specific league for copycat patterns using correlation analysis.
        
        Args:
            league: League to analyze
            league_predictions: List of predictions with match data to analyze
            
        Returns:
            Tuple of (suspicious_miner_ids, miners_to_penalize)
        """
        bt.logging.info(f"Analyzing {len(league_predictions)} league predictions for copycat patterns: {league.name}")

        if not league_predictions:
            bt.logging.warning(f"No predictions found for {league.name}")
            return set(), set()

        # Validate predictions
        valid_predictions = []
        for pred in league_predictions:
            if pred.prediction.matchId and pred.prediction.minerId is not None:
                valid_predictions.append(pred)
            else:
                bt.logging.warning(f"Found invalid prediction: matchId={pred.prediction.matchId}, minerId={pred.prediction.minerId}")

        bt.logging.info(f"Found {len(valid_predictions)} valid predictions out of {len(league_predictions)} total predictions")

        # Group predictions by miner
        miner_predictions = defaultdict(list)
        for pred in valid_predictions:
            miner_predictions[pred.prediction.minerId].append(pred)

        bt.logging.info(f"Found predictions for {len(miner_predictions)} unique miners")

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(miner_predictions)
        bt.logging.info(f"Calculated correlation matrix with {len(correlation_matrix)} miner pairs")

        # Generate penalties based on correlation scores
        suspicious_miners, miners_to_penalize = self._generate_penalties(correlation_matrix)

        # Print analysis results
        self._print_analysis_results(correlation_matrix, suspicious_miners, miners_to_penalize, league)

        return suspicious_miners, miners_to_penalize

    def _calculate_correlation_matrix(
        self,
        miner_predictions: Dict[int, List[MatchPredictionWithMatchData]]
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Calculate correlation matrix between miners based on their predictions.
        
        Args:
            miner_predictions: Dictionary mapping miner IDs to their predictions
            
        Returns:
            Dictionary mapping miner pairs to their correlation metrics
        """
        correlation_matrix = {}
        miners = list(miner_predictions.keys())

        for i in range(len(miners)):
            for j in range(i + 1, len(miners)):
                miner_i = miners[i]
                miner_j = miners[j]

                # Get shared predictions (now matches by time interval)
                shared_predictions = self._get_shared_predictions_by_interval(
                    miner_predictions[miner_i],
                    miner_predictions[miner_j]
                )

                if len(shared_predictions) < self.min_sample_size:
                    continue

                # Limit to max sample size
                if len(shared_predictions) > self.max_sample_size:
                    shared_predictions = shared_predictions[:self.max_sample_size]

                # Sort by matchId for cleaner grouping
                shared_predictions = sorted(shared_predictions, key=lambda x: x[0].prediction.matchId)

                # Calculate similarity score
                similarity_score, score_components = self._calculate_similarity_score(shared_predictions)

                if similarity_score is not None:
                    correlation_matrix[(miner_i, miner_j)] = {
                        'similarity_score': similarity_score,
                        'num_shared_predictions': len(shared_predictions),
                        'shared_predictions': shared_predictions,
                        'score_components': score_components
                    }

        return correlation_matrix

    def _get_shared_predictions_by_interval(
        self,
        predictions_i: List[MatchPredictionWithMatchData],
        predictions_j: List[MatchPredictionWithMatchData]
    ) -> List[Tuple[MatchPredictionWithMatchData, MatchPredictionWithMatchData]]:
        """
        Get predictions that both miners made for the same matches at the same time intervals.
        
        Args:
            predictions_i: First miner's predictions
            predictions_j: Second miner's predictions
            
        Returns:
            List of tuples containing matching predictions (same match, same time interval)
        """
        
        def _classify_prediction_interval(prediction_date, match_date) -> str:
            """
            Classify a prediction into one of the 4 time intervals based on when it was made
            relative to the match date.
            
            Args:
                prediction_date: When the prediction was made
                match_date: When the match is scheduled
                
            Returns:
                String representing the time interval ('T-24h', 'T-12h', 'T-4h', 'T-10m', or None)
            """
            if prediction_date is None or match_date is None:
                return None
                
            # Calculate time difference in minutes (positive = before match)
            time_diff_minutes = (match_date - prediction_date).total_seconds() / 60
            
            # Only accept predictions made BEFORE the match
            if time_diff_minutes < 0:
                return None
            
            # Classify based on time ranges in minutes
            if 1380 <= time_diff_minutes <= 1440:  # 23h to 24h (1380-1440 minutes)
                return 'T-24h'
            elif 660 <= time_diff_minutes <= 720:   # 11h to 12h (660-720 minutes)
                return 'T-12h'
            elif 180 <= time_diff_minutes <= 240:   # 3h to 4h (180-240 minutes)
                return 'T-4h'
            elif 0 <= time_diff_minutes <= 10:      # 0 to 10 minutes
                return 'T-10m'
            else:
                return None
        
        # Create lookup for second miner's predictions by (matchId, interval)
        pred_j_lookup = {}
        for pred in predictions_j:
            interval = _classify_prediction_interval(pred.prediction.predictionDate, pred.prediction.matchDate)
            if interval:  # Only include predictions that fall within valid intervals
                key = (pred.prediction.matchId, interval)
                pred_j_lookup[key] = pred
        
        # Find matching predictions (same match AND same time interval)
        shared_predictions = []
        for pred_i in predictions_i:
            interval = _classify_prediction_interval(pred_i.prediction.predictionDate, pred_i.prediction.matchDate)
            if interval:  # Only include predictions that fall within valid intervals
                key = (pred_i.prediction.matchId, interval)
                if key in pred_j_lookup:
                    shared_predictions.append((pred_i, pred_j_lookup[key]))
        
        return shared_predictions

    def _calculate_similarity_score(
        self,
        shared_predictions: List[Tuple[MatchPredictionWithMatchData, MatchPredictionWithMatchData]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate similarity score between two miners based on their shared predictions.
        
        Args:
            shared_predictions: List of tuples containing matching predictions
            
        Returns:
            Tuple of (similarity_score, score_components)
        """
        # Extract probabilities and choices
        probs_i = []
        probs_j = []
        probs_favs_i = []
        probs_favs_j = []
        choices_i = []
        choices_j = []
        non_obvious_count = 0
        no_bet_count = 0
        no_bet_threshold = 0.01  # 1% difference threshold for "no bet" detection
        
        for pred_i, pred_j in shared_predictions:
            # Get market implied probability for the predicted outcome
            odds = pred_i.get_closing_odds_for_predicted_outcome()
            if odds is None:
                continue
                
            market_prob = 1 / odds
            
            # Skip if prediction is too close to market probability (no bet)
            if abs(pred_i.prediction.probability - market_prob) < no_bet_threshold:
                no_bet_count += 1
                continue

            # Skip if market odds are too low (e.g., equal/under our obviousness threshold)
            if odds <= self.obviousness_threshold:
                continue

            # Append choices before gathering probabilities
            choices_i.append(pred_i.prediction.probabilityChoice)
            choices_j.append(pred_j.prediction.probabilityChoice)

            # Append if prediction is for market favorite
            probs_favs_i.append(False if pred_i.is_prediction_for_underdog(LEAGUES_ALLOWING_DRAWS) else True)
            probs_favs_j.append(False if pred_j.is_prediction_for_underdog(LEAGUES_ALLOWING_DRAWS) else True)

            # log for verification
            if (pred_i.prediction.minerId == 119 and pred_j.prediction.minerId == 161) or \
               (pred_i.prediction.minerId == 17 and pred_j.prediction.minerId == 22):
                isDiff = " - DIFF" if pred_i.prediction.probabilityChoice != pred_j.prediction.probabilityChoice else ""
                print(f"Match: {pred_i.prediction.matchId}, Miner {pred_i.prediction.minerId} Prob: {pred_i.prediction.probability} ({pred_i.prediction.probabilityChoice} - {pred_i.prediction.predictionDate}), Miner {pred_j.prediction.minerId} Prob: {pred_j.prediction.probability} ({pred_j.prediction.probabilityChoice} - {pred_j.prediction.predictionDate}){isDiff}")

            # Only compare predictions with same choice
            if pred_i.prediction.probabilityChoice != pred_j.prediction.probabilityChoice:
                continue
                
            # Check if prediction is non-obvious
            if odds > self.obviousness_threshold:
                non_obvious_count += 1
                
            probs_i.append(pred_i.prediction.probability)
            probs_j.append(pred_j.prediction.probability)
            

        if len(probs_i) < self.min_sample_size or non_obvious_count < self.min_non_obvious_predictions:
            return None, None

        # Calculate Spearman correlation on probabilities (ranks-based)
        prob_correlation, _ = spearmanr(probs_i, probs_j)
        
        # Handle NaN correlation (happens when all values are identical)
        if np.isnan(prob_correlation):
            prob_correlation = 1.0 if len(set(probs_i)) == 1 and len(set(probs_j)) == 1 else 0.0
        
        # Calculate choice agreement percentage
        choice_agreement = sum(1 for c1, c2 in zip(choices_i, choices_j) if c1 == c2) / len(choices_i)

        # Calculate percentage of favorite predictions
        miner_a_fav_percentage = sum(probs_favs_i) / len(probs_favs_i) if probs_favs_i else 0.0
        miner_b_fav_percentage = sum(probs_favs_j) / len(probs_favs_j) if probs_favs_j else 0.0
        
        # Apply obviousness filter
        filtered_correlation, obviousness_impact = self._apply_obviousness_filter(prob_correlation, shared_predictions)
        
        # Calculate normalized sample size score (capped at 1.0)
        sample_size_score = min(1.0, len(shared_predictions) / (self.min_sample_size * 2))
        
        # Calculate normalized non-obvious ratio
        non_obvious_ratio = non_obvious_count / len(shared_predictions)
        
        # Calculate composite similarity score
        similarity_score = (
            self.prob_correlation_weight * filtered_correlation +
            self.choice_agreement_weight * choice_agreement
        )
        
        # Store score components
        score_components = {
            'prob_correlation': prob_correlation,
            'choice_agreement': choice_agreement,
            'miner_a_fav_percentage': miner_a_fav_percentage,
            'miner_b_fav_percentage': miner_b_fav_percentage,
            'obviousness_impact': obviousness_impact,
            'filtered_correlation': filtered_correlation,
            'similarity_score': similarity_score,
            'non_obvious_ratio': non_obvious_ratio,
            'no_bet_ratio': no_bet_count / len(shared_predictions),
            'sample_size_score': sample_size_score
        }
        
        return similarity_score, score_components

    def _apply_obviousness_filter(
        self,
        correlation: float,
        shared_predictions: List[Tuple[MatchPredictionWithMatchData, MatchPredictionWithMatchData]]
    ) -> Tuple[float, float]:
        """
        Apply obviousness filter to correlation score.
        
        Args:
            correlation: Raw correlation score
            shared_predictions: List of shared predictions
            
        Returns:
            Tuple of (filtered_correlation, obviousness_impact)
        """
        # Count obvious predictions (where odds strongly favor one outcome)
        obvious_count = 0
        for pred_i, _ in shared_predictions:
            odds = pred_i.get_closing_odds_for_predicted_outcome()
            if odds is not None and odds < self.obviousness_threshold:
                obvious_count += 1
        
        # Calculate obviousness impact
        obviousness_ratio = obvious_count / len(shared_predictions)
        obviousness_impact = 0.5 if obviousness_ratio > 0.5 else 1.0
        
        # If most predictions are obvious, reduce correlation score
        filtered_correlation = correlation * obviousness_impact
        
        return filtered_correlation, obviousness_impact

    def _generate_penalties(
        self,
        correlation_matrix: Dict[Tuple[int, int], Dict]
    ) -> Tuple[Set[int], Set[int]]:
        """
        Generate penalties based on correlation scores.
        
        Args:
            correlation_matrix: Dictionary mapping miner pairs to correlation metrics
            
        Returns:
            Tuple of (suspicious_miners, miners_to_penalize)
        """
        suspicious_miners = set()
        miners_to_penalize = set()

        for (miner_i, miner_j), metrics in correlation_matrix.items():
            components = metrics['score_components']
            
            # Only flag if both similarity score and confidence score are high
            if (metrics['similarity_score'] >= 0.9 or
                components['choice_agreement'] >= 0.9):
                
                # Add miners to suspicious miners set
                suspicious_miners.add(miner_i)
                suspicious_miners.add(miner_j)
                
                # Penalize if they have a very high percentage of exact choices
                if components['choice_agreement'] >= 0.9:
                    miners_to_penalize.add(miner_i)
                    miners_to_penalize.add(miner_j)
                # Penalize if they have a very high similarity score
                elif components['similarity_score'] >= 0.9:
                    miners_to_penalize.add(miner_i)
                    miners_to_penalize.add(miner_j)

        return suspicious_miners, miners_to_penalize

    def _print_analysis_results(
        self,
        correlation_matrix: Dict[Tuple[int, int], Dict],
        suspicious_miners: Set[int],
        miners_to_penalize: Set[int],
        league: League
    ) -> None:
        """
        Print analysis results in a readable format.
        
        Args:
            correlation_matrix: Dictionary mapping miner pairs to correlation metrics
            suspicious_miners: Set of suspicious miner IDs
            miners_to_penalize: Set of miners to penalize
            league: League being analyzed
        """
        print(f"\nAnalysis Results for {league.name}:")
        print(f"Total suspicious miners: {len(suspicious_miners)}")
        print(f"Miners to penalize: {len(miners_to_penalize)}")
        
        if correlation_matrix:
            # Show all pairs above similarity threshold
            similarity_threshold = 0.85
            choice_agreement_threshold = 0.85
            print(f"\nMiner pairs with similarity score > {similarity_threshold} and/or choice agreement > {choice_agreement_threshold}:")
            sorted_pairs = sorted(
                correlation_matrix.items(),
                key=lambda x: x[1]['similarity_score'],
                reverse=True
            )
            
            # Group miners by their relationships
            miner_relationships = defaultdict(set)
            
            # Prepare table data
            table_data = []
            for (miner_i, miner_j), metrics in sorted_pairs:
                components = metrics['score_components']
                if metrics['similarity_score'] > similarity_threshold or components['choice_agreement'] > choice_agreement_threshold:
                    
                    miner_relationships[miner_i].add(miner_j)
                    miner_relationships[miner_j].add(miner_i)
                    
                    table_data.append([
                        f"{miner_i}, {miner_j}",
                        metrics['num_shared_predictions'],
                        f"{components['prob_correlation']:.3f}",
                        f"{components['choice_agreement']:.3f}",
                        f"{components['miner_a_fav_percentage']:.3f}",
                        f"{components['miner_b_fav_percentage']:.3f}",
                        f"{components['non_obvious_ratio']:.3f}",
                        f"{components['no_bet_ratio']:.3f}",
                        f"{components['sample_size_score']:.3f}",
                        f"{components['similarity_score']:.3f}"
                    ])
            
            # Print correlation table
            if table_data:
                headers = [
                    "Miner Pair",
                    "Shared Preds",
                    "Spearman Corr",
                    "Choice Agree",
                    "Miner A Fav %",
                    "Miner B Fav %",
                    "Non-Obv Ratio",
                    "No-Bet Ratio",
                    "Sample Score",
                    "Final Score"
                ]
                print(tabulate(
                    table_data,
                    headers=headers,
                    tablefmt="grid",
                    numalign="right",
                    stralign="left"
                ))
            
            # Print relationship groups
            print("\nSuspicious Miner Groups:")
            processed_miners = set()
            for miner, related in miner_relationships.items():
                if miner not in processed_miners:
                    group = {miner} | related
                    processed_miners.update(group)
                    print(f"Group: {', '.join(str(m) for m in sorted(group))}")