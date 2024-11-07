import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

"""
This is a test that gives two miners the same scores for the same leagues where they only differ in their rho values.
We will compare how the old mechanism and its usage of rho is impacting the scores versus how the new mechanism and its
usage of rho, after pareto distribution, is impacting the scores.

Each 'miner' will have a league 1 (L1) score of 100 and a league 2 (L2) score of 150.
These scores are the assumed products of v * sigma * gfilter because that product is not in question.
"""

LEAGUE_WEIGHTS = {'League1': 0.4, 'League2': 0.6}  # Arbitrary weights
PARETO_ALPHA = 1.0
PARETO_MU = 1.0
M1_L1_RHO = 0.73
M1_L2_RHO = 0.84
M2_L1_RHO = 0.25
M2_L2_RHO = 0.41
L1_SCORE = 100 # This will be shared by both 'miners' so we can see how rho + old vs new method impacts
L2_SCORE = 150 # This will be shared by both 'miners' so we can see how rho + old vs new method impacts
ALL_UIDS = [0, 1]

def apply_pareto(all_scores: List[float], all_uids: List[int], mu: float, alpha: int) -> List[float]:
    """
    Apply a Pareto distribution to the scores.

    :param all_scores: List of scores to apply the Pareto distribution to
    :param all_uids: List of UIDs corresponding to the scores
    :param mu: Minimum value for the Pareto distribution
    :param alpha: Shape parameter for the Pareto distribution
    :return: List of scores after applying the Pareto distribution
    """
    scores_array = np.array(all_scores)
    
    # Treat all non-positive scores as zero
    positive_mask = scores_array > 0
    positive_scores = scores_array[positive_mask]
    
    transformed_scores = np.zeros_like(scores_array, dtype=float)
    
    if len(positive_scores) > 0:
        # Transform positive scores
        range_transformed = (positive_scores - np.min(positive_scores)) + 1
        transformed_positive = mu * np.power(range_transformed, alpha)
        transformed_scores[positive_mask] = transformed_positive
    
    return transformed_scores

def simulate_old_mechanism():
    
    # In old mechanism, rho is applied immediately to league scores before going into pareto distribution.
    
    # Miner 1 scores
    m1_l1_score = L1_SCORE * M1_L1_RHO  # Score * rho for League 1
    m1_l2_score = L2_SCORE * M1_L2_RHO  # Score * rho for League 2
    m1_total_score = (m1_l1_score * LEAGUE_WEIGHTS['League1'] + 
                  m1_l2_score * LEAGUE_WEIGHTS['League2'])
    
    # Miner 2 scores
    m2_l1_score = L1_SCORE * M2_L1_RHO  # Score * rho for League 1
    m2_l2_score = L2_SCORE * M2_L2_RHO  # Score * rho for League 2
    m2_total_score = (m2_l1_score * LEAGUE_WEIGHTS['League1'] + 
                  m2_l2_score * LEAGUE_WEIGHTS['League2'])
    
    print(f"OLD SCORES GOING INTO PARETO: miner1: {m1_total_score}, miner2: {m2_total_score}")
    # Just use the two scores
    all_scores = np.array([m1_total_score, m2_total_score])
    
    pareto_scores = apply_pareto(all_scores, ALL_UIDS, PARETO_MU, PARETO_ALPHA)
    
    return pareto_scores[0], pareto_scores[1]

def simulate_new_mechanism():
    # In new mechanism, rho is applied after Pareto distribution
    
    # Initial total scores without rho
    m1_league_weighted = (L1_SCORE * LEAGUE_WEIGHTS['League1'] + 
                  L2_SCORE * LEAGUE_WEIGHTS['League2'])
    
    m2_league_weighted = (L1_SCORE * LEAGUE_WEIGHTS['League1'] + 
                  L2_SCORE * LEAGUE_WEIGHTS['League2'])
    
    print(f"NEW SCORES GOING INTO PARETO: miner1: {m1_league_weighted}, miner2: {m2_league_weighted}")

    # Just use the two scores
    all_scores = np.array([m1_league_weighted, m2_league_weighted])
    
    pareto_scores = apply_pareto(all_scores, ALL_UIDS, PARETO_MU, PARETO_ALPHA)
    
    # Apply rho after Pareto. To do so, we will unwrap each leagues score from the total score.
    m1_with_pareto_and_rho_applied = (pareto_scores[0] * L1_SCORE * M1_L1_RHO) + (pareto_scores[0] * L2_SCORE * M1_L2_RHO)
    m2_with_pareto_and_rho_applied = (pareto_scores[1] * L1_SCORE * M2_L1_RHO) + (pareto_scores[0] * L2_SCORE * M2_L2_RHO)

    return m1_with_pareto_and_rho_applied, m2_with_pareto_and_rho_applied

print(f"""
Miner 1 League 1 rho = {M1_L1_RHO}, Miner 1 League 2 rho = {M1_L2_RHO}
Miner 2 League 1 rho = {M2_L1_RHO}, Miner 2 League 2 rho = {M2_L2_RHO}
Both Miners using same league performance scores. 
""")

# Run simulations
old_m1, old_m2 = simulate_old_mechanism()
new_m1, new_m2 = simulate_new_mechanism()

# Create comparison table
comparison = pd.DataFrame({
    'Miner': ['Miner 1', 'Miner 2'],
    'Old Mechanism Score': [old_m1, old_m2],
    'New Mechanism Score': [new_m1, new_m2],
    'Difference': [new_m1 - old_m1, new_m2 - old_m2],
    'Percentage Change': [(new_m1/old_m1 - 1)*100, (new_m2/old_m2 - 1)*100]
})

print("\nScoring Mechanism Comparison:")
print(comparison.to_string(float_format=lambda x: '{:.6f}'.format(x)))