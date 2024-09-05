import math

# Global constants
THRESHOLD_PREDICTIONS = 100  # n0
SENSITIVITY_ALPHA = 0.05     # alpha
TRANSITION_KAPPA = 1.5       # kappa
EXTREMIS_BETA = 0.1          # beta

def calculate_edge(team, bet_time, match_start_time, model_win_prob, consensus_win_prob, actual_winner):
    """
    Calculate the edge in expected return for a bet on a two-sided market.
    
    :param team: str, either 'A' or 'B', representing the team chosen
    :param bet_time: float, time of the bet
    :param match_start_time: float, time of the start of the match
    :param model_win_prob: function, weak learner's probability of winning for the chosen team
    :param consensus_win_prob: function, consensus probability of winning for the chosen team
    :param actual_winner: str, either 'A' or 'B', representing the team that actually won
    :return: float, the calculated edge
    """
    if bet_time >= match_start_time:
        raise ValueError("Bet time must be before the match start time")
    
    model_prediction_correct = (team == actual_winner)
    reward_punishment = 1 if model_prediction_correct else -1
    edge = (1 / model_win_prob(bet_time)) - consensus_win_prob(match_start_time)
    result = reward_punishment * edge
    
    return result

def calculate_incentive_score(n, n0=THRESHOLD_PREDICTIONS, alpha=SENSITIVITY_ALPHA):
    """
    Calculate the incentive score based on the number of predictions.
    """
    exponent = -alpha * (n - n0)
    denominator = 1 + math.exp(exponent)
    return 1 / denominator

def calculate_advanced_incentive(delta_t, z, clv, kappa=TRANSITION_KAPPA, beta=EXTREMIS_BETA):
    """
    Calculate the advanced incentive score considering time differential and closing line value.
    """
    time_component = math.exp(-delta_t)
    clv_component = (1 - beta) / (1 + math.exp(-kappa * clv)) + beta
    incentive_score = z * time_component + (1 - time_component) * clv_component
    return incentive_score

def calculate_combined_incentive_score(n, predictions, n0=THRESHOLD_PREDICTIONS, alpha=SENSITIVITY_ALPHA):
    """
    Calculate the combined incentive score for a series of predictions.
    """
    rho = calculate_incentive_score(n, n0, alpha)
    
    total_score = 0
    for i, pred in enumerate(predictions, start=1):
        if i > n:
            break
        
        v = calculate_advanced_incentive(
            pred['delta_t'], pred['z'], pred['clv'],
            kappa=pred.get('kappa', TRANSITION_KAPPA), 
            beta=pred.get('beta', EXTREMIS_BETA)
        )
        total_score += v * pred['sigma']
    
    return rho * total_score

if __name__ == "__main__":
    # Test cases
    edge = calculate_edge(team='A', bet_time=0.5, match_start_time=1.0, 
                          model_win_prob=lambda x: 0.6, consensus_win_prob=lambda x: 0.5, 
                          actual_winner='A')
    print(f"Edge: {edge:.4f}")

    score = calculate_incentive_score(n=50)
    print(f"Incentive score: {score:.4f}")

    score = calculate_advanced_incentive(delta_t=2.0, z=0.7, clv=0.8)
    print(f"Advanced incentive score: {score:.4f}")

    predictions = [
        {'delta_t': 2.0, 'z': 0.7, 'clv': 0.8, 'sigma': 1.2},
        {'delta_t': 1.5, 'z': 0.8, 'clv': 0.7, 'sigma': 1.1},
        {'delta_t': 1.0, 'z': 0.9, 'clv': 0.6, 'sigma': 1.0},
        {'delta_t': 0.5, 'z': 1.0, 'clv': 0.5, 'sigma': 0.9},
        {'delta_t': 0.0, 'z': 1.1, 'clv': 0.4, 'sigma': 0.8},
    ]
    final_score = calculate_combined_incentive_score(n=len(predictions), predictions=predictions)
    print(f"Final combined incentive score: {final_score:.4f}")