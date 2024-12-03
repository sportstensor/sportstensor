import numpy as np
import math
import matplotlib.pyplot as plt

def compute_significance_score(num_miner_predictions: int, num_threshold_predictions: int, alpha: float) -> float:
    """
    Based on the number of predictions, calculate the statistical signifigance score.

    :param num_miner_predictions: int, the number of predictions made by the miner
    :param num_threshold_predictions: int, the number of predictions made by the threshold
    :param alpha: float, the sensitivity alpha
    :return: float, the calculated significance score
    """
    exponent = -alpha * (num_miner_predictions - num_threshold_predictions)
    denominator = 1 + math.exp(exponent)
    return 1 / denominator


# Define variables
alpha_new = 0.01
alpha_new_num_threshold_predictions = 75
alpha_old = 0.025
alpha_old_num_threshold_predictions = 250
num_miner_predictions_range = np.linspace(0, 1500, 1500)

# Calculate scores for both alpha values
scores_alpha_new = [compute_significance_score(n, alpha_new_num_threshold_predictions, alpha_new) 
                    for n in num_miner_predictions_range]
scores_alpha_old = [compute_significance_score(n, alpha_old_num_threshold_predictions, alpha_old) 
                    for n in num_miner_predictions_range]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_miner_predictions_range, scores_alpha_new, color='blue', label="New Constants (alpha, threshold) = 0.01/75")
plt.axvline(x=alpha_new_num_threshold_predictions, color='blue', linestyle='--')

# Fainter color for alpha_old variables
plt.plot(num_miner_predictions_range, scores_alpha_old, color='grey', alpha=0.25, label="Old Constants (alpha, threshold) = 0.025/250")
plt.axvline(x=alpha_old_num_threshold_predictions, color='grey', linestyle='--', alpha=0.25)

plt.xlabel("Number of Miner Predictions")
plt.ylabel("Significance Score")
plt.title("Significance Score vs. Number of Miner Predictions")
plt.legend()
plt.grid(True)
plt.show()