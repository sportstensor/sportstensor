import numpy as np

def apply_pareto(all_scores, all_uids, mu, alpha):
    scores_array = np.array(all_scores)
    
    # Treat all non-positive scores less than 1 as zero
    positive_mask = scores_array >= 1
    positive_scores = scores_array[positive_mask]
    
    transformed_scores = np.zeros_like(scores_array, dtype=float)
    
    if len(positive_scores) > 0:
        # Transform positive scores
        transformed_positive = mu * np.power(positive_scores, alpha)
        transformed_scores[positive_mask] = transformed_positive
    
    return transformed_scores, all_uids

# Example usage
#all_uids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#all_scores = [17, 12, -1, 38, 0, 0.000001, 0.04, 1, 1.2, 0.007]

all_uids = list(range(73))  # 0 to 72, matching the UIDs in the log
all_scores = [
    -1, 21.6962, -0.556654, -0.994077, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 6.80035, -1, -1, -1, -1, -1, -1,
    -1, -1.07155, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 7.58157, -1, -1, 13.1217, -1, -1, 35.0172, -1, -1,
    -1, -1, -1, -1, -1, 81.4527, -1, -1, 21.7832, 20.602,
    21.5432, -1, -1
]

mu = 1
alpha = 0.9

transformed_scores, uids = apply_pareto(all_scores, all_uids, mu, alpha)

print("Original scores:", all_scores)
print("Transformed scores:", transformed_scores)
print("UIDs:", uids)

# Additional information
for uid, original, transformed in zip(uids, all_scores, transformed_scores):
    print(f"UID: {uid}, Original: {original}, Transformed: {transformed:.2f}")