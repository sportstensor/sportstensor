import numpy as np

def apply_pareto(all_scores, all_uids, mu, alpha):
    scores_array = np.array(all_scores)
    
    # Treat all non-positive scores less than 1 as zero
    positive_mask = scores_array > 0
    positive_scores = scores_array[positive_mask]
    
    transformed_scores = np.zeros_like(scores_array, dtype=float)

    print(f"Positive scores: {positive_scores}")
    print(f"Min: {min(positive_scores)}")
    range_transformed = (positive_scores - np.min(positive_scores)) + 1
    print(f"Range transformed: {range_transformed}")
    
    if len(positive_scores) > 0:
        # Transform positive scores
        print(f"Power: {np.power(range_transformed, alpha)}")
        transformed_positive = mu * np.power(range_transformed, alpha)
        #transformed_positive = mu * np.power(positive_scores, alpha)
        transformed_scores[positive_mask] = transformed_positive
    
    return transformed_scores, all_uids

# Example usage
#all_uids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#all_scores = [17, 12, -1, 38, 0, 0.000001, 0.04, 1, 1.2, 0.007]

all_uids = list(range(73))  # 0 to 72, matching the UIDs in the log
all_scores = [
    -3, -2.5231, -2.91235, -2.99362, -3, 0.24, -3, -3, -3, -3,
    -3, -3, -3, -2.0793, -3, -3, -3, -3, -3, -3,
    -3, 0.0296598, -3, -3, -3, -3, -3, -3, -3, -3,
    -3, -3, -3, -3, -3, -3, -3, 0, -3, -3,
    -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
    -3, -2.52537, -3, -3, -2.32464, -3, -3, 2.13207, -3, -3,
    -3, -3, -3, -3, -3, 1.38769, -3, -3, -1.93823, -1.63695,
    1.12741, -3, -3, -3, -3
]

mu = 1
alpha = 1.0

transformed_scores, uids = apply_pareto(all_scores, all_uids, mu, alpha)

print("Original scores:", all_scores)
print("Transformed scores:", transformed_scores)
print("UIDs:", uids)

# Additional information
for uid, original, transformed in zip(uids, all_scores, transformed_scores):
    print(f"UID: {uid}, Original: {original}, Transformed: {transformed:.2f}")