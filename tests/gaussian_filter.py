import numpy as np

ODDS = [1.04, 1.91, 2.5, 10]
PROBABILITIES = [1.0, 0.999, 0.99, 0.9, 0.8, 0.5, 0.2857, 0.3333, 0.1, 0.05, 0.01, 0.001, 0.0001, .00001]

def main():
    results = []
    for odds in ODDS:
        for prob in PROBABILITIES:
            gfilter = apply_gaussian_filter(odds, prob)
            implied_edge = odds - 1/prob
            w = (odds - 1.0) * np.log(odds) / 2
            results.append({
                "odds": odds,
                "prob": prob,
                "gfilter": gfilter,
                "implied edge": implied_edge,
                "w": w
            })
    
    for result in results:
        print(result)

def apply_gaussian_filter(odds, prob):
    sigma = np.log(1/(prob*prob))
    w = (odds - 1.0) * np.log(odds) / 2
    diff = abs(odds - 1/prob)
    
    exp_component = 1.0 if diff <= w else np.exp(-np.power(diff, 2) / (4 * np.power(sigma, 2)))
    
    return exp_component

if __name__ == "__main__":
    main()