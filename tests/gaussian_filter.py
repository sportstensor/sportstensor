import numpy as np
import pandas as pd
from tabulate import tabulate

#ODDS = [1.65, 1.75, 1.79]
#PROBABILITIES = [0.999, 0.99, 0.9, 0.8, 0.7, 0.671, 0.62, 0.6, 0.57, 0.5, 0.45, 0.41, 0.37, 0.35, 0.3333, 0.2857, 0.25, 0.22, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, .00001]

ODDS = [14.84, 13.45, 10.22, 9.01, 6.03, 5.04, 4.59, 4.167, 3.89, 3.31, 3.24, 2.31, 1.75, 1.6, 1.44, 1.4, 1.25, 1.15]
PROBABILITIES = [0.001, 0.01, 0.0467, 0.0716, 0.0844, 0.0945, 0.10, 0.111, 0.1214, 0.1343, 0.18, 0.2, 0.225, 0.267, 0.3, 0.33, 0.41, 0.465, 0.52, 0.54, 0.6145, 0.669, 0.71, 0.8, 0.88, 0.91, 0.94, 0.97, 0.99]

def run_filter_comparison(odds_array, probabilities_array, filter_versions):
    """
    Run multiple versions of gaussian filters and compare results with grouped columns
    
    Parameters:
    odds_array: array of odds values to test
    probabilities_array: array of probability values to test
    filter_versions: dict of filter functions to compare, e.g. {'v1': apply_gaussian_filter, 'v2': apply_gaussian_filter_v2}
    """
    results = []
    
    for odds in odds_array:
        for prob in probabilities_array:
            # Calculate base metrics that don't depend on filter version
            wp_odds = 1/prob
            market_prob = 1/odds
            implied_edge = odds - 1/prob
            
            # Create base result dictionary with input parameters
            result = {
                "odds": odds,
                "wp odds": wp_odds,
                "prob": market_prob,
                "wp": prob,
                "implied_edge": implied_edge,
                "is_lay": "lay" if 1/prob > odds else ""
            }
            
            # Create grouped dictionaries for filter results
            filter_results = {}
            edge_results = {}
            
            # Run each filter version and group results
            for version_name, filter_func in filter_versions.items():
                gfilter = filter_func(odds, prob)
                # If implied edge is negative, the prediction is incorrect. apply penalty by setting gfilter to 1
                """
                if implied_edge < 0 and round(gfilter, 2) > 0 and gfilter < 1:
                    final_edge = implied_edge * (1)
                    filter_results[f"gfilter_{version_name}"] = str(round(gfilter, 2)) + " (1.0)"
                else:
                    final_edge = implied_edge * gfilter
                    filter_results[f"gfilter_{version_name}"] = round(gfilter, 2)
                """
                final_edge = implied_edge * gfilter
                filter_results[f"gfilter_{version_name}"] = round(gfilter, 2)
                edge_results[f"final_edge_{version_name}"] = final_edge
            
            # Update result dictionary with grouped metrics
            result.update(filter_results)
            result.update(edge_results)
            
            #if result['is_lay'] == 'lay':
                #results.append(result)
            results.append(result)
    
    # Convert to DataFrame and organize columns
    df = pd.DataFrame(results)
    
    # Reorder columns to group metrics together
    base_cols = ["odds", "wp odds", "prob", "wp", "implied_edge"]
    filter_cols = [col for col in df.columns if col.startswith("gfilter_")]
    edge_cols = [col for col in df.columns if col.startswith("final_edge_")]
    lay_col = ["is_lay"]
    
    # Combine columns in desired order
    ordered_cols = base_cols + filter_cols + edge_cols + lay_col
    
    return df[ordered_cols]

def display_comparison(df, float_format=".4f"):
    """
    Display the comparison results in a formatted table
    """
    # Sort by odds and probability for better readability
    df_sorted = df.sort_values(['odds', 'prob'])
    
    # Format the table
    table = tabulate(
        df_sorted,
        headers='keys',
        tablefmt='pipe',
        floatfmt=float_format,
        showindex=False
    )
    
    print(table)

# Example usage:
def main():
    # Define filter versions to compare
    filter_versions = {
        'v1': apply_gaussian_filter,
        #'v2': apply_gaussian_filter_v2,
        #'v3': apply_gaussian_filter_v3,
        #'v4': apply_gaussian_filter_v4
    }
    
    # Run comparison
    results_df = run_filter_comparison(ODDS, PROBABILITIES, filter_versions)
    
    # Display results
    display_comparison(results_df)
    
    # Optionally save to CSV
    # results_df.to_csv('filter_comparison.csv', index=False)
    
    return results_df  # Return DataFrame for further analysis if needed

def main_old():
    results = []
    for odds in ODDS:
        for prob in PROBABILITIES:
            gfilter = apply_gaussian_filter_v2(odds, prob)
            implied_edge = odds - 1/prob
            final_edge = implied_edge * gfilter
            #w = (odds - 1.0) * np.log(odds) / 2
            results.append({
                "odds": odds,
                "prob": prob,
                "gfilter": gfilter,
                "implied edge": implied_edge,
                'final edge': final_edge
                #"w": w
            })
    
    for result in results:
        #if result['final edge'] > 3 or result['final edge'] < -3:
            #print(result)
        print(result)

""" Deprecated original v1 of gaussian filter 
def apply_gaussian_filter(odds, prob):
    sigma = np.log(1/(prob*prob))
    w = (odds - 1.0) * np.log(odds) / 2
    diff = abs(odds - 1/prob)
    exp_component = 1.0 if diff <= w else np.exp(-np.power(diff, 2) / (4 * np.power(sigma, 2)))
    return exp_component
"""

def apply_gaussian_filter(odds, prob):
    """
    Apply a Gaussian filter to the closing odds and prediction probability. 
    This filter is used to suppress the score when the prediction is far from the closing odds, simulating a more realistic prediction.

    :param pwmd: MatchPredictionWithMatchData
    :return: float, the calculated Gaussian filter
    """
    t = 0.5 # Controls the spread/width of the Gaussian curve outside the plateau region. Larger t means slower decay in the exponential term
    t2 = 0.05 # Controls the spread/width of the Gaussian curve inside the plateau region. t2 is used on lay predictions
    #t2 = 0.0001 # Controls the spread/width of the Gaussian curve inside the plateau region. t2 is used on lay predictions
    a = 0.25 # Controls the height of the plateau boundary. More negative a means lower plateau boundary
    b = 0.3 # Controls how quickly the plateau boundary changes with odds. Larger b means faster exponential decay in plateau width
    c = 0.25 # Minimum plateau width/boundary
    pwr = 1.1 # Power to raise the difference between odds and 1/prob to in the exponential term

    # Plateau width calculation
    w = c - a * np.exp(-b * (odds - 1))
    diff = abs(odds - 1 / prob)

    # If wp is less than implied wp, or wp odds is greater than implied odds, then use t2
    if odds < 1 / prob:
        t = t2

    # Plateaud curve with with uniform decay
    exp_component = 1.0 if diff <= w else np.exp( -(diff - w) / (t * np.power((odds-1),pwr)) )
    
    return exp_component

def apply_gaussian_filter_v2(odds, prob):
    ########################
    # note that sigma^2 = odds now
    ########################
    t = 1.0
    a = -2
    b = 0.3
    c = 3

    w = a * np.exp(-b * (odds-1)) + c

    diff = abs(odds - 1/prob)
    exp_component = 1.0 if diff <= w else np.exp(-np.power(diff, 2) / (t*2*odds))
    return exp_component

def apply_gaussian_filter_v3(odds, prob):
    ########################
    # note that sigma^2 = odds now
    ########################
    t = 0.5
    a = -2
    b = 0.3
    c = 1

    # Plateau width calculation
    w = c - a * np.exp(-b * (odds - 1))

    # Difference calculation
    diff = abs(odds - 1 / prob)

    # Updated equation with uniform decay
    exp_component = 1.0 if diff <= w else np.exp(-(diff - w) / t)
    
    return exp_component

def apply_gaussian_filter_v4(odds, prob):
    ########################
    # note that sigma^2 = odds now
    ########################
    t = 1
    a = -2
    b = 0.3
    c = 2

    w = a * np.exp(-b * (odds-1)) + c

    diff = abs(odds - 1/prob)
    exp_component = 1.0 if diff <= w else np.exp(-np.power(diff, 2) / (t*2*odds))
    return exp_component

if __name__ == "__main__":
    main()