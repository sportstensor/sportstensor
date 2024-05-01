from retrieve_data import get_data, scale_data, prep_pred_input
from model import load_or_run_model
from predictions import predict

import numpy as np

def activate(match_datetime, homeTeamName, awayTeamName):
    scrape_more_data = False
    data = get_data(scrape_more_data)

    scalers, X_scaled, y_scaled = scale_data(data)

    model = load_or_run_model(scalers, X_scaled, y_scaled)

    pred_input, hist_score = prep_pred_input(match_datetime, homeTeamName, awayTeamName, scalers)
    predicted_outcome = model.predict(pred_input)
    
    predicted_outcome[:,0] = np.round(scalers['HT_SC'].inverse_transform(predicted_outcome[:, 0].reshape(-1, 1)).reshape(-1))
    predicted_outcome[:,1] = np.round(scalers['AT_SC'].inverse_transform(predicted_outcome[:, 1].reshape(-1, 1)).reshape(-1))

    # Ensure that predictions dictionary is always returned
    predictions = {(homeTeamName, awayTeamName): (predicted_outcome[0][0], predicted_outcome[0][1])}
    # print({'DATE': match_datetime, 'HT': homeTeamName, 'AT': awayTeamName},
    #      'predicted score:', predicted_outcome[0], 'actual score:', hist_score) 

    return predictions
