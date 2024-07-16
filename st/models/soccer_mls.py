import random
import bittensor as bt

import os
from datetime import datetime
import time
from typing import Tuple
import math
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download

import tensorflow as tf
from keras._tf_keras.keras.models import load_model, Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, InputLayer
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras._tf_keras.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

from st.models.soccer import SoccerPredictionModel


class MLSSoccerPredictionModel(SoccerPredictionModel):
    def __init__(self, prediction):
        super().__init__(prediction)
        self.huggingface_model = "sportstensor/basic_mls_model"
        self.mls_fixture_data_filepath = "mls_fixture_data.xlsx"
        self.mls_model_filepath = "basic_mls_model.keras"
        self.mls_combined_table_filepath = "combined_table.csv"

    def make_prediction(self):
        bt.logging.info("Predicting soccer match...")
        matchDate = self.prediction.matchDate.strftime("%Y-%m-%d")
        homeTeamName = self.prediction.homeTeamName
        awayTeamName = self.prediction.awayTeamName

        predictions = self.activate(matchDate, homeTeamName, awayTeamName)

        if predictions is not None and (homeTeamName, awayTeamName) in predictions:
            pred_scores = predictions[(homeTeamName, awayTeamName)]
            self.prediction.homeTeamScore = int(pred_scores[0])
            self.prediction.awayTeamScore = int(pred_scores[1])
        else:
            self.prediction.homeTeamScore = random.randint(0, 10)
            self.prediction.awayTeamScore = random.randint(0, 10)

    def activate(self, matchDate, homeTeamName, awayTeamName):
        data = self.get_data()

        scalers, X_scaled, y_scaled = self.scale_data(data)

        model = self.load_or_run_model(scalers, X_scaled, y_scaled)

        pred_input, hist_score = self.prep_pred_input(matchDate, homeTeamName, awayTeamName, scalers)

        predicted_outcome = model.predict(pred_input)
            
        predicted_outcome[:,0] = np.round(scalers['HT_SC'].inverse_transform(predicted_outcome[:, 0].reshape(-1, 1)).reshape(-1))
        predicted_outcome[:,1] = np.round(scalers['AT_SC'].inverse_transform(predicted_outcome[:, 1].reshape(-1, 1)).reshape(-1))

        # Ensure that predictions dictionary is always returned
        predictions = {(homeTeamName, awayTeamName): (predicted_outcome[0][0], predicted_outcome[0][1])}

        return predictions
    
    def get_data(self) -> pd.DataFrame:
        # grab data from our base model on HF
        file_path = hf_hub_download(repo_id=self.huggingface_model, filename=self.mls_fixture_data_filepath)
               
        if os.path.exists(file_path):
            fixtures = pd.read_excel(file_path)
        else:
            print('No file found, scrape data first.')
            fixtures = pd.DataFrame()    

        return fixtures
    
    def load_or_run_model(self, scalers: dict, X_scaled: np.ndarray, y_scaled: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        file_path = hf_hub_download(repo_id=self.huggingface_model, filename=self.mls_model_filepath)

        if not os.path.exists(file_path):
            model = Sequential([
                InputLayer(input_shape=(X_scaled.shape[1],)),
                Dense(units=2, activation='relu')
            ])

            opt = Adam()
            model.compile(optimizer=opt, loss='mean_squared_error')
            es = EarlyStopping(monitor='loss', mode='min', patience=6)
            mcp_save = ModelCheckpoint(file_path, save_best_only=True, monitor='loss', mode='min')
            model.fit(X_train, y_train, epochs=150, batch_size=32, callbacks=[es, mcp_save])

            predicted_scores_validate = model.predict(X_test)

            # Rescale back to original range    
            home_predicted_scores = np.round(scalers['HT_SC'].inverse_transform(predicted_scores_validate[:, 0].reshape(-1, 1)))     
            away_predicted_scores = np.round(scalers['AT_SC'].inverse_transform(predicted_scores_validate[:, 1].reshape(-1, 1)))

            # Calculate metrics
            home_mse_test = mean_squared_error(y_test[:, 0], home_predicted_scores)
            home_MAE_test = mean_absolute_error(y_test[:, 0], home_predicted_scores)
            home_R2val_test = r2_score(y_test[:, 0], home_predicted_scores)

            away_mse_test = mean_squared_error(y_test[:, 1], away_predicted_scores)
            away_MAE_test = mean_absolute_error(y_test[:, 1], away_predicted_scores)
            away_R2val_test = r2_score(y_test[:, 1], away_predicted_scores)

            print('RMSE Home = {}, Away = {}'.format(math.sqrt(home_mse_test), math.sqrt(away_mse_test)))
            print('MAE Home = {}, Away = {}'.format(home_MAE_test, away_MAE_test))
            print('R2 Value Home = {}, Away = {}'.format(home_R2val_test, away_R2val_test))

        else:
            model = load_model(file_path)

        return model
    
    def scale_data(self, data:pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray]:

        ## Scaling data so it is normalized and ready for ingestion ##
        X_scaled = data[['HT', 'AT', 'HT_GD', 'AT_GD']].values #, 'HT_ELO', 'AT_ELO'
        y_scaled = data[['HT_SC', 'AT_SC']].values.astype(float)

        columns_in_input = ['HT', 'AT', 'HT_GD', 'AT_GD', 'HT_SC', 'AT_SC'] #, 'HT_ELO', 'AT_ELO'

        # Scale features
        scalers = {}
        index = 0
        for column in columns_in_input:
            scaler = MinMaxScaler(feature_range=(0, 1))

            if index < X_scaled.shape[1]:
                X_scaled[:,index] = scaler.fit_transform(X_scaled[:,index].reshape(-1,1)).reshape(1,-1)
            else:
                y_scaled[:,index-X_scaled.shape[1]] = scaler.fit_transform(y_scaled[:,index-X_scaled.shape[1]].reshape(-1,1)).reshape(1,-1)

            scalers[column] = scaler
            index += 1

        return scalers, X_scaled, y_scaled

    def prep_pred_input(self, date:str, home_team:str, away_team:str, scalers:dict) -> np.array:

        file_path = hf_hub_download(repo_id=self.huggingface_model, filename=self.mls_combined_table_filepath)
        fixture_data_file_path = hf_hub_download(repo_id=self.huggingface_model, filename=self.mls_fixture_data_filepath)

        date_formatted = datetime.strptime(date, '%Y-%m-%d')
        home_val = self.get_team_sorted_val(home_team)
        away_val = self.get_team_sorted_val(away_team)

        current_date = datetime.now().date()

        if date_formatted.date() < current_date:
            
            if not os.path.exists(fixture_data_file_path):
                print('Data needed, scrape it and store it in order to get input')
                input = 0
            else:
                fixtures = pd.read_excel(fixture_data_file_path)

                try:
                    matching_input = fixtures[(fixtures['DATE'] == date_formatted) & (fixtures['HT'] == home_val) & (fixtures['AT'] == away_val)]
                except:
                    matching_input = 0
                    print('Match could not be found in data source, scrape more data or check inputs.')  

                input = matching_input[['HT', 'AT', 'HT_GD', 'AT_GD']].values.astype(float) #, 'HT_ELO', 'AT_ELO'
                
                index = 0
                for column in scalers.keys():                
                    if index < input.shape[1]:
                        input[:,index] = scalers[column].transform(input[:,index].reshape(-1,1)).reshape(1,-1)
                    index += 1
                output = matching_input[['HT_SC', 'AT_SC']].values

                return input, output
        else:        

            input = {}
            input['HT'] = home_val
            input['AT'] = away_val
    
            # Fetch all tables from the CSV
            prem_table_current = pd.read_csv(file_path)
            return prem_table_current
        

    def get_team_sorted_val(self, team_name:str):

        ### This dictionary orders the teams by appearances in the BPL from 2002/03 -> 2023/24. Allows categorical team input into model. ###
        
        ### Please adjust teams if using different league or order if you find something preferable. ### 

        ### Each team should have a unique value, it is not representing any numeric quantity although the model 
        # may make that assumption hence some smarts to order necessary. ###

        # If teams change name over years, make sure both names are included in dictionary with the same value #

        team_vals = {
            'D.C. United': 32,
            'DC United': 32,
            'LA Galaxy': 31,
            'L.A. Galaxy': 31,
            'New England Revolution': 30,
            'Colorado Rapids': 29,
            'Columbus Crew': 28,
            'FC Dallas': 27,
            'Dallas Burn': 27,
            'San Jose Earthquakes': 26,
            'San Jose Clash': 26,
            'Sporting Kansas City': 25,
            'Kansas City Wiz': 25,
            'New York Red Bulls': 24,
            'New York': 24,
            'New Jersey MetroStars': 24,
            'Chicago Fire': 23,
            'Real Salt Lake': 22,
            'Toronto FC': 21,
            'Houston Dynamo': 20,
            'Seattle Sounders FC': 19,
            'Philadelphia Union': 18,
            'Portland Timbers': 17,
            'Vancouver Whitecaps': 16,
            'CF Montréal': 15,
            'Montreal Impact': 15,
            'Orlando City': 14,
            'New York City FC': 13,
            'Atlanta United': 12,
            'Minnesota United': 11,
            'Los Angeles FC': 10,
            'FC Cincinnati': 9,
            'Inter Miami': 8,
            'Nashville SC': 7,
            'Austin FC': 6,
            'Charlotte FC': 5,
            'St. Louis City SC': 4,
            'St. Louis City': 4,
            'St. Louis': 4,
            'Chivas USA': 3,
            'Tampa Bay Mutiny': 2,
            'Miami Fusion': 1,
        }

        return team_vals[team_name]

    def get_team_name_clubelo_format(self, team_name:str):

        ### Retrieves format of team name that works for ClubElo

        club_elo_team_name_format = {
            'Sporting Kansas City': 'Sporting Kansas City',
            'Austin FC': 'Austin FC',
            'FC Dallas': 'FC Dallas',
            'Chicago Fire': 'Chicago Fire',
            'Philadelphia Union': 'Philadelphia Union',
            'Colorado Rapids': 'Colorado Rapids',
            'Portland Timbers': 'Portland Timbers',
            'Vancouver Whitcaps': 'Vancouver Whitcaps',
            'Nashville SC': 'Nashville',
            'Charlotte FC': 'Charlotte FC',
            'Columbus Crew': 'Columbus Crew',
            'Seattle Sounders FC': 'Seattle',
            'Los Angeles FC': 'Los Angeles FC',
            'Real Salt Lake': 'Real Salt Lake',
            'Inter Miami': 'Inter Miami',
            'FC Cincinnati': 'FC Cincinnati',
            'New York City FC': 'New York City FC',
            'NY Red Bulls': 'NY Red Bulls',
            'St. Louis City': 'St. Louis City',
            'Toronto FC': 'Toronto FC',
            'LA Galaxy': 'LA Galaxy',
            'Minnesota United': 'Minnesota United',
            'Houston Dynamo': 'Houston Dynamo',
            'D.C. United': 'D.C. United',
            'Orlando City': 'Orlando City',
            'CF Montréal': 'CF Montréal',
            'Atlanta United': 'Atlanta United',
            'San Jose Earthquakes': 'San Jose Earthquakes',
            'NE Revolution': 'NE Revolution'
            }

        return club_elo_team_name_format[team_name]


    def get_team_name_fbref_format(self, team_name:str):

        ### Retrieves format of team name that works for Fbref

        fbref_team_name_format = {
            'Sporting Kansas City': 'Sporting KC',
            'Austin FC': 'Austin',
            'FC Dallas': 'FC Dallas',
            'Chicago Fire': 'Fire',
            'Philadelphia Union': 'Philadelphia',
            'Colorado Rapids': 'Rapids',
            'Portland Timbers': 'Portland Timbers',
            'Vancouver Whitcaps': 'Vancouver Whitcaps',
            'Nashville SC': 'Nashville',
            'Charlotte FC': 'Charlotte',
            'Columbus Crew': 'Crew',
            'Seattle Sounders FC': 'Seattle Sounders FC',
            'Los Angeles FC': 'LAFC',
            'Real Salt Lake': 'RSL',
            'Inter Miami': 'Inter Miami',
            'FC Cincinnati': 'FC Cincinnati',
            'New York City FC': 'NYCFC',
            'New York Red Bulls': 'NY Red Bulls',
            'St. Louis City SC': 'St. Louis',
            'St. Louis City': 'St. Louis',
            'Toronto FC': 'Toronto FC',
            'L.A. Galaxy': 'LA Galaxy',
            'Minnesota United': 'Minnesota Utd',
            'Houston Dynamo': 'Dynamo FC',
            'DC United': 'D.C. United',
            'Orlando City': 'Orlando City',
            'CF Montréal': 'CF Montréal',
            'Atlanta United': 'Atlanta Utd',
            'San Jose Earthquakes': 'SJ Earthquakes',
            'New England Revolution': 'NE Revolution',
            'Vancouver Whitecaps': 'Vancouver W\'caps',

            }

        return fbref_team_name_format[team_name]

    def randomised_sleep_time(self, lower_bound, upper_bound):    
        delay = random.uniform(lower_bound, upper_bound)
        print(f" Sleeping for {delay:.2f} seconds...")
        time.sleep(delay)
