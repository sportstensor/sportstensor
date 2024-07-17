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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Added import for MinMaxScaler
from matplotlib import pyplot as plt

from st.models.baseball import BaseballPredictionModel


def get_data() -> pd.DataFrame:
    file_path = 'data_and_models/mlb_model_ready_data_comb.csv'
    data = pd.read_csv(file_path)
    #print(f"Data loaded with shape: {data.shape}")

    # Removing outliers 
    ht_sc_95 = np.percentile(data['HT_SC'], 95)
    at_sc_95 = np.percentile(data['AT_SC'], 95)
    #print(f"Data percentiles before removing outliers: 95% HT_SC={np.percentile(data['HT_SC'], 95)}, AT_SC={np.percentile(data['AT_SC'], 95)}")
    data = data[(data['HT_SC'] <= ht_sc_95) & (data['AT_SC'] <= at_sc_95)]
    #print(f"Data shape after removing outliers: {data.shape}")

    return data


def load_or_run_model(scalers: dict, X_scaled: np.ndarray, y_scaled: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    file_path = 'data_and_models/basic_model_mlb.keras'

    model = load_model(file_path)
    print(f"Model loaded from {file_path}")

    return model


class MLBBaseballPredictionModel(BaseballPredictionModel):
    def __init__(self, prediction):
        super().__init__(prediction)
        self.huggingface_model = "sportstensor/basic_mls_model"
        self.mls_fixture_data_filepath = "mls_fixture_data.xlsx"
        self.mls_model_filepath = "basic_mls_model.keras"
        self.mls_combined_table_filepath = "combined_table.csv"

    def activate(self, matchDate, homeTeamName, awayTeamName):
        data = get_data()

        scalers, X_scaled, y_scaled = self.scale_data(data)

        model = load_or_run_model(scalers, X_scaled, y_scaled)

        pred_input, hist_score = self.prep_pred_input(matchDate, homeTeamName, awayTeamName, scalers)

        predicted_outcome = model.predict(pred_input)

        home_pred_unrounded = scalers['HT_SC'].inverse_transform(predicted_outcome[:, 0].reshape(-1, 1))[0][0]
        away_pred_unrounded = scalers['AT_SC'].inverse_transform(predicted_outcome[:, 1].reshape(-1, 1))[0][0]

        home_pred = round(home_pred_unrounded)
        away_pred = round(away_pred_unrounded)
        #print(f"Final predicted scores: Home={home_pred}, Away={away_pred}")

        if home_pred == away_pred and home_pred_unrounded > away_pred_unrounded:
            away_pred -= 1
        elif home_pred == away_pred and home_pred_unrounded < away_pred_unrounded:
            home_pred -= 1

        # Ensure that predictions dictionary is always returned
        predictions = {
            homeTeamName: home_pred,
            awayTeamName: away_pred
        }

        return predictions

    def make_prediction(self):
        bt.logging.info("Predicting MLB match...")
        matchDate = self.prediction.matchDate.strftime("%Y-%m-%d")
        homeTeamName = self.prediction.homeTeamName
        awayTeamName = self.prediction.awayTeamName
        
        predictions = self.activate(matchDate, homeTeamName, awayTeamName)

        if predictions is not None and homeTeamName in predictions and awayTeamName in predictions:
            # Set our final predictions
            bt.logging.info("Setting final predictions from model...")
            self.prediction.homeTeamScore = int(predictions[homeTeamName])
            self.prediction.awayTeamScore = int(predictions[awayTeamName])
        else:
            bt.logging.warning("Failed to get predictions from model, setting random scores")
            self.prediction.homeTeamScore = random.randint(0, 10)
            self.prediction.awayTeamScore = random.randint(0, 10)

        #print(f"Assigned final predictions: Home={predictions[homeTeamName]}, Away={predictions[awayTeamName]}")
        return True

    def scale_data(self, data: pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray]:

        ## Scaling data so it is normalized and ready for ingestion ##
        X_scaled = data[
            ['HT', 'AT', 'HT_RD', 'AT_RD', 'HT_ELO', 'AT_ELO', 'HT_HPG', 'AT_HPG', 'HT_PREV_SC', 'AT_PREV_SC',
             'HT_WL_RATIO', 'AT_WL_RATIO', 'HT_AVG_SC', 'AT_AVG_SC']].values
        y_scaled = data[['HT_SC', 'AT_SC']].values.astype(float)

        columns_for_model_input = ['HT', 'AT', 'HT_RD', 'AT_RD', 'HT_ELO', 'AT_ELO', 'HT_HPG', 'AT_HPG', 'HT_PREV_SC',
                                   'AT_PREV_SC', 'HT_WL_RATIO', 'AT_WL_RATIO', 'HT_AVG_SC', 'AT_AVG_SC', 'HT_SC',
                                   'AT_SC']

        # Scale features
        scalers = {}
        index = 0
        for column in columns_for_model_input:
            scaler = MinMaxScaler(feature_range=(0, 1))

            if index < X_scaled.shape[1]:
                X_scaled[:, index] = scaler.fit_transform(X_scaled[:, index].reshape(-1, 1)).reshape(1, -1)
            else:
                y_scaled[:, index - X_scaled.shape[1]] = scaler.fit_transform(
                    y_scaled[:, index - X_scaled.shape[1]].reshape(-1, 1)).reshape(1, -1)

            scalers[column] = scaler
            #print(f"Scaler for {column}: min={scaler.min_[0]}, scale={scaler.scale_[0]}")
            index += 1

        return scalers, X_scaled, y_scaled

    def prep_pred_input(self, date: str, home_team: str, away_team: str, scalers: dict) -> np.array:

        date_formatted = datetime.strptime(date, '%Y-%m-%d')
        current_date = datetime.now().date()

        current_team_stats = pd.read_excel('data_and_models/current_team_data.xlsx')

        home_abrv = current_team_stats[current_team_stats['team'] == home_team]['abrv'].values[0]
        away_abrv = current_team_stats[current_team_stats['team'] == away_team]['abrv'].values[0]

        home_id = get_id_from_ABRV(home_abrv)
        away_id = get_id_from_ABRV(away_abrv)

        home_val = get_teamcode_from_id(home_id)
        away_val = get_teamcode_from_id(away_id)

        #print(home_abrv, away_abrv)
        #print(home_team, away_team)
        #print(home_val, away_val)
        #print(date_formatted)

        if date_formatted.date() < current_date:

            file_path = 'data_and_models/mlb_model_ready_data_comb.xlsx'
            if not os.path.exists(file_path):
                print('Data needed, scrape it and store it in order to get input')
                input = 0
            else:

                fixtures = pd.read_excel(file_path)
                try:
                    matching_input = fixtures[(fixtures['DATE'] == date_formatted) & (fixtures['HT'] == home_val) & (
                                fixtures['AT'] == away_val)]
                except:
                    matching_input = 0
                    print('Match could not be found in data source, scrape more data or check inputs.')

                input = matching_input[
                    ['HT', 'AT', 'HT_RD', 'AT_RD', 'HT_ELO', 'AT_ELO', 'HT_HPG', 'AT_HPG', 'HT_PREV_SC', 'AT_PREV_SC',
                     'HT_WL_RATIO', 'AT_WL_RATIO', 'HT_AVG_SC', 'AT_AVG_SC']].values.astype(float)

                index = 0
                for column in scalers.keys():
                    if index < input.shape[1]:
                        input[:, index] = scalers[column].transform(input[:, index].reshape(-1, 1)).reshape(1, -1)
                    index += 1
                output = matching_input[['HT_SC', 'AT_SC']].values
        else:

            home_stats = current_team_stats[(current_team_stats['abrv'] == home_abrv)]
            away_stats = current_team_stats[(current_team_stats['abrv'] == away_abrv)]

            input = {}
            input['HT'] = home_val
            input['AT'] = away_val
            input['HT_RD'] = home_stats['rd'].to_numpy()[0]
            input['AT_RD'] = away_stats['rd'].to_numpy()[0]
            input['HT_ELO'] = home_stats['elo'].to_numpy()[0]
            input['AT_ELO'] = away_stats['elo'].to_numpy()[0]
            input['HT_HPG'] = home_stats['hpg'].to_numpy()[0]
            input['AT_HPG'] = away_stats['hpg'].to_numpy()[0]
            input['HT_PREV_SC'] = home_stats['prev_sc'].to_numpy()[0]
            input['AT_PREV_SC'] = away_stats['prev_sc'].to_numpy()[0]
            input['HT_WL_RATIO'] = home_stats['w'].to_numpy()[0] / home_stats['l'].to_numpy()[0]
            input['AT_WL_RATIO'] = away_stats['w'].to_numpy()[0] / away_stats['l'].to_numpy()[0]
            input['HT_AVG_SC'] = home_stats['rpg'].to_numpy()[0]
            input['AT_AVG_SC'] = away_stats['rpg'].to_numpy()[0]

            input = np.array(list(input.values())).reshape(1, -1)

            index = 0
            for column in scalers.keys():
                if index < input.shape[1]:
                    input[:, index] = scalers[column].transform(input[:, index].reshape(-1, 1)).reshape(1, -1)
                index += 1

            output = '...'

        return input, output

    def update_current_team_database():
        ### Update database for current statistics ###
        print('Database updated successfully')


def get_id_from_ABRV(ABRV: str):
    team_id = {
        'SD': 135278,
        'CHC': 135269,
        'LAD': 135272,
        'TEX': 135264,
        'COL': 135271,
        'MIN': 135259,
        'STL': 135280,
        'TOR': 135265,
        'NYM': 135275,
        'SF': 135279,
        'KC': 135257,
        'OAK': 135261,
        'CWS': 135253,
        'LAA': 135258,
        'ARI': 135267,
        'BAL': 135251,
        'CLE': 135254,
        'DET': 135255,
        'TB': 135263,
        'WSH': 135281,
        'NYY': 135260,
        'PHI': 135276,
        'MIL': 135274,
        'SEA': 135262,
        'PIT': 135277,
        'BOS': 135252,
        'MIA': 135273,
        'HOU': 135256,
        'CIN': 135270,
        'ATL': 135268
    }

    return team_id[ABRV]


def get_teamcode_from_id(id: float):
    teamcode = {
        135260: 30,
        135272: 29,
        135280: 28,
        135279: 27,
        135268: 26,
        135269: 25,
        135261: 24,
        135252: 23,
        135255: 22,
        135270: 21,
        135277: 20,
        135276: 19,
        135251: 18,
        135259: 17,
        135253: 16,
        135254: 15,
        135275: 14,
        135256: 13,
        135257: 12,
        135264: 11,
        135278: 10,
        135265: 9,
        135263: 8,
        135273: 7,
        135271: 6,
        135258: 5,
        135267: 4,
        135281: 3,
        135274: 2,
        135262: 1,
    }

    return teamcode[id]


def randomised_sleep_time(self, lower_bound, upper_bound):
    delay = random.uniform(lower_bound, upper_bound)
    print(f" Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)
