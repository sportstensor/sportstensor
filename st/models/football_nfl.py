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
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)
from matplotlib import pyplot as plt

from st.models.football import FootballPredictionModel


class NFLFootballPredictionModel(FootballPredictionModel):
    def __init__(self, prediction):
        super().__init__(prediction)
        self.huggingface_model = "sportstensor/basic_model"
        self.nfl_current_team_data_filepath = "nfl/fixture_data.xlsx"
        self.nfl_model_filepath = "nfl/basic_model.keras"
        self.nfl_model_ready_data_comb_filepath = "nfl/combined_table.csv"

    def load_or_run_model(
        self, scalers: dict, X_scaled: np.ndarray, y_scaled: np.ndarray
    ):

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        file_path = hf_hub_download(
            repo_id=self.huggingface_model, filename=self.nfl_model_filepath
        )

        model = load_model(file_path)
        print(f"Model loaded from {file_path}")

        return model

    def get_data(self) -> pd.DataFrame:

        file_path = hf_hub_download(
            repo_id=self.huggingface_model,
            filename=self.nfl_model_ready_data_comb_filepath,
        )

        # check number of entries with na's removed and all stats    
        all_stats_df = pd.read_csv(file_path)
        all_stats_df.rename(columns={'home_team': 'HT', 'away_team': 'AT', 'home_score':'HT_SC', 'away_score':'AT_SC', 'home_moneyline':'HT_ML', 'away_moneyline':'AT_ML', 'spread_line':'SL', 'temp':'T', 'home_elo':'HT_ELO', 'away_elo':'AT_ELO', 'home_points_diff':'HT_PD', 'away_points_diff':'AT_PD', 'home_qb_rating':'HT_QBR', 'away_qb_rating':'AT_QBR'}, inplace=True)

        all_stats_df.to_csv(file_path)

        all_stats_df = all_stats_df.dropna()
        all_stats_df = remove_outliers(all_stats_df)
        all_stats_df = all_stats_df.dropna()


        data = all_stats_df
        teamcode = {
            'KC':32,
            'BUF':31,
            'LA':30,
            'NO':29,
            'BAL':28,
            'PHI':27,
            'GB':26,
            'SF':25,
            'DAL':24,
            'PIT':23,
            'NE':22,
            'MIN':21,
            'SEA':20,
            'TEN':19,
            'TB':18,
            'LAC':17,
            'MIA':16,
            'CIN':15,
            'IND':14,
            'CLE':13,
            'ATL':12,
            'OAK':11,
            'LV':11,
            'CHI':10,
            'DET':9,
            'HOU':8,
            'JAX':7,
            'DEN':6,
            'WAS':5,
            'ARI':4,
            'CAR':3,
            'NYG':2,
            'NYJ':1
        }
        data['HT'] = data['HT'].map(teamcode)
        data['AT'] = data['AT'].map(teamcode)
        data = data.dropna()

        return data

    def activate(self, matchDate, homeTeamName, awayTeamName):
        data = self.get_data()

        scalers, X_scaled, y_scaled = self.scale_data(data)

        model = self.load_or_run_model(scalers, X_scaled, y_scaled)

        pred_input, hist_score = self.prep_pred_input(
            matchDate, homeTeamName, awayTeamName, scalers
        )

        predicted_outcome = model.predict(pred_input)

        home_pred_unrounded = scalers["HT_SC"].inverse_transform(
            predicted_outcome[:, 0].reshape(-1, 1)
        )[0][0]
        away_pred_unrounded = scalers["AT_SC"].inverse_transform(
            predicted_outcome[:, 1].reshape(-1, 1)
        )[0][0]

        home_pred = round(home_pred_unrounded)
        away_pred = round(away_pred_unrounded)
        # print(f"Final predicted scores: Home={home_pred}, Away={away_pred}")

        if home_pred == away_pred and home_pred_unrounded > away_pred_unrounded:
            away_pred -= 1
        elif home_pred == away_pred and home_pred_unrounded < away_pred_unrounded:
            home_pred -= 1

        # Ensure that predictions dictionary is always returned
        predictions = {homeTeamName: home_pred, awayTeamName: away_pred}

        return predictions

    def make_prediction(self):
        bt.logging.info("Predicting NFL match...")
        matchDate = self.prediction.matchDate.strftime("%Y-%m-%d")
        homeTeamName = self.prediction.homeTeamName
        awayTeamName = self.prediction.awayTeamName

        predictions = self.activate(matchDate, homeTeamName, awayTeamName)

        if (
            predictions is not None
            and homeTeamName in predictions
            and awayTeamName in predictions
        ):
            # Set our final predictions
            bt.logging.info("Setting final predictions from model...")
            self.prediction.homeTeamScore = int(predictions[homeTeamName])
            self.prediction.awayTeamScore = int(predictions[awayTeamName])
        else:
            bt.logging.warning(
                "Failed to get predictions from model, setting random scores"
            )
            self.prediction.homeTeamScore = random.randint(0, 10)
            self.prediction.awayTeamScore = random.randint(0, 10)

        # print(f"Assigned final predictions: Home={predictions[homeTeamName]}, Away={predictions[awayTeamName]}")
        return True

    def scale_data(self, data: pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray]:

        ## Scaling data so it is normalized and ready for ingestion ##
        X_scaled = data[
            [
                "HT",
                "AT",
                "HT_ELO",
                "AT_ELO",
                "HT_PD",
                "AT_PD",
            ]
        ].values
        y_scaled = data[["HT_SC", "AT_SC"]].values.astype(float)

        columns_for_model_input = [
            "HT",
            "AT",
            "HT_ELO",
            "AT_ELO",
            "HT_PD",
            "AT_PD",
            "HT_SC",
            "AT_SC",
        ]

        # Scale features
        scalers = {}
        index = 0
        for column in columns_for_model_input:
            scaler = MinMaxScaler(feature_range=(0, 1))

            if index < X_scaled.shape[1]:
                X_scaled[:, index] = scaler.fit_transform(
                    X_scaled[:, index].reshape(-1, 1)
                ).reshape(1, -1)
            else:
                y_scaled[:, index - X_scaled.shape[1]] = scaler.fit_transform(
                    y_scaled[:, index - X_scaled.shape[1]].reshape(-1, 1)
                ).reshape(1, -1)

            scalers[column] = scaler
            # print(f"Scaler for {column}: min={scaler.min_[0]}, scale={scaler.scale_[0]}")
            index += 1

        return scalers, X_scaled, y_scaled

    def prep_pred_input(
        self, date: str, home_team: str, away_team: str, scalers: dict
    ) -> np.array:

        date_formatted = datetime.strptime(date, "%Y-%m-%d")
        current_date = datetime.now().date()

        file_path = hf_hub_download(
            repo_id=self.huggingface_model, filename=self.nfl_current_team_data_filepath
        )

        current_team_stats = pd.read_excel(file_path)

        home_abbrv = current_team_stats[current_team_stats["team"] == home_team][
            "abbrv"
        ].values[0]
        away_abbrv = current_team_stats[current_team_stats["team"] == away_team][
            "abbrv"
        ].values[0]

        home_abbrv_dict = home_abbrv.split(' ')[4]
        away_abbrv_dict = away_abbrv.split(' ')[4]

        home_stats = current_team_stats[(current_team_stats['abbrv'] == home_abbrv)]
        away_stats = current_team_stats[(current_team_stats['abbrv'] == away_abbrv)]

        input = {}
        input['HT'] = get_teamcode_from_abrv(home_abbrv_dict)
        input['AT'] = get_teamcode_from_abrv(away_abbrv_dict)
        input['HT_ELO'] = home_stats['elo'].to_numpy()[0]
        input['AT_ELO'] = away_stats['elo'].to_numpy()[0]
        input['HT_RD'] = home_stats['pd'].to_numpy()[0]
        input['AT_RD'] = away_stats['pd'].to_numpy()[0]

        input = np.array(list(input.values())).reshape(1,-1)

        index = 0
        for column in scalers.keys():
            if index < input.shape[1]:
                        input[:, index] = (
                            scalers[column]
                            .transform(input[:, index].reshape(-1, 1))
                            .reshape(1, -1)
                        )
            index += 1
        output = '...'
        return input, output

    def update_current_team_database():
        ### Update database for current statistics ###
        print("Database updated successfully")


def get_teamcode_from_abrv(ABBRV: str):
    teamcode = {
        'KC':32,
        'BUF':31,
        'LA':30,
        'NO':29,
        'BAL':28,
        'PHI':27,
        'GB':26,
        'SF':25,
        'DAL':24,
        'PIT':23,
        'NE':22,
        'MIN':21,
        'SEA':20,
        'TEN':19,
        'TB':18,
        'LAC':17,
        'MIA':16,
        'CIN':15,
        'IND':14,
        'CLE':13,
        'ATL':12,
        'OAK':11,
        'LV':11,
        'CHI':10,
        'DET':9,
        'HOU':8,
        'JAX':7,
        'DEN':6,
        'WAS':5,
        'ARI':4,
        'CAR':3,
        'NYG':2,
        'NYJ':1
        }
    return teamcode[ABBRV]

def remove_outliers(data):

    # Removing outliers 
    ht_sc_95 = np.percentile(data['HT_SC'], 95)
    at_sc_95 = np.percentile(data['AT_SC'], 95)
    
    ht_pd_5 = np.percentile(data['HT_PD'], 5)
    ht_pd_95 = np.percentile(data['HT_PD'], 95)

    at_pd_5 = np.percentile(data['AT_PD'], 5)
    at_pd_95 = np.percentile(data['AT_PD'], 95)
    
    ht_elo_5 = np.percentile(data['HT_ELO'], 3)
    ht_elo_95 = np.percentile(data['HT_ELO'], 97)

    at_elo_5 = np.percentile(data['AT_ELO'], 3)
    at_elo_95 = np.percentile(data['AT_ELO'], 97)
    
    data = data[(data['HT_SC'] <= ht_sc_95) & (data['AT_SC'] <= at_sc_95)]
    data = data[(data['HT_PD'] >= ht_pd_5) & (data['HT_PD'] <= ht_pd_95)]
    data = data[(data['AT_PD'] >= at_pd_5) & (data['AT_PD'] <= at_pd_95)]
    data = data[(data['HT_ELO'] >= ht_elo_5) & (data['HT_ELO'] <= ht_elo_95)]
    data = data[(data['AT_ELO'] >= at_elo_5) & (data['AT_ELO'] <= at_elo_95)]

    return data


def randomised_sleep_time(self, lower_bound, upper_bound):
    delay = random.uniform(lower_bound, upper_bound)
    print(f" Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)
