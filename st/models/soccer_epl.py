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
from keras._tf_keras.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

from st.models.soccer import SoccerPredictionModel

class EPLSoccerPredictionModel(SoccerPredictionModel):
    def __init__(self, prediction):
        super().__init__(prediction)
        self.huggingface_model = "sportstensor/basic_model"
        self.epl_fixture_data_filepath = "epl/fixture_data.csv"
        self.epl_model_filepath = "epl/basic_model.keras"
        self.epl_combined_table_filepath = "epl/combined_table.csv"

    def make_prediction(self):
        bt.logging.info("Predicting EPL soccer match...")
        matchDate = self.prediction.matchDate.strftime("%Y-%m-%d")
        homeTeamName = self.prediction.homeTeamName
        awayTeamName = self.prediction.awayTeamName

        print(f"Making prediction for {homeTeamName} vs {awayTeamName} on {matchDate}")

        predictions = self.activate(matchDate, homeTeamName, awayTeamName)

        if predictions is not None and (homeTeamName, awayTeamName) in predictions:
            pred_scores = predictions[(homeTeamName, awayTeamName)]
            self.prediction.homeTeamScore = int(pred_scores[0])
            self.prediction.awayTeamScore = int(pred_scores[1])
        else:
            self.prediction.homeTeamScore = random.randint(0, 10)
            self.prediction.awayTeamScore = random.randint(0, 10)
            print(f"Using random scores: {homeTeamName} {self.prediction.homeTeamScore} - {self.prediction.awayTeamScore} {awayTeamName}")

    def activate(self, matchDate, homeTeamName, awayTeamName):

        data = self.get_data()

        scalers, X_scaled, y_scaled = self.scale_data(data)

        model = self.load_or_run_model(scalers, X_scaled, y_scaled)

        pred_input = self.prep_pred_input(
            matchDate, homeTeamName, awayTeamName, scalers
        )

        if pred_input.size == 0:
            print("Error: Empty prediction input")
            return None

        predicted_outcome = model.predict(pred_input)

        predicted_outcome[:, 0] = np.round(
            scalers["HT_SC"]
            .inverse_transform(predicted_outcome[:, 0].reshape(-1, 1))
            .reshape(-1)
        )
        predicted_outcome[:, 1] = np.round(
            scalers["AT_SC"]
            .inverse_transform(predicted_outcome[:, 1].reshape(-1, 1))
            .reshape(-1)
        )

        # Ensure that predictions dictionary is always returned
        predictions = {
            (homeTeamName, awayTeamName): (
                predicted_outcome[0][0],
                predicted_outcome[0][1],
            )
        }

        return predictions

    def get_data(self) -> pd.DataFrame:
        file_path = hf_hub_download(
            repo_id=self.huggingface_model, filename=self.epl_fixture_data_filepath
        )

        try:
            if os.path.exists(file_path):
                fixtures = pd.read_csv(file_path)

                expected_columns = ['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_GD', 'AT_GD', 'HT_SC', 'AT_SC']
                missing_columns = set(expected_columns) - set(fixtures.columns)
                if missing_columns:
                    print(f"Warning: Missing columns in fixture_data.csv: {missing_columns}")

                return fixtures
            else:
                print(f"Error: File not found at {file_path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading fixture data: {str(e)}")
            return pd.DataFrame()

    def load_or_run_model(
        self, scalers: dict, X_scaled: np.ndarray, y_scaled: np.ndarray
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )


        file_path = hf_hub_download(
            repo_id=self.huggingface_model, filename=self.epl_model_filepath
        )

        local_model_path = hf_hub_download(
            repo_id=self.huggingface_model, filename=self.epl_model_filepath
        )
        if not os.path.exists(local_model_path):
            model = Sequential(
                [
                    InputLayer(input_shape=(X_scaled.shape[1],)),
                    Dense(units=2, activation="relu"),
                ]
            )

            opt = Adam()
            model.compile(optimizer=opt, loss="mean_squared_error")
            es = EarlyStopping(monitor="loss", mode="min", patience=6)
            mcp_save = ModelCheckpoint(
                file_path, save_best_only=True, monitor="loss", mode="min"
            )
            model.fit(
                X_train, y_train, epochs=150, batch_size=32, callbacks=[es, mcp_save]
            )

            predicted_scores_validate = model.predict(X_test)

            # Rescale back to original range
            home_predicted_scores = np.round(
                scalers["HT_SC"].inverse_transform(
                    predicted_scores_validate[:, 0].reshape(-1, 1)
                )
            )
            away_predicted_scores = np.round(
                scalers["AT_SC"].inverse_transform(
                    predicted_scores_validate[:, 1].reshape(-1, 1)
                )
            )

            # Calculate metrics
            home_mse_test = mean_squared_error(y_test[:, 0], home_predicted_scores)
            home_MAE_test = mean_absolute_error(y_test[:, 0], home_predicted_scores)
            home_R2val_test = r2_score(y_test[:, 0], home_predicted_scores)

            away_mse_test = mean_squared_error(y_test[:, 1], away_predicted_scores)
            away_MAE_test = mean_absolute_error(y_test[:, 1], away_predicted_scores)
            away_R2val_test = r2_score(y_test[:, 1], away_predicted_scores)

            print(
                "RMSE Home = {}, Away = {}".format(
                    math.sqrt(home_mse_test), math.sqrt(away_mse_test)
                )
            )
            print("MAE Home = {}, Away = {}".format(home_MAE_test, away_MAE_test))
            print(
                "R2 Value Home = {}, Away = {}".format(home_R2val_test, away_R2val_test)
            )

        else:
            model = load_model(file_path)

        return model

    def scale_data(self, data: pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray]:

        ## Scaling data so it is normalized and ready for ingestion ##
        X_scaled = data[['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_GD', 'AT_GD']].values
        y_scaled = data[["HT_SC", "AT_SC"]].values.astype(float)

        columns_in_input = ['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_GD', 'AT_GD', 'HT_SC', 'AT_SC']

        # Scale features
        scalers = {}
        index = 0
        for column in columns_in_input:
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
            index += 1

        return scalers, X_scaled, y_scaled

    def prep_pred_input(
        self, date: str, home_team: str, away_team: str, scalers: dict
    ) -> np.array:
        file_path = hf_hub_download(
            repo_id=self.huggingface_model, filename=self.epl_fixture_data_filepath
        )

        date_formatted = datetime.strptime(date, "%Y-%m-%d")
        home_val = self.get_team_sorted_val(home_team)
        away_val = self.get_team_sorted_val(away_team)

        current_date = datetime.now().date()

        print(f"Preparing prediction input for {home_team} vs {away_team} on {date}")

        if date_formatted.date() < current_date:
            if not os.path.exists(file_path):
                print("Data needed, scrape it and store it in order to get input")
                return np.array([])

            fixtures = pd.read_csv(file_path)

            try:
                matching_input = fixtures[
                    (fixtures["DATE"] == date_formatted)
                    & (fixtures["HT"] == home_val)
                    & (fixtures["AT"] == away_val)
                ]
                print(f"Matching input found: {matching_input}")
            except KeyError as e:
                print(f"KeyError: {e}")
                print("Match could not be found in data source, scrape more data or check inputs.")
                return np.array([])

            if matching_input.empty:
                print("No matching input found in fixtures data.")
                return np.array([])

            input_data = matching_input[['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_GD', 'AT_GD']].values.astype(float)

            for i, column in enumerate(scalers.keys()):
                if i < input_data.shape[1]:
                    input_data[:, i] = scalers[column].transform(input_data[:, i].reshape(-1, 1)).reshape(1, -1)

            return input_data
        else:
            prem_table_current = pd.read_csv(file_path)

            home_data = prem_table_current[prem_table_current['HT'] == home_val]
            away_data = prem_table_current[prem_table_current['AT'] == away_val]

            if home_data.empty or away_data.empty:
                print(f"Data for {home_team} (value: {home_val}) or {away_team} (value: {away_val}) not found in the current table.")
                return np.array([])

            home_elo, home_gd = home_data[['HT_ELO', 'HT_GD']].values[0]
            away_elo, away_gd = away_data[['AT_ELO', 'AT_GD']].values[0]

            input_data = np.array([
                [
                    scalers["HT"].transform([[home_val]])[0][0],
                    scalers["AT"].transform([[away_val]])[0][0],
                    scalers["HT_ELO"].transform([[home_elo]])[0][0],
                    scalers["AT_ELO"].transform([[away_elo]])[0][0],
                    scalers["HT_GD"].transform([[home_gd]])[0][0],
                    scalers["AT_GD"].transform([[away_gd]])[0][0],
                ]
            ])

            return input_data

    def get_team_sorted_val(self, team_name: str):

        ### This dictionary orders the teams by appearances in the BPL from 2002/03 -> 2023/24. Allows categorical team input into model. ###

        ### Please adjust teams if using different league or order if you find something preferable. ###

        ### Each team should have a unique value, it is not representing any numeric quantity although the model
        # may make that assumption hence some smarts to order necessary. ###

        # If teams change name over years, make sure both names are included in dictionary with the same value #

        team_vals = {
            'Ipswich':41,
            'Arsenal':40,
            'Chelsea':39,
            'Manchester United':38,
            'Liverpool':37,
            'Tottenham Hotspur':36,
            'Everton':35,
            'Manchester City':34,
            'Aston Villa':33,
            'Newcastle':32,
            'West Ham':31,
            'Southampton':30,
            'Fulham':29,
            'Crystal Palace':28,
            'Leicester City':27,
            'West Bromwich Albion':26,
            'Swansea City':25,
            'Burnley':24,
            'Bournemouth':23,
            'Brighton':22,
            'Wolves':21,
            'Stoke City':20,
            'Sunderland':19,
            'Norwich City':18,
            'Watford':17,
            'Birmingham':16,
            'Blackburn Rovers':15,
            'Wigan Athletic':14,
            'Middlesbrough':13,
            'Bolton Wanderers':12,
            'Leeds United':11,
            'Queens Park Rangers':10,
            'Hull City':9,
            'Sheffield United':8,
            'Brentford':7,
            'Reading':6,
            'Cardiff City':5,
            'Nottingham Forest':4,
            'Huddersfield Town':3,
            'AFC Sunderland':2,
            'Luton Town':1
        }

        return team_vals[team_name]

    def get_team_name_clubelo_format(self, team_name: str):

        ### Retrieves format of team name that works for ClubElo

        club_elo_team_name_format = {
            'Ipswich':'Ipswich',
            'Arsenal': 'Arsenal',
            'Chelsea':'Chelsea',
            'Manchester United':'ManUnited',
            'Liverpool':'Liverpool',
            'Tottenham Hotspur':'Tottenham',
            'Everton':'Everton',
            'Manchester City':'ManCity',
            'Aston Villa':'AstonVilla',
            'Newcastle':'Newcastle',
            'West Ham':'WestHam',
            'Southampton':'Southampton',
            'Fulham':'Fulham',
            'Crystal Palace':'CrystalPalace',
            'Leicester City':'Leicester',
            'West Bromwich Albion':'WestBrom',
            'Swansea City':'Swansea',
            'Burnley':'Burnley',
            'Bournemouth':'Bournemouth',
            'Brighton':'Brighton',
            'Wolves':'Wolves',
            'Stoke City':'Stoke',
            'Sunderland':'Sunderland',
            'Norwich City':'Norwich',
            'Watford':'Watford',
            'Birmingham':'Birmingham',
            'Blackburn Rovers':'Blackburn',
            'Wigan Athletic':'Wigan',
            'Middlesbrough':'Middlesbrough',
            'Bolton Wanderers':'Bolton',
            'Leeds United':'Leeds',
            'Queens Park Rangers':'QPR',
            'Hull City':'Hull',
            'Sheffield United':'SheffieldUnited',
            'Brentford':'Brentford',
            'Reading':'Reading',
            'Cardiff City':'Cardiff',
            'Nottingham Forest':'Forest',
            'Huddersfield Town':'Huddersfield',
            'AFC Sunderland':'AFCSunderland',
            'Luton Town':'Luton'
        }

        return club_elo_team_name_format[team_name]

    def get_team_name_fbref_format(self, team_name: str):

        ### Retrieves format of team name that works for Fbref

        fbref_team_name_format = {
            'Ipswich':'Ipswich',
            'Arsenal': 'Arsenal',
            'Chelsea':'Chelsea',
            'Manchester United':'Manchester Utd',
            'Liverpool':'Liverpool',
            'Tottenham Hotspur':'Tottenham',
            'Everton':'Everton',
            'Manchester City':'Manchester City',
            'Aston Villa':'Aston Villa',
            'Newcastle':'Newcastle Utd',
            'West Ham':'West Ham',
            'Southampton':'Southampton',
            'Fulham':'Fulham',
            'Crystal Palace':'Crystal Palace',
            'Leicester City':'Leicester',
            'West Bromwich Albion':'WestBrom',
            'Swansea City':'Swansea',
            'Burnley':'Burnley',
            'Bournemouth':'Bournemouth',
            'Brighton':'Brighton',
            'Wolves':'Wolves',
            'Stoke City':'Stoke',
            'Sunderland':'Sunderland',
            'Norwich City':'Norwich',
            'Watford':'Watford',
            'Birmingham':'Birmingham',
            'Blackburn Rovers':'Blackburn',
            'Wigan Athletic':'Wigan',
            'Middlesbrough':'Middlesbrough',
            'Bolton Wanderers':'Bolton',
            'Leeds United':'Leeds',
            'Queens Park Rangers':'QPR',
            'Hull City':'Hull',
            'Sheffield United':'Sheffield Utd',
            'Brentford':'Brentford',
            'Reading':'Reading',
            'Cardiff City':'Cardiff',
            'Nottingham Forest':"Nott'ham Forest",
            'Huddersfield Town':'Huddersfield',
            'AFC Sunderland':'AFC Sunderland',
            'Luton Town':'Luton Town'
        }

        return fbref_team_name_format[team_name]


    def randomised_sleep_time(self, lower_bound, upper_bound):
        delay = random.uniform(lower_bound, upper_bound)
        print(f" Sleeping for {delay:.2f} seconds...")
        time.sleep(delay)
