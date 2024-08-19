import random
import bittensor as bt

import os
from datetime import datetime
import time
from typing import Tuple
import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download

from keras._tf_keras.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
)
from matplotlib import pyplot as plt

from st.sport_prediction_model import SportPredictionModel


class PlayerStatsPredictionModel(SportPredictionModel):
    def __init__(self, prediction):
        super().__init__(prediction)
        self.huggingface_model = "sportstensor/basic_model" # TODO: train and add a basic model here
        self.mlb_current_team_data_filepath = "mlb/fixture_data.xlsx"
        self.mls_model_filepath = "mlb/basic_model.keras"
        self.mlb_model_ready_data_comb_filepath = "mlb/combined_table.csv"

    def load_or_run_model(
        self, scalers: dict, X_scaled: np.ndarray, y_scaled: np.ndarray
    ):
        file_path = hf_hub_download(
            repo_id=self.huggingface_model, filename=self.mls_model_filepath
        )

        model = load_model(file_path)
        print(f"Model loaded from {file_path}")

        return model

    def get_data(self) -> pd.DataFrame:

        file_path = hf_hub_download(
            repo_id=self.huggingface_model,
            filename=self.mlb_model_ready_data_comb_filepath,
        )

        data = pd.read_csv(file_path)

        return data

    def activate(self, matchDate, sport, league, homeTeamName, awayTeamName, playerName, statNames):
        data = self.get_data()

        scalers, X_scaled, y_scaled = self.scale_data(data)

        model = self.load_or_run_model(scalers, X_scaled, y_scaled)

        pred_input = self.prep_pred_input(
            matchDate, sport, league, homeTeamName, awayTeamName, playerName, statNames, scalers
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
        bt.logging.info("Predicting MLB match...")
        matchDate = self.prediction.matchDate.strftime("%Y-%m-%d")
        homeTeamName = self.prediction.homeTeamName
        awayTeamName = self.prediction.awayTeamName
        statNames = self.prediction.statNames
        sport = self.prediction.sport
        league = self.prediction.league
        playerName = self.prediction.playerName

        predictions = self.activate(matchDate, sport, league, homeTeamName, awayTeamName, playerName, statNames)

        if (
            predictions is not None
            and homeTeamName in predictions
            and awayTeamName in predictions
        ):
            # Set our final predictions
            bt.logging.info("Setting final predictions from model...")
            self.prediction.statValues = int(predictions[homeTeamName])
        else:
            bt.logging.warning(
                "Failed to get predictions from model, setting random scores"
            )
            self.prediction.homeTeamScore = random.randint(0, 10)
            self.prediction.awayTeamScore = random.randint(0, 10)

        # print(f"Assigned final predictions: Home={predictions[homeTeamName]}, Away={predictions[awayTeamName]}")
        return True

    def scale_data(self, data: pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray]:
        pass

    def prep_pred_input(
        self, date: str, home_team: str, away_team: str, scalers: dict
    ) -> np.array:
        pass

    def update_current_team_database():
        ### Update database for current statistics ###
        print("Database updated successfully")

