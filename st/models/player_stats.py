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

    def activate(self, matchDate, sport, league, playerName, playerTeam, playerPosition, statName, statType):
        data = self.get_data()

        scalers, X_scaled, y_scaled = self.scale_data(data)

        model = self.load_or_run_model(scalers, X_scaled, y_scaled)

        pred_input = self.prep_pred_input(
            matchDate, sport, league, playerName, playerTeam, playerPosition, statName, statType, scalers
        )

        predicted_outcome = model.predict(pred_input)

        # TODO: model outcome

        prediction = predicted_outcome

        return prediction

    def make_prediction(self):
        bt.logging.info("Predicting an individual player's stat...")
        matchDate = self.prediction.matchDate.strftime("%Y-%m-%d")
        playerName = self.prediction.playerName
        playerTeam = self.prediction.playerTeam
        playerPosition = self.prediction.playerPosition
        sport = self.prediction.sport
        league = self.prediction.league
        statName = self.prediction.statName
        statType = self.prediction.statType

        predictions = self.activate(matchDate, sport, league, playerName, playerTeam, playerPosition, statName, statType)

        if predictions:
            self.prediction.statValue = predictions
        else:
            bt.logging.warning(
                "Failed to get predictions from model, setting random value"
            )
            self.prediction.statValue = 0

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

