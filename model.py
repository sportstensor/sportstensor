import os
import numpy as np
import math
import pandas as pd
import openpyxl
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from huggingface_hub import hf_hub_download

def load_or_run_model(scalers: dict, X_scaled: np.ndarray, y_scaled: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    repo_id = "sportstensor/basic_mls_model"
    filename = "basic_mls_model.keras"

    file_path = hf_hub_download(repo_id=repo_id, filename=filename)

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
