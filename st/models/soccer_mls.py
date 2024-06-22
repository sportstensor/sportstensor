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
import requests
import ScraperFC
import openpyxl
from huggingface_hub import hf_hub_download

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from st.models.soccer import SoccerPredictionModel


class MLSSoccerPredictionModel(SoccerPredictionModel):
    def __init__(self, prediction):
        super().__init__(prediction)
        self.huggingface_model = "sportstensor/basic_mls_model"
        self.mls_fixture_data_filepath = "mls_fixture_data.xlsx"
        self.mls_model_filepath = "basic_mls_model.keras"

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
        scrape_more_data = False
        data = self.get_data(scrape_more_data)

        scalers, X_scaled, y_scaled = self.scale_data(data)

        model = self.load_or_run_model(scalers, X_scaled, y_scaled)

        pred_input, hist_score = self.prep_pred_input(matchDate, homeTeamName, awayTeamName, scalers)

        predicted_outcome = model.predict(pred_input)
            
        predicted_outcome[:,0] = np.round(scalers['HT_SC'].inverse_transform(predicted_outcome[:, 0].reshape(-1, 1)).reshape(-1))
        predicted_outcome[:,1] = np.round(scalers['AT_SC'].inverse_transform(predicted_outcome[:, 1].reshape(-1, 1)).reshape(-1))

        # Ensure that predictions dictionary is always returned
        predictions = {(homeTeamName, awayTeamName): (predicted_outcome[0][0], predicted_outcome[0][1])}
        # print({'DATE': match_datetime, 'HT': homeTeamName, 'AT': awayTeamName},
        #      'predicted score:', predicted_outcome[0], 'actual score:', hist_score) 

        return predictions
    
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
    
    def get_data(self, more_data:bool) -> pd.DataFrame:

        file_path = hf_hub_download(repo_id=self.huggingface_model, filename=self.mls_fixture_data_filepath)
        
        if more_data:

            current_year = datetime.now().year 
            years_back = 0 ## roughly 380 fixtures a year ## default 10    
            years_of_interest = [int(year) for year in range(current_year - years_back, current_year + 1) if year != 2020]
            league_of_interest = 'MLS' # Other selections: "Ligue-1", "Bundesliga", "Serie-A", "La-Liga"
            row_index = 0

            fixtures = pd.DataFrame(columns = ['HT-AT-DATE', 'DATE', 'HT', 'AT', 'HT_GD', 'AT_GD', 'HT_SC', 'AT_SC']) # 'HT_ELO', 'AT_ELO'
            for year in years_of_interest:
                url_retrieval = ScraperFC.FBRef()
                urls = url_retrieval.get_match_links(year = year, league = league_of_interest)   

                ## Change range to +1 than the amount of teams in dictionaries (32 + 1)
                goal_diff_dict = {key: 0 for key in range(1, 33)}
            
                with open('proxies.txt', 'r') as file:
                    proxy_list = file.readlines()
                proxy_index_counter = 1

                scraped_urls = []
                for match_url in urls:

                    if match_url in scraped_urls:
                        break

                    match_input = {}
                    index = match_url.split('/')[-1].split('-Major-League-Soccer')[0]
                    match_input['HT-AT-DATE'] = index

                    ##### ADD PROXY LOGIC #######################################################
                    if proxy_index_counter == len(proxy_list):
                        proxy_index_counter = 1
                    proxy_for_match = proxy_list[proxy_index_counter - 1]
                    proxy_for_match = proxy_for_match[:-1]
                    proxy_index_counter += 1
                    proxy_parts = proxy_for_match.split(':')
                    proxy_html = proxy_parts[2] + ':' + proxy_parts[3] + '@' + proxy_parts[0] + ':' + proxy_parts[1]
                    proxy_html_dict = {
                        'http' : 'http://' + proxy_html #, 'https' : 'https://' + proxy_html
                        }
                    ##############################################################################

                    html = requests.get(match_url, proxies = proxy_html_dict)
                    self.randomised_sleep_time(1,7)
                    soup = BeautifulSoup(html.text, 'html.parser')

                    ## Get Scores and calculate GD ## 
                    scores = soup.find_all('div', {'class': 'score'})
                    if len(scores) == 0:
                        continue

                    home_goals = int(scores[0].text)
                    away_goals = int(scores[1].text)
                    match_input['HT_SC'] = home_goals
                    match_input['AT_SC'] = away_goals
                    home_goal_difference = home_goals - away_goals

                    ## Get Team Names and assigning team values ##
                    names = soup.find('div', {'id' : 'content'}).text.split(' vs. ')

                    home_name = names[0].split('\n')[1]
                    match_input['HT'] = self.get_team_sorted_val(home_name)
                    #home_name_elo = get_team_name_clubelo_format(home_name)

                    away_name = names[1].split(' Match Report')[0]
                    match_input['AT'] = self.get_team_sorted_val(away_name)
                    #away_name_elo = get_team_name_clubelo_format(away_name)

                    ## Retrieving goal difference prior to match and adding values post match ##
                    match_input['HT_GD'] = goal_diff_dict[self.get_team_sorted_val(home_name)]
                    goal_diff_dict[self.get_team_sorted_val(home_name)] += home_goal_difference

                    match_input['AT_GD'] = goal_diff_dict[self.get_team_sorted_val(away_name)]
                    goal_diff_dict[self.get_team_sorted_val(away_name)] -= home_goal_difference

                    ## Formatting date of fixture ##
                    date_basic = index.split('-')[-3:]
                    month_num = datetime.strptime(date_basic[0], "%B").month
                    date_string = date_basic[2] + '-' + str(month_num) + '-' + date_basic[1]            
                    match_input['DATE'] = datetime.strptime(date_string, '%Y-%m-%d')

                    ## Retrieving ELO ##
                    # away_elo = ScraperFC.ClubElo()
                    # away_elo = away_elo.scrape_team_on_date(away_name_elo, date_string)
                    # home_elo = ScraperFC.ClubElo()
                    # home_elo = home_elo.scrape_team_on_date(home_name_elo, date_string)
                    # match_input['HT_ELO'] = home_elo
                    # match_input['AT_ELO'] = away_elo

                    ## Adding to dataframe and adjusting index ##
                    fixtures.loc[row_index] = match_input
                    scraped_urls.append(match_url)
                    print('Match Added: ' + index)
                    row_index += 1
                
                fixtures = fixtures[(fixtures != -1).all(axis=1)]
                if os.path.exists(file_path):
                    existing_df = pd.read_excel(file_path)
                    fixtures = pd.concat([existing_df, fixtures], ignore_index=True)
                    fixtures = fixtures.drop_duplicates(subset='HT-AT-DATE', keep='last')        
                fixtures.to_excel(file_path, index=False)
                print('Year added to excel - ' + str(year) + ', df length = ' + str(len(fixtures)))
        else:        
            if os.path.exists(file_path):
                fixtures = pd.read_excel(file_path)
            else:
                print('No file found, scrape data first.')
                fixtures = pd.DataFrame()    

        return fixtures

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

        date_formatted = datetime.strptime(date, '%Y-%m-%d')
        home_val = self.get_team_sorted_val(home_team)
        away_val = self.get_team_sorted_val(away_team)

        current_date = datetime.now().date()

        if date_formatted.date() < current_date:
            
            if not os.path.exists(self.mls_fixture_data_filepath):
                print('Data needed, scrape it and store it in order to get input')
                input = 0
            else:
                fixtures = pd.read_excel(self.mls_fixture_data_filepath)

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
        else:        

            input = {}
            input['HT'] = home_val
            input['AT'] = away_val

            # home_name_elo = get_team_name_clubelo_format(home_team)
            # home_elo = ScraperFC.ClubElo()
            # home_elo = home_elo.scrape_team_on_date(home_name_elo, date)

            # away_name_elo = get_team_name_clubelo_format(away_team)
            # away_elo = ScraperFC.ClubElo()
            # away_elo = away_elo.scrape_team_on_date(away_name_elo, date)

            # input['HT_ELO'] = home_elo
            # input['AT_ELO'] = away_elo
    
            # Fetch all tables from the webpage
            url = 'https://fbref.com/en/comps/22/Major-League-Soccer-Seasons'
            tables = pd.read_html(url)

            # Combine tables 0 and 2 (assuming they have the same structure)
            prem_table_current = pd.concat([tables[0], tables[2]], ignore_index=True)

            home_team_fb = self.get_team_name_fbref_format(home_team)
            home_team_row = prem_table_current.loc[prem_table_current['Squad'] == home_team_fb]
            home_gd = home_team_row['GD'].values[0]
            input['HT_GD'] = home_gd  
            away_team_fb = self.get_team_name_fbref_format(away_team)
            away_team_row = prem_table_current[prem_table_current['Squad'] == away_team_fb]
            away_gd = away_team_row['GD'].values[0]
            input['AT_GD'] = away_gd
            
            input = np.array(list(input.values())).reshape(1,-1)
            index = 0
            for column in scalers.keys():                
                if index < input.shape[1]:
                    input[:,index] = scalers[column].transform(input[:,index].reshape(-1,1)).reshape(1,-1)
                index += 1

            output = 'Outcome not known yet as game not taken place'

        return input, output

    def get_team_sorted_val(self, team_name:str):

        ### This dictionary orders the teams by appearances in the BPL from 2002/03 -> 2023/24. Allows categorical team input into model. ###
        
        ### Please adjust teams if using different league or order if you find something preferable. ### 

        ### Each team should have a unique value, it is not representing any numeric quantity although the model 
        # may make that assumption hence some smarts to order necessary. ###

        # If teams change name over years, make sure both names are included in dictionary with the same value #

        team_vals = {
            'D.C. United': 32,
            'LA Galaxy': 31,
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
