import pandas as pd
import numpy as np
from typing import Tuple
import ScraperFC
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import os
import openpyxl
from sklearn.preprocessing import MinMaxScaler

def get_data(more_data:bool) -> pd.DataFrame:       
    
    file_path = 'data_and_models/fixtures_advanced.xlsx'
    if more_data:

        current_year = datetime.now().year 
        years_back = 0 ## roughly 380 fixtures a year ## default 10    
        years_of_interest = [int(year) for year in range(current_year - years_back, current_year + 1) if year != 2020]
        league_of_interest = 'EPL' # Other selections: "Ligue-1", "Bundesliga", "Serie-A", "La-Liga"
        row_index = 0

        fixtures = pd.DataFrame(columns = ['HT-AT-DATE', 'DATE', 'HT', 'AT', 'HT_GD', 'AT_GD', 'HT_ELO', 'AT_ELO', 'HT_SC', 'AT_SC'])
        for year in years_of_interest:
            url_retrieval = ScraperFC.FBRef()
            urls = url_retrieval.get_match_links(year = year, league = league_of_interest)   

            goal_diff_dict = {key: 0 for key in range(1, 41)}

            for match_url in urls:
                match_input = {}
                index = match_url.split('/')[-1].split('-Premier-League')[0]
                match_input['HT-AT-DATE'] = index
                html = requests.get(match_url)
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
                match_input['HT'] = get_team_sorted_val(home_name)
                home_name_elo = get_team_name_clubelo_format(home_name)

                away_name = names[1].split(' Match Report')[0]
                match_input['AT'] = get_team_sorted_val(away_name)
                away_name_elo = get_team_name_clubelo_format(away_name)

                ## Retrieving goal difference prior to match and adding values post match ##
                match_input['HT_GD'] = goal_diff_dict[get_team_sorted_val(home_name)]
                goal_diff_dict[get_team_sorted_val(home_name)] += home_goal_difference

                match_input['AT_GD'] = goal_diff_dict[get_team_sorted_val(away_name)]
                goal_diff_dict[get_team_sorted_val(away_name)] -= home_goal_difference

                ## Formatting date of fixture ##
                date_basic = index.split('-')[-3:]
                month_num = datetime.strptime(date_basic[0], "%B").month
                date_string = date_basic[2] + '-' + str(month_num) + '-' + date_basic[1]            
                match_input['DATE'] = datetime.strptime(date_string, '%Y-%m-%d')

                ## Retrieving ELO ##
                away_elo = ScraperFC.ClubElo()
                away_elo = away_elo.scrape_team_on_date(away_name_elo, date_string)
                home_elo = ScraperFC.ClubElo()
                home_elo = home_elo.scrape_team_on_date(home_name_elo, date_string)
                match_input['HT_ELO'] = home_elo
                match_input['AT_ELO'] = away_elo

                ## Adding to dataframe and adjusting index ##
                fixtures.loc[row_index] = match_input
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

def scale_data(data:pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray]:

    ## Scaling data so it is normalized and ready for ingestion ##
    X_scaled = data[['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_GD', 'AT_GD']].values
    y_scaled = data[['HT_SC', 'AT_SC']].values.astype(float)

    columns_in_input = ['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_GD', 'AT_GD', 'HT_SC', 'AT_SC']

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

def prep_pred_input(date:str, home_team:str, away_team:str, scalers:dict) -> np.array:

    date_formatted = datetime.strptime(date, '%Y-%m-%d')
    home_val = get_team_sorted_val(home_team)
    away_val = get_team_sorted_val(away_team)

    current_date = datetime.now().date()

    if date_formatted.date() < current_date:
        
        file_path = 'data_and_models/fixtures_advanced.xlsx'
        if not os.path.exists(file_path):
            print('Data needed, scrape it and store it in order to get input')
            input = 0
        else:

            fixtures = pd.read_excel(file_path)

            try:
                matching_input = fixtures[(fixtures['DATE'] == date_formatted) & (fixtures['HT'] == home_val) & (fixtures['AT'] == away_val)]
            except:
                matching_input = 0
                print('Match could not be found in data source, scrape more data or check inputs.')  

            input = matching_input[['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_GD', 'AT_GD']].values.astype(float)
            
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

        home_name_elo = get_team_name_clubelo_format(home_team)
        home_elo = ScraperFC.ClubElo()
        home_elo = home_elo.scrape_team_on_date(home_name_elo, date)

        away_name_elo = get_team_name_clubelo_format(away_team)
        away_elo = ScraperFC.ClubElo()
        away_elo = away_elo.scrape_team_on_date(away_name_elo, date)

        input['HT_ELO'] = home_elo
        input['AT_ELO'] = away_elo
   
        prem_table_current = pd.read_html('https://fbref.com/en/comps/9/Premier-League-Stats')[0]
        home_team_fb = get_team_name_fbref_format(home_team)
        home_team_row = prem_table_current.loc[prem_table_current['Squad'] == home_team_fb]
        home_gd = home_team_row['GD'].values[0]
        input['HT_GD'] = home_gd  
        away_team_fb = get_team_name_fbref_format(away_team)
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

def get_team_sorted_val(team_name:str):

    ### This dictionary orders the teams by appearances in the BPL from 2002/03 -> 2023/24. Allows categorical team input into model. ###
    
    ### Please adjust teams if using different league or order if you find something preferable. ### 

    ### Each team should have a unique value, it is not representing any numeric quantity although the model 
    # may make that assumption hence some smarts to order necessary. ###

    team_vals = {
        'Arsenal':40,
        'Chelsea':39,
        'Manchester United':38,
        'Liverpool':37,
        'Tottenham Hotspur':36,
        'Everton':35,
        'Manchester City':34,
        'Aston Villa':33,
        'Newcastle United':32,
        'West Ham United':31,
        'Southampton':30,
        'Fulham':29,
        'Crystal Palace':28,
        'Leicester City':27,
        'West Bromwich Albion':26,
        'Swansea City':25,
        'Burnley':24,
        'Bournemouth':23,
        'Brighton & Hove Albion':22,
        'Wolverhampton Wanderers':21,
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

def get_team_name_clubelo_format(team_name:str):

    ### Retrieves format of team name that works for ClubElo

    club_elo_team_name_format = {
        'Arsenal': 'Arsenal',
        'Chelsea':'Chelsea',
        'Manchester United':'ManUnited',
        'Liverpool':'Liverpool',
        'Tottenham Hotspur':'Tottenham',
        'Everton':'Everton',
        'Manchester City':'ManCity',
        'Aston Villa':'AstonVilla',
        'Newcastle United':'Newcastle',
        'West Ham United':'WestHam',
        'Southampton':'Southampton',
        'Fulham':'Fulham',
        'Crystal Palace':'CrystalPalace',
        'Leicester City':'Leicester',
        'West Bromwich Albion':'WestBrom',
        'Swansea City':'Swansea',
        'Burnley':'Burnley',
        'Bournemouth':'Bournemouth',
        'Brighton & Hove Albion':'Brighton',
        'Wolverhampton Wanderers':'Wolves',
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


def get_team_name_fbref_format(team_name:str):

    ### Retrieves format of team name that works for ClubElo

    fbref_team_name_format = {
        'Arsenal': 'Arsenal',
        'Chelsea':'Chelsea',
        'Manchester United':'Manchester Utd',
        'Liverpool':'Liverpool',
        'Tottenham Hotspur':'Tottenham',
        'Everton':'Everton',
        'Manchester City':'Manchester City',
        'Aston Villa':'Aston Villa',
        'Newcastle United':'Newcastle Utd',
        'West Ham United':'West Ham',
        'Southampton':'Southampton',
        'Fulham':'Fulham',
        'Crystal Palace':'Crystal Palace',
        'Leicester City':'Leicester',
        'West Bromwich Albion':'WestBrom',
        'Swansea City':'Swansea',
        'Burnley':'Burnley',
        'Bournemouth':'Bournemouth',
        'Brighton & Hove Albion':'Brighton',
        'Wolverhampton Wanderers':'Wolves',
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
