import requests
import api.db as db
import schedule
import time
import logging
from datetime import datetime, timezone
from api.config import ODDS_API_KEY
import pytz

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SPORTS_TYPES = ['baseball_mlb', 'americanfootball_nfl', 'soccer_usa_mls', 'soccer_epl']

league_mapping = {
    'EPL': 'English Premier League',
    'MLS': 'American Major League Soccer',
    'MLB': 'MLB',
    'NFL': 'NFL'
}

mismatch_teams_mapping = {
    'Orlando City SC': 'Orlando City',
    'Inter Miami CF': 'Inter Miami',
    'Atlanta United FC': 'Atlanta United',
    'Montreal Impact': 'CF Montréal',
    'D.C. United': 'DC United',
    'Tottenham Hotspur': 'Tottenham',
    'Columbus Crew SC': 'Columbus Crew',
    'Minnesota United FC': 'Minnesota United',
    'Vancouver Whitecaps FC': 'Vancouver Whitecaps',
    'Leicester City': 'Leicester',
    'West Ham United': 'West Ham',
    'Ipswich Town': 'Ipswich',
    'Vancouver Whitecaps FC': 'Vancouver Whitecaps',
    'Brighton and Hove Albion': 'Brighton',
    'Wolverhampton Wanderers': 'Wolves',
    'Newcastle United': 'Newcastle',
    'LA Galaxy': 'L.A. Galaxy'
}

# Fetch JSON data from the API
def fetch_odds():
    all_odds = []
    for type in SPORTS_TYPES:
        api_url = f"https://api.the-odds-api.com/v4/sports/{type}/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "bookmakers": "pinnacle"
        }
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_odds.extend(data)
        else:
            logging.error("Failed to fetch odds:", response.status_code)
            return None
    
    if not all_odds:
        logging.info("No odds data found in the API responses")
        return
    reduced_odds = []
    try:
        for odds in all_odds:
            api_id = odds["id"]  # Get the odds ID
            sport_title = league_mapping.get(odds["sport_title"], odds["sport_title"])
            home_team = mismatch_teams_mapping.get(odds["home_team"], odds["home_team"])
            away_team = mismatch_teams_mapping.get(odds["away_team"], odds["away_team"])
            commence_time = odds["commence_time"]
            home_team_odds = None
            away_team_odds = None
            draw_odds = None
            if odds["bookmakers"]:
                for bookmaker in odds["bookmakers"]:
                    if bookmaker["key"] == "pinnacle":
                        for market in bookmaker["markets"]:
                            if market["key"] == "h2h":
                                outcomes = market["outcomes"]

                                # Map odds to the correct columns
                                for outcome in outcomes:
                                    if outcome["name"] == odds["home_team"]:
                                        home_team_odds = outcome["price"]
                                    elif outcome["name"] == odds["away_team"]:
                                        away_team_odds = outcome["price"]
                                    elif outcome["name"] == "Draw":
                                        draw_odds = outcome["price"]

                # Append the odds directly since they will not be None
                reduced_odds.append({
                    "api_id": api_id,
                    "sport_title": sport_title,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_team_odds": home_team_odds,
                    "away_team_odds": away_team_odds,
                    "draw_odds": draw_odds,
                    "commence_time": commence_time
                })
        return reduced_odds

    except Exception as e:
        logging.error("Failed getting reduced odds data", exc_info=True)
        return []

def check_if_odds_should_be_stored(stored_odds, odds):
    api_id = odds.get('api_id')
    home_team_odds = odds.get('home_team_odds')
    away_team_odds = odds.get('away_team_odds')
    draw_odds = odds.get('draw_odds')
    commence_time = odds.get('commence_time')
    current_utc_time = datetime.now(timezone.utc)
    current_utc_time = current_utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Use a generator expression to find the first matching odds
    matched_odds = next((item for item in stored_odds if item['oddsapiMatchId'] == api_id), None)
    if matched_odds:
        # Check if any of the odds have changed
        matched_odds_commence = pytz.utc.localize(matched_odds['commence_time']).strftime("%Y-%m-%dT%H:%M:%SZ")
        should_update_match_odds = (
            matched_odds['homeTeamOdds'] != home_team_odds or
            matched_odds['awayTeamOdds'] != away_team_odds or
            matched_odds['drawOdds'] != draw_odds
        ) and (matched_odds_commence > current_utc_time)
        # Check if commence_time has changed
        should_update_odds = matched_odds_commence != commence_time
        return should_update_match_odds, should_update_odds
    else:
        # If no match found, indicate that new odds should be stored
        return True, True

def create_match__odds_id():
    # generate a unique uuid for the match. make sure it does not already exist.
    match_odds_id = db.generate_uuid()
    while db.match_odds_id_exists(match_odds_id):
        print(f"Match Odds Table ID {match_odds_id} already exists. Generating a new one.")
        match_odds_id = db.generate_uuid()
    return match_odds_id

def fetch_and_store_match_odds():
    logging.info(f"=============Starting to fetch and store match odds=============")
    all_odds = fetch_odds()
    current_utc_time = datetime.now(timezone.utc)
    current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")
    stored_odds = db.get_stored_odds()
    match_odds_data = []
    odds_to_store = []
    
    try:
        for odds in all_odds:
            match_odds_id = db.generate_uuid()
            should_update_match_odds, should_update_odds = check_if_odds_should_be_stored(stored_odds, odds)  # Get odd for the current match
            if should_update_match_odds:
                match_odds_data.append((match_odds_id, odds['api_id'], odds['home_team_odds'], odds['away_team_odds'], odds['draw_odds'], current_utc_time))
            if should_update_odds:
                odds_to_store.append((odds['api_id'], odds['sport_title'], odds['home_team'], odds['away_team'], odds['commence_time'], current_utc_time))

        if match_odds_data:
            result = db.insert_match_odds_bulk(match_odds_data)
            if result:
                logging.info(f"Inserted odds data for {len(match_odds_data)} matches into the match odds database")
                for match_odd in match_odds_data:
                    logging.info(f"Inserted odds data {match_odd[1]} into the match odds database")
        if odds_to_store:
            result = db.insert_odds_bulk(odds_to_store)
            if result:
                logging.info(f"Inserted odds data for {len(odds_to_store)} matches into the match odds database")
                for odds in odds_to_store:
                    logging.info(f"Inserted {odds[0]} odds for a match(homeTeam: {odds[2]}, awayTeam: {odds[3]}) into the odds database")
    except Exception as e:
        logging.error("Failed inserting odds data into the MySQL database", exc_info=True)

if __name__ == "__main__":
    # Schedule the function to run every 4 minutes
    schedule.every(4).minutes.do(fetch_and_store_match_odds)

    # Initial fetch and store to ensure setup is correct
    fetch_and_store_match_odds()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)
