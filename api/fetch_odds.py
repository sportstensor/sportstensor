import requests
import api.db as db
from api.config import ODDS_API_KEY
import schedule
import time
from datetime import datetime
import logging

SPORTS_TYPES = ['baseball_mlb', 'americanfootball_nfl', 'soccer_usa_mls', 'soccer_epl']

league_mapping = {
    'EPL': 'English Premier League',
    'MLS': 'American Major League Soccer',
    'MLB': 'MLB',
    'NFL': 'NFL'
}

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Fetch JSON data from the API
def fetch_and_store_odds():
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
    try:
        for odds in all_odds:
            api_id = odds["id"]  # Get the odds ID
            sport_title = league_mapping.get(odds["sport_title"], odds["sport_title"])
            home_team = odds["home_team"]
            away_team = odds["away_team"]
            commence_time_str = odds["commence_time"]
            commence_time = datetime.strptime(commence_time_str, '%Y-%m-%dT%H:%M:%SZ')
            home_team_odds = None
            away_team_odds = None
            draw_odds = None
            for bookmaker in odds["bookmakers"]:
                if bookmaker["key"] == "pinnacle":
                    for market in bookmaker["markets"]:
                        if market["key"] == "h2h":
                            outcomes = market["outcomes"]

                            # Map odds to the correct columns
                            for outcome in outcomes:
                                if outcome["name"] == home_team:
                                    home_team_odds = outcome["price"]
                                elif outcome["name"] == away_team:
                                    away_team_odds = outcome["price"]
                                elif outcome["name"] == "Draw":
                                    draw_odds = outcome["price"]
            
            dbresult = db.insert_odds(
                api_id, sport_title, home_team, away_team, home_team_odds, away_team_odds, draw_odds, commence_time
            )

            if dbresult:
                matchId = db.query_match_id_with_odds_data(home_team, away_team, sport_title, commence_time)
                if matchId:
                    result = db.insert_odds_match_lookup(matchId, api_id)
                    if result:
                        logging.info(f"Inserted matchId {matchId} and oddsAPIMatchId {api_id} lookup into the database")                

    except Exception as e:
        logging.error("Failed inserting odds into the MySQL database", exc_info=True)

if __name__ == "__main__":
    # Schedule the function to run every 10 minutes
    schedule.every(1440).minutes.do(fetch_and_store_odds)

    # Initial fetch and store to ensure setup is correct
    fetch_and_store_odds()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)