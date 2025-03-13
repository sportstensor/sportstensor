import requests
import api.db as db
import logging
import schedule
import time
from api.config import ODDS_API_KEY
import api.fetch_odds as fo

league_sports_types_mapping = {
    'English Premier League': 'soccer_epl',
    'American Major League Soccer': 'soccer_usa_mls',
    'NFL': 'americanfootball_nfl',
    'MLB': 'baseball_mlb',
    'NBA': 'basketball_nba',
}

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_odds(api_url, event_id, start):
    # Initialize the current interval index
    current_index = 0

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,eu,uk",
        "date": start
    }
    logging.info(f"Fetching odds for event {event_id} start============>{start}")
    
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        odds = data.get("data")
        odds = [odds] if odds else []
        reduced_odds = fo.get_reduced_odds(odds)
        if reduced_odds:
            lastUpdated = reduced_odds[0]["last_updated"]
            stored_odds = db.get_stored_odds(lastUpdated)
            fo.store_match_odds(reduced_odds, stored_odds)
    else:
        logging.error("Failed to fetch odds: %s", response.status_code)
        return

def fetch_tipoff_odds():
    live_matches = db.get_live_matches()
    
    if not live_matches:
        logging.info("No live matches found. Skipping fetching tip off odds.")
        return
    
    logging.info(f"Fetching tip off odds for {len(live_matches)} live matches")
    for match in live_matches:
        matchDate = match['matchDate']
        matchLeague = match['matchLeague']
        oddsapiMatchId = match['oddsapiMatchId']
        t_0 = matchDate.strftime("%Y-%m-%dT%H:%M:%SZ")
    
        api_url = f"https://api.the-odds-api.com/v4/historical/sports/{league_sports_types_mapping.get(matchLeague, matchLeague)}/events/{oddsapiMatchId}/odds/"
        fetch_odds(api_url, oddsapiMatchId, t_0)

if __name__ == "__main__":
    # Schedule the function to run every 15 minutes
    schedule.every(15).minutes.do(fetch_tipoff_odds)

    # Initial fetch and store to ensure setup is correct
    fetch_tipoff_odds()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)
