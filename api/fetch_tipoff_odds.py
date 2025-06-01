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

def fetch_historical_tipoff_odds():
    #league = "MLB"
    #start_date = "2025-03-27 00:00:00"
    #end_date = None

    #league = "English Premier League"
    #start_date = "2024-08-16 00:00:00"
    #end_date = None

    league = "NBA"
    start_date = "2024-10-21 00:00:00"
    end_date = None

    #league = "American Major League Soccer"
    #start_date = "2025-02-21 00:00:00"
    #end_date = None

    matches = db.get_completed_matches(start_date=start_date, end_date=end_date, league=league, no_spread_odds=True)
    
    if not matches:
        logging.info("No matches found. Skipping fetching historical tip off odds.")
        return
    
    logging.info(f"Fetching tip off odds for {len(matches)} {league} matches...")
    for match in matches:
        matchDate = match['matchDate']
        matchLeague = match['matchLeague']
        oddsapiMatchId = match['oddsapiMatchId']
        t_0 = matchDate.strftime("%Y-%m-%dT%H:%M:%SZ")
    
        api_url = f"https://api.the-odds-api.com/v4/historical/sports/{league_sports_types_mapping.get(matchLeague, matchLeague)}/events/{oddsapiMatchId}/odds/"
        fetch_odds(api_url, oddsapiMatchId, t_0, do_h2h=False, do_spread=True)
        time.sleep(1)  # Sleep to avoid hitting API rate limits too quickly

def fetch_odds(api_url, event_id, start, do_h2h=True, do_spread=True):
    # Initialize the current interval index
    current_index = 0
    
    markets = []
    if do_h2h:
        markets.append("h2h")
    if do_spread:
        markets.append("spreads")
    if not markets:
        logging.error("No markets specified for fetching odds.")
        return
    # Convert the markets list to a comma-separated string
    markets_str = ",".join(markets)

    # eu region should be all be need because it includes Pinnacle and others and we only get the first available from others if Pinnacle is not available
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": markets_str,
        "date": start
    }
    logging.info(f"Fetching odds for event {event_id} start============>{start}")
    
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        odds = data.get("data")
        odds = [odds] if odds else []

        if do_h2h:
            reduced_odds = fo.get_reduced_odds(odds)
            if reduced_odds:
                lastUpdated = reduced_odds[0]["last_updated"]
                stored_odds = db.get_stored_odds(lastUpdated)
                fo.store_match_odds(reduced_odds, stored_odds)

        if do_spread:
            reduced_spread_odds = fo.get_reduced_spread_odds(odds)
            if reduced_spread_odds:
                lastUpdated = reduced_spread_odds[0]["last_updated"]
                stored_spread_odds = db.get_stored_spread_odds(lastUpdated)
                fo.store_match_spread_odds(reduced_spread_odds, stored_spread_odds)
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
    #fetch_historical_tipoff_odds()
    #records = db.get_team_records("Milwaukee Bucks", "NBA", "2024-10-21 00:00:00", "2025-04-14 00:00:00")
    #print(records)
    
    # Schedule the function to run every 15 minutes
    schedule.every(15).minutes.do(fetch_tipoff_odds)

    # Initial fetch and store to ensure setup is correct
    fetch_tipoff_odds()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)
