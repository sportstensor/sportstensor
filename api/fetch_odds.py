import requests
import db as db
from api.config import ODDS_API_KEY
import schedule
import time

SPORTS_TYPES = ['baseball_mlb', 'americanfootball_nfl', 'soccer_usa_mls', 'soccer_epl']

# Fetch JSON data from the API
def fetch_and_store_odds():
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
            db.insert_odds(data)
        else:
            print("Failed to fetch odds:", response.status_code)
            return None

if __name__ == "__main__":
    # Schedule the function to run every 10 minutes
    schedule.every(1440).minutes.do(fetch_and_store_odds)

    # Initial fetch and store to ensure setup is correct
    fetch_and_store_odds()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)