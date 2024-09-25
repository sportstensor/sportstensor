import requests
import db as db
from api.config import API_KEYS

# Fetch JSON data from the API
def fetch_odds():
    api_url = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/odds/"
    params = {
        "apiKey": API_KEYS[0],
        "regions": "uk",
        "bookmakers": "pinnacle"
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch odds:", response.status_code)
        return None


# Fetch the odds data
odds_data = fetch_odds()

# Insert the odds data into the database
if odds_data:
    db.insert_odds(odds_data)
