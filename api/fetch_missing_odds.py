import requests
import api.db as db
import logging
from api.config import ODDS_API_KEY
import json
from datetime import timedelta, datetime
import time
from collections import defaultdict
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

# Function to merge overlapping intervals
def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]

        # Check if there is an overlap
        if current_start <= last_end:
            # Merge the intervals
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged

def fetch_odds(api_url, start, intervals):
    # Initialize the current interval index
    current_index = 0

    while current_index < len(intervals):
        # Get the current interval
        current_interval = intervals[current_index]
        interval_start, interval_end = current_interval

        # Check if the start is within the current interval
        if not (interval_start <= start <= interval_end):
            logging.error("Start timestamp is outside the interval.")
            return

        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us,eu,uk",
            "date": start
        }
        logging.info(f"start============>{start}")
        
        response = requests.get(api_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            next_timestamp = data.get("next_timestamp")
            odds = data.get("data")
            reduced_odds = fo.get_reduced_odds(odds)
            if reduced_odds:
                lastUpdated = reduced_odds[0]["last_updated"]
                stored_odds = db.get_stored_odds(lastUpdated)
                fo.store_match_odds(reduced_odds, stored_odds)
            # Check if next_timestamp is within the current interval
            if next_timestamp and interval_start <= next_timestamp <= interval_end:
                # Continue with the same interval
                start = next_timestamp
            else:
                # Move to the next interval
                current_index += 1
                if current_index < len(intervals):
                    start = intervals[current_index][0]  # Reset start to the beginning of the next interval
                else:
                    logging.info("No more intervals to process. Stopping.")
                    logging.info(f"Next timestamp===>{next_timestamp}")
                    break
        else:
            logging.error("Failed to fetch odds: %s", response.status_code)
            return

    logging.info("All intervals processed.")

def fetch_missing_odds():
    completed_matches = db.get_matches_with_missing_odds()
    history_intervals = []
    for completed_match in completed_matches:
        matchDate = completed_match['matchDate']
        matchLeague = completed_match['matchLeague']
        oddsapiMatchId = completed_match['oddsapiMatchId']
        t_24 = (matchDate - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
        t_0 = matchDate.strftime("%Y-%m-%dT%H:%M:%SZ")
        if completed_match['oddsData']:
            oddsData = json.loads(completed_match['oddsData'])
            # datetime_objects = [datetime.strptime(odds.get('lastUpdated'), "%Y-%m-%d %H:%M:%S.%f") for odds in oddsData]
            datetime_objects = [
                datetime.strptime(odds['lastUpdated'], "%Y-%m-%d %H:%M:%S.%f")
                for odds in oddsData 
                if odds.get('lastUpdated') is not None
            ]
            if datetime_objects:
                min_timeStamp = min(datetime_objects).strftime("%Y-%m-%dT%H:%M:%SZ")
                max_timeStamp = max(datetime_objects).strftime("%Y-%m-%dT%H:%M:%SZ")

                if max_timeStamp <= t_24 or min_timeStamp >= t_0:
                    history_intervals.append((t_24, t_0, matchLeague, oddsapiMatchId))
                elif min_timeStamp > t_24 and max_timeStamp < t_0:
                    history_intervals.append((t_24, min_timeStamp, matchLeague, oddsapiMatchId))
                    history_intervals.append((max_timeStamp, t_0, matchLeague, oddsapiMatchId))
                elif max_timeStamp < t_0 and min_timeStamp < t_24:
                    history_intervals.append((max_timeStamp, t_0, matchLeague, oddsapiMatchId))
                elif min_timeStamp > t_24 and max_timeStamp > t_0:
                    history_intervals.append((t_24, min_timeStamp, matchLeague, oddsapiMatchId))
            else:
                history_intervals.append((t_24, t_0, matchLeague, oddsapiMatchId))  
        else:
            history_intervals.append((t_24, t_0, matchLeague, oddsapiMatchId))
    
    league_intervals = defaultdict(list)
    for start, end, league, oddsapiMatchId in history_intervals:
        league_intervals[league].append((start, end))
    
    # Merge intervals by league
    merged_by_league = {}
    
    for league, intervals in league_intervals.items():
        merged_intervals = merge_intervals(intervals)
        merged_by_league[league] = [(start, end) for start, end in merged_intervals]
    
    for league, intervals in merged_by_league.items():
        api_url = f"https://api.the-odds-api.com/v4/historical/sports/{league_sports_types_mapping.get(league, league)}/odds/"
        start, end = intervals[0]
        fetch_odds(api_url, start, intervals)
        time.sleep(1)

if __name__ == "__main__":
    fetch_missing_odds()
