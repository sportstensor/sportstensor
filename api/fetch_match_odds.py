import requests
import api.db as db
import schedule
import time
import logging
from datetime import datetime, timezone
from api.config import ODDS_API_KEY

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
            home_team = odds["home_team"]
            away_team = odds["away_team"]
            commence_time = odds["commence_time"]
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

def get_odds_by_match(all_odds, match):
    homeTeamName = match.get("homeTeamName")
    awayTeamName = match.get("awayTeamName")
    matchDate = match.get("matchDate")
    matchLeague = match.get("matchLeague")
    homeTeamOdds = match.get("homeTeamOdds")
    awayTeamOdds = match.get("awayTeamOdds")
    drawOdds = match.get("drawOdds")
    # Extract just the date part from match_date
    match_date_only = matchDate.date()

    matching_odds = []

    for odds in all_odds:
        # Extract date from commence_time
        commence_time_only = datetime.strptime(odds["commence_time"], '%Y-%m-%dT%H:%M:%SZ').date()

        if (mismatch_teams_mapping.get(odds["home_team"], odds["home_team"]) == homeTeamName and
            mismatch_teams_mapping.get(odds["away_team"], odds["away_team"]) == awayTeamName and
            commence_time_only == match_date_only and
            odds["sport_title"] == matchLeague):
            matching_odds.append(odds)

    # Check if there are any matching odds
    if matching_odds:
        match_odds = matching_odds[0]
        should_update = (match_odds['home_team_odds'] != homeTeamOdds or
                        match_odds['away_team_odds'] != awayTeamOdds or
                        match_odds['draw_odds'] != drawOdds)
    else:
        match_odds = None
        should_update = False

    return match_odds, should_update


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
    match_list = db.get_upcoming_matches()

    match_odds_data = []
    match_lookup_data = []
    
    try:
        for match in match_list:
            match_id = match.get("matchId")

            match_odds_id = db.generate_uuid()
            odds, should_update = get_odds_by_match(all_odds, match)  # Get odd for the current match
            if odds and should_update:
                match_odds_data.append((match_odds_id, odds['api_id'], odds['home_team_odds'], odds['away_team_odds'], odds['draw_odds'], current_utc_time))
                if match.get('oddsapi_id') is None:
                    match_lookup_data.append((match_id, odds['api_id']))

        if match_odds_data:
            result1 = db.insert_match_odds_bulk(match_odds_data)
            result2 = db.insert_match_lookups_bulk(match_lookup_data)
            if result1:
                logging.info(f"Inserted odds data for {len(match_odds_data)} matches into the match odds database")
                for match_odd in match_odds_data:
                    logging.info(f"Inserted odds data {match_odd[1]} into the match odds database")
            if result2:
                logging.info(f"Inserted odds data for {len(match_lookup_data)} matches into the match odds database")
                for match_lookup in match_lookup_data:
                    logging.info(f"Inserted {match_lookup[0]} match and {match_lookup[1]} odds into the match odds lookup database")
    except Exception as e:
        logging.error("Failed inserting odds data into the MySQL database", exc_info=True)

if __name__ == "__main__":
    # Schedule the function to run every 1 minute
    schedule.every(1).minutes.do(fetch_and_store_match_odds)

    # Initial fetch and store to ensure setup is correct
    fetch_and_store_match_odds()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)
