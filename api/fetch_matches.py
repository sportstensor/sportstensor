import requests
import api.db as db
import schedule
import time
import logging
from datetime import datetime, timedelta, timezone
from api.config import ODDS_API_KEY
from fetch_odds import league_mapping
# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from api.config import NETWORK

# Define sport mapping
sport_mapping = {"soccer_epl": 1, "soccer_usa_mls": 1, "americanfootball_nfl": 2, "baseball_mlb": 3, "basketball_nba": 4}

league_odds_type_mapping = {
    'English Premier League': 'soccer_epl',
    'NBA': 'basketball_nba',
    'MLB': 'baseball_mlb',
    'American Major League Soccer': 'soccer_usa_mls',
    'NFL': 'americanfootball_nfl'
}

def parse_datetime_with_optional_timezone(timestamp):
    if isinstance(timestamp, (int, float)) or timestamp.isdigit():
        # Handle Unix timestamp (seconds since epoch)
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    
    try:
        # Try parsing with timezone
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        try:
            # Try parsing without timezone and assume UTC
            dt_naive = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
            return dt_naive.replace(tzinfo=timezone.utc)
        except ValueError:
            # If all else fails, raise an informative error
            raise ValueError(f"Unable to parse timestamp: {timestamp}. "
                             "Expected format: YYYY-MM-DDTHH:MM:SS[±HHMM] or Unix timestamp.")


def create_match_id():
    # generate a unique uuid for the match. make sure it does not already exist.
    match_id = db.generate_uuid()
    while db.match_id_exists(match_id):
        print(f"Match ID {match_id} already exists. Generating a new one.")
        match_id = db.generate_uuid()
    return match_id

def create_match_id_deprecated(home_team, away_team, match_date):
    # Extract the first 10 letters of home and away team names
    home_prefix = home_team.replace(" ", "")[:10]
    away_prefix = away_team.replace(" ", "")[:10]

    formatted_date = parse_datetime_with_optional_timezone(match_date).strftime(
        "%Y%m%d%H%M"
    )

    return f"{home_prefix}{away_prefix}{formatted_date}"


def fetch_and_store_events():
    api_endpoints_url = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=1997764475&single=true&output=csv"
        if NETWORK == "test"
        else "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=0&single=true&output=csv"
    )

    # get event API endpoints from CSV URL and load them into our urls list
    try:
        response = requests.get(api_endpoints_url)
        response.raise_for_status()

        # split the response text into lines
        lines = response.text.split("\n")
        # filter the lines to include only those where column C is "Active"
        activeLeagues = [
            line.split(",")[0].strip()
            for line in lines[1:]
            if line.split(",")[5].strip() == "Active"
        ]
        logging.info(f"Loaded {len(activeLeagues)} API endpoints from {api_endpoints_url}")

    except Exception as e:
        logging.error(f"Error loading API endpoints from URL {api_endpoints_url}: {e}")
        logging.info(f"Using fallback hardcoded API endpoints.")
        urls = [
            #'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4328&s=2023-2024', #English Premier League
            #'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4387&s=2023-2024', #NBA
            #'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4424&s=2024', #MLB
            "https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4346&s=2024"  # American Major League Soccer
        ]

    logging.info("Fetching data from APIs")
    all_matches = []

    current_utc_time = datetime.now(timezone.utc)

    for league in activeLeagues:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{league_odds_type_mapping.get(league)}/scores/"
            params = {
                "apiKey": ODDS_API_KEY,
                "daysFrom": 3,
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                all_matches.extend(data)
            else:
                logging.error("Failed to fetch matches:", response.status_code)
        except Exception as e:
            logging.error(
                f"Failed to fetch or parse API data from {url}", exc_info=True
            )

    if not all_matches:
        logging.info("No matches data found in the API responses")
        return

    current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        for match in all_matches:
            oddspaiMatchId = match.get("id")

            # Query the lookup table
            match_id = db.query_sportsdb_match_lookup(oddspaiMatchId)
            new_match = False
            if match_id is None:
                match_id = create_match_id()
                new_match = True

            is_complete = match.get("completed")
            sport_type = sport_mapping.get(
                match.get("sport_key"), 0
            )  # Default to 0 if sport is unknown
            match.update({"sport_title": league_mapping.get(match.get("sport_title"), match.get("sport_title"))})
            commence_time_str = match.get("commence_time")
            if isinstance(commence_time_str, (int, float)) or commence_time_str.isdigit():
                # Handle Unix timestamp (seconds since epoch)
                matchTimestamp = datetime.fromtimestamp(commence_time_str, tz=timezone.utc)
                matchTimestampStr = matchTimestamp.strftime("%Y-%m-%d %H:%M:%S")
                match.update({"commence_time": matchTimestampStr})
            elif commence_time_str:
                # Parse the ISO 8601 format and convert to the desired format
                commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                match.update({"commence_time": commence_time})

            dbresult = db.insert_match(
                match_id, match, sport_type, is_complete, current_utc_time
            )
            if dbresult and new_match:
                dbresult2 = db.insert_sportsdb_match_lookup(match_id, oddspaiMatchId)
                if dbresult2:
                    logging.info(f"Inserted matchId {match_id} and oddsapiMatchId {oddspaiMatchId} lookup into the database")

    except Exception as e:
        logging.error("Failed inserting events into the MySQL database", exc_info=True)

if __name__ == "__main__":
    # Schedule the function to run every 10 minutes
    schedule.every(10).minutes.do(fetch_and_store_events)

    # Initial fetch and store to ensure setup is correct
    fetch_and_store_events()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)
