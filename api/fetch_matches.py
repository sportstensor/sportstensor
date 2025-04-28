import requests
import api.db as db
import schedule
import time
import logging
from datetime import datetime, timedelta, timezone

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from api.config import NETWORK, ODDS_API_KEY

# Define sport mapping
sport_mapping = {"SOCCER": 1, "AMERICAN FOOTBALL": 2, "BASEBALL": 3, "BASKETBALL": 4}
ODDSAPI_SPORTS_TYPES = [
    {
        'sport_key': 'baseball_mlb',
        'sport_id': 3,
        'league': 'MLB',
    },
    {
        'sport_key': 'americanfootball_nfl',
        'sport_id': 2,
        'league': 'NFL',
    },
    {
        'sport_key': 'soccer_usa_mls',
        'sport_id': 1,
        'league': 'American Major League Soccer',
    },
    {
        'sport_key': 'soccer_epl',
        'sport_id': 1,
        'league': 'English Premier League',
    },
    {
        'sport_key': 'basketball_nba',
        'sport_id': 4,
        'league': 'NBA',
    },
]


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
        urls = [
            line.split(",")[4].strip()
            for line in lines[1:]
            if line.split(",")[5].strip() == "Active"
        ]
        logging.info(f"Loaded {len(urls)} API endpoints from {api_endpoints_url}")

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
    all_events = []

    current_utc_time = datetime.now(timezone.utc)
    start_period = current_utc_time - timedelta(days=10)
    end_period = current_utc_time + timedelta(hours=48)

    for url in urls:
        try:
            response = requests.get(url)
            data = response.json()
            if "events" in data:
                filtered_events = [
                    event
                    for event in data["events"]
                    if start_period
                    <= parse_datetime_with_optional_timezone(event["strTimestamp"])
                    <= end_period
                ]
                all_events.extend(filtered_events)
        except Exception as e:
            logging.error(
                f"Failed to fetch or parse API data from {url}", exc_info=True
            )

    if not all_events:
        logging.info("No events data found in the API responses")
        return

    current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        for event in all_events:
            event_id = event.get("idEvent")

            # Query the lookup table
            match_id = db.query_sportsdb_match_lookup(event_id)
            new_match = False
            if match_id is None:
                match_id = create_match_id()
                new_match = True

            status = event.get("strStatus")
            # Initial assessment of completion status from this data source
            initial_is_complete = (
                0
                if status in ("Not Started", "NS")
                else 1 if status in ("Match Finished", "FT", "After Over Time", "AOT") else 0
            )
            
            # Only do the cross-check with oddsapi if this isn't a new match
            final_is_complete = initial_is_complete
            # If oddsapi says match is not complete, don't mark it as such
            if not new_match:
                # Get the match data from oddsapi
                oddsapi_match = db.get_match_oddsapi_by_id(match_id)
                
                if oddsapi_match:
                    # Check if both sources agree the match is complete
                    if oddsapi_match.get("isComplete") == 1:
                        final_is_complete = 1
                    elif oddsapi_match.get("isComplete") == 0:
                        final_is_complete = 0
            
            sport_type = sport_mapping.get(
                event.get("strSport").upper(), 0
            )  # Default to 0 if sport is unknown
            
            if isinstance(event.get("strTimestamp"), (int, float)) or event.get("strTimestamp").isdigit():
                # Handle Unix timestamp (seconds since epoch)
                matchTimestamp = datetime.fromtimestamp(float(event.get("strTimestamp")), tz=timezone.utc)
                matchTimestampStr = matchTimestamp.strftime("%Y-%m-%d %H:%M:%S")
                event.update({"strTimestamp": matchTimestampStr})

            # Use the final_is_complete value that considers both data sources
            dbresult = db.insert_match(
                match_id, event, sport_type, final_is_complete, current_utc_time
            )
            if dbresult and new_match:
                dbresult2 = db.insert_sportsdb_match_lookup(match_id, event_id)
                if dbresult2:
                    logging.info(f"Inserted matchId {match_id} and sportsdbMatchId {event_id} lookup into the database")

    except Exception as e:
        logging.error("Failed inserting events into the MySQL database", exc_info=True)

def fetch_and_store_events_oddsapi():
    api_endpoints_url = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=1997764475&single=true&output=csv"
        if NETWORK == "test"
        else "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=0&single=true&output=csv"
    )

    # get leagues that are active
    active_leagues = []
    try:
        response = requests.get(api_endpoints_url)
        response.raise_for_status()

        # split the response text into lines
        lines = response.text.split("\n")
        # filter the lines to include only those where column C is "Active"
        active_leagues = [
            line.split(",")[0].strip()
            for line in lines[1:]
            if line.split(",")[5].strip() == "Active"
        ]
        logging.info(f"Loaded {len(active_leagues)} active leagues from {api_endpoints_url}")

    except Exception as e:
        logging.error(f"Error loading active leagues from URL {api_endpoints_url}: {e}")
        return

    logging.info("Fetching data from The Odds Api")
    all_events = []

    current_utc_time = datetime.now(timezone.utc)
    #start_period = current_utc_time - timedelta(days=10)
    #end_period = current_utc_time + timedelta(hours=48)

    filtered_sports_types = [
        sports_type for sports_type in ODDSAPI_SPORTS_TYPES if sports_type['league'] in active_leagues
    ]

    for sports_type in filtered_sports_types:
        api_url = f"https://api.the-odds-api.com/v4/sports/{sports_type['sport_key']}/scores/"
        params = {
            "apiKey": ODDS_API_KEY,
            "daysFrom": 3,
        }
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_events.extend(data)
        else:
            logging.error(f"Failed to fetch events for sport {sports_type['sport_key']}:", response.status_code)
            return None

    if not all_events:
        logging.info("No events data found in the Odds Api responses")
        return

    current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        for event in all_events:
            event_id = event.get("id")

            """ No need for lookup table right now.
            # Query the lookup table
            match_id = db.query_sportsdb_match_lookup(event_id)
            new_match = False
            if match_id is None:
                match_id = create_match_id()
                new_match = True
            """

            status = event.get("completed")
            is_complete = 0
            if status:
                is_complete = 1
            
            sport_key = event.get("sport_key", "").lower()
            
            sport_type = 0
            for sport in filtered_sports_types:
                if sport['sport_key'] == sport_key:
                    sport_type = sport['sport_id']
                    break
            
            # convert commence_time from YYYY-MM-DDTHH:MM:SSZ to YYYY-MM-DD HH:MM:SS in UTC
            if isinstance(event.get("commence_time"), str):
                try:
                    matchTimestamp = parse_datetime_with_optional_timezone(event.get("commence_time"))
                    matchTimestampStr = matchTimestamp.strftime("%Y-%m-%d %H:%M:%S")
                    event.update({"commence_time": matchTimestampStr})
                except ValueError as ve:
                    logging.error(f"Failed to parse commence_time for event {event_id}: {ve}")
                    continue

            home_team_score, away_team_score = None, None
            if event.get("scores") is not None and len(event.get("scores")) > 0:
                for score in event.get("scores"):
                    if score.get("name") is not None and score.get("name") == event.get("home_team"):
                        home_team_score = score.get("score")
                    if score.get("name") is not None and score.get("name") == event.get("away_team"):
                        away_team_score = score.get("score")

            dbresult = db.insert_match_oddsapi(
                event_id, event, sport_type, home_team_score, away_team_score, is_complete, current_utc_time
            )
            """ No need for lookup table right now.
            if dbresult and new_match:
                dbresult2 = db.insert_sportsdb_match_lookup(match_id, event_id)
                if dbresult2:
                    logging.info(f"Inserted matchId {match_id} and sportsdbMatchId {event_id} lookup into the database")
            """

    except Exception as e:
        logging.error("Failed inserting Odds Api events into the MySQL database", exc_info=True)

if __name__ == "__main__":
    # Schedule the function to run every 10 minutes
    schedule.every(10).minutes.do(fetch_and_store_events)
    schedule.every(10).minutes.do(fetch_and_store_events_oddsapi)

    # Initial fetch and store to ensure setup is correct
    fetch_and_store_events()
    # Initial fetch and store for Odds API
    fetch_and_store_events_oddsapi()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)
