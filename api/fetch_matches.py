import requests
import api.db as db
import schedule
import time
import logging
from datetime import datetime, timedelta, timezone

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define sport mapping
sport_mapping = {
    'SOCCER': 1,
    'FOOTBALL': 2,
    'BASEBALL': 3,
    'BASKETBALL': 4
}

def parse_datetime_with_optional_timezone(timestamp):
    try:
        # First, try parsing with timezone
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')
    except ValueError:
        # If that fails, parse without the timezone and assume UTC
        dt_naive = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
        return dt_naive.replace(tzinfo=timezone.utc)


def create_match_id(home_team, away_team, match_date):
    # Extract the first 10 letters of home and away team names
    home_prefix = home_team.replace(' ', '')[:10]
    away_prefix = away_team.replace(' ', '')[:10]
    
    formatted_date = parse_datetime_with_optional_timezone(match_date).strftime('%Y%m%d%H%M')
    
    return f"{home_prefix}{away_prefix}{formatted_date}"

def fetch_and_store_events():
    logging.info("Fetching data from APIs")
    urls = [
        'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4328&s=2023-2024',
        'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4387&s=2023-2024',
        'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4424&s=2024',
        'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4346&s=2024'

    ]
    
    all_events = []
    
    current_utc_time = datetime.now(timezone.utc)
    start_period = current_utc_time - timedelta(days=10)
    end_period = current_utc_time + timedelta(hours=48)

    for url in urls:
        try:
            response = requests.get(url)
            data = response.json()
            if 'events' in data:
                filtered_events = [
                    event for event in data['events']
                    if start_period <= parse_datetime_with_optional_timezone(event['strTimestamp']) <= end_period
                ]
                all_events.extend(filtered_events)
        except Exception as e:
            logging.error(f"Failed to fetch or parse API data from {url}", exc_info=True)

    if not all_events:
        logging.info("No events data found in the API responses")
        return

    current_utc_time = current_utc_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        for event in all_events:
            match_id = create_match_id(event.get('strHomeTeam'), event.get('strAwayTeam'), event.get('strTimestamp'))
            status = event.get('strStatus')
            is_complete = 0 if status in ('Not Started', 'NS') else 1 if status in ('Match Finished', 'FT') else 0
            sport_type = sport_mapping.get(event.get('strSport').upper(), 0)  # Default to 0 if sport is unknown
            
            dbresult = db.insert_match(
                match_id,
                event,
                sport_type,
                is_complete,
                current_utc_time
            )
    except Exception as e:
        logging.error("Failed inserting events into the MySQL database", exc_info=True)

# Schedule the function to run every 10 minutes
schedule.every(10).minutes.do(fetch_and_store_events)

# Initial fetch and store to ensure setup is correct
fetch_and_store_events()

# Run the scheduler in an infinite loop
while True:
    schedule.run_pending()
    time.sleep(1)
