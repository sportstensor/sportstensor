import api.db as db
import schedule
import time
import logging
from datetime import datetime, timezone

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_match__odds_id():
    # generate a unique uuid for the match. make sure it does not already exist.
    match_odds_id = db.generate_uuid()
    while db.match_odds_id_exists(match_odds_id):
        print(f"Match Odds Table ID {match_odds_id} already exists. Generating a new one.")
        match_odds_id = db.generate_uuid()
    return match_odds_id

def fetch_and_store_match_odds():
    current_utc_time = datetime.now(timezone.utc)
    current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")
    match_list = db.get_matches(all=True)

    match_data = []
    match_ids = [match.get("matchId") for match in match_list]  # Collect all match IDs
    all_odds = db.query_all_match_odds(match_ids)  # Fetch all odds at once
    try:
        for match in match_list:
            match_id = match.get("matchId")

            match_odds_id = db.generate_uuid()
            odds = all_odds.get(match_id)  # Get odds from the dictionary
            if odds:
                match_data.append((match_odds_id, match_id, odds[0], odds[1], odds[2], current_utc_time))
            else:
                match_data.append((match_odds_id, match_id, None, None, None, current_utc_time))
        
        if match_data:
            result = db.insert_match_odds_bulk(match_data)
            if result:
                logging.info(f"Inserted odds data for {len(match_data)} matches into the match odds database")
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
