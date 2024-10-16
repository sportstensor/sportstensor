import api.db as db
import schedule
import time
import logging
import pytz

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_odds_apis_by_match(stored_odds, match):
    matchId = match.get("matchId")
    if matchId is None:
        return None
    homeTeamName = match.get("homeTeamName")
    awayTeamName = match.get("awayTeamName")
    matchDate = match.get("matchDate")
    matchLeague = match.get("matchLeague")
    str_matchDate = pytz.utc.localize(matchDate).strftime("%Y-%m-%dT%HZ")
    # Extract just the date part from match_date
    odds_apis = []
    for odds in stored_odds:
        # Extract date from commence_time
        commence_time = odds["commence_time"]
        str_commence_time = pytz.utc.localize(commence_time).strftime("%Y-%m-%dT%HZ")
        if (odds["homeTeamName"] == homeTeamName and
            odds["awayTeamName"] == awayTeamName and
            str_matchDate == str_commence_time and
            odds["league"] == matchLeague):
            odds_apis.append(odds['oddsapiMatchId'])
    if odds_apis:
        return matchId, odds_apis[0]
    else:
        return None

def fetch_and_store_match_lookups():
    logging.info(f"=============Starting to update matches lookup with new odds apis=============")
    matchesWithNoOdds = db.get_matches_with_no_odds()
    stored_odds = db.get_stored_odds()
    matches_odds_apis = []
    try:
        for match in matchesWithNoOdds:
            matches_odds_api = get_odds_apis_by_match(stored_odds, match)
            if matches_odds_api:
                matches_odds_apis.append(matches_odds_api)
        result = db.insert_match_lookups_bulk(matches_odds_apis)
        if result:
            logging.info(f"Inserted odds data for {len(matches_odds_apis)} matches into the match odds database")
            for matches_odds in matches_odds_apis:
                logging.info(f"Inserted {matches_odds[0]} match and {matches_odds[1]} odds into the match odds lookup database")
    except Exception as e:
        logging.error("Failed inserting oddsapiMatchId data into the matches_lookup database", exc_info=True)

if __name__ == "__main__":
    # Schedule the function to run every 15 minutes
    schedule.every(15).minutes.do(fetch_and_store_match_lookups)

    # Initial fetch and store to ensure setup is correct
    fetch_and_store_match_lookups()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)