import os
from mysql.connector import Error
import datetime as dt
from datetime import timezone
import time
import logging

from api.config import NETWORK, NETUID, IS_PROD
import api.db as db
import bittensor
import typing

def get_uids(
    metagraph: bittensor.metagraph,
    exclude: typing.List[int] = None
) -> typing.List[int]:
    """Returns k available random uids from the metagraph.
    Args:
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (List[int]): Available uids.
    """
    avail_uids = []

    for uid in range(metagraph.n.item()):
        uid_is_not_excluded = exclude is None or uid not in exclude
        
        if uid_is_not_excluded:
            avail_uids.append(uid)
    
    return avail_uids

def take_snapshot():

    # attempt to connect to metagraph 3 times before quitting
    attempts = 3
    for attempt in range(attempts):
        try:
            subtensor = bittensor.subtensor(network=NETWORK)
            metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)
            break  # Exit the loop if successful
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < attempts - 1:
                time.sleep(1)  # Optional: wait a bit before retrying
            else:
                raise RuntimeError("Failed to connect to subtensor and metagraph after 3 attempts")

    uids = get_uids(metagraph)
    hotkey_incentives = {}

    # Get the incentive for each hotkey
    for uid in uids:
        hotkey = metagraph.hotkeys[uid]
        incentive = metagraph.I[uid].item()
        hotkey_incentives[hotkey] = incentive

    try:
        conn = db.get_db_conn()
        c = conn.cursor()

        current_date = dt.datetime.now(timezone.utc)

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"

        # Fetch data from the prediction scores table
        fetch_query = f"""
        SELECT 
            id, 
            miner_hotkey,
            miner_coldkey,
            miner_uid, 
            miner_is_registered,
            miner_age,
            league, 
            sport, 
            total_predictions, 
            winner_predictions, 
            avg_score, 
            last_updated
        FROM {prediction_scores_table_name}
        """
        c.execute(fetch_query)
        results = c.fetchall()

        # Add incentive to the results
        modified_results = []
        for row in results:
            miner_hotkey = row[1]
            incentive = hotkey_incentives.get(miner_hotkey, None)
            modified_row = row + (incentive,)
            modified_results.append((current_date,) + modified_row)
        
        # Insert modified data into the snapshot table
        insert_snapshot_query = """
        INSERT INTO MPRSnapshots (
            snapshot_date, id, miner_hotkey, miner_coldkey, miner_uid, miner_is_registered, miner_age, league, sport, total_predictions, winner_predictions, avg_score, last_updated, incentive
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        c.executemany(insert_snapshot_query, modified_results)
        conn.commit()
        return True

    except Exception as e:
        logging.error("Failed to insert match prediction results snapshot in MySQL database", exc_info=True)
        return False

    finally:
        c.close()
        conn.close()

def main():
    try:
        # Take a snapshot of the current data
        result = take_snapshot()
        if result:
            print("Snapshot taken successfully")
        else:
            print("Failed to take snapshot")

    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()