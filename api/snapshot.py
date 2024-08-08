import mysql.connector
from mysql.connector import Error
import datetime as dt
import logging

from api.config import IS_PROD
import api.db as db

def take_snapshot():
    try:
        conn = db.get_db_conn()
        c = conn.cursor()

        current_date = dt.datetime.now().date()

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
        insert_snapshot_query = f"""
        INSERT INTO MPRSnapshots (
            snapshot_date, id, miner_hotkey, miner_uid, miner_is_registered, league, sport, total_predictions, winner_predictions, avg_score, last_updated
        )
        SELECT %s, id, miner_hotkey, miner_uid, miner_is_registered, league, sport, total_predictions, winner_predictions, avg_score, last_updated
        FROM {prediction_scores_table_name}
        """
        c.execute(insert_snapshot_query, (current_date,))
        conn.commit()

    except Exception as e:
        logging.error("Failed to insert match prediction results snapshot in MySQL database", exc_info=True)

    finally:
        c.close()
        conn.close()

def main():
    try:
        # Take a snapshot of the current data
        take_snapshot()

        print("Snapshot taken successfully")

    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()