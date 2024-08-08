import mysql.connector
from mysql.connector import Error
import datetime as dt
import logging

from api.config import IS_PROD
import api.db as db

def create_snapshot_table():
    try:
        conn = db.get_db_conn()
        c = conn.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS MPRSnapshots (
            snapshot_id INT AUTO_INCREMENT PRIMARY KEY,
            snapshot_date DATE NOT NULL,
            id INT,
            miner_hotkey VARCHAR(64),
            miner_uid INTEGER,
            miner_is_registered TINYINT(1),
            league VARCHAR(50),
            sport INTEGER,
            total_predictions INTEGER,
            winner_predictions INTEGER,
            avg_score FLOAT,
            last_updated TIMESTAMP,
            UNIQUE (snapshot_date, id)
        )
        """
        c.execute(create_table_query)
        conn.commit()

    except Exception as e:
        logging.error("Failed to create match prediction results snapshot table in MySQL database", exc_info=True)

    finally:
        c.close()
        conn.close()

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
        # Create snapshot table if it doesn't exist
        create_snapshot_table()

        # Take a snapshot of the current data
        take_snapshot()

        print("Snapshot taken successfully")

    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()