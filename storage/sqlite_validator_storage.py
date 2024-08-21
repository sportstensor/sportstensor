import os
import contextlib
import datetime as dt
import time
import bittensor as bt
import sqlite3
import threading
import random
from pydantic import ValidationError
from typing import Any, Dict, Optional, Set, Tuple, List
from common.data import (
    Sport,
    Match,
    Prediction,
    MatchPrediction,
    League,
    MatchPredictionWithMatchData,
)
from common.protocol import GetMatchPrediction
from common.constants import (
    IS_DEV,
    MIN_PREDICTION_TIME_THRESHOLD,
    MAX_PREDICTION_DAYS_THRESHOLD,
    SCORING_CUTOFF_IN_DAYS,
)
from storage.validator_storage import ValidatorStorage


class SqliteValidatorStorage(ValidatorStorage):
    """Sqlite in-memory backed Validator Storage"""

    LEAGUES_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS Leagues (
                            leagueId      VARCHAR(50)     PRIMARY KEY,
                            leagueName    VARCHAR(50)     NOT NULL,
                            sport         INTEGER         NOT NULL,
                            isActive      INTEGER         DEFAULT 0,
                            lastUpdated   TIMESTAMP(6)    NOT NULL
                            )"""

    MATCHES_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS Matches (
                            matchId       VARCHAR(50)     PRIMARY KEY,
                            matchDate     TIMESTAMP(6)    NOT NULL,
                            sport         INTEGER         NOT NULL,
                            league        VARCHAR(50)     NOT NULL,
                            homeTeamName  VARCHAR(30)     NOT NULL,
                            awayTeamName  VARCHAR(30)     NOT NULL,
                            homeTeamScore INTEGER         NULL,
                            awayTeamScore INTEGER         NULL,
                            isComplete    INTEGER         DEFAULT 0,
                            lastUpdated   TIMESTAMP(6)    NOT NULL
                            )"""

    MATCHPREDICTIONS_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS MatchPredictions (
                            predictionId  INTEGER         PRIMARY KEY,
                            minerId       INTEGER         NOT NULL,
                            hotkey        VARCHAR(64)     NOT NULL,
                            matchId       VARCHAR(50)     NOT NULL,
                            matchDate     TIMESTAMP(6)    NOT NULL,
                            sport         INTEGER         NOT NULL,
                            league        VARCHAR(50)     NOT NULL,
                            homeTeamName  VARCHAR(30)     NOT NULL,
                            awayTeamName  VARCHAR(30)     NOT NULL,
                            homeTeamScore INTEGER         NULL,
                            awayTeamScore INTEGER         NULL,
                            isScored      INTEGER         DEFAULT 0,
                            scoredDate    TIMESTAMP(6)    NULL,
                            lastUpdated   TIMESTAMP(6)    NOT NULL
                            )"""

    def __init__(self):
        sqlite3.register_converter("timestamp", tz_aware_timestamp_adapter)

        self.continuous_connection_do_not_reuse = self._create_connection()

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()

            # Create the Matches table (if it does not already exist).
            # cursor.execute(SqliteValidatorStorage.LEAGUES_TABLE_CREATE)

            # Create the Matches table (if it does not already exist).
            cursor.execute(SqliteValidatorStorage.MATCHES_TABLE_CREATE)

            # Create the MatchPredictions table (if it does not already exist).
            cursor.execute(SqliteValidatorStorage.MATCHPREDICTIONS_TABLE_CREATE)

            # Lock to avoid concurrency issues on interacting with the database.
            self.lock = threading.RLock()

        # Execute cleanup queries
        self.cleanup()

    def _create_connection(self):
        # Create the database if it doesn't exist, defaulting to the local directory.
        # Use PARSE_DECLTYPES to convert accessed values into the appropriate type.
        connection = sqlite3.connect(
            "SportsTensor.db",
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=120.0,
        )
        # Avoid using a row_factory that would allow parsing results by column name for performance.
        # connection.row_factory = sqlite3.Row
        connection.isolation_level = None
        return connection
    
    def cleanup(self):
        """Cleanup the database."""
        print("========================== Database status checks and cleanup ==========================")
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                
                db_size_bytes = os.path.getsize("SportsTensor.db")
                db_size_gb = db_size_bytes / (1024 ** 3)
                db_size_mb = db_size_bytes / (1024 ** 2)
                print(f"SportsTensor.db size: {db_size_gb:.2f} GB ({db_size_mb:.2f} MB)")
                
                # Print the total number of rows in the MatchPredictions table
                cursor.execute("SELECT COUNT(*) FROM MatchPredictions")
                total_rows = cursor.fetchone()[0]
                print(f"Total number of rows in MatchPredictions: {total_rows}")

                # Print the total number of rows in the MatchPredictions table
                cursor.execute(f"SELECT COUNT(*) FROM MatchPredictions WHERE isScored = 0 AND lastUpdated < DATETIME('now', '-{SCORING_CUTOFF_IN_DAYS} day')")
                total_unscored_rows = cursor.fetchone()[0]
                print(f"Total number of MatchPredictions {SCORING_CUTOFF_IN_DAYS}+ days old that haven't been scored: {total_unscored_rows}")
                
                try:
                    # Execute cleanup queries
                    if total_unscored_rows > 0:
                        # Clean up old predictions that haven't been scored (for whatever reason) and never will be
                        print(f"Deleting abandoned predictions older than {SCORING_CUTOFF_IN_DAYS} days...")
                        cursor.execute(
                            f"DELETE FROM MatchPredictions WHERE isScored = 0 AND lastUpdated < DATETIME('now', '-{SCORING_CUTOFF_IN_DAYS} day')"
                        )
                        connection.commit()

                    # Clean up bug where we accidentally inserted 'REDACTED' as scores
                    cursor.execute(
                        """DELETE FROM MatchPredictions WHERE homeTeamScore='REDACTED' OR awayTeamScore='REDACTED'"""
                    )
                    connection.commit()

                    # Run VACUUM to reclaim unused space
                    #print("Running VACUUM to reclaim unused space...")
                    #cursor.execute("VACUUM")
                    
                    # Check database integrity
                    print("Checking database integrity...")
                    cursor.execute("PRAGMA integrity_check")
                    integrity_result = cursor.fetchone()[0]
                    if integrity_result != "ok":
                        print("*** ERROR: Database integrity check failed! Contact Sportstensor admin. ***")
                    else:
                        print("Database integrity check passed.")
                
                except Exception as e:
                    print(f"An error occurred during cleanup: {e}")
                    raise e
        print("========================================================================================")

    def insert_leagues(self, leagues: List[League]):
        """Stores leagues associated with sports. Indicates which leagues are active to run predictions on."""
        values = []
        for league in leagues:
            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            # Parse every League into a list of values to insert.
            values.append(
                [
                    league.leagueId,
                    league.leagueName,
                    league.sport,
                    league.isActive,
                    now_str,
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """INSERT OR IGNORE INTO Leagues (leagueId, leagueName, sport, isActive, lastUpdated) VALUES (?, ?, ?, ?, ?)""",
                    values,
                )
                connection.commit()

    def update_leagues(self, leagues: List[League]):
        """Updates leagues. Mainly for activating or deactivating"""
        values = []
        for league in leagues:
            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # Parse Leagues into a list of values to update.
            values.append(
                [league.leagueName, league.isActive, now_str, league.leagueId]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE Leagues SET leagueName = ?, isActive = ?, lastUpdated = ? WHERE leagueId = ?""",
                    values,
                )
                connection.commit()

    def insert_matches(self, matches: List[Match]):
        """Stores official matches to score predictions from miners on."""
        values = []
        for match in matches:
            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # Parse every Match into a list of values to insert.
            values.append(
                [
                    match.matchId,
                    match.matchDate,
                    match.sport,
                    match.league,
                    match.homeTeamName,
                    match.awayTeamName,
                    match.homeTeamScore,
                    match.awayTeamScore,
                    match.isComplete,
                    now_str,
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """INSERT OR IGNORE INTO Matches (matchId, matchDate, sport, league, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, isComplete, lastUpdated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
                connection.commit()

    def update_matches(self, matches: List[Match]):
        """Updates matches. Typically only used when updating final score."""
        values = []
        for match in matches:
            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # Parse Matches into a list of values to update.
            values.append(
                [
                    match.homeTeamScore,
                    match.awayTeamScore,
                    match.isComplete,
                    now_str,
                    match.matchId,
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE Matches SET homeTeamScore = ?, awayTeamScore = ?, isComplete = ?, lastUpdated = ? WHERE matchId = ?""",
                    values,
                )
                connection.commit()

    def check_match(self, matchId: str) -> Match:
        """Check if a match with the given ID exists in the database."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """SELECT EXISTS(SELECT 1 FROM matches WHERE matchId = ?)""",
                    (matchId,),
                )
                return cursor.fetchone()[0]

    def get_matches_to_predict(self, batchsize: int = 10) -> List[Match]:
        """Gets batchsize number of matches ready to be predicted."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:

                # Calculate the current timestamp
                current_timestamp = int(time.time())
                # Calculate the lower bound timestamp (earliest match date allowed for predictions)
                lower_bound_timestamp = (
                    current_timestamp + MIN_PREDICTION_TIME_THRESHOLD
                )
                # Calculate the upper bound timestamp (latest match date allowed for predictions)
                upper_bound_timestamp = (
                    current_timestamp + MAX_PREDICTION_DAYS_THRESHOLD * 24 * 3600
                )
                # Convert timestamps to strings in 'YYYY-MM-DD HH:MM:SS' format
                lower_bound_str = dt.datetime.utcfromtimestamp(
                    lower_bound_timestamp
                ).strftime("%Y-%m-%d %H:%M:%S")
                upper_bound_str = dt.datetime.utcfromtimestamp(
                    upper_bound_timestamp
                ).strftime("%Y-%m-%d %H:%M:%S")

                cursor = connection.cursor()
                cursor.execute(
                    """
                    SELECT * 
                    FROM Matches
                    WHERE isComplete = 0
                    AND matchDate BETWEEN ? AND ?
                    ORDER BY RANDOM() LIMIT ?
                    """,
                    (lower_bound_str, upper_bound_str, batchsize),
                )
                # print("matchDate > ", lower_bound_str, "matchDate < ", upper_bound_str)
                results = cursor.fetchall()
                if not results:
                    return []

                # Convert the raw database results into Pydantic models
                matches = [
                    Match(
                        **dict(zip([column[0] for column in cursor.description], row))
                    )
                    for row in results
                ]
                return matches

    def insert_match_predictions(self, predictions: List[GetMatchPrediction]):
        """Stores unscored predictions returned from miners."""
        values = []
        for prediction in predictions:
            """
            bt.logging.debug(f" \
                    [{prediction.match_prediction.minerId}] {prediction.match_prediction.hotkey}: Upserting prediction for match {str(prediction.match_prediction.matchId)}, \
                    {prediction.match_prediction.awayTeamName} at {prediction.match_prediction.homeTeamName} \
                    on {str(prediction.match_prediction.matchDate)} \
            ")
            """

            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            if IS_DEV:
                random_uid = random.randint(
                    1, 16
                )  # Generate a random integer between 1 and 16

            # Parse every MatchPrediction into a list of values to insert.
            values.append(
                [
                    prediction.match_prediction.minerId if not IS_DEV else random_uid,
                    (
                        prediction.match_prediction.hotkey
                        if not IS_DEV
                        else f"DEV_{str(random_uid)}"
                    ),
                    prediction.match_prediction.matchId,
                    prediction.match_prediction.matchDate,
                    prediction.match_prediction.sport,
                    prediction.match_prediction.league,
                    prediction.match_prediction.homeTeamName,
                    prediction.match_prediction.awayTeamName,
                    prediction.match_prediction.homeTeamScore,
                    prediction.match_prediction.awayTeamScore,
                    now_str,
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """INSERT OR IGNORE INTO MatchPredictions (minerId, hotkey, matchId, matchDate, sport, league, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, lastUpdated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
                connection.commit()

    def get_match_predictions_to_score(
        self, batchsize: int = 10, matchDateCutoff: int = SCORING_CUTOFF_IN_DAYS
    ) -> Optional[List[MatchPredictionWithMatchData]]:
        """Gets batchsize number of predictions that need to be scored and are eligible to be scored (the match is complete)"""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()

                # Calculate the current timestamp
                current_timestamp = int(time.time())
                # Calculate cutoff date timestamp
                match_cutoff_timestamp = current_timestamp - (
                    matchDateCutoff * 24 * 3600
                )
                # Convert timestamps to strings in 'YYYY-MM-DD HH:MM:SS' format
                match_cutoff_str = dt.datetime.utcfromtimestamp(
                    match_cutoff_timestamp
                ).strftime("%Y-%m-%d %H:%M:%S")

                cursor.execute(
                    """
                    SELECT mp.*, m.homeTeamScore as actualHomeTeamScore, m.awayTeamScore as actualAwayTeamScore
                    FROM MatchPredictions mp
                    JOIN Matches m ON (m.matchId = mp.matchId)
                    WHERE mp.isScored = 0
                    AND m.isComplete = 1
                    AND mp.matchDate > ?
                    AND mp.homeTeamScore IS NOT NULL
                    AND mp.awayTeamScore IS NOT NULL
                    AND m.homeTeamScore IS NOT NULL
                    AND m.awayTeamScore IS NOT NULL
                    ORDER BY mp.matchDate ASC
                    LIMIT ?
                    """,
                    [match_cutoff_str, batchsize],
                )
                results = cursor.fetchall()
                if not results:
                    return []

                # Convert the raw database results into the new combined Pydantic model
                combined_predictions = []
                for row in results:
                    prediction_data = {
                        "predictionId": row[0],
                        "minerId": row[1],
                        "hotkey": row[2],
                        "matchId": row[3],
                        "matchDate": row[4],
                        "sport": row[5],
                        "league": row[6],
                        "homeTeamName": row[7],
                        "awayTeamName": row[8],
                        "homeTeamScore": row[9],
                        "awayTeamScore": row[10],
                        "isScored": row[11],
                        # row[12] is scoredDate
                        # row[13] is lastUpdated
                    }
                    try:
                        combined_predictions.append(
                            MatchPredictionWithMatchData(
                                prediction=MatchPrediction(**prediction_data),
                                actualHomeTeamScore=row[14],
                                actualAwayTeamScore=row[15],
                            )
                        )
                    except ValidationError as e:
                        bt.logging.error(f"Validation error for row {row}: {e}")

                return combined_predictions

    def update_match_predictions(self, predictions: List[MatchPrediction]):
        """Updates predictions. Typically only used when marking predictions as being scored."""
        values = []
        for prediction in predictions:
            bt.logging.trace(
                f"{prediction.hotkey}: Marking prediction {str(prediction.predictionId)} as scored"
            )

            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # Parse MatchPredictions into a list of values to update, marking each as scored with a timestamp of now.
            values.append([1, now_str, prediction.predictionId])

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE MatchPredictions SET isScored = ?, scoredDate = ? WHERE predictionId = ?""",
                    values,
                )
                connection.commit()

    def get_miner_match_predictions(
        self, miner_hotkey: str, scored=False
    ) -> Optional[List[MatchPrediction]]:
        """Gets a list of all predictions made by a miner."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT * FROM MatchPredictions WHERE hotkey = ?",
                    [miner_hotkey],
                )
                results = cursor.fetchall()
                if not results:
                    return []

                # Convert the raw database results into Pydantic models
                predictions = [
                    MatchPrediction(
                        **dict(zip([column[0] for column in cursor.description], row))
                    )
                    for row in results
                ]
                return predictions

    def read_miner_last_prediction(self, miner_hotkey: str) -> Optional[dt.datetime]:
        """Gets when a specific miner last returned a prediction."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT MAX(lastUpdated) FROM MatchPrediction WHERE hotkey = ?",
                    [miner_hotkey],
                )
                result = cursor.fetchone()
                if result is not None:
                    return result[0]
                else:
                    return None

    def delete_miner(self, hotkey: str):
        """Removes the predictions and miner information for the specified miner."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "DELETE FROM MatchPredictions WHERE hotkey = ?", [hotkey]
                )


# Use a timezone aware adapter for timestamp columns.
def tz_aware_timestamp_adapter(val):
    datepart, timepart = val.split(b" ")
    year, month, day = map(int, datepart.split(b"-"))

    if b"+" in timepart:
        timepart, tz_offset = timepart.rsplit(b"+", 1)
        if tz_offset == b"00:00":
            tzinfo = dt.timezone.utc
        else:
            hours, minutes = map(int, tz_offset.split(b":", 1))
            tzinfo = dt.timezone(dt.timedelta(hours=hours, minutes=minutes))
    elif b"-" in timepart:
        timepart, tz_offset = timepart.rsplit(b"-", 1)
        if tz_offset == b"00:00":
            tzinfo = dt.timezone.utc
        else:
            hours, minutes = map(int, tz_offset.split(b":", 1))
            tzinfo = dt.timezone(dt.timedelta(hours=-hours, minutes=-minutes))
    else:
        tzinfo = None

    timepart_full = timepart.split(b".")
    hours, minutes, seconds = map(int, timepart_full[0].split(b":"))

    if len(timepart_full) == 2:
        microseconds = int("{:0<6.6}".format(timepart_full[1].decode()))
    else:
        microseconds = 0

    val = dt.datetime(year, month, day, hours, minutes, seconds, microseconds, tzinfo)

    return val
