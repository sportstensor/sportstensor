import contextlib
import datetime as dt
import time
import bittensor as bt
import sqlite3
import threading
from typing import Any, Dict, Optional, Set, Tuple, List
from common.data import Sport, Match, Prediction, MatchPrediction
from common.constants import MIN_PREDICTION_TIME_THRESHOLD, MAX_PREDICTION_DAYS_THRESHOLD
from storage.validator.validator_storage import ValidatorStorage


class SqliteValidatorStorage(ValidatorStorage):
    """Sqlite in-memory backed Validator Storage"""

    LEAGUES_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS Leagues (
                            leagueId      VARCHAR(50)     PRIMARY KEY,
                            leagueName    VARCHAR(50)     NOT NULL,
                            sport         INTEGER         NOT NULL,
                            isActive      INTEGER         DEFAULT 0,
                            lastUpdated   TIMESTAMP(6)    NOT NULL,
                            )"""
    
    MATCHES_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS Matches (
                            matchId       VARCHAR(50)     PRIMARY KEY,
                            matchDate     TIMESTAMP(6)    NOT NULL,
                            sport         INTEGER         NOT NULL,
                            homeTeamName  VARCHAR(30)     NOT NULL,
                            awayTeamName  VARCHAR(30)     NOT NULL,
                            homeTeamScore INTEGER         NULL,
                            awayTeamScore INTEGER         NULL,
                            isComplete    INTEGER         DEFAULT 0,
                            lastUpdated   TIMESTAMP(6)    NOT NULL,
                            )"""

    MATCHPREDICTIONS_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS MatchPredictions (
                            predictionId  INTEGER         PRIMARY KEY,
                            minerId       INTEGER         NOT NULL,
                            hotkey        VARCHAR(64)     NOT NULL,
                            matchId       VARCHAR(50)     NOT NULL,
                            matchDate     TIMESTAMP(6)    NOT NULL,
                            sport         INTEGER         NOT NULL,
                            homeTeamName  VARCHAR(30)     NOT NULL,
                            awayTeamName  VARCHAR(30)     NOT NULL,
                            homeTeamScore INTEGER         NULL,
                            awayTeamScore INTEGER         NULL,
                            isScored      INTEGER         DEFAULT 0,
                            scoredDate    TIMESTAMP(6)    NULL,
                            lastUpdated   TIMESTAMP(6)    NOT NULL,
                            )"""

    def __init__(self):
        sqlite3.register_converter("timestamp", tz_aware_timestamp_adapter)

        self.continuous_connection_do_not_reuse = self._create_connection()

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()

            # Create the Matches table (if it does not already exist).
            cursor.execute(SqliteValidatorStorage.LEAGUES_TABLE_CREATE)

            # Create the Matches table (if it does not already exist).
            cursor.execute(SqliteValidatorStorage.MATCHES_TABLE_CREATE)

            # Create the MatchPredictions table (if it does not already exist).
            cursor.execute(SqliteValidatorStorage.MATCHPREDICTIONS_TABLE_CREATE)
            
            # Lock to avoid concurrency issues on interacting with the database.
            self.lock = threading.RLock()

    def _create_connection(self):
        # Create the database if it doesn't exist, defaulting to the local directory.
        # Use PARSE_DECLTYPES to convert accessed values into the appropriate type.
        connection = sqlite3.connect(
            "file::memory:?cache=shared",
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=120.0,
        )
        # Avoid using a row_factory that would allow parsing results by column name for performance.
        # connection.row_factory = sqlite3.Row
        connection.isolation_level = None
        return connection
    
    def insert_leagues(self, leagues: List[League]):
        """Stores leagues associated with sports. Indicates which leagues are active to run predictions on."""
        for league in leagues:
          now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
          # Parse every League into a list of values to insert.
          values = []
          values.append(
            [
                league.leaguehId,
                league.leagueName,
                league.sport,
                league.isActive,                
                now_str
            ]
          )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                # Batch in groups of 1m if necessary to avoid congestion issues.
                value_subsets = [
                    values[x : x + 1_000_000] for x in range(0, len(values), 1_000_000)
                ]
                for value_subset in value_subsets:
                    cursor.executemany(
                        """INSERT OR IGNORE INTO Leagues (leagueId, leagueName, sport, isActive, lastUpdated) VALUES (?, ?, ?, ?, ?)""",
                        value_subset,
                    )
                connection.commit()
    
    def update_leagues(self, leagues: List[League]):
        """Updates leagues. Mainly for activating or deactivating"""
        for league in leagues:
          now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")

          # Parse Leagues into a list of values to update.
          values = []
          values.append(
            [
                league.leagueName,
                league.isActive,                
                now_str
                league.leagueId
            ]
          )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE Leagues SET leagueName = ?, isActive = ?, lastUpdated = ? WHERE matchId = ?""",
                    values,
                )
                connection.commit()
    
    def insert_matches(self, matches: List[Match]):
        """Stores official matches to score predictions from miners on."""
        for match in matches:
          now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")

          # Parse every Match into a list of values to insert.
          values = []
          values.append(
            [
                match.matchId,
                match.matchDatetime,
                match.sport,
                match.homeTeamName,
                match.awayTeamName,
                now_str
            ]
          )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                # Batch in groups of 1m if necessary to avoid congestion issues.
                value_subsets = [
                    values[x : x + 1_000_000] for x in range(0, len(values), 1_000_000)
                ]
                for value_subset in value_subsets:
                    cursor.executemany(
                        """INSERT OR IGNORE INTO Matches (matchId, matchDate, sport, homeTeamName, awayTeamName, lastUpdated) VALUES (?, ?, ?, ?, ?, ?)""",
                        value_subset,
                    )
                connection.commit()

    def update_matches(self, matches: List[Match]):
        """Updates matches. Typically only used when updating final score."""
        for match in matches:
          now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")

          # Parse Matches into a list of values to update.
          values = []
          values.append(
            [
                match.homeTeamScore,
                match.awayTeamScore,
                match.isComplete,
                now_str,
                match.matchId
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

    def get_matches_to_predict(self, batchsize: int = 10) -> List[Match]:
        """Gets batchsize number of matches ready to be predicted."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                
                # Calculate the current timestamp
                current_timestamp = int(time.time())
                # Calculate the lower bound timestamp (earliest match date allowed for predictions)
                lower_bound_timestamp = current_timestamp + MIN_PREDICTION_TIME_THRESHOLD
                # Calculate the upper bound timestamp (latest match date allowed for predictions)
                upper_bound_timestamp = current_timestamp + MAX_PREDICTION_DAYS_THRESHOLD * 24 * 3600

                cursor = connection.cursor()
                cursor.execute(
                    """
                    SELECT * 
                    FROM Matches
                    WHERE isComplete = 0 
                    AND matchDate > ? 
                    AND matchDate < ? 
                    ORDER BY RANDOM() LIMIT ?
                    """,
                    (lower_bound_timestamp, upper_bound_timestamp, batchsize),
                )
                results = cursor.fetchall()
                if results is None:
                    return None
                
                return results
    
    def insert_match_predictions(self, predictions: List[MatchPrediction]):
        """Stores unscored predictions returned from miners."""
        for prediction in predictions:
          bt.logging.trace(
              f"{prediction.axon.hotkey}: Upserting prediction for match {str(prediction.matchId)}, {prediction.awayTeamName} at {prediction.homeTeamName} on {str(prediction.matchDatetime)}"
          )

          now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")

          # Parse every MatchPrediction into a list of values to insert.
          values = []
          values.append(
            [
                prediction.axon.uid,
                prediction.axon.hotkey,
                prediction.matchId,
                prediction.matchDatetime,
                prediction.sport,
                prediction.homeTeamName,
                prediction.awayTeamName,
                prediction.homeTeamScore,
                prediction.awayTeamScore,
                now_str
            ]
          )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                # Batch in groups of 1m if necessary to avoid congestion issues.
                value_subsets = [
                    values[x : x + 1_000_000] for x in range(0, len(values), 1_000_000)
                ]
                for value_subset in value_subsets:
                    cursor.executemany(
                        """INSERT OR IGNORE INTO MatchPredictions (minerId, hotkey, matchId, matchDate, sport, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, lastUpdated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        value_subset,
                    )
                connection.commit()

    def get_match_predictions_to_score(self, batchsize: int = 10, matchdate_before: dt.datetime = None, end_datetime: dt.datetime = None) -> Optional[List[MatchPrediction]]:
        """Gets batchsize number of predictions that need to be scored."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT * FROM MatchPredictions WHERE isScored = 0 ORDER BY matchDate ASC LIMIT ?",
                    [batchsize],
                )
                results = cursor.fetchall()
                if results is None:
                    return None
                
                return results
    
    def update_match_predictions(self, predictions: List[MatchPrediction]):
        """Updates predictions. Typically only used when marking predictions as being scored."""
        for prediction in predictions:
          bt.logging.trace(
              f"{prediction.axon.hotkey}: Marking prediction {str(prediction.predictionId)} as scored"
          )

          now_str = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")

          # Parse MatchPredictions into a list of values to update, marking each as scored with a timestamp of now.
          values = []
          values.append(
            [
                1,
                now_str,
                prediction.predictionId
            ]
          )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE MatchPredictions SET isScored = ?, scoredDate = ? WHERE predictionId = ?""",
                    values,
                )
                connection.commit()
        

    def get_miner_match_predictions(self, miner_hotkey: str, scored = False) -> Optional[List[MatchPrediction]]:
        """Gets a list of all predictions made by a miner."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT * FROM MatchPredictions WHERE hotkey = ?",
                    [miner_hotkey],
                )
                results = cursor.fetchall()
                if results is None:
                    return None
                
                return results

    def read_miner_last_prediction(self, miner_hotkey: str) -> Optional[dt.datetime]:
        """Gets when a specific miner last returned a prediction."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT MAX(lastUpdated) FROM MatchPrediction WHERE hotkey = ?", [miner_hotkey]
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
                cursor.execute("DELETE FROM MatchPredictions WHERE hotkey = ?", [hotkey])


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
