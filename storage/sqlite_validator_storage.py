import os
import csv
import shutil
import contextlib
import datetime as dt
import time
import bittensor as bt
import sqlite3
import threading
import random
from pydantic import ValidationError
from typing import Any, Dict, Optional, List, Tuple
from common.data import (
    Match,
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
    _instance: Optional['SqliteValidatorStorage'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'SqliteValidatorStorage':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    """Sqlite in-memory backed Validator Storage"""

    LEAGUES_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS Leagues (
                            leagueId      VARCHAR(50)     PRIMARY KEY,
                            leagueName    VARCHAR(50)     NOT NULL,
                            sport         INTEGER         NOT NULL,
                            isActive      INTEGER         DEFAULT 0,
                            lastUpdated   TIMESTAMP(6)    NOT NULL
                            )"""

    MATCHES_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS Matches (
                            matchId         VARCHAR(50)     PRIMARY KEY,
                            matchDate       TIMESTAMP(6)    NOT NULL,
                            sport           INTEGER         NOT NULL,
                            league          VARCHAR(50)     NOT NULL,
                            homeTeamName    VARCHAR(30)     NOT NULL,
                            awayTeamName    VARCHAR(30)     NOT NULL,
                            homeTeamScore   INTEGER         NULL,
                            awayTeamScore   INTEGER         NULL,
                            isComplete      INTEGER         DEFAULT 0,
                            lastUpdated     TIMESTAMP(6)    NOT NULL,
                            homeTeamOdds    FLOAT           NULL,
                            awayTeamOdds    FLOAT           NULL,
                            drawOdds        FLOAT           NULL
                            )"""
    
    MATCH_ODDS_CREATE = """CREATE TABLE IF NOT EXISTS MatchOdds (
                            matchId         VARCHAR(50)     NOT NULL,
                            homeTeamOdds    FLOAT           NULL,
                            awayTeamOdds    FLOAT           NULL,
                            drawOdds        FLOAT           NULL,
                            lastUpdated     TIMESTAMP(6)    NOT NULL
                            )"""
    
    MATCHPREDICTIONREQUESTS_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS MatchPredictionRequests (
                            matchId             VARCHAR(50) PRIMARY KEY,
                            prediction_24_hour  BOOLEAN     DEFAULT FALSE,
                            prediction_12_hour  BOOLEAN     DEFAULT FALSE,
                            prediction_4_hour   BOOLEAN     DEFAULT FALSE,
                            prediction_10_min   BOOLEAN     DEFAULT FALSE,
                            lastUpdated         TIMESTAMP   NOT NULL
                            )"""

    MATCHPREDICTIONS_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS MatchPredictions (
                            predictionId        INTEGER         PRIMARY KEY,
                            minerId             INTEGER         NOT NULL,
                            hotkey              VARCHAR(64)     NOT NULL,
                            matchId             VARCHAR(50)     NOT NULL,
                            matchDate           TIMESTAMP(6)    NOT NULL,
                            sport               INTEGER         NOT NULL,
                            league              VARCHAR(50)     NOT NULL,
                            homeTeamName        VARCHAR(30)     NOT NULL,
                            awayTeamName        VARCHAR(30)     NOT NULL,
                            homeTeamScore       INTEGER         NULL,
                            awayTeamScore       INTEGER         NULL,
                            isScored            INTEGER         DEFAULT 0,
                            scoredDate          TIMESTAMP(6)    NULL,
                            lastUpdated         TIMESTAMP(6)    NOT NULL,
                            predictionDate      TIMESTAMP(6)    NOT NULL,
                            probabilityChoice   VARCHAR(10)     NULL,
                            probability         FLOAT           NULL,
                            closingEdge         FLOAT           NULL,
                            isArchived          INTEGER         DEFAULT 0
                            )"""
    
    HOTFIX_MLB_20250410_MARKER_FILE = "HOTFIX_MLB_20250410_MARKER_FILE.txt"

    def __init__(self):
        self._initialized = False
        self.continuous_connection_do_not_reuse: Optional[sqlite3.Connection] = None
        self.lock = threading.RLock()

    def initialize(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
    
            sqlite3.register_converter("timestamp", tz_aware_timestamp_adapter)

            self.continuous_connection_do_not_reuse = self._create_connection()

            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()

                # Create the Matches table (if it does not already exist).
                # cursor.execute(SqliteValidatorStorage.LEAGUES_TABLE_CREATE)

                # Create the Matches table (if it does not already exist).
                cursor.execute(SqliteValidatorStorage.MATCHES_TABLE_CREATE)

                # Create the MatchOdds table (if it does not already exist).
                cursor.execute(SqliteValidatorStorage.MATCH_ODDS_CREATE)

                # Create the MatchPredictionRequests table (if it does not already exist).
                cursor.execute(SqliteValidatorStorage.MATCHPREDICTIONREQUESTS_TABLE_CREATE)

                # Create the MatchPredictions table (if it does not already exist).
                cursor.execute(SqliteValidatorStorage.MATCHPREDICTIONS_TABLE_CREATE)

                # Execute db hotfixes
                self.execute_db_hotfixes()

                # Commit the changes and close the connection
                connection.commit()

                # Lock to avoid concurrency issues on interacting with the database.
                self.lock = threading.RLock()

            # Execute archiving. Must run before cleanup.
            self.archive_old_data()
            # Execute cleanup queries
            self.cleanup()

    def _create_connection(self):
        # Create the database if it doesn't exist, defaulting to the local directory.
        # Use PARSE_DECLTYPES to convert accessed values into the appropriate type.
        connection = sqlite3.connect(
            "SportsTensorEdge.db",
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=120.0,
        )
        # Avoid using a row_factory that would allow parsing results by column name for performance.
        # connection.row_factory = sqlite3.Row
        connection.isolation_level = None
        return connection
    
    def get_connection(self):
        if not self._initialized:
            raise RuntimeError("SqliteValidatorStorage has not been initialized")
        return self.continuous_connection_do_not_reuse
    
    def execute_db_hotfixes(self):
        if os.path.exists(self.HOTFIX_MLB_20250410_MARKER_FILE):
            print(f"{self.HOTFIX_MLB_20250410_MARKER_FILE} hotfix has already been executed. Skipping.")
            return

        try:
            with self.lock:
                with contextlib.closing(self._create_connection()) as connection:
                    cursor = connection.cursor()
                    cursor.execute(
                        """UPDATE MatchPredictions SET isArchived = 1 WHERE matchId IN ('b270fb69f9d1bc85c1ea6d4bd1241f95', '45c43e101810be48851e88fde3cc15fd', '4700aae86174a44f726ab023b8889f2a')""",
                    )
                    cursor.execute(
                        """DELETE FROM Matches WHERE matchId IN ('30ad5abd755a61dbcf4ff79759e32a2f', '1c7a5a8373c1bae26e044037d6c12f21', 'ef1aa8b53826d5dbe374291e08cd6b2b')""",
                    )
                    connection.commit()
            
            # Create the marker file
            with open(self.HOTFIX_MLB_20250410_MARKER_FILE, "w") as f:
                f.write("Hotfix executed on: " + dt.datetime.now(dt.timezone.utc).isoformat())
            
            print(f"Marker file created: {self.HOTFIX_MLB_20250410_MARKER_FILE}. This script will not run the hotfix again.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def cleanup(self):
        """Cleanup the database."""
        print("========================== Database status checks and cleanup ==========================")
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                
                db_size_bytes = os.path.getsize("SportsTensorEdge.db")
                db_size_gb = db_size_bytes / (1024 ** 3)
                db_size_mb = db_size_bytes / (1024 ** 2)
                print(f"SportsTensorEdge.db size: {db_size_gb:.2f} GB ({db_size_mb:.2f} MB)")
                
                # Print the total number of rows in the MatchPredictions table
                cursor.execute("SELECT COUNT(*) FROM MatchPredictions")
                total_rows = cursor.fetchone()[0]
                print(f"Total number of rows in MatchPredictions: {total_rows}")
                
                try:
                    # Print the total number of rows in the MatchPredictions table
                    cursor.execute(f"SELECT COUNT(*) FROM MatchPredictions WHERE isScored = 0 AND lastUpdated < DATETIME('now', '-{SCORING_CUTOFF_IN_DAYS} day')")
                    total_unscored_rows = cursor.fetchone()[0]
                    print(f"Total number of MatchPredictions {SCORING_CUTOFF_IN_DAYS}+ days old that haven't been scored: {total_unscored_rows}")

                    # Execute cleanup queries
                    if total_unscored_rows > 0:
                        # Clean up old predictions that haven't been scored (for whatever reason) and never will be
                        print(f"Deleting abandoned predictions older than {SCORING_CUTOFF_IN_DAYS} days...")
                        cursor.execute(
                            f"DELETE FROM MatchPredictions WHERE isScored = 0 AND lastUpdated < DATETIME('now', '-{SCORING_CUTOFF_IN_DAYS} day')"
                        )
                        connection.commit()

                    # Run VACUUM to reclaim unused space
                    print("Running VACUUM to reclaim unused space...")
                    cursor.execute("VACUUM")
                    
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

    def archive_old_data(self):
        """
        Archive leagues based on CSV configuration.
        
        The CSV should have the following format:
        LEAGUE,SHORTNAME,ARCHIVING STATUS,SEASON/YEAR,START,END
        
        If ARCHIVING STATUS is 'Active', we will archive the league.
        """
        print("================================== Database archiving ==================================")
        
        # Create archives directory if it doesn't exist
        archive_dir = os.path.join("storage", "archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Path to CSV file containing archiving configuration
        csv_path = os.path.join("storage", "league_archiving.csv")
        
        if not os.path.exists(csv_path):
            print(f"Archiving CSV file not found at {csv_path}. Skipping archiving.")
            return
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                leagues_to_archive = []
                
                for row in reader:
                    # Check if this league should be archived
                    if row['ARCHIVING STATUS'].strip() == 'Active':
                        leagues_to_archive.append({
                            'league': row['LEAGUE'].strip(),
                            'shortname': row['SHORTNAME'].strip(),
                            'season': row['SEASON/YEAR'].strip(),
                            'start': row['START'].strip(),
                            'end': row['END'].strip()
                        })
            
            if not leagues_to_archive:
                print("No leagues marked for archiving in the CSV.")
                return
            
            # make a copy of the main database, just in case
            shutil.copy("SportsTensorEdge.db", "SportsTensorEdge_preArchive.db")
            print(f"Created pre archive copy of main db: SportsTensorEdge_preArchive.db")

            print(f"Found {len(leagues_to_archive)} leagues to archive: {[l['league'] for l in leagues_to_archive]}")
            
            with self.lock:
                # Work through each league that needs to be archived
                for league_info in leagues_to_archive:
                    print("-" * 100)
                    league_name = league_info['league']
                    shortname = league_info['shortname']
                    season = league_info['season']
                    league_start = league_info['start']
                    league_end = league_info['end']
                    
                    # Define the archive filename
                    archive_filename = f"SportsTensorEdge_{shortname}_{season}.db"
                    archive_path = os.path.join(archive_dir, archive_filename)
                    
                    # Check if archive already exists
                    if os.path.exists(archive_path):
                        print(f"Archive for {league_name} ({season}) already exists at {archive_path}. Skipping.")
                        continue
                    
                    print(f"Archiving {league_name} ({season})...")
                    
                    # 1. Copy the main database to create the archive
                    shutil.copy("SportsTensorEdge.db", archive_path)
                    print(f"Created archive copy at {archive_path}")
                    
                    # 2. Connect to the archive database and delete non-relevant data
                    self._process_archive_db(archive_path, league_name, league_start, league_end)
                    
                    # 3. Delete the archived league data from the main database
                    self._delete_archived_league_from_main_db(league_name, league_start, league_end)
                    
                    print(f"Successfully archived {league_name} ({season})")
                
                print("-" * 100)
                print("Archiving process completed successfully.")
                
        except Exception as e:
            print(f"Error during archiving process: {e}")
            raise e
        
        print("========================================================================================")
    
    def _process_archive_db(self, archive_path, league_name, league_start, league_end):
        """Process the archive database by removing non-relevant data"""
        try:
            # Connect to the archive database
            with contextlib.closing(sqlite3.connect(archive_path)) as archive_conn:
                cursor = archive_conn.cursor()
                
                # Begin transaction
                cursor.execute("BEGIN TRANSACTION")
                
                # Keep only data related to the specified league
                # First, get list of matchIds for this league
                cursor.execute("SELECT matchId FROM Matches WHERE league = ? AND matchDate >= ? AND matchDate <= ?", (league_name, league_start, league_end))
                relevant_match_ids = [row[0] for row in cursor.fetchall()]
                
                if relevant_match_ids:
                    match_ids_tuple = tuple(relevant_match_ids)
                    
                    # Delete data from related tables
                    if len(match_ids_tuple) == 1:
                        # Handle single item tuples
                        match_id_sql = f"!= '{match_ids_tuple[0]}'"
                    else:
                        match_id_sql = f"NOT IN {match_ids_tuple}"
                    
                    print(f"  Archive DB: Removing data from MatchPredictions...")
                    cursor.execute(f"DELETE FROM MatchPredictions WHERE matchId {match_id_sql}")
                    
                    print(f"  Archive DB: Removing data from MatchOdds...")
                    cursor.execute(f"DELETE FROM MatchOdds WHERE matchId {match_id_sql}")
                    
                    print(f"  Archive DB: Removing data from MatchPredictionRequests...")
                    cursor.execute(f"DELETE FROM MatchPredictionRequests WHERE matchId {match_id_sql}")
                    
                    print(f"  Archive DB: Removing non-relevant matches...")
                    cursor.execute(f"DELETE FROM Matches WHERE matchId {match_id_sql}")
                
                    # Commit the transaction
                    cursor.execute("COMMIT")
                    
                    # Optimize the database
                    print(f"  Archive DB: Running VACUUM to optimize database...")
                    cursor.execute("VACUUM")
                    
                    # Check the integrity of the archive
                    cursor.execute("PRAGMA integrity_check")
                    integrity_result = cursor.fetchone()[0]
                    if integrity_result != "ok":
                        print(f"  Archive DB: *** WARNING: Integrity check failed for {archive_path}! ***")
                    else:
                        print(f"  Archive DB: Integrity check passed.")
                    
                    # Get statistics on archived data
                    cursor.execute("SELECT COUNT(*) FROM Matches")
                    matches_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM MatchPredictions")
                    predictions_count = cursor.fetchone()[0]
                    
                    print(f"  Archive DB: Contains {matches_count} matches and {predictions_count} predictions.")
                    
                    db_size_bytes = os.path.getsize(archive_path)
                    db_size_mb = db_size_bytes / (1024 ** 2)
                    print(f"  Archive DB: Size: {db_size_mb:.2f} MB")
                
                else:
                    print(f"  Archive DB: No matches found for league {league_name} ({league_start} - {league_end}). Deleting archive.")
                    cursor.execute("ROLLBACK")
                    # Delete the archive file if no relevant data was found
                    os.remove(archive_path)
                
        except Exception as e:
            print(f"Error processing archive database: {e}")
            raise e
    
    def _delete_archived_league_from_main_db(self, league_name, league_start, league_end):
        """Delete the archived league data from the main database"""
        try:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                
                # Begin transaction
                cursor.execute("BEGIN TRANSACTION")
                
                # First, find all match IDs for this league
                cursor.execute("SELECT matchId FROM Matches WHERE league = ? AND matchDate >= ? AND matchDate <= ?", (league_name, league_start, league_end))
                match_ids = [row[0] for row in cursor.fetchall()]
                
                if not match_ids:
                    print(f"  Main DB: No matches found for league {league_name}. Nothing to delete.")
                    cursor.execute("ROLLBACK")
                    return
                
                match_ids_tuple = tuple(match_ids)
                
                # Delete related data
                if len(match_ids_tuple) == 1:
                    # Handle single item tuples
                    match_id_sql = f"= '{match_ids_tuple[0]}'"
                else:
                    match_id_sql = f"IN {match_ids_tuple}"
                
                print(f"  Main DB: Deleting {len(match_ids)} matches and related data for {league_name}...")
                
                # Delete predictions
                cursor.execute(f"SELECT COUNT(*) FROM MatchPredictions WHERE matchId {match_id_sql}")
                predictions_count = cursor.fetchone()[0]
                print(f"  Main DB: Deleting {predictions_count} rows from MatchPredictions...")
                cursor.execute(f"DELETE FROM MatchPredictions WHERE matchId {match_id_sql}")
                
                # Delete odds
                cursor.execute(f"SELECT COUNT(*) FROM MatchOdds WHERE matchId {match_id_sql}")
                odds_count = cursor.fetchone()[0]
                print(f"  Main DB: Deleting {odds_count} rows from MatchOdds...")
                cursor.execute(f"DELETE FROM MatchOdds WHERE matchId {match_id_sql}")
                
                # Delete prediction requests
                cursor.execute(f"SELECT COUNT(*) FROM MatchPredictionRequests WHERE matchId {match_id_sql}")
                requests_count = cursor.fetchone()[0]
                print(f"  Main DB: Deleting {requests_count} rows from MatchPredictionRequests...")
                cursor.execute(f"DELETE FROM MatchPredictionRequests WHERE matchId {match_id_sql}")
                
                # Delete matches
                print(f"  Main DB: Deleting matches for {league_name}...")
                cursor.execute(f"DELETE FROM Matches WHERE matchId {match_id_sql}")
                
                # Commit changes
                cursor.execute("COMMIT")
                
                # Log successful deletion
                print(f"  Main DB: Successfully deleted {league_name} data ({len(match_ids)} matches, {predictions_count} predictions)")

                # Optimize the database
                print(f"  Running VACUUM to optimize database...")
                cursor.execute("VACUUM")
                
                # Check the integrity of the archive
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                if integrity_result != "ok":
                    print(f"  *** WARNING: Integrity check failed after archiving {league_name} ({league_start} - {league_end})! ***")
                else:
                    print(f"  Integrity check passed.")
                
                # Size after deletion
                db_size_bytes = os.path.getsize("SportsTensorEdge.db")
                db_size_mb = db_size_bytes / (1024 ** 2)
                print(f"  Main DB: Current size: {db_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"Error deleting league data from main database: {e}")
            raise e

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
                    match.homeTeamOdds,
                    match.awayTeamOdds,
                    match.drawOdds,
                    match.isComplete,
                    now_str,
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """
                        INSERT OR IGNORE INTO Matches (matchId, matchDate, sport, league, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, homeTeamOdds, awayTeamOdds, drawOdds, isComplete, lastUpdated) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
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
                    match.matchDate,
                    match.homeTeamName,
                    match.awayTeamName,
                    match.homeTeamScore,
                    match.awayTeamScore,
                    match.homeTeamOdds,
                    match.awayTeamOdds,
                    match.drawOdds,
                    match.isComplete,
                    now_str,
                    match.matchId,
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE Matches SET matchDate = ?, homeTeamName = ?, awayTeamName = ?, homeTeamScore = ?, awayTeamScore = ?, homeTeamOdds = ?, awayTeamOdds = ?, drawOdds = ?, isComplete = ?, lastUpdated = ? WHERE matchId = ?""",
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
            
    def insert_match_odds(self, match_odds: List[tuple[str, float, float, float, dt.datetime]]):
        """Stores match odds in the database."""
        values = []
        for odds in match_odds:
            values.append(
                [
                    odds[0],
                    odds[1],
                    odds[2],
                    odds[3],
                    odds[4],
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """
                        INSERT OR IGNORE INTO MatchOdds (matchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated) 
                        VALUES (?, ?, ?, ?, ?)
                    """,
                    values,
                )
                connection.commit()
    
    def delete_match_odds(self):
        """Deletes all match odds from the database."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """
                        DELETE FROM MatchOdds
                    """,
                )
                connection.commit()

    def check_match_odds(self, matchId: str, lastUpdated: str) -> bool:
        """Check if match odds with the given ID exists in the database."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """SELECT EXISTS(SELECT 1 FROM MatchOdds WHERE matchId = ? AND lastUpdated = ?)""",
                    (matchId, lastUpdated),
                )
                return cursor.fetchone()[0]

    def get_match_odds(self, matchId: str = None):
        """Gets all the match odds for the provided matchId."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                if matchId:
                    cursor.execute(
                        """
                        SELECT * 
                        FROM MatchOdds
                        WHERE matchId = ?
                        ORDER BY lastUpdated ASC
                        """,
                        (matchId,)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * 
                        FROM MatchOdds
                        ORDER BY lastUpdated ASC
                        """,
                    )

                results = cursor.fetchall()
                if not results:
                    return []
                
                return results
            
    def get_match_odds_by_batch(self, matchIds: List[str]) -> Dict[str, List[Tuple[str, float, float, float, dt.datetime]]]:
        """Gets all the match odds for the provided matchIds in a single query."""
        if not matchIds:
            return {}
        
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                
                # Create placeholders for IN clause
                placeholders = ', '.join(['?' for _ in matchIds])
                
                query = f"""
                SELECT matchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated
                FROM MatchOdds 
                WHERE matchId IN ({placeholders})
                ORDER BY matchId, lastUpdated ASC
                """
                
                cursor.execute(query, matchIds)
                results = cursor.fetchall()
                
                if not results:
                    return {}
                
                # Group odds by matchId
                odds_by_match = {}
                for row in results:
                    match_id = row[0]
                    odds_tuple = (
                        row[0],  # matchId
                        row[1],  # homeTeamOdds
                        row[2],  # awayTeamOdds
                        row[3],  # drawOdds
                        row[4]   # lastUpdated
                    )
                    
                    if match_id not in odds_by_match:
                        odds_by_match[match_id] = []
                    odds_by_match[match_id].append(odds_tuple)
                
                return odds_by_match

    def get_matches_to_predict(self, batchsize: Optional[int] = None) -> List[Match]:
        """Gets batchsize number of matches ready to be predicted."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:

                # Calculate the current timestamp
                current_timestamp = int(time.time())
                # Calculate the lower bound timestamp (earliest match date allowed for predictions)
                lower_bound_timestamp = (
                    current_timestamp + MIN_PREDICTION_TIME_THRESHOLD
                )
                # Calculate the upper bound timestamp (latest match date allowed for predictions) with buffer of 1 hour
                upper_bound_timestamp = (
                    current_timestamp + (MAX_PREDICTION_DAYS_THRESHOLD * 24 * 3600) + 3600
                )
                # Convert timestamps to strings in 'YYYY-MM-DD HH:MM:SS' format
                lower_bound_str = dt.datetime.utcfromtimestamp(
                    lower_bound_timestamp
                ).strftime("%Y-%m-%d %H:%M:%S")
                upper_bound_str = dt.datetime.utcfromtimestamp(
                    upper_bound_timestamp
                ).strftime("%Y-%m-%d %H:%M:%S")

                cursor = connection.cursor()
                query = """
                    SELECT * 
                    FROM Matches
                    WHERE isComplete = 0
                    AND matchDate BETWEEN ? AND ?
                    ORDER BY RANDOM()
                    """
                if batchsize:
                    query += "LIMIT ?"
                    cursor.execute(query, (lower_bound_str, upper_bound_str, batchsize))
                else:
                    cursor.execute(query, (lower_bound_str, upper_bound_str))
                
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
            
    def get_recently_completed_matches(self, matchDateSince: dt.datetime, league: Optional[League] = None) -> List[Match]:
        """Gets completed matches since the passed in date."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                # Convert datetime to string in 'YYYY-MM-DD HH:MM:SS' format
                dateSince = matchDateSince.strftime("%Y-%m-%d %H:%M:%S")

                cursor = connection.cursor()
                params = [dateSince]
                query = """
                    SELECT * 
                    FROM Matches
                    WHERE isComplete = 1
                    AND matchDate > ?
                    """
                
                if league:
                    query += "AND (league = ? OR league = ?)"
                    params.append(league.name)
                    params.append(league.value)
                
                cursor.execute(query, params)
                
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
            
    def get_total_prediction_requests_count(self, matchDateSince: dt.datetime, league: Optional[League] = None) -> int:
        """Gets total count of prediction requests sent to miners since the passed in date."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                # Convert datetime to string in 'YYYY-MM-DD HH:MM:SS' format
                dateSince = matchDateSince.strftime("%Y-%m-%d %H:%M:%S")
                
                cursor = connection.cursor()
                params = [dateSince]
                
                # Count rows where any type of prediction was requested
                query = """
                    SELECT SUM(
                        CASE WHEN prediction_24_hour THEN 1 ELSE 0 END +
                        CASE WHEN prediction_12_hour THEN 1 ELSE 0 END +
                        CASE WHEN prediction_4_hour THEN 1 ELSE 0 END +
                        CASE WHEN prediction_10_min THEN 1 ELSE 0 END
                    ) as total_requests
                    FROM MatchPredictionRequests mpr
                    JOIN Matches m ON mpr.matchId = m.matchId
                    WHERE m.matchDate > ?
                    AND m.homeTeamOdds IS NOT NULL
                    AND m.awayTeamOdds IS NOT NULL
                    AND m.isComplete = 1
                    """
                
                if league:
                    query += "AND (m.league = ? OR m.league = ?)"
                    params.append(league.name)
                    params.append(league.value)
                
                cursor.execute(query, params)
                
                results = cursor.fetchone()
                return results[0] if results and results[0] is not None else 0
            
    def update_match_prediction_request(self, matchId: str, request_time: str):
        """Updates a match prediction request with the status of the request_time."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                cursor = connection.cursor()
                cursor.execute(
                    f"""
                    INSERT INTO MatchPredictionRequests (matchId, {request_time}, lastUpdated)
                    VALUES (?, TRUE, ?)
                    ON CONFLICT(matchId) DO UPDATE SET
                    {request_time} = TRUE,
                    lastUpdated = ?
                    """,
                    (matchId, now_str, now_str)
                )
                connection.commit()

    def get_match_prediction_requests(self, matchId: Optional[str] = None) -> Dict[str, Dict[str, bool]]:
        """Gets all match prediction requests or a specific match prediction request."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                if matchId:
                    cursor.execute(
                        """
                        SELECT mpr.matchId, mpr.prediction_24_hour, mpr.prediction_12_hour, mpr.prediction_4_hour, mpr.prediction_10_min
                        FROM MatchPredictionRequests mpr
                        WHERE mpr.matchId = ?
                        """,
                        (matchId,)
                    )
                    row = cursor.fetchone()
                    if row:
                        return {row[0]: {
                            '24_hour': bool(row[1]),
                            '12_hour': bool(row[2]),
                            '4_hour': bool(row[3]),
                            '10_min': bool(row[4])
                        }}
                    else:
                        return {}
                else:
                    cursor.execute(
                        """
                        SELECT mpr.matchId, mpr.prediction_24_hour, mpr.prediction_12_hour, mpr.prediction_4_hour, mpr.prediction_10_min
                        FROM MatchPredictionRequests mpr
                        """
                    )
                    return {row[0]: {
                        '24_hour': bool(row[1]),
                        '12_hour': bool(row[2]),
                        '4_hour': bool(row[3]),
                        '10_min': bool(row[4])
                    } for row in cursor.fetchall()}
                
    def check_and_fix_match_prediction_requests(self, matchId: Optional[str] = None) -> None:
        """Checks and fixes the match prediction requests in the database."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                if matchId:
                    # Check if the matchId exists in the MatchPredictionRequests table last updated more than 2 days ago for wiggle room
                    cursor.execute(
                        """
                        SELECT mpr.matchId, mpr.prediction_24_hour, mpr.prediction_12_hour, mpr.prediction_4_hour, mpr.prediction_10_min, mpr.lastUpdated
                        FROM MatchPredictionRequests mpr
                        WHERE mpr.matchId = ?
                        AND mpr.lastUpdated < datetime('now', '-2 day')
                        """,
                        (matchId,)
                    )
                    row = cursor.fetchone()
                    # if record exists, let's delete it and force a new entry for this matchId
                    if row:
                        bt.logging.debug(f"Found MatchPredictionRequests for matchId {matchId} that is older than 2 days ({row[5]}). Deleting it for reset (shouldn't exist).")
                        self.delete_match_prediction_requests(matchId)
                
    def delete_match_prediction_requests(self, matchId: Optional[str] = None) -> None:
        """Deletes specific match prediction requests, or match prediction requests from matches that are older than 1 day."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                if matchId:
                    # Delete the specific matchId from MatchPredictionRequests
                    cursor.execute(
                        """
                        DELETE FROM MatchPredictionRequests
                        WHERE matchId = ?
                        """,
                        (matchId,)
                    )
                else:
                    cursor.execute(
                        """
                        DELETE FROM MatchPredictionRequests
                        WHERE matchId IN (
                            SELECT mpr.matchId
                            FROM MatchPredictionRequests mpr
                            JOIN matches m ON mpr.matchId = m.matchId
                            WHERE datetime(m.matchDate) < datetime('now', '-1 day')
                        )
                        """
                    )
                connection.commit()

    def insert_match_predictions(self, predictions: List[GetMatchPrediction]):
        """Stores unscored predictions returned from miners."""
        values = []
        for prediction in predictions:
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
                    prediction.match_prediction.probabilityChoice,
                    prediction.match_prediction.probability,
                    prediction.match_prediction.predictionDate,
                    now_str,
                ]
            )

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """
                        INSERT OR IGNORE INTO MatchPredictions (minerId, hotkey, matchId, matchDate, sport, league, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, probabilityChoice, probability, predictionDate, lastUpdated) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                connection.commit()

    def delete_unscored_deregistered_match_predictions(self, miner_hotkeys: List[str], miner_uids: List[int]):
        """Deletes unscored predictions returned from miners that are no longer registered."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()

                # Get the unique hotkeys and count of predictions to be deleted
                cursor.execute(
                    """
                    SELECT hotkey, COUNT(*) 
                    FROM MatchPredictions 
                    WHERE isScored = 0 AND hotkey NOT IN ({})
                    GROUP BY hotkey
                    """.format(",".join("?" * len(miner_hotkeys))),
                    list(miner_hotkeys),
                )
                deletion_data = cursor.fetchall()
                
                if deletion_data:
                    hotkeys_to_delete = [row[0] for row in deletion_data]
                    counts_to_delete = [row[1] for row in deletion_data]
                    total_count_to_delete = sum(counts_to_delete)

                    # Delete the predictions that are not from registered hotkeys
                    cursor.execute(
                        "DELETE FROM MatchPredictions WHERE isScored = 0 AND hotkey NOT IN ({})".format(
                            ",".join("?" * len(miner_hotkeys))
                        ),
                        list(miner_hotkeys),
                    )
                    connection.commit()

                    # Log the details
                    bt.logging.info(f"Deleted a total of {total_count_to_delete} unscored predictions from {len(hotkeys_to_delete)} deregistered miners.")
                    bt.logging.info(f"Affected hotkeys and their prediction counts: {dict(zip(hotkeys_to_delete, counts_to_delete))}")
                else:
                    bt.logging.info("No unscored predictions from deregistered miners found for deletion.")

                # Next, loop through active hotkeys and delete any rows with different uids
                values = []
                for miner_hotkey, miner_uid in zip(miner_hotkeys, miner_uids):
                    # Parse MatchPredictions into a list of values to check/delete
                    values.append([miner_hotkey, miner_uid])

                with self.lock:
                    with contextlib.closing(self._create_connection()) as connection:
                        cursor = connection.cursor()
                        cursor.executemany(
                            """DELETE FROM MatchPredictions WHERE isScored = 0 AND hotkey = ? AND minerId != ?""",
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
                    SELECT mp.*, m.homeTeamScore as actualHomeTeamScore, m.awayTeamScore as actualAwayTeamScore, m.homeTeamOdds, m.awayTeamOdds, COALESCE(m.drawOdds, 0) as drawOdds
                    FROM MatchPredictions mp
                    JOIN Matches m ON (m.matchId = mp.matchId)
                    WHERE mp.isScored = 0
                    AND m.isComplete = 1
                    AND mp.matchDate > ?
                    AND mp.probabilityChoice IS NOT NULL
                    AND mp.probability IS NOT NULL
                    AND m.homeTeamScore IS NOT NULL
                    AND m.awayTeamScore IS NOT NULL
                    AND m.homeTeamOdds IS NOT NULL
                    AND m.awayTeamOdds IS NOT NULL
                    ORDER BY RANDOM()
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
                        "lastUpdated": row[13],
                        "predictionDate": row[14],
                        "probabilityChoice": row[15],
                        "probability": row[16],
                        #"closingEdge": row[17],
                        #"isArchived": row[18]
                    }
                    try:
                        # Ensure all required fields are present in prediction_data
                        prediction_data.setdefault('scoredDate', None)
                        prediction_data.setdefault('closingEdge', None)

                        combined_predictions.append(
                            MatchPredictionWithMatchData(
                                prediction=MatchPrediction(**prediction_data),
                                actualHomeTeamScore=row[19],
                                actualAwayTeamScore=row[20],
                                homeTeamOdds=row[21],
                                awayTeamOdds=row[22],
                                drawOdds=row[23],
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
            values.append([prediction.closingEdge, 1, now_str, prediction.predictionId])

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE MatchPredictions SET closingEdge = ?, isScored = ?, scoredDate = ? WHERE predictionId = ?""",
                    values,
                )
                connection.commit()

    def archive_match_predictions(self, miner_hotkeys: List[str], miner_uids: List[int]):
        """Updates predictions with isArchived 1. Typically only used when marking predictions achived after miner has been deregistered."""
        # Archive the predictions that are not from registered hotkeys
        bt.logging.trace(
            f"Archiving predictions for deregistered miners"
        )
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                params = [1] + miner_hotkeys
                cursor.execute(
                    "UPDATE MatchPredictions SET isArchived = ? WHERE isScored = 1 AND hotkey NOT IN ({})".format(
                        ",".join("?" * len(miner_hotkeys))
                    ),
                    params,
                )
                connection.commit()

        # Next, loop through active hotkeys and mark any rows with different uids as archived
        values = []
        for miner_hotkey, miner_uid in zip(miner_hotkeys, miner_uids):
            # Parse MatchPredictions into a list of values to update
            values.append([1, miner_hotkey, miner_uid])

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(
                    """UPDATE MatchPredictions SET isArchived = ? WHERE isScored = 1 AND hotkey = ? AND minerId != ?""",
                    values,
                )
                connection.commit()

    def get_total_match_predictions_by_miner(self, miner_hotkey: str, miner_uid: int, matchDateSince: Optional[dt.datetime] = None, league: Optional[League] = None) -> int:
        """Gets the total number of predictions a miner has made since being registered. Must be scored and not archived."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()

                query = """
                    SELECT COUNT(*)
                    FROM MatchPredictions
                    WHERE hotkey = ?
                    AND minerId = ?
                    AND isScored = 1
                    AND isArchived = 0
                """
                params = [miner_hotkey, miner_uid]
                
                if matchDateSince:
                    query += " AND matchDate > ? "
                    params.append(matchDateSince.strftime("%Y-%m-%d %H:%M:%S"))
                
                if league:
                    query += " AND league = ? "
                    params.append(league.value)
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                if result is not None:
                    return result[0]
                else:
                    return 0
    
    def get_miner_match_predictions(
        self, miner_hotkey: str, miner_uid: int, league: League=None, scored: bool=False, batchSize: int=None
    ) -> Optional[List[MatchPredictionWithMatchData]]:
        """Gets a list of all predictions made by a miner. Include match data."""
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()

                query = """
                    SELECT mp.*, m.homeTeamScore as actualHomeTeamScore, m.awayTeamScore as actualAwayTeamScore, m.homeTeamOdds, m.awayTeamOdds, COALESCE(m.drawOdds, 0) as drawOdds
                    FROM MatchPredictions mp
                    JOIN Matches m ON (m.matchId = mp.matchId)
                    WHERE mp.hotkey = ?
                    AND mp.minerId = ?
                    AND mp.isArchived = 0
                """

                params = [miner_hotkey, miner_uid]
                if league:
                    query += " AND mp.league = ? "
                    params.append(league.value)
                if scored:
                    query += " AND mp.isScored = 1 "
                    query += " AND mp.closingEdge IS NOT NULL "
                    query += " AND m.isComplete = 1 "
                    query += " AND m.homeTeamOdds IS NOT NULL"
                    query += " AND m.awayTeamOdds IS NOT NULL"
                else:
                    query += " AND mp.isScored = 0 "
                
                query += " ORDER BY mp.predictionDate DESC"
                
                if batchSize:
                    query += f" LIMIT {batchSize}"

                cursor.execute(
                    query,
                    params,
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
                        "scoredDate": row[12],
                        "lastUpdated": row[13],
                        "predictionDate": row[14],
                        "probabilityChoice": row[15],
                        "probability": round(row[16], 4),
                        "closingEdge": row[17],
                        "isArchived": row[18]
                    }
                    try:
                        combined_predictions.append(
                            MatchPredictionWithMatchData(
                                prediction=MatchPrediction(**prediction_data),
                                actualHomeTeamScore=row[19],
                                actualAwayTeamScore=row[20],
                                homeTeamOdds=row[21],
                                awayTeamOdds=row[22],
                                drawOdds=row[23],
                            )
                        )
                    except ValidationError as e:
                        bt.logging.error(f"Validation error for row {row}: {e}")
                
                return combined_predictions
            
    def get_miner_match_predictions_by_batch(
        self, miner_data: List[tuple[str, int]], league: League=None, scored: bool=True, batch_size: int=None
    ) -> Optional[Dict[int, List[MatchPredictionWithMatchData]]]:
        """Gets a dictionary of miner UIDs to a list of predictions made by that miner. Include match data."""
        if not miner_data:
            return {}

        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()

                # Create placeholders for the IN clause
                miner_uid_placeholders = ', '.join(['?' for _ in miner_data])
                miner_hotkey_placeholders = ', '.join(['?' for _ in miner_data])
                miner_pair_placeholders = ', '.join(['(?, ?)' for _ in miner_data])
                
                # Build the scored conditions to match your original query exactly
                scored_conditions = ""
                if scored:
                    scored_conditions = """
                        AND mp.isScored = 1
                        AND mp.closingEdge IS NOT NULL
                        AND m.isComplete = 1
                        AND m.homeTeamOdds IS NOT NULL
                        AND m.awayTeamOdds IS NOT NULL
                    """
                else:
                    scored_conditions = "AND mp.isScored = 0"
                
                query = f"""
                WITH RankedPredictions AS (
                    SELECT 
                        mp.predictionId,
                        mp.minerId,
                        mp.hotkey,
                        mp.matchId,
                        mp.matchDate,
                        mp.sport,
                        mp.league,
                        mp.homeTeamName,
                        mp.awayTeamName,
                        mp.homeTeamScore,
                        mp.awayTeamScore,
                        mp.isScored,
                        mp.scoredDate,
                        mp.lastUpdated,
                        mp.predictionDate,
                        mp.probabilityChoice,
                        mp.probability,
                        mp.closingEdge,
                        mp.isArchived,
                        m.homeTeamScore as actualHomeTeamScore,
                        m.awayTeamScore as actualAwayTeamScore,
                        m.homeTeamOdds,
                        m.awayTeamOdds,
                        COALESCE(m.drawOdds, 0) as drawOdds,
                        ROW_NUMBER() OVER (
                            PARTITION BY mp.minerId, mp.hotkey 
                            ORDER BY mp.predictionDate DESC
                        ) as rn
                    FROM MatchPredictions mp
                    JOIN Matches m ON (m.matchId = mp.matchId)
                    WHERE mp.minerId IN ({miner_uid_placeholders})
                    AND mp.hotkey IN ({miner_hotkey_placeholders})
                    AND mp.league = ?
                    AND mp.isArchived = 0
                    AND (mp.minerId, mp.hotkey) IN ({miner_pair_placeholders})
                    {scored_conditions}
                )
                SELECT * 
                FROM RankedPredictions 
                WHERE rn <= ?
                ORDER BY minerId, predictionDate DESC
                """
                
                # Prepare parameters
                miner_uids = [uid for uid, hotkey in miner_data]
                miner_hotkeys = [hotkey for uid, hotkey in miner_data]
                miner_pairs = [item for uid, hotkey in miner_data for item in (uid, hotkey)]
                
                params = (
                    *miner_uids,           # For miner_uid IN clause
                    *miner_hotkeys,        # For miner_hotkey IN clause  
                    league.value,          # For league = ?
                    *miner_pairs,          # For (miner_uid, miner_hotkey) IN clause
                    batch_size             # For rn <= ?
                )
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                if not results:
                    return {}
                
                # Group by miner_uid - use tuple indexing consistently
                predictions_by_miner = {}
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
                        "scoredDate": row[12],
                        "lastUpdated": row[13],
                        "predictionDate": row[14],
                        "probabilityChoice": row[15],
                        "probability": round(row[16], 4),
                        "closingEdge": row[17],
                        "isArchived": row[18]
                    }
                    try:
                        uid = row[1]  # minerId is at index 1
                        if uid not in predictions_by_miner:
                            predictions_by_miner[uid] = []

                        predictions_by_miner[uid].append(
                            MatchPredictionWithMatchData(
                                prediction=MatchPrediction(**prediction_data),
                                actualHomeTeamScore=row[19],  # actualHomeTeamScore
                                actualAwayTeamScore=row[20],  # actualAwayTeamScore
                                homeTeamOdds=row[21],         # homeTeamOdds
                                awayTeamOdds=row[22],         # awayTeamOdds
                                drawOdds=row[23],             # drawOdds
                            )
                        )
                    except ValidationError as e:
                        bt.logging.error(f"Validation error for row {row}: {e}")
                
                return predictions_by_miner

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

# Global accessor function
def get_storage() -> SqliteValidatorStorage:
    return SqliteValidatorStorage.get_instance()
