import mysql.connector
import logging
import datetime as dt
from datetime import timezone
from api.config import IS_PROD, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
import os


def get_matches():
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT * FROM matches
            WHERE matchDate BETWEEN NOW() - INTERVAL 10 DAY AND NOW() + INTERVAL 48 HOUR
        """
        )
        match_list = cursor.fetchall()

        return match_list

    except Exception as e:
        logging.error(
            "Failed to retrieve matches from the MySQL database", exc_info=True
        )
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def get_match_by_id(match_id):
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM matches WHERE matchId = %s", (match_id,))
        match = cursor.fetchone()

        return match

    except Exception as e:
        logging.error("Failed to retrieve match from the MySQL database", exc_info=True)
        return False
    finally:
        cursor.close()
        conn.close()


def insert_match(match_id, event, sport_type, is_complete, current_utc_time):
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO matches (matchId, matchDate, sport, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, matchLeague, isComplete, lastUpdated) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                matchDate=VALUES(matchDate),
                sport=VALUES(sport),
                homeTeamName=VALUES(homeTeamName),
                awayTeamName=VALUES(awayTeamName),
                homeTeamScore=VALUES(homeTeamScore),
                awayTeamScore=VALUES(awayTeamScore),
                matchLeague=VALUES(matchLeague),
                isComplete=VALUES(isComplete),
                lastUpdated=VALUES(lastUpdated)
            """,
            (
                match_id,
                event.get("strTimestamp"),
                sport_type,
                event.get("strHomeTeam"),
                event.get("strAwayTeam"),
                event.get("intHomeScore"),
                event.get("intAwayScore"),
                event.get("strLeague"),
                is_complete,
                current_utc_time,
            ),
        )

        conn.commit()
        logging.info("Data inserted or updated in database")

    except Exception as e:
        logging.error("Failed to insert match in MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()


def upload_prediction_results(prediction_results):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        current_utc_time = dt.datetime.now(timezone.utc)
        current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")

        """
        {
            'scores': prediction_scores,
            'correct_winner_results': correct_winner_results,
            'uids': prediction_rewards_uids,
            'hotkeys': prediction_results_hotkeys,
            'sports': prediction_sports,
            'leagues': prediction_leagues
        }
        """

        # Prepare the data for executemany
        data_to_insert = list(
            zip(
                prediction_results["hotkeys"],
                prediction_results["uids"],
                prediction_results["leagues"],
                prediction_results["sports"],
                [1] * len(prediction_results["uids"]),
                prediction_results["correct_winner_results"],
                prediction_results["scores"],
            )
        )

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
        c.executemany(
            f"""
            INSERT INTO {prediction_scores_table_name} (
                miner_hotkey,
                miner_uid,
                league,
                sport,
                total_predictions,
                winner_predictions,
                avg_score,
                last_updated
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, NOW()
            ) ON DUPLICATE KEY UPDATE
                total_predictions = total_predictions + VALUES(total_predictions),
                winner_predictions = winner_predictions + VALUES(winner_predictions),
                avg_score = ((avg_score * (total_predictions - VALUES(total_predictions))) + (VALUES(avg_score) * VALUES(total_predictions))) / total_predictions,
                last_updated = NOW();
            """,
            data_to_insert,
        )

        conn.commit()
        logging.info("Prediction results data inserted or updated in database")
        return True

    except Exception as e:
        logging.error("Failed to insert match in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()


def update_miner_reg_statuses(active_uids, active_hotkeys):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"

        # mark all as unregistered first as we'll update only the active ones next
        c.execute(
            f"""
            UPDATE {prediction_scores_table_name}
            SET miner_is_registered = 0
            """,
        )
        conn.commit()

        # loop through zipped uids and hotkeys and update the miner_is_registered status
        for uid, hotkey in zip(active_uids, active_hotkeys):
            c.execute(
                f"""
                UPDATE {prediction_scores_table_name}
                SET miner_is_registered = 1
                WHERE miner_uid = %s AND miner_hotkey = %s
                """,
                (uid, hotkey),
            )
            conn.commit()
        
        logging.info("Miner registration statuses updated in database")
        return True

    except Exception as e:
        logging.error("Failed to update miner registration statuses in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()


def update_miner_coldkeys_and_ages(data_to_update):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"

        c.executemany(
            f"""
            UPDATE {prediction_scores_table_name}
            SET miner_coldkey = %s, miner_age = %s
            WHERE miner_hotkey = %s
            """,
            [(coldkey, age, hotkey) for coldkey, age, hotkey in data_to_update],
        )
        conn.commit()
        logging.info("Miner coldkeys and ages updated in database")

    except Exception as e:
        logging.error("Failed to update miner coldkeys and ages in MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()


def get_prediction_stats_by_league(league, miner_hotkey=None, group_by_miner=False):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"

        query = f"""
            SELECT
                league,
                AVG(avg_score) AS avg_score,
                SUM(total_predictions) AS total_predictions,
                SUM(winner_predictions) AS winner_predictions
        """

        if group_by_miner:
            query += ", miner_hotkey, miner_coldkey, miner_uid, miner_age"

        query += f"""
            FROM {prediction_scores_table_name}
            WHERE league = %s
        """

        params = [league]

        if miner_hotkey:
            query += " AND miner_hotkey = %s"
            params.append(miner_hotkey)
        else:
            query += " AND miner_is_registered = 1"

        if group_by_miner:
            query += " GROUP BY league, miner_hotkey, miner_coldkey, miner_uid, miner_age"
        else:
            query += " GROUP BY league"

        c.execute(query, params)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query league prediction stats from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
        conn.close()


def get_prediction_stats_by_sport(sport, miner_hotkey=None, group_by_miner=False):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"

        query = f"""
            SELECT
                sport,
                AVG(avg_score) AS avg_score,
                SUM(total_predictions) AS total_predictions,
                SUM(winner_predictions) AS winner_predictions
        """

        if group_by_miner:
            query += ", miner_hotkey, miner_coldkey, miner_uid, miner_age"

        query += f"""
            FROM {prediction_scores_table_name}
            WHERE sport = %s
        """

        params = [sport]

        if miner_hotkey:
            query += " AND miner_hotkey = %s"
            params.append(miner_hotkey)
        else:
            query += " AND miner_is_registered = 1"

        if group_by_miner:
            query += " GROUP BY sport, miner_hotkey, miner_coldkey, miner_uid, miner_age"
        else:
            query += " GROUP BY sport"

        c.execute(query, params)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query sport prediction stats from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
        conn.close()


def get_prediction_stats_total(miner_hotkey=None, group_by_miner=False):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"

        query = f"""
            SELECT
                AVG(avg_score) AS avg_score,
                SUM(total_predictions) AS total_predictions,
                SUM(winner_predictions) AS winner_predictions
        """

        if group_by_miner:
            query += ", miner_hotkey, miner_coldkey, miner_age"

        query += f"""
            FROM {prediction_scores_table_name}
            WHERE 1=1
        """

        params = []

        if miner_hotkey:
            query += " AND miner_hotkey = %s"
            params.append(miner_hotkey)
        else:
            query += " AND miner_is_registered = 1"

        if group_by_miner:
            query += " GROUP BY miner_hotkey, miner_coldkey, miner_uid, miner_age"

        c.execute(query, params)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query total prediction stats from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
        conn.close()


def get_prediction_stat_snapshots(sport=None, league=None, miner_hotkey=None):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)

        query = f"""
            SELECT *
            FROM MPRSnapshots
            WHERE 1=1
        """

        params = []
        if sport:
            query += " AND sport = %s"
            params.append(sport)

        if league:
            query += " AND league = %s"
            params.append(league)

        if miner_hotkey:
            query += " AND miner_hotkey = %s"
            params.append(miner_hotkey)
        else:
            query += " AND miner_is_registered = 1"

        query += " ORDER BY snapshot_date ASC"
        
        c.execute(query, params)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query match prediction snapshots from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
        conn.close()


def get_prediction_stat_snapshots(sport=None, league=None, miner_hotkey=None):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)

        query = f"""
            SELECT *
            FROM MPRSnapshots
            WHERE 1=1
        """

        params = []
        if sport:
            query += " AND sport = %s"
            params.append(sport)

        if league:
            query += " AND league = %s"
            params.append(league)

        if miner_hotkey:
            query += " AND miner_hotkey = %s"
            params.append(miner_hotkey)
        else:
            query += " AND miner_is_registered = 1"

        query += " ORDER BY snapshot_date ASC"
        
        c.execute(query, params)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query match prediction snapshots from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
        conn.close()


def upsert_app_match_prediction(prediction, vali_hotkey):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        current_utc_time = dt.datetime.now(timezone.utc)
        current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")

        c.execute(
            """
            INSERT INTO AppMatchPredictions (app_request_id, matchId, matchDate, sport, league, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, isComplete, lastUpdated, miner_hotkey, vali_hotkey) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                matchId=VALUES(matchId), 
                matchDate=VALUES(matchDate), 
                sport=VALUES(sport),
                league=VALUES(league),
                homeTeamName=VALUES(homeTeamName), 
                awayTeamName=VALUES(awayTeamName),
                homeTeamScore=VALUES(homeTeamScore), 
                awayTeamScore=VALUES(awayTeamScore),
                isComplete=VALUES(isComplete), 
                lastUpdated=VALUES(lastUpdated),
                miner_hotkey=VALUES(miner_hotkey),
                vali_hotkey=VALUES(vali_hotkey)
            """,
            (
                prediction["app_request_id"],
                prediction["matchId"],
                prediction["matchDate"],
                prediction["sport"],
                prediction["league"],
                prediction["homeTeamName"],
                prediction["awayTeamName"],
                prediction.get("homeTeamScore"),  # These can be None, hence using get
                prediction.get("awayTeamScore"),
                prediction.get("isComplete", 0),  # Default to 0 if not provided
                current_utc_time,
                prediction.get("miner_hotkey"),  # This can be None
                vali_hotkey,  # This can be None
            ),
        )

        conn.commit()
        logging.info("Data inserted or updated in database")
        return True

    except Exception as e:
        logging.error("Failed to insert app match prediction in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()

def update_app_match_predictions(predictions):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        current_utc_time = dt.datetime.now(timezone.utc)
        current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")

        predictions_to_update = []
        predictions_with_issues = []
        for prediction in predictions:
            if "minerHasIssue" in prediction and (prediction["minerHasIssue"] == 1 or prediction["minerHasIssue"] == True):
                predictions_with_issues.append(
                    (
                        current_utc_time,
                        1,
                        prediction["minerIssueMessage"],
                        prediction["app_request_id"],
                    )
                )
                continue

            if "homeTeamScore" not in prediction or "awayTeamScore" not in prediction:
                logging.error("Missing homeTeamScore or awayTeamScore for prediction in update_app_match_predictions")
                continue
            
            predictions_to_update.append(
                (
                    prediction["homeTeamScore"],
                    prediction["awayTeamScore"],
                    1,
                    current_utc_time,
                    prediction["app_request_id"],
                )
            )

        if predictions_to_update:
            c.executemany(
                """
                UPDATE AppMatchPredictions
                SET homeTeamScore = %s, awayTeamScore = %s, isComplete = %s, lastUpdated = %s
                WHERE app_request_id = %s
                """,
                predictions_to_update
            )
        if predictions_with_issues:
            c.executemany(
                """
                UPDATE AppMatchPredictions
                SET valiLastUpdated = %s, minerHasIssue = %s, minerIssueMessage = %s
                WHERE app_request_id = %s
                """,
                predictions_with_issues
            )

        conn.commit()
        logging.info("Data inserted or updated in database")
        return True

    except Exception as e:
        logging.error("Failed to insert match in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()


def get_app_match_predictions(vali_hotkey=None, batch_size=-1):
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)

        query = "SELECT * FROM AppMatchPredictions WHERE isComplete = 0"
        
        params = []
        if vali_hotkey is not None:
            query += " AND vali_hotkey = %s"
            params.append(vali_hotkey)
        if batch_size > 0:
            query += " LIMIT %s"
            params.append(batch_size)
            
        cursor.execute(query, params)
        match_list = cursor.fetchall()

        # if results, we need to update the valiLastUpdated field
        if match_list and vali_hotkey is not None:
            cursor.execute(
                "UPDATE AppMatchPredictions SET valiLastUpdated = NOW() WHERE app_request_id IN (%s)"
                % ",".join(["%s"] * len(match_list)),
                [match["app_request_id"] for match in match_list],
            )
            conn.commit()

        return match_list

    except Exception as e:
        logging.error(
            "Failed to retrieve app match predictions from the MySQL database",
            exc_info=True,
        )
        return False
    finally:
        cursor.close()
        conn.close()


def get_app_match_predictions_unfulfilled(unfulfilled_threshold=5):
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            f"SELECT * FROM AppMatchPredictions WHERE isComplete = 0 AND valiLastUpdated IS NULL AND lastUpdated < NOW() - INTERVAL {unfulfilled_threshold} MINUTE"
        )
        match_list = cursor.fetchall()

        return match_list

    except Exception as e:
        logging.error(
            "Failed to retrieve unfulfilled app match predictions from the MySQL database",
            exc_info=True,
        )
        return False
    finally:
        cursor.close()
        conn.close()


def get_prediction_by_id(app_id):
    try:
        # Log the received app_id
        logging.info(f"Fetching prediction for app_request_id: {app_id}")

        # Ensure app_id is a string
        if not isinstance(app_id, str):
            app_id = str(app_id)

        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT m.*, apm.app_request_id, apm.isComplete AS predictionIsComplete, apm.homeTeamScore AS predictedHomeTeamScore, apm.awayTeamScore AS predictedAwayTeamScore, apm.lastUpdated AS predictionLastUpdated,
                       apm.miner_hotkey, apm.vali_hotkey, apm.valiLastUpdated, apm.minerHasIssue, apm.minerIssueMessage
            FROM AppMatchPredictions apm 
            LEFT JOIN matches m ON (m.matchId = apm.matchId) 
            WHERE app_request_id = %s
        """, (app_id,)
        )
        prediction = cursor.fetchone()

        return prediction
    except Exception as e:
        logging.error(
            "Failed to retrieve prediction from the MySQL database", exc_info=True
        )
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def create_tables():
    c = None
    conn = None
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS matches (
            matchId VARCHAR(50) PRIMARY KEY,
            matchDate TIMESTAMP NOT NULL,
            sport INTEGER NOT NULL,
            homeTeamName VARCHAR(30) NOT NULL,
            awayTeamName VARCHAR(30) NOT NULL,
            homeTeamScore INTEGER,
            awayTeamScore INTEGER,
            matchLeague VARCHAR(50),
            isComplete BOOLEAN DEFAULT FALSE,
            lastUpdated TIMESTAMP NOT NULL
        )"""
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS MatchPredictionResults (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_hotkey VARCHAR(64) NOT NULL,
            miner_coldkey VARCHAR(64) NOT NULL,
            miner_uid INTEGER NOT NULL,
            miner_is_registered TINYINT(1) DEFAULT 1,
            miner_age INTEGER NOT NULL DEFAULT 0,
            league VARCHAR(50) NOT NULL,
            sport INTEGER NOT NULL,
            total_predictions INTEGER NOT NULL,
            winner_predictions INTEGER NOT NULL,
            avg_score FLOAT NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            UNIQUE (miner_hotkey, league)
        )"""
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS MatchPredictionResults_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_hotkey VARCHAR(64) NOT NULL,
            miner_coldkey VARCHAR(64) NOT NULL,
            miner_uid INTEGER NOT NULL,
            miner_is_registered TINYINT(1) DEFAULT 1,
            miner_age INTEGER NOT NULL DEFAULT 0,
            league VARCHAR(50) NOT NULL,
            sport INTEGER NOT NULL,
            total_predictions INTEGER NOT NULL,
            winner_predictions INTEGER NOT NULL,
            avg_score FLOAT NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            UNIQUE (miner_hotkey, league)
        )"""
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS MPRSnapshots (
            snapshot_id INT AUTO_INCREMENT PRIMARY KEY,
            snapshot_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            id INT,
            miner_hotkey VARCHAR(64),
            miner_coldkey VARCHAR(64),
            miner_uid INTEGER,
            miner_is_registered TINYINT(1),
            miner_age INTEGER NOT NULL DEFAULT 0,
            league VARCHAR(50),
            sport INTEGER,
            total_predictions INTEGER,
            winner_predictions INTEGER,
            avg_score FLOAT,
            last_updated TIMESTAMP
        )"""
        )
        conn.commit()
    except Exception as e:
        logging.error("Failed to create matches table in MySQL database", exc_info=True)
    finally:
        if c is not None:
            c.close()
        if conn is not None:
            conn.close()


def create_app_tables():
    c = None
    conn = None
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS AppMatchPredictions (
            app_request_id VARCHAR(50) PRIMARY KEY,
            matchId VARCHAR(50) NOT NULL,
            matchDate TIMESTAMP NOT NULL,
            sport INTEGER NOT NULL,
            league VARCHAR(50) NOT NULL,
            homeTeamName VARCHAR(30) NOT NULL,
            awayTeamName VARCHAR(30) NOT NULL,
            homeTeamScore INTEGER,
            awayTeamScore INTEGER,
            isComplete BOOLEAN DEFAULT FALSE,
            lastUpdated TIMESTAMP NOT NULL,
            miner_hotkey VARCHAR(64) NULL,
            vali_hotkey VARCHAR(64) NULL,
            valiLastUpdated TIMESTAMP NULL,
            minerHasIssue BOOLEAN DEFAULT FALSE,
            minerIssueMessage VARCHAR(255) NULL
        )"""
        )
        conn.commit()
    except Exception as e:
        logging.error("Failed to create matches table in MySQL database", exc_info=True)
    finally:
        if c is not None:
            c.close()
        if conn is not None:
            conn.close()


def get_db_conn():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        return conn
    except mysql.connector.Error as e:
        logging.error(f"Failed to connect to MySQL database: {e}")
        return None  # Explicitly return None if there is an error


create_tables()
create_app_tables()
