import mysql.connector
import logging
import datetime as dt
from datetime import timezone
from api.config import IS_PROD, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
import os
import time

def generate_uuid():
    return os.urandom(16).hex()

GET_MATCH_QUERY = """
    SELECT
        mlo.*,
        mo.homeTeamOdds,
        mo.awayTeamOdds,
        mo.drawOdds,
        mo.lastUpdated,
        (SELECT COUNT(*) FROM match_odds mo WHERE mlo.oddsapiMatchId = mo.oddsapiMatchId AND mo.lastUpdated IS NOT NULL) AS odds_count,
        CASE 
            WHEN EXISTS (
                SELECT 1 
                FROM match_odds mo
                WHERE mo.oddsapiMatchId = mlo.oddsapiMatchId 
                AND mo.lastUpdated < DATE_SUB(mlo.matchDate, INTERVAL 24 HOUR)
            ) THEN TRUE 
            ELSE FALSE 
        END AS t_24h,
        CASE 
            WHEN EXISTS (
                SELECT 1 
                FROM match_odds mo
                WHERE mo.oddsapiMatchId = mlo.oddsapiMatchId 
                AND mo.lastUpdated >= DATE_SUB(mlo.matchDate, INTERVAL 24 HOUR)
                AND mo.lastUpdated < DATE_SUB(mlo.matchDate, INTERVAL 12 HOUR)
            ) THEN TRUE 
            ELSE FALSE 
        END AS t_12h,
        CASE 
            WHEN EXISTS (
                SELECT 1 
                FROM match_odds mo
                WHERE mo.oddsapiMatchId = mlo.oddsapiMatchId 
                AND mo.lastUpdated >= DATE_SUB(mlo.matchDate, INTERVAL 12 HOUR)
                AND mo.lastUpdated < DATE_SUB(mlo.matchDate, INTERVAL 4 HOUR)
            ) THEN TRUE 
            ELSE FALSE 
        END AS t_4h,
        CASE 
            WHEN EXISTS (
                SELECT 1 
                FROM match_odds mo
                WHERE mo.oddsapiMatchId = mlo.oddsapiMatchId 
                AND mo.lastUpdated >= DATE_SUB(mlo.matchDate, INTERVAL 4 HOUR)
                AND mo.lastUpdated < DATE_SUB(mlo.matchDate, INTERVAL 10 MINUTE)
            ) THEN TRUE 
            ELSE FALSE 
        END AS t_10m
    FROM (
        SELECT
            m.matchId,
            m.matchDate,
            m.homeTeamName,
            m.awayTeamName,
            m.sport,
            CASE
                WHEN m.isComplete = 1 THEN COALESCE(m.homeTeamScore, 0)
                ELSE m.homeTeamScore 
            END AS homeTeamScore,
            CASE 
                WHEN m.isComplete = 1 THEN COALESCE(m.awayTeamScore, 0)
                ELSE m.awayTeamScore 
            END AS awayTeamScore,
            m.matchLeague,
            m.isComplete,
            ml.oddsapiMatchId
        FROM matches m
        LEFT JOIN matches_lookup ml ON m.matchId = ml.matchId
    ) mlo
    LEFT JOIN match_odds mo ON mlo.oddsapiMatchId = mo.oddsapiMatchId AND mo.lastUpdated = (
        SELECT lastUpdated
        FROM match_odds AS mo_inner
        WHERE mo_inner.oddsapiMatchId = mlo.oddsapiMatchId AND mo_inner.lastUpdated < (mlo.matchDate + INTERVAL 5 MINUTE)
        ORDER BY lastUpdated DESC
        LIMIT 1
    ) AND mo.id = (
        SELECT id
        FROM match_odds AS mo_inner
        WHERE mo_inner.oddsapiMatchId = mlo.oddsapiMatchId AND mo_inner.lastUpdated < (mlo.matchDate + INTERVAL 5 MINUTE)
        ORDER BY lastUpdated DESC
        LIMIT 1
    )
"""

def match_id_exists(match_id):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM matches WHERE matchId = %s", (match_id,))
        count = cursor.fetchone()[0]

        return count > 0

    except Exception as e:
        logging.error("Failed to check if match exists in MySQL database", exc_info=True)
        return False
    finally:
        cursor.close()
        conn.close()

def match_odds_id_exists(match_odds_id):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM match_odds WHERE id = %s", (match_odds_id,))
        count = cursor.fetchone()[0]

        return count > 0

    except Exception as e:
        logging.error("Failed to check if match_odds exists in MySQL database", exc_info=True)
        return False
    finally:
        cursor.close()
        conn.close()

def get_matches(all=False):
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SET @current_time_utc = CONVERT_TZ(NOW(), @@session.time_zone, '+00:00')")
        query = GET_MATCH_QUERY

        if not all:
            query += """
                WHERE mlo.matchDate BETWEEN @current_time_utc - INTERVAL 10 DAY AND @current_time_utc + INTERVAL 48 HOUR
            """
        
        cursor.execute(query)
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

def get_upcoming_matches():
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        # Set the current time in UTC
        cursor.execute("SET @current_time_utc = CONVERT_TZ(NOW(), @@session.time_zone, '+00:00')")

        query = GET_MATCH_QUERY + """
           WHERE mlo.matchDate BETWEEN @current_time_utc AND @current_time_utc + INTERVAL 48 HOUR
        """

        cursor.execute(query)
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

def get_matches_with_no_odds():
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT ml.*, m.*
            FROM matches_lookup ml
            LEFT JOIN matches m
            ON ml.matchId = m.matchId
        """

        cursor.execute(query)
        matches = cursor.fetchall()

        return matches

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

def get_stored_odds(lastUpdated = None):
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        if lastUpdated:
            query = """
                SELECT mo.*, o.homeTeamName, o.awayTeamName, o.commence_time, o.league
                FROM (
                    SELECT id, oddsapiMatchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated
                    FROM match_odds
                    WHERE (oddsapiMatchId, lastUpdated) IN (
                        SELECT oddsapiMatchId, MAX(lastUpdated)
                        FROM match_odds
                        WHERE lastUpdated <= %s
                        GROUP BY oddsapiMatchId 
                    )
                ) mo
                INNER JOIN odds o
                ON mo.oddsapiMatchId = o.oddsapiMatchId
            """
            cursor.execute(query, (lastUpdated,))
            stored_odds = cursor.fetchall()
        else:
            query = """
                SELECT mo.*, o.homeTeamName, o.awayTeamName, o.commence_time, o.league
                FROM (
                    SELECT id, oddsapiMatchId, homeTeamOdds, awayTeamOdds, drawOdds, lastUpdated
                    FROM match_odds
                    WHERE (oddsapiMatchId, lastUpdated) IN (
                        SELECT oddsapiMatchId, MAX(lastUpdated)
                        FROM match_odds
                        GROUP BY oddsapiMatchId 
                    )
                ) mo
                INNER JOIN odds o
                ON mo.oddsapiMatchId = o.oddsapiMatchId
            """
            cursor.execute(query)
            stored_odds = cursor.fetchall()

        return stored_odds

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

def get_match_odds_by_id(match_id):
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        if match_id:
            query = """
                SELECT mo.*, ml.matchId
                FROM match_odds mo
                LEFT JOIN matches_lookup ml ON mo.oddsapiMatchId = ml.oddsapiMatchId
                WHERE ml.matchId = %s AND mo.lastUpdated IS NOT NULL
                ORDER BY mo.lastUpdated ASC
            """
            cursor.execute(query, (match_id,))
            match_odds = cursor.fetchall()
        else:
            query = """
                SELECT
                    mlo.*,
                    (
                        SELECT
                            JSON_ARRAYAGG(
                                JSON_OBJECT(
                                    "oddsapiMatchId", mo.oddsapiMatchId,
                                    "homeTeamOdds", mo.homeTeamOdds,
                                    "awayTeamOdds", mo.awayTeamOdds,
                                    "drawOdds", mo.drawOdds,
                                    "lastUpdated", mo.lastUpdated
                                )
                            )
                        FROM match_odds mo
                        WHERE mlo.oddsapiMatchId = mo.oddsapiMatchId
                    ) AS oddsData
                FROM (
                    SELECT
                        m.matchId,
                        m.matchDate,
                        m.homeTeamName,
                        m.awayTeamName,
                        m.sport,
                        CASE 
                            WHEN m.isComplete = 1 THEN COALESCE(m.homeTeamScore, 0)
                            ELSE m.homeTeamScore 
                        END AS homeTeamScore,
                        CASE 
                            WHEN m.isComplete = 1 THEN COALESCE(m.awayTeamScore, 0)
                            ELSE m.awayTeamScore 
                        END AS awayTeamScore,
                        m.matchLeague,
                        m.isComplete,
                        ml.oddsapiMatchId
                    FROM matches m
                    LEFT JOIN matches_lookup ml ON m.matchId = ml.matchId
                ) mlo
            """
            cursor.execute(query)
            match_odds = cursor.fetchall()
        
        return match_odds

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

def get_matches_with_missing_odds():
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        # Set the current time in UTC
        cursor.execute("SET @current_time_utc = CONVERT_TZ(NOW(), @@session.time_zone, '+00:00')")

        query = """
            SELECT
                mlo.*,
                (
                    SELECT
                        JSON_ARRAYAGG(
                            JSON_OBJECT(
                                "oddsapiMatchId", mo.oddsapiMatchId,
                                "homeTeamOdds", mo.homeTeamOdds,
                                "awayTeamOdds", mo.awayTeamOdds,
                                "drawOdds", mo.drawOdds,
                                "lastUpdated", mo.lastUpdated
                            )
                        )
                    FROM match_odds mo
                    WHERE mlo.oddsapiMatchId = mo.oddsapiMatchId
                    ORDER BY mo.lastUpdated ASC
                ) AS oddsData
            FROM (
                SELECT
                    m.matchId,
                    m.matchDate,
                    m.homeTeamName,
                    m.awayTeamName,
                    m.sport,
                    CASE 
                        WHEN m.isComplete = 1 THEN COALESCE(m.homeTeamScore, 0)
                        ELSE m.homeTeamScore 
                    END AS homeTeamScore,
                    CASE 
                        WHEN m.isComplete = 1 THEN COALESCE(m.awayTeamScore, 0)
                        ELSE m.awayTeamScore 
                    END AS awayTeamScore,
                    m.matchLeague,
                    m.isComplete,
                    ml.oddsapiMatchId
                FROM matches m
                LEFT JOIN matches_lookup ml ON m.matchId = ml.matchId
                WHERE m.matchDate BETWEEN @current_time_utc - INTERVAL 10 DAY AND @current_time_utc AND m.isComplete = 1 
            ) mlo
        """

        cursor.execute(query)
        matches = cursor.fetchall()

        return matches

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

        query = GET_MATCH_QUERY + """
            WHERE mlo.matchId = %s
        """
        cursor.execute(query, (match_id,))
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
                homeTeamScore=CASE 
                    WHEN VALUES(isComplete) = 1 THEN COALESCE(VALUES(homeTeamScore), 0)
                    ELSE VALUES(homeTeamScore)
                END,
                awayTeamScore=CASE 
                    WHEN VALUES(isComplete) = 1 THEN COALESCE(VALUES(awayTeamScore), 0)
                    ELSE VALUES(awayTeamScore)
                END,
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
        return True

    except Exception as e:
        logging.error("Failed to insert match in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()

def insert_sportsdb_match_lookup(match_id, sportsdb_match_id):
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute(
            """
            INSERT IGNORE INTO matches_lookup (matchId, sportsdbMatchId) 
            VALUES (%s, %s)
            """,
            (
                match_id,
                sportsdb_match_id
            ),
        )

        conn.commit()
        logging.info("Match lookup inserted in database")
        return True

    except Exception as e:
        logging.error("Failed to insert match lookup in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()

def query_sportsdb_match_lookup(sportsdb_match_id):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT matchId FROM matches_lookup WHERE sportsdbMatchId = %s",
            (sportsdb_match_id,),
        )
        match_id = cursor.fetchone()

        return match_id[0] if match_id else None

    except Exception as e:
        logging.error("Failed to query sportsdb match lookup in MySQL database", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

def setup_match_odds_table():
    try:
        conn = get_db_conn()
        c = conn.cursor()
        
        # Check if the pinnacle_bookmaker column exists
        c.execute("SHOW COLUMNS FROM match_odds LIKE 'pinnacle_bookmaker';")
        result = c.fetchone()

        # If the column does not exist, add it
        if not result:
            c.execute(
                """
                ALTER TABLE match_odds
                ADD COLUMN pinnacle_bookmaker TINYINT(1) DEFAULT 1;
                """
            )
            conn.commit()
            logging.info("Added pinnacle_bookmaker column to match_odds table.")

        # Update existing records to set a default value for pinnacle_bookmaker
        c.execute(
            """
            UPDATE match_odds
            SET pinnacle_bookmaker = 1
            WHERE pinnacle_bookmaker IS NULL;
            """
        )
        conn.commit()
        logging.info("Updated existing records in match_odds table.")

    except Exception as e:
        logging.error("Failed to setup match_odds table", exc_info=True)
    finally:
        c.close()
        conn.close()

def insert_match_odds_bulk(match_data):
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.executemany(
            """
            INSERT IGNORE INTO match_odds (id, oddsapiMatchId, homeTeamOdds, awayTeamOdds, drawOdds, pinnacle_bookmaker, lastUpdated) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            match_data,
        )

        conn.commit()
        return True

    except Exception as e:
        logging.error("Failed to insert match odds in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()

def get_average_pinnacle():
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute(
            """
                SELECT AVG(vig) AS average_vig
                FROM (
                    SELECT 
                        (1 / homeTeamOdds + 1 / awayTeamOdds + 
                        (CASE WHEN drawOdds IS NOT NULL THEN 1 / drawOdds ELSE 0 END) - 1) AS vig
                    FROM match_odds
                    WHERE pinnacle_bookmaker = 1
                ) AS vig_table
            """
        )

        average_vig = c.fetchone()[0]  # Fetch the result
        conn.commit()
        return average_vig  # Return the average vig

    except Exception as e:
        logging.error("Failed to get average pinnacle vig in match odds table", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()

def insert_match_lookups_bulk(match_lookup_data):
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.executemany(
            """
            INSERT IGNORE INTO matches_lookup (matchId, oddsapiMatchId) 
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE
                oddsapiMatchId=VALUES(oddsapiMatchId)
            """,
            match_lookup_data,
        )

        conn.commit()
        return True

    except Exception as e:
        logging.error("Failed to insert match lookup in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()

def insert_odds_bulk(odds_to_store):
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.executemany(
            """
            INSERT IGNORE INTO odds (oddsapiMatchId, league, homeTeamName, awayTeamName, commence_time, lastUpdated)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                commence_time=VALUES(commence_time),
                lastUpdated=VALUES(lastUpdated);
            """,
            odds_to_store,
        )

        conn.commit()
        return True

    except Exception as e:
        logging.error("Failed to insert odds in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()

def upload_prediction_edge_results(prediction_results):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        """
        {
            'miner_scores': [
                {
                    "uid": 123,
                    "hotkey": 'abcdefg',
                    "vali_hotkey": 'hijklmnop',
                    "total_score": 5.5,
                    "total_pred_count": 100,
                    "mlb_score": 1.1,
                    "mlb_pred_count": 35,
                    "nfl_score": 3.12,
                    "nfl_pred_count": 20,
                    "nba_score": 0.5,
                    "nba_pred_count": 45,
                    "mls_score": 0,
                    "mls_pred_count": 0,
                    "epl_score": 0,
                    "epl_pred_count": 0,
                    "lastUpdated": '2024-10-28 00:00:00'
                }
            ],
        }
        """

        # Convert the miner_scores dictionary into a list of tuples
        values_list = []
        for uid, score_data in prediction_results["miner_scores"].items():
            values_tuple = (
                score_data["uid"],
                score_data["hotkey"],
                score_data["vali_hotkey"],
                score_data["total_score"],
                score_data["total_pred_count"],
                score_data["mlb_score"],
                score_data["mlb_pred_count"],
                score_data["nfl_score"],
                score_data["nfl_pred_count"],
                score_data["nba_score"],
                score_data["nba_pred_count"],
                score_data["mls_score"],
                score_data["mls_pred_count"],
                score_data["epl_score"],
                score_data["epl_pred_count"]
            )
            values_list.append(values_tuple)

        prediction_edge_scores_table_name = "MatchPredictionEdgeResults"
        if not IS_PROD:
            prediction_edge_scores_table_name += "_test"
        c.executemany(
            f"""
            INSERT INTO {prediction_edge_scores_table_name} (
                miner_uid,
                miner_hotkey,
                vali_hotkey,
                total_score,
                total_pred_count,
                mlb_score,
                mlb_pred_count,
                nfl_score,
                nfl_pred_count,
                nba_score,
                nba_pred_count,
                mls_score,
                mls_pred_count,
                epl_score,
                epl_pred_count,
                lastUpdated
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
            """,
            values_list,
        )

        conn.commit()
        logging.info("Prediction edge results data inserted or updated in database")
        return True

    except Exception as e:
        logging.error("Failed to insert prediction edge results data in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()


def get_prediction_edge_results(vali_hotkey, miner_hotkey=None, miner_id=None, league=None, date=None, count=10):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)

        prediction_edge_results_table_name = "MatchPredictionEdgeResults"
        params = [vali_hotkey]
        if not IS_PROD:
            prediction_edge_results_table_name += "_test"

        query = f"""
            SELECT *
            FROM {prediction_edge_results_table_name}
            WHERE vali_hotkey = %s
        """

        if miner_hotkey:
            query += " AND miner_hotkey = %s"
            params.append(miner_hotkey)
        
        if miner_id:
            query += " AND miner_uid = %s"
            params.append(miner_id)

        if league:
            query += f" AND {league.lower()}_pred_count > 0 AND {league.lower()}_score > 0"
        
        if date:
            query += " AND DATE(lastUpdated) >= %s"
            date = dt.datetime.strptime(date, "%Y-%m-%d")
            params.append(date)
            if not count:
                count = 1

        query += """
            ORDER BY lastUpdated DESC
        """

        if league:
            query += f"""
                , {league.lower()}_score DESC
            """

        if count:
            query += " LIMIT %s"
            params.append(count)

        if params:
            c.execute(query, params)
        else:
            c.execute(query)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query league prediction stats from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
        conn.close()


def upload_scored_predictions(predictions, vali_hotkey):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        current_utc_time = dt.datetime.now(timezone.utc)
        current_utc_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Content of predictions: {predictions}")

        predictions = predictions.get('predictions', [])

        # Ensure predictions is a list
        if not isinstance(predictions, list):
            raise ValueError("predictions must be a list of dictionaries")

        # Prepare the data for executemany
        data_to_insert = [
            (
                int(prediction.get("minerId")),
                prediction.get("hotkey"),
                vali_hotkey,
                prediction.get("predictionDate"),
                prediction.get("matchId"),
                prediction.get("matchDate"),
                int(prediction.get("sport")),
                prediction.get("league"),
                1,
                current_utc_time,
                prediction.get("homeTeamName"),
                prediction.get("awayTeamName"),
                prediction.get("homeTeamScore"),
                prediction.get("awayTeamScore"),
                prediction.get("probabilityChoice"),
                float(prediction.get("probability")),
                float(prediction.get("closingEdge")) if prediction.get("closingEdge") is not None else None,
            )
            for prediction in predictions
        ]

        prediction_scores_table_name = "MatchPredictionsScored"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
        c.executemany(
            f"""
            INSERT IGNORE INTO {prediction_scores_table_name} (
                miner_id,
                miner_hotkey,
                vali_hotkey,
                predictionDate,
                matchId,
                matchDate,
                sport,
                league,
                isScored,
                scoredDate,
                homeTeamName,
                awayTeamName,
                homeTeamScore,
                awayTeamScore,
                probabilityChoice,
                probability,
                closingEdge
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            );
            """,
            data_to_insert,
        )

        conn.commit()
        logging.info("Scored predictions inserted in database")
        return True

    except Exception as e:
        logging.error("Failed to insert scored predictions in MySQL database", exc_info=True)
        return False
    finally:
        c.close()
        conn.close()


def update_miner_reg_statuses(active_uids, active_hotkeys):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        miners_table_name = "Miners"
        if not IS_PROD:
            miners_table_name += "_test"

        # mark all as unregistered first as we'll update only the active ones next
        c.execute(
            f"""
            UPDATE {miners_table_name}
            SET miner_is_registered = 0
            """,
        )
        conn.commit()

        # loop through zipped uids and hotkeys and update the miner_is_registered status
        for uid, hotkey in zip(active_uids, active_hotkeys):
            c.execute(
                f"""
                UPDATE {miners_table_name}
                SET miner_is_registered = 1, miner_uid = %s
                WHERE miner_hotkey = %s
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


def insert_or_update_miner_coldkeys_and_ages(data_to_update):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        miners_table_name = "Miners"
        if not IS_PROD:
            miners_table_name += "_test"

        # mark all as unregistered first as we'll update only the active ones next
        cursor.execute(
            f"""
            UPDATE {miners_table_name}
            SET miner_is_registered = 0
            """,
        )
        conn.commit()

        cursor.executemany(f"""
            INSERT INTO {miners_table_name} (miner_hotkey, miner_coldkey, miner_uid, miner_age, miner_is_registered, last_updated) 
            VALUES (%s, %s, %s, %s, 1, NOW())
            ON DUPLICATE KEY UPDATE
                miner_coldkey=VALUES(miner_coldkey), 
                miner_age=VALUES(miner_age), 
                miner_is_registered=VALUES(miner_is_registered)
        """, data_to_update)

        # Commit the changes
        conn.commit()

    except Exception as e:
        logging.error("Failed to update miner coldkeys and ages in MySQL database", exc_info=True)
    finally:
        cursor.close()
        conn.close()


def get_prediction_stats_by_league(vali_hotkey, miner_hotkey=None, cutoff = None, include_deregged = None):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)
        
        params = [vali_hotkey]
        miners_table_name = "Miners"
        prediction_edges_table_name = "MatchPredictionEdgeResults"
        if not IS_PROD:
            miners_table_name += "_test"
            prediction_edges_table_name += "_test"

        query = f"""
            SELECT
                mpe.miner_uid,
                mpe.miner_hotkey,
                mpe.total_score,
                mpe.mlb_score,
                mpe.nba_score,
                mpe.mls_score,
                mpe.epl_score,
                mpe.nfl_score
            FROM {prediction_edges_table_name} mpe
            LEFT JOIN {miners_table_name} m ON mpe.miner_hotkey = m.miner_hotkey AND mpe.miner_uid = m.miner_uid
            WHERE (mpe.miner_hotkey, mpe.vali_hotkey, mpe.lastUpdated) IN (
                SELECT miner_hotkey, vali_hotkey, MAX(lastUpdated)
                FROM {prediction_edges_table_name}
                GROUP BY miner_hotkey, vali_hotkey
            )
            AND mpe.vali_hotkey = %s
        """

        if cutoff:
            # Calculate the current timestamp
            current_timestamp = int(time.time())
            # Calculate cutoff date timestamp
            match_cutoff_timestamp = current_timestamp - (
                cutoff * 24 * 3600
            )
            # Convert timestamps to strings in 'YYYY-MM-DD HH:MM:SS' format
            match_cutoff_str = dt.datetime.utcfromtimestamp(
                match_cutoff_timestamp
            ).strftime("%Y-%m-%d %H:%M:%S")
            query += " AND mpe.lastUpdated > %s"
            params.append(match_cutoff_str)

        if miner_hotkey:
            query += " AND mpe.miner_hotkey = %s"
            params.append(miner_hotkey)
        if include_deregged is None or include_deregged == 0:
            query += " AND m.miner_is_registered = 1"
        
        query += """ GROUP BY 
        mpe.miner_uid, 
        mpe.miner_hotkey,
        mpe.total_score,
        mpe.nfl_score,
        mpe.nba_score,
        mpe.mls_score,
        mpe.epl_score,
        mpe.mlb_score;"""

        #logging.info(f"query============>{query}")

        if params:
            c.execute(query, params)
        else:
            c.execute(query)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query league prediction stats from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
        conn.close()

def get_prediction_results_by_league(vali_hotkey, league=None, miner_hotkey=None):
    try:
        conn = get_db_conn()
        c = conn.cursor(dictionary=True)

        prediction_scores_table_name = "MatchPredictionsScored"
        params = [vali_hotkey]
        miners_table_name = "Miners"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
            miners_table_name += "_test"

        query = f"""
            SELECT
                DATE(mps.scoredDate) AS scoreDate,
                COUNT(*) AS total_predictions
            FROM {prediction_scores_table_name} mps
            LEFT JOIN {miners_table_name} m ON mps.miner_hotkey = m.miner_hotkey
            WHERE m.miner_is_registered = 1 AND mps.vali_hotkey = %s
        """
        if league:
            query += " AND league = %s"
            params.append(league)

        if miner_hotkey:
            query += " AND mps.miner_hotkey = %s"
            params.append(miner_hotkey)
        query += """
            GROUP BY DATE(mps.scoredDate)
            ORDER BY scoreDate ASC;
        """
        if params:
            c.execute(query, params)
        else:
            c.execute(query)
        return c.fetchall()

    except Exception as e:
        logging.error(
            "Failed to query league prediction stats from MySQL database", exc_info=True
        )
        return False
    finally:
        c.close()
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
            lastUpdated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )"""
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS matches_lookup (
            matchId VARCHAR(50) PRIMARY KEY,
            sportsdbMatchId VARCHAR(50) DEFAULT NULL,
            oddsapiMatchId VARCHAR(50) DEFAULT NULL
        )"""
        )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS match_odds (
            id VARCHAR(50) PRIMARY KEY,
            oddsapiMatchId VARCHAR(50) DEFAULT NULL,
            homeTeamOdds FLOAT,
            awayTeamOdds FLOAT,
            drawOdds FLOAT,
            pinnacle_bookmaker TINYINT(1) DEFAULT 1,
            lastUpdated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )"""
        )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS odds (
            oddsapiMatchId VARCHAR(50) PRIMARY KEY,
            league VARCHAR(30) NOT NULL,
            homeTeamName VARCHAR(30) NOT NULL,
            awayTeamName VARCHAR(30) NOT NULL,
            commence_time TIMESTAMP NOT NULL,
            lastUpdated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
       )"""
       )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS MatchPredictionsScored (
            miner_id INTEGER NOT NULL,
            miner_hotkey VARCHAR(64) NOT NULL,
            vali_hotkey VARCHAR(64) NOT NULL,
            predictionDate TIMESTAMP NOT NULL,
            matchId VARCHAR(50) NOT NULL,
            matchDate TIMESTAMP NOT NULL,
            sport INTEGER NOT NULL,
            league VARCHAR(50) NOT NULL,
            homeTeamName VARCHAR(30) NOT NULL,
            awayTeamName VARCHAR(30) NOT NULL,
            homeTeamScore INTEGER,
            awayTeamScore INTEGER,
            probabilityChoice VARCHAR(10) NOT NULL,
            probability FLOAT NOT NULL,
            closingEdge FLOAT NOT NULL,
            isScored BOOLEAN DEFAULT FALSE,
            scoredDate TIMESTAMP DEFAULT NULL,
            lastUpdated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (miner_hotkey, vali_hotkey, matchId)
        )"""
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS MatchPredictionsScored_test (
            miner_id INTEGER NOT NULL,
            miner_hotkey VARCHAR(64) NOT NULL,
            vali_hotkey VARCHAR(64) NOT NULL,
            predictionDate TIMESTAMP NOT NULL,
            matchId VARCHAR(50) NOT NULL,
            matchDate TIMESTAMP NOT NULL,
            sport INTEGER NOT NULL,
            league VARCHAR(50) NOT NULL,
            homeTeamName VARCHAR(30) NOT NULL,
            awayTeamName VARCHAR(30) NOT NULL,
            homeTeamScore INTEGER,
            awayTeamScore INTEGER,
            probabilityChoice VARCHAR(10) NOT NULL,
            probability FLOAT NOT NULL,
            closingEdge FLOAT NOT NULL,
            isScored BOOLEAN DEFAULT FALSE,
            scoredDate TIMESTAMP DEFAULT NULL,
            lastUpdated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (miner_hotkey, vali_hotkey, matchId)
        )"""
        )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS MatchPredictionEdgeResults (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_uid INTEGER NOT NULL,
            miner_hotkey VARCHAR(64) NOT NULL,
            vali_hotkey VARCHAR(64) NOT NULL,
            total_score FLOAT NOT NULL,
            total_pred_count INTEGER NOT NULL,
            mlb_score FLOAT NOT NULL,
            mlb_pred_count INTEGER NOT NULL,
            nfl_score FLOAT NOT NULL,
            nfl_pred_count INTEGER NOT NULL,
            nba_score FLOAT NOT NULL,
            nba_pred_count INTEGER NOT NULL,
            mls_score FLOAT NOT NULL,
            mls_pred_count INTEGER NOT NULL,
            epl_score FLOAT NOT NULL,
            epl_pred_count INTEGER NOT NULL,
            lastUpdated TIMESTAMP NOT NULL
        )"""
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS MatchPredictionEdgeResults_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_uid INTEGER NOT NULL,
            miner_hotkey VARCHAR(64) NOT NULL,
            vali_hotkey VARCHAR(64) NOT NULL,
            total_score FLOAT NOT NULL,
            total_pred_count INTEGER NOT NULL,
            mlb_score FLOAT NOT NULL,
            mlb_pred_count INTEGER NOT NULL,
            nfl_score FLOAT NOT NULL,
            nfl_pred_count INTEGER NOT NULL,
            nba_score FLOAT NOT NULL,
            nba_pred_count INTEGER NOT NULL,
            mls_score FLOAT NOT NULL,
            mls_pred_count INTEGER NOT NULL,
            epl_score FLOAT NOT NULL,
            epl_pred_count INTEGER NOT NULL,
            lastUpdated TIMESTAMP NOT NULL
        )"""
        )
        
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS Miners (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_hotkey VARCHAR(64) NOT NULL,
            miner_coldkey VARCHAR(64) NOT NULL,
            miner_uid INTEGER NOT NULL,
            miner_is_registered TINYINT(1) DEFAULT 1,
            miner_age INTEGER NOT NULL DEFAULT 0,
            last_updated TIMESTAMP NOT NULL,
            UNIQUE (miner_hotkey, miner_uid)
        )"""
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS Miners_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_hotkey VARCHAR(64) NOT NULL,
            miner_coldkey VARCHAR(64) NOT NULL,
            miner_uid INTEGER NOT NULL,
            miner_is_registered TINYINT(1) DEFAULT 1,
            miner_age INTEGER NOT NULL DEFAULT 0,
            last_updated TIMESTAMP NOT NULL,
            UNIQUE (miner_hotkey, miner_uid)
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
