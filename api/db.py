import mysql.connector
import logging
import datetime as dt
from datetime import timezone
from api.config import IS_PROD
import os

def get_matches():
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT * FROM matches
            WHERE matchDate BETWEEN NOW() - INTERVAL 10 DAY AND NOW() + INTERVAL 48 HOUR
        """)
        match_list = cursor.fetchall()
        
        return match_list
    
    except Exception as e:
        logging.error("Failed to retrieve matches from the MySQL database", exc_info=True)
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
        
        cursor.execute('SELECT * FROM matches WHERE matchId = %s', (id,))
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
        c.execute('''
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
            ''',
            (
                match_id,
                event.get('strTimestamp'),
                sport_type,
                event.get('strHomeTeam'),
                event.get('strAwayTeam'),
                event.get('intHomeScore'),
                event.get('intAwayScore'),
                event.get('strLeague'), 
                is_complete,
                current_utc_time
            ))
        
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
        current_utc_time = current_utc_time.strftime('%Y-%m-%d %H:%M:%S')

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
        data_to_insert = list(zip(
            prediction_results['hotkeys'],
            prediction_results['uids'],
            prediction_results['leagues'],
            prediction_results['sports'],
            [1] * len(prediction_results['uids']),
            prediction_results['correct_winner_results'],
            prediction_results['scores']
        ))

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
        c.executemany(f'''
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
            ''',
            data_to_insert
        )
        
        conn.commit()
        logging.info("Prediction results data inserted or updated in database")

    except Exception as e:
        logging.error("Failed to insert match in MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()

def get_prediction_stats_by_league(league, miner_hotkey=None):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
        query = f'''
            SELECT
                league,
                AVG(avg_score) AS avg_score,
                SUM(total_predictions) AS total_predictions,
                SUM(winner_predictions) AS winner_predictions
            FROM {prediction_scores_table_name}
            WHERE league = %s
        '''
        params = [league]
        
        if miner_hotkey:
            query += ' AND miner_hotkey = %s'
            params.append(miner_hotkey)
        
        query += ' GROUP BY league'
        
        c.execute(query, params)
        return c.fetchall()
    
    except Exception as e:
        logging.error("Failed to query league prediction stats from MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()

def get_prediction_stats_by_sport(sport, miner_hotkey=None):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
        query = f'''
            SELECT
                sport,
                AVG(avg_score) AS avg_score,
                SUM(total_predictions) AS total_predictions,
                SUM(winner_predictions) AS winner_predictions
            FROM {prediction_scores_table_name}
            WHERE sport = %s
        '''
        params = [sport]
        
        if miner_hotkey:
            query += ' AND miner_hotkey = %s'
            params.append(miner_hotkey)
        
        query += ' GROUP BY sport'
        
        c.execute(query, params)
        return c.fetchall()

    except Exception as e:
        logging.error("Failed to query sport prediction stats from MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()

def get_prediction_stats_total(miner_hotkey=None):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        prediction_scores_table_name = "MatchPredictionResults"
        if not IS_PROD:
            prediction_scores_table_name += "_test"
        query = f'''
            SELECT
                AVG(avg_score) AS avg_score,
                SUM(total_predictions) AS total_predictions,
                SUM(winner_predictions) AS winner_predictions
            FROM {prediction_scores_table_name}
        '''
        params = []
        
        if miner_hotkey:
            query += ' WHERE miner_hotkey = %s'
            params.append(miner_hotkey)
        
        c.execute(query, params)
        return c.fetchall()
    
    except Exception as e:
        logging.error("Failed to query total prediction stats from MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()

def upsert_app_match_prediction(prediction):
    try:
        conn = get_db_conn()
        c = conn.cursor()

        current_utc_time = dt.datetime.now(timezone.utc)
        current_utc_time = current_utc_time.strftime('%Y-%m-%d %H:%M:%S')

        c.execute('''
            INSERT INTO AppMatchPredictions (app_request_id, matchId, matchDate, sport, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, isComplete, lastUpdated, miner_hotkey) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                matchId=VALUES(matchId), 
                matchDate=VALUES(matchDate), 
                sport=VALUES(sport),
                homeTeamName=VALUES(homeTeamName), 
                awayTeamName=VALUES(awayTeamName),
                homeTeamScore=VALUES(homeTeamScore), 
                awayTeamScore=VALUES(awayTeamScore),
                isComplete=VALUES(isComplete), 
                lastUpdated=VALUES(lastUpdated),
                miner_hotkey=VALUES(miner_hotkey)
            ''',
            (
                prediction['app_request_id'],
                prediction['matchId'],
                prediction['matchDate'],
                prediction['sport'],
                prediction['homeTeamName'],
                prediction['awayTeamName'],
                prediction.get('homeTeamScore'),  # These can be None, hence using get
                prediction.get('awayTeamScore'),
                prediction.get('isComplete', 0),  # Default to 0 if not provided
                current_utc_time,
                prediction.get('miner_hotkey')  # This can be None
            ))
        
        conn.commit()
        logging.info("Data inserted or updated in database")

    except Exception as e:
        logging.error("Failed to insert match in MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()

def get_app_match_predictions():
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM AppMatchPredictions")
        match_list = cursor.fetchall()
        
        return match_list
    
    except Exception as e:
        logging.error("Failed to retrieve app match predictions from the MySQL database", exc_info=True)
        return False
    finally:
        cursor.close()
        conn.close()

def create_tables():
    c = None
    conn = None
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute('''
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
        )''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS MatchPredictionResults (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_hotkey VARCHAR(64) NOT NULL,
            miner_uid INTEGER NOT NULL,
            league VARCHAR(50) NOT NULL,
            sport INTEGER NOT NULL,
            total_predictions INTEGER NOT NULL,
            winner_predictions INTEGER NOT NULL,
            avg_score FLOAT NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            UNIQUE (miner_hotkey, league)
        )''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS MatchPredictionResults_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            miner_hotkey VARCHAR(64) NOT NULL,
            miner_uid INTEGER NOT NULL,
            league VARCHAR(50) NOT NULL,
            sport INTEGER NOT NULL,
            total_predictions INTEGER NOT NULL,
            winner_predictions INTEGER NOT NULL,
            avg_score FLOAT NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            UNIQUE (miner_hotkey, league)
        )''')
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
        c.execute('''
        CREATE TABLE IF NOT EXISTS AppMatchPredictions (
            app_request_id VARCHAR(50) PRIMARY KEY,
            matchId VARCHAR(50) NOT NULL,
            matchDate TIMESTAMP NOT NULL,
            sport INTEGER NOT NULL,
            homeTeamName VARCHAR(30) NOT NULL,
            awayTeamName VARCHAR(30) NOT NULL,
            homeTeamScore INTEGER,
            awayTeamScore INTEGER,
            matchLeague VARCHAR(50),
            isComplete BOOLEAN DEFAULT FALSE,
            lastUpdated TIMESTAMP NOT NULL,
            miner_hotkey VARCHAR(64) NULL
        )''')
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
            host='localhost', 
            database='sportstensor', 
            user=os.getenv('DB_USER'), 
            password=os.getenv('DB_PASSWORD')

        )
        return conn
    except mysql.connector.Error as e:
        logging.error(f"Failed to connect to MySQL database: {e}")
        return None  # Explicitly return None if there is an error


create_tables()
create_app_tables()