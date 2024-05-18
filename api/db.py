import mysql.connector
import logging


def get_matches():
    try:
        conn = get_db_conn()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM matches")
        match_list = cursor.fetchall()
        
        return match_list
    
    except Exception as e:
        logging.error("Failed to retrieve matches from the MySQL database", exc_info=True)
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

def create_database():
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
            lastUpdated TIMESTAMP NOT NULL,
            sportstensorId VARCHAR(50)
        )''')
        conn.commit()
    except Exception as e:
        logging.error("Failed to create matches table in MySQL database", exc_info=True)
    finally:
        c.close()
        conn.close()

def get_db_conn():
    try:
        conn = mysql.connector.connect(
            host='localhost', 
            database='sports_events', 
            user='root', 
            password='Cunnaredu1996@'
        )
    except Exception as e:
        logging.error("Failed to connect to MySQL database", exc_info=True)

create_database()