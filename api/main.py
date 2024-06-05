from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import api.db as db

app = FastAPI()

@app.get("/matches")
def read_matches():
    conn = db.get_db_conn()
    if conn is not None:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM matches")
        match_list = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"matches": match_list}
    else:
        return {"error": "Failed to connect to database"}

def serialize_datetime(value):
    """Serialize datetime to JSON-compatible format, if necessary."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value

@app.get('/get-match')
async def get_match(id: str):
    conn = db.get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")

    try:
        cur = conn.cursor(dictionary=True)
        cur.execute('SELECT * FROM matches WHERE matchId = %s', (id,))
        match = cur.fetchone()
    except Error as e:
        conn.close()
        print(f"Error executing the query: {e}")
        raise HTTPException(status_code=500, detail="Error executing the query")
    finally:
        cur.close()
        conn.close()

    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    # Apply datetime serialization to all fields in the dictionary that need it
    if match:
        match = {key: serialize_datetime(value) for key, value in match.items()}
    
    return JSONResponse(content=match)


@app.post("/AddAppPrediction")
async def upsert_prediction(prediction: dict = Body(...)):
    conn = db.get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")

    query = """
    INSERT INTO AppMatchPredictions (
        app_request_id, matchId, matchDate, sport, homeTeamName, awayTeamName,
        homeTeamScore, awayTeamScore, isComplete, lastUpdated, miner_hotkey
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    matchId=VALUES(matchId), matchDate=VALUES(matchDate), sport=VALUES(sport),
    homeTeamName=VALUES(homeTeamName), awayTeamName=VALUES(awayTeamName),
    homeTeamScore=VALUES(homeTeamScore), awayTeamScore=VALUES(awayTeamScore),
    isComplete=VALUES(isComplete), lastUpdated=VALUES(lastUpdated),
    miner_hotkey=VALUES(miner_hotkey)
    """
    values = (
        prediction['app_request_id'],
        prediction['matchId'],
        prediction['matchDate'],
        prediction['sport'],
        prediction['homeTeamName'],
        prediction['awayTeamName'],
        prediction.get('homeTeamScore'),  # These can be None, hence using get
        prediction.get('awayTeamScore'),
        prediction.get('isComplete', 0),  # Default to 0 if not provided
        datetime.now(),
        prediction.get('miner_hotkey')  # This can be None
    )

    try:
        cur = conn.cursor()
        cur.execute(query, values)
        conn.commit()
    except Error as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Error executing upsert: {e}")
    finally:
        cur.close()
        conn.close()

    return {"message": "Prediction upserted successfully"}

@app.get("/AppMatchPredictions")
def read_matches():
    conn = db.get_db_conn()()
    if conn is not None:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM AppMatchPredictions")
        match_list = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"matches": match_list}
    else:
        return {"error": "Failed to connect to database"}

def serialize_datetime(value):
    """Serialize datetime to JSON-compatible format, if necessary."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value