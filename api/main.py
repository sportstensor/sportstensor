from fastapi import FastAPI
import mysql.connector
from mysql.connector import Error

app = FastAPI()

def get_database_connection():
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='sports_events',
                                             user='root',
                                             password='Cunnaredu1996@')
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

@app.get("/matches")
def read_matches():
    conn = get_database_connection()
    if conn is not None:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM matches")
        match_list = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"matches": match_list}
    else:
        return {"error": "Failed to connect to database"}

