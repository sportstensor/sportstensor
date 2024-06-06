import sqlite3
import random
from datetime import datetime, timedelta

# Function to create a random MatchPrediction
def create_random_match_prediction(matchId, matchDate, sport, homeTeamName, awayTeamName):
    random_uid = random.randint(1, 16)  # Generate a random integer between 1 and 16
    
    return {
        'minerId': random_uid,
        'hotkey': f'DEV_{str(random_uid)}',
        'matchId': matchId,
        'matchDate': matchDate,
        'sport': sport,
        'homeTeamName': homeTeamName,
        'awayTeamName': awayTeamName,
        'homeTeamScore': random.randint(0, 10),
        'awayTeamScore': random.randint(0, 10),
        'lastUpdated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# Connect to the SQLite database
conn = sqlite3.connect('SportsTensor.db')
cursor = conn.cursor()

# Calculate the date 2 days ago
two_days_ago = datetime.now() - timedelta(days=2)
two_days_ago_str = two_days_ago.strftime('%Y-%m-%d %H:%M:%S')

# Query for matches that have completed in the last 2 days
query = """
SELECT * FROM Matches
WHERE matchDate >= ?
AND isComplete = 1
LIMIT 5
"""
cursor.execute(query, (two_days_ago_str,))
matches = cursor.fetchall()

# Create random MatchPredictions for the matches
match_predictions = [create_random_match_prediction(match[0], match[1], match[2], match[3], match[4]) for match in matches]

# Insert the random MatchPredictions into the MatchPredictions table
insert_query = """
INSERT OR IGNORE INTO MatchPredictions (minerId, hotkey, matchId, matchDate, sport, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, lastUpdated)
VALUES (:minerId, :hotkey, :matchId, :matchDate, :sport, :homeTeamName, :awayTeamName, :homeTeamScore, :awayTeamScore, :lastUpdated)
"""
cursor.executemany(insert_query, match_predictions)

# Commit the changes and close the connection
conn.commit()
conn.close()

print(f"Inserted {len(match_predictions)} random MatchPredictions into the database.")