import sqlite3
import random
from datetime import datetime, timedelta
from common.data import ProbabilityChoice
from st.sport_prediction_model import generate_random_probability_no_tie, generate_random_probabilities_with_tie

# Function to create a random MatchPrediction
def create_random_match_prediction(matchId, matchDate, sport, league, homeTeamName, awayTeamName):
    uid = random.randint(1, 16)  # Generate a random integer between 1 and 16
    uid = 1

    hotkey = f'DEV_{str(uid)}'
    hotkey = "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8"

    prob_a, prob_b = generate_random_probability_no_tie()
    if prob_a > prob_b:
        probabilityChoice = ProbabilityChoice.HOMETEAM
    else:
        probabilityChoice = ProbabilityChoice.AWAYTEAM

    probability = max(prob_a, prob_b)
    
    return {
        'minerId': uid,
        'hotkey': hotkey,
        'matchId': matchId,
        'matchDate': matchDate,
        'sport': sport,
        'league': league,
        'homeTeamName': homeTeamName,
        'awayTeamName': awayTeamName,
        'homeTeamScore': random.randint(0, 10),
        'awayTeamScore': random.randint(0, 10),
        'probabilityChoice': probabilityChoice.value,
        'probability': probability,
        'predictionDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'lastUpdated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# Connect to the SQLite database
conn = sqlite3.connect('SportsTensorEdge.db')
cursor = conn.cursor()

# Calculate the date 2 days ago
two_days_ago = datetime.now() - timedelta(days=2)
two_days_ago_str = two_days_ago.strftime('%Y-%m-%d %H:%M:%S')

# Query for matches that have completed in the last 2 days
query = """
SELECT * FROM Matches
WHERE matchDate >= ?
AND isComplete = 1
AND homeTeamOdds IS NOT NULL
AND awayTeamOdds IS NOT NULL
LIMIT 30
"""
cursor.execute(query, (two_days_ago_str,))
matches = cursor.fetchall()

# Create random MatchPredictions for the matches
match_predictions = [create_random_match_prediction(match[0], match[1], match[2], match[3], match[4], match[5]) for match in matches]

# Insert the random MatchPredictions into the MatchPredictions table
insert_query = """
INSERT OR IGNORE INTO MatchPredictions (minerId, hotkey, matchId, matchDate, sport, league, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, probabilityChoice, probability, predictionDate, lastUpdated)
VALUES (:minerId, :hotkey, :matchId, :matchDate, :sport, :league, :homeTeamName, :awayTeamName, :homeTeamScore, :awayTeamScore, :probabilityChoice, :probability, :predictionDate, :lastUpdated)
"""
cursor.executemany(insert_query, match_predictions)

# Commit the changes and close the connection
conn.commit()
conn.close()

print(f"Inserted {len(match_predictions)} random MatchPredictions into the database.")