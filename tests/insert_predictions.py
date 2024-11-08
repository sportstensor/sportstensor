import sqlite3
import random
from datetime import datetime, timedelta
from common.data import ProbabilityChoice

# List of miner UIDs and hotkeys
MINERS = [
    (1, "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8"),
    (54, "5EkPSGp7Yt63j1WdFvKduvv2fuXaJKhPyN8cm7jQ5cp1hvBC"),
    (57, "5HDkQ6hUR31yXBvuwXXQzrB14xseuUmgPBbm1ApR1tN3uw7q"),
    (65, "5DPdXPrYCTnsUDh2nYZMCAUb3d6h8eouDCF3zhdw8ru3czSm"),
    (69, "5HNAS5jXy3xX4kUKH4qSoncTTGpwJKNCNNE3CxK8GRtCU5XU"),
    (68, "5Cg6KJrRV1ceYK8uWFPSY2T1DeJ3guivS8VAfgQqkrKFBSEz"),
    (70, "5DPvseipTrLhBikPsH6WNCVnYSVtJt9Sf9h39UWNpSipymGj"),
    (51, "5Gjbs1prGBoXyK1uRmdnsvBkumrRb65n6ufb41D9x2RZgCpR"),
    (13, "5H1GFPwHKdBeE9GacGuxUcJt8vT8Qri4LU5MGQj98AmjRqR3")
]

# Define groups of miners that might submit duplicate predictions
DUPLICATE_GROUPS = [
    [(1, "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8"), 
     (54, "5EkPSGp7Yt63j1WdFvKduvv2fuXaJKhPyN8cm7jQ5cp1hvBC")],
    [(57, "5HDkQ6hUR31yXBvuwXXQzrB14xseuUmgPBbm1ApR1tN3uw7q"),
     (65, "5DPdXPrYCTnsUDh2nYZMCAUb3d6h8eouDCF3zhdw8ru3czSm"),
     (69, "5HNAS5jXy3xX4kUKH4qSoncTTGpwJKNCNNE3CxK8GRtCU5XU")]
]

def connect_to_db(db_name='SportsTensorEdge.db'):
    conn = sqlite3.connect(db_name)
    return conn, conn.cursor()

def close_db(conn):
    conn.commit()
    conn.close()

def get_recently_completed_matches(cursor, days_ago=5, limit=30):
    query = """
    SELECT m.matchId, m.matchDate, m.sport, m.league, m.homeTeamName, m.awayTeamName, 
           m.homeTeamScore, m.awayTeamScore, m.homeTeamOdds, m.awayTeamOdds, m.drawOdds
    FROM Matches m
    WHERE m.matchDate >= datetime('now', ?)
    AND m.isComplete = 1
    AND m.homeTeamOdds IS NOT NULL
    AND m.awayTeamOdds IS NOT NULL
    ORDER BY m.matchDate DESC
    LIMIT ?
    """
    cursor.execute(query, (f'-{days_ago} days', limit))
    return cursor.fetchall()

def odds_to_probability(odds):
    return 1 / odds if odds else 0

def generate_prediction(match, time_to_match, miner_info=None):
    match_id, match_date, sport, league, home_team, away_team, home_score, away_score, home_odds, away_odds, draw_odds = match
    
    # Determine the actual outcome
    if home_score > away_score:
        actual_outcome = ProbabilityChoice.HOMETEAM
        actual_odds = home_odds
    elif away_score > home_score:
        actual_outcome = ProbabilityChoice.AWAYTEAM
        actual_odds = away_odds
    else:
        actual_outcome = ProbabilityChoice.DRAW
        actual_odds = draw_odds
    
    # Generate a "good" prediction based on the actual outcome
    if time_to_match.total_seconds() < 3600:  # Last hour predictions are very accurate
        if random.random() < 0.4:
            wrong_choices = [choice for choice in ProbabilityChoice if choice != actual_outcome]
            prediction = random.choice(wrong_choices)
        else:
            prediction = actual_outcome
        probability = random.uniform(0.8, 0.95)
    else:
        if random.random() < 0.3:
            prediction = actual_outcome
            probability = random.uniform(0.6, 0.85)
        else:
            wrong_choices = [choice for choice in ProbabilityChoice if choice != actual_outcome]
            prediction = random.choice(wrong_choices)
            probability = random.uniform(0.5, 0.7)
    
    projected_edge = (actual_odds - (1 / probability)) * (1 if (actual_outcome == prediction) else -1)
    prediction_date = datetime.strptime(match_date, '%Y-%m-%d %H:%M:%S') - time_to_match
    
    # Use provided miner info or randomly select a miner
    if miner_info is None:
        miner_id, hotkey = random.choice(MINERS)
    else:
        miner_id, hotkey = miner_info
    
    return {
        'minerId': miner_id,
        'hotkey': hotkey,
        'matchId': match_id,
        'matchDate': match_date,
        'sport': sport,
        'league': league,
        'homeTeamName': home_team,
        'awayTeamName': away_team,
        'homeTeamScore': None,
        'awayTeamScore': None,
        'probabilityChoice': prediction.value,
        'probability': probability,
        'predictionDate': prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
        'lastUpdated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def generate_duplicate_predictions(match, time_to_match, duplicate_group):
    """Generate identical predictions for a group of miners with small timing variations."""
    base_prediction = generate_prediction(match, time_to_match, duplicate_group[0])
    predictions = []
    
    for miner_info in duplicate_group:
        dup_prediction = base_prediction.copy()
        dup_prediction['minerId'] = miner_info[0]
        dup_prediction['hotkey'] = miner_info[1]
        
        # Add small random variations to timestamps (within 2 seconds)
        pred_date = datetime.strptime(dup_prediction['predictionDate'], '%Y-%m-%d %H:%M:%S')
        pred_date += timedelta(seconds=random.uniform(0, 2))
        dup_prediction['predictionDate'] = pred_date.strftime('%Y-%m-%d %H:%M:%S')
        
        last_updated = datetime.strptime(dup_prediction['lastUpdated'], '%Y-%m-%d %H:%M:%S')
        last_updated += timedelta(seconds=random.uniform(0, 2))
        dup_prediction['lastUpdated'] = last_updated.strftime('%Y-%m-%d %H:%M:%S')
        
        predictions.append(dup_prediction)
    
    return predictions

def insert_predictions(cursor, predictions):
    insert_query = """
    INSERT OR IGNORE INTO MatchPredictions (minerId, hotkey, matchId, matchDate, sport, league, homeTeamName, awayTeamName, homeTeamScore, awayTeamScore, probabilityChoice, probability, predictionDate, lastUpdated)
    VALUES (:minerId, :hotkey, :matchId, :matchDate, :sport, :league, :homeTeamName, :awayTeamName, :homeTeamScore, :awayTeamScore, :probabilityChoice, :probability, :predictionDate, :lastUpdated)
    """
    cursor.executemany(insert_query, predictions)

def main():
    conn, cursor = connect_to_db()
    
    matches = get_recently_completed_matches(cursor, limit=10)
    all_predictions = []
    
    time_intervals = [
        timedelta(hours=24),
        timedelta(hours=12),
        timedelta(hours=4),
        timedelta(minutes=10)
    ]
    
    for match in matches:
        # Generate regular predictions
        for interval in time_intervals:
            # Regular independent predictions
            prediction = generate_prediction(match, interval)
            all_predictions.append(prediction)
            
            # Random chance to generate duplicate predictions
            if random.random() < 0.3:  # 30% chance of duplicates for each interval
                duplicate_group = random.choice(DUPLICATE_GROUPS)
                duplicate_predictions = generate_duplicate_predictions(match, interval, duplicate_group)
                print(f"Generated {len(duplicate_predictions)} duplicate predictions for match {match[0]}")
                all_predictions.extend(duplicate_predictions)
    
    insert_predictions(cursor, all_predictions)
    
    close_db(conn)
    print(f"Inserted {len(all_predictions)} predictions into the database.")

if __name__ == "__main__":
    main()