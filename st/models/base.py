import os
import bittensor as bt
import aiohttp
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from common.data import MatchPrediction, League, ProbabilityChoice, get_league_from_string
from common.constants import LEAGUES_ALLOWING_DRAWS
from st.sport_prediction_model import SportPredictionModel

MINER_ENV_PATH = 'neurons/miner.env'
load_dotenv(dotenv_path=MINER_ENV_PATH)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise ValueError(f"ODDS_API_KEY not found in {MINER_ENV_PATH}")

API_URL = "https://api.the-odds-api.com/v4/sports/"

# Team name mappings for normalization
mismatch_teams_mapping = {
    "Orlando City SC": "Orlando City",
    "Inter Miami CF": "Inter Miami",
    "Atlanta United FC": "Atlanta United",
    "Montreal Impact": "CF Montréal",
    "D.C. United": "DC United",
    "Tottenham Hotspur": "Tottenham",
    "Columbus Crew SC": "Columbus Crew",
    "Minnesota United FC": "Minnesota United",
    "Vancouver Whitecaps FC": "Vancouver Whitecaps",
    "Leicester City": "Leicester",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "LA Galaxy": "L.A. Galaxy",
    "Oakland Athletics": "Athletics",
}

SPORTS_TYPES = [
    {
        'sport_key': 'baseball_mlb',
        'region': 'us,eu',
    },
    {
        'sport_key': 'americanfootball_nfl',
        'region': 'us,eu'
    },
    {
        'sport_key': 'soccer_usa_mls',
        'region': 'us,eu'
    },
    {
        'sport_key': 'soccer_epl',
        'region': 'uk,eu'
    },
    {
        'sport_key': 'basketball_nba',
        'region': 'us,eu'
    },
]

league_mapping = {
    'NBA': 'NBA',
    'NFL': 'NFL',
    'MLS': 'MLS',
    'EPL': 'EPL',
    'MLB': 'MLB',
}

class SportstensorBaseModel(SportPredictionModel):
    def __init__(self, prediction: MatchPrediction):
        super().__init__(prediction)
        self.boost_min_percent = 0.03
        self.boost_max_percent = 0.10
        self.probability_cap = 0.95
        self.max_retries = 3
        self.retry_delay = 0.5
        self.timeout = 3

    async def fetch_odds(self, sport_key: str, region: str) -> Optional[dict]:
        """Fetch odds from the new API."""
        url = f"{API_URL}{sport_key}/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": region,
            "bookmakers": "pinnacle",
            "markets": "h2h"
        }
        async with aiohttp.ClientSession() as session:
            try:
                bt.logging.debug("Fetching odds from API...")
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        print(f"\n=== API Error ===\nStatus: {response.status}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"\n=== API Exception ===\n{str(e)}")
                return None

    def map_team_name(self, team_name: str) -> str:
        """Map team names using mismatch mapping."""
        return mismatch_teams_mapping.get(team_name, team_name)

    def odds_to_probabilities(self, home_odds: float, away_odds: float, draw_odds: Optional[float] = None) -> Dict[str, float]:
        """Convert odds to probabilities."""
        try:
            if home_odds is None or away_odds is None:
                print("Missing required odds values")
                return None

            # Convert odds to probabilities
            home_prob = 1 / home_odds if home_odds > 0 else 0
            away_prob = 1 / away_odds if away_odds > 0 else 0
            draw_prob = 1 / draw_odds if draw_odds and draw_odds > 0 else 0

            # Normalize probabilities
            total = home_prob + away_prob + draw_prob
            if total <= 0:
                print("Invalid odds values resulted in zero total probability")
                return None

            probabilities = {
                "home": home_prob / total,
                "away": away_prob / total,
            }

            if draw_odds:
                probabilities["draw"] = draw_prob / total
            
            return probabilities
        
        except Exception as e:
            print(f"Error converting odds to probabilities: {str(e)}")
            return None
        except Exception as e:
            bt.logging.error(f"Error converting odds to probabilities: {str(e)}")
            return None
    
    async def make_prediction(self):
        """Synchronous wrapper for async prediction logic."""
        bt.logging.info(f"Predicting {self.prediction.league} game...")
        
        try:
            # Convert the league to enum if it's not already one
            if not isinstance(self.prediction.league, League):
                try:
                    league_enum = get_league_from_string(str(self.prediction.league))
                    if league_enum is None:
                        bt.logging.error(f"Unknown league: {self.prediction.league}. Returning.")
                    self.prediction.league = league_enum
                except ValueError as e:
                    bt.logging.error(f"Failed to convert league: {self.prediction.league}. Error: {e}")
            else:
                league_enum = self.prediction.league

            if not isinstance(self.prediction.league, League):
                bt.logging.error(f"Invalid league type: {type(self.prediction.league)}. Expected League enum.")
                return self.prediction
            
            # Dynamically determine sport_key
            league_to_sport_key = {
                "NBA": "basketball_nba",
                "NFL": "americanfootball_nfl",
                "MLS": "soccer_usa_mls",
                "EPL": "soccer_epl",
                "MLB": "baseball_mlb",
                "English Premier League": "soccer_epl",
                "American Major League Soccer": "soccer_usa_mls",
            }

            league_key = self.prediction.league.name
            sport_key = league_to_sport_key.get(league_key)

            if not sport_key:
                bt.logging.error(f"Unknown league: {league_key}. Unable to determine sport_key.")
                return self.prediction

            # Determine the region (optional customization for regions)
            region = "us,eu" if sport_key in ["baseball_mlb", "americanfootball_nfl", "basketball_nba"] else "uk,eu"
            
            odds_data = await self.fetch_odds(sport_key, region)

            if not odds_data:
                bt.logging.error("No odds data fetched.")
                return self.prediction

            # Find the match
            for odds in odds_data:
                home_team = self.map_team_name(self.prediction.homeTeamName)
                away_team = self.map_team_name(self.prediction.awayTeamName)

                if odds["home_team"] == home_team and odds["away_team"] == away_team:
                    bookmaker = next((b for b in odds["bookmakers"] if b["key"] == "pinnacle"), None)
                    if not bookmaker:
                        bt.logging.error("No Pinnacle odds found")
                        continue

                    market = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
                    if not market:
                        bt.logging.error("No h2h market found")
                        continue

                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    home_odds = outcomes.get(home_team)
                    away_odds = outcomes.get(away_team)
                    draw_odds = outcomes.get("Draw") if self.prediction.league in LEAGUES_ALLOWING_DRAWS else None

                    bt.logging.debug(f"Raw odds: {outcomes}")

                    if home_odds is None or away_odds is None:
                        bt.logging.error("Missing odds for one or both teams")
                        continue

                    probabilities = self.odds_to_probabilities(home_odds, away_odds, draw_odds)
                    bt.logging.debug(f"Calculated probabilities: {probabilities}")

                    if probabilities:
                        # Find the highest probability outcome
                        max_prob = max(probabilities["home"], probabilities["away"], probabilities.get("draw", 0))

                        if max_prob == probabilities["home"]:
                            self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                        elif max_prob == probabilities["away"]:
                            self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                        else:
                            self.prediction.probabilityChoice = ProbabilityChoice.DRAW

                        self.prediction.probability = max_prob
                        bt.logging.info(f"Prediction made: {self.prediction.probabilityChoice} with probability {self.prediction.probability}")
                        return

            bt.logging.warning("Match not found in fetched odds data.")
            return
            
        except Exception as e:
            bt.logging.error(f"Failed to make prediction: {str(e)}")
