import bittensor as bt
import aiohttp
import asyncio
from typing import Optional, Dict, Any
from common.data import MatchPrediction, Sport, League, ProbabilityChoice
from common.constants import LEAGUES_ALLOWING_DRAWS
from st.sport_prediction_model import SportPredictionModel

API_URL = "https://api.the-odds-api.com/v4/sports/"
ODDS_API_KEY = "your_api_key"  # The Odds API key

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
    "LA Galaxy": "L.A. Galaxy"
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
        print(f"\n=== API Request ===\nURL: {url}\nParams: {params}\n")
        async with aiohttp.ClientSession() as session:
            try:
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

            print(f"\n=== Odds to Probabilities Conversion ===")
            print(f"Input odds: Home={home_odds}, Away={away_odds}, Draw={draw_odds}")

            # Convert odds to probabilities
            home_prob = 1 / home_odds if home_odds > 0 else 0
            away_prob = 1 / away_odds if away_odds > 0 else 0
            draw_prob = 1 / draw_odds if draw_odds and draw_odds > 0 else 0

            print(f"Initial probabilities: Home={home_prob}, Away={away_prob}, Draw={draw_prob}")

            # Normalize probabilities
            total = home_prob + away_prob + draw_prob
            if total <= 0:
                print("Invalid odds values resulted in zero total probability")
                return None

            print(f"Normalization total: {total}")

            probabilities = {
                "home": home_prob / total,
                "away": away_prob / total,
            }

            if draw_odds:
                probabilities["draw"] = draw_prob / total

            print(f"Final normalized probabilities: {probabilities}")
            return probabilities
        except Exception as e:
            print(f"Error converting odds to probabilities: {str(e)}")
            return None

            bt.logging.info(f"Final normalized probabilities: {probabilities}")
            return probabilities
        except Exception as e:
            bt.logging.error(f"Error converting odds to probabilities: {str(e)}")
            return None

    async def make_prediction(self):
        """Override to fetch odds and make predictions."""
        try:
            bt.logging.info(f"Starting prediction for league type {type(self.prediction.league)}")
            if not isinstance(self.prediction.league, League):
                bt.logging.error(f"Invalid league type: {type(self.prediction.league)}. Expected League enum.")
                return self.prediction

            bt.logging.info(f"Starting prediction for {self.prediction.league.name} match: {self.prediction.homeTeamName} vs {self.prediction.awayTeamName}")

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

            bt.logging.info(f"League key: {league_key}, Determined sport_key: {sport_key}")

            if not sport_key:
                bt.logging.error(f"Unknown league: {league_key}. Unable to determine sport_key.")
                return self.prediction

            # Determine the region (optional customization for regions)
            region = "us,eu" if sport_key in ["baseball_mlb", "americanfootball_nfl", "basketball_nba"] else "uk,eu"

            # Fetch odds using the determined sport_key
            odds_data = await self.fetch_odds(sport_key, region)
            bt.logging.info(f"Fetched odds data")

            if not odds_data:
                bt.logging.error("No odds data fetched.")
                return self.prediction

            # Find the match
            for odds in odds_data:
                home_team = self.map_team_name(self.prediction.homeTeamName)
                away_team = self.map_team_name(self.prediction.awayTeamName)

                bt.logging.info(f"Comparing teams - Looking for: {home_team} vs {away_team}")
                bt.logging.info(f"Current odds entry: {odds.get('home_team')} vs {odds.get('away_team')}")

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
                    draw_odds = outcomes.get("Draw") if str(self.prediction.league) in ["English Premier League", "American Major League Soccer"] else None

                    print(f"\n=== Parsed Odds ===\nRaw outcomes: {outcomes}\nHome odds: {home_odds}\nAway odds: {away_odds}\nDraw odds: {draw_odds}\n")

                    if home_odds is None or away_odds is None:
                        bt.logging.error("Missing odds for one or both teams")
                        continue

                    probabilities = self.odds_to_probabilities(home_odds, away_odds, draw_odds)
                    bt.logging.info(f"Calculated probabilities: {probabilities}")

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
                        return self.prediction

            bt.logging.warning("Match not found in fetched odds data.")
            return self.prediction
        except Exception as e:
            bt.logging.error(f"Error making prediction: {str(e)}")
            return self.prediction