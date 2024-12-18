import random
import asyncio
import aiohttp
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel
from common.constants import LEAGUES_ALLOWING_DRAWS
from common.data import ProbabilityChoice, League
from typing import Optional

API_URL = "https://api.the-odds-api.com/v4/sports/"
ODDS_API_KEY = "your_api_key"  # Replace with your API key

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
    'EPL': 'English Premier League',
    'MLS': 'American Major League Soccer',
    'MLB': 'MLB',
    'NFL': 'NFL',
    'NBA': 'NBA',
}

class SportstensorBaseModel(SportPredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boost_min_percent = 0.03
        self.boost_max_percent = 0.10
        self.probability_cap = 0.95
        self.max_retries = 3
        self.retry_delay = 0.5
        self.timeout = 3

    async def fetch_odds(self, sport_key: str, region: str) -> Optional[dict]:
        """Fetch odds from TOA API."""
        url = f"{API_URL}{sport_key}/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": region,
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        bt.logging.error(f"Failed to fetch odds: {response.status}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                bt.logging.error(f"Error fetching odds: {str(e)}")
                return None

    def map_team_name(self, team_name: str) -> str:
        """Map team names using mismatch mapping."""
        return mismatch_teams_mapping.get(team_name, team_name)

    def odds_to_probabilities(self, home_odds: float, away_odds: float, draw_odds: Optional[float] = None):
        """Convert odds to probabilities."""
        home_prob = 1 / home_odds
        away_prob = 1 / away_odds
        draw_prob = 1 / draw_odds if draw_odds else 0

        total_prob = home_prob + away_prob + draw_prob

        return {
            "home": home_prob / total_prob,
            "away": away_prob / total_prob,
            "draw": draw_prob / total_prob if draw_odds else None
        }

    async def make_prediction(self):
        """Override to fetch odds and make predictions."""
        try:
            # Dynamically determine sport_key
            league_to_sport_key = {
                League.NBA: "basketball_nba",
                League.NFL: "americanfootball_nfl",
                League.MLS: "soccer_usa_mls",
                League.EPL: "soccer_epl",
                League.MLB: "baseball_mlb",
            }

            sport_key = league_to_sport_key.get(self.prediction.league, None)

            if not sport_key:
                bt.logging.error(f"Unknown league: {self.prediction.league}. Unable to determine sport_key.")
                return


            # Determine the region (optional customization for regions)
            region = "us,eu" if sport_key in ["baseball_mlb", "americanfootball_nfl", "basketball_nba"] else "uk,eu"

            # Fetch odds using the determined sport_key
            odds_data = await self.fetch_odds(sport_key, region)

            if not odds_data:
                bt.logging.error("No odds data fetched.")
                return

            bt.logging.info(f"Fetched odds data: {odds_data}")

            # Find the match
            for odds in odds_data:
                home_team = self.map_team_name(self.prediction.homeTeamName)
                away_team = self.map_team_name(self.prediction.awayTeamName)

                if odds["home_team"] == home_team and odds["away_team"] == away_team:
                    if not odds.get("bookmakers") or not odds["bookmakers"][0].get("markets"):
                        bt.logging.error("No bookmakers or markets data found")
                        return

                    # Find the correct outcome for each team
                    outcomes = odds["bookmakers"][0]["markets"][0]["outcomes"]
                    bt.logging.info(f"Processing outcomes: {outcomes}")

                    home_odds = None
                    away_odds = None
                    draw_odds = None

                    for outcome in outcomes:
                        if outcome["name"] == home_team:
                            home_odds = outcome["price"]
                        elif outcome["name"] == away_team:
                            away_odds = outcome["price"]
                        elif outcome["name"].lower() == "draw":
                            draw_odds = outcome["price"]

                    if not home_odds or not away_odds:
                        bt.logging.error(f"Could not find odds for both teams. Home: {home_odds}, Away: {away_odds}")
                        return

                    probabilities = self.odds_to_probabilities(home_odds, away_odds, draw_odds)

                    # Find the highest probability outcome
                    max_prob = max(
                        filter(None, [
                            probabilities["home"],
                            probabilities["away"],
                            probabilities.get("draw", 0)
                        ])
                    )

                    if max_prob == probabilities["home"]:
                        self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                    elif max_prob == probabilities["away"]:
                        self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                    elif probabilities.get("draw") and max_prob == probabilities["draw"]:
                        self.prediction.probabilityChoice = ProbabilityChoice.DRAW

                    self.prediction.probability = max_prob
                    bt.logging.info(f"Prediction made: {self.prediction.probabilityChoice} with probability {self.prediction.probability}")
                    return

            bt.logging.warning("Match not found in fetched odds data.")

            bt.logging.warning("Match not found in fetched odds data.")
        except Exception as e:
            bt.logging.error(f"Error making prediction: {str(e)}")
