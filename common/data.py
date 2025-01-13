import datetime as dt
from enum import IntEnum, Enum
from typing import Optional
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
    field_validator,
    model_serializer,
)


class StrictBaseModel(BaseModel):
    """A BaseModel that enforces stricter validation constraints"""

    class Config:
        # JSON serialization doesn't seem to work correctly without
        # enabling `use_enum_values`. It's possible this isn't an
        # issue with newer version of pydantic, which we can't use.
        use_enum_values = True


class Sport(IntEnum):
    """The sport a prediction pertains to. This will be expanded over time as we increase the types of sports we predict."""

    SOCCER = 1
    FOOTBALL = 2
    BASEBALL = 3
    BASKETBALL = 4
    CRICKET = 5
    # Additional enum values reserved for yet to be implemented sources.
    UNKNOWN_6 = 6
    UNKNOWN_7 = 7


class League(Enum):
    """Represents a sports league, mainly used for mapping and indicating active status to run predictions on."""

    MLB = "MLB"
    NFL = "NFL"
    NBA = "NBA"
    NHL = "NHL"
    EPL = "English Premier League"
    MLS = "American Major League Soccer"

    """
    leagueId: PositiveInt = Field(
        description="Unique ID that represents a league."
    )
    leagueName: str = Field(
        description="Name of the league. i.e. English Premiere League, NFL, MLB"
    )
    sport: Sport
    isActive: bool = False
    """
    
    def __eq__(self, other):
        if isinstance(other, League):
            return self is other
        return self.value == other

    def __hash__(self):
        return hash(self.name)


def get_league_from_string(league_string: str) -> League:
    for league in League:
        if league_string.upper() == league.name or league_string == league.value:
            return league
    raise ValueError(f"Invalid league: {league_string}")


class Match(StrictBaseModel):
    """Represents a match/game, sport agnostic."""

    matchId: str = Field(info={"description": "Unique ID that represents a match."})

    # The datetime of the starting time of the match. Should be in UTC?
    matchDate: dt.datetime

    sport: Sport
    league: League

    # Set variable to keep track if the match has completed. Default to False.
    isComplete: bool = False

    homeTeamName: str
    awayTeamName: str
    homeTeamScore: Optional[int]
    awayTeamScore: Optional[int]
    
    # Define our odds for the match. These change over time. Final value are the closing odds.
    homeTeamOdds: Optional[float]
    awayTeamOdds: Optional[float]
    drawOdds: Optional[float]

    # Validators to ensure immutability
    @field_validator(
        "matchId",
        "matchDate",
        "sport",
        "league",
        "homeTeamName",
        "awayTeamName",
        mode="before",
    )
    def match_fields_are_immutable(cls, v, info):
        if info.data.get(info.field_name) is not None and v != info.data[info.field_name]:
            raise ValueError(f"{info.field_name} is immutable and cannot be changed")
        return v


class Prediction(StrictBaseModel):
    """Represents a base prediction, sport agnostic."""

    predictionId: Optional[PositiveInt] = Field(
        default=None,
        description="Unique ID that represents a predication."
    )

    minerId: Optional[NonNegativeInt] = Field(
        default=None,
        description="Unique ID that represents a miner."
    )

    hotkey: Optional[str] = Field(
        default=None, 
        description="A unique identifier for the miner."
    )

    predictionDate: Optional[dt.datetime] = Field(
        default=None,
        description="The datetime the prediction was made. Should be UTC"
    )

    matchId: str = Field(description="Unique ID that represents a match.")

    # The datetime of the starting time of the match. Should be in UTC
    matchDate: dt.datetime = Field(
        description="The datetime of the starting time of the match. Should be UTC"
    )

    sport: Sport
    league: League

    # Set variable to keep track if the prediction has been scored. Default to False.
    isScored: bool = False
    scoredDate: Optional[dt.datetime] = Field(default=None)

    # Validators to ensure immutability
    @field_validator(
        "predictionId",
        "matchId",
        "matchDate",
        "sport",
        "league",
        mode="before",
    )
    def match_fields_are_immutable(cls, v, info):
        if info.data.get(info.field_name) is not None and v != info.data[info.field_name]:
            raise ValueError(f"{info.field_name} is immutable and cannot be changed")
        return v

    def __str__(self):
        sport_name = (
            self.sport.name if isinstance(self.sport, Sport) else Sport(self.sport).name
        )
        league_name = (
            self.league.name
            if isinstance(self.league, League)
            else League(self.league).name
        )
        return (
            f"Prediction(predictionId={self.predictionId}, "
            f"minerId={self.minerId}, hotkey={self.hotkey}, "
            f"matchId={self.matchId}, matchDate={self.matchDate}, "
            f"sport={sport_name}, league={league_name}, "
            f"isScored={self.isScored}, scoredDate={self.scoredDate})"
        )


class ProbabilityChoice(Enum):
    """Represents the choice a miner can make for a prediction."""

    HOMETEAM = "HomeTeam"
    AWAYTEAM = "AwayTeam"
    DRAW = "Draw"


def get_probablity_choice_from_string(choice_str: str) -> Optional[ProbabilityChoice]:
    """Utility function to get a ProbabilityChoice enum from a string."""
    for choice in ProbabilityChoice:
        if choice.value == choice_str:
            return choice
    return None


class MatchPrediction(Prediction):
    """Represents a prediction of a sports match."""

    homeTeamName: str
    awayTeamName: str
    homeTeamScore: Optional[int] = Field(default=None)
    awayTeamScore: Optional[int] = Field(default=None)

    probabilityChoice: Optional[ProbabilityChoice] = Field(default=None)
    probability: Optional[float] = Field(default=None)
    
    closingEdge: Optional[float] = Field(default=None)

    def get_predicted_team(self) -> str:
        """Get the predicted team or Draw based on the probability choice."""
        if self.probabilityChoice == ProbabilityChoice.HOMETEAM or self.probabilityChoice == ProbabilityChoice.HOMETEAM.value:
            return self.homeTeamName
        elif self.probabilityChoice == ProbabilityChoice.AWAYTEAM or self.probabilityChoice == ProbabilityChoice.AWAYTEAM.value:
            return self.awayTeamName
        elif self.probabilityChoice == ProbabilityChoice.DRAW or self.probabilityChoice == ProbabilityChoice.DRAW.value:
            return "Draw"
        else:
            return "Unknown"
        
    @model_serializer(mode="wrap")
    def serialize_dates(self, handler):
        data = handler(self)
        # Convert datetime objects to ISO strings for serialization
        if isinstance(data['matchDate'], dt.datetime):
            data['matchDate'] = data['matchDate'].isoformat()
        if isinstance(data['predictionDate'], dt.datetime):
            data['predictionDate'] = data['predictionDate'].isoformat()
        if isinstance(data['scoredDate'], dt.datetime):
            data['scoredDate'] = data['scoredDate'].isoformat()
        return data

    # Validators to ensure immutability
    @field_validator(
        "homeTeamName", "awayTeamName", mode="before"
    )
    def match_fields_are_immutable(cls, v, info):
        if info.data.get(info.field_name) is not None and v != info.data[info.field_name]:
            raise ValueError(f"{info.field_name} is immutable and cannot be changed")
        return v

    def __str__(self):
        base_str = super().__str__()
        return (
            f"{base_str[:-1]}, "  # Remove the closing parenthesis from the base string
            f"homeTeamName={self.homeTeamName}, awayTeamName={self.awayTeamName}, "
            #f"homeTeamScore={self.homeTeamScore}, awayTeamScore={self.awayTeamScore}, "
            f"probabilityChoice={self.probabilityChoice}, probability={self.probability})"
        )
    
    def pretty_print(self):
        league_name = (
            self.league.name
            if isinstance(self.league, League)
            else League(self.league).name
        )
        return (
            f"Prediction for {league_name} match between {self.homeTeamName} and {self.awayTeamName}.\n"
            f"Predicted winner: {self.get_predicted_team()} with a probability of {self.probability:.2f}.\n"
            f"Match date: {self.matchDate.strftime('%Y-%m-%d %H:%M:%S')}"
        )


class MatchPredictionWithMatchData(BaseModel):
    prediction: MatchPrediction
    actualHomeTeamScore: int
    actualAwayTeamScore: int
    homeTeamOdds: float
    awayTeamOdds: float
    drawOdds: float

    def get_closing_odds_for_predicted_outcome(self) -> Optional[float]:
        """
        Get the closing odds for the predicted outcome.

        :return: float, the closing odds for the predicted outcome, or None if unavailable
        """
        predicted_team = self.prediction.get_predicted_team()
        if predicted_team == self.prediction.homeTeamName:
            return self.homeTeamOdds
        elif predicted_team == self.prediction.awayTeamName:
            return self.awayTeamOdds
        elif predicted_team == "Draw":
            return self.drawOdds
        else:
            return None

    def get_actual_winner(self) -> str:
        """Get the actual winner of the match."""
        if self.actualHomeTeamScore > self.actualAwayTeamScore:
            return self.prediction.homeTeamName
        elif self.actualHomeTeamScore < self.actualAwayTeamScore:
            return self.prediction.awayTeamName
        return "Draw"
    
    def get_actual_winner_odds(self) -> float:
        """Get the odds of the actual winner of the match."""
        if self.actualHomeTeamScore > self.actualAwayTeamScore:
            return self.homeTeamOdds
        elif self.actualHomeTeamScore < self.actualAwayTeamScore:
            return self.awayTeamOdds
        return self.drawOdds
    
    def get_actual_loser_odds(self) -> float:
        """Get the odds of the actual loser of the match."""
        if self.actualHomeTeamScore > self.actualAwayTeamScore:
            return self.awayTeamOdds
        elif self.actualHomeTeamScore < self.actualAwayTeamScore:
            return self.homeTeamOdds
        return self.drawOdds

    def is_prediction_for_underdog(self, LEAGUES_ALLOWING_DRAWS: list) -> bool:
        """
        Check if the prediction supports the underdog.
        
        :param LEAGUES_ALLOWING_DRAWS: List of leagues where draws are common.
        :return: True if the prediction is for the underdog, False otherwise.
        """
        # Compare odds to determine underdog status
        predicted_odds = self.get_closing_odds_for_predicted_outcome()
        if predicted_odds is None:
            return False

        # Determine if the prediction aligns with the underdog
        favorite_odds = min([self.homeTeamOdds, self.awayTeamOdds])
        if self.prediction.league in LEAGUES_ALLOWING_DRAWS and self.drawOdds > 0:
            favorite_odds = min([self.homeTeamOdds, self.awayTeamOdds, self.drawOdds])
        return predicted_odds > favorite_odds
    