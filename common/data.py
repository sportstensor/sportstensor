import dataclasses
import time

# from common import constants
# from . import utils
import datetime as dt
from enum import IntEnum, Enum
from typing import Any, Dict, List, Type, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    NonNegativeInt,
    validator,
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


def get_league_from_string(league_str: str) -> Optional[League]:
    """Utility function to get a League enum from a string."""
    for league in League:
        if league.value == league_str:
            return league
    return None


class Match(StrictBaseModel):
    """Represents a match/game, sport agnostic."""

    matchId: str = Field(description="Unique ID that represents a match.")

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

    # Validators to ensure immutability
    @validator(
        "matchId",
        "matchDate",
        "sport",
        "league",
        "homeTeamName",
        "awayTeamName",
        pre=True,
        always=True,
        check_fields=False,
    )
    def match_fields_are_immutable(cls, v, values, field):
        if field.name in values and v != values[field.name]:
            raise ValueError(f"{field.name} is immutable and cannot be changed")
        return v


class Prediction(StrictBaseModel):
    """Represents a base prediction, sport agnostic."""

    predictionId: Optional[PositiveInt] = Field(
        description="Unique ID that represents a predication."
    )

    minerId: Optional[NonNegativeInt] = Field(
        description="Unique ID that represents a miner."
    )

    hotkey: Optional[str] = Field(description="A unique identifier for the miner.")

    matchId: str = Field(description="Unique ID that represents a match.")

    # The datetime of the starting time of the match. Should be in UTC
    matchDate: dt.datetime = Field(
        description="The datetime of the starting time of the match. Should be UTC"
    )

    sport: Sport
    league: League

    # Set variable to keep track if the prediction has been scored. Default to False.
    isScored: bool = False
    scoredDate: Optional[dt.datetime]

    # Validators to ensure immutability
    @validator(
        "predictionId",
        "matchId",
        "matchDate",
        "sport",
        "league",
        pre=True,
        always=True,
        check_fields=False,
    )
    def base_fields_are_immutable(cls, v, values, field):
        if field.name in values and v != values[field.name]:
            raise ValueError(f"{field.name} is immutable and cannot be changed")
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


class MatchPrediction(Prediction):
    """Represents a prediction of a sports match."""

    homeTeamName: str
    awayTeamName: str
    homeTeamScore: Optional[int]
    awayTeamScore: Optional[int]

    # Validators to ensure immutability
    @validator(
        "homeTeamName", "awayTeamName", pre=True, always=True, check_fields=False
    )
    def match_fields_are_immutable(cls, v, values, field):
        if field.name in values and v != values[field.name]:
            raise ValueError(f"{field.name} is immutable and cannot be changed")
        return v

    def __str__(self):
        base_str = super().__str__()
        return (
            f"{base_str[:-1]}, "  # Remove the closing parenthesis from the base string
            f"homeTeamName={self.homeTeamName}, awayTeamName={self.awayTeamName}, "
            f"homeTeamScore={self.homeTeamScore}, awayTeamScore={self.awayTeamScore})"
        )


class MatchPredictionWithMatchData(BaseModel):
    prediction: MatchPrediction
    actualHomeTeamScore: int
    actualAwayTeamScore: int
