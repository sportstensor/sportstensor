import dataclasses
import time
#from common import constants
#from . import utils
import datetime as dt
from enum import IntEnum
from typing import Any, Dict, List, Type, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
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
    # Additional enum values reserved for yet to be implemented sources.
    UNKNOWN_5 = 5
    UNKNOWN_6 = 6
    UNKNOWN_7 = 7

class League(StrictBaseModel):
    """Represents a sports league, mainly used for mapping and indicating active status to run predictiosn on."""

    leagueId: PositiveInt = Field(
        description="Unique ID that represents a league."
    )
    leagueName: str = Field(
        description="Name of the league. i.e. English Premiere League, NFL, MLB"
    )
    sport: Sport
    isActive: bool = False

class Match(StrictBaseModel):
    """Represents a match/game, sport agnostic."""

    matchId: str = Field(
        description="Unique ID that represents a match."
    )

    # The datetime of the starting time of the match. Should be in UTC?
    matchDatetime: dt.datetime

    sport: Sport
    
    # Set variable to keep track if the match has completed. Default to False.
    isComplete: bool = False

    homeTeamName: str
    awayTeamName: str
    homeTeamScore: Optional[int]
    awayTeamScore: Optional[int]

    # Validators to ensure immutability
    @validator('matchId', 'matchDatetime', 'sport', 'homeTeamName', 'awayTeamName', pre=True, always=True, check_fields=False)
    def match_fields_are_immutable(cls, v, values, field):
        if field.name in values and v != values[field.name]:
            raise ValueError(f"{field.name} is immutable and cannot be changed")
        return v

class Prediction(StrictBaseModel):
    """Represents a base prediction, sport agnostic."""

    predictionId: Optional[PositiveInt] = Field(
        description="Unique ID that represents a predication."
    )

    matchId: PositiveInt = Field(
        description="Unique ID that represents a match."
    )

    # The datetime of the starting time of the match. Should be in UTC?
    matchDatetime: dt.datetime

    sport: Sport
    
    # Set variable to keep track if the prediction has been scored. Default to False.
    isScored: bool = False
    scoredDate: Optional[dt.datetime]
    
    # Validators to ensure immutability
    @validator('predictionId', 'matchId', 'matchDatetime', 'sport', pre=True, always=True, check_fields=False)
    def base_fields_are_immutable(cls, v, values, field):
        if field.name in values and v != values[field.name]:
            raise ValueError(f"{field.name} is immutable and cannot be changed")
        return v

class MatchPrediction(Prediction):
    """Represents a prediction of a sports match."""
    
    homeTeamName: str
    awayTeamName: str
    homeTeamScore: Optional[int]
    awayTeamScore: Optional[int]

    # Validators to ensure immutability
    @validator('homeTeamName', 'awayTeamName', pre=True, always=True, check_fields=False)
    def match_fields_are_immutable(cls, v, values, field):
        if field.name in values and v != values[field.name]:
            raise ValueError(f"{field.name} is immutable and cannot be changed")
        return v

class MatchPredictionWithMatchData(BaseModel):
    prediction: MatchPrediction
    actualHomeTeamScore: int
    actualAwayTeamScore: int