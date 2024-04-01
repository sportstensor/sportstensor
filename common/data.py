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

class MatchPrediction(StrictBaseModel):
    """Represents a prediction of a sports match."""

    matchID: PositiveInt = Field(
        description="Unique ID that represents a match."
    )

    # The datetime of the starting time of the match. Should be in UTC?
    datetime: dt.datetime
    homeTeamName: str
    awayTeamName: str
    homeTeamScore: Optional[int]
    awayTeamScore: Optional[int]