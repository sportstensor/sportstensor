# The MIT License (MIT)
# Copyright © 2024 sportstensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import pydantic
from common.data import (
    MatchPrediction,
)
from typing import Dict, List, Optional, Tuple


class BaseProtocol(bt.Synapse):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    version: Optional[int] = pydantic.Field(
        description="Protocol version", default=None
    )


class GetMatchPrediction(BaseProtocol):
    """
    Protocol by which Validators can retrieve a Match Prediction from a Miner.

    Attributes:
    - match_predicton: A single MatchPrediction object that the Miner can serve.
    """
    
    match_prediction: MatchPrediction = pydantic.Field(
        description="The MatchPrediction object being requested",
        frozen=False,
        repr=False,
        default=None
    )

    def __str__(self):
        return f"GetMatchPrediction(match_prediction={self.match_prediction}, axon={self.axon})"
    __repr__ = __str__

"""
class GetSoccerPrediction(BaseProtocol):
    
    Protocol by which Validators can retrieve a Soccer Prediction from a Miner.

    Attributes:
    - soccer_prediction_request: The SoccerPredictionRequest object that the requester is asking to be predicted by a Miner.
    - soccer_prediction_response: The SoccerPredictionResponse object that the requester is asking for.
    

    # Required request input, filled by sending dendrite caller.
    soccer_prediction_request: SoccerPredictionRequest = pydantic.Field(
        title="data_entity_bucket_id",
        description="The identifier for the requested DataEntityBucket.",
        frozen=True,
        repr=False,
        default=None,
    )
"""