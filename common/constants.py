import datetime

# Collection of constants for use throughout the codebase.

# The current protocol version (int)
PROTOCOL_VERSION = 1

# Validator API endpoint timeout in seconds
VALIDATOR_TIMEOUT = 120

# The base FloatTensor score for all miners that return a valid prediction
BASE_MINER_PREDICTION_SCORE = 0.01

# Number of miners to send predictions to. -1 == all
NUM_MINERS_TO_SEND_TO = -1

# Max number of days in the future for allowable predictions. 
MAX_PREDICTION_DAYS_THRESHOLD = 2

# Max number of predictions that can be scored at a time
MAX_BATCHSIZE_FOR_SCORING = 10

# The maximum number of characters a team name can have.
MAX_TEAM_NAME_LENGTH = 32

