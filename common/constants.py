import datetime

# Collection of constants for use throughout the codebase.

# The current protocol version (int)
PROTOCOL_VERSION = 1

# Interval in minutes that we sync match data
DATA_SYNC_INTERVAL_IN_MINUTES = 30

# Validator API endpoint timeout in seconds
VALIDATOR_TIMEOUT = 120

# Have Validators pull match data ever X seconds.
VALI_REFRESH_MATCHES = 60 * 30

# The base FloatTensor score for all miners that return a valid prediction
BASE_MINER_PREDICTION_SCORE = 0.01

# Number of miners to send predictions to. -1 == all
NUM_MINERS_TO_SEND_TO = -1

# Minimum time in seconds predictions are allowed before match begins
MIN_PREDICTION_TIME_THRESHOLD = 60 * 30

# Max number of days in the future for allowable predictions. 
MAX_PREDICTION_DAYS_THRESHOLD = 2

# Max number of predictions that can be scored at a time
MAX_BATCHSIZE_FOR_SCORING = 10

# Cut off days to attempt to score predictions. i.e. Any predictions not scored with X days will be left behind
SCORING_CUTOFF_IN_DAYS = 3

# Interval in minutes that we attempt to score predictions
SCORING_INTERVAL_IN_MINUTES = 5

# The maximum number of characters a team name can have.
MAX_TEAM_NAME_LENGTH = 32

########## SCORING CONSTANTS ##############
CORRECT_MATCH_WINNER_SCORE = 0.5

MAX_SCORE_DIFFERENCE = 10
MAX_SCORE_DIFFERENCE_SOCCER = 10
MAX_SCORE_DIFFERENCE_FOOTBALL = 50
MAX_SCORE_DIFFERENCE_BASKETBALL = 50
MAX_SCORE_DIFFERENCE_BASEBALL = 20

