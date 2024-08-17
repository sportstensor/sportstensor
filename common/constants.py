import datetime

IS_DEV = False
# Controls if validators should process our SportsTensor App-based logic
ENABLE_APP = True

# The current protocol version (int)
PROTOCOL_VERSION = 1

# Interval in minutes that we sync match data
DATA_SYNC_INTERVAL_IN_MINUTES = 30

# Interval in minutes that we poll the api for prediction requests from the app
APP_DATA_SYNC_INTERVAL_IN_MINUTES = 1

# Validator API endpoint timeout in seconds
VALIDATOR_TIMEOUT = 120

# Have Validators pull match data ever X seconds.
VALI_REFRESH_MATCHES = 60 * 30

# The base FloatTensor score for all miners that return a valid prediction
BASE_MINER_PREDICTION_SCORE = 0.01

# Number of miners to send predictions to. -1 == all
NUM_MINERS_TO_SEND_TO = 20

# Minimum time in seconds predictions are allowed before match begins
MIN_PREDICTION_TIME_THRESHOLD = 60 * 30

# Max number of days in the future for allowable predictions. 
MAX_PREDICTION_DAYS_THRESHOLD = 2

# Max number of predictions that can be scored at a time
MAX_BATCHSIZE_FOR_SCORING = 25

# Cut off days to attempt to score predictions. i.e. Any predictions not scored with X days will be left behind
SCORING_CUTOFF_IN_DAYS = 10

# Interval in minutes that we attempt to score predictions
SCORING_INTERVAL_IN_MINUTES = 1

# The maximum number of characters a team name can have.
MAX_TEAM_NAME_LENGTH = 32

########## SCORING CONSTANTS ##############
CORRECT_MATCH_WINNER_SCORE = 0.5
# The score a miner must achieve to earn weights
TOTAL_SCORE_THRESHOLD = 0.4

MAX_SCORE_DIFFERENCE = 10
MAX_SCORE_DIFFERENCE_SOCCER = 10
MAX_SCORE_DIFFERENCE_FOOTBALL = 50
MAX_SCORE_DIFFERENCE_BASKETBALL = 50
MAX_SCORE_DIFFERENCE_BASEBALL = 20
MAX_SCORE_DIFFERENCE_CRICKET = 20