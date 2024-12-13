import datetime
from common.data import League

IS_DEV = False
# Controls if validators should process our SportsTensor App-based logic
ENABLE_APP = False

# The current protocol version (int)
PROTOCOL_VERSION = 1

# Interval in minutes that we sync match data
DATA_SYNC_INTERVAL_IN_MINUTES = 30

# Interval in minutes that we ask for league commitments from miners
LEAGUE_COMMITMENT_INTERVAL_IN_MINUTES = 15

# Interval in minutes that we poll the api for prediction requests from the app
APP_DATA_SYNC_INTERVAL_IN_MINUTES = 1

# Interval in minutes that we give validators to respond to an assigned app prediction request
APP_PREDICTIONS_UNFULFILLED_THRESHOLD = 3

# Validator enforced miner response timeout in seconds
VALIDATOR_TIMEOUT = 10

# Have Validators pull match data every X seconds.
VALI_REFRESH_MATCHES = 60 * 30

# Have Validators run the cleaning process every X minutes.
PURGE_DEREGGED_MINERS_INTERVAL_IN_MINUTES = 5

# Minimum time in seconds predictions are allowed before match begins
MIN_PREDICTION_TIME_THRESHOLD = 60 * 5

# Max number of days in the future for allowable predictions. 
MAX_PREDICTION_DAYS_THRESHOLD = 1

# Max number of predictions that can be scored at a time
MAX_BATCHSIZE_FOR_SCORING = 500

# Cut off days to attempt to score predictions. i.e. Any predictions not scored with X days will be left behind
SCORING_CUTOFF_IN_DAYS = 30

# Interval in minutes that we attempt to score predictions
SCORING_INTERVAL_IN_MINUTES = 1

# The maximum number of characters a team name can have.
MAX_TEAM_NAME_LENGTH = 32

########## SCORING CONSTANTS ##############
# min probability for a prediction to be considered valid. 0.5 removes lay predictions
MIN_PROBABILITY = 0.5
MIN_PROB_FOR_DRAWS = 0.3340

NO_LEAGUE_COMMITMENT_PENALTY = -0.25
NO_PREDICTION_RESPONSE_PENALTY = -0.1
MAX_GFILTER_FOR_WRONG_PREDICTION = 0.3

# Copycat punishment constants
COPYCAT_PUNISHMENT_START_DATE = datetime.datetime(2024, 9, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
COPYCAT_PENALTY_SCORE = 0
COPYCAT_VARIANCE_THRESHOLD = 0.70
EXACT_MATCH_PREDICTIONS_THRESHOLD = 15
SUSPICIOUS_CONSECUTIVE_MATCHES_THRESHOLD = 10

ACTIVE_LEAGUES = [
    League.EPL,
    League.NFL,
    League.NBA
]

LEAGUES_ALLOWING_DRAWS = [
    League.EPL,
    League.MLS
]

ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE = {
    League.MLB: 250,
    League.NBA: 180,
    League.EPL: 60,
    League.MLS: 60,
    League.NFL: 64
}

# MUST ADD UP to 1.0 (100%)
LEAGUE_SCORING_PERCENTAGES = {
    League.MLB: 0.0,
    League.NBA: 0.35,
    League.EPL: 0.30,
    League.MLS: 0.0,
    League.NFL: 0.35
}

# ALPHA controls how many predictions are needed to start getting rewards. Higher the ALPHA, the less predictions needed.
LEAGUE_SENSITIVITY_ALPHAS = {
    League.MLB: 0.025,
    League.NBA: 0.03,
    League.EPL: 0.1,
    League.MLS: 0.1,
    League.NFL: 0.1
}

# Single sensitivity alpha depcrecated for league-specific sensitivity alphas
SENSITIVITY_ALPHA = 0.025
# GAMMA controls the time decay of CLV. Higher the GAMMA, the faster the decay.
GAMMA = 0.00125
# KAPPA controls the sharpness of the interchange between CLV and Time. Higher the KAPPA, the sharper the interchange.
TRANSITION_KAPPA = 35
# BETA controls the ranges that the CLV component lives within. Higher the BETA, the tighter the range.
EXTREMIS_BETA = 0.25
# PARETO_MU is the minimum value for the Pareto distribution
PARETO_MU = 1.0
# PARETO_ALPHA is the shape of the Pareto distribution. The lower the ALPHA, the more suppressed the distribution.
PARETO_ALPHA = 1.0
