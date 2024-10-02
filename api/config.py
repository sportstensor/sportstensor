import os
from dotenv import load_dotenv
import json

# Define the path to the api.env file
env_path = os.path.join(os.path.dirname(__file__), 'api.env')

# Load the environment variables from the api.env file
load_dotenv(dotenv_path=env_path)

IS_PROD = False if os.getenv("IS_PROD") == "False" else True

NETWORK = None
NETUID = 41

if not IS_PROD:
    NETWORK = "test"
    NETUID = 172

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

API_KEYS = os.getenv('API_KEYS')
ODDS_API_KEY=os.getenv('ODDS_API_KEY')

TESTNET_VALI_HOTKEYS = json.loads(os.environ["TESTNET_VALI_HOTKEYS"])
