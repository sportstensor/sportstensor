import os

IS_PROD = os.environ.get("IS_PROD", "false").lower() == "true"

NETWORK = "mainnet"
NETUID = 41

if not IS_PROD:
    NETWORK = "test"
    NETUID = 172