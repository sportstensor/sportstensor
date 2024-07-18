import os

IS_PROD = os.environ.get("IS_PROD", "false").lower() == "true"

if IS_PROD:
    NETWORK = "mainnet"
    NETUID = 41
else:
    NETWORK = "test"
    NETUID = 172