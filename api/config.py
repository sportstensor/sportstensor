import os

NETWORK = os.environ["NETWORK"]
NETUID = int(os.environ["NETUID"])

IS_PROD = os.environ.get("IS_PROD", "false").lower() == "true"