import os

IS_PROD = True

NETWORK = None
NETUID = 41

if not IS_PROD:
    NETWORK = "test"
    NETUID = 172