from typing import Annotated, List, Optional
from pydantic import conint
from traceback import print_exception

import bittensor
import uvicorn
import asyncio
import logging
import random
from fastapi import FastAPI, HTTPException, Depends, Body, Path, Security, Request, BackgroundTasks
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from starlette import status
from substrateinterface import Keypair

from fastapi import FastAPI
import sentry_sdk

# mysqlclient install issues: https://stackoverflow.com/a/77020207
import mysql.connector
from mysql.connector import Error

from datetime import datetime, timedelta
import api.db as db
from api.config import NETWORK, NETUID, IS_PROD, API_KEYS, TESTNET_VALI_HOTKEYS
from common.constants import ENABLE_APP, APP_PREDICTIONS_UNFULFILLED_THRESHOLD

sentry_sdk.init(
    dsn="https://d9cce5fe3664e00bf8857b2e425d9ec5@o4507644404236288.ingest.de.sentry.io/4507644429271120",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

app = FastAPI()

# define the APIKeyHeader for API authorization to our APP endpoints
api_key_header = APIKeyHeader(name="ST_API_KEY", auto_error=False)
security = HTTPBasic()

# Cache configuration
MATCHES_CACHE_TTL = 300  # 5 minutes

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

async def get_api_key(api_key_header: Optional[str] = Security(api_key_header)) -> Optional[str]:
    if api_key_header is not None and api_key_header in API_KEYS:
        return api_key_header
    return None

def get_hotkey(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> Optional[str]:
    if not credentials:
        return None
    
    keypair = Keypair(ss58_address=credentials.username)

    if keypair.verify(credentials.username, credentials.password):
        return credentials.username
    
    return None

def authenticate_with_bittensor(hotkey, metagraph):
    if hotkey not in metagraph.hotkeys:
        print(f"Hotkey not found in metagraph.")
        return False

    uid = metagraph.hotkeys.index(hotkey)
    if not metagraph.validator_permit[uid] and NETWORK != "test":
        print("Bittensor validator permit required")
        return False

    if metagraph.S[uid] < 1000 and NETWORK != "test":
        print("Bittensor validator requires 1000+ staked TAO")
        return False

    return True


async def main():
    app = FastAPI()

    matches_cache = {
        "data": None,
        "last_updated": None
    }

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            try:
                # Sync the metagraph.
                metagraph.sync(subtensor=subtensor)

            # In case of unforeseen errors, the api will log the error and continue operations.
            except Exception as err:
                print("Error during metagraph sync", str(err))
                print_exception(type(err), err, err.__traceback__)

            await asyncio.sleep(300)

    async def resync_miner_statuses():
        while True:
            """Checks active hotkeys on the metagraph and updates our results table. Also updates the miner coldkey and age of miner on the subnet."""
            print("resync_miner_statuses()")

            try:
                active_uids = []
                active_hotkeys = []
                active_coldkeys = []
                ages = []
                current_block = subtensor.get_current_block()
                # Assuming an average block time of 12 seconds (adjust as necessary)
                block_time_seconds = 12
                for uid in range(metagraph.n.item()):
                    active_uids.append(uid)
                    active_hotkeys.append(metagraph.hotkeys[uid])
                    active_coldkeys.append(metagraph.coldkeys[uid])
                    
                    # calculate the age of the miner in hours
                    # query the subtensor for the block at registration
                    registration_block = subtensor.query_module('SubtensorModule','BlockAtRegistration',None,[NETUID,uid]).value
                    duration_in_blocks = current_block - registration_block
                    duration_seconds = duration_in_blocks * block_time_seconds
                    duration_hours = duration_seconds / 3600
                    ages.append(duration_hours)

                # Combine the data into a list of tuples
                data_to_update = list(zip(active_hotkeys, active_coldkeys, active_uids, ages))
                db.insert_or_update_miner_coldkeys_and_ages(data_to_update)

            # In case of unforeseen errors, the api will log the error and continue operations.
            except Exception as err:
                print("Error during miner reg statuses sync", str(err))
                print_exception(type(err), err, err.__traceback__)

            await asyncio.sleep(300)

    async def authenticate_user(
        request: Request,
        api_key: Optional[str] = Depends(get_api_key, use_cache=False),
        #credentials: Optional[HTTPBasicCredentials] = Depends(security)
    ):
        """Unified authentication function that allows access via either API key or hotkey."""
        # API Key is valid, allow access
        if api_key:
            return {"authenticated_by": "api_key", "user": api_key}

        # Try to get HTTP Basic credentials manually
        credentials: Optional[HTTPBasicCredentials] = await security(request)
        hotkey = get_hotkey(credentials) if credentials else None

        if hotkey and authenticate_with_bittensor(hotkey, metagraph):
            return {"authenticated_by": "hotkey", "user": hotkey}

        # If neither authentication method is valid, deny access
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Valid API key or hotkey required."
        )

    @app.get("/")
    def healthcheck():
        return {"status": "ok", "message": datetime.utcnow()}

    @app.get("/matches")
    def get_matches():
        current_time = datetime.now()
        
        try:
            # Use cached data if it exists and is still valid
            if (matches_cache["data"] is not None and 
                matches_cache["last_updated"] is not None and
                current_time - matches_cache["last_updated"] < timedelta(seconds=MATCHES_CACHE_TTL)):
                logging.info("-- Returning cached matches data.")
                return {"matches": matches_cache["data"]}
            
            # Cache expired or doesn't exist, fetch fresh data
            match_list = db.get_matches()
            
            # Update cache
            matches_cache["data"] = match_list
            matches_cache["last_updated"] = current_time
            
            if match_list:
                return {"matches": match_list}
            else:
                return {"matches": []}
                
        except Exception as e:
            logging.error(f"Error retrieving matches: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.get("/matches/all")
    def get_all_matches():
        try:
            match_list = db.get_matches(all=True)
            if match_list:
                return {"matches": match_list}
            else:
                return {"matches": []}
        except Exception as e:
            logging.error(f"Error retrieving matches: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/matches/upcoming")
    def get_upcoming_matches():
        try:
            match_list = db.get_upcoming_matches()
            if match_list:
                return {"matches": match_list}
            else:
                return {"matches": []}
        except Exception as e:
            logging.error(f"Error retrieving matches: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/get-match")
    async def get_match(id: str):
        try:
            match = db.get_match_by_id(id)
            if match:
                # Apply datetime serialization to all fields in the dictionary that need it
                match = {key: serialize_datetime(value) for key, value in match.items()}
                return match
            else:
                return {"message": "No match found for the given ID."}
        except Exception as e:
            logging.error(f"Error retrieving get-match: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/matchOdds")
    async def get_match_odds(
        matchId: Optional[str] = None,
        auth: dict = Depends(authenticate_user),
    ):
        """Allows access if authenticated via either an API key or hotkey.
        user = auth["user"]
        auth_method = auth["authenticated_by"]
        
        # If authenticated via hotkey, enforce metagraph validation
        if auth_method == "hotkey":
            uid = metagraph.hotkeys.index(user)
        else:
            uid = None  # Not needed for API key access
        """

        try:
            match_odds = db.get_match_odds_by_id(matchId)
            if match_odds:
                return {"match_odds": match_odds}
            else:
                return {"match_odds": []}
        except Exception as e:
            logging.error(f"Error retrieving match odds by match id: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.get("/teamRecords")
    async def get_team_records(
        team_name: Optional[str] = None,
        league: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        #auth: dict = Depends(authenticate_user),
    ):
        try:
            team_records = db.get_team_records(team_name, league, start_date, end_date)
            if team_records:
                return {"records": team_records}
            else:
                return {"records": []}
        except Exception as e:
            logging.error(f"Error retrieving team records: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/predictionEdgeResults")
    async def upload_prediction_edge_results(
        prediction_edge_results: dict = Body(...),
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
        background_tasks: BackgroundTasks = None,
    ):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # get uid of bittensor validator
        uid = metagraph.hotkeys.index(hotkey)

        try:
            background_tasks.add_task(db.upload_prediction_edge_results, prediction_edge_results)

            """
            result = db.upload_prediction_edge_results(prediction_edge_results)
            if result:
                return {
                    "message": "Prediction edge results uploaded successfully from validator "
                    + str(uid)
                }
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to upload prediction edge results from validator "
                    + str(uid)
                )
            """
            
            return {
                "message": "Prediction edge results upload scheduled from validator "
                + str(uid)
            }
        
        except Exception as e:
            logging.error(f"Error posting predictionEdgeResults: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.get("/predictionEdgeResults")
    async def get_prediction_edge_results(
        vali_hotkey: str,
        miner_hotkey: Optional[str] = None,
        miner_id: Optional[int] = None,
        league: Optional[str] = None,
        date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_deregistered: Optional[bool] = None,
        count: Optional[int] = None,
    ):
        try:
            results = db.get_prediction_edge_results(vali_hotkey, miner_hotkey, miner_id, league, date, end_date, include_deregistered, count)

            if results:
                return {"results": results}
            else:
                return {"results": []}
        except Exception as e:
            logging.error(f"Error retrieving predictionEdgeResults: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
    
    @app.post("/scoredPredictions")
    async def upload_scored_predictions(
        predictions: dict = Body(...),
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
    ):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # get uid of bittensor validator
        uid = metagraph.hotkeys.index(hotkey)

        try:
            result = db.upload_scored_predictions(predictions, hotkey)
            if result:
                return {
                    "message": "Scored predictions uploaded successfully from validator "
                    + str(uid)
                }
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to upload scored predictions from validator "
                    + str(uid)
                )
        except Exception as e:
            logging.error(f"Error posting scoredPredictions: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/predictionResults")
    async def get_prediction_results(
        vali_hotkey: str,
        miner_hotkey: Optional[str] = None,
        league: Optional[str] = None,
    ):
        try:
            results = db.get_prediction_results_by_league(vali_hotkey, league, miner_hotkey)

            if results:
                return {"results": results}
            else:
                return {"results": []}
        except Exception as e:
            logging.error(f"Error retrieving predictionResultsPerMiner: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
    
    @app.post("/admin/clear-matches-cache")
    def clear_matches_cache():
        matches_cache["data"] = None
        matches_cache["last_updated"] = None
        return {"message": "Matches cache cleared successfully"}

    def serialize_datetime(value):
        """Serialize datetime to JSON-compatible format, if necessary."""
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    await asyncio.gather(
        resync_metagraph(),
        asyncio.to_thread(
            uvicorn.run,
            app,
            host="0.0.0.0",
            port=443,
            ssl_certfile="/root/origin-cert.pem",
            ssl_keyfile="/root/origin-key.key",
        ),
        resync_miner_statuses(),
    )


if __name__ == "__main__":
    asyncio.run(main())


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
