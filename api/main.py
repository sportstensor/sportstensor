from typing import Annotated, List, Optional
from traceback import print_exception

import bittensor
import uvicorn
import asyncio
import logging
import random
from fastapi import FastAPI, HTTPException, Depends, Body, Path, Security
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

from datetime import datetime
import api.db as db
from api.config import NETWORK, NETUID, IS_PROD, API_KEYS

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


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is not None and api_key_header in API_KEYS:
        return api_key_header
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )


def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    keypair = Keypair(ss58_address=credentials.username)

    if keypair.verify(credentials.username, credentials.password):
        return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Signature mismatch",
    )


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


# Get a random active validator hotkey with vTrust >= 0.9
def get_active_vali_hotkey(metagraph):
    avail_uids = []
    for uid in range(metagraph.n.item()):
        if metagraph.validator_permit[uid]:
            avail_uids.append(uid)

    vali_vtrusts = [(uid, metagraph.hotkeys[uid], metagraph.Tv[uid].item()) for uid in avail_uids if metagraph.Tv[uid] >= 0.8 and metagraph.active[uid] == 1]

    if len(vali_vtrusts) == 0:
        print("No active validators with vTrust >= 0.8 found.")
        return None
    
    # Get the hotkey of a random validator with vTrust >= 0.8
    random_vali_hotkey = random.choice(vali_vtrusts)[1]
    
    return random_vali_hotkey


async def main():
    app = FastAPI()

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

            await asyncio.sleep(90)

    @app.get("/")
    def healthcheck():
        return {"status": "ok", "message": datetime.utcnow()}

    @app.get("/matches")
    # def get_matches(hotkey: Annotated[str, Depends(get_hotkey)]):
    def get_matches():
        match_list = db.get_matches()
        if match_list:
            return {"matches": match_list}
        else:
            return {"error": "Failed to retrieve match data."}

    @app.get("/get-match")
    async def get_match(id: str):
        match = db.get_match_by_id(id)
        if match:
            # Apply datetime serialization to all fields in the dictionary that need it
            match = {key: serialize_datetime(value) for key, value in match.items()}
            return match
        else:
            return {"error": "Failed to retrieve match data."}

    @app.get("/get-prediction")
    async def get_prediction(id: str):
        print(
            f"API called with id: {id}"
        )  # Print statement to confirm the endpoint is hit
        logging.info(f"API called with id: {id}")

        prediction = db.get_prediction_by_id(id)
        if prediction:
            # Apply datetime serialization to all fields in the dictionary that need it
            prediction = {
                key: serialize_datetime(value) for key, value in prediction.items()
            }
            return prediction
        else:
            return {"error": "Failed to retrieve prediction data."}

    @app.post("/AddAppPrediction")
    async def upsert_app_prediction(api_key: str = Security(get_api_key), prediction: dict = Body(...)):
        vali_hotkey = None
        for attempt in range(10):
            # Get a valid validator hotkey with vTrust >= 0.8
            vali_hotkey = get_active_vali_hotkey(metagraph)
            if vali_hotkey is not None:
                print(f"Random active validator hotkey with vTrust >= 0.8: {vali_hotkey}")
                break
            print(f"Attempt {attempt + 1} failed to get a valid hotkey.")
        else:
            return {"message": "Failed to find a valid validator hotkey after 10 attempts"}

        result = db.upsert_app_match_prediction(prediction, vali_hotkey)
        return {"message": "Prediction upserted successfully"}

    @app.get("/AppMatchPredictions")
    def get_app_match_predictions(api_key: str = Security(get_api_key)):
        predictions = db.get_app_match_predictions()
        if predictions:
            return {"requests": predictions}
        else:
            return {"error": "Failed to retrieve match predictions data."}
        
    @app.get("/AppMatchPredictionsForValidators")
    def get_app_match_predictions(hotkey: Annotated[str, Depends(get_hotkey)] = None,):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        # Get a batch of 10 match predictions from the app for the calling validator
        predictions = db.get_app_match_predictions(hotkey, 10)
        if predictions:
            return {"requests": predictions}
        else:
            return {"error": "Failed to retrieve match predictions data."}
        
    @app.post("/AppMatchPredictionsForValidators")
    def upload_app_match_predictions(
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
        predictions: List[dict] = Body(...),
    ):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )

    @app.post("/predictionResults")
    async def upload_prediction_results(
        prediction_results: dict = Body(...),
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

        result = db.upload_prediction_results(prediction_results)
        return {
            "message": "Prediction results uploaded successfully from validator "
            + str(uid)
        }

    @app.get("/predictionResults")
    async def get_prediction_results(
        miner_hotkey: Optional[str] = None,
        sport: Optional[str] = None,
        league: Optional[str] = None,
    ):
        if league is not None:
            results = db.get_prediction_stats_by_league(league, miner_hotkey)
        elif sport is not None:
            results = db.get_prediction_stats_by_sport(sport, miner_hotkey)
        else:
            results = db.get_prediction_stats_total(miner_hotkey)

        if results:
            return {"results": results}
        else:
            return {"error": "Failed to retrieve prediction results data."}

    @app.get("/predictionResultsPerMiner")
    async def get_prediction_results_per_miner(
        miner_hotkey: Optional[str] = None,
        sport: Optional[str] = None,
        league: Optional[str] = None,
    ):
        if league is not None:
            results = db.get_prediction_stats_by_league(league, miner_hotkey, True)
        elif sport is not None:
            results = db.get_prediction_stats_by_sport(sport, miner_hotkey, True)
        else:
            results = db.get_prediction_stats_total(miner_hotkey, True)

        if results:
            return {"results": results}
        else:
            return {"error": "Failed to retrieve prediction results data."}

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
    )


if __name__ == "__main__":
    asyncio.run(main())


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
