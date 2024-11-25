from typing import Annotated, List, Optional
from pydantic import conint, BaseModel
from traceback import print_exception
import subprocess
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
import os
from fastapi import FastAPI
import sentry_sdk
import socket

# mysqlclient install issues: https://stackoverflow.com/a/77020207
import mysql.connector
from mysql.connector import Error

from datetime import datetime
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

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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


# Get a random active validator hotkey with vTrust >= 0.8
def get_active_vali_hotkey(metagraph, exclude_hotkeys=[]):
    avail_uids = []

    if NETWORK == "test":
        # Get the hotkeys of all testnet validators
        avail_hotkeys = [hotkey for hotkey in TESTNET_VALI_HOTKEYS]
        return random.choice(avail_hotkeys)

    for uid in range(metagraph.n.item()):
        if metagraph.validator_permit[uid] and metagraph.hotkeys[uid] not in exclude_hotkeys:
            avail_uids.append(uid)

    vali_vtrusts = [(uid, metagraph.hotkeys[uid], metagraph.Tv[uid].item()) for uid in avail_uids if metagraph.Tv[uid] >= 0.8 and metagraph.active[uid] == 1]

    if len(vali_vtrusts) == 0:
        print("No active validators with vTrust >= 0.8 found.")
        return None
    
    # Get the hotkey of a random validator with vTrust >= 0.8
    random_vali_hotkey = random.choice(vali_vtrusts)[1]
    
    return random_vali_hotkey

def find_available_port():
    """Find an available port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to an available port
        return s.getsockname()[1]  # Return the port number

def run_btcli_command(command):
    try:
        # Execute the command
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred when runing btcli: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate wallet {e.stderr}")

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

                # # Update the miner registration statuses
                # db.update_miner_reg_statuses(active_uids, active_hotkeys)

                # Combine the data into a list of tuples
                data_to_update = list(zip(active_hotkeys, active_coldkeys, active_uids, ages))
                db.insert_or_update_miner_coldkeys_and_ages(data_to_update)

            # In case of unforeseen errors, the api will log the error and continue operations.
            except Exception as err:
                print("Error during miner reg statuses sync", str(err))
                print_exception(type(err), err, err.__traceback__)

            await asyncio.sleep(300)

    async def check_vali_app_match_prediction_requests():
        if not ENABLE_APP:
            return
        while True:
            """Checks if any AppMatchPredictions have NOT been picked up by a validator within APP_PREDICTIONS_UNFULFILLED_THRESHOLD minutes."""
            print("check_vali_app_match_prediction_requests()")

            try:
                requests = db.get_app_match_predictions_unfulfilled(APP_PREDICTIONS_UNFULFILLED_THRESHOLD)
                if requests:
                    print(f"Unfulfilled app match prediction requests: {requests}")
                    for request in requests:
                        vali_hotkey = None
                        for attempt in range(10):
                            # Get a valid validator hotkey with vTrust >= 0.8
                            vali_hotkey = get_active_vali_hotkey(metagraph, exclude_hotkeys=[request["vali_hotkey"]])
                            if vali_hotkey is not None:
                                print(f"Random active validator hotkey with vTrust >= 0.8: {vali_hotkey}")
                                break
                            print(f"Attempt {attempt + 1} failed to get a valid hotkey.")
                        else:
                            return {"message": "Failed to find a valid validator hotkey after 10 attempts"}
                        
                        if vali_hotkey is not None:
                            print(f"Random active validator hotkey with vTrust >= 0.8: {vali_hotkey}")
                            db.upsert_app_match_prediction(request, vali_hotkey)
                        else:
                            print("Failed to find a valid validator hotkey.")

            # In case of unforeseen errors, the api will log the error and continue operations.
            except Exception as err:
                print("Error checking unfulfilled app match predictions", str(err))
                print_exception(type(err), err, err.__traceback__)

            await asyncio.sleep(10)

    @app.get("/")
    def healthcheck():
        return {"status": "ok", "message": datetime.utcnow()}

    @app.get("/matches")
    def get_matches():
        try:
            match_list = db.get_matches()
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
    async def get_match_odds(matchId: Optional[str] = None):
        try:
            match_odds = db.get_match_odds_by_id(matchId)
            if match_odds:
                return {"match_odds": match_odds}
            else:
                return {"match_odds": []}
        except Exception as e:
            logging.error(f"Error retrieving match odds by match id: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/get-prediction")
    async def get_prediction(id: str):
        try:
            prediction = db.get_prediction_by_id(id)
            if prediction:
                # Apply datetime serialization to all fields in the dictionary that need it
                prediction = {
                    key: serialize_datetime(value) for key, value in prediction.items()
                }
                return prediction
            else:
                return {"message": "No prediction found for the given ID."}
        except Exception as e:
            logging.error(f"Error retrieving get-prediction: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.post("/get-predictions")
    async def get_predictions(api_key: str = Security(get_api_key), prediction_ids: dict = Body(...)):
        try:
            if "ids" not in prediction_ids:
                return {"message": "No prediction IDs provided."}
            predictions = db.get_app_match_predictions_by_ids(prediction_ids["ids"])
            if predictions:
                return {"requests": predictions}
            else:
                return {"requests": []}
        except Exception as e:
            logging.error(f"Error retrieving get-predictions: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

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

        try:
            result = db.upsert_app_match_prediction(prediction, vali_hotkey)
            if result:
                return {"message": "Prediction upserted successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to upsert app prediction request.")
        except Exception as e:
            logging.error(f"Error upserting in AddAppPrediction: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/AppMatchPredictions")
    def get_app_match_predictions(api_key: str = Security(get_api_key)):
        try:
            predictions = db.get_app_match_predictions()
            if predictions:
                return {"requests": predictions}
            else:
                return {"requests": []}
        except Exception as e:
            logging.error(f"Error retrieving AppMatchPredictions: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.get("/AppMatchPredictionsForValidators")
    def get_app_match_predictions(hotkey: Annotated[str, Depends(get_hotkey)] = None,):
        if not authenticate_with_bittensor(hotkey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotkey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        try:
            # Get a batch of 10 match predictions from the app for the calling validator
            predictions = db.get_app_match_predictions(hotkey, 10)
            if predictions:
                return {"requests": predictions}
            else:
                return {"message": "No prediction requests found for the calling validator."}
        except Exception as e:
            logging.error(f"Error retrieving AppMatchPredictionsForValidators: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.post("/AppMatchPredictionsForValidators")
    def update_app_match_predictions(
        hotkey: Annotated[str, Depends(get_hotkey)] = None,
        predictions: List[dict] = Body(...),
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
            result = db.update_app_match_predictions(predictions)
            if result:
                return {
                    "message": "Prediction results uploaded successfully from validator "
                    + str(uid)
                }
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to update app prediction requests from validator "
                    + str(uid)
                )
        except Exception as e:
            logging.error(f"Error posting AppMatchPredictionsForValidators: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/predictionEdgeResults")
    async def upload_prediction_edge_results(
        prediction_edge_results: dict = Body(...),
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
        except Exception as e:
            logging.error(f"Error posting predictionEdgeResults: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
        
    @app.get("/predictionEdgeResults")
    async def get_prediction_edge_results(
        vali_hotkey: str,
        miner_hotkey: Optional[str] = None,
        miner_id: Optional[int] = None,
    ):
        try:
            results = db.get_prediction_edge_results(vali_hotkey, miner_hotkey, miner_id)

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

    class PredictionRequest(BaseModel):
        homeTeamName: str
        awayTeamName: str
        matchDate: str
        league: str

    @app.post("/inference")
    async def retreive_model_inference_result(request: PredictionRequest):
        try:
            # Log input data
            logging.info("Received prediction request: %s", request.dict())

            # Set up their model inference
            # prediction = make_prediction(homeTeamName, awayTeamName, matchDate, league)
            # return {
            #     'choice': prediction.choice,
            #     'probability': prediction.probability
            # }
            return {
                'choice': 'HomeTeam',
                'probability': 0.2
            }
        except Exception as e:
            logging.error(f"Error inferencing models: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")
    class SetupRequest(BaseModel):
        coldKey: str
        hotKey: str
        hotKeyMnemonic: str
        externalAPI: str
        minerId: int
        league_committed: str

    @app.post("/setup-miner")
    async def setup_miner_for_non_builder(request: SetupRequest):
        # Log input data
        logging.info("Received setup-miner request: %s", request.dict())
        setupRequest = request.dict()
        coldKey = setupRequest.get('coldKey')
        hotKey = setupRequest.get('hotKey')
        hotKeyMnemonic = setupRequest.get('hotKeyMnemonic')
        externalAPI = setupRequest.get('externalAPI')
        minerId = setupRequest.get('minerId')
        league_committed = setupRequest.get('league_committed')
        if not authenticate_with_bittensor(hotKey, metagraph):
            print(f"Valid hotkey required, returning 403. hotkey: {hotKey}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        try:
            
            # Find an available port
            port = find_available_port()

            # Construct the absolute path to the miner.py script
            script_path = os.path.join(os.path.dirname(__file__), '../neurons/miner.py')
            # Create a wallet on the server and run pm2 instance
            wallet_name = f"auto-gen-wallet-{coldKey}"
            wallet_hotkey_name = f"auto-gen-wallet-hotkey-{hotKey}"
            regen_coldkeypub_command = f"btcli wallet regen_coldkeypub --ss58_address {coldKey} --wallet.name {wallet_name}"
            regen_hotkey_command = f"btcli wallet regen_hotkey --wallet.name {wallet_name} --wallet.hotkey {wallet_hotkey_name} --mnemonic {hotKeyMnemonic} --no-use-password"
            isWalletAlreadyGenerated = db.walletAlreadyGenerated(coldKey)
            if not isWalletAlreadyGenerated:
                coldkey_pub_output = run_btcli_command(regen_coldkeypub_command)
                if coldkey_pub_output:
                    logging.info(f"Wallet regenerated: {wallet_name}")
                    # Regenerate hotkey
                    hotkey_output = run_btcli_command(regen_hotkey_command)
                    logging.info(f"Wallet Hotkey regenerated: {wallet_name}")
                    if hotkey_output is None:
                        raise HTTPException(status_code=500, detail=f"Failed to regenerate hotkey wallet")
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to regenerate wallet")
            else:
                logging.info(f"Wallet already generated: {wallet_name}")
                isWalletHotkeyAlreadyGenerated = db.walletAlreadyGenerated(coldKey, hotKey, hotKeyMnemonic)
                if not isWalletHotkeyAlreadyGenerated:
                    # Regenerate hotkey
                    hotkey_output = run_btcli_command(regen_hotkey_command)
                    if hotkey_output is None:
                        raise HTTPException(status_code=500, detail=f"Failed to regenerate hotkey wallet")
                else:
                    logging.info(f"Wallet hotkey already generated: {wallet_hotkey_name}")

            # Determine netuid and subtensor.network based on environment
            netuid = "41" if IS_PROD else "172"
            subtensor_network = "finney" if IS_PROD else "test"
            # Prepare the pm2 command
            pm2_command = [
                "pm2", "start", script_path,
                "--name", f"Sportstensor-{minerId}",
                "--", "--netuid", netuid,
                "--subtensor.network", subtensor_network,
                "--wallet.name", wallet_name,
                "--wallet.hotkey", wallet_hotkey_name,
                "--axon.port", str(port),
                "--blacklist.validator_min_stake", "0",
                "--non_builder_miner_id", str(minerId)
            ]
            
            # Run the pm2 command and capture output
            result = subprocess.run(pm2_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Log the output
            logging.info(f"pm2 command output: {result.stdout.decode().strip()}")
            logging.info(f"pm2 command error (if any): {result.stderr.decode().strip()}")

            result = db.storeDataForNonBuilderMiner(minerId, coldKey, hotKey, hotKeyMnemonic, externalAPI, league_committed, port)
            if result:
                logging.info(f"Stored non-builder miner{minerId}(hotkey-{hotKey})'s neuron data")
                return {"status": "success", "detail": f"Starting miner on port {port}"}
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running pm2 command: {e}")
            logging.error(f"Command output: {e.stdout.decode().strip()}")
            logging.error(f"Command error output: {e.stderr.decode().strip()}")
            raise HTTPException(status_code=500, detail=f"Failed to start miner process. {e.stderr.decode().strip()}")
        except Exception as e:
            logging.error(f"Error setting up and running miner instance: {e}")
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

    @app.get("/predictionResultsPerMiner")
    async def get_prediction_results_per_miner(
        vali_hotkey: str,
        miner_hotkey: Optional[str] = None,
        league: Optional[str] = None,
        cutoff: Optional[int] = None,
        include_deregged: Optional[conint(ge=0, le=1)] = None
    ):
        try:
            results = db.get_prediction_stats_by_league(vali_hotkey, league, miner_hotkey, cutoff, include_deregged)

            if results:
                return {"results": results}
            else:
                return {"results": []}
        except Exception as e:
            logging.error(f"Error retrieving predictionResults: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")    
    
    @app.get("/predictionResultsSnapshots")
    async def get_prediction_results_snapshots(
        miner_hotkey: Optional[str] = None,
        sport: Optional[str] = None,
        league: Optional[str] = None,
    ):
        try:
            results = db.get_prediction_stat_snapshots(sport, league, miner_hotkey)

            if results:
                return {"results": results}
            else:
                return {"results": []}
        except Exception as e:
            logging.error(f"Error retrieving predictionResultsSnapshots: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

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
        check_vali_app_match_prediction_requests(),
    )


if __name__ == "__main__":
    asyncio.run(main())


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
