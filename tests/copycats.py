import logging
import time
import json
from typing import List, Set
from pathlib import Path
import requests

from storage.sqlite_validator_storage import get_storage
import bittensor

from common.data import League, MatchPrediction, MatchPredictionWithMatchData
from common.constants import (
    ACTIVE_LEAGUES,
    ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE,
)

from vali_utils.copycat_controller import CopycatDetectionController

    
def fetch_prediction_results(
    base_url: str,
    miner_id: int,
    miner_hotkey: str,
    vali_hotkey: str,
    force_api_fetch: bool = False
) -> List[MatchPredictionWithMatchData]:
    """
    Fetch prediction results from local cache or API and convert them to MatchPrediction objects.
    
    Args:
        base_url: str - The base URL for the API
        miner_id: int - The miner's ID
        miner_hotkey: str - The miner's hotkey
        vali_hotkey: str - The validator's hotkey
        force_api_fetch: bool - If True, fetch from API even if cache exists
    
    Returns:
        List[MatchPrediction] - List of MatchPrediction objects
        
    Raises:
        requests.RequestException: If API request fails
        json.JSONDecodeError: If response is not valid JSON
        KeyError: If expected fields are missing from response
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path('.api-json')
    cache_dir.mkdir(exist_ok=True)
    
    # Define cache file path for this miner
    cache_file = cache_dir / f"miner_{miner_id}_{miner_hotkey[:8]}.json"
    
    # Try to load from cache if allowed
    if not force_api_fetch and cache_file.exists():
        try:
            with cache_file.open('r') as f:
                response_data = json.load(f)
                return _parse_prediction_data(response_data, miner_id)
        except Exception as e:
            logging.warning(f"Failed to load from cache: {e}. Falling back to API.")
    
    # Fetch from API if needed
    try:
        # Construct API URL
        url = f"{base_url}/predictionResultsPerMiner"
        params = {
            "miner_hotkey": miner_hotkey,
            "vali_hotkey": vali_hotkey
        }
        
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse response JSON
        response_data = response.json()
        
        # Save to cache
        try:
            with cache_file.open('w') as f:
                json.dump(response_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to write to cache: {e}")
        
        time.sleep(1)  # Add delay to avoid rate limiting
        return _parse_prediction_data(response_data, miner_id)
        
    except requests.RequestException as e:
        logging.error(f"API request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse API response: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def _parse_prediction_data(response_data: dict, miner_id: int) -> List[MatchPredictionWithMatchData]:
    """
    Parse prediction data from either API response or cached file.
    
    Args:
        response_data: dict - The JSON response data
        miner_id: int - The miner's ID
    
    Returns:
        List[MatchPredictionWithMatchData] - List of parsed predictions
    """
    predictions = []
    for result in response_data.get("results", []):
        try:
            # Parse the nested JSON string in the data field
            predictions_data = json.loads(result["data"])
            
            # Convert each prediction dict to Prediction object
            for pred_dict in predictions_data:
                try:
                    prediction = MatchPrediction(**pred_dict)
                    prediction.minerId = miner_id
                    pwmd = MatchPredictionWithMatchData(
                        prediction=prediction,
                        actualHomeTeamScore=0,
                        actualAwayTeamScore=0,
                        homeTeamOdds=0,
                        awayTeamOdds=0,
                        drawOdds=0
                    )
                    predictions.append(pwmd)
                except (TypeError, ValueError) as e:
                    logging.warning(f"Failed to parse prediction: {e}")
                    continue
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse data field for miner {result.get('miner_id')}: {e}")
            continue
        except KeyError as e:
            logging.error(f"Missing required field in result: {e}")
            continue
    
    return predictions
    
def get_predictions_from_api(miner_uid, miner_hotkey, league):
    BASE_URL = "https://api.sportstensor.com"
    VALI_HOTKEY = "5GKH9FPPnWSUoeeTJp19wVtd84XqFW4pyK2ijV2GsFbhTrP1"
    
    try:
        # Fetch all predictions
        predictions = fetch_prediction_results(BASE_URL, miner_uid, miner_hotkey, VALI_HOTKEY)
        #print(f"Successfully fetched {len(predictions)} predictions for miner {miner_uid}")
        
        # Filter predictions by league
        league_predictions = [p for p in predictions if p.prediction.league == league]
        #print(f"-- Found {len(league_predictions)} predictions for league {league.name}")

        return league_predictions
            
    except Exception as e:
        print(f"Error fetching predictions: {e}")

def main_api():
    # Initialize our subtensor and metagraph
    NETWORK = None # "test" or None
    NETUID = 41
    if NETWORK == "test":
        NETUID = 172
    
    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)
    all_uids = metagraph.uids.tolist()

    # Initialize controller
    controller = CopycatDetectionController()
    
    # Optional: Limit analysis to specific leagues or miners for testing
    # test_leagues = [League.NFL]
    # test_miners = controller.metagraph.uids.tolist()[:10]
    
    # Run analysis
    leagues = ACTIVE_LEAGUES
    final_duplicates = set()
    final_penalties = set()

    for league in leagues:
        league_predictions = []
        for index, uid in enumerate(all_uids):
            miner_hotkey = metagraph.hotkeys[uid]
            predictions = get_predictions_from_api(uid, miner_hotkey, league)
            if not predictions:
                continue
            league_predictions.extend(predictions)
        
        duplicates, penalties = controller.analyze_league(league, league_predictions)
        final_duplicates.update(duplicates)
        final_penalties.update(penalties)

    # Print final results
    print(f"\n==============================================================================")
    print(f"Total miners with duplicates across all leagues: {len(final_duplicates)}")
    print(f"Miners: {', '.join(str(m) for m in sorted(final_duplicates))}")

    print(f"\nTotal miners to penalize across all leagues: {len(final_penalties)}")
    print(f"Miners: {', '.join(str(m) for m in sorted(final_penalties))}")
    print(f"==============================================================================")

def main():
    # Initialize our subtensor and metagraph
    NETWORK = None # "test" or None
    NETUID = 41
    if NETWORK == "test":
        NETUID = 172
    
    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)
    all_uids = metagraph.uids.tolist()

    # Initialize database to get data
    storage = get_storage()

    # Initialize controller
    controller = CopycatDetectionController()
    
    # Optional: Limit analysis to specific leagues or miners for testing
    # test_leagues = [League.NFL]
    # test_miners = controller.metagraph.uids.tolist()[:10]
    
    # Run analysis
    leagues = ACTIVE_LEAGUES
    final_duplicates = set()
    final_penalties = set()

    for league in leagues:
        league_predictions = []
        for index, uid in enumerate(all_uids):
            miner_hotkey = metagraph.hotkeys[uid]

            predictions_with_match_data = storage.get_miner_match_predictions(
                miner_hotkey=miner_hotkey,
                miner_uid=uid,
                league=league,
                scored=True,
                batchSize=(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * 2)
            )

            if not predictions_with_match_data:
                continue  # No predictions for this league, keep score as 0

            league_predictions.extend(predictions_with_match_data)

        
        duplicates, penalties = controller.analyze_league(league, league_predictions)
        final_duplicates.update(duplicates)
        final_penalties.update(penalties)

    # Print final results
    print(f"\n==============================================================================")
    print(f"Total miners with duplicates across all leagues: {len(final_duplicates)}")
    print(f"Miners: {', '.join(str(m) for m in sorted(final_duplicates))}")

    print(f"\nTotal miners to penalize across all leagues: {len(final_penalties)}")
    print(f"Miners: {', '.join(str(m) for m in sorted(final_penalties))}")
    print(f"==============================================================================")

if __name__ == "__main__":
    #main()
    main_api()
