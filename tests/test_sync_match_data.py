import asyncio
from aiohttp import ClientSession
import logging
from storage.sqlite_validator_storage import SqliteValidatorStorage  # Ensure correct import path

logging.basicConfig(level=logging.INFO)

API_ENDPOINT = "http://95.179.153.99:8000/matches"  # Use a test endpoint if necessary

async def sync_match_data(match_data_endpoint) -> bool:
    storage = SqliteValidatorStorage()  
    try:
        async with ClientSession() as session:
            async with session.get(match_data_endpoint) as response:
                response.raise_for_status()
                match_data = await response.json()
        
        if not match_data:
            logging.info("No match data returned from API")
            return False
        
        # UPSERT logic
        matches_to_insert = [match for match in match_data if not storage.check_match(match['matchId'])]
        matches_to_update = [match for match in match_data if storage.check_match(match['matchId'])]

        if matches_to_insert:
            storage.insert_matches(matches_to_insert)
            logging.info(f"Inserted {len(matches_to_insert)} new matches.")
        if matches_to_update:
            storage.update_matches(matches_to_update)
            logging.info(f"Updated {len(matches_to_update)} existing matches.")

        return True

    except Exception as e:
        logging.error(f"Error getting match data: {e}")
        return False

async def test_sync_match_data():
    result = await sync_match_data(API_ENDPOINT)
    print(f"Test result: {'Success' if result else 'Failure'}")

if __name__ == "__main__":
    asyncio.run(test_sync_match_data())
