# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 sportstensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os
from typing import List, Dict
import datetime as dt
import time
import asyncio
import threading
import traceback
import requests

# Bittensor
import bittensor as bt
import wandb

# Bittensor Validator Template:
from common.protocol import GetLeagueCommitments, GetMatchPrediction
from common.data import League, get_league_from_string
from common.constants import (
    ENABLE_APP,
    DATA_SYNC_INTERVAL_IN_MINUTES,
    APP_DATA_SYNC_INTERVAL_IN_MINUTES,
    PURGE_DEREGGED_MINERS_INTERVAL_IN_MINUTES,
    MAX_BATCHSIZE_FOR_SCORING,
    SCORING_INTERVAL_IN_MINUTES,
    ACTIVE_LEAGUES,
    LEAGUE_COMMITMENT_INTERVAL_IN_MINUTES,
    LEAGUE_SCORING_PERCENTAGES,
    ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE,
    LEAGUE_SENSITIVITY_ALPHAS,
    SENSITIVITY_ALPHA,
    GAMMA,
    TRANSITION_KAPPA,
    EXTREMIS_BETA,
    PARETO_MU,
    PARETO_ALPHA,
)
import vali_utils.utils as utils
import vali_utils.scoring_utils as scoring_utils

# import base validator class which takes care of most of the boilerplate
from base.validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.wandb_run_start = None
        if not self.config.wandb.off:
            if os.getenv("WANDB_API_KEY"):
                self.new_wandb_run()
            else:
                bt.logging.exception(
                    "WANDB_API_KEY not found. Set it with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with --wandb.off, but it is strongly recommended to run with W&B enabled."
                )
                self.config.wandb.off = True
        else:
            bt.logging.warning(
                "Running with --wandb.off. It is strongly recommended to run with W&B enabled."
            )

        # Load league controls from CSV URL
        self.league_controls_url = (
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=1997764475&single=true&output=csv"
            if self.config.subtensor.network == "test"
            else "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=0&single=true&output=csv"
        )
        self.load_league_controls_start = dt.datetime.now()
        bt.logging.info(f"Loading league controls from URL: {self.league_controls_url}")
        self.load_league_controls()

        # Load scoring controls from CSV URL
        self.scoring_controls_url = (
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=1135799581&single=true&output=csv"
            if self.config.subtensor.network == "test"
            else "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJcedkDc0c3rijp6gX9eSiq1QDRpMlbiZMywPc3amzznLyiLSOqc6dfbz5Hd18dqPgQVbvp91NSCSE/pub?gid=420555119&single=true&output=csv"
        )
        self.load_scoring_controls_start = dt.datetime.now()
        bt.logging.info(f"Loading scoring controls from URL: {self.scoring_controls_url}")
        self.load_scoring_controls()

        api_root = (
            "https://dev-api.sportstensor.com"
            if self.config.subtensor.network == "test"
            else "https://api.sportstensor.com"
        )
        bt.logging.info(f"Using Sportstensor API: {api_root}")
        self.match_data_endpoint = f"{api_root}/matches"
        self.match_odds_endpoint = f"{api_root}/matchOdds"
        self.prediction_results_endpoint = f"{api_root}/predictionResults"
        self.prediction_edge_results_endpoint = f"{api_root}/predictionEdgeResults"
        self.scored_predictions_endpoint = f"{api_root}/scoredPredictions"
        self.app_prediction_requests_endpoint = f"{api_root}/AppMatchPredictionsForValidators"
        self.app_prediction_responses_endpoint = f"{api_root}/AppMatchPredictionsForValidators"
        
        self.next_match_syncing_datetime = dt.datetime.now(dt.timezone.utc)
        self.next_predictions_cleanup_datetime = dt.datetime.now(dt.timezone.utc)
        self.next_league_commitments_datetime = dt.datetime.now(dt.timezone.utc)
        self.next_scoring_datetime = dt.datetime.now(dt.timezone.utc)
        self.next_app_predictions_syncing_datetime = dt.datetime.now(dt.timezone.utc)
        
        self.accumulated_league_commitment_penalties = {}
        self.accumulated_league_commitment_penalties_lock = threading.RLock()

        # Create a set of uids to corresponding leagues that miners are committed to.
        self.uids_to_leagues: Dict[int, List[League]] = {}
        self.uids_to_leagues_lock = threading.RLock()
        self.uids_to_last_leagues: Dict[int, List[League]] = {}
        self.uids_to_last_leagues_lock = threading.RLock()
        self.uids_to_leagues_last_updated: Dict[int, dt.datetime] = {}
        self.uids_to_leagues_last_updated_lock = threading.RLock()

        # Initialize the incentive scoring and weight setting thread
        self.stop_event = threading.Event()
        self.weight_thread = threading.Thread(
            target=self.incentive_scoring_and_set_weights,
            args=(300,),
            daemon=True,
        )
        self.weight_thread.start()

    
    def incentive_scoring_and_set_weights(self, ttl: int):
        # Continually loop and execute at the 30-minute mark
        while not self.stop_event.is_set():
            current_time = dt.datetime.utcnow()
            minutes = current_time.minute
            hour = current_time.hour
            
            # Check if we're at a 30-minute mark
            if minutes % 30 == 0 or self.config.immediate:
                # Skip processing if we haven't loaded our league commitments yet
                if len(self.uids_to_leagues) == 0:
                    bt.logging.info("Skipping calculating incentives, updating scores, and setting weights. League commitments not loaded yet.")
                    time.sleep(60)
                    continue

                try:
                    bt.logging.info(
                        "*** Syncing the latest match odds data to local validator storage. ***"
                    )
                    sync_result = utils.sync_match_odds_data(self, self.match_odds_endpoint)
                    if sync_result:
                        bt.logging.info("Successfully synced match odds data.")
                    else:
                        bt.logging.warning("Issue syncing match odds data")
                except Exception as e:
                    bt.logging.error(f"Error syncing match odds: {str(e)}")

                try:
                    bt.logging.debug("Calculating incentives.")
                    (
                        league_scores,
                        league_edge_scores,
                        league_roi_scores,
                        league_roi_counts,
                        league_roi_payouts,
                        league_roi_market_payouts,
                        league_roi_incr_counts,
                        league_roi_incr_payouts,
                        league_roi_incr_market_payouts,
                        league_pred_counts,
                        league_pred_win_counts,
                        all_scores,
                    ) = scoring_utils.calculate_incentives_and_update_scores(self)
                    bt.logging.debug("Finished calculating incentives.")
                except Exception as e:
                    bt.logging.error(f"Error calculating incentives: {str(e)}")
                    bt.logging.error(f"Error details: {traceback.format_exc()}")

                try:
                    if not self.config.neuron.disable_set_weights and not self.config.offline:
                        bt.logging.debug("Setting weights.")
                        self.set_weights()
                        bt.logging.debug("Finished setting weights.")
                        if self.config.immediate:
                            time.sleep(3600)
                except asyncio.TimeoutError:
                    bt.logging.error(f"Failed to set weights after {ttl} seconds")

                try:
                    # Post scores to API after scoring. Overwrites current day's scores.
                    if (
                        league_scores and len(league_scores) > 0 and
                        ((self.config.subtensor.network == "test") or 
                        (self.config.subtensor.network != "test" and self.metagraph.validator_permit[self.uid] and self.metagraph.S[self.uid] >= 200_000))
                    ):
                        bt.logging.info("Posting league scores to API.")
                        post_result = utils.post_prediction_edge_results(
                            self,
                            self.prediction_edge_results_endpoint,
                            league_scores,
                            league_edge_scores,
                            league_roi_scores,
                            league_roi_counts,
                            league_roi_payouts,
                            league_roi_market_payouts,
                            league_roi_incr_counts,
                            league_roi_incr_payouts,
                            league_roi_incr_market_payouts,
                            league_pred_counts,
                            league_pred_win_counts,
                            all_scores
                        )

                except Exception as e:
                    bt.logging.error(f"Error posting league scores to API: {str(e)}")

            else:
                # only log every 5 minutes
                if minutes % 5 == 0:
                    bt.logging.debug(f"Skipping setting weights. Only set weights at 30-minute marks.")

            # sleep for 1 minute before checking again
            time.sleep(60)


    def validate_league_percentages(self, percentages: Dict[League, float]):
        total = sum(percentages.values())
        if not abs(total - 1.0) < 1e-9:  # Using a small epsilon for float comparison
            raise ValueError(f"LEAGUE_SCORING_PERCENTAGES do not sum to 1.0. Current sum: {total}")
        bt.logging.info("LEAGUE_SCORING_PERCENTAGES are valid and sum to 1.0")

    
    def load_league_controls(self):
        # get league controls from CSV URL and load them into our settings
        try:
            response = requests.get(self.league_controls_url)
            response.raise_for_status()

            # split the response text into lines
            lines = response.text.split("\n")
            # filter the lines to include only those where column C is "Active". Skip the first line which is the header.
            active_leagues = [
                line.split(",")[0].strip()
                for line in lines[1:]
                if line.split(",")[3].strip() == "Active"
            ]
            active_league_percentages = [
                float(line.split(",")[1].strip())
                for line in lines[1:]
                if line.split(",")[3].strip() == "Active"
            ]
            active_league_rolling_thresholds = [
                int(line.split(",")[2].strip())
                for line in lines[1:]
                if line.split(",")[3].strip() == "Active"
            ]
            active_league_sensitivity_alphas = [
                float(line.split(",")[6].strip())
                for line in lines[1:]
                if line.split(",")[3].strip() == "Active"
            ]
            self.ACTIVE_LEAGUES = [get_league_from_string(league) for league in active_leagues]
            self.LEAGUE_SCORING_PERCENTAGES = {
                get_league_from_string(league): percentage for league, percentage in zip(active_leagues, active_league_percentages)
            }
            self.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE = {
                get_league_from_string(league): rolling_threshold for league, rolling_threshold in zip(active_leagues, active_league_rolling_thresholds)
            }
            self.LEAGUE_SENSITIVITY_ALPHAS = {
                get_league_from_string(league): alpha for league, alpha in zip(active_leagues, active_league_sensitivity_alphas)
            }
            bt.logging.info("************ Setting active leagues ************")
            for league in self.ACTIVE_LEAGUES:
                bt.logging.info(f"  • {league}")
            bt.logging.info("************************************************")

            bt.logging.info("************ Setting leagues scoring percentages ************")
            for league, percentage in self.LEAGUE_SCORING_PERCENTAGES.items():
                bt.logging.info(f"  • {league}: {percentage*100}%")
            bt.logging.info("*************************************************************")

            bt.logging.info("************ Setting leagues rolling prediction thresholds ************")
            for league, rolling_threshold in self.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE.items():
                bt.logging.info(f"  • {league}: {rolling_threshold}")
            bt.logging.info("*************************************************************")

            bt.logging.info("************ Setting leagues sensitivity alphas ************")
            for league, alpha in self.LEAGUE_SENSITIVITY_ALPHAS.items():
                bt.logging.info(f"  • {league}: {alpha}")
            bt.logging.info("*************************************************************")

            # Validate the league scoring percentages to make sure we're good.
            self.validate_league_percentages(self.LEAGUE_SCORING_PERCENTAGES)
            bt.logging.info(f"Loaded league controls successfully.")

        except Exception as e:
            bt.logging.error(f"Error loading league controls from URL {self.league_controls_url}: {e}")
            bt.logging.info(f"Using fallback control constants.")
            self.ACTIVE_LEAGUES = ACTIVE_LEAGUES
            self.ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE = ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE
            self.LEAGUE_SENSITIVITY_ALPHAS = LEAGUE_SENSITIVITY_ALPHAS
            self.LEAGUE_SCORING_PERCENTAGES = LEAGUE_SCORING_PERCENTAGES
            # Validate the league scoring percentages to make sure we're good.
            self.validate_league_percentages(self.LEAGUE_SCORING_PERCENTAGES)
            


    def load_scoring_controls(self):
        # get scoring constant controls from CSV URL and load them into our settings
        try:
            response = requests.get(self.scoring_controls_url)
            response.raise_for_status()

            # split the response text into lines
            lines = response.text.split("\n")
            
            # Dictionary to store our constants
            constants = {}

            # Skip the header row
            for line in lines[1:]:
                # Split each line by comma and unpack
                constant, value, *_ = line.split(',')
                # Strip any quotation marks and convert value to float
                constants[constant.strip('"')] = float(value.strip('"'))

            # Set class attributes based on CSV data
            self.SENSITIVITY_ALPHA = constants.get('SENSITIVITY_ALPHA', SENSITIVITY_ALPHA)
            self.GAMMA = constants.get('GAMMA', GAMMA)
            self.TRANSITION_KAPPA = constants.get('TRANSITION_KAPPA', TRANSITION_KAPPA)
            self.EXTREMIS_BETA = constants.get('EXTREMIS_BETA', EXTREMIS_BETA)
            self.PARETO_MU = constants.get('PARETO_MU', PARETO_MU)
            self.PARETO_ALPHA = constants.get('PARETO_ALPHA', PARETO_ALPHA)

            bt.logging.info("************ Setting scoring controls ************")
            bt.logging.info(f" SENSITIVITY_ALPHA: {self.SENSITIVITY_ALPHA}")
            bt.logging.info(f" GAMMA: {self.GAMMA}")
            bt.logging.info(f" TRANSITION_KAPPA: {self.TRANSITION_KAPPA}")
            bt.logging.info(f" EXTREMIS_BETA: {self.EXTREMIS_BETA}")
            bt.logging.info(f" PARETO_MU: {self.PARETO_MU}")
            bt.logging.info(f" PARETO_ALPHA: {self.PARETO_ALPHA}")
            bt.logging.info("************************************************")

            bt.logging.info(f"Loaded scoring controls successfully.")

        except Exception as e:
            bt.logging.error(f"Error loading scoring controls from URL {self.scoring_controls_url}: {e}")
            bt.logging.info(f"Using fallback control constants.")
            self.SENSITIVITY_ALPHA = SENSITIVITY_ALPHA
            self.GAMMA = GAMMA
            self.TRANSITION_KAPPA = TRANSITION_KAPPA
            self.EXTREMIS_BETA = EXTREMIS_BETA


    def new_wandb_run(self):
        # Shoutout SN13 for the wandb snippet!
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        now = dt.datetime.now()
        self.wandb_run_start = now
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project="sportstensor-vali-logs",
            entity="sportstensor",
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "validator",
            },
            allow_val_change=True,
            anonymous="allow",
        )

        bt.logging.debug(f"Started a new wandb run: {name}")


    def get_miner_uids_committed_to_league(self, league: League) -> List[int]:
        """
        Returns a list of miner uids that are committed to a league.
        """
        with self.uids_to_leagues_lock:
            result = []
            for uid, leagues in self.uids_to_leagues.items():
                for l in leagues:
                    if league == l:
                        result.append(uid)
                        break
            return result
            

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Periodically updating match data.
        - Querying the miners for league commitments
        - Generating the prediction queries
        - Querying the miners for predictions
        - Getting the responses and validating predictions
        - Storing prediction responses
        - Scoring (calculating closing edge) on eligible past prediction responses

        The forward function is called by the validator every run step.

        It is responsible for querying the network and scoring the responses.

        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
        """

        """ START MATCH SYNCING """
        if self.next_match_syncing_datetime <= dt.datetime.now(dt.timezone.utc):
            bt.logging.info(
                "*** Syncing the latest match data to local validator storage. ***"
            )
            sync_result = await utils.sync_match_data(self.match_data_endpoint)
            if sync_result:
                bt.logging.info("Successfully synced match data.")
            else:
                bt.logging.warning("Issue syncing match data")
            self.next_match_syncing_datetime = dt.datetime.now(
                dt.timezone.utc
            ) + dt.timedelta(minutes=DATA_SYNC_INTERVAL_IN_MINUTES)
        """ END MATCH SYNCING """

        """ START LEAGUE COMMITMENTS REQUESTS """
        if self.next_league_commitments_datetime <= dt.datetime.now(dt.timezone.utc):
            # Get all miner uids by passing in high number (300)
            miner_uids = utils.get_random_uids(self, k=300)
            
            # Send league commitments request to miners
            bt.logging.info(
                f"*** Sending league commitments requests to all miners. ***"
            )
            input_synapse = GetLeagueCommitments()
            await utils.send_league_commitments_to_miners(
                self, input_synapse, miner_uids
            )
            # write the uids to league mapping to a file
            with open("_uids_to_league.txt", mode="w") as txt_file:
                txt_file.write("uids_to_league = {\n")
                # Sort by UID before iterating
                for uid in sorted(self.uids_to_last_leagues.keys()):
                    leagues = self.uids_to_last_leagues[uid]
                    first_league = leagues[0] if isinstance(leagues, (list, set, tuple)) and leagues else leagues
                    txt_file.write(f"    {repr(uid)}: {repr(first_league.name)},\n")
                txt_file.write("}\n")
            self.next_league_commitments_datetime = dt.datetime.now(
                dt.timezone.utc
            ) + dt.timedelta(minutes=LEAGUE_COMMITMENT_INTERVAL_IN_MINUTES)
        """ END LEAGUE COMMITMENTS REQUESTS """

        """ START MATCH PREDICTION REQUESTS """
        # Get prediction requests to send to miners
        match_prediction_requests, next_match_request_info = utils.get_match_prediction_requests(self)

        if len(match_prediction_requests) > 0:
            # The dendrite client queries the network.
            bt.logging.info(
                f"*** Sending {len(match_prediction_requests)} matches to miners for predictions. ***"
            )
            # Loop through predictions and send to miners
            for mpr in match_prediction_requests:
                input_synapse = GetMatchPrediction(match_prediction=mpr)

                # Gather all miner uids that have committed to the league in the match prediction
                miner_uids = self.get_miner_uids_committed_to_league(mpr.league)
                if len(miner_uids) == 0:
                    bt.logging.info(f"No miners committed to send requests to for league: {mpr.league}")
                    continue

                bt.logging.info(f"Sending miners {miner_uids} prediction request for match: {mpr}")

                # Send prediction requests to miners and store their responses
                finished_responses, working_miner_uids = (
                    await utils.send_predictions_to_miners(
                        self, input_synapse, miner_uids
                    )
                )

                # Update the scores of miner uids NOT working. Default to 0. Miners who commit to a league but don't respond to a prediction request will be penalized.
                not_working_miner_uids = []
                for uid in miner_uids:
                    if uid not in working_miner_uids:
                        not_working_miner_uids.append(uid)
                        
                if len(not_working_miner_uids) > 0:
                    bt.logging.info(
                        f"Miners {not_working_miner_uids} did not respond or had invalid responses."
                    )
        else:
            bt.logging.info("No matches available to send for predictions.")
            bt.logging.info(f"{next_match_request_info}")
        """ END MATCH PREDICTION REQUESTS """

        """ START MATCH PREDICTIONS CLEANUP """
        # Clean up any unscored predictions from miners that are no longer registered. Archive predictions from miners that are no longer registered.
        if self.next_predictions_cleanup_datetime <= dt.datetime.now(dt.timezone.utc):
            bt.logging.info(
                "*** Cleaning up unscored predictions from miners that are no longer registered. ***"
            )
            # Sync the metagraph to get the latest miner hotkeys
            self.resync_metagraph()

            # Get active hotkeys and uids
            active_hotkeys = []
            active_uids = []
            for uid in range(self.metagraph.n.item()):
                active_uids.append(uid)
                active_hotkeys.append(self.metagraph.axons[uid].hotkey)

            # Delete unscored predictions from miners that are no longer registered
            utils.clean_up_unscored_deregistered_match_predictions(active_hotkeys, active_uids)

            # Archive predictions from miners that are no longer registered
            utils.archive_deregistered_match_predictions(active_hotkeys, active_uids)
            
            self.next_predictions_cleanup_datetime = dt.datetime.now(
                dt.timezone.utc
            ) + dt.timedelta(minutes=PURGE_DEREGGED_MINERS_INTERVAL_IN_MINUTES)
        """ END MATCH PREDICTIONS CLEANUP """

        """ START MATCH PREDICTION SCORING """
        # Check if we're ready to score another batch of predictions
        if self.next_scoring_datetime <= dt.datetime.now(dt.timezone.utc):
            bt.logging.info(f"*** Checking if there are predictions to score. ***")

            (
                predictions,
                edge_scores,
                correct_winner_results,
                prediction_miner_uids,
                prediction_sports,
                prediction_leagues,
            ) = utils.find_and_score_edge_match_predictions(MAX_BATCHSIZE_FOR_SCORING)

            if len(edge_scores) > 0:
                bt.logging.info(
                    f"Scoring (calculating Closing Edge) {len(prediction_miner_uids)} predictions for miners {prediction_miner_uids}."
                )
                bt.logging.info(
                    f"Closing Edge scores: {edge_scores}"
                )

                """ 3/18/2025: Commenting out the post_scored_predictions call to the API. Not needed at this time. 
                # Post scored predictions to API for storage/analysis
                post_result = await utils.post_scored_predictions(
                    self,
                    self.scored_predictions_endpoint,
                    predictions,
                )
                """

            else:
                bt.logging.info("No predictions to score.")
            # Update next sync time
            self.next_scoring_datetime = dt.datetime.now(
                dt.timezone.utc
            ) + dt.timedelta(minutes=SCORING_INTERVAL_IN_MINUTES)
        """ END MATCH PREDICTION SCORING """

        """ START APP PREDICTION FLOW """
        if ENABLE_APP:
            # Check if we're ready to poll the API for prediction requests from the app
            if self.next_app_predictions_syncing_datetime <= dt.datetime.now(
                dt.timezone.utc
            ):
                bt.logging.info(
                    "*** Checking the latest app prediction request data for requests for this validator. ***"
                )
                process_result = await utils.process_app_prediction_requests(
                    self,
                    self.app_prediction_requests_endpoint,
                    self.app_prediction_responses_endpoint,
                )
                if process_result:
                    bt.logging.info(
                        "Successfully processed app match prediction requests."
                    )
                self.next_app_predictions_syncing_datetime = dt.datetime.now(
                    dt.timezone.utc
                ) + dt.timedelta(minutes=APP_DATA_SYNC_INTERVAL_IN_MINUTES)
        """ END APP PREDICTION FLOW """


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    Validator().run()
