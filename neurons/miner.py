# The MIT License (MIT)
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

import asyncio
from collections import defaultdict
import copy
import sys
import threading
import time
import traceback
import typing
import bittensor as bt
import datetime as dt
from common import constants, utils
from common.data import MatchPrediction
from common.protocol import (
    GetMatchPrediction,
    REQUEST_LIMIT_BY_TYPE_PER_PERIOD,
)
from common.predictions import make_match_prediction
from neurons.config import NeuronType
from neurons.config import NeuronType, check_config, create_config


class Miner:
    """The Sports Tensor Miner."""

    def __init__(self, config=None):
        self.config = copy.deepcopy(config or create_config(NeuronType.MINER))
        check_config(self.config)

        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(self.config)

        
        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}.")

        # The subtensor is our connection to the Bittensor blockchain.
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}.")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}.")

        # Each miner gets a unique identity (UID) in the network for differentiation.
        # TODO: Stop doing meaningful work in the constructor to make neurons more testable.
        if self.wallet.hotkey.ss58_address in self.metagraph.hotkeys:
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(
                f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}."
            )
        else:
            self.uid = 0
            bt.logging.warning(
                f"Hotkey {self.wallet.hotkey.ss58_address} not found in metagraph. Assuming this is a test."
            )

        self.last_sync_timestamp = dt.datetime.min
        self.step = 0

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.get_index,
            blacklist_fn=self.get_index_blacklist,
            priority_fn=self.get_index_priority,
        )
        bt.logging.success(f"Axon created: {self.axon}.")

        # Instantiate runners.
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = threading.RLock()

        # TODO: Instantiate base level LLM?

        # Configure per hotkey per request limits.
        self.request_lock = threading.RLock()
        self.last_cleared_request_limits = dt.datetime.now()
        self.requests_by_type_by_hotkey = defaultdict(lambda: defaultdict(lambda: 0))
    

    def run(self):
        """
        Initiates and manages the main loop for the miner.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}."
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        self.last_sync_timestamp = dt.datetime.now()
        bt.logging.success(f"Miner starting at {self.last_sync_timestamp}.")

        while not self.should_exit:
            # This loop maintains the miner's operations until intentionally stopped.
            try:
                
                # Epoch length defaults to 100 blocks at 12 seconds each for 20 minutes.
                while dt.datetime.now() - self.last_sync_timestamp < (
                    dt.timedelta(seconds=12 * self.config.neuron.epoch_length)
                ):
                    # Wait before checking again.
                    time.sleep(12)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph.
                self.sync()

                self._log_status(self.step)

                self.last_sync_timestamp = dt.datetime.now()
                self.step += 1

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                if not self.config.offline:
                    self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                sys.exit()

            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
    

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("Attempting to resync the metagraph.")

        # Sync the metagraph.
        new_metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        with self.lock:
            self.metagraph = new_metagraph

        bt.logging.success("Successfuly resynced the metagraph.")

    def _log_status(self, step: int):
        """Logs a summary of the miner status in the subnet."""
        relative_incentive = self.metagraph.I[self.uid].item() / max(self.metagraph.I)
        incentive_and_hk = zip(self.metagraph.I, self.metagraph.hotkeys)
        incentive_and_hk = sorted(incentive_and_hk, key=lambda x: x[0], reverse=True)
        position = -1
        for i, (_, hk) in enumerate(incentive_and_hk):
            if hk == self.wallet.hotkey.ss58_address:
                position = i
                break
        log = (
            f"Step:{step} | "
            f"Block:{self.metagraph.block.item()} | "
            f"Stake:{self.metagraph.S[self.uid]} | "
            f"Incentive:{self.metagraph.I[self.uid]} | "
            f"Relative Incentive:{relative_incentive} | "
            f"Position:{position} | "
            f"Emission:{self.metagraph.E[self.uid]}"
        )
        bt.logging.info(log)
    

    async def get_match_prediction(
        self, synapse: GetMatchPrediction
    ) -> GetMatchPrediction:
        """Runs after the GetMatchPrediction synapse has been deserialized (i.e. after synapse.data is available)."""
        bt.logging.info(
            f"Received GetMatchPrediction request from {synapse.dendrite.hotkey}."
        )

        # Make the match prediction based on the requested MatchPrediction object
        # TODO: does this need to by async?
        synapse.match_prediction = make_match_prediction(synapse.match_prediction)
        synapse.version = constants.PROTOCOL_VERSION

        bt.logging.success(
            f"Returning MatchPrediction ID: {str(synapse.match_prediction.matchID)} to {synapse.dendrite.hotkey}."
        )

        return synapse

    def default_priority(self, synapse: bt.Synapse) -> float:
        """The default priority that prioritizes by validator stake."""
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}.",
        )
        return priority

    def get_config_for_test(self) -> bt.config:
        return self.config

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner.
        """
        # Ensure miner hotkey is still registered on the network.
        self.check_registered()
        self.resync_metagraph()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )
            sys.exit(1)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(60)
