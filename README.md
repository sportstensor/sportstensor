
<div align="center">

# Sportstensor: The future of sports prediction algorithms <!-- omit in toc -->

</div>

- [Introduction](#introduction)
- [Why is this important?](#why-is-this-important)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Roadmap](#roadmap)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [Community](#community)
- [License](#license)

## Introduction

Welcome to Sportstensor—the convergence of cutting-edge technology and sports data analytics. We are pioneering unprecedented innovation in sports prediction algorithms, powered by the Bittensor network.

Our Sportstensor subnet is designed to incentivize the discovery of competitive advantages over closing market odds, enabling top miners within the network to establish machine-driven dominance across the sports prediction landscape.

## Why is this important?
- Closing odds represent the pinnacle of market efficiency, determined by thousands of advanced machine learning algorithms.
- Our subnet fosters the development of true machine intelligence by outperforming competing algorithms in a highly competitive AI-versus-AI environment.
- Even with sophisticated models, bettors struggle to be profitable, as bookmakers frequently impose strict limits on consistent winners.
- There is substantial demand for high-performing predictive AI from betting operators, financial firms, syndicates, and algorithmic traders seeking more accurate models.
- We attract top AI and machine learning talent from diverse industries, encouraging them to revolutionize sports prediction markets.
- By decentralizing the creation and improvement of predictive models, we reduce reliance on any single entity or algorithm, enhancing resilience and driving innovation in the sports prediction market.

## Miner and Validator Functionality

### Miner

- Receives requests from the Validator containing specific information such as team names and match details.
- Accesses historical data and current statistics relevant to the teams involved in the query from sports databases.
- Utilizes trained machine learning models to analyze the data and predict the team they think will win and the probability.
- Returns the prediction back to the Validator for confirmation and further action.

Miners must return two key pieces of information when responding to prediction requests:
1. **probabilityChoice**: The predicted outcome (Home team, Away team, Draw).
2. **probability**: The probability for that outcome, as a float between 0 and 1. e.g. 0.6 represents 60% probability.

Miners who fail to respond or provide incorrect responses will be penalized.

### Validator

- **Match Syncing**: The validator operates in an endless loop, syncing match data every 30 minutes. This includes checking upcoming, in-progress, and completed games.
- **League Commitment**: Validators send requests every 15 minutes to acquire the active leagues a miner is committed to. Miners are required to respond with predictions for all matches in their committed leagues or receive a penalty. Miners who fail to commit to at least one league will be incrementally penalized until committing or deregistering.
- **Match Prediction Requests**: Prediction requests are sent out at specific time intervals (24 hours, 12 hours, 4 hours, and 10 minutes before a match). Miners are penalized for non-responses.
- **Closing Edge Scoring**: After match completion, the validator calculates the closing edge scores for each prediction and updates the local database to be used later in the scoring and weights logic.
- **Prediction Cleanup**: Non-registered miners and outdated predictions are regularly cleaned from the system to ensure only valid data is kept.

### Scoring and Weights

- **Incentive Mechanism**: 
   - Incentives and scores are calculated every 20 minutes in a background thread.
   - Each active league is iterated through to calculate scores for that league.
   - During each league iteration, every miner is scored for their prediction accuracy.
   - The max number of predictions included for a miner per league is determined by the league’s **ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE** multiplied by 2.
      - This creates a constantly rolling forward set of predictions per miner per league, so older predictions outside these thresholds will not be scored.
   - Incentive scores are calculated through a series of complex algorithms. Please see our whitepaper for more details. Also analyze `vali_utils/scoring_utils.py`.
   - After all active leagues have been scored, league-specific scoring percentages are applied.
   - Final scores are aggregated and logged for weight setting.
   - Validators set the miners' weights on the chain based on these scores.

## Roadmap

### Phase 1: Foundation (Q3 2024)
- [x] Launch on testnet (172)
- [x] Develop baseline model for soccer (Major League Soccer)
- [x] Develop baseline model for baseball (Major League Baseball)
- [x] Launch website (sportstensor.com)
- [x] Begin marketing for brand awareness and interest
- [x] Build dashboard
- [x] Launch front-end application

### Phase 2: Upgrade (Q4 2024)
- [ ] Upgrade incentive mechanism to v2
- [ ] Upgrade dashboard to track miner performance and progress
- [ ] Achieve machine learning dominance over the sports prediction market by our top miners
- [ ] Collaborations and partnerships with synergistic companies and subnets
- [ ] Build application to monetize validator bandwidth
- [ ] Expand on commitable leagues

### Phase 3: Expand (Q1 2025)
- [ ] Market and B2B sales expansion
- [ ] Grow the team
- [ ] Explore niche sports offerings
- [ ] Develop additional products

## Running Miners and Validators

### Running a Miner
#### Requirements
- Python 3.8+
- Pip
- CPU
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.

#### Setup
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/sportstensor/sportstensor/
cd sportstensor
```
2. Install pm2 if you don't already have it: [pm2.io](https://pm2.io/docs/runtime/guide/installation/).
3. Next, install the `Sportstensor` package: `pip install -e .`
4. Copy the example environment file for miner configuration:
```bash
cp neurons/example.miner.env neurons/miner.env
```
5. Update `miner.env` with your league commitments. Ensure that your commitments reflect the leagues you are participating in. This will enable the system to send you predictions for the appropriate matches.

#### Run with PM2
```bash
pm2 start neurons/miner.py --name Sportstensor-miner -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --axon.external_ip {ip} \
    --blacklist.force_validator_permit
```

#### League Committments Return Format
When responding to league commentment requests, miners need to provide the active leagues they are committing to.
The `miner.env` file will allow miners to define their league commitments without having to restart their miner.
```bash
# NFL, MLB, NBA, etc. -- check common/data.py for all available leagues
LEAGUE_COMMITMENTS=NFL,MLB,EPL
```

#### Prediction Return Format

When responding to prediction requests, miners need to provide two key pieces of information:
1. **probabilityChoice**: The predicted outcome (Home team, Away team, Draw).
2. **probability**: The probability for that outcome, as a float between 0 and 1. e.g. 0.6 represents 60% probability.

Failure to respond or incorrectly formatted responses will result in penalties as described in the scoring and incentive mechanisms.

### Running a Validator
#### Requirements
- Python 3.8+
- Pip
- CPU
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.

#### Importance of the Database
The subnet creates a sqlite database in the root directory titled `SportsTensorEdge.db`. This database is extremely important to the functionality of the subnet as it holds historical prediction data for all miners which are used in the scoring mechanism.

**If you need to reinstall or migrate the subnet code, please be very careful and bring along the database with you!!!**

Please reach out to the subnet team for any questions or assistance.

#### Recommended
- Utilizing wandb. Set environment variable with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable wandb with --wandb.off

#### Setup
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/sportstensor/sportstensor/
cd sportstensor
```
2. Install pm2 if you don't already have it: [pm2.io/docs/runtime/guide/installation/].
3. Next, install the `Sportstensor` package: `pip install -e .`

#### Run auto-updating validator with PM2 (recommended)
```bash
pm2 start vali_auto_update.sh --name Sportstensor-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```
Note: you might need to adjust "python" to "python3" within the `vali_auto_update.sh` depending on your preferred system python.

Additionally, validators can use the flag `--neuron.batch_size X` to set a different batch size for sending requests to miners.

#### Run basic validator with PM2
```bash
pm2 start neurons/validator.py --name Sportstensor-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```

## Environments

| Network | Netuid |
| ----------- | -----: |
| Mainnet     |     41 |
| Testnet     |    172 |

## Community

Join the vibrant Bittensor community and find our channel `#פ • nun • 41` on [Discord](https://discord.gg/bittensor).

## License

The Sportstensor subnet is released under the [MIT License](./LICENSE).

---

</div>
