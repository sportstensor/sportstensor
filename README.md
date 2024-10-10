
<div align="center">

# Sportstensor: The world's most accurate sports prediction algorithm <!-- omit in toc -->
[![Sportstensor](/docs/sportstensor_header.png)](https://sportstensor.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

---
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Roadmap](#roadmap)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [Community](#community)
- [License](#license)

---
## Introduction

Welcome to Sportstensor, where cutting-edge technology meets sports analytics. We're pioneering the world's most accurate decentralized sports prediction algorithm, powered by the Bittensor network. Our subnet tackles the challenges of high-quality data sourcing, complex data analysis and limited access to advanced machine learning models.

Traditionally, the most accurate sports prediction models have been proprietary, restricting innovation to isolated silos. Collectively, miners and validators are redefining what's possible in this field with the collaborative power of decentralization.

## Key Features
🔑 **Open source model development**
- Sportstensor continuously builds and develops base models for various sports
- Available on HuggingFace for miners to train and improve on

🏅 **Advanced sports analytics**
- Strategic planning and performance analysis with predictions made
- Insights and predictions based on historical and real-time data

💰 **Performance-based incentives**
- Rewards for comprehensive dataset sourcing
- Incentives for developing high-performance predictive models

🌐 **User-friendly integration**
- Intuitive front-end application
- Seamless access to miner predictions

📈 **Scalable improvement**
- Dashboard for miner rankings and proximity to target prediction accuracy
- Designed to foster continous model enhancement for world class results

## Miner and Validator Functionality

### Miner

- Receives requests from the Validator containing specific information such as team names and match details.
- Accesses historical data and current statistics relevant to the teams involved in the query from sports databases.
- Utilizes trained machine learning models to analyze the data and predict outcomes such as match scores.
-  Submits the predicted score and relevant analysis back to the Validator for confirmation and further action.

Miners must return two key pieces of information when responding to prediction requests:
1. **probabilityChoice**: The predicted outcome (Home team, Away team, Draw).
2. **probability**: The probability for that outcome, as a float.

Miners who fail to respond or provide incorrect responses will be penalized.

### Validator

- **Main Loop**: The validator operates in an endless loop, syncing match data every 30 minutes. This includes checking upcoming, in-progress, and completed games.
- **League Commitment**: Validators send requests every 30 minutes to miners for them to commit to active leagues. Miners must respond with predictions for matches in the committed leagues or they receive a penalty score of 0. Miners who fail to commit to any league are penalized -1.
- **Match Prediction Requests**: Prediction requests are sent out at specific time intervals (24 hours, 12 hours, 4 hours, and 10 minutes before a match). Miners are penalized for non-responses.
- **Closing Edge Scoring**: After match completion, the validator calculates the closing edge scores for each prediction and updates the local database.
- **Prediction Cleanup**: Non-registered miners and outdated predictions are regularly cleaned from the system to ensure only valid data is kept.

### Scoring and Weights

- **Incentive Mechanism**: 
   - The validator calculates incentives based on miners’ commitments to leagues and their prediction accuracy.
   - Incentives and scores are updated every 20 minutes. League-specific scoring percentages are applied, and scores are aggregated and logged. 
   - Validators adjust the miners' weights on the chain based on these scores.
   - The max number of predictions included for a miner per league is determined by the league’s **ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE** * 2.
   - A separate background thread handles the processing of calculating scores and setting weights, which runs every 20 minutes.

## Roadmap

### Phase 1: Foundation (Q3 2024)
- [x] Launch on testnet (172)
- [x] Develop baseline model for soccer (Major League Soccer)
- [x] Develop baseline model for baseball (Major League Baseball)
- [x] Launch website (sportstensor.com)
- [ ] Begin marketing for brand awareness and interest

### Phase 2: Expansion (Q4 2024)
- [ ] Launch front-end application
- [ ] Introduce next level of prediction queries and validation metrics
- [ ] Collaborations and partnerships with synergistic companies and subnets
- [ ] Build our proprietary database for miners
- [ ] Achieve competitive baseline prediction accuracy
- [ ] Build proprietary LLM chatbot hooked up to miner predictions
- [ ] Monetize predictions through front-end

### Phase 3: Refinement (Q1 2025)
- [ ] Market and sales expansion
- [ ] Further develop baseline models
- [ ] Expand to basketball and NFL to cover the whole year for sports
- [ ] Explore niche sports such as eSports and UFC
- [ ] Monetize API access to predictions and proprietary database
- [ ] Build super secret Sportstensor tool 😉

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
git clone https://github.com/xzistance/sportstensor/
cd sportstensor
```
2. Install pm2 if you don't already have it: [pm2.io](https://pm2.io/docs/runtime/guide/installation/).
3. Next, install the `Sportstensor` package: `pip install -e .`
4. Copy the example environment file for miner configuration:
```bash
cp neurons/example.miner.env neurons/miner.env
```
5. Update `miner.env` with your league commitments. Ensure that your commitments reflect the leagues you are participating in. This will enable the system to send you predictions for the appropriate matches.

#### Prediction Return Format

When responding to prediction requests, miners need to provide two key pieces of information:
1. **probabilityChoice**: The predicted outcome (Home team, Away team, Draw).
2. **probability**: The probability for that outcome, as a float.

Failure to respond or incorrectly formatted responses will result in penalties as described in the scoring and incentive mechanisms.

### Running a Validator
#### Requirements
- Python 3.8+
- Pip
- CPU
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.

#### Recommended
- Utilizing wandb. Set environment variable with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable wandb with --wandb.off

#### Setup
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/xzistance/sportstensor/
cd sportstensor
```
2. Install pm2 if you don't already have it: [pm2.io/docs/runtime/guide/installation/].
3. Next, install the `Sportstensor` package: `pip install -e .`

#### Run auto-updating validator with PM2 (recommended)
```bash
pm2 start vali_auto_update.sh --name Sportstensor-validator --     --netuid {netuid}     --wallet.name {wallet}     --wallet.hotkey {hotkey}     --axon.port {port}     --logging.trace
```
Note: you might need to adjust "python" to "python3" within the `vali_auto_update.sh` depending on your preferred system python.

Additionally, validators can use the flag --neuron.batch_size X to set a different batch size for sending requests to miners.

#### Run basic validator with PM2
```bash
pm2 start neurons/validator.py --name Sportstensor-validator --     --netuid {netuid}     --wallet.name {wallet}     --wallet.hotkey {hotkey}     --axon.port {port}     --logging.trace
```

## Community

Join the vibrant Bittensor community and find our channel `#פ • nun • 41` on [Discord](https://discord.gg/bittensor).

## License

The Sportstensor subnet is released under the [MIT License](./LICENSE).

---

</div>
