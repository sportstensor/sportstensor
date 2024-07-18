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

### Validator

- Collects predictions from the Miner, which include the anticipated outcomes and relevant statistical analyses of sports events.
- Compares the Miner's predictions with the actual outcomes of the matches, which are sourced from trusted sports databases and official results.
- Logs the results of the validations for auditing and continuous improvement of the predictive system on the Sportstensor platform

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

#### Scoring Mechanism
1. Home and Away Score Accuracy (Max 0.25 each):

The scoring function calculates the absolute difference between the predicted and actual scores for both the home and away teams.
The difference for each team is normalized by a maximum score difference parameter to maintain a scale between 0 and 1.
Each team's score accuracy contributes up to 0.25 to the total score. A smaller difference results in a higher score, meaning a perfect prediction (no difference) results in the full 0.25 points.

2. Correct Winner Prediction (Max 0.5):

The function determines the winner based on the predicted scores and compares this to the actual match outcome.
If the predicted winner matches the actual winner, the full 0.5 points are awarded. If not, no points are awarded for this part of the prediction.
Total Score Calculation:

The total prediction score is the sum of the individual scores for the home team, away team, and the correct winner prediction.
The maximum possible score is 1.0, achieved by perfectly predicting the home and away scores (0.25 + 0.25) and correctly identifying the match winner (0.5).


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
2. Install pm2 if you don't already have it: [pm2.io](https://pm2.io/docs/runtime/guide/installation/).
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

#### Run basic validator with PM2
```bash
pm2 start neurons/validator.py --name Sportstensor-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```

## Community

Join the vibrant Bittensor community and find our channel `#פ • pe • 41` on [Discord](https://discord.gg/bittensor).

## License

The Sportstensor subnet is released under the [MIT License](./LICENSE).

---

</div>
