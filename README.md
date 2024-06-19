<div align="center">

# SportsTensor Bittensor Subnet: Sports insights through state-of-the-art & decentralized AI <!-- omit in toc -->
[![OMEGA](/docs/header_bg.png)](https://sportstensor.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Be, and it becomes ... <!-- omit in toc -->
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
- [Contributing](#contributing)
- [License](#license)

---
## Introduction

Welcome to SportsTensor, a cutting-edge platform designed to transform sports analytics through the decentralized capabilities of the Bittensor network. Our platform tackles the challenges of data overload, complex data analysis, and limited access to actionable sports insights by leveraging competitive machine learning and AI to deliver real-time and highly accurate sports analytics to teams, leagues, and sports enthusiasts.


## Key Features

- 🌍 **Data-Driven Insights**: Provides actionable insights and predictions based on historical and real-time data for strategic planning and performance analysis.
- 🧠 **Decentralized and Reliable**: Utilizes the decentralized Bittensor network, enhancing the accuracy and reliability of data through community-verified contributions.
- 💰 **Incentivized Data Collection**: Miners are rewarded to develop high-performance predictive models on our Bittensor-based subnet.
- 🤖 **Easy Access and Integration**: Offers convenient access via mobile and web platforms, with APIs for easy integration into existing systems.
- 🎮 **Scalable Solution**: Designed to seamlessly scale up and accommodate growing data needs without disrupting user experience.

## Miner and Validator Functionality

### Miner

- Receives requests from the Validator containing specific information such as team names and match details.
- Accesses historical data and current statistics relevant to the teams involved in the query from sports databases.
- Utilizes trained machine learning models to analyze the data and predict outcomes such as match scores.
-  Submits the predicted score and relevant analysis back to the Validator for confirmation and further action.

### Validator

- Collects predictions from the Miner, which include the anticipated outcomes and relevant statistical analyses of sports events.
- Compares the Miner's predictions with the actual outcomes of the matches, which are sourced from trusted sports databases and official results.
- Logs the results of the validations for auditing and continuous improvement of the predictive system on the SportsTensor platform

## Roadmap

### Phase 1: Foundation (Q3 2024)
- [x] Launch on testnet, develop baseline models for soccer and baseball.

### Phase 2: Expansion (Q3 2024)
- [ ] Introduce validation metrics, begin brand awareness campaigns, launch front-end.

### Phase 3: Refinement (Q3 2024)
- [ ] Market and sales expansion, further develop sports models, including for basketball and NFL.

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
3. Next, install the `sportstensor` package: `pip install -e .`

#### Run with PM2
```bash
pm2 start neurons/miner.py --name sportstensor-miner -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
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

#### Setup
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/xzistance/sportstensor/
cd sportstensor
```
2. Install pm2 if you don't already have it: [pm2.io](https://pm2.io/docs/runtime/guide/installation/).
3. Next, install the `sportstensor` package: `pip install -e .`

#### Run auto-updating validator with PM2 (recommended)
```bash
pm2 start auto_updating_validator.sh --name sportstensor-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```
Note: you might need to adjust "python" to "python3" within the `neurons/auto_updating_validator.sh` depending on your preferred system python.

#### Run basic validator with PM2
```bash
pm2 start neurons/validator.py --name sportstensor-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```

## Contributing

Join our vibrant community on [Discord](https://discord.gg/opentensor).

## License

The Sportstensor Bittensor subnet is released under the [MIT License](./LICENSE).

---

</div>