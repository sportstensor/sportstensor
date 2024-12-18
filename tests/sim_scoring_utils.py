import os
import matplotlib.pyplot as plt
import numpy as np

import datetime as dt
from datetime import timezone
import random
from typing import Dict, List
from tabulate import tabulate

import bittensor
from storage.sqlite_validator_storage import get_storage

from common.data import Match, League, MatchPrediction, MatchPredictionWithMatchData
from common.constants import (
    ACTIVE_LEAGUES,
    MAX_PREDICTION_DAYS_THRESHOLD,
    ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE,
    LEAGUE_SENSITIVITY_ALPHAS,
    COPYCAT_PUNISHMENT_START_DATE,
    MAX_GFILTER_FOR_WRONG_PREDICTION,
    MIN_PROBABILITY,
    MIN_PROB_FOR_DRAWS,
    LEAGUES_ALLOWING_DRAWS,
    SENSITIVITY_ALPHA,
    GAMMA,
    TRANSITION_KAPPA,
    EXTREMIS_BETA,
    LEAGUE_SCORING_PERCENTAGES,
    COPYCAT_PENALTY_SCORE,
    PARETO_MU,
    PARETO_ALPHA
)

from vali_utils.copycat_controller import CopycatDetectionController

from vali_utils.scoring_utils import (
    calculate_edge,
    compute_significance_score,
    calculate_clv,
    calculate_incentive_score,
    apply_gaussian_filter,
    apply_pareto,
)


def calculate_incentives_and_update_scores():
    """
    Calculate the incentives and update the scores for all miners with predictions.

    This function:
    1. Loops through every league
    2. For each league, loops through every miner
    4. Calculates incentives for miners committed to the league
    5. Updates scores for each miner for each league
    6. Updates the validator scores for each miner to set their weights
    7. Logs detailed results for each league and final scores

    :param vali: Validator, the validator object
    """
    # Initialize our subtensor and metagraph
    NETWORK = None # "test" or None
    NETUID = 41
    if NETWORK == "test":
        NETUID = 172
    
    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    storage = get_storage()
    all_uids = metagraph.uids.tolist()
    #all_uids = metagraph.uids.tolist()[:10]

    # Initialize Copycat Detection Controller
    copycat_controller = CopycatDetectionController()
    final_suspicious_miners = set()
    final_copycat_penalties = set()
    final_exact_matches = set()
    
    # Initialize league_scores dictionary
    league_scores: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_pred_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in ACTIVE_LEAGUES}
    # Use this to get payouts to calculate ROI
    league_roi_counts: Dict[League, List[int]] = {league: [0] * len(all_uids) for league in ACTIVE_LEAGUES}
    league_roi_payouts: Dict[League, List[float]] = {league: [0.0] * len(all_uids) for league in ACTIVE_LEAGUES}

    leagues_to_analyze = ACTIVE_LEAGUES
    #leagues_to_analyze = [League.NBA]

    uids_to_last_leagues = {}
    uids_to_leagues_last_updated = {}

    uid_to_best_league = {
        0: 'NBA',      # NBA: 0.00549531, EPL: -0.000862287, NFL: 0.00021872
        4: 'NBA',      # NBA: 1.39478, EPL: -0.5678, NFL: 0.483975
        6: 'EPL',      # NBA: -0.00354889, EPL: 0.00244482, NFL: -1.98545e-05
        7: 'NBA',      # NBA: 0.00489986, EPL: -0.00155182, NFL: 0.000497882
        8: 'NBA',      # NBA: 0.0651583, EPL: -0.00991251, NFL: -0.000736007
        9: 'NBA',      # NBA: 0.0609043, EPL: 0.00220152, NFL: -4.7169e-05
        10: 'NBA',     # NBA: -0.00199353, EPL: -0.00124317, NFL: 0.000517759
        12: 'NFL',     # NBA: -1.44031, EPL: -0.957016, NFL: -1.04315
        14: 'EPL',     # NBA: -0.0320488, EPL: -0.00622819
        15: 'EPL',     # NBA: -14.0798, EPL: 3.15163e-09
        16: 'NBA',     # NBA: 0.0134838, EPL: -0.00895769
        17: 'EPL',     # NBA: -2.00295, EPL: -0.842251, NFL: -1.88506
        18: 'NBA',     # NBA: 0.0628245, EPL: -0.00563762, NFL: -0.000667659
        19: 'EPL',     # NBA: -0.0160819, EPL: -0.00403653
        20: 'EPL',     # NBA: -2.78309, EPL: -0.498345, NFL: -2.4467
        21: 'NFL',     # NBA: 0.51433, EPL: -2.18173, NFL: 10.693
        22: 'EPL',     # NBA: -0.017392, EPL: -0.00212223
        23: 'NFL',     # NBA: -27.3268, EPL: -4.03054, NFL: 1.03913
        24: 'NBA',     # NBA: 0.0101252, EPL: -0.0197205, NFL: -0.00175025
        25: 'NBA',     # NBA: 0.0606919, EPL: -0.0122528, NFL: -0.00183062
        26: 'EPL',     # NBA: -5.35848, EPL: -1.05867, NFL: -1.77626
        29: 'NBA',     # NBA: 1.43969, EPL: -0.299053, NFL: 0.469992
        30: 'EPL',     # EPL: -0.00328628
        31: 'EPL',     # NBA: -0.576609, EPL: 0.152137, NFL: -3.03276
        32: 'EPL',     # NBA: -0.00140011, EPL: -0.00170567
        33: 'NBA',     # NBA: 0.00761529, EPL: 0.000397166, NFL: 0.000317264
        34: 'EPL',     # NBA: -0.95135, EPL: -1.02697, NFL: -2.22173
        35: 'NBA',     # NBA: -2.56843, EPL: -0.314112, NFL: -1.12526
        36: 'NBA',     # NBA: 0.0554796, EPL: -0.0089094, NFL: -0.00183599
        37: 'NBA',     # NBA: 0.0100823, EPL: -0.00284436, NFL: 0.000111423
        38: 'EPL',     # NBA: -2.1585, EPL: 0.541175
        41: 'NBA',     # NBA: 1.62597, EPL: -0.540154, NFL: 0.412724
        42: 'EPL',     # NBA: -4.25973, EPL: -0.464677, NFL: -0.590194
        43: 'EPL',     # NBA: -0.0012138, EPL: -0.00453559, NFL: 8.87072e-05
        44: 'EPL',     # NBA: -6.18743, EPL: -3.76418, NFL: -0.430153
        45: 'EPL',     # EPL: -0.00119587
        46: 'EPL',     # NBA: -0.00451565, EPL: 8.37977e-07, NFL: -4.10008e-05
        47: 'NFL',     # NBA: -5.20914, EPL: -0.765292, NFL: 1.22808
        48: 'NFL',     # NBA: -0.108784, EPL: -0.0102622, NFL: 0.00248338
        49: 'EPL',     # NBA: -0.00381958, EPL: -0.00112558, NFL: -0.000124026
        51: 'NBA',     # NBA: 3.06211, EPL: 0.398022, NFL: 2.05531
        52: 'NBA',     # NBA: 0.0655432, EPL: -0.00526663, NFL: -0.00183571
        53: 'NBA',     # NBA: 0.00191485, EPL: 0.000818648, NFL: -1.67742e-05
        54: 'EPL',     # NBA: -1.47925, EPL: 0.0410756, NFL: -2.62303
        57: 'NFL',     # NBA: -0.0888912, NFL: 7.11162e-06
        58: 'NBA',     # NBA: 0.0680194, EPL: -0.00210125, NFL: -0.000730763
        59: 'NBA',     # NBA: 0.0466925, EPL: -0.0230185, NFL: -0.00183544
        60: 'NBA',     # NBA: 0.0836556, EPL: -0.000835455, NFL: -6.55415e-05
        61: 'NBA',     # NBA: 0.0561207, EPL: -0.0105183, NFL: -0.00183492
        62: 'NBA',     # NBA: 0.0691787, EPL: 0.00359952, NFL: -1.1449e-05
        64: 'EPL',     # NBA: -0.012283, EPL: -0.00286962
        66: 'EPL',     # NBA: -3.17573, EPL: -1.45463, NFL: -2.01198
        67: 'NFL',     # NBA: -0.0769223, EPL: -2.1949, NFL: 10.5947
        68: 'EPL',     # NBA: -0.0114733, EPL: -0.0019125
        69: 'NBA',     # NBA: 0.0644989, EPL: -0.0109959, NFL: -0.00183585
        70: 'NBA',     # NBA: 0.0404441, EPL: -0.00535964, NFL: -0.00183493
        71: 'NFL',     # NBA: 1.4152, EPL: -0.760337, NFL: 1.74726
        72: 'EPL',     # NBA: -0.00139814, EPL: -0.000471776, NFL: -0.00013663
        73: 'NBA',     # NBA: 0.00133798, EPL: -0.00368824
        74: 'NBA',     # NBA: 0.00177158, EPL: -0.00528752
        75: 'NBA',     # NBA: 0.0110365, EPL: -0.0193257
        76: 'NFL',     # NBA: -0.00740535, EPL: -0.00592235, NFL: 0.000575382
        78: 'NBA',     # NBA: 6.44141, EPL: -0.857837, NFL: 0.399258
        79: 'EPL',     # NBA: -0.00169266, EPL: 0.00491857
        80: 'EPL',     # NBA: -1.34398, EPL: -0.095412
        81: 'NBA',     # NBA: 0.0502373, EPL: -0.00772791, NFL: -0.00183485
        82: 'NBA',     # NBA: 0.00325992, EPL: -0.00094375, NFL: 0.000197089
        83: 'NBA',     # NBA: 0.0590018, EPL: -0.0081072, NFL: -0.00183318
        84: 'NBA',     # NBA: 1.30737, EPL: -0.485886, NFL: 0.467725
        86: 'NBA',     # NBA: 0.00751199, EPL: -0.00559624, NFL: 0.000607101
        87: 'NBA',     # NBA: 0.00944507, EPL: -0.00125586, NFL: -0.000447663
        88: 'NBA',     # NBA: 0.00116602, EPL: -0.0091584, NFL: 0.000532489
        90: 'EPL',     # NBA: -2.48621, EPL: -0.303108, NFL: -2.60368
        91: 'EPL',     # NBA: -4.21748, EPL: -1.15689, NFL: -0.263983
        92: 'NFL',     # NBA: -2.85562, EPL: -0.688206, NFL: -2.06567
        93: 'NBA',     # NBA: 0.00677752, EPL: -0.00967879
        95: 'NBA',     # NBA: 0.0630967, EPL: -0.0276388, NFL: -8.70028e-05
        97: 'NBA',     # NBA: 0.0108071, EPL: -0.0162234
        98: 'NBA',     # NBA: 0.00131656, EPL: -0.00378768, NFL: -0.000266354
        99: 'NBA',     # NBA: 0.0757327, EPL: 0.00362265, NFL: -6.61011e-05
        100: 'NBA',    # NBA: 1.4847, EPL: -0.392039, NFL: 0.469005
        101: 'EPL',    # NBA: -1.36457, EPL: -0.938313, NFL: -2.61521
        102: 'NFL',    # NBA: 0.958472, EPL: -0.0846324, NFL: 1.1339
        104: 'NFL',    # NBA: -0.116212, EPL: -0.0273425, NFL: 0.00254156
        105: 'EPL',    # EPL: 1.35832e-05
        106: 'NFL',    # NBA: -0.00233833, EPL: -0.00416307, NFL: 0.000135698
        107: 'NBA',    # NBA: 0.000546167, EPL: 0.000285085, NFL: 0.000480232
        108: 'NFL',    # NBA: -0.00547655, EPL: -0.0024336, NFL: 0.000114651
        109: 'EPL',    # EPL: -0.0093971
        110: 'EPL',    # NBA: -0.00129921, EPL: -0.00617534
        111: 'NFL',    # NBA: -0.145606, EPL: -0.0157812, NFL: 0.00231793
        112: 'NBA',    # NBA: 7.07387, EPL: 0.475188, NFL: -0.303994
        113: 'EPL',    # NBA: -1.20353, EPL: -0.754569, NFL: -1.69617
        114: 'NBA',    # NBA: 0.00146815, EPL: -0.0113652
        115: 'NBA',    # NBA: 0.0455808, EPL: -0.00411722, NFL: -0.00183554
        116: 'NBA',    # NBA: 0.00402246, EPL: -0.0106408, NFL: 0.000192879
        117: 'EPL',    # NBA: -1.76052, EPL: 0.135151, NFL: -3.13276
        118: 'NBA',    # NBA: 0.0542371, EPL: -0.00748668, NFL: -0.00183586
        119: 'EPL',    # EPL: -0.00300641
        120: 'NBA',    # NBA: 0.0497496, EPL: -0.00720801, NFL: -0.00183594
        121: 'NBA',    # NBA: 0.0699909, EPL: -0.00509823, NFL: -0.000114014
        123: 'EPL',    # NBA: -2.86581, EPL: -0.442533, NFL: -3.31823
        124: 'NBA',    # NBA: 0.00257092, EPL: -0.00426434
        126: 'NBA',    # NBA: 3.60816, EPL: -0.277219, NFL: -0.858553
        127: 'NBA',    # NBA: 0.00446108, EPL: 0.00761244, NFL: -5.47913e-05
        128: 'NBA',    # NBA: 0.0608795, EPL: 0.000347714, NFL: -0.00183598
        129: 'NBA',    # NBA: 0.0129139, EPL: -0.0143942
        130: 'EPL',    # NBA: -0.00821843, EPL: 0.00745341, NFL: 0.000295534
        131: 'NBA',    # NBA: 0.00330906, EPL: -0.00824692
        132: 'NBA',    # NBA: 0.00172342, EPL: -0.00658816
        133: 'EPL',    # EPL: 1.13233e-05
        134: 'EPL',    # NBA: -0.196475, EPL: -0.315017, NFL: -2.44037
        136: 'NFL',    # NBA: -5.94737, EPL: -0.649522, NFL: 1.98784
        137: 'NBA',    # NBA: 3.25088, EPL: -1.42787, NFL: -1.99108
        138: 'NBA',    # NBA: 6.81623, EPL: 0.149003, NFL: -0.316528
        140: 'EPL',    # NBA: -0.0442832, EPL: -0.00147859
        141: 'NFL',    # NBA: -2.319, EPL: -0.57382, NFL: 0.0038594
        143: 'EPL',    # NBA: 0.00394266, EPL: 0.0053887, NFL: 0.000721167
        144: 'NFL',    # NBA: 0.20831, EPL: 0.173064, NFL: 1.35287
        145: 'NBA',    # NBA: -0.00259412, EPL: 0.00530704, NFL: 0.000170917
        146: 'EPL',    # NBA: -8.89797, EPL: -4.44068, NFL: -2.55026
        147: 'NBA',    # NBA: -0.000846921, EPL: -0.00610042, NFL: 0.000631325
        148: 'NBA',    # NBA: 0.00364044, EPL: 0.00198855
        149: 'NBA',    # NBA: 0.00702232, EPL: -4.33612e-05, NFL: 0.000135642
        150: 'NBA',    # NBA: 0.0520572, EPL: -0.00634325, NFL: -0.00183595
        151: 'NBA',    # NBA: 0.00372613, EPL: -0.0022979, NFL: 0.000259929
        152: 'NBA',    # NBA: 0.00788707, EPL: -0.00640902
        153: 'EPL',    # NBA: -2.24458, EPL: 0.116087, NFL: -1.69738
        154: 'EPL',    # NBA: -0.000358148, EPL: 0.00330378
        155: 'NBA',    # NBA: 0.00645906, EPL: -0.00707944
        156: 'EPL',    # NBA: -2.55718, EPL: 0.126685, NFL: -2.61517
        157: 'NBA',    # NBA: 0.0053556, EPL: -0.0167484
        158: 'NBA',    # NBA: 0.0559163, EPL: -0.00494299, NFL: -0.00183596
        159: 'NFL',    # NBA: -25.1885, EPL: -3.93117, NFL: 0.565918
        160: 'EPL',    # NBA: -1.05652, EPL: -0.618558, NFL: -2.47087
        162: 'NBA',    # NBA: 0.00974715, EPL: 0.00472273, NFL: -0.000299714
        163: 'NBA',    # NBA: 0.0774234, EPL: -0.000207802, NFL: -6.97179e-05
        164: 'NFL',    # NBA: -3.41864, EPL: -0.122055, NFL: 0.832228
        165: 'NBA',    # NBA: 0.0026242, EPL: -0.0192809, NFL: -0.00058774
        167: 'NBA',    # NBA: 0.00945336, EPL: 0.00103349, NFL: 0.000135958
        168: 'NBA',    # NBA: 1.50736, EPL: -0.414925, NFL: 0.485616
        169: 'NFL',    # NBA: -0.82635, EPL: -0.585383, NFL: -0.192169
        170: 'NBA',    # NBA: 0.0593174, EPL: -0.00122672, NFL: -0.00011532
        171: 'NBA',    # NBA: 0.054277, EPL: -0.00769591, NFL: -0.00183568
        172: 'NBA',    # NBA: -0.00738182, EPL: 0.00126581, NFL: 0.000531876
        173: 'NBA',    # NBA: 0.0581027, EPL: -0.0121944, NFL: -0.00183483
        175: 'NBA',    # NBA: 0.0504422, EPL: -0.0127049, NFL: -0.00183309
        177: 'NBA',    # NBA: 0.00104637, EPL: -0.00336229, NFL: -0.000191538
        178: 'NBA',    # NBA: 0.054363, EPL: -0.00242708, NFL: -0.0018357
        179: 'NBA',    # NBA: 0.0015858, EPL: 0.000702935
        181: 'NBA',    # NBA: 0.0701791, EPL: -0.000711832, NFL: -0.000123168
        187: 'NBA',    # NBA: 0.23674, EPL: -0.0465601, NFL: 0.0158337
        189: 'EPL',    # EPL: -0.00406252
        191: 'EPL',    # NBA: -3.92285, EPL: 0.617871, NFL: -2.48809
        192: 'NBA',    # NBA: 0.043373, EPL: -0.00683138, NFL: -0.0018359
        193: 'EPL',    # NBA: -25.782, EPL: -3.60521, NFL: -5.64857
        194: 'EPL',    # EPL: -0.000816973
        196: 'NBA',    # NBA: 0.00159681, EPL: 0.000559424
        197: 'EPL',    # NBA: -1.39436, EPL: -0.438678, NFL: -1.40288
        201: 'NBA',    # NBA: -0.0108703, EPL: 0.00567548, NFL: 0.000688346
        202: 'NBA',    # NBA: 0.00279876, EPL: -0.00536884, NFL: 0.00010553
        204: 'NFL',    # NBA: -0.0582234, EPL: -0.00546268, NFL: 0.00273393
        205: 'NFL',    # NBA: 1.71136, EPL: 0.182421, NFL: 1.59435
        206: 'NBA',    # NBA: 0.0668281, EPL: -0.00736011, NFL: -4.75822e-05
        207: 'NBA',    # NBA: 0.0527657, EPL: -0.00628494, NFL: -0.000756083
        208: 'NBA',    # NBA: 0.0329942, EPL: -0.00910205, NFL: -0.000348921
        209: 'NBA',    # NBA: -0.00736627, EPL: 0.000301095, NFL: 0.000647093
        210: 'NBA',    # NBA: 1.60345, EPL: -0.293229, NFL: 0.467796
        211: 'NBA',    # NBA: 1.41409, EPL: -0.664063, NFL: 0.422437
        213: 'NBA',    # NBA: 0.0409417, EPL: -0.0193769, NFL: -0.00106812
        215: 'EPL',    # NBA: -0.00153733, EPL: 0.00627947
        216: 'NFL',    # NBA: -0.0632884, EPL: -0.00194304, NFL: 0.00266722
        217: 'NBA',    # NBA: 0.00317017, EPL: 0.00324113, NFL: -5.171e-05
        219: 'NFL',    # NBA: -0.9618, EPL: -0.0940649, NFL: 1.32615
        220: 'EPL',    # NBA: -4.00546, EPL: 0.160441, NFL: -1.55814
        221: 'NBA',    # NBA: 0.0085603, EPL: 0.00449205, NFL: -0.000101153
        222: 'EPL',    # EPL: -0.00192885
        223: 'EPL',    # NBA: -5.08371, EPL: 0.709119, NFL: -1.66794
        224: 'NFL',    # NBA: 0.415878, EPL: -1.70702, NFL: 10.8193
        225: 'EPL',    # NBA: -2.10695, EPL: -0.679679, NFL: -2.0811
        226: 'NBA',    # NBA: -0.00735229, EPL: -0.010633, NFL: -0.000178607
        230: 'NBA',    # NBA: 0.329733, EPL: -0.69332, NFL: 0.168542
        231: 'EPL',    # NBA: -1.86121, EPL: -0.337417, NFL: -3.38152
        232: 'NBA',    # NBA: 6.88975, EPL: -0.229296, NFL: 0.412949
        234: 'NBA',    # NBA: 0.00136211, EPL: 0.0143301, NFL: 0.000361869
        235: 'NBA',    # NBA: 0.00089815, EPL: -0.000859684
        236: 'NBA',    # NBA: 0.00882884, EPL: -0.00160914, NFL: 0.000421872
        237: 'NBA',    # NBA: 0.0571384, EPL: -0.00626461, NFL: -0.00183036
        238: 'EPL',    # NBA: -11.6765, EPL: -4.18825, NFL: -2.62263
        241: 'NBA',    # NBA: 1.35027, EPL: -0.418818, NFL: 0.468425
        242: 'NBA',    # NBA: -0.00107452, EPL: 0.00322214, NFL: -1.08686e-05
        243: 'NBA',    # NBA: -0.00652929, EPL: -0.000979794, NFL: 0.00049079
        244: 'NBA',    # NBA: -0.00053507, EPL: 0.00248046, NFL: -7.14271e-05
        245: 'NFL',    # NBA: -31.0071, EPL: -2.48581, NFL: -0.715581
        246: 'NFL',    # NBA: -10.8554, EPL: -3.8437, NFL: -0.254066
        249: 'NFL',    # NBA: 0.366677, EPL: -0.440426, NFL: 0.741298
        250: 'NBA',    # NBA: 0.0585908, EPL: -0.00704407, NFL: -0.00183294
        252: 'EPL',    # EPL: -0.00240596
        253: 'NBA',    # NBA: 0.0578772, EPL: -0.0137094, NFL: -0.00183294
        254: 'EPL',    # NBA: -0.9618, EPL: -0.929817, NFL: -1.33151
        255: 'NBA'     # NBA: 0.0698644, EPL: -0.00341698, NFL: -0.000143775
    }

    for league in leagues_to_analyze:
        print(f"Processing league: {league.name} (Rolling Pred Threshold: {ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}, Rho Sensitivity Alpha: {LEAGUE_SENSITIVITY_ALPHAS[league]:.4f})")
        league_table_data = []
        predictions_for_copycat_analysis = []
        matches_without_odds = []

        # Get all miners committed to this league within the grace period
        league_miner_uids = []
        for uid in all_uids:
            if uid not in uid_to_best_league:
                continue
            if uid_to_best_league[uid] == league or uid_to_best_league[uid] == league.name:
                league_miner_uids.append(uid)
                uids_to_last_leagues[uid] = [league]
                uids_to_leagues_last_updated[uid] = dt.datetime.now()

            # Randomly select a subset of miners to commit to the league. UIDs 0-90 goto NBA. UIDs 91-180 goto NFL. UIDs 181-240 goto EPL.
            """
            if league == League.NBA and uid < 90:
                league_miner_uids.append(uid)
                uids_to_last_leagues[uid] = [league]
                uids_to_leagues_last_updated[uid] = dt.datetime.now()
            elif league == League.NFL and 90 <= uid < 180:
                league_miner_uids.append(uid)
                uids_to_last_leagues[uid] = [league]
                uids_to_leagues_last_updated[uid] = dt.datetime.now()
            elif league == League.EPL and 180 <= uid < 256:
                league_miner_uids.append(uid)
                uids_to_last_leagues[uid] = [league]
                uids_to_leagues_last_updated[uid] = dt.datetime.now()
            """

        for index, uid in enumerate(all_uids):
            total_score, rho = 0, 0
            predictions_with_match_data = []
            # Only process miners that are committed to the league
            if uid in league_miner_uids:
                hotkey = metagraph.hotkeys[uid]

                predictions_with_match_data = storage.get_miner_match_predictions(
                    miner_hotkey=hotkey,
                    miner_uid=uid,
                    league=league,
                    scored=True,
                    batchSize=(ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league] * 2)
                )

                if not predictions_with_match_data:
                    continue  # No predictions for this league, keep score as 0

                # Add eligible predictions to predictions_for_copycat_analysis
                predictions_for_copycat_analysis.extend([p for p in predictions_with_match_data if p.prediction.predictionDate.replace(tzinfo=timezone.utc) >= COPYCAT_PUNISHMENT_START_DATE])

                # Calculate rho
                rho = compute_significance_score(
                    num_miner_predictions=len(predictions_with_match_data),
                    num_threshold_predictions=ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league],
                    alpha=LEAGUE_SENSITIVITY_ALPHAS[league]
                )

                total_score = 0
                total_wrong_pred_pos_edge_penalty_preds = 0
                for pwmd in predictions_with_match_data:
                    log_prediction = random.random() < 0.1
                    log_prediction = False
                    #if pwmd.prediction.probability <= MIN_PROBABILITY:
                        #log_prediction = True
                    if log_prediction:
                        print(f"Randomly logged prediction for miner {uid} in league {league.name}:")
                        print(f"  • Number of predictions: {len(predictions_with_match_data)}")
                        print(f"  • League rolling threshold count: {ROLLING_PREDICTION_THRESHOLD_BY_LEAGUE[league]}")
                        print(f"  • Rho: {rho:.4f}")

                    # Grab the match odds from local db
                    match_odds = storage.get_match_odds(matchId=pwmd.prediction.matchId)
                    if match_odds is None or len(match_odds) == 0:
                        print(f"Odds were not found for matchId {pwmd.prediction.matchId}. Skipping calculation of this prediction.")
                        continue

                    # if predictionDate within 10 minutes of matchDate, calculate roi payout
                    if (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 10 and pwmd.prediction.predictionDate >= dt.datetime(2024, 12, 3, 0, 0, 0):
                        league_roi_counts[league][index] += 1
                        if pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner():
                            league_roi_payouts[league][index] += 100 * (pwmd.get_actual_winner_odds()-1)
                        else:
                            league_roi_payouts[league][index] -= 100

                    # Ensure prediction.matchDate is offset-aware
                    if pwmd.prediction.matchDate.tzinfo is None:
                        match_date = pwmd.prediction.matchDate.replace(tzinfo=dt.timezone.utc)
                    else:
                        match_date = pwmd.prediction.matchDate
                    # Ensure prediction.predictionDate is offset-aware
                    if pwmd.prediction.predictionDate.tzinfo is None:
                        prediction_date = pwmd.prediction.predictionDate.replace(tzinfo=dt.timezone.utc)
                    else:
                        prediction_date = pwmd.prediction.predictionDate

                    # Calculate time delta in minutes    
                    delta_t = min(MAX_PREDICTION_DAYS_THRESHOLD * 24 * 60, (match_date - prediction_date).total_seconds() / 60)
                    if log_prediction:
                        print(f"      • Time delta: {delta_t:.4f}")
                    
                    # Calculate closing line value
                    clv = calculate_clv(match_odds, pwmd, log_prediction)
                    if clv is None:
                        if (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 10:
                            t_interval = "T-10m"
                        elif (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 240:
                            t_interval = "T-4h"
                        elif (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 720:
                            t_interval = "T-12h"
                        elif (pwmd.prediction.matchDate - pwmd.prediction.predictionDate).total_seconds() / 60 <= 1440:
                            t_interval = "T-24h"
                        # only add to matches_without_odds if the match and t-interval are not already in the list
                        if (pwmd.prediction.matchId, t_interval) not in matches_without_odds:
                            matches_without_odds.append((pwmd.prediction.matchId, t_interval))
                        continue
                    elif log_prediction:
                        print(f"      • Closing line value: {clv:.4f}")

                    v = calculate_incentive_score(
                        delta_t=delta_t,
                        clv=clv,
                        gamma=GAMMA,
                        kappa=TRANSITION_KAPPA, 
                        beta=EXTREMIS_BETA
                    )
                    if log_prediction:
                        print(f"      • Incentive score (v): {v:.4f}")

                    # Get sigma, aka the closing edge
                    sigma = pwmd.prediction.closingEdge
                    #sigma, correct_winner_score = calculate_edge(
                    #    prediction_team=pwmd.prediction.get_predicted_team(),
                    #    prediction_prob=pwmd.prediction.probability,
                    #    actual_team=pwmd.get_actual_winner(),
                    #    closing_odds=pwmd.get_closing_odds_for_predicted_outcome(),
                    #)
                    if log_prediction:
                        print(f"      • Sigma (aka Closing Edge): {sigma:.4f}")
                    
                    # Calculate the Gaussian filter
                    gfilter = apply_gaussian_filter(pwmd)
                    if log_prediction:
                        print(f"      • Gaussian filter: {gfilter:.4f}")
                    
                    # Zero out all lay predictions, that is if the prediction probability is less than MIN_PROBABILITY
                    #if (pwmd.prediction.league in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.probability <= MIN_PROB_FOR_DRAWS) or pwmd.prediction.probability <= MIN_PROBABILITY:
                        #gfilter = 0
                    # Apply a penalty if the prediction was incorrect and the Gaussian filter is less than 1 and greater than 0
                    #elif pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner() and round(gfilter, 4) > 0 and gfilter < 1 and sigma < 0:
                    
                    # Apply a penalty if the prediction was incorrect and the Gaussian filter is less than 1 and greater than 0
                    if  (
                            (pwmd.prediction.probability > MIN_PROBABILITY and league not in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner())
                            or 
                            (pwmd.prediction.probability < MIN_PROBABILITY and league not in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner())
                        ) or \
                        (
                            (pwmd.prediction.probability > MIN_PROB_FOR_DRAWS and league in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() != pwmd.get_actual_winner())
                            or
                            (pwmd.prediction.probability < MIN_PROB_FOR_DRAWS and league in LEAGUES_ALLOWING_DRAWS and pwmd.prediction.get_predicted_team() == pwmd.get_actual_winner())
                        ) \
                        and round(gfilter, 4) > 0 and gfilter < 1 \
                        and sigma < 0:
                        
                        gfilter = max(MAX_GFILTER_FOR_WRONG_PREDICTION, gfilter)
                        if log_prediction:
                            print(f"      • Penalty applied for wrong prediction. gfilter: {gfilter:.4f}")
                
                    # Apply sigma and G (gaussian filter) to v
                    total_score += v * sigma * gfilter
                
                    if log_prediction:
                        print(f"      • Total prediction score: {(v * sigma * gfilter):.4f}")
                        print("-" * 50)

            final_score = rho * total_score
            league_scores[league][index] = final_score
            league_pred_counts[league][index] = len(predictions_with_match_data)
            total_lay_preds = len([
                pwmd for pwmd in predictions_with_match_data if pwmd.get_closing_odds_for_predicted_outcome() < 1 / pwmd.prediction.probability
            ])
            avg_pred_score = final_score / len(predictions_with_match_data) if len(predictions_with_match_data) > 0 else 0.0
            roi = league_roi_payouts[league][index] / (league_roi_counts[league][index] * 100) if league_roi_counts[league][index] > 0 else 0.0
            # Only log scores for miners committed to the league
            if uid in league_miner_uids:
                league_table_data.append([uid, round(final_score, 2), len(predictions_with_match_data), round(avg_pred_score, 4), round(rho, 2), str(total_lay_preds) + "/" + str(len(predictions_with_match_data)), str(round(roi*100, 2)) + "%"])
                #league_table_data.append([uid, final_score, len(predictions_with_match_data), rho, total_wrong_pred_pos_edge_penalty_preds])

        # Log league scores
        if league_table_data:
            print(f"\nScores for {league.name}:")
            print(tabulate(league_table_data, headers=['UID', 'Score', '# Predictions', 'Avg Pred Score', 'Rho', '# Lay Predictions', 'ROI'], tablefmt='grid'))
            #print(tabulate(league_table_data, headers=['UID', 'Score', '# Predictions', 'Rho', '# Wrong Pred Pos Edge'], tablefmt='grid'))
        else:
            print(f"No non-zero scores for {league.name}")

        if len(matches_without_odds) > 0:
            print(f"\n==============================================================================")
            print(f"Odds were not found for the following matches within {league.name}:")
            for mwo in matches_without_odds:
                print(f"{mwo[0]} - {mwo[1]}")
            print(f"==============================================================================")

        # Analyze league for copycat patterns
        earliest_match_date = min([p.prediction.matchDate for p in predictions_for_copycat_analysis], default=None)
        pred_matches = []
        if earliest_match_date is not None:
            pred_matches = storage.get_recently_completed_matches(earliest_match_date, league)
        ordered_matches = [(match.matchId, match.matchDate) for match in pred_matches]
        ordered_matches.sort(key=lambda x: x[1])  # Ensure chronological order
        suspicious_miners, penalties, exact_matches = copycat_controller.analyze_league(league, predictions_for_copycat_analysis, ordered_matches)
        #suspicious_miners, penalties, exact_matches = [], [], []
        # Print league results
        print(f"\n==============================================================================")
        print(f"Total suspicious miners in {league.name}: {len(suspicious_miners)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(suspicious_miners))}")

        print(f"\nTotal miners with exact matches in {league.name}: {len(exact_matches)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(exact_matches))}")
        
        print(f"\nTotal miners to penalize in {league.name}: {len(penalties)}")
        print(f"Miners: {', '.join(str(m) for m in sorted(penalties))}")
        print(f"==============================================================================")
        final_suspicious_miners.update(suspicious_miners)
        final_copycat_penalties.update(penalties)
        final_exact_matches.update(exact_matches)

    # Update all_scores with weighted sum of league scores for each miner
    print("************ Applying leagues scoring percentages to scores ************")
    for league, percentage in LEAGUE_SCORING_PERCENTAGES.items():
        print(f"  • {league}: {percentage*100}%")
    print("*************************************************************")
    all_scores = [0.0] * len(all_uids)
    for i in range(len(all_uids)):
        all_scores[i] = sum(league_scores[league][i] * LEAGUE_SCORING_PERCENTAGES[league] for league in ACTIVE_LEAGUES)

    # Check and penalize miners that are not committed to any active leagues
    #all_scores = check_and_apply_league_commitment_penalties(vali, all_scores, all_uids)
    # Apply penalties for miners that have not responded to prediction requests
    #all_scores = apply_no_prediction_response_penalties(vali, all_scores, all_uids)

    # Log final copycat results
    print(f"********************* Copycat Controller Findings  *********************")
    # Get a unique list of coldkeys from metagraph
    coldkeys = list(set(metagraph.coldkeys))
    for coldkey in coldkeys:
        uids_for_coldkey = []
        for miner_uid in final_suspicious_miners:
            if metagraph.coldkeys[miner_uid] == coldkey:
                if miner_uid in final_copycat_penalties:
                    miner_uid = f"{miner_uid} 💀"
                uids_for_coldkey.append(str(miner_uid))
        if len(uids_for_coldkey) > 0:
            print(f"\nColdkey: {coldkey}")
            print(f"Suspicious Miners: {', '.join(str(m) for m in sorted(uids_for_coldkey))}")

    print(f"\nTotal suspicious miners across all leagues: {len(final_suspicious_miners)}")
    if len(final_suspicious_miners) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_suspicious_miners))}")

    print(f"\nTotal miners with exact matches across all leagues: {len(final_exact_matches)}")
    if len(final_exact_matches) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_exact_matches))}")

    print(f"\nTotal miners to penalize across all leagues: {len(final_copycat_penalties)}")
    if len(final_copycat_penalties) > 0:
        print(f"Miners: {', '.join(str(m) for m in sorted(final_copycat_penalties))}")
    print(f"************************************************************************")

    for uid in final_copycat_penalties:
        # Apply the penalty to the score
        all_scores[uid] = COPYCAT_PENALTY_SCORE
    
    # Apply Pareto to all scores
    print(f"Applying Pareto distribution (mu: {PARETO_MU}, alpha: {PARETO_ALPHA}) to scores...")
    final_scores = apply_pareto(all_scores, all_uids, PARETO_MU, PARETO_ALPHA)

    # Prepare final scores table
    final_scores_table = []
    for i, uid in enumerate(all_uids):
        miner_league = ""
        miner_league_last_updated = ""
        if uid in uids_to_last_leagues and len(uids_to_last_leagues[uid]) > 0:
            miner_league = uids_to_last_leagues[uid][0].name
        if uid in uids_to_leagues_last_updated and uids_to_leagues_last_updated[uid] is not None:
            miner_league_last_updated = uids_to_leagues_last_updated[uid].strftime("%Y-%m-%d %H:%M")

        final_scores_table.append([uid, miner_league, miner_league_last_updated, all_scores[i], final_scores[i]])

    # Log final scores
    print("\nFinal Weighted Scores:")
    print(tabulate(final_scores_table, headers=['UID', 'League', 'Last Commitment', 'Pre-Pareto Score', 'Final Score'], tablefmt='grid'))

    # Create top 50 scores table
    top_scores_table = []
    # Sort the final scores in descending order. We need to sort the uids as well so they match
    top_scores, top_uids = zip(*sorted(zip(final_scores, all_uids), reverse=True))
    for i in range(75):
        miner_league = ""
        if top_uids[i] in uids_to_last_leagues and len(uids_to_last_leagues[top_uids[i]]) > 0:
            miner_league = uids_to_last_leagues[top_uids[i]][0].name
        is_cabal = ""
        if metagraph.coldkeys[top_uids[i]] in ["5CB89UjzLgWxZQNqV69AvTSS8AA5DP5QD6kudbDPDLxYcMvD", "5EUVRikyyXG1TbGMt8NpwVyzkKbdnDvVS2sRDDPAikEzAmbv", "5CQNGGzd1k6EP98X9Z4UAMVY1LwA38ipR5VhedyeBGrW3WpA", "5Enevga9siU68f1fpwreDzPFjXH4sH3Bv7hiHq4KjYMSHW4G"]:
            is_cabal = "✔"
        top_scores_table.append([top_uids[i], top_scores[i], miner_league, is_cabal, metagraph.coldkeys[top_uids[i]][:8]])
    print("\nTop 75 Miner Scores:")
    print(tabulate(top_scores_table, headers=['UID', 'Final Score', 'League', 'Cabal?', 'Coldkey'], tablefmt='grid'))

    # Log summary statistics
    non_zero_scores = [score for score in final_scores if score > 0]
    if non_zero_scores:
        print(f"\nScore Summary:")
        print(f"Number of miners with non-zero scores: {len(non_zero_scores)}")
        print(f"Average non-zero score: {sum(non_zero_scores) / len(non_zero_scores):.6f}")
        print(f"Highest score: {max(final_scores):.6f}")
        print(f"Lowest non-zero score: {min(non_zero_scores):.6f}")
    else:
        print("\nNo non-zero scores recorded.")

    # Generate graph of Pre-Pareto vs Final Pareto Scores
    graph_results(all_uids, all_scores, final_scores)

    cabal_uids = ""
    for uid in all_uids:
        if metagraph.coldkeys[uid] in []:
            cabal_uids += f"{uid},"
    print(f"\nCabal UIDs: {cabal_uids}")


def graph_results(all_uids, all_scores, final_scores):
    """
    Graphs the Pre-Pareto and Final Pareto scores with smaller, transparent dots and improved aesthetics.

    :param all_uids: List of unique identifiers for the miners
    :param all_scores: List of Pre-Pareto scores
    :param final_scores: List of Final Pareto scores
    """
    # Sort the miners based on Pre-Pareto Scores
    sorted_indices = np.argsort(all_scores)
    sorted_pre_pareto_scores = np.array(all_scores)[sorted_indices]
    sorted_final_pareto_scores = np.array(final_scores)[sorted_indices]

    # X-axis for the miners (from 0 to number of miners)
    x_axis = np.arange(len(all_uids))

    # Create the output directory if it doesn't exist
    output_dir = "tests/imgs"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the graph with smaller, transparent dots
    plt.figure(figsize=(12, 6))
    #plt.scatter(x_axis, sorted_pre_pareto_scores, label="Pre-Pareto Score", color='blue', s=10, alpha=0.6)
    plt.scatter(x_axis, sorted_final_pareto_scores, label="Final Pareto Score", color='orange', s=10, alpha=0.6)
    plt.xlabel("Miners (sorted by Pre-Pareto Score)")
    plt.ylabel("Scores")
    plt.title("Final Scores")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the graph as an image
    output_path = os.path.join(output_dir, "pareto_scores.png")
    plt.savefig(output_path)
    plt.close()

    return output_path


if "__main__" == __name__:
    calculate_incentives_and_update_scores()