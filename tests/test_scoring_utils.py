import unittest
import threading
import math
import datetime as dt
import numpy as np

import vali_utils.scoring_utils as scoring_utils
from common.data import Sport, League, ProbabilityChoice, MatchPrediction, MatchPredictionWithMatchData
from neurons.validator import Validator
from base.validator import BaseValidatorNeuron


# config = BaseValidatorNeuron.config()
# config.wallet._mock = True
# config.metagraph._mock = True
# config.subtensor._mock = True
# test_vali = Validator(config)
test_vali = {}

class TestCalculateEdge(unittest.TestCase):
    def test_correct_prediction_with_closing_odds(self):
        prediction_team = 'A'
        prediction_prob = 0.8
        actual_team = 'A'
        closing_odds = 1.5
        expected_edge = 0.25
        expected_correctness = 1
        edge, correctness = scoring_utils.calculate_edge(prediction_team, prediction_prob, actual_team, closing_odds)
        self.assertAlmostEqual(edge, expected_edge)
        self.assertEqual(correctness, expected_correctness)

    def test_incorrect_prediction_with_closing_odds(self):
        prediction_team = 'A'
        prediction_prob = 0.8
        actual_team = 'B'
        closing_odds = 1.5
        expected_edge = -0.25
        expected_correctness = 0
        edge, correctness = scoring_utils.calculate_edge(prediction_team, prediction_prob, actual_team, closing_odds)
        self.assertAlmostEqual(edge, expected_edge)
        self.assertEqual(correctness, expected_correctness)

    def test_correct_prediction_no_closing_odds(self):
        prediction_team = 'A'
        prediction_prob = 0.8
        actual_team = 'A'
        closing_odds = None
        expected_edge = 0.0
        expected_correctness = 0
        edge, correctness = scoring_utils.calculate_edge(prediction_team, prediction_prob, actual_team, closing_odds)
        self.assertEqual(edge, expected_edge)
        self.assertEqual(correctness, expected_correctness)

    def test_incorrect_prediction_no_closing_odds(self):
        prediction_team = 'A'
        prediction_prob = 0.8
        actual_team = 'B'
        closing_odds = None
        expected_edge = 0.0
        expected_correctness = 0
        edge, correctness = scoring_utils.calculate_edge(prediction_team, prediction_prob, actual_team, closing_odds)
        self.assertEqual(edge, expected_edge)
        self.assertEqual(correctness, expected_correctness)

    def test_correct_prediction_with_high_closing_odds(self):
        prediction_team = 'A'
        prediction_prob = 0.8
        actual_team = 'A'
        closing_odds = 5.0
        expected_edge = 3.75
        expected_correctness = 1
        edge, correctness = scoring_utils.calculate_edge(prediction_team, prediction_prob, actual_team, closing_odds)
        self.assertEqual(edge, expected_edge)
        self.assertEqual(correctness, expected_correctness)

    def test_incorrect_prediction_with_high_closing_odds(self):
        prediction_team = 'A'
        prediction_prob = 0.8
        actual_team = 'B'
        closing_odds = 5.0
        expected_edge = -3.75
        expected_correctness = 0
        edge, correctness = scoring_utils.calculate_edge(prediction_team, prediction_prob, actual_team, closing_odds)
        self.assertEqual(edge, expected_edge)
        self.assertEqual(correctness, expected_correctness)


class TestComputeSignificanceScore(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.alpha = 0.025
        self.num_threshold_predictions = 60

    def test_significance_score_high_predictions(self):
        num_miner_predictions = 100
        expected_score = 0.7311
        score = scoring_utils.compute_significance_score(num_miner_predictions, self.num_threshold_predictions, self.alpha)
        self.assertEqual(round(score, 4), expected_score)

    def test_significance_score_low_predictions(self):
        num_miner_predictions = 20
        expected_score = 0.2689
        score = scoring_utils.compute_significance_score(num_miner_predictions, self.num_threshold_predictions, self.alpha)
        self.assertEqual(round(score, 4), expected_score)

    def test_significance_score_equal_predictions(self):
        num_miner_predictions = 60
        expected_score = 0.5
        score = scoring_utils.compute_significance_score(num_miner_predictions, self.num_threshold_predictions, self.alpha)
        self.assertEqual(round(score, 4), expected_score)

    def test_significance_score_zero_predictions(self):
        num_miner_predictions = 0
        expected_score = 0.1824
        score = scoring_utils.compute_significance_score(num_miner_predictions, self.num_threshold_predictions, self.alpha)
        self.assertEqual(round(score, 4), expected_score)


class TestApplyGaussianFilter(unittest.TestCase):

    def setUp(self):
        self.match_prediction_home = MatchPrediction(
            matchId=1234,
            matchDate=dt.datetime.strptime("2024-08-20", "%Y-%m-%d"),
            sport=Sport.SOCCER,
            league=League.MLS,
            homeTeamName="Inter Miami",
            awayTeamName="Miami Fusion",
            probability=0.8,
            probabilityChoice=ProbabilityChoice.HOMETEAM
        )

        self.match_prediction_draw = MatchPrediction(
            matchId=1234,
            matchDate=dt.datetime.strptime("2024-08-20", "%Y-%m-%d"),
            sport=Sport.SOCCER,
            league=League.MLS,
            homeTeamName="Inter Miami",
            awayTeamName="Miami Fusion",
            probability=0.4,
            probabilityChoice=ProbabilityChoice.DRAW
        )

    def test_prediction_matches_actual_winner(self):
        pwmd = MatchPredictionWithMatchData(
            prediction=self.match_prediction_home,
            actualHomeTeamScore=2,
            actualAwayTeamScore=1,
            homeTeamOdds=1.5,
            awayTeamOdds=2.5,
            drawOdds=3.0
        )
        result = scoring_utils.apply_gaussian_filter(pwmd)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.8)
        self.assertLessEqual(result, 1)

    def test_prediction_matches_actual_draw(self):
        pwmd = MatchPredictionWithMatchData(
            prediction=self.match_prediction_draw,
            actualHomeTeamScore=1,
            actualAwayTeamScore=1,
            homeTeamOdds=1.5,
            awayTeamOdds=2.5,
            drawOdds=3.0
        )
        result = scoring_utils.apply_gaussian_filter(pwmd)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.8)
        self.assertLessEqual(result, 1)

    def test_prediction_does_not_match_actual_winner(self):
        pwmd = MatchPredictionWithMatchData(
            prediction=self.match_prediction_home,
            actualHomeTeamScore=1,
            actualAwayTeamScore=2,
            homeTeamOdds=1.5,
            awayTeamOdds=2.5,
            drawOdds=3.0
        )
        result = scoring_utils.apply_gaussian_filter(pwmd)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.8)
        self.assertLessEqual(result, 1)

    def test_high_closing_odds(self):
        pwmd = MatchPredictionWithMatchData(
            prediction=self.match_prediction_home,
            actualHomeTeamScore=2,
            actualAwayTeamScore=1,
            homeTeamOdds=10.0,
            awayTeamOdds=2.5,
            drawOdds=3.0
        )
        result = scoring_utils.apply_gaussian_filter(pwmd)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.9)
        self.assertLessEqual(result, 1)

    def test_high_closing_odds_with_incorrect_prediction(self):
        pwmd = MatchPredictionWithMatchData(
            prediction=self.match_prediction_home,
            actualHomeTeamScore=1,
            actualAwayTeamScore=2,
            homeTeamOdds=10.0,
            awayTeamOdds=2.5,
            drawOdds=3.0
        )
        result = scoring_utils.apply_gaussian_filter(pwmd)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.9)
        self.assertLessEqual(result, 1)

    def test_low_closing_odds(self):
        pwmd = MatchPredictionWithMatchData(
            prediction=self.match_prediction_home,
            actualHomeTeamScore=2,
            actualAwayTeamScore=1,
            homeTeamOdds=1.01,
            awayTeamOdds=2.5,
            drawOdds=3.0
        )
        result = scoring_utils.apply_gaussian_filter(pwmd)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 0.1)


class TestCalculateIncentiveScore(unittest.TestCase):
    def test_typical_values(self):
        self.assertEqual(round(scoring_utils.calculate_incentive_score(30, 0.5, 0.1, 1.0, 0.2), 4), 0.4551)
        self.assertEqual(round(scoring_utils.calculate_incentive_score(60, 1.0, 0.1, 1.0, 0.2), 4), 0.5)

    def test_zero_delta_t(self):
        self.assertEqual(round(scoring_utils.calculate_incentive_score(0, 0.5, 0.1, 1.0, 0.2), 4), 1)

    def test_large_delta_t(self):
        self.assertEqual(round(scoring_utils.calculate_incentive_score(1000, 0.5, 0.1, 1.0, 0.2), 4), 0.4265)

    def test_extreme_clv(self):
        self.assertEqual(round(scoring_utils.calculate_incentive_score(30, 10.0, 0.1, 1.0, 0.2), 4), 0.2399)
        self.assertEqual(round(scoring_utils.calculate_incentive_score(30, -10.0, 0.1, 1.0, 0.2), 4), 0.8)


class TestCalculateCLV(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.match_prediction = MatchPrediction(
            matchId=1234,
            matchDate=dt.datetime.strptime("2024-08-20", "%Y-%m-%d"),
            predictionDate=dt.datetime.strptime("2024-08-20", "%Y-%m-%d"),
            sport=Sport.SOCCER,
            league=League.MLS,
            homeTeamName="Inter Miami",
            awayTeamName="Miami Fusion",
            probability=0.8,
            probabilityChoice=ProbabilityChoice.HOMETEAM
        )
        self.pwmd = MatchPredictionWithMatchData(
            prediction=self.match_prediction,
            actualHomeTeamScore=2,
            actualAwayTeamScore=1,
            homeTeamOdds=1.5,
            awayTeamOdds=2.5,
            drawOdds=3.0
        )

    def test_typical_values(self):
        match_odds = [["", 1.7, 2.0, 3.5, dt.datetime.strptime("2024-08-20", "%Y-%m-%d")]]
        result = scoring_utils.calculate_clv(match_odds, self.pwmd, True)
        print(result)
        self.assertEqual(result, 0.2)

    def test_closing_odds_none(self):
        match_odds = [["", 1.7, 2.0, 3.5, dt.datetime.strptime("2024-08-19", "%Y-%m-%d")]]
        result = scoring_utils.calculate_clv(match_odds, self.pwmd, True)
        self.assertEqual(result, 0)


class TestFindClosestOdds(unittest.TestCase):
    def test_typical_values(self):
        match_odds = [["", 1.5, 2.0, 2.5, dt.datetime.strptime("2024-08-20", "%Y-%m-%d")]]
        prediction_time = dt.datetime.strptime("2024-08-20", "%Y-%m-%d")
        choice = ProbabilityChoice.HOMETEAM
        result = scoring_utils.find_closest_odds(match_odds, prediction_time, choice, True)
        self.assertEqual(result, 1.5)

    def test_datetime_not_found(self):
        match_odds = [["", 1.5, 2.0, 2.5, dt.datetime.strptime("2024-08-19", "%Y-%m-%d")]]
        prediction_time = dt.datetime.strptime("2024-08-18", "%Y-%m-%d")
        choice = ProbabilityChoice.HOMETEAM
        result = scoring_utils.find_closest_odds(match_odds, prediction_time, choice, True)
        self.assertIsNone(result)

    
class TestApplyPareto(unittest.TestCase):
    def test_typical_values(self):
        mu = 1
        alpha = 1.0
        all_uids = list(range(73))
        all_scores = [
            -3, -2.5231, -2.91235, -2.99362, -3, 0.24, -3, -3, -3, -3,
            -3, -3, -3, -2.0793, -3, -3, -3, -3, -3, -3,
            -3, 0.0296598, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, 0, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -2.52537, -3, -3, -2.32464, -3, -3, 2.13207, -3, -3,
            -3, -3, -3, -3, -3, 1.38769, -3, -3, -1.93823, -1.63695,
            1.12741, -3, -3, -3, -3
        ]
        max_score = max(all_scores)
        
        result = scoring_utils.apply_pareto(all_scores, all_uids, mu, alpha)
        max_result = max(result)
        
        top_10_percent = sorted(result)[-14:]  # top 10%
        mid_10_percent = sorted(result)[35:45]  # mid 10%
        bottom_10_percent = sorted(result)[:14]  # bottom 10%

        self.assertTrue(np.all(top_10_percent >= np.median(mid_10_percent)))
        self.assertTrue(np.all(mid_10_percent >= np.median(bottom_10_percent)))
        self.assertTrue(all_scores.index(max_result),  all_scores.index(max_score))


class TestCheckAndApplyLeagueCommitmentPenalties(unittest.TestCase):
    def test_typical_league_commitment(self):
        all_uids = list(range(73))  # 0 to 72, matching the UIDs in the log
        all_scores = [
            -3, -2.5231, -2.91235, -2.99362, -3, 0.24, -3, -3, -3, -3,
            -3, -3, -3, -2.0793, -3, -3, -3, -3, -3, -3,
            -3, 0.0296598, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, 0, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -2.52537, -3, -3, -2.32464, -3, -3, 2.13207, -3, -3,
            -3, -3, -3, -3, -3, 1.38769, -3, -3, -1.93823, -1.63695,
            1.12741, -3, -3, -3, -3
        ]
        # result = scoring_utils.check_and_apply_league_commitment_penalties(test_vali, all_scores, all_uids)
        # self.assertEqual(result, expected)

    def test_no_penalties(self):
        all_uids = list(range(73))  # 0 to 72, matching the UIDs in the log
        all_scores = [
            -3, -2.5231, -2.91235, -2.99362, -3, 0.24, -3, -3, -3, -3,
            -3, -3, -3, -2.0793, -3, -3, -3, -3, -3, -3,
            -3, 0.0296598, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, 0, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -2.52537, -3, -3, -2.32464, -3, -3, 2.13207, -3, -3,
            -3, -3, -3, -3, -3, 1.38769, -3, -3, -1.93823, -1.63695,
            1.12741, -3, -3, -3, -3
        ]
        # result = scoring_utils.check_and_apply_league_commitment_penalties(test_vali, all_scores, all_uids)
        # self.assertEqual(result, expected)


class TestCalculateIncentivesAndUpdateScores(unittest.TestCase):
    def test_typical_scenario(self):
        # scoring_utils.calculate_incentives_and_update_scores(test_vali)
        pass


class TestUpdateMinerScores(unittest.TestCase):
    def test_typical_scenario(self):
        all_uids = list(range(73))
        all_scores = [
            -3, -2.5231, -2.91235, -2.99362, -3, 0.24, -3, -3, -3, -3,
            -3, -3, -3, -2.0793, -3, -3, -3, -3, -3, -3,
            -3, 0.0296598, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, 0, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -2.52537, -3, -3, -2.32464, -3, -3, 2.13207, -3, -3,
            -3, -3, -3, -3, -3, 1.38769, -3, -3, -1.93823, -1.63695,
            1.12741, -3, -3, -3, -3
        ]
        # scoring_utils.update_miner_scores(test_vali, all_scores, all_uids)

    def test_no_miners(self):
        all_uids = []
        all_scores = []
        # scoring_utils.update_miner_scores(test_vali, all_scores, all_uids)


if __name__ == '__main__':
    unittest.main()