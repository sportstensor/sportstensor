import math
import datetime as dt


MAX_PREDICTION_DAYS_THRESHOLD = 1
gamma = 0.00125

def calc_time_component(match_date, prediction_date, label):
    delta_t = min(MAX_PREDICTION_DAYS_THRESHOLD * 24 * 60, (match_date - prediction_date).total_seconds() / 60)
    time_component = math.exp(-gamma * delta_t)
    print(f"{label}: delta_t: {delta_t}, time_component: {time_component}")
    return time_component

def pretty_print(label, score, score_sum):
    print(f"{label}: {round(score / score_sum * 100, 2)}%")

def test_time_component():

    # T-10m == 75%
    # T-4h == 15%
    # T-12h == 7%
    # T-24h == 3%

    match_date = dt.datetime(2024, 10, 30, 12, 0, 0)
    pred_date_10m = dt.datetime(2024, 10, 30, 11, 59, 0)
    pred_date_4h = dt.datetime(2024, 10, 30, 8, 10, 0)
    pred_date_12h = dt.datetime(2024, 10, 30, 0, 10, 0)
    pred_date_24h = dt.datetime(2024, 10, 29, 12, 10, 0)

    t_10m = calc_time_component(match_date, pred_date_10m, "T-10m")
    t_4h = calc_time_component(match_date, pred_date_4h, "T-4h")
    t_12h = calc_time_component(match_date, pred_date_12h, "T-12h")
    t_24h = calc_time_component(match_date, pred_date_24h, "T-24h")
    score_sum = t_24h + t_12h + t_4h + t_10m
    print()

    pretty_print("T-10m", t_10m, score_sum)
    pretty_print("T-4h", t_4h, score_sum)
    pretty_print("T-12h", t_12h, score_sum)
    pretty_print("T-24h", t_24h, score_sum)
    print()

    print("1 - time_component:")
    print(1- t_10m, 1 - t_4h, 1 - t_12h, 1 - t_24h)
    print()

    score_sum = 4 - (t_24h + t_12h + t_4h + t_10m)

    pretty_print("T-10m", 1 - t_10m, score_sum)
    pretty_print("T-4h", 1 - t_4h, score_sum)
    pretty_print("T-12h", 1 - t_12h, score_sum)
    pretty_print("T-24h", 1 - t_24h, score_sum)


if __name__ == "__main__":
    test_time_component()