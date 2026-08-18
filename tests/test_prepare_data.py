import pandas as pd

from concurrent_schedules_plot.cs_plot import prepare_data

DATA_DIR = "tests/data"


def _run(csv_name):
    """Run a toy fixture through prepare_data with the canonical schema."""
    _, trials_df = prepare_data(
        data=f"{DATA_DIR}/{csv_name}",
        sep=";",
        rs_a_col="resp_a",
        rs_b_col="resp_b",
        ref_a_col="reinf_a",
        ref_b_col="reinf_b",
        time_col="time_ms",
    )
    return trials_df


def test_basic_three_trials_alternating_reinforcement():
    result = _run("basic.csv")

    expected = pd.DataFrame({
        "Trial": [1, 2, 3],
        "Responses A": [2, 4, 1],
        "Responses B": [2, 1, 3],
        "Reinforcement A": [0, 1, 0],
        "Reinforcement B": [1, 0, 1],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_single_response_no_reinforcement_yet():
    result = _run("single_response.csv")

    expected = pd.DataFrame({
        "Trial": [1],
        "Responses A": [1],
        "Responses B": [0],
        "Reinforcement A": [0],
        "Reinforcement B": [0],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_schedule_a_never_reinforced():
    result = _run("no_reinforcers_a.csv")

    expected = pd.DataFrame({
        "Trial": [1, 2],
        "Responses A": [2, 3],
        "Responses B": [2, 1],
        "Reinforcement A": [0, 0],
        "Reinforcement B": [1, 1],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_trailing_responses_after_prior_reinforcement():
    result = _run("trailing_responses.csv")

    expected = pd.DataFrame({
        "Trial": [1, 2],
        "Responses A": [2, 1],
        "Responses B": [1, 3],
        "Reinforcement A": [1, 0],
        "Reinforcement B": [0, 0],
    })

    pd.testing.assert_frame_equal(result, expected)
