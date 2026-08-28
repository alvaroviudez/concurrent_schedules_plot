from pathlib import Path

import pandas as pd

from concurrent_schedules_plot.cs_plot import prepare_data

DATA_DIR = Path(__file__).parent / "data"


def _run(csv_name):
    """Run a toy fixture through prepare_data with the canonical schema."""
    _, trials_df = prepare_data(
        data=DATA_DIR / csv_name,
        sep=";",
        resp_a_col="resp_a",
        resp_b_col="resp_b",
        reinf_a_col="reinf_a",
        reinf_b_col="reinf_b",
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


def test_time_unit_seconds_converts_to_milliseconds():
    """time_unit='s' should scale the time column by 1000, not leave it untouched."""
    raw = pd.read_csv(f"{DATA_DIR}/basic.csv", sep=";")

    filtered_df, _ = prepare_data(
        data=f"{DATA_DIR}/basic.csv",
        sep=";",
        resp_a_col="resp_a",
        resp_b_col="resp_b",
        reinf_a_col="reinf_a",
        reinf_b_col="reinf_b",
        time_col="time_ms",
        time_unit="s",
    )

    expected = pd.DataFrame({
        "Time": raw["time_ms"] * 1000,
        "Responses A": raw["resp_a"],
        "Responses B": raw["resp_b"],
        "Reinforcement A": raw["reinf_a"],
        "Reinforcement B": raw["reinf_b"],
    })

    pd.testing.assert_frame_equal(filtered_df.reset_index(drop=True), expected)


def test_time_unit_minutes_converts_to_milliseconds():
    """time_unit='min' should scale the time column by 60000, not leave it untouched."""
    raw = pd.read_csv(f"{DATA_DIR}/basic.csv", sep=";")

    filtered_df, _ = prepare_data(
        data=f"{DATA_DIR}/basic.csv",
        sep=";",
        resp_a_col="resp_a",
        resp_b_col="resp_b",
        reinf_a_col="reinf_a",
        reinf_b_col="reinf_b",
        time_col="time_ms",
        time_unit="min",
    )

    expected = pd.DataFrame({
        "Time": raw["time_ms"] * 60000,
        "Responses A": raw["resp_a"],
        "Responses B": raw["resp_b"],
        "Reinforcement A": raw["reinf_a"],
        "Reinforcement B": raw["reinf_b"],
    })

    pd.testing.assert_frame_equal(filtered_df.reset_index(drop=True), expected)