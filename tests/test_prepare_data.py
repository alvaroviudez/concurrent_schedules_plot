from pathlib import Path

import pandas as pd
import pytest

from concurrent_schedules_plot.cs_plot import _validate_session, prepare_data

DATA_DIR = Path(__file__).parent / "data"


def _run(csv_name, canonical_schema):
    """Run a toy fixture through prepare_data with the canonical schema."""
    _, trials_df = prepare_data(data=DATA_DIR / csv_name, **canonical_schema)
    return trials_df


def test_basic_three_trials_alternating_reinforcement(canonical_schema):
    result = _run("basic.csv", canonical_schema)

    expected = pd.DataFrame({
        "Trial": [1, 2, 3],
        "Responses A": [2, 4, 1],
        "Responses B": [2, 1, 3],
        "Reinforcement A": [0, 1, 0],
        "Reinforcement B": [1, 0, 1],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_single_response_no_reinforcement_yet(canonical_schema):
    result = _run("single_response.csv", canonical_schema)

    expected = pd.DataFrame({
        "Trial": [1],
        "Responses A": [1],
        "Responses B": [0],
        "Reinforcement A": [0],
        "Reinforcement B": [0],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_schedule_a_never_reinforced(canonical_schema):
    result = _run("no_reinforcers_a.csv", canonical_schema)

    expected = pd.DataFrame({
        "Trial": [1, 2],
        "Responses A": [2, 3],
        "Responses B": [2, 1],
        "Reinforcement A": [0, 0],
        "Reinforcement B": [1, 1],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_trailing_responses_after_prior_reinforcement(canonical_schema):
    result = _run("trailing_responses.csv", canonical_schema)

    expected = pd.DataFrame({
        "Trial": [1, 2],
        "Responses A": [2, 1],
        "Responses B": [1, 3],
        "Reinforcement A": [1, 0],
        "Reinforcement B": [0, 0],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_time_unit_seconds_converts_to_milliseconds(canonical_schema):
    """time_unit='s' should scale the time column by 1000, not leave it untouched."""
    raw = pd.read_csv(DATA_DIR / "basic.csv", sep=";")

    filtered_df, _ = prepare_data(
        data=DATA_DIR / "basic.csv",
        time_unit="s",
        **canonical_schema,
    )

    expected = pd.DataFrame({
        "Time": raw["time_ms"] * 1000,
        "Responses A": raw["resp_a"],
        "Responses B": raw["resp_b"],
        "Cumulative Reinforcement A": raw["reinf_a"],
        "Cumulative Reinforcement B": raw["reinf_b"],
    })

    pd.testing.assert_frame_equal(filtered_df.reset_index(drop=True), expected)


def test_time_unit_minutes_converts_to_milliseconds(canonical_schema):
    """time_unit='min' should scale the time column by 60000, not leave it untouched."""
    raw = pd.read_csv(f"{DATA_DIR}/basic.csv", sep=";")

    filtered_df, _ = prepare_data(
        data=f"{DATA_DIR}/basic.csv",
        time_unit="min",
        **canonical_schema,
    )

    expected = pd.DataFrame({
        "Time": raw["time_ms"] * 60000,
        "Responses A": raw["resp_a"],
        "Responses B": raw["resp_b"],
        "Cumulative Reinforcement A": raw["reinf_a"],
        "Cumulative Reinforcement B": raw["reinf_b"],
    })

    pd.testing.assert_frame_equal(filtered_df.reset_index(drop=True), expected)


def test_validate_session_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["time_ms", "resp_a", "resp_b", "reinf_a", "reinf_b"])
    with pytest.raises(ValueError, match="empty"):
        _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")


def test_validate_session_rejects_non_numeric_column():
    df = pd.DataFrame({
        "time_ms": [0, 1000, 2000],
        "resp_a": ["a", "b", "c"],
        "resp_b": [0, 1, 2],
        "reinf_a": [0, 0, 0],
        "reinf_b": [0, 0, 0],
    })
    with pytest.raises(ValueError, match="resp_a.*must be numeric"):
        _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")


def test_validate_session_rejects_decreasing_count():
    df = pd.DataFrame({
        "time_ms": [0, 1000, 2000, 3000],
        "resp_a": [0, 5, 2, 6],
        "resp_b": [0, 1, 2, 3],
        "reinf_a": [0, 1, 1, 1],
        "reinf_b": [0, 0, 0, 0],
    })
    with pytest.raises(ValueError, match="non-decreasing integer count"):
        _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")


def test_validate_session_rejects_fractional_count():
    df = pd.DataFrame({
        "time_ms": [0, 1000, 2000],
        "resp_a": [0.0, 1.5, 3.0],
        "resp_b": [0, 0, 0],
        "reinf_a": [0, 0, 0],
        "reinf_b": [0, 0, 0],
    })
    with pytest.raises(ValueError, match="non-decreasing integer count"):
        _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")


def test_validate_session_rejects_non_increasing_time():
    df = pd.DataFrame({
        "time_ms": [0, 1000, 1000, 3000],
        "resp_a": [0, 1, 2, 3],
        "resp_b": [0, 0, 0, 0],
        "reinf_a": [0, 0, 0, 0],
        "reinf_b": [0, 0, 0, 0],
    })
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")


def test_validate_session_accepts_large_steps_between_rows():
    """Response counts may jump by more than one unit between consecutive rows
    — e.g. when the logger samples at fixed intervals rather than one row per
    event. This does not apply to reinforcer columns — see
    test_validate_session_rejects_multiple_reinforcers_in_one_row."""
    df = pd.DataFrame({
        "time_ms": [0, 1000, 2000],
        "resp_a": [0, 409, 774],
        "resp_b": [0, 0, 0],
        "reinf_a": [0, 1, 1],
        "reinf_b": [0, 0, 0],
    })
    _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")


def test_validate_session_rejects_multiple_reinforcers_in_one_row():
    """Unlike response counts, reinforcer counts must step by at most 1 per
    row: prepare_data marks a trial boundary wherever the reinforcer count
    changes, so a jump of more than one would silently fold several
    reinforcement events — and the responses interleaved between them —
    into a single trial."""
    df = pd.DataFrame({
        "time_ms": [0, 1000, 2000],
        "resp_a": [0, 409, 774],
        "resp_b": [0, 0, 0],
        "reinf_a": [0, 3, 3],
        "reinf_b": [0, 0, 0],
    })
    with pytest.raises(ValueError, match="must not deliver more than one reinforcer"):
        _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")


def test_validate_session_accepts_all_existing_fixtures():
    paths = [DATA_DIR / name for name in [
        "basic.csv",
        "single_response.csv",
        "trailing_responses.csv",
        "no_reinforcers_a.csv",
    ]]
    paths.append(DATA_DIR.parent.parent / "examples" / "sample_session.csv")

    for path in paths:
        df = pd.read_csv(path, sep=";")
        _validate_session(df, "resp_a", "resp_b", "reinf_a", "reinf_b", "time_ms")
