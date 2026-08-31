from pathlib import Path

import pandas as pd
import pytest

from concurrent_schedules_plot.cs_plot import (
    back_to_back_bar_plot,
    cumulative_records_plot,
    prepare_data,
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.mpl_image_compare(tolerance=10)
def test_cumulative_records_plot_basic():
    filtered_df, _ = prepare_data(
        data=DATA_DIR / "basic.csv",
        sep=";",
        resp_a_col="resp_a",
        resp_b_col="resp_b",
        reinf_a_col="reinf_a",
        reinf_b_col="reinf_b",
        time_col="time_ms",
    )
    fig, _ax = cumulative_records_plot(
        filtered_df=filtered_df,
        label_a="A",
        label_b="B",
    )
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_back_to_back_bar_plot_basic():
    _, trials_df = prepare_data(
        data=DATA_DIR / "basic.csv",
        sep=";",
        resp_a_col="resp_a",
        resp_b_col="resp_b",
        reinf_a_col="reinf_a",
        reinf_b_col="reinf_b",
        time_col="time_ms",
    )
    fig, _ax = back_to_back_bar_plot(
        trials_df=trials_df,
        step=1,
        label_a="A",
        label_b="B",
    )
    return fig


def test_cumulative_records_plot_xticks_are_minutes():
    """X-axis tick labels should read as minutes when Time is in milliseconds."""
    filtered_df = pd.DataFrame({
        "Time": [0, 60000, 120000, 180000, 240000, 300000],
        "Responses A": [0, 2, 4, 6, 8, 10],
        "Responses B": [0, 1, 3, 5, 7, 9],
        "Cumulative Reinforcement A": [0, 0, 1, 0, 0, 1],
        "Cumulative Reinforcement B": [0, 1, 0, 0, 1, 0],
    })

    _fig, ax = cumulative_records_plot(filtered_df)
    labels = [t.get_text() for t in ax.get_xticklabels()]

    assert labels == ["0", "1", "2", "3", "4", "5"]


def test_cumulative_records_plot_includes_final_tick_at_exact_multiple():
    """A session whose length is an exact multiple of the tick step must
    still show its final tick — np.arange(0, max, step) would silently
    drop it since np.arange excludes its stop value."""
    filtered_df = pd.DataFrame({
        "Time": [0, 300000, 600000],
        "Responses A": [0, 5, 10],
        "Responses B": [0, 3, 6],
        "Cumulative Reinforcement A": [0, 0, 1],
        "Cumulative Reinforcement B": [0, 1, 1],
    })

    _fig, ax = cumulative_records_plot(filtered_df)
    labels = [t.get_text() for t in ax.get_xticklabels()]

    assert labels[-1] == "10"


def test_cumulative_records_plot_no_phantom_tick_at_start():
    """The first row must not be flagged as a reinforcement event unless it
    actually is one — a naive != comparison against the shifted column treats
    NaN as different from anything, marking row 0 as reinforced by default."""
    filtered_df = pd.DataFrame({
        "Time": [0, 1000, 2000, 3000, 4000, 5000, 6000],
        "Responses A": [0, 1, 1, 2, 2, 3, 3],
        "Responses B": [0, 0, 1, 1, 2, 2, 3],
        "Cumulative Reinforcement A": [0, 0, 0, 0, 0, 0, 0],
        "Cumulative Reinforcement B": [0, 0, 0, 0, 1, 1, 2],
    })

    _fig, ax = cumulative_records_plot(filtered_df)

    n_ticks_a = len(ax.collections[0].get_offsets())
    n_ticks_b = len(ax.collections[1].get_offsets())

    assert n_ticks_a == 0
    assert n_ticks_b == 2


def test_back_to_back_bar_plot_rejects_all_zero_responses():
    """A trials_df where no trial has any response leaves the x-axis with
    no range — matplotlib would silently expand a singular (0, 0) xlim
    rather than fail, so this must be rejected explicitly."""
    trials_df = pd.DataFrame({
        "Trial": [1],
        "Responses A": [0],
        "Responses B": [0],
        "Reinforcement A": [0],
        "Reinforcement B": [0],
    })
    with pytest.raises(ValueError, match="no recorded responses"):
        back_to_back_bar_plot(trials_df)
