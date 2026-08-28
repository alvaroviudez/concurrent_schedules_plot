from pathlib import Path

import pandas as pd
import pytest

from concurrent_schedules_plot.cs_plot import (
    back_to_back_bar_plot,
    cumulative_records_plot,
    prepare_data,
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.mpl_image_compare
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


@pytest.mark.mpl_image_compare
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
        "Reinforcement A": [0, 0, 1, 0, 0, 1],
        "Reinforcement B": [0, 1, 0, 0, 1, 0],
    })

    _fig, ax = cumulative_records_plot(filtered_df)
    labels = [t.get_text() for t in ax.get_xticklabels()]

    assert labels == ["0", "1", "2", "3", "4"]