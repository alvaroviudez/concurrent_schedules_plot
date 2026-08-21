import pytest
from pathlib import Path

from concurrent_schedules_plot.cs_plot import (
    cumulative_records_plot,
    plot_cs,
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
    fig, ax = cumulative_records_plot(
        filtered_df=filtered_df,
        label_a="A",
        label_b="B",
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cs_basic():
    _, trials_df = prepare_data(
        data=DATA_DIR / "basic.csv",
        sep=";",
        resp_a_col="resp_a",
        resp_b_col="resp_b",
        reinf_a_col="reinf_a",
        reinf_b_col="reinf_b",
        time_col="time_ms",
    )
    fig, ax = plot_cs(
        trials_df=trials_df,
        step=1,
        label_a="A",
        label_b="B",
    )
    return fig
