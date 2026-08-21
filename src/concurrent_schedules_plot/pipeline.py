"""Orchestration pipeline for concurrent schedules analysis.

Ties together data preparation and plotting into a single entry point:
reads a raw session CSV, validates it, generates both plots and both
processed DataFrames, and saves everything to disk.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from concurrent_schedules_plot.cs_plot import (
    cumulative_records_plot,
    back_to_back_bar_plot,
    prepare_data,
)


def run_pipeline(
    data: str | Path,
    sep: str = ",",
    time_col: str = "time_ms",
    resp_a_col: str = "resp_a",
    resp_b_col: str = "resp_b",
    reinf_a_col: str = "reinf_a",
    reinf_b_col: str = "reinf_b",
    time_unit: str | None = None,
    label_a: str = "Schedule A",
    label_b: str = "Schedule B",
    color_a: str = "#D55E00",
    color_b: str = "#0072B2",
    x_tick_step: int = 10,
    outdir: str | Path = "output",
    prefix: str | None = None,
    fmt: str = "png",
    dpi: int = 300,
    show: bool = False,
) -> dict[str, Path]:
    """
    Run the full concurrent schedules pipeline on a raw session CSV.

    Validates that the required columns exist, prepares the data,
    generates the cumulative record and back-to-back bar plots, and
    saves both figures and both processed DataFrames to `outdir`.

    Parameters
    ----------
    data : str or Path
        Path to the raw session CSV file.
    sep : str, optional
        CSV delimiter, used both for reading `data` and for writing the
        output CSVs. Default is ",".
    time_col : str, optional
        Column name for timestamps in `data`. Default is "time_ms".
    resp_a_col : str, optional
        Column name for cumulative responses on schedule A. Default is "resp_a".
    resp_b_col : str, optional
        Column name for cumulative responses on schedule B. Default is "resp_b".
    reinf_a_col : str, optional
        Column name for cumulative reinforcers on schedule A. Default is "reinf_a".
    reinf_b_col : str, optional
        Column name for cumulative reinforcers on schedule B. Default is "reinf_b".
    time_unit : str, optional
        Time unit of `time_col`: "s" for seconds, "min" for minutes.
        If None, assumes time is already in milliseconds.
    label_a : str, optional
        Label for schedule A in both plot legends.
    label_b : str, optional
        Label for schedule B in both plot legends.
    color_a : str, optional
        Hex color for schedule A in both plots.
    color_b : str, optional
        Hex color for schedule B in both plots.
    x_tick_step : int, optional
        X-axis tick interval for the back-to-back bar plot. Default is 50.
    outdir : str or Path, optional
        Directory where output files are saved. Created if missing.
        Default is "output".
    prefix : str, optional
        Prefix for output filenames. If None, derived from the stem of
        `data` (e.g. "subject113.csv" -> "subject113").
    fmt : str, optional
        Image format for saved figures (e.g. "png", "pdf", "svg"). Default is "png".
    dpi : int, optional
        Resolution for saved figures. Default is 300.
    show : bool, optional
        Whether to display the figures on screen. Default is False.

    Returns
    -------
    dict of str -> Path
        Paths to the four generated files, keyed as "cumulative_record",
        "back_to_back_bar", "filtered_data", and "trials_data".

    Raises
    ------
    ValueError
        If any of the required columns is missing from `data`.
    """
    user_cols = [time_col, resp_a_col, resp_b_col, reinf_a_col, reinf_b_col]

    # Read only the header (nrows=0) to validate columns without loading
    # the full file — session CSVs can span years of data.
    data_cols = pd.read_csv(data, sep=sep, nrows=0).columns

    for col in user_cols:
        if col not in data_cols:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(data_cols)}"
            )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    filtered_df, trials_df = prepare_data(
        data=data,
        sep=sep,
        resp_a_col=resp_a_col,
        resp_b_col=resp_b_col,
        reinf_a_col=reinf_a_col,
        reinf_b_col=reinf_b_col,
        time_col=time_col,
        time_unit=time_unit,
    )

    fig_cumulative, ax_cumulative = cumulative_records_plot(
        filtered_df=filtered_df,
        label_a=label_a,
        label_b=label_b,
        color_a=color_a,
        color_b=color_b,
    )

    fig_bar, ax_bar = back_to_back_bar_plot(
        trials_df=trials_df,
        step=x_tick_step,
        label_a=label_a,
        label_b=label_b,
        color_a=color_a,
        color_b=color_b,
    )

    if prefix is None:
        prefix = Path(data).stem

    path_cumulative = outdir / f"{prefix}_cumulative.{fmt}"
    path_bar = outdir / f"{prefix}_back_to_back_bar.{fmt}"
    path_filtered = outdir / f"{prefix}_filtered.csv"
    path_trials = outdir / f"{prefix}_trials.csv"

    fig_cumulative.savefig(path_cumulative, format=fmt, dpi=dpi)
    fig_bar.savefig(path_bar, format=fmt, dpi=dpi)
    filtered_df.to_csv(path_filtered, sep=sep, index=False)
    trials_df.to_csv(path_trials, sep=sep, index=False)

    if show:
        plt.show()

    # Release figures from memory now that they're saved — matters when
    # run_pipeline is called in a loop over many subjects.
    for fig in [fig_cumulative, fig_bar]:
        plt.close(fig)

    result = {
        "cumulative_record": path_cumulative,
        "back_to_back_bar": path_bar,
        "filtered_data": path_filtered,
        "trials_data": path_trials,
    }

    return result