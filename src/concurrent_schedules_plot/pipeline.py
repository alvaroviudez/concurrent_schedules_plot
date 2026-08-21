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
    x_tick_step: int = 50,
    outdir: str | Path = "output",
    prefix: str | None = None,
    fmt: str = "png",
    dpi: int = 300,
    show: bool = False,
) -> dict[str, Path]:
    
    user_cols = [time_col, resp_a_col, resp_b_col, reinf_a_col, reinf_b_col]
    data_cols = pd.read_csv(data, sep=sep, nrows=0).columns

    for col in user_cols:
        if col not in data_cols:
            raise ValueError(
                f"Columna '{col}' no encontrada. Columnas disponibles: {list(data_cols)}"
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
        time_unit=time_unit
    )

    fig_cumulative, ax_cumulative = cumulative_records_plot(
        filtered_df=filtered_df,
        label_a=label_a,
        label_b=label_b,
        color_a=color_a,
        color_b=color_b
    )

    fig_bar, ax_bar = back_to_back_bar_plot(
            trials_df=trials_df,
            step=x_tick_step,
            label_a=label_a,
            label_b=label_b,
            color_a=color_a,
            color_b=color_b
        )

    if prefix is None:
        prefix = Path(data).stem

    path_cumulative = outdir / f"{prefix}_cumulative.{fmt}"
    path_bar = outdir / f"{prefix}_bar.{fmt}"
    path_filtered = outdir / f"{prefix}_filtered.csv"
    path_trials = outdir / f"{prefix}_trials.csv"
    
    fig_cumulative.savefig(path_cumulative, format=fmt, dpi=dpi)
    fig_bar.savefig(path_bar, format=fmt, dpi=dpi)
    filtered_df.to_csv(path_filtered, sep=sep, index=False)
    trials_df.to_csv(path_trials, sep=sep, index=False)

    if show:
        plt.show()
    for fig in [fig_cumulative, fig_bar]:        
        plt.close(fig)

    result = {
    "cumulative_record": path_cumulative,
    "concurrent_schedules": path_bar,
    "filtered_data": path_filtered,
    "trials_data": path_trials
    }

    return result