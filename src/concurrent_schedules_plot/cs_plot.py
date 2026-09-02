import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch


def _legend_patches(color_a, color_b, label_a, label_b):
    """
    Build legend patches for schedules A and B.

    Parameters
    ----------
    color_a : str
        Hex color for schedule A.
    color_b : str
        Hex color for schedule B.
    label_a : str
        Label for schedule A in legend.
    label_b : str
        Label for schedule B in legend.

    Returns
    -------
    list
        List of two matplotlib Patch objects for use as legend handles.
    """
    legend_handles = [
            Patch(facecolor=color_a, label=f"{label_a}"),
            Patch(facecolor=color_b, label=f"{label_b}")
        ]
    return legend_handles

def _validate_session(df, resp_a_col, resp_b_col, reinf_a_col, reinf_b_col, time_col):
    """
    Validate that a session DataFrame matches the expected data contract before
    any further processing: non-empty, numeric columns, non-decreasing integer
    counts, at most one reinforcer delivered per row, and strictly increasing
    time.

    Response counts may jump by more than one unit between consecutive rows
    (e.g. a logger sampling at fixed intervals rather than one row per
    event). Reinforcer counts may not: `prepare_data` treats each row where a
    reinforcer count changes as a single trial boundary, so a jump of more
    than one reinforcer in one row would silently collapse multiple
    reinforcement events — and the responses between them — into one trial.

    Parameters
    ----------
    df : pd.DataFrame
        Raw session data, as loaded from CSV.
    resp_a_col : str
        Column name for cumulative responses on schedule A.
    resp_b_col : str
        Column name for cumulative responses on schedule B.
    reinf_a_col : str
        Column name for cumulative reinforcers on schedule A.
    reinf_b_col : str
        Column name for cumulative reinforcers on schedule B.
    time_col : str
        Column name for timestamps.

    Raises
    ------
    ValueError
        If the data is empty, a required column is not numeric, a cumulative
        count decreases or changes by a non-integer amount between rows, a
        reinforcer count changes by more than one in a single row, or time
        does not strictly increase between rows.
    """
    if len(df) == 0:
        raise ValueError("Data file is empty.")

    numeric_cols = [resp_a_col, resp_b_col, reinf_a_col, reinf_b_col, time_col]
    for numeric_col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[numeric_col]):
            raise ValueError(
                f"Column '{numeric_col}' must be numeric, got dtype '{df[numeric_col].dtype}'."
            )
        if df[numeric_col].isna().any():
            bad_row = df[numeric_col].isna().idxmax()
            raise ValueError(
                f"Column '{numeric_col}' contains a missing value at row {bad_row}. "
                f"All required columns must be fully populated."
            )

    count_cols = [resp_a_col, resp_b_col, reinf_a_col, reinf_b_col]
    for count_col in count_cols:
        diffs = df[count_col].diff()
        invalid = (diffs < 0) | (diffs != diffs.round())
        invalid.iloc[0] = False  # first row has no predecessor to violate
        if invalid.any():
            bad_row = invalid.idxmax()
            raise ValueError(
                f"Column '{count_col}' must be a non-decreasing integer count. "
                f"At row {bad_row}, value {df[count_col].iloc[bad_row]} follows "
                f"{df[count_col].iloc[bad_row - 1]} (step of {diffs.iloc[bad_row]})."
            )

    # Reinforcer columns additionally cap the step at 1: prepare_data marks a
    # trial boundary wherever the reinforcer count changes, so a jump of more
    # than one would mean several reinforcement events — and the responses
    # interleaved between them — get folded into a single trial with no way
    # to recover their true order.
    reinf_cols = [reinf_a_col, reinf_b_col]
    for reinf_col in reinf_cols:
        if df[reinf_col].iloc[0] > 1:
            raise ValueError(
                f"Column '{reinf_col}' must start at 0 or 1 in the first row, "
                f"got {df[reinf_col].iloc[0]}. A value greater than 1 in the "
                f"first row means at least one reinforcer was delivered before "
                f"the logged session began, which this tool cannot account for."
            )
        diffs = df[reinf_col].diff()
        invalid = diffs > 1
        invalid.iloc[0] = False
        if invalid.any():
            bad_row = invalid.idxmax()
            raise ValueError(
                f"Column '{reinf_col}' must not deliver more than one reinforcer "
                f"per row. At row {bad_row}, value {df[reinf_col].iloc[bad_row]} "
                f"follows {df[reinf_col].iloc[bad_row - 1]} (step of "
                f"{diffs.iloc[bad_row]}). If your logger can record multiple "
                f"reinforcers in a single sample, split that row so each "
                f"reinforcer has its own row and timestamp."
            )

    time_diffs = df[time_col].diff()
    invalid_time = time_diffs <= 0
    invalid_time.iloc[0] = False
    if invalid_time.any():
        bad_row = invalid_time.idxmax()
        raise ValueError(
            f"Column '{time_col}' must be strictly increasing. "
            f"At row {bad_row}, value {df[time_col].iloc[bad_row]} does not exceed "
            f"{df[time_col].iloc[bad_row - 1]}."
        )

def lighten_color(color_hex, factor=0.5):
    """
    Lighten a hex color by interpolating towards white.

    Parameters
    ----------
    color_hex : str
        Hex color code (e.g., '#BA84F0').
    factor : float, optional
        Lightening factor in [0, 1]. Higher values produce lighter colors.
        Default is 0.5.

    Returns
    -------
    tuple
        RGBA tuple with lightened color.
    """
    r, g, b, a = to_rgba(color_hex)
    r_light = r + (1 - r) * factor
    g_light = g + (1 - g) * factor
    b_light = b + (1 - b) * factor
    return (r_light, g_light, b_light, a)

def prepare_data(
    data, sep, resp_a_col, resp_b_col, reinf_a_col, reinf_b_col, time_col, time_unit=None
):
    """
    Parse raw data into trial-based format for concurrent schedule analysis.

    This function expects CUMULATIVE response and reinforcement counts as input and detects
    reinforcement delivery events to segment the session into trials.

    Input data must satisfy the following contract, checked before any processing:
    the file is non-empty; the response, reinforcer, and time columns are numeric;
    response counts are non-decreasing integers that may jump by more than one
    between rows (to allow for loggers that record more than one event per row);
    reinforcer counts are non-decreasing integers that may jump by at most one
    per row and must start at 0 or 1 in the first row, since each reinforcer
    marks a trial boundary; and the time column is strictly increasing.

    Parameters
    ----------
    data : str
        Path to CSV file containing session data.
    sep : str
        CSV delimiter.
    resp_a_col : str
        Column name for cumulative responses on schedule A.
    resp_b_col : str
        Column name for cumulative responses on schedule B.
    reinf_a_col : str
        Column name for cumulative reinforcers on schedule A.
    reinf_b_col : str
        Column name for cumulative reinforcers on schedule B.
    time_col : str
        Column name for timestamps.
    time_unit : str, optional
        Time unit (used for conversion to miliseconds): 's' for seconds, 'min' for minutes.
        If None, assumes time is already in miliseconds.

    Returns
    -------
    tuple
        (filtered_df, trials_df) where filtered_df contains the raw data columns
        and trials_df contains trial-by-trial response and reinforcement counts.

    Raises
    ------
    ValueError
        If the input data does not satisfy the contract described above, or if
        `time_unit` is not one of 's', 'min', 'ms', or None.
    """
    # Load and filter relevant columns, test input data
    df = pd.read_csv(data, sep=sep)
    _validate_session(df, resp_a_col, resp_b_col, reinf_a_col, reinf_b_col, time_col)

    # Convert time units to miliseconds if necessary
    time_unit = time_unit.lower() if time_unit else time_unit

    if time_unit in ("ms", None):
        pass
    elif time_unit == "s":
        df[time_col] = df[time_col] * 1000
    elif time_unit == "min":
        df[time_col] = df[time_col] * 60000
    else:
        raise ValueError(
            f"time_unit must be 's', 'min', 'ms', or unspecified (defaults to ms), "
            f"not '{time_unit}'"
        )

    # Select relevant columns for filtered_df
    filtered_df = df[[resp_a_col, resp_b_col, reinf_a_col, reinf_b_col, time_col]].copy()

    # Detect when reinforcers are delivered (change in cumulative count)
    new_ref_a = filtered_df[reinf_a_col].diff().fillna(0) > 0
    new_ref_b = filtered_df[reinf_b_col].diff().fillna(0) > 0

    # Mark trial boundaries: any row where a reinforcer is delivered marks the END of a trial
    trial_end = new_ref_a | new_ref_b

    # Get only the rows where trials end (where reinforcements occur)
    trial_ends_df = filtered_df[trial_end].copy()

    # Add reinforcement flags
    trial_ends_df["new_ref_a"] = new_ref_a[trial_end].astype(int)
    trial_ends_df["new_ref_b"] = new_ref_b[trial_end].astype(int)

    # For each schedule: responses = difference between consecutive trial endings
    # But we need to handle them separately because they accumulate independently

    # Create groups for each schedule based on which schedule was reinforced
    # When A is reinforced: B continues accumulating, A resets
    # When B is reinforced: A continues accumulating, B resets

    # Calculate cumulative responses at the last reinforcement of each schedule
    last_a_when_a_reinforced = (
        trial_ends_df[resp_a_col].where(trial_ends_df["new_ref_a"] == 1).ffill().shift(1).fillna(0)
    )
    last_b_when_b_reinforced = (
        trial_ends_df[resp_b_col].where(trial_ends_df["new_ref_b"] == 1).ffill().shift(1).fillna(0)
    )

    # Responses in each trial = current cumulative - last time that schedule was reinforced
    trial_ends_df["Responses A"] = (
        trial_ends_df[resp_a_col] - last_a_when_a_reinforced
    ).astype(int)
    trial_ends_df["Responses B"] = (
        trial_ends_df[resp_b_col] - last_b_when_b_reinforced
    ).astype(int)

    # Check if there are responses after the last reinforcement (final trial without
    # reinforcement). Compare row positions, not index labels — filtered_df's index
    # is not guaranteed to be a clean RangeIndex, so labels and positions can diverge.
    last_trial_pos = (
        filtered_df.index.get_indexer_for([trial_ends_df.index[-1]])[0]
        if len(trial_ends_df) > 0
        else -1
    )
    last_data_pos = len(filtered_df) - 1

    if last_trial_pos < last_data_pos:
        # There are data rows after the last reinforcement - create final trial
        final_resp_a = filtered_df.iloc[last_data_pos][resp_a_col]
        final_resp_b = filtered_df.iloc[last_data_pos][resp_b_col]

        # Get last reinforcement values for each schedule
        if len(trial_ends_df) > 0:
            last_ref_a_val = (
                trial_ends_df[resp_a_col].where(trial_ends_df["new_ref_a"] == 1).ffill().iloc[-1]
            )
            last_ref_b_val = (
                trial_ends_df[resp_b_col].where(trial_ends_df["new_ref_b"] == 1).ffill().iloc[-1]
            )
            if pd.isna(last_ref_a_val):
                last_ref_a_val = 0
            if pd.isna(last_ref_b_val):
                last_ref_b_val = 0
        else:
            last_ref_a_val = 0
            last_ref_b_val = 0

        # Append final trial
        final_trial = pd.DataFrame({
            resp_a_col: [final_resp_a],
            resp_b_col: [final_resp_b],
            "new_ref_a": [0],
            "new_ref_b": [0],
            "Responses A": [int(final_resp_a - last_ref_a_val)],
            "Responses B": [int(final_resp_b - last_ref_b_val)]
        })
        trial_ends_df = pd.concat([trial_ends_df, final_trial], ignore_index=True)

    # Build trials DataFrame
    trials_df = pd.DataFrame({
        "Trial": np.arange(1, len(trial_ends_df) + 1),
        "Responses A": trial_ends_df["Responses A"].to_numpy(),
        "Responses B": trial_ends_df["Responses B"].to_numpy(),
        "Reinforcement A": trial_ends_df["new_ref_a"].to_numpy(),
        "Reinforcement B": trial_ends_df["new_ref_b"].to_numpy()
    })

    # Return original filtered_df after renaming the columns
    filtered_df = df[[time_col, resp_a_col, resp_b_col, reinf_a_col, reinf_b_col]]
    filtered_df.columns = [
        "Time", "Responses A", "Responses B",
        "Cumulative Reinforcement A", "Cumulative Reinforcement B"
    ]

    return filtered_df, trials_df

def cumulative_records_plot(filtered_df, label_a="Schedule A", label_b="Schedule B",
                           color_a="#D55E00", color_b="#0072B2", time_unit_display="min"):
    """
    Generate a cumulative record plot for concurrent schedules.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        DataFrame containing response and reinforcement data. The "Time"
        column is assumed to be in milliseconds, matching the output of
        `prepare_data`.
    label_a : str, optional
        Label for schedule A in legend.
    label_b : str, optional
        Label for schedule B in legend.
    color_a : str, optional
        Hex color for schedule A (default: orange).
    color_b : str, optional
        Hex color for schedule B (default: blue).
    time_unit_display : str, optional
        Unit to display on the x-axis: "ms", "s", or "min". Only affects
        the axis labels and tick values; the input data is always
        expected in milliseconds. Default is "min".

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.

    Raises
    ------
    ValueError
        If `time_unit_display` is not one of "ms", "s", "min".
    """
    df = filtered_df.copy()

    # Identify reinforcement delivery points
    refs_a = df[df["Cumulative Reinforcement A"].diff() > 0]
    refs_b = df[df["Cumulative Reinforcement B"].diff() > 0]

    # Create figure and plot cumulative responses
    fig, ax = plt.subplots()
    ax.plot("Time", "Responses A", data=df, color=color_a)
    ax.plot("Time", "Responses B", data=df, color=color_b)
    ax.scatter(refs_a["Time"], refs_a["Responses A"], marker=r"$\backslash$", color=color_a)
    ax.scatter(refs_b["Time"], refs_b["Responses B"], marker=r"$\backslash$", color=color_b)

    # Configure axes appearance
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", linestyle=":")

    # Set axis labels and limits
    time_unit_factors = {"ms": 1, "s": 1000, "min": 60000}
    if time_unit_display not in time_unit_factors:
        raise ValueError(
            f"time_unit_display must be 'ms', 's', or 'min', not '{time_unit_display}'"
        )
    unit_factor = time_unit_factors[time_unit_display]

    ax.set_xlabel(f"Time ({time_unit_display})", fontdict={"weight": "bold"})
    ax.set_ylabel("Responses", fontdict={"weight": "bold"})
    ax.set_ylim(0, df[["Responses A", "Responses B"]].max().max())

    # Configure x-axis ticks in the requested display unit.
    max_tick = int(np.floor(df["Time"].max() / unit_factor)) * unit_factor
    xticks = np.arange(0, max_tick + unit_factor, step=unit_factor)
    xticks = xticks[xticks <= df["Time"].max()]
    xtick_labels = (xticks / unit_factor).astype(int)

    ax.set_xlim(0, df["Time"].max())
    ax.set_xticks(xticks, labels=xtick_labels)

    # Add legend
    legend_handles = _legend_patches(color_a, color_b, label_a, label_b)
    ax.legend(handles=legend_handles)

    return fig, ax

def back_to_back_bar_plot(trials_df, step=10, label_a="Schedule A", label_b="Schedule B",
           color_a="#D55E00", color_b="#0072B2"):
    """
    Generate a concurrent schedules plot with a back_to_back bar plot.

    Creates a trial-by-trial visualization where schedule A responses extend
    left and schedule B responses extend right. Reinforced trials are shown
    in full color, unreinforced trials in lighter shades.

    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame with trial-by-trial data (output from prepare_data).
    step : int, optional
        X-axis tick interval. Must be a positive number (default: 10).
    label_a : str, optional
        Label for schedule A in legend.
    label_b : str, optional
        Label for schedule B in legend.
    color_a : str, optional
        Hex color for schedule A (default: orange).
    color_b : str, optional
        Hex color for schedule B (default: blue).

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.

    Raises
    ------
    ValueError
        If `step` is not a positive number, or if no trial in `trials_df`
        has any recorded response on either schedule, since that leaves
        the x-axis with no range to display.
    """
    df = trials_df.copy()

    if step <= 0:
        raise ValueError(f"step must be a positive number, got {step}.")

    # Calculate x-axis range based on maximum response count
    rs_max = df[["Responses A", "Responses B"]].max().max()
    if rs_max == 0:
        raise ValueError(
            "trials_df has no recorded responses on either schedule "
            "(Responses A and Responses B are all zero) — there is nothing "
            "to plot on the x-axis."
        )
    scale_x_axis = np.ceil(rs_max / step)
    x_axis_range = np.arange(
        int(-scale_x_axis * step),
        int(scale_x_axis * step + 1),
        step=step
    )

    # Flip schedule A responses to extend leftward
    df["Responses A"] = df["Responses A"] * -1

    # Map reinforcement status to colors (lighter for unreinforced trials)
    color_map_a = {0: lighten_color(color_a), 1: color_a}
    color_map_b = {0: lighten_color(color_b), 1: color_b}
    colors_a = [color_map_a[ref_a] for ref_a in df["Reinforcement A"]]
    colors_b = [color_map_b[ref_b] for ref_b in df["Reinforcement B"]]

    # Create horizontal bar plot
    fig, ax = plt.subplots()
    ax.barh(y=df["Trial"], width=df["Responses A"], color=colors_a)
    ax.barh(y=df["Trial"], width=df["Responses B"], color=colors_b)

    # Configure axes
    ax.set_ylim(len(df) + 1, 0.1)
    ax.set_xlim(int(-scale_x_axis * step), int(scale_x_axis * step))
    ax.set_xticks(x_axis_range, labels=abs(x_axis_range))
    ax.set_yticklabels("")
    ax.tick_params(axis='y', direction='inout', width=1, length=10)

    # Center y-axis and hide unnecessary spines
    ax.spines["left"].set_position(('data', 0))
    ax.spines[["right", "top"]].set_visible(False)

    # Labels and styling
    ax.set_xlabel("Responses", fontdict={"weight": "bold"})
    ax.grid(axis="x", linestyle=":")

    # Add legend
    legend_handles = _legend_patches(color_a, color_b, label_a, label_b)
    ax.legend(handles=legend_handles)

    return fig, ax
