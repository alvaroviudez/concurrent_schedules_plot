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

def prepare_data(data, sep, rs_a_col, rs_b_col, ref_a_col, ref_b_col, time_col, time_unit=None):
    """
    Parse raw data into trial-based format for concurrent schedule analysis.
    
    This function expects CUMULATIVE response and reinforcement counts as input and detects 
    reinforcement delivery events to segment the session into trials.
    
    Parameters
    ----------
    data : str
        Path to CSV file containing session data.
    sep : str
        CSV delimiter.
    rs_a_col : str
        Column name for cumulative responses on schedule A.
    rs_b_col : str
        Column name for cumulative responses on schedule B.
    ref_a_col : str
        Column name for cumulative reinforcers on schedule A.
    ref_b_col : str
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
    """
    # Load and filter relevant columns
    df = pd.read_csv(data, sep=sep)

    # Convert time units to miliseconds if necessary
    if time_unit == "s":
        to_ms_factor = 1000
    elif time_unit == "min":
        to_ms_factor = 60000
    else:
        to_ms_factor = 1
    df[time_col] = df[time_col] * to_ms_factor

    # Select relevant columns for filtered_df
    filtered_df = df[[rs_a_col, rs_b_col, ref_a_col, ref_b_col, time_col]].copy()   
    
    # Detect when reinforcers are delivered (change in cumulative count)
    new_ref_a = filtered_df[ref_a_col].diff().fillna(0) > 0
    new_ref_b = filtered_df[ref_b_col].diff().fillna(0) > 0
    
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
    last_a_when_a_reinforced = trial_ends_df[rs_a_col].where(trial_ends_df["new_ref_a"] == 1).ffill().shift(1).fillna(0)
    last_b_when_b_reinforced = trial_ends_df[rs_b_col].where(trial_ends_df["new_ref_b"] == 1).ffill().shift(1).fillna(0)
    
    # Responses in each trial = current cumulative - last time that schedule was reinforced
    trial_ends_df["Responses A"] = (trial_ends_df[rs_a_col] - last_a_when_a_reinforced).astype(int)
    trial_ends_df["Responses B"] = (trial_ends_df[rs_b_col] - last_b_when_b_reinforced).astype(int)
    
    # Check if there are responses after the last reinforcement (final trial without reinforcement)
    last_trial_idx = trial_ends_df.index[-1] if len(trial_ends_df) > 0 else -1
    last_data_idx = len(filtered_df) - 1
    
    if last_trial_idx < last_data_idx:
        # There are data rows after the last reinforcement - create final trial
        final_resp_a = filtered_df.loc[last_data_idx, rs_a_col]
        final_resp_b = filtered_df.loc[last_data_idx, rs_b_col]
        
        # Get last reinforcement values for each schedule
        if len(trial_ends_df) > 0:
            last_ref_a_val = trial_ends_df[rs_a_col].where(trial_ends_df["new_ref_a"] == 1).ffill().iloc[-1]
            last_ref_b_val = trial_ends_df[rs_b_col].where(trial_ends_df["new_ref_b"] == 1).ffill().iloc[-1]
            if pd.isna(last_ref_a_val):
                last_ref_a_val = 0
            if pd.isna(last_ref_b_val):
                last_ref_b_val = 0
        else:
            last_ref_a_val = 0
            last_ref_b_val = 0
        
        # Append final trial
        final_trial = pd.DataFrame({
            rs_a_col: [final_resp_a],
            rs_b_col: [final_resp_b],
            "new_ref_a": [0],
            "new_ref_b": [0],
            "Responses A": [int(final_resp_a - last_ref_a_val)],
            "Responses B": [int(final_resp_b - last_ref_b_val)]
        })
        trial_ends_df = pd.concat([trial_ends_df, final_trial], ignore_index=True)
    
    # Build trials DataFrame
    trials_df = pd.DataFrame({
        "Trial": np.arange(1, len(trial_ends_df) + 1),
        "Responses A": trial_ends_df["Responses A"].values,
        "Responses B": trial_ends_df["Responses B"].values,
        "Reinforcement A": trial_ends_df["new_ref_a"].values,
        "Reinforcement B": trial_ends_df["new_ref_b"].values
    })
    
    # Return original filtered_df after renaming the columns
    filtered_df = df[[time_col, rs_a_col, rs_b_col, ref_a_col, ref_b_col]]
    filtered_df.columns = ["Time", "Responses A", "Responses B", "Reinforcement A", "Reinforcement B"]
    
    return filtered_df, trials_df

def cumulative_records_plot(filtered_df, label_a="Schedule A", label_b="Schedule B", 
                           color_a="#D55E00", color_b="#0072B2"):
    """
    Generate a cumulative record plot for concurrent schedules.
    
    Parameters
    ----------
    filtered_df : pd.DataFrame
        DataFrame containing response and reinforcement data.
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
    """
    df = filtered_df.copy()
    
    # Identify reinforcement delivery points
    refs_a = df[df["Reinforcement A"] != df["Reinforcement A"].shift(1)]
    refs_b = df[df["Reinforcement B"] != df["Reinforcement B"].shift(1)]

    # Create figure and plot cumulative responses
    fig, ax = plt.subplots()
    ax.plot("Time", "Responses A", data=df, color=color_a)
    ax.plot("Time", "Responses B", data=df, color=color_b)
    ax.scatter(refs_a["Time"], refs_a["Responses A"], marker=r"$\backslash$", color=color_a)
    ax.scatter(refs_b["Time"], refs_b["Responses B"], marker=r"$\backslash$", color=color_b)

    # Configure axes appearance
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", linestyle=":")

    # Convert time units to minutes if necessary
    to_min_factor = 60000

    # Set axis labels and limits
    ax.set_xlabel("Time (min)", fontdict={"weight": "bold"})
    ax.set_ylabel("Responses", fontdict={"weight": "bold"})
    ax.set_xlim(0, df["Time"].max())
    ax.set_ylim(0, df[["Responses A", "Responses B"]].max().max())

    # Configure x-axis ticks in minutes
    ax.set_xticks(
        np.arange(0, df["Time"].max(), step=to_min_factor),
        labels=np.arange(0, df["Time"].max() / to_min_factor, step=1).astype(int)
    )

    # Add legend
    legend_handles = _legend_patches(color_a, color_b, label_a, label_b)
    ax.legend(handles=legend_handles)
    
    return fig, ax

def plot_cs(trials_df, step=50, label_a="Schedule A", label_b="Schedule B", 
           color_a="#D55E00", color_b="#0072B2"):
    """
    Generate a concurrent schedules plot with horizontal bars.
    
    Creates a trial-by-trial visualization where schedule A responses extend
    left and schedule B responses extend right. Reinforced trials are shown
    in full color, unreinforced trials in lighter shades.
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame with trial-by-trial data (output from prepare_data).
    step : int, optional
        X-axis tick interval (default: 1).
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
    """
    df = trials_df.copy()

    # Calculate x-axis range based on maximum response count
    rs_max = df[["Responses A", "Responses B"]].max().max()
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
