import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch


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

def prepare_data(data, sep, rs_a_col, rs_b_col, ref_a_col, ref_b_col, time_col):
    """
    Parse raw data into trial-based format for concurrent schedule analysis.
    
    This function expects CUMULATIVE response and reinforcement counts as input.
    It detects reinforcement delivery events and segments the session into trials,
    where each trial ends when a reinforcer is delivered on either schedule.
    
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
    
    Returns
    -------
    tuple
        (filtered_df, trials_df) where filtered_df contains the raw data columns
        and trials_df contains trial-by-trial response and reinforcement counts.
    """
    # Load and filter relevant columns
    df = pd.read_csv(data, sep=sep)
    filtered_df = df[[rs_a_col, rs_b_col, ref_a_col, ref_b_col, time_col]]
    last_row = len(df)

    # Initialize trial tracking lists
    trials = [0]
    rs_a = [0]
    rs_b = [0]
    ref_a = [0]
    ref_b = [0]

    for row in filtered_df.itertuples(index=True):
        cur_trial = trials[-1]
        index = row.Index
        cur_rs_a = row[1]
        cur_rs_b = row[2]
        cur_ref_a = row[3]
        cur_ref_b = row[4]

        # First row: initialize response and reinforcement counts
        if index == 0:
            rs_a[cur_trial] = cur_rs_a
            rs_b[cur_trial] = cur_rs_b
            
            # Check for reinforcer on schedule A
            if cur_ref_a > 0:
                ref_a[cur_trial] += cur_ref_a
                
                if index + 1 < last_row:
                    trials.append(cur_trial + 1)
                    ref_a.append(0)
                    rs_a.append(0)
                    ref_b.append(0)
                    rs_b.append(rs_b[-1])

            # Check for reinforcer on schedule B
            if cur_ref_b > 0:
                ref_b[cur_trial] += cur_ref_b
                
                if index + 1 < last_row:
                    trials.append(cur_trial + 1)
                    ref_b.append(0)
                    rs_b.append(0)
                    ref_a.append(0)
                    rs_a.append(rs_a[-1])

        # Subsequent rows: detect changes in cumulative counts
        if index > 0:
            # Detect new response on schedule A
            if filtered_df.iloc[index][rs_a_col] > filtered_df.iloc[index - 1][rs_a_col]:
                rs_a[cur_trial] += 1
            
            # Detect new response on schedule B
            if filtered_df.iloc[index][rs_b_col] > filtered_df.iloc[index - 1][rs_b_col]:
                rs_b[cur_trial] += 1

            # Detect new reinforcer on schedule A
            if filtered_df.iloc[index][ref_a_col] > filtered_df.iloc[index - 1][ref_a_col]:
                ref_a[cur_trial] += 1
                
                if index + 1 < last_row:
                    trials.append(cur_trial + 1)
                    ref_a.append(0)
                    rs_a.append(0)
                    ref_b.append(0)
                    rs_b.append(rs_b[-1])

            # Detect new reinforcer on schedule B
            if filtered_df.iloc[index][ref_b_col] > filtered_df.iloc[index - 1][ref_b_col]:
                ref_b[cur_trial] += 1
                
                if index + 1 < last_row:
                    trials.append(cur_trial + 1)
                    ref_b.append(0)
                    rs_b.append(0)
                    ref_a.append(0)
                    rs_a.append(rs_a[-1])

    # Convert to 1-indexed trials
    trials = [trial + 1 for trial in trials]

    # Create output DataFrame
    trials_df = pd.DataFrame({
        "Trial": trials,
        "Responses A": rs_a,
        "Responses B": rs_b,
        "Reinforcement A": ref_a,
        "Reinforcement B": ref_b
    })

    return filtered_df, trials_df
    
def cumulative_records_plot(filtered_df, rs_a_col, rs_b_col, ref_a_col, 
                           ref_b_col, time_col, time_to_min=None, 
                           label_a="", label_b="", 
                           color_a="#D55E00", color_b="#0072B2"):
    """
    Generate a cumulative record plot for concurrent schedules.
    
    Parameters
    ----------
    filtered_df : pd.DataFrame
        DataFrame containing response and reinforcement data.
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
    time_to_min : str, optional
        Time unit conversion: 's' for seconds, 'ms' for milliseconds.
        If None, assumes time is already in minutes.
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
    refs_a = df[df[ref_a_col] != df[ref_a_col].shift(1)]
    refs_b = df[df[ref_b_col] != df[ref_b_col].shift(1)]

    # Create figure and plot cumulative responses
    fig, ax = plt.subplots()
    ax.plot(time_col, rs_a_col, data=df, color=color_a)
    ax.plot(time_col, rs_b_col, data=df, color=color_b)
    ax.scatter(refs_a[time_col], refs_a[rs_a_col], marker=r"$\backslash$", color=color_a)
    ax.scatter(refs_b[time_col], refs_b[rs_b_col], marker=r"$\backslash$", color=color_b)

    # Configure axes appearance
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", linestyle=":")

    # Convert time units to minutes if necessary
    if time_to_min == "s":
        to_min_factor = 60
    elif time_to_min == "ms":
        to_min_factor = 60000
    else:
        to_min_factor = 1

    # Set axis labels and limits
    ax.set_xlabel("Time (min)", fontdict={"weight": "bold"})
    ax.set_ylabel("Responses", fontdict={"weight": "bold"})
    ax.set_xlim(0, df[time_col].max())
    ax.set_ylim(0, df[[rs_a_col, rs_b_col]].max().max())

    # Configure x-axis ticks in minutes
    ax.set_xticks(
        np.arange(0, df[time_col].max(), step=to_min_factor),
        labels=np.arange(0, df[time_col].max() / to_min_factor, step=1).astype(int)
    )

    # Add legend
    legend_handles = [
        Patch(facecolor=color_a, label=f"{label_a}"),
        Patch(facecolor=color_b, label=f"{label_b}")
    ]
    ax.legend(handles=legend_handles)
    
    plt.show()

    return fig, ax

def plot_cs(trials_df, rs_a_col, rs_b_col, ref_a_col, ref_b_col, 
           step=1, label_a="", label_b="", 
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
    rs_a_col : str
        Column name for responses on schedule A.
    rs_b_col : str
        Column name for responses on schedule B.
    ref_a_col : str
        Column name for reinforcement on schedule A (0 or 1).
    ref_b_col : str
        Column name for reinforcement on schedule B (0 or 1).
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
    rs_max = df[[rs_a_col, rs_b_col]].max().max()
    scale_x_axis = np.ceil(rs_max / step)
    x_axis_range = np.arange(
        int(-scale_x_axis * step), 
        int(scale_x_axis * step + 1), 
        step=step
    )

    # Flip schedule A responses to extend leftward
    df[rs_a_col] = df[rs_a_col] * -1

    # Map reinforcement status to colors (lighter for unreinforced trials)
    color_map_a = {0: lighten_color(color_a), 1: color_a}
    color_map_b = {0: lighten_color(color_b), 1: color_b}
    colors_a = [color_map_a[ref_a] for ref_a in df[ref_a_col]]
    colors_b = [color_map_b[ref_b] for ref_b in df[ref_b_col]]

    # Create horizontal bar plot
    fig, ax = plt.subplots()
    bar_a = ax.barh(y=df["Trial"], width=df[rs_a_col], color=colors_a)
    bar_b = ax.barh(y=df["Trial"], width=df[rs_b_col], color=colors_b)
    
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
    legend_handles = [
        Patch(facecolor=color_a, label=f"{label_a}"),
        Patch(facecolor=color_b, label=f"{label_b}")
    ]
    ax.legend(handles=legend_handles)
    
    plt.show()

    return fig, ax



filtered_df, trials_df = prepare_data(
    data="./Data/subject-1-13.csv",
    sep=";",
    rs_a_col = "respFi",
    rs_b_col = "respCh",
    ref_a_col = "reinfFi",
    ref_b_col = "reinfCh",
    time_col = "current_time"
    )

cumulative_records_plot(
    filtered_df = filtered_df,
    rs_a_col = "respFi",
    rs_b_col = "respCh",
    ref_a_col = "reinfFi",
    ref_b_col = "reinfCh",
    time_col = "current_time",
    time_to_min= "ms",
    label_a="Schedule A",
    label_b="Schedule B"
    )

plot_cs(
    trials_df = trials_df,
    rs_a_col = "Responses A",
    rs_b_col = "Responses B",
    ref_a_col = "Reinforcement A",
    ref_b_col = "Reinforcement B",
    step=50,
    label_a="Schedule A",
    label_b="Schedule B"
    )
