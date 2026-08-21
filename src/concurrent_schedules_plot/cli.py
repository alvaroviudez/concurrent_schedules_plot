"""Command-line interface for the concurrent schedules pipeline.

Parses terminal arguments and delegates to `run_pipeline`.
"""

import argparse

from concurrent_schedules_plot.pipeline import run_pipeline


def main():
    """Entry point for the `cs-plot` command."""
    parser = argparse.ArgumentParser(
        prog="cs-plot",
        description=(
            "Generate cumulative record and back-to-back bar plots "
            "from a concurrent schedules session CSV."
        ),
    )

    # Required positional argument: the input file itself.
    parser.add_argument("data", help="Path to the raw session CSV file")

    # Column names — default to the canonical schema, so a user whose
    # CSV already matches it can omit all of these.
    parser.add_argument("--sep", default=",", help="CSV delimiter (default: ',')")
    parser.add_argument("--time-col", default="time_ms", help="Timestamp column name")
    parser.add_argument("--resp-a-col", default="resp_a", help="Schedule A responses column")
    parser.add_argument("--resp-b-col", default="resp_b", help="Schedule B responses column")
    parser.add_argument("--reinf-a-col", default="reinf_a", help="Schedule A reinforcers column")
    parser.add_argument("--reinf-b-col", default="reinf_b", help="Schedule B reinforcers column")

    # Time unit — required, no default, so the user is forced to check
    # their own data instead of silently assuming milliseconds.
    parser.add_argument(
        "--time-unit",
        choices=["ms", "s", "min"],
        required=True,
        help="Time unit of --time-col",
    )

    # Plot appearance — optional, defaults live in cs_plot.py itself.
    parser.add_argument("--label-a", default="Schedule A", help="Legend label for schedule A")
    parser.add_argument("--label-b", default="Schedule B", help="Legend label for schedule B")
    parser.add_argument("--color-a", default="#D55E00", help="Hex color for schedule A")
    parser.add_argument("--color-b", default="#0072B2", help="Hex color for schedule B")
    parser.add_argument("--x-tick-step", type=int, default=10, help="X-axis tick interval for the bar plot")

    # Output control.
    parser.add_argument("--outdir", default="output", help="Directory for output files")
    parser.add_argument("--prefix", default=None, help="Prefix for output filenames")
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png", help="Image format"
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for saved figures")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the figures on screen (default: True)",
    )

    args = parser.parse_args()

    result = run_pipeline(
        data=args.data,
        sep=args.sep,
        time_col=args.time_col,
        resp_a_col=args.resp_a_col,
        resp_b_col=args.resp_b_col,
        reinf_a_col=args.reinf_a_col,
        reinf_b_col=args.reinf_b_col,
        time_unit=args.time_unit,
        label_a=args.label_a,
        label_b=args.label_b,
        color_a=args.color_a,
        color_b=args.color_b,
        x_tick_step=args.x_tick_step,
        outdir=args.outdir,
        prefix=args.prefix,
        fmt=args.format,
        dpi=args.dpi,
        show=args.show,
    )

    for key, path in result.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()