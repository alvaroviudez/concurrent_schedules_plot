import sys
from pathlib import Path

import pytest

from concurrent_schedules_plot.cli import main

DATA_DIR = Path(__file__).parent / "data"


def test_cli_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cs-plot",
            str(DATA_DIR / "basic.csv"),
            "--sep", ";",
            "--time-unit", "ms",
            "--outdir", str(tmp_path),
            "--no-show",
        ],
    )

    main()

    assert (tmp_path / "basic_cumulative.png").exists()


def test_cli_requires_time_unit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cs-plot", str(DATA_DIR / "basic.csv"), "--sep", ";"],
    )

    with pytest.raises(SystemExit):
        main()
