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


def test_cli_exits_cleanly_on_missing_file(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cs-plot", "does_not_exist.csv", "--time-unit", "ms"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "does_not_exist.csv" in capsys.readouterr().err
