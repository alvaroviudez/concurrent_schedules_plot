from pathlib import Path

import pytest

from concurrent_schedules_plot.pipeline import run_pipeline

DATA_DIR = Path(__file__).parent / "data"


def test_run_pipeline_creates_all_files(tmp_path):
    result = run_pipeline(
        data=DATA_DIR / "basic.csv",
        sep=";",
        outdir=tmp_path,
        prefix="test",
    )

    for path in result.values():
        assert path.exists()


def test_run_pipeline_returns_expected_keys(tmp_path):
    result = run_pipeline(
        data=DATA_DIR / "basic.csv",
        sep=";",
        outdir=tmp_path,
        prefix="test",
    )

    expected_keys = {"cumulative_record", "back_to_back_bar", "filtered_data", "trials_data"}
    assert set(result.keys()) == expected_keys


def test_run_pipeline_raises_on_missing_column(tmp_path):
    with pytest.raises(ValueError):
        run_pipeline(
            data=DATA_DIR / "basic.csv",
            sep=";",
            resp_a_col="columna_que_no_existe",
            outdir=tmp_path,
        )


def test_run_pipeline_default_prefix_from_filename(tmp_path):
    result = run_pipeline(
        data=DATA_DIR / "basic.csv",
        sep=";",
        outdir=tmp_path,
    )

    assert result["cumulative_record"].name == "basic_cumulative.png"
