import pytest


@pytest.fixture
def canonical_schema():
    """Column-name and delimiter kwargs matching the canonical CSV schema
    (time_ms, resp_a, resp_b, reinf_a, reinf_b), shared across test files
    to avoid repeating the same block of arguments in every call to
    prepare_data."""
    return {
        "sep": ";",
        "resp_a_col": "resp_a",
        "resp_b_col": "resp_b",
        "reinf_a_col": "reinf_a",
        "reinf_b_col": "reinf_b",
        "time_col": "time_ms",
    }
