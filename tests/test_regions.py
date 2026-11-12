import pandas as pd
import pytest

from src.temperature.regions import hotspot_region_shares


def test_shares_on_small_table():
    df = pd.DataFrame({"T_Max1": [36.5, 36.0, 35.0, None], "canthiMax1": [36.5, 35.2, 34.0, 36.0],
                       "T_FH_Max1": [35.0, 36.0, 34.9, 35.0], "T_OR_Max1": [34.0, 35.0, 33.0, 34.0]})
    r = hotspot_region_shares(df)
    assert r["n"] == 3
    assert r["matches_any"]["inner canthi"] == pytest.approx(1 / 3)
    assert r["matches_any"]["forehead"] == pytest.approx(1 / 3)
    assert r["matches_any"]["mouth"] == 0.0
    assert r["none_of_the_three"] == pytest.approx(1 / 3)


def test_empty_input_is_rejected():
    df = pd.DataFrame({"T_Max1": [None], "canthiMax1": [None], "T_FH_Max1": [None], "T_OR_Max1": [None]})
    with pytest.raises(ValueError):
        hotspot_region_shares(df)
