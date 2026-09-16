import pandas as pd

from src.features import add_player_features, create_training_pairs, safe_divide


def test_safe_divide_handles_zero():
    result = safe_divide(pd.Series([10, 4]), pd.Series([2, 0]))
    assert result.tolist() == [5.0, 0.0]


def test_feature_formulas():
    raw = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Test Guard"],
            "team": ["TST"],
            "season": ["2024-25"],
            "age": [22],
            "gp": [10],
            "min": [200],
            "fgm": [50],
            "fga": [100],
            "fg3m": [20],
            "fg3a": [50],
            "ftm": [30],
            "fta": [40],
            "oreb": [5],
            "dreb": [25],
            "reb": [30],
            "ast": [40],
            "tov": [20],
            "stl": [10],
            "blk": [2],
            "pf": [15],
            "pts": [150],
            "plus_minus": [5],
        }
    )
    result = add_player_features(raw).iloc[0]
    assert result["ppg"] == 15
    assert result["pts_per36"] == 27
    assert result["ast_tov"] == 2
    assert round(result["efg_pct"], 3) == 0.6


def test_training_pairs_only_match_next_season():
    base = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "season_start": [2023, 2024, 2024],
            "ppg": [5.0, 10.0, 7.0],
            "mpg": [10.0, 20.0, 15.0],
        }
    )
    pairs = create_training_pairs(base)
    assert len(pairs) == 1
    assert pairs.iloc[0]["next_ppg"] == 10.0
    assert pairs.iloc[0]["breakout"] == 1

