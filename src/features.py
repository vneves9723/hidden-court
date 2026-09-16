from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.div(denominator.replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_player_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn season totals into per-game, per-36 and efficiency features."""
    data = raw.copy()
    data.columns = [column.lower() for column in data.columns]

    for stat in ["min", "pts", "reb", "ast", "stl", "blk", "tov"]:
        data[f"{stat}_pg" if stat != "min" else "mpg"] = safe_divide(
            data[stat], data["gp"]
        )

    data = data.rename(
        columns={
            "pts_pg": "ppg",
            "reb_pg": "rpg",
            "ast_pg": "apg",
            "stl_pg": "spg",
            "blk_pg": "bpg",
        }
    )

    for stat in ["pts", "reb", "ast", "stl", "blk", "tov"]:
        data[f"{stat}_per36"] = 36 * safe_divide(data[stat], data["min"])

    data["ts_pct"] = safe_divide(data["pts"], 2 * (data["fga"] + 0.44 * data["fta"]))
    data["efg_pct"] = safe_divide(data["fgm"] + 0.5 * data["fg3m"], data["fga"])
    data["three_rate"] = safe_divide(data["fg3a"], data["fga"])
    data["ft_rate"] = safe_divide(data["fta"], data["fga"])
    data["ast_tov"] = safe_divide(data["ast"], data["tov"])

    data["season_start"] = data["season"].str[:4].astype(int)
    numeric = data.select_dtypes(include="number").columns
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    return data


def create_training_pairs(featured: pd.DataFrame) -> pd.DataFrame:
    """Match each player-season to that player's following NBA season."""
    current = featured.copy()
    future_columns = ["player_id", "season_start", "ppg", "mpg"]
    future = featured[future_columns].copy()
    future["season_start"] -= 1
    future = future.rename(columns={"ppg": "next_ppg", "mpg": "next_mpg"})

    pairs = current.merge(future, on=["player_id", "season_start"], how="inner")
    pairs["ppg_growth"] = pairs["next_ppg"] - pairs["ppg"]
    pairs["mpg_growth"] = pairs["next_mpg"] - pairs["mpg"]
    pairs["breakout"] = (
        (pairs["ppg_growth"] >= 4.0) & (pairs["mpg_growth"] >= 3.0)
    ).astype(int)
    return pairs

