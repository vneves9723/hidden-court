from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from src.config import RAW_DATA_PATH, SEASONS, ensure_directories


COLUMNS = {
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ABBREVIATION": "team",
    "AGE": "age",
    "GP": "gp",
    "MIN": "min",
    "FGM": "fgm",
    "FGA": "fga",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FTM": "ftm",
    "FTA": "fta",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "TOV": "tov",
    "STL": "stl",
    "BLK": "blk",
    "PF": "pf",
    "PTS": "pts",
    "PLUS_MINUS": "plus_minus",
}


def fetch_season(season: str, retries: int = 3, timeout: int = 30) -> pd.DataFrame:
    """Download one regular season of NBA player totals."""
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="Totals",
                timeout=timeout,
            )
            frame = response.get_data_frames()[0]
            frame = frame[list(COLUMNS)].rename(columns=COLUMNS)
            frame["season"] = season
            return frame
        except Exception as exc:  # pragma: no cover - depends on the NBA endpoint
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download season {season}") from error


def download_all_seasons(
    seasons: list[str] | None = None,
    output_path: Path = RAW_DATA_PATH,
    pause_seconds: float = 0.7,
) -> pd.DataFrame:
    """Download, combine, validate and save all configured seasons."""
    ensure_directories()
    seasons = seasons or SEASONS
    frames: list[pd.DataFrame] = []

    for index, season in enumerate(seasons, start=1):
        print(f"[{index}/{len(seasons)}] Downloading {season}...")
        frames.append(fetch_season(season))
        time.sleep(pause_seconds)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["player_id", "season"], keep="first")
    data = data.sort_values(["season", "player_name"]).reset_index(drop=True)

    if data.empty or data["season"].nunique() != len(seasons):
        raise ValueError("The downloaded dataset is incomplete.")
    if data[["player_id", "season"]].duplicated().any():
        raise ValueError("Duplicate player-season rows remain in the dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data):,} player-seasons to {output_path}")
    return data

