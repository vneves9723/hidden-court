from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_DATA_PATH = RAW_DIR / "player_seasons.csv"
TRAINING_DATA_PATH = PROCESSED_DIR / "training_pairs.csv"
PROSPECTS_PATH = PROCESSED_DIR / "prospects.csv"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
MODEL_PATH = MODEL_DIR / "hidden_court_models.joblib"

SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2026)]
CURRENT_SEASON = "2025-26"
PROJECTION_SEASON = "2026-27"

FEATURES = [
    "age",
    "gp",
    "mpg",
    "ppg",
    "rpg",
    "apg",
    "spg",
    "bpg",
    "tov_pg",
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "stl_per36",
    "blk_per36",
    "tov_per36",
    "ts_pct",
    "efg_pct",
    "three_rate",
    "ft_rate",
    "ast_tov",
]


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

