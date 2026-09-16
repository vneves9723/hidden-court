from __future__ import annotations

import pandas as pd

from src.config import (
    CURRENT_SEASON,
    METRICS_PATH,
    MODEL_PATH,
    PROSPECTS_PATH,
    RAW_DATA_PATH,
    TRAINING_DATA_PATH,
    ensure_directories,
)
from src.features import add_player_features, create_training_pairs
from src.modeling import (
    save_artifacts,
    score_current_prospects,
    temporal_train_and_evaluate,
)


def main() -> None:
    ensure_directories()
    raw = pd.read_csv(RAW_DATA_PATH)
    featured = add_player_features(raw)
    pairs = create_training_pairs(featured)
    pairs.to_csv(TRAINING_DATA_PATH, index=False)

    bundle, metrics = temporal_train_and_evaluate(pairs)
    prospects = score_current_prospects(featured, bundle, CURRENT_SEASON)
    prospects.to_csv(PROSPECTS_PATH, index=False)
    save_artifacts(bundle, metrics, MODEL_PATH, METRICS_PATH)

    print("Holdout metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("\nTop 10 hidden-gem candidates:")
    print(
        prospects[
            ["player_name", "team", "age", "hidden_gem_score", "projected_ppg_growth"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

