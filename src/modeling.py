from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)

from src.config import FEATURES


@dataclass
class ModelBundle:
    classifier: RandomForestClassifier
    regressor: RandomForestRegressor
    features: list[str]
    holdout_season: str


def _build_models(random_state: int = 42):
    classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    regressor = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=random_state,
        n_jobs=-1,
    )
    return classifier, regressor


def temporal_train_and_evaluate(
    pairs: pd.DataFrame,
    holdout_season: str = "2024-25",
) -> tuple[ModelBundle, dict]:
    """Evaluate on the latest labeled season, then refit on every labeled row."""
    eligible = pairs[(pairs["gp"] >= 15) & (pairs["mpg"] >= 5)].copy()
    train = eligible[eligible["season"] < holdout_season]
    test = eligible[eligible["season"] == holdout_season]
    if train.empty or test.empty:
        raise ValueError("Temporal train or holdout split is empty.")

    classifier, regressor = _build_models()
    classifier.fit(train[FEATURES], train["breakout"])
    regressor.fit(train[FEATURES], train["next_ppg"])

    probability = classifier.predict_proba(test[FEATURES])[:, 1]
    projected_ppg = regressor.predict(test[FEATURES])

    top_decile_size = max(1, int(np.ceil(len(test) * 0.10)))
    top_decile_index = np.argsort(probability)[-top_decile_size:]
    top_decile_actual = test["breakout"].to_numpy()[top_decile_index]
    precision_top_decile = float(top_decile_actual.mean())
    recall_top_decile = float(top_decile_actual.sum() / max(test["breakout"].sum(), 1))
    base_rate = float(test["breakout"].mean())

    metrics = {
        "holdout_season": holdout_season,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "breakout_rate_holdout": round(float(test["breakout"].mean()), 4),
        "roc_auc": round(float(roc_auc_score(test["breakout"], probability)), 4),
        "average_precision": round(
            float(average_precision_score(test["breakout"], probability)), 4
        ),
        "precision_top_decile": round(precision_top_decile, 4),
        "recall_top_decile": round(recall_top_decile, 4),
        "lift_top_decile": round(precision_top_decile / max(base_rate, 1e-9), 3),
        "next_ppg_mae": round(float(mean_absolute_error(test["next_ppg"], projected_ppg)), 3),
    }

    final_classifier, final_regressor = _build_models()
    final_classifier.fit(eligible[FEATURES], eligible["breakout"])
    final_regressor.fit(eligible[FEATURES], eligible["next_ppg"])
    bundle = ModelBundle(final_classifier, final_regressor, FEATURES, holdout_season)
    return bundle, metrics


def percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average") * 100


def score_current_prospects(
    featured: pd.DataFrame,
    bundle: ModelBundle,
    current_season: str,
) -> pd.DataFrame:
    current = featured[
        (featured["season"] == current_season)
        & (featured["age"] <= 25)
        & (featured["gp"] >= 20)
        & (featured["mpg"].between(7, 28))
    ].copy()
    if current.empty:
        raise ValueError("No current-season prospects meet the eligibility rules.")

    current["breakout_probability"] = bundle.classifier.predict_proba(
        current[bundle.features]
    )[:, 1]
    current["projected_ppg"] = bundle.regressor.predict(current[bundle.features])
    current["projected_ppg_growth"] = current["projected_ppg"] - current["ppg"]

    efficiency = 0.65 * percentile(current["ts_pct"]) + 0.35 * percentile(
        current["ast_tov"]
    )
    growth = percentile(current["projected_ppg_growth"])
    opportunity = 100 - percentile(current["mpg"])
    current["hidden_gem_score"] = (
        0.55 * current["breakout_probability"] * 100
        + 0.25 * growth
        + 0.10 * efficiency
        + 0.10 * opportunity
    ).clip(0, 100)

    current["model_reason"] = current.apply(_reason, axis=1)
    keep = [
        "player_id",
        "player_name",
        "team",
        "age",
        "gp",
        "mpg",
        "ppg",
        "rpg",
        "apg",
        "spg",
        "bpg",
        "ts_pct",
        "efg_pct",
        "ast_tov",
        "pts_per36",
        "ast_per36",
        "reb_per36",
        "breakout_probability",
        "projected_ppg",
        "projected_ppg_growth",
        "hidden_gem_score",
        "model_reason",
    ]
    return current[keep].sort_values("hidden_gem_score", ascending=False).reset_index(drop=True)


def _reason(row: pd.Series) -> str:
    reasons = []
    if row["ts_pct"] >= 0.58:
        reasons.append("eficiência de arremesso")
    if row["ast_tov"] >= 2.0:
        reasons.append("boa relação assistência/erro")
    if row["pts_per36"] >= 18:
        reasons.append("produção por 36 minutos")
    if row["mpg"] <= 18:
        reasons.append("papel atual limitado")
    if not reasons:
        reasons.append("perfil estatístico semelhante a breakouts históricos")
    return ", ".join(reasons[:2]).capitalize()


def feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": bundle.features,
            "importance": bundle.classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


def save_artifacts(
    bundle: ModelBundle,
    metrics: dict,
    model_path: Path,
    metrics_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
